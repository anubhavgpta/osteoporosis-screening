from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import load_yaml
from .evaluation import (
    classification_metrics,
    threshold_for_sensitivity,
)
from .knhanes import harmonize_knhanes
from .modeling import _xgb_classifier


REFERENCE_PREVALENCE = 0.056
RECALL_FLOOR = 0.90
NHANES_HOLDOUT = "2017-2018"
KNHANES_HOLDOUT = 2011

BASELINE_POLICY = "global_conservative"
RECALL_PRESERVING_POLICY = "source_specific_recall_preserving"
FIXED_RECALL_POLICY = "source_specific_fixed_90"

CONSISTENT_FEATURES = [
    "age",
    "sex",
    "education",
    "age_sex_interaction",
    "bmi",
    "height_cm",
    "weight_kg",
    "menopausal_status",
    "hormone_therapy_ever",
    "arthritis_history",
    "thyroid_problem",
]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    training_sources: tuple[str, ...]
    weight_policy: str
    feature_policy: str


@dataclass(frozen=True)
class IntegrationExperiment:
    metrics: pd.DataFrame
    ranking: pd.DataFrame
    model_selection: pd.DataFrame
    strategy_audit: pd.DataFrame
    predictions: pd.DataFrame
    constrained_model_selection: pd.DataFrame
    policy_metrics: pd.DataFrame
    policy_comparison: pd.DataFrame
    policy_intervals: pd.DataFrame
    policy_predictions: pd.DataFrame
    manifest: dict


STRATEGIES = (
    StrategySpec("nhanes_only", ("nhanes",), "native", "full"),
    StrategySpec("knhanes_only", ("knhanes",), "native", "full"),
    StrategySpec(
        "pooled_naive",
        ("nhanes", "knhanes"),
        "native",
        "full",
    ),
    StrategySpec(
        "pooled_dataset_balanced",
        ("nhanes", "knhanes"),
        "equal_dataset_total",
        "full",
    ),
    StrategySpec(
        "pooled_reduced_consistent",
        ("nhanes", "knhanes"),
        "equal_dataset_total",
        "consistent",
    ),
)


def prevalence_standardized_weights(
    outcome, sample_weight, target_prevalence: float = REFERENCE_PREVALENCE
) -> np.ndarray:
    if not 0 < target_prevalence < 1:
        raise ValueError("target_prevalence must lie strictly between zero and one")
    y = np.asarray(outcome, dtype=int)
    weight = np.asarray(sample_weight, dtype=float)
    positive_total = weight[y == 1].sum()
    negative_total = weight[y == 0].sum()
    if positive_total <= 0 or negative_total <= 0:
        raise ValueError("prevalence standardization requires both outcome classes")
    result = weight.copy()
    result[y == 1] *= target_prevalence / positive_total
    result[y == 0] *= (1 - target_prevalence) / negative_total
    return result / result.mean()


def training_weights(frame: pd.DataFrame, policy: str) -> pd.Series:
    weight = pd.to_numeric(frame["native_weight"], errors="coerce")
    if weight.isna().any() or weight.le(0).any() or ~np.isfinite(weight).all():
        raise ValueError("Every training row must have a finite positive native weight")
    if policy == "native":
        result = weight
    elif policy == "equal_dataset_total":
        source_total = weight.groupby(frame["source_dataset"]).transform("sum")
        result = weight / source_total
    else:
        raise ValueError(f"Unknown integration weight policy: {policy}")
    return (result / result.mean()).rename("training_weight")


def _age_sex_interaction(frame: pd.DataFrame) -> pd.Series:
    age_band = pd.cut(
        frame["age"],
        [-np.inf, 64, 74, np.inf],
        labels=["under_65", "65_74", "75_plus"],
    ).astype("string")
    sex = frame["sex"].astype("Int64").astype("string")
    return ("sex_" + sex + "__age_" + age_band).where(
        frame["age"].notna() & frame["sex"].notna()
    )


def _load_cohorts(root: Path, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    nhanes = pd.read_csv(root / "data" / "processed" / "classification_cohort.csv.gz")
    nhanes = nhanes.loc[
        nhanes["survey_cycle"].isin(
            ["2007-2008", "2009-2010", "2013-2014", NHANES_HOLDOUT]
        )
        & nhanes["osteoporosis"].notna()
    ].copy()
    nhanes["participant_id"] = "NH_" + nhanes["participant_id"].astype("Int64").astype(
        "string"
    )
    nhanes["source_dataset"] = "nhanes"
    nhanes["cohort_id"] = "nhanes_" + nhanes["survey_cycle"].astype("string")
    nhanes["native_weight"] = pd.to_numeric(nhanes["mec_weight"], errors="coerce")

    harmonized_path = (
        root / "data" / "processed" / "knhanes_2008_2011_harmonized.csv.gz"
    )
    if harmonized_path.is_file():
        knhanes = pd.read_csv(harmonized_path)
    else:
        harmonized = harmonize_knhanes(root)
        knhanes = harmonized.frame
        knhanes.to_csv(harmonized_path, index=False, compression="gzip")
    knhanes = knhanes.loc[
        knhanes["age"].ge(50) & knhanes["femoral_neck_bmd"].notna()
    ].copy()
    knhanes["osteoporosis"] = (
        ((knhanes["femoral_neck_bmd"] - 0.858) / 0.120).le(-2.5).astype("int8")
    )
    knhanes["participant_id"] = "KN_" + knhanes["participant_id"].astype("string")
    knhanes["source_dataset"] = "knhanes"
    knhanes["cohort_id"] = "knhanes_" + knhanes["survey_year"].astype(int).astype(str)
    knhanes["native_weight"] = pd.to_numeric(
        knhanes["external_weight"], errors="coerce"
    )
    if "age_sex_interaction" not in knhanes:
        knhanes["age_sex_interaction"] = _age_sex_interaction(knhanes)

    required = set(features) | {
        "participant_id",
        "source_dataset",
        "cohort_id",
        "native_weight",
        "osteoporosis",
        "survey_psu",
        "survey_stratum",
    }
    for label, frame in (("NHANES", nhanes), ("KNHANES", knhanes)):
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(f"{label} integration cohort is missing: {', '.join(missing)}")
    columns = list(required)
    return nhanes[columns].copy(), knhanes[columns].copy()


def _candidate_estimators(config: dict) -> dict[str, object]:
    candidates: dict[str, object] = {
        "regularized_logistic": LogisticRegression(
            C=1.0, max_iter=3000, random_state=config["seed"]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            n_jobs=config.get("cv_n_jobs", 1),
            random_state=config["seed"],
        ),
    }
    xgb = _xgb_classifier(config)
    if xgb is not None:
        xgb.set_params(n_estimators=250)
        candidates["xgboost"] = xgb
    return candidates


def _integration_pipeline(
    features: list[str], categorical: set[str], estimator
) -> Pipeline:
    categorical_features = [feature for feature in features if feature in categorical]
    numeric_features = [feature for feature in features if feature not in categorical]
    numeric = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            ("scale", StandardScaler().set_fit_request(sample_weight=False)),
        ]
    )
    category = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="missing",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("categorical", category, categorical_features),
        ],
        remainder="drop",
    )
    if hasattr(estimator, "set_fit_request"):
        estimator = estimator.set_fit_request(sample_weight=True)
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _fit_group_oof(
    training: pd.DataFrame,
    features: list[str],
    categorical: set[str],
    weight: pd.Series,
    estimator,
    config: dict,
) -> tuple[np.ndarray, list[dict]]:
    groups = training["cohort_id"]
    if groups.nunique() < 2:
        raise ValueError("Cohort-level validation requires at least two training cohorts")
    oof = np.full(len(training), np.nan)
    rows = []
    splitter = LeaveOneGroupOut()
    for train_index, validation_index in splitter.split(training, groups=groups):
        pipeline = _integration_pipeline(features, categorical, clone(estimator))
        pipeline.fit(
            training.iloc[train_index][features],
            training.iloc[train_index]["osteoporosis"],
            sample_weight=weight.iloc[train_index].to_numpy(),
        )
        probability = pipeline.predict_proba(training.iloc[validation_index][features])[:, 1]
        oof[validation_index] = probability
        validation = training.iloc[validation_index]
        validation_weight = weight.iloc[validation_index]
        y = validation["osteoporosis"].to_numpy()
        standardized = prevalence_standardized_weights(y, validation_weight)
        rows.append(
            {
                "validation_cohort": validation["cohort_id"].iloc[0],
                "validation_source": validation["source_dataset"].iloc[0],
                "validation_n": len(validation),
                "validation_prevalence": float(
                    np.average(y, weights=validation_weight)
                ),
                "roc_auc": roc_auc_score(y, probability, sample_weight=validation_weight),
                "pr_auc": average_precision_score(
                    y, probability, sample_weight=validation_weight
                ),
                "pr_auc_standardized_0_056": average_precision_score(
                    y, probability, sample_weight=standardized
                ),
            }
        )
    if not np.isfinite(oof).all():
        raise ValueError("Grouped out-of-fold prediction did not cover every training row")
    return oof, rows


def _platt_calibrator(outcome, raw_probability, weight) -> LogisticRegression:
    clipped = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    calibrator.fit(logits, outcome, sample_weight=weight)
    return calibrator


def _apply_calibrator(calibrator: LogisticRegression, raw_probability) -> np.ndarray:
    clipped = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def _ppv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    numerator = sensitivity * prevalence
    denominator = numerator + (1 - specificity) * (1 - prevalence)
    return numerator / denominator if denominator else np.nan


def source_threshold_policy(
    training: pd.DataFrame,
    probability,
    sample_weight,
    recall_targets: dict[str, float],
    guard_by_cohort: bool = True,
) -> tuple[dict[str, float], list[dict]]:
    """Select and audit one training-only threshold for each represented source."""
    probability = np.asarray(probability, dtype=float)
    weight = np.asarray(sample_weight, dtype=float)
    thresholds: dict[str, float] = {}
    rows: list[dict] = []
    for source in sorted(training["source_dataset"].unique()):
        mask = training["source_dataset"].eq(source).to_numpy()
        target = float(recall_targets[source])
        if guard_by_cohort and "cohort_id" in training:
            cohort_values = training["cohort_id"].astype("string").to_numpy()
        else:
            cohort_values = np.full(len(training), source, dtype=object)
        cohort_thresholds = {}
        for cohort in sorted(np.unique(cohort_values[mask])):
            cohort_mask = mask & (cohort_values == cohort)
            cohort_thresholds[str(cohort)] = threshold_for_sensitivity(
                training.loc[cohort_mask, "osteoporosis"],
                probability[cohort_mask],
                target,
                sample_weight=weight[cohort_mask],
            )
        # A lower threshold predicts more positives. The minimum cohort-level
        # threshold is therefore the conservative value that meets the recall
        # target in every held-out training cycle/year for this source.
        limiting_cohort = min(cohort_thresholds, key=cohort_thresholds.get)
        threshold = float(cohort_thresholds[limiting_cohort])
        metrics = classification_metrics(
            training.loc[mask, "osteoporosis"],
            probability[mask],
            threshold,
            sample_weight=weight[mask],
        )
        cohort_recalls = []
        for cohort in cohort_thresholds:
            cohort_mask = mask & (cohort_values == cohort)
            cohort_metrics = classification_metrics(
                training.loc[cohort_mask, "osteoporosis"],
                probability[cohort_mask],
                threshold,
                sample_weight=weight[cohort_mask],
            )
            cohort_recalls.append(cohort_metrics["sensitivity"])
        minimum_cohort_recall = float(np.min(cohort_recalls))
        thresholds[source] = float(threshold)
        rows.append(
            {
                "source_dataset": source,
                "target_recall": target,
                "threshold": float(threshold),
                "limiting_training_cohort": limiting_cohort,
                "oof_recall": metrics["sensitivity"],
                "minimum_cohort_oof_recall": minimum_cohort_recall,
                "oof_specificity": metrics["specificity"],
                "oof_ppv": metrics["ppv"],
                "oof_accuracy": metrics["accuracy"],
                "oof_balanced_accuracy": metrics["balanced_accuracy"],
                "oof_ppv_standardized_0_056": _ppv_at_prevalence(
                    metrics["sensitivity"],
                    metrics["specificity"],
                    REFERENCE_PREVALENCE,
                ),
                "recall_constraint_met": bool(
                    minimum_cohort_recall + 1e-12 >= target
                ),
            }
        )
    return thresholds, rows


def select_precision_constrained_model(candidate_rows: list[dict]) -> str:
    """Choose the best worst-source standardized PPV among recall-feasible models."""
    frame = pd.DataFrame(candidate_rows)
    required = {
        "model",
        "source_dataset",
        "recall_constraint_met",
        "oof_ppv_standardized_0_056",
        "oof_specificity",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Constrained model rows are missing: {', '.join(sorted(missing))}")
    summary = (
        frame.groupby("model", as_index=False)
        .agg(
            recall_feasible=("recall_constraint_met", "all"),
            minimum_standardized_ppv=("oof_ppv_standardized_0_056", "min"),
            mean_standardized_ppv=("oof_ppv_standardized_0_056", "mean"),
            minimum_specificity=("oof_specificity", "min"),
        )
        .loc[lambda data: data["recall_feasible"]]
    )
    if summary.empty:
        raise ValueError("No candidate model met every source-specific recall constraint")
    return str(
        summary.sort_values(
            [
                "minimum_standardized_ppv",
                "mean_standardized_ppv",
                "minimum_specificity",
                "model",
            ],
            ascending=[False, False, False, True],
        )["model"].iloc[0]
    )


def _binary_operating_metrics(outcome, predicted, sample_weight) -> dict[str, float]:
    y = np.asarray(outcome, dtype=int)
    prediction = np.asarray(predicted, dtype=bool)
    weight = np.asarray(sample_weight, dtype=float)
    tp = weight[prediction & (y == 1)].sum()
    fp = weight[prediction & (y == 0)].sum()
    fn = weight[~prediction & (y == 1)].sum()
    tn = weight[~prediction & (y == 0)].sum()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "ppv_standardized_0_056": _ppv_at_prevalence(
            sensitivity, specificity, REFERENCE_PREVALENCE
        ),
    }


def _resample_psus_within_strata(data: pd.DataFrame, rng) -> pd.DataFrame:
    sampled = []
    valid = data.dropna(subset=["survey_stratum", "survey_psu"])
    for _, stratum in valid.groupby("survey_stratum", observed=True):
        psus = stratum["survey_psu"].dropna().unique()
        for psu in rng.choice(psus, size=len(psus), replace=True):
            sampled.append(stratum.loc[stratum["survey_psu"].eq(psu)])
    return pd.concat(sampled, ignore_index=True) if sampled else data.iloc[0:0]


def paired_policy_intervals(
    policy_predictions: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Paired PSU-within-stratum bootstrap differences versus the baseline policy."""
    rng = np.random.default_rng(seed)
    reported = (
        "accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "balanced_accuracy",
        "ppv_standardized_0_056",
    )
    rows = []
    group_columns = ["strategy", "holdout"]
    for (strategy, holdout), group in policy_predictions.groupby(
        group_columns, observed=True
    ):
        baseline = group.loc[group["policy"].eq(BASELINE_POLICY)].copy()
        baseline = baseline.rename(columns={"prediction": "baseline_prediction"})
        baseline = baseline[
            [
                "participant_id",
                "osteoporosis",
                "analysis_weight",
                "survey_psu",
                "survey_stratum",
                "baseline_prediction",
            ]
        ]
        baseline_metrics = _binary_operating_metrics(
            baseline["osteoporosis"],
            baseline["baseline_prediction"],
            baseline["analysis_weight"],
        )
        for policy in sorted(set(group["policy"]) - {BASELINE_POLICY}):
            candidate = group.loc[group["policy"].eq(policy), ["participant_id", "prediction"]]
            candidate = candidate.rename(columns={"prediction": "candidate_prediction"})
            paired = baseline.merge(candidate, on="participant_id", how="inner", validate="one_to_one")
            candidate_metrics = _binary_operating_metrics(
                paired["osteoporosis"],
                paired["candidate_prediction"],
                paired["analysis_weight"],
            )
            draws = {metric: [] for metric in reported}
            for _ in range(iterations):
                sample = _resample_psus_within_strata(paired, rng)
                if sample.empty or sample["osteoporosis"].nunique() < 2:
                    continue
                baseline_draw = _binary_operating_metrics(
                    sample["osteoporosis"],
                    sample["baseline_prediction"],
                    sample["analysis_weight"],
                )
                candidate_draw = _binary_operating_metrics(
                    sample["osteoporosis"],
                    sample["candidate_prediction"],
                    sample["analysis_weight"],
                )
                for metric in reported:
                    delta = candidate_draw[metric] - baseline_draw[metric]
                    if np.isfinite(delta):
                        draws[metric].append(delta)
            for metric in reported:
                values = np.asarray(draws[metric], dtype=float)
                rows.append(
                    {
                        "strategy": strategy,
                        "holdout": holdout,
                        "policy": policy,
                        "metric": metric,
                        "baseline_estimate": baseline_metrics[metric],
                        "policy_estimate": candidate_metrics[metric],
                        "delta": candidate_metrics[metric] - baseline_metrics[metric],
                        "delta_ci_low": np.quantile(values, 0.025) if len(values) else np.nan,
                        "delta_ci_high": np.quantile(values, 0.975) if len(values) else np.nan,
                        "successful_bootstrap_iterations": len(values),
                        "requested_bootstrap_iterations": iterations,
                        "resampling_design": (
                            "paired policies; PSUs resampled with replacement within survey strata"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _evaluate_holdout(
    strategy: str,
    selected_model: str,
    holdout_name: str,
    holdout: pd.DataFrame,
    features: list[str],
    pipeline,
    calibrator: LogisticRegression,
    threshold: float,
) -> tuple[dict, pd.DataFrame]:
    raw_probability = pipeline.predict_proba(holdout[features])[:, 1]
    probability = _apply_calibrator(calibrator, raw_probability)
    weight = holdout["native_weight"].to_numpy()
    outcome = holdout["osteoporosis"].to_numpy()
    metrics = classification_metrics(outcome, probability, threshold, weight)
    standardized_weight = prevalence_standardized_weights(outcome, weight)
    diagnostic_threshold = threshold_for_sensitivity(
        outcome, probability, 0.90, sample_weight=weight
    )
    diagnostic = classification_metrics(
        outcome, probability, diagnostic_threshold, weight
    )
    row = {
        "strategy": strategy,
        "selected_model": selected_model,
        "holdout": holdout_name,
        "n": len(holdout),
        "events": int(outcome.sum()),
        "weighted_prevalence": float(np.average(outcome, weights=weight)),
        "mean_predicted_probability": float(np.average(probability, weights=weight)),
        "pr_auc_standardized_0_056": average_precision_score(
            outcome, probability, sample_weight=standardized_weight
        ),
        "ppv_standardized_0_056": _ppv_at_prevalence(
            metrics["sensitivity"], metrics["specificity"], REFERENCE_PREVALENCE
        ),
        "diagnostic_threshold_at_90_recall": diagnostic_threshold,
        "diagnostic_recall": diagnostic["sensitivity"],
        "diagnostic_specificity": diagnostic["specificity"],
        "diagnostic_ppv": diagnostic["ppv"],
        "diagnostic_ppv_standardized_0_056": _ppv_at_prevalence(
            diagnostic["sensitivity"],
            diagnostic["specificity"],
            REFERENCE_PREVALENCE,
        ),
        **metrics,
    }
    predictions = holdout[
        [
            "participant_id",
            "source_dataset",
            "cohort_id",
            "osteoporosis",
            "survey_psu",
            "survey_stratum",
        ]
    ].copy()
    predictions.insert(0, "strategy", strategy)
    predictions["holdout"] = holdout_name
    predictions["analysis_weight"] = weight
    predictions["probability"] = probability
    predictions["training_selected_threshold"] = threshold
    predictions["prediction"] = probability >= threshold
    return row, predictions


def run_integration_experiment(root: Path) -> IntegrationExperiment:
    root = root.resolve()
    metadata = json.loads(
        (root / "artifacts" / "classification_model_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    full_features = list(metadata["features"])
    config = load_yaml(root / "configs" / "model_config.yaml")
    recall_floor = float(config.get("screening_sensitivity_target", RECALL_FLOOR))
    variable_config = load_yaml(root / "configs" / "variables.yaml")
    categorical = set(variable_config["categorical_features"])
    nhanes, knhanes = _load_cohorts(root, full_features)
    for frame in (nhanes, knhanes):
        for feature in categorical.intersection(full_features):
            values = frame[feature].astype("string").astype("object")
            frame[feature] = values.where(pd.notna(values), np.nan)
    nhanes_training = nhanes.loc[
        ~nhanes["cohort_id"].eq(f"nhanes_{NHANES_HOLDOUT}")
    ].copy()
    knhanes_training = knhanes.loc[
        ~knhanes["cohort_id"].eq(f"knhanes_{KNHANES_HOLDOUT}")
    ].copy()
    holdouts = {
        "nhanes_2017_2018": nhanes.loc[
            nhanes["cohort_id"].eq(f"nhanes_{NHANES_HOLDOUT}")
        ].copy(),
        "knhanes_2011": knhanes.loc[
            knhanes["cohort_id"].eq(f"knhanes_{KNHANES_HOLDOUT}")
        ].copy(),
    }

    metrics_rows = []
    selection_rows = []
    audit_rows = []
    prediction_frames = []
    constrained_selection_rows = []
    policy_metric_rows = []
    policy_prediction_frames = []
    for spec in STRATEGIES:
        pieces = []
        if "nhanes" in spec.training_sources:
            pieces.append(nhanes_training)
        if "knhanes" in spec.training_sources:
            pieces.append(knhanes_training)
        training = pd.concat(pieces, ignore_index=True)
        features = full_features if spec.feature_policy == "full" else CONSISTENT_FEATURES
        weight = training_weights(training, spec.weight_policy)
        source_weight_share = (
            weight.groupby(training["source_dataset"]).sum() / weight.sum()
        )
        audit_rows.append(
            {
                "strategy": spec.name,
                "training_n": len(training),
                "training_events": int(training["osteoporosis"].sum()),
                "training_cohorts": "|".join(sorted(training["cohort_id"].unique())),
                "weight_policy": spec.weight_policy,
                "feature_policy": spec.feature_policy,
                "feature_count": len(features),
                "nhanes_weight_share": float(source_weight_share.get("nhanes", 0)),
                "knhanes_weight_share": float(source_weight_share.get("knhanes", 0)),
            }
        )

        candidate_oof = {}
        candidate_summaries = []
        for model_name, estimator in _candidate_estimators(config).items():
            oof, fold_rows = _fit_group_oof(
                training, features, categorical, weight, estimator, config
            )
            candidate_oof[model_name] = oof
            for row in fold_rows:
                row.update({"strategy": spec.name, "model": model_name})
                selection_rows.append(row)
            candidate_summaries.append(
                {
                    "model": model_name,
                    "mean_standardized_pr_auc": float(
                        np.mean([row["pr_auc_standardized_0_056"] for row in fold_rows])
                    ),
                    "mean_roc_auc": float(np.mean([row["roc_auc"] for row in fold_rows])),
                }
            )
        selected_summary = sorted(
            candidate_summaries,
            key=lambda row: (row["mean_standardized_pr_auc"], row["mean_roc_auc"]),
            reverse=True,
        )[0]
        selected_model = selected_summary["model"]
        selected_estimator = _candidate_estimators(config)[selected_model]
        raw_oof = candidate_oof[selected_model]
        calibrator = _platt_calibrator(
            training["osteoporosis"], raw_oof, weight.to_numpy()
        )
        calibrated_oof = _apply_calibrator(calibrator, raw_oof)
        represented_sources = sorted(training["source_dataset"].unique())
        fixed_targets = {source: recall_floor for source in represented_sources}
        source_thresholds, _ = source_threshold_policy(
            training,
            calibrated_oof,
            weight,
            fixed_targets,
            guard_by_cohort=False,
        )
        # A lower probability threshold predicts more positives. Selecting the
        # minimum source-specific threshold enforces the training recall target
        # in every represented dataset without consulting either holdout.
        threshold = float(min(source_thresholds.values()))
        audit_rows[-1]["selected_model"] = selected_model
        audit_rows[-1]["training_selected_threshold"] = threshold
        incumbent_recall_targets = {}
        for source in represented_sources:
            source_mask = training["source_dataset"].eq(source).to_numpy()
            source_metrics = classification_metrics(
                training.loc[source_mask, "osteoporosis"],
                calibrated_oof[source_mask],
                threshold,
                sample_weight=weight.loc[source_mask],
            )
            audit_rows[-1][f"training_oof_recall_{source}"] = source_metrics[
                "sensitivity"
            ]
            audit_rows[-1][f"training_oof_specificity_{source}"] = source_metrics[
                "specificity"
            ]
            incumbent_recall_targets[source] = source_metrics["sensitivity"]
        final_pipeline = _integration_pipeline(
            features, categorical, clone(selected_estimator)
        )
        final_pipeline.fit(
            training[features],
            training["osteoporosis"],
            sample_weight=weight.to_numpy(),
        )
        for holdout_name, holdout in holdouts.items():
            row, predictions = _evaluate_holdout(
                spec.name,
                selected_model,
                holdout_name,
                holdout,
                features,
                final_pipeline,
                calibrator,
                threshold,
            )
            row["training_n"] = len(training)
            row["feature_count"] = len(features)
            metrics_rows.append(row)
            prediction_frames.append(predictions)
            policy_row = dict(row)
            policy_row.update(
                {
                    "policy": BASELINE_POLICY,
                    "threshold_scope": "one_global_threshold",
                    "training_recall_target": recall_floor,
                }
            )
            policy_metric_rows.append(policy_row)
            policy_predictions = predictions.copy()
            policy_predictions["policy"] = BASELINE_POLICY
            policy_prediction_frames.append(policy_predictions)

        policy_targets = {
            RECALL_PRESERVING_POLICY: incumbent_recall_targets,
            FIXED_RECALL_POLICY: fixed_targets,
        }
        for policy, targets in policy_targets.items():
            local_selection_rows = []
            candidate_calibrators = {}
            candidate_thresholds = {}
            for model_name, candidate_raw_oof in candidate_oof.items():
                candidate_calibrator = _platt_calibrator(
                    training["osteoporosis"], candidate_raw_oof, weight.to_numpy()
                )
                candidate_calibrated_oof = _apply_calibrator(
                    candidate_calibrator, candidate_raw_oof
                )
                thresholds, policy_rows = source_threshold_policy(
                    training,
                    candidate_calibrated_oof,
                    weight,
                    targets,
                )
                candidate_calibrators[model_name] = candidate_calibrator
                candidate_thresholds[model_name] = thresholds
                for policy_row in policy_rows:
                    policy_row.update(
                        {
                            "strategy": spec.name,
                            "policy": policy,
                            "model": model_name,
                        }
                    )
                    local_selection_rows.append(policy_row)
            policy_model = select_precision_constrained_model(local_selection_rows)
            for policy_row in local_selection_rows:
                policy_row["selected"] = policy_row["model"] == policy_model
            constrained_selection_rows.extend(local_selection_rows)

            policy_estimator = _candidate_estimators(config)[policy_model]
            policy_pipeline = _integration_pipeline(
                features, categorical, clone(policy_estimator)
            )
            policy_pipeline.fit(
                training[features],
                training["osteoporosis"],
                sample_weight=weight.to_numpy(),
            )
            thresholds = candidate_thresholds[policy_model]
            policy_calibrator = candidate_calibrators[policy_model]
            for holdout_name, holdout in holdouts.items():
                holdout_source = str(holdout["source_dataset"].iloc[0])
                source_available = holdout_source in thresholds
                policy_threshold = float(
                    thresholds.get(holdout_source, min(thresholds.values()))
                )
                row, predictions = _evaluate_holdout(
                    spec.name,
                    policy_model,
                    holdout_name,
                    holdout,
                    features,
                    policy_pipeline,
                    policy_calibrator,
                    policy_threshold,
                )
                row.update(
                    {
                        "training_n": len(training),
                        "feature_count": len(features),
                        "policy": policy,
                        "threshold_scope": (
                            f"source_specific_{holdout_source}"
                            if source_available
                            else "unseen_source_fallback"
                        ),
                        "training_recall_target": targets.get(holdout_source, np.nan),
                    }
                )
                policy_metric_rows.append(row)
                predictions["policy"] = policy
                policy_prediction_frames.append(predictions)

        for row in selection_rows:
            if row["strategy"] == spec.name:
                row["selected"] = row["model"] == selected_model

    metrics = pd.DataFrame(metrics_rows)
    ranking = (
        metrics.groupby("strategy", as_index=False)
        .agg(
            selected_model=("selected_model", "first"),
            mean_roc_auc=("roc_auc", "mean"),
            worst_roc_auc=("roc_auc", "min"),
            mean_standardized_pr_auc=("pr_auc_standardized_0_056", "mean"),
            worst_standardized_pr_auc=("pr_auc_standardized_0_056", "min"),
            minimum_recall=("sensitivity", "min"),
            mean_recall=("sensitivity", "mean"),
            minimum_specificity=("specificity", "min"),
            mean_specificity=("specificity", "mean"),
            mean_standardized_ppv=("ppv_standardized_0_056", "mean"),
        )
        .sort_values(
            ["mean_standardized_pr_auc", "worst_standardized_pr_auc"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking["meets_90_recall_on_both_holdouts"] = ranking["minimum_recall"].ge(0.90)
    screening_order = ranking.sort_values(
        [
            "meets_90_recall_on_both_holdouts",
            "mean_standardized_ppv",
            "mean_standardized_pr_auc",
        ],
        ascending=False,
    )
    screening_rank = {
        strategy: rank
        for rank, strategy in enumerate(screening_order["strategy"], start=1)
    }
    ranking["screening_rank"] = ranking["strategy"].map(screening_rank)
    policy_metrics = pd.DataFrame(policy_metric_rows)
    policy_predictions = pd.concat(policy_prediction_frames, ignore_index=True)
    comparison_metrics = [
        "accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "balanced_accuracy",
        "ppv_standardized_0_056",
    ]
    comparison_keys = ["strategy", "holdout"]
    baseline = policy_metrics.loc[
        policy_metrics["policy"].eq(BASELINE_POLICY),
        comparison_keys + ["selected_model", "threshold", *comparison_metrics],
    ].rename(
        columns={
            "selected_model": "baseline_model",
            "threshold": "baseline_threshold",
            **{metric: f"baseline_{metric}" for metric in comparison_metrics},
        }
    )
    policy_comparison = policy_metrics.loc[
        ~policy_metrics["policy"].eq(BASELINE_POLICY),
        comparison_keys
        + [
            "policy",
            "selected_model",
            "threshold",
            "threshold_scope",
            "training_recall_target",
            *comparison_metrics,
        ],
    ].merge(baseline, on=comparison_keys, how="left", validate="many_to_one")
    for metric in comparison_metrics:
        policy_comparison[f"delta_{metric}"] = (
            policy_comparison[metric] - policy_comparison[f"baseline_{metric}"]
        )
    policy_comparison["precision_and_recall_both_improved"] = (
        policy_comparison["delta_ppv"].gt(0)
        & policy_comparison["delta_sensitivity"].ge(0)
    )
    policy_comparison["accuracy_precision_recall_all_improved"] = (
        policy_comparison["delta_accuracy"].gt(0)
        & policy_comparison["delta_ppv"].gt(0)
        & policy_comparison["delta_sensitivity"].ge(0)
    )
    policy_intervals = paired_policy_intervals(
        policy_predictions.loc[policy_predictions["strategy"].eq("pooled_naive")],
        iterations=int(config.get("bootstrap_iterations", 500)),
        seed=int(config["seed"]),
    )
    manifest = {
        "experiment": "nhanes_knhanes_training_integration",
        "strategies": [spec.name for spec in STRATEGIES],
        "training_holdouts_excluded": ["NHANES 2017-2018", "KNHANES 2011"],
        "internal_validation": "leave-one-survey-cycle-or-year-out",
        "preprocessing": (
            "median numeric imputation and explicit categorical missing levels, "
            "fitted separately within every training fold"
        ),
        "model_selection_metric": "mean PR-AUC standardized to 5.6% prevalence",
        "operating_threshold_selection": (
            "minimum of source-specific grouped-OOF thresholds targeting 90% recall; "
            "ensures the target separately in each represented training dataset"
        ),
        "ranking_metric": "mean standardized PR-AUC across both validation holdouts",
        "experimental_threshold_policies": {
            RECALL_PRESERVING_POLICY: (
                "maximize worst-source prevalence-standardized PPV while matching or "
                "exceeding each source's incumbent grouped-OOF recall"
            ),
            FIXED_RECALL_POLICY: (
                f"maximize worst-source prevalence-standardized PPV subject to "
                f"grouped-OOF recall >= {recall_floor:.3f} in every training cycle/year"
            ),
        },
        "constrained_model_selection_metric": (
            "maximum worst-source PPV standardized to 5.6% prevalence among models "
            "meeting every source recall constraint"
        ),
        "policy_comparison_uncertainty": (
            "paired PSU-within-stratum bootstrap on unchanged holdout participants; "
            "prespecified for the native-weight pooled incumbent strategy"
        ),
        "reference_prevalence": REFERENCE_PREVALENCE,
        "holdouts_used_for_final_strategy_ranking": True,
        "final_independent_confirmation_still_required": True,
        "model_artifact_overwritten": False,
    }
    return IntegrationExperiment(
        metrics=metrics,
        ranking=ranking,
        model_selection=pd.DataFrame(selection_rows),
        strategy_audit=pd.DataFrame(audit_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
        constrained_model_selection=pd.DataFrame(constrained_selection_rows),
        policy_metrics=policy_metrics,
        policy_comparison=policy_comparison,
        policy_intervals=policy_intervals,
        policy_predictions=policy_predictions,
        manifest=manifest,
    )
