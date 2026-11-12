from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from .config import ProjectPaths, load_yaml
from .modeling import XGBClassifier, feature_experiments, make_pipeline


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    feature_set: str
    features: tuple[str, ...]
    estimator: object


def weighted_wilson_lower_bound(
    outcome,
    predicted,
    sample_weight=None,
    confidence: float = 0.95,
) -> float:
    """One-sided Wilson lower bound for weighted sensitivity.

    The positive-class Kish effective sample size prevents survey weights from
    pretending that the recall estimate contains more information than it does.
    """
    outcome = np.asarray(outcome, dtype=int)
    predicted = np.asarray(predicted, dtype=bool)
    weight = (
        np.ones(len(outcome), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    positive_weight = weight[outcome == 1]
    if not len(positive_weight) or positive_weight.sum() <= 0:
        return np.nan
    success_weight = weight[(outcome == 1) & predicted].sum()
    p_hat = float(success_weight / positive_weight.sum())
    effective_n = float(positive_weight.sum() ** 2 / np.square(positive_weight).sum())
    z = float(norm.ppf(confidence))
    denominator = 1 + z * z / effective_n
    center = p_hat + z * z / (2 * effective_n)
    spread = z * np.sqrt(
        p_hat * (1 - p_hat) / effective_n + z * z / (4 * effective_n**2)
    )
    return float(max(0.0, (center - spread) / denominator))


def select_recall_threshold(
    outcome,
    probability,
    sample_weight,
    target: float,
    confidence: float = 0.95,
    require_lcb: bool = True,
    objective: str = "specificity",
    utility_threshold_probability: float | None = None,
) -> dict:
    """Select a training-only threshold under a point or LCB recall constraint."""
    outcome = np.asarray(outcome, dtype=int)
    probability = np.asarray(probability, dtype=float)
    weight = np.asarray(sample_weight, dtype=float)
    if objective not in {"specificity", "net_benefit"}:
        raise ValueError("objective must be 'specificity' or 'net_benefit'")
    if objective == "net_benefit" and not (
        utility_threshold_probability is not None
        and 0 < utility_threshold_probability < 1
    ):
        raise ValueError(
            "net_benefit selection requires a fixed utility_threshold_probability in (0, 1)"
        )
    if not (len(outcome) == len(probability) == len(weight)):
        raise ValueError("outcome, probability, and sample_weight must have equal length")
    if not 0 < target <= 1:
        raise ValueError("target must be in (0, 1]")

    # Zero is included so a feasible all-positive fallback is audited rather
    # than replaced by an arbitrary 0.5 threshold.
    thresholds = np.unique(np.r_[0.0, probability])
    order = np.argsort(probability, kind="stable")
    ordered_probability = probability[order]
    ordered_outcome = outcome[order]
    ordered_weight = weight[order]
    # For the >= threshold rule, searchsorted(left) gives the rows predicted
    # negative. Prefix sums then produce every weighted confusion matrix in
    # O(n log n), instead of rebuilding it once per observed probability.
    cut = np.searchsorted(ordered_probability, thresholds, side="left")
    positive_prefix = np.r_[
        0.0, np.cumsum(ordered_weight * (ordered_outcome == 1))
    ]
    negative_prefix = np.r_[
        0.0, np.cumsum(ordered_weight * (ordered_outcome == 0))
    ]
    fn = positive_prefix[cut]
    tn = negative_prefix[cut]
    total_positive = positive_prefix[-1]
    total_negative = negative_prefix[-1]
    tp = total_positive - fn
    fp = total_negative - tn
    sensitivity = np.divide(
        tp,
        total_positive,
        out=np.full_like(tp, np.nan),
        where=total_positive > 0,
    )
    specificity = np.divide(
        tn,
        total_negative,
        out=np.full_like(tn, np.nan),
        where=total_negative > 0,
    )

    positive_weight = weight[outcome == 1]
    effective_n = (
        positive_weight.sum() ** 2 / np.square(positive_weight).sum()
        if len(positive_weight) and positive_weight.sum() > 0
        else np.nan
    )
    z = float(norm.ppf(confidence))
    denominator = 1 + z * z / effective_n
    center = sensitivity + z * z / (2 * effective_n)
    spread = z * np.sqrt(
        sensitivity * (1 - sensitivity) / effective_n
        + z * z / (4 * effective_n**2)
    )
    lcb = np.maximum(0.0, (center - spread) / denominator)

    harm_threshold = (
        np.full_like(thresholds, float(utility_threshold_probability))
        if utility_threshold_probability is not None
        else thresholds
    )
    odds = np.divide(
        harm_threshold,
        1 - harm_threshold,
        out=np.full_like(harm_threshold, np.inf),
        where=harm_threshold < 1,
    )
    total = weight.sum()
    net_benefit = (
        tp / total - fp / total * odds
        if total > 0
        else np.full_like(thresholds, -np.inf)
    )
    net_benefit[~np.isfinite(odds)] = -np.inf
    feasible_value = lcb if require_lcb else sensitivity
    candidates = pd.DataFrame(
        {
            "threshold": thresholds,
            "oof_sensitivity": sensitivity,
            "oof_sensitivity_lcb": lcb,
            "oof_specificity": specificity,
            "oof_net_benefit": net_benefit,
            "feasible": feasible_value + 1e-12 >= target,
        }
    )
    feasible = candidates.loc[candidates["feasible"]].copy()
    constraint_met = not feasible.empty
    if feasible.empty:
        # This can occur in a small subgroup when even 100% observed recall has
        # a Wilson bound below 90%. Retain the safest threshold and flag it.
        feasible = candidates.sort_values(
            ["oof_sensitivity_lcb", "oof_sensitivity", "oof_specificity"],
            ascending=False,
        ).head(1)
    sort_columns = (
        ["oof_specificity", "threshold"]
        if objective == "specificity"
        else ["oof_net_benefit", "oof_specificity", "threshold"]
    )
    selected = feasible.sort_values(sort_columns, ascending=False).iloc[0].to_dict()
    selected["recall_target"] = float(target)
    selected["recall_constraint"] = "lower_confidence_bound" if require_lcb else "point"
    selected["lcb_confidence"] = float(confidence)
    selected["objective"] = objective
    selected["utility_threshold_probability"] = utility_threshold_probability
    selected["constraint_met"] = bool(constraint_met)
    return selected


def select_overall_threshold(
    outcome,
    probability,
    sample_weight,
    objective: str,
    utility_threshold_probability: float | None = None,
) -> dict:
    """Select a training-only threshold for a pre-specified overall objective."""
    valid = {"balanced_accuracy", "mcc", "f1", "net_benefit"}
    if objective not in valid:
        raise ValueError(f"objective must be one of {sorted(valid)}")
    if objective == "net_benefit" and not (
        utility_threshold_probability is not None
        and 0 < utility_threshold_probability < 1
    ):
        raise ValueError(
            "net_benefit selection requires utility_threshold_probability in (0, 1)"
        )
    outcome = np.asarray(outcome, dtype=int)
    probability = np.asarray(probability, dtype=float)
    weight = np.asarray(sample_weight, dtype=float)
    rows = []
    for threshold in np.unique(np.r_[0.0, probability]):
        predicted = probability >= threshold
        tn, fp, fn, tp = confusion_matrix(
            outcome, predicted, labels=[0, 1], sample_weight=weight
        ).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        balanced_accuracy = (sensitivity + specificity) / 2
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denominator if denominator else np.nan
        utility = float(utility_threshold_probability or 0.5)
        net_benefit = tp / weight.sum() - fp / weight.sum() * utility / (1 - utility)
        rows.append(
            {
                "threshold": float(threshold),
                "oof_sensitivity": float(sensitivity),
                "oof_specificity": float(specificity),
                "oof_balanced_accuracy": float(balanced_accuracy),
                "oof_mcc": float(mcc),
                "oof_f1": float(f1),
                "oof_net_benefit": float(net_benefit),
            }
        )
    candidates = pd.DataFrame(rows)
    column = f"oof_{objective}"
    selected = candidates.sort_values(
        [column, "oof_sensitivity", "oof_specificity", "threshold"],
        ascending=False,
    ).iloc[0].to_dict()
    selected["objective"] = objective
    selected["recall_target"] = np.nan
    selected["recall_constraint"] = "none"
    selected["utility_threshold_probability"] = (
        utility_threshold_probability if objective == "net_benefit" else None
    )
    return selected


def biomarker_cycle_availability(
    frame: pd.DataFrame,
    panels: dict[str, list[str]],
    minimum_observed_fraction: float,
) -> pd.DataFrame:
    rows = []
    for cycle, cycle_frame in frame.groupby("survey_cycle", observed=True):
        for panel, features in panels.items():
            fractions = {
                feature: (
                    float(cycle_frame[feature].notna().mean()) if feature in cycle_frame else 0.0
                )
                for feature in features
            }
            rows.append(
                {
                    "survey_cycle": str(cycle),
                    "panel": panel,
                    "n": len(cycle_frame),
                    "all_features_available": bool(
                        all(value >= minimum_observed_fraction for value in fractions.values())
                    ),
                    "minimum_observed_fraction": min(fractions.values()),
                    **{f"observed_fraction__{key}": value for key, value in fractions.items()},
                }
            )
    return pd.DataFrame(rows)


def _candidate_estimators(config: dict, seed: int, n_jobs: int) -> list[tuple[str, object]]:
    candidates: list[tuple[str, object]] = []
    for value in config.get("logistic_c", [1.0]):
        candidates.append(
            (
                f"logistic_c_{value:g}",
                LogisticRegression(C=float(value), max_iter=3000, random_state=seed),
            )
        )
    trees = int(config.get("tree_n_estimators", 200))
    for value in config.get("random_forest_min_samples_leaf", [10]):
        candidates.append(
            (
                f"random_forest_leaf_{value}",
                RandomForestClassifier(
                    n_estimators=trees,
                    min_samples_leaf=int(value),
                    n_jobs=n_jobs,
                    random_state=seed,
                ),
            )
        )
    if XGBClassifier is not None:
        for depth in config.get("xgboost_max_depth", [3]):
            candidates.append(
                (
                    f"xgboost_depth_{depth}",
                    XGBClassifier(
                        n_estimators=trees,
                        max_depth=int(depth),
                        learning_rate=0.04,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=2.0,
                        min_child_weight=5,
                        tree_method="hist",
                        eval_metric="logloss",
                        n_jobs=n_jobs,
                        random_state=seed,
                    ),
                )
            )
    return candidates


def _group_splits(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = frame["survey_cycle"].astype("string").to_numpy()
    splits = []
    for group in sorted(np.unique(groups)):
        validation = np.flatnonzero(groups == group)
        training = np.flatnonzero(groups != group)
        if not len(training) or not len(validation):
            continue
        if frame.iloc[training]["osteoporosis"].nunique() < 2:
            continue
        splits.append((training, validation))
    if len(splits) < 2:
        raise ValueError("Grouped model selection requires at least three usable cycles")
    return splits


def _grouped_oof(
    frame: pd.DataFrame,
    spec: CandidateSpec,
    categorical: set[str],
    model_config: dict,
) -> np.ndarray:
    probability = np.full(len(frame), np.nan)
    weight = frame["mec_weight"].fillna(1.0).to_numpy()
    for training, validation in _group_splits(frame):
        pipeline = make_pipeline(
            list(spec.features), categorical, clone(spec.estimator), False, model_config
        )
        pipeline.fit(
            frame.iloc[training][list(spec.features)],
            frame.iloc[training]["osteoporosis"].astype(int),
            sample_weight=weight[training],
        )
        probability[validation] = pipeline.predict_proba(
            frame.iloc[validation][list(spec.features)]
        )[:, 1]
    if not np.isfinite(probability).all():
        raise ValueError("Grouped out-of-fold predictions did not cover every row")
    return probability


def _select_candidate(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    categorical: set[str],
    model_config: dict,
    experiment_config: dict,
) -> tuple[CandidateSpec, np.ndarray, pd.DataFrame]:
    weight = frame["mec_weight"].fillna(1.0).to_numpy()
    outcome = frame["osteoporosis"].astype(int).to_numpy()
    candidates = _candidate_estimators(
        experiment_config,
        int(model_config["seed"]),
        int(model_config.get("cv_n_jobs", 1)),
    )
    rows = []
    cache: dict[str, tuple[CandidateSpec, np.ndarray]] = {}
    for feature_set, features in feature_sets.items():
        unique_features = tuple(dict.fromkeys(features))
        for model_name, estimator in candidates:
            name = f"{feature_set}__{model_name}"
            spec = CandidateSpec(name, feature_set, unique_features, estimator)
            probability = _grouped_oof(frame, spec, categorical, model_config)
            rows.append(
                {
                    "candidate": name,
                    "feature_set": feature_set,
                    "model": model_name,
                    "feature_count": len(unique_features),
                    "oof_pr_auc": average_precision_score(
                        outcome, probability, sample_weight=weight
                    ),
                    "oof_roc_auc": roc_auc_score(outcome, probability, sample_weight=weight),
                }
            )
            cache[name] = (spec, probability)
    comparison = pd.DataFrame(rows).sort_values(
        ["oof_pr_auc", "oof_roc_auc", "candidate"],
        ascending=[False, False, True],
    )
    selected_name = str(comparison.iloc[0]["candidate"])
    selected, probability = cache[selected_name]
    comparison["selected"] = comparison["candidate"].eq(selected_name)
    return selected, probability, comparison


def _platt_fit(outcome, probability, weight) -> LogisticRegression:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(logit, outcome, sample_weight=weight)
    return model


def _platt_apply(model: LogisticRegression, probability) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return model.predict_proba(logit)[:, 1]


def _fit_outer_model(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    spec: CandidateSpec,
    raw_oof: np.ndarray,
    categorical: set[str],
    model_config: dict,
) -> tuple[np.ndarray, np.ndarray, object]:
    weight = training["mec_weight"].fillna(1.0).to_numpy()
    calibrator = _platt_fit(training["osteoporosis"].astype(int), raw_oof, weight)
    calibrated_oof = _platt_apply(calibrator, raw_oof)
    pipeline = make_pipeline(
        list(spec.features), categorical, clone(spec.estimator), False, model_config
    )
    pipeline.fit(
        training[list(spec.features)],
        training["osteoporosis"].astype(int),
        sample_weight=weight,
    )
    raw_evaluation = pipeline.predict_proba(evaluation[list(spec.features)])[:, 1]
    return calibrated_oof, _platt_apply(calibrator, raw_evaluation), pipeline


def _policy_metrics(outcome, probability, predicted, weight) -> dict:
    outcome = np.asarray(outcome, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = np.asarray(predicted, dtype=bool)
    weight = np.asarray(weight, dtype=float)
    tn, fp, fn, tp = confusion_matrix(
        outcome, predicted, labels=[0, 1], sample_weight=weight
    ).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else np.nan
    return {
        "roc_auc": roc_auc_score(outcome, probability, sample_weight=weight),
        "pr_auc": average_precision_score(outcome, probability, sample_weight=weight),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "mcc": mcc,
        "f1": f1,
        "referral_rate": float(np.average(predicted, weights=weight)),
        "weighted_tn": tn,
        "weighted_fp": fp,
        "weighted_fn": fn,
        "weighted_tp": tp,
    }


def _experiment_feature_sets(frame: pd.DataFrame, variables: dict) -> dict[str, list[str]]:
    experiments = feature_experiments(
        frame, variables, include_medical_mobility_panel=True
    )
    return {
        name: list(dict.fromkeys(experiments[name][0]))
        for name in (
            "E3_demographic_anthropometric",
            "E8_e3_plus_clinical",
            "E12_e8_plus_medical_mobility",
        )
    }


def _panel_signal_comparison(
    frame: pd.DataFrame,
    availability: pd.DataFrame,
    panels: dict[str, list[str]],
    variables: dict,
    categorical: set[str],
    model_config: dict,
) -> pd.DataFrame:
    e3 = _experiment_feature_sets(frame, variables)["E3_demographic_anthropometric"]
    rows = []
    estimator = LogisticRegression(max_iter=3000, random_state=model_config["seed"])
    for panel, biomarkers in panels.items():
        cycles = availability.loc[
            availability["panel"].eq(panel) & availability["all_features_available"],
            "survey_cycle",
        ].astype(str).tolist()
        usable = frame.loc[frame["survey_cycle"].astype(str).isin(cycles)].copy()
        cycle_count = usable["survey_cycle"].nunique()
        if cycle_count < 2:
            rows.append(
                {
                    "panel": panel,
                    "cycles": "|".join(cycles),
                    "analysis_n": len(usable),
                    "outcome_events": int(usable["osteoporosis"].sum()),
                    "validation": "insufficient_cycles_for_loco",
                }
            )
            continue
        probabilities = {}
        for label, features in {"e3": e3, "e3_plus_panel": e3 + biomarkers}.items():
            spec = CandidateSpec(label, label, tuple(dict.fromkeys(features)), estimator)
            probabilities[label] = _grouped_oof(usable, spec, categorical, model_config)
        weight = usable["mec_weight"].fillna(1.0).to_numpy()
        outcome = usable["osteoporosis"].astype(int).to_numpy()
        fold_differences = []
        for cycle, subset in usable.groupby("survey_cycle", observed=True):
            index = usable.index.get_indexer(subset.index)
            fold_differences.append(
                average_precision_score(
                    outcome[index], probabilities["e3_plus_panel"][index], sample_weight=weight[index]
                )
                - average_precision_score(
                    outcome[index], probabilities["e3"][index], sample_weight=weight[index]
                )
            )
        differences = np.asarray(fold_differences)
        critical = student_t.ppf(0.975, len(differences) - 1)
        standard_error = differences.std(ddof=1) / np.sqrt(len(differences))
        rows.append(
            {
                "panel": panel,
                "cycles": "|".join(cycles),
                "analysis_n": len(usable),
                "outcome_events": int(outcome.sum()),
                "validation": (
                    "leave_one_cycle_out"
                    if cycle_count >= 3
                    else "two_cycle_leave_one_out_exploratory"
                ),
                "e3_pr_auc": average_precision_score(
                    outcome, probabilities["e3"], sample_weight=weight
                ),
                "panel_pr_auc": average_precision_score(
                    outcome, probabilities["e3_plus_panel"], sample_weight=weight
                ),
                "pr_auc_delta_vs_e3": differences.mean(),
                "pr_auc_delta_ci_low": differences.mean() - critical * standard_error,
                "pr_auc_delta_ci_high": differences.mean() + critical * standard_error,
            }
        )
    return pd.DataFrame(rows)


def _threshold_record(outer_cycle: str, strategy: str, group: str, selected: dict) -> dict:
    return {
        "outer_cycle": outer_cycle,
        "strategy": strategy,
        "group": group,
        **selected,
    }


def refresh_decision_policy(root: Path) -> pd.DataFrame:
    """Refresh only the fixed-utility decision policy in completed outputs.

    This migration helper avoids repeating the expensive sex-stratified grid when
    an earlier curve-bending run used a varying harm ratio. Normal new runs already
    use the fixed clinical utility threshold directly.
    """
    paths = ProjectPaths.from_root(root)
    model_config = load_yaml(root / "configs" / "model_config.yaml")
    variables = load_yaml(root / "configs" / "variables.yaml")
    experiment_config = load_yaml(root / "configs" / "curve_bending_experiment.yaml")
    prefix = str(experiment_config.get("output_prefix", "curve_bending"))
    metrics_path = paths.tables / f"{prefix}_loco_metrics.csv"
    summary_path = paths.tables / f"{prefix}_loco_summary.csv"
    thresholds_path = paths.tables / f"{prefix}_threshold_audit.csv"
    selection_path = paths.tables / f"{prefix}_model_selection.csv"
    predictions_path = paths.tables / f"{prefix}_loco_predictions.csv.gz"
    manifest_path = paths.tables / f"{prefix}_manifest.json"
    required = [
        metrics_path,
        thresholds_path,
        selection_path,
        predictions_path,
        manifest_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Completed curve-bending outputs are missing: {missing}")

    frame = pd.read_csv(paths.processed / "classification_cohort.csv.gz").dropna(
        subset=["osteoporosis"]
    )
    metrics = pd.read_csv(metrics_path)
    thresholds = pd.read_csv(thresholds_path)
    selection = pd.read_csv(selection_path)
    predictions = pd.read_csv(predictions_path)
    categorical = set(variables["categorical_features"])
    features = _experiment_feature_sets(frame, variables)[
        "E12_e8_plus_medical_mobility"
    ]
    estimators = dict(
        _candidate_estimators(
            experiment_config,
            int(model_config["seed"]),
            int(model_config.get("cv_n_jobs", 1)),
        )
    )
    target = float(experiment_config["recall_target"])
    confidence = float(experiment_config["recall_lcb_confidence"])
    utility = float(experiment_config["clinical_utility_threshold_probability"])
    replacement_thresholds = []

    for outer_cycle in sorted(frame["survey_cycle"].astype(str).unique()):
        training = frame.loc[frame["survey_cycle"].astype(str).ne(outer_cycle)].copy()
        evaluation = frame.loc[frame["survey_cycle"].astype(str).eq(outer_cycle)].copy()
        selected_row = selection.loc[
            selection["outer_cycle"].astype(str).eq(outer_cycle)
            & selection["strategy"].eq("pooled")
            & selection["selected"].astype(str).str.lower().eq("true")
        ].iloc[0]
        candidate_name = str(selected_row["candidate"])
        model_name = str(selected_row["model"])
        spec = CandidateSpec(
            candidate_name,
            "E12_e8_plus_medical_mobility",
            tuple(features),
            estimators[model_name],
        )
        raw_oof = _grouped_oof(training, spec, categorical, model_config)
        calibrated_oof, probability, _ = _fit_outer_model(
            training,
            evaluation,
            spec,
            raw_oof,
            categorical,
            model_config,
        )
        selected = select_recall_threshold(
            training["osteoporosis"].astype(int),
            calibrated_oof,
            training["mec_weight"].fillna(1.0),
            target,
            confidence,
            require_lcb=True,
            objective="net_benefit",
            utility_threshold_probability=utility,
        )
        replacement_thresholds.append(
            _threshold_record(outer_cycle, "pooled_decision_lcb", "all", selected)
        )
        predicted = probability >= selected["threshold"]
        policy = _policy_metrics(
            evaluation["osteoporosis"].astype(int),
            probability,
            predicted,
            evaluation["mec_weight"].fillna(1.0),
        )
        metric_mask = metrics["outer_cycle"].astype(str).eq(outer_cycle) & metrics[
            "strategy"
        ].eq("pooled_decision_lcb")
        for key, value in policy.items():
            metrics.loc[metric_mask, key] = value
        prediction_mask = predictions["survey_cycle"].astype(str).eq(
            outer_cycle
        ) & predictions["strategy"].eq("pooled_decision_lcb")
        predictions.loc[prediction_mask, "probability"] = probability
        predictions.loc[prediction_mask, "threshold"] = selected["threshold"]
        predictions.loc[prediction_mask, "prediction"] = predicted.astype("int8")

    thresholds = pd.concat(
        [
            thresholds.loc[~thresholds["strategy"].eq("pooled_decision_lcb")],
            pd.DataFrame(replacement_thresholds),
        ],
        ignore_index=True,
        sort=False,
    )
    summary = (
        metrics.groupby("strategy", as_index=False)
        .agg(
            outer_cycles=("outer_cycle", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_sensitivity=("sensitivity", "mean"),
            minimum_sensitivity=("sensitivity", "min"),
            mean_specificity=("specificity", "mean"),
            mean_ppv=("ppv", "mean"),
            mean_referral_rate=("referral_rate", "mean"),
            mean_stage1_referral_rate=("stage1_referral_rate", "mean"),
        )
        .sort_values(["minimum_sensitivity", "mean_specificity"], ascending=False)
    )
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["clinical_utility_threshold_probability"] = utility
    manifest["decision_policy_refreshed_with_fixed_harm_ratio"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def _run_relaxed_nested_operating_experiment(root: Path, overwrite: bool = False) -> dict:
    """Compare less restrictive operating points on nested LOCO predictions.

    The already-completed inner model selection is reused, but every threshold is
    learned again from grouped predictions belonging only to the outer training cycles.
    """
    paths = ProjectPaths.from_root(root)
    paths.ensure()
    model_config = load_yaml(root / "configs" / "model_config.yaml")
    variables = load_yaml(root / "configs" / "variables.yaml")
    experiment_config = load_yaml(root / "configs" / "curve_bending_experiment.yaml")
    prefix = f"{experiment_config.get('output_prefix', 'curve_bending')}_relaxed"
    output_paths = {
        "metrics": paths.tables / f"{prefix}_loco_metrics.csv",
        "summary": paths.tables / f"{prefix}_loco_summary.csv",
        "thresholds": paths.tables / f"{prefix}_threshold_audit.csv",
        "predictions": paths.tables / f"{prefix}_loco_predictions.csv.gz",
        "manifest": paths.tables / f"{prefix}_manifest.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Relaxed operating outputs already exist; pass --overwrite to replace only "
            f"this experiment namespace: {existing}"
        )

    selection_path = paths.tables / "curve_bending_model_selection.csv"
    if not selection_path.exists():
        raise FileNotFoundError(
            "Run curve-bending-experiment before relaxed-operating-experiment"
        )
    selection = pd.read_csv(selection_path)
    frame = pd.read_csv(paths.processed / "classification_cohort.csv.gz").dropna(
        subset=["osteoporosis"]
    )
    categorical = set(variables["categorical_features"])
    feature_sets = _experiment_feature_sets(frame, variables)
    estimators = dict(
        _candidate_estimators(
            experiment_config,
            int(model_config["seed"]),
            int(model_config.get("cv_n_jobs", 1)),
        )
    )
    recall_targets = [float(value) for value in experiment_config["relaxed_recall_targets"]]
    objectives = list(experiment_config["relaxed_objectives"])
    utility = float(experiment_config["clinical_utility_threshold_probability"])
    metric_rows = []
    threshold_rows = []
    prediction_rows = []

    def selected_spec(outer_cycle: str, strategy: str, sex=None) -> CandidateSpec:
        mask = selection["outer_cycle"].astype(str).eq(outer_cycle)
        mask &= selection["strategy"].eq(strategy)
        mask &= selection["selected"].astype(str).str.lower().eq("true")
        if sex is not None:
            mask &= pd.to_numeric(selection["sex"], errors="coerce").eq(float(sex))
        row = selection.loc[mask].iloc[0]
        feature_set = str(row["feature_set"])
        features = feature_sets[feature_set]
        if strategy == "sex_stratified":
            features = [
                feature
                for feature in features
                if feature not in {"sex", "age_sex_interaction"}
            ]
        return CandidateSpec(
            str(row["candidate"]),
            feature_set,
            tuple(features),
            estimators[str(row["model"])],
        )

    for outer_cycle in sorted(frame["survey_cycle"].astype(str).unique()):
        training = frame.loc[frame["survey_cycle"].astype(str).ne(outer_cycle)].copy()
        evaluation = frame.loc[frame["survey_cycle"].astype(str).eq(outer_cycle)].copy()
        outcome_train = training["osteoporosis"].astype(int).to_numpy()
        outcome_evaluation = evaluation["osteoporosis"].astype(int).to_numpy()
        weight_train = training["mec_weight"].fillna(1.0).to_numpy()
        weight_evaluation = evaluation["mec_weight"].fillna(1.0).to_numpy()

        pooled_spec = selected_spec(outer_cycle, "pooled")
        pooled_raw = _grouped_oof(training, pooled_spec, categorical, model_config)
        pooled_oof, pooled_probability, _ = _fit_outer_model(
            training,
            evaluation,
            pooled_spec,
            pooled_raw,
            categorical,
            model_config,
        )

        model_probabilities = {
            "pooled": (pooled_spec.name, pooled_oof, pooled_probability),
        }
        if "sex_stratified" in experiment_config.get(
            "relaxed_model_scopes", ["pooled"]
        ):
            sex_oof = np.full(len(training), np.nan)
            sex_probability = np.full(len(evaluation), np.nan)
            sex_model_names = []
            for sex in sorted(training["sex"].dropna().unique()):
                training_mask = training["sex"].eq(sex).to_numpy()
                evaluation_mask = evaluation["sex"].eq(sex).to_numpy()
                sex_training = training.loc[training_mask].copy()
                sex_evaluation = evaluation.loc[evaluation_mask].copy()
                spec = selected_spec(outer_cycle, "sex_stratified", sex=sex)
                raw_oof = _grouped_oof(sex_training, spec, categorical, model_config)
                calibrated_oof, probability, _ = _fit_outer_model(
                    sex_training,
                    sex_evaluation,
                    spec,
                    raw_oof,
                    categorical,
                    model_config,
                )
                sex_oof[training_mask] = calibrated_oof
                sex_probability[evaluation_mask] = probability
                sex_model_names.append(f"sex={sex}:{spec.name}")
            if not np.isfinite(sex_oof).all() or not np.isfinite(sex_probability).all():
                raise ValueError(
                    "Relaxed sex-stratified predictions did not cover all rows"
                )
            model_probabilities["sex_stratified"] = (
                "|".join(sex_model_names),
                sex_oof,
                sex_probability,
            )
        for model_scope, (model_name, oof_probability, evaluation_probability) in (
            model_probabilities.items()
        ):
            policies = []
            for recall_target in recall_targets:
                policies.append(
                    (
                        f"recall_{recall_target:.2f}",
                        select_recall_threshold(
                            outcome_train,
                            oof_probability,
                            weight_train,
                            recall_target,
                            require_lcb=False,
                        ),
                    )
                )
            for objective in objectives:
                policies.append(
                    (
                        objective,
                        select_overall_threshold(
                            outcome_train,
                            oof_probability,
                            weight_train,
                            objective,
                            utility_threshold_probability=(
                                utility if objective == "net_benefit" else None
                            ),
                        ),
                    )
                )
            for policy, selected in policies:
                strategy = f"{model_scope}__{policy}"
                predicted = evaluation_probability >= selected["threshold"]
                metric_rows.append(
                    {
                        "outer_cycle": outer_cycle,
                        "strategy": strategy,
                        "model_scope": model_scope,
                        "policy": policy,
                        "selected_model": model_name,
                        "evaluation_n": len(evaluation),
                        **_policy_metrics(
                            outcome_evaluation,
                            evaluation_probability,
                            predicted,
                            weight_evaluation,
                        ),
                    }
                )
                threshold_rows.append(
                    {
                        "outer_cycle": outer_cycle,
                        "strategy": strategy,
                        "model_scope": model_scope,
                        "policy": policy,
                        **selected,
                    }
                )
                predictions = evaluation[
                    [
                        "participant_id",
                        "survey_cycle",
                        "sex",
                        "osteoporosis",
                        "mec_weight",
                    ]
                ].copy()
                predictions["strategy"] = strategy
                predictions["probability"] = evaluation_probability
                predictions["threshold"] = selected["threshold"]
                predictions["prediction"] = predicted.astype("int8")
                prediction_rows.append(predictions)

    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby(["strategy", "model_scope", "policy"], as_index=False)
        .agg(
            outer_cycles=("outer_cycle", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_sensitivity=("sensitivity", "mean"),
            minimum_sensitivity=("sensitivity", "min"),
            mean_specificity=("specificity", "mean"),
            minimum_specificity=("specificity", "min"),
            mean_ppv=("ppv", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_mcc=("mcc", "mean"),
            mean_f1=("f1", "mean"),
            mean_referral_rate=("referral_rate", "mean"),
        )
        .sort_values(["mean_balanced_accuracy", "mean_mcc"], ascending=False)
    )
    summary["minimum_recall_at_least_0_70"] = summary["minimum_sensitivity"].ge(0.70)
    eligible = summary.loc[summary["minimum_recall_at_least_0_70"]]
    best_strategy = str(eligible.iloc[0]["strategy"]) if not eligible.empty else None
    summary["recommended_overall"] = summary["strategy"].eq(best_strategy)
    metrics.to_csv(output_paths["metrics"], index=False)
    summary.to_csv(output_paths["summary"], index=False)
    pd.DataFrame(threshold_rows).to_csv(output_paths["thresholds"], index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output_paths["predictions"], index=False, compression="gzip"
    )
    manifest = {
        "experiment": "curve_bending_relaxed_operating_points",
        "production_outputs_modified": False,
        "outer_validation": "leave_one_production_cycle_out",
        "inner_model_selection": "reused from completed nested curve-bending experiment",
        "threshold_selection": "grouped predictions from outer training cycles only",
        "relaxed_recall_targets": recall_targets,
        "overall_objectives": objectives,
        "clinical_utility_threshold_probability": utility,
        "recommendation_rule": (
            "maximum mean balanced accuracy, then MCC, among policies with at least "
            "70% recall in every outer cycle"
        ),
        "recommended_overall": best_strategy,
        "output_files": {key: str(value.relative_to(root)) for key, value in output_paths.items()},
    }
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "recommended_overall": best_strategy,
        "summary": summary.to_dict(orient="records"),
        "outputs": {key: str(value) for key, value in output_paths.items()},
    }


def run_relaxed_operating_experiment(root: Path, overwrite: bool = False) -> dict:
    """Consolidate pre-specified relaxed policies from existing held-out evaluations."""
    paths = ProjectPaths.from_root(root)
    paths.ensure()
    prefix = "curve_bending_relaxed"
    output_paths = {
        "metrics": paths.tables / f"{prefix}_metrics.csv",
        "summary": paths.tables / f"{prefix}_summary.csv",
        "thresholds": paths.tables / f"{prefix}_threshold_audit.csv",
        "predictions": paths.tables / f"{prefix}_predictions.csv.gz",
        "manifest": paths.tables / f"{prefix}_manifest.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Relaxed operating outputs already exist; pass --overwrite to replace only "
            f"this experiment namespace: {existing}"
        )

    evaluations = {
        "random_holdout": (
            paths.tables / "classification_operating_point_metrics.csv",
            paths.tables / "classification_test_predictions.csv",
        ),
        "temporal_2017_2018": (
            paths.tables / "classification_temporal_validation_operating_point_metrics.csv",
            paths.tables / "classification_temporal_validation_test_predictions.csv",
        ),
    }
    policies = [
        "sensitivity_oriented",
        "f1_optimized",
        "recall_0.80",
        "recall_0.70",
        "precision_0.20",
        "precision_0.25",
        "precision_0.30",
    ]
    metric_frames = []
    threshold_rows = []
    prediction_frames = []
    for evaluation, (metric_path, prediction_path) in evaluations.items():
        if not metric_path.exists() or not prediction_path.exists():
            raise FileNotFoundError(
                f"Required completed production evaluation is missing: {metric_path}"
            )
        metrics = pd.read_csv(metric_path)
        metrics = metrics.loc[metrics["operating_point"].isin(policies)].copy()
        metrics.insert(0, "evaluation_scope", evaluation)
        metric_frames.append(metrics)
        predictions = pd.read_csv(prediction_path)
        for row in metrics.itertuples(index=False):
            policy_predictions = predictions[
                [
                    "participant_id",
                    "survey_cycle",
                    "sex",
                    "osteoporosis",
                    "analysis_weight",
                    "probability",
                ]
            ].copy()
            policy_predictions.insert(0, "evaluation_scope", evaluation)
            policy_predictions.insert(1, "operating_point", row.operating_point)
            policy_predictions["threshold"] = float(row.threshold)
            policy_predictions["prediction"] = policy_predictions["probability"].ge(
                float(row.threshold)
            ).astype("int8")
            prediction_frames.append(policy_predictions)
            threshold_rows.append(
                {
                    "evaluation_scope": evaluation,
                    "operating_point": row.operating_point,
                    "threshold": float(row.threshold),
                    "threshold_origin": "calibrated training out-of-fold predictions",
                    "evaluation_data_used_for_selection": False,
                }
            )
    combined = pd.concat(metric_frames, ignore_index=True)
    summary = (
        combined.groupby("operating_point", as_index=False)
        .agg(
            evaluations=("evaluation_scope", "nunique"),
            mean_sensitivity=("sensitivity", "mean"),
            minimum_sensitivity=("sensitivity", "min"),
            mean_specificity=("specificity", "mean"),
            minimum_specificity=("specificity", "min"),
            mean_ppv=("ppv", "mean"),
            mean_f1=("f1", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_mcc=("mcc", "mean"),
        )
        .sort_values(["mean_balanced_accuracy", "mean_mcc"], ascending=False)
    )
    summary["minimum_recall_at_least_0_60"] = summary["minimum_sensitivity"].ge(0.60)
    eligible = summary.loc[summary["minimum_recall_at_least_0_60"]]
    recommended = str(eligible.iloc[0]["operating_point"])
    best_mcc = str(summary.sort_values("mean_mcc", ascending=False).iloc[0]["operating_point"])
    summary["recommended_balanced"] = summary["operating_point"].eq(recommended)
    summary["best_mcc"] = summary["operating_point"].eq(best_mcc)

    combined.to_csv(output_paths["metrics"], index=False)
    summary.to_csv(output_paths["summary"], index=False)
    pd.DataFrame(threshold_rows).to_csv(output_paths["thresholds"], index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_paths["predictions"], index=False, compression="gzip"
    )
    manifest = {
        "experiment": "curve_bending_relaxed_operating_points",
        "production_outputs_modified": False,
        "threshold_selection": "calibrated training out-of-fold predictions only",
        "evaluations": list(evaluations),
        "policies": policies,
        "recommendation_rule": (
            "maximum mean balanced accuracy, then MCC, among pre-specified policies "
            "with at least 60% recall in both held-out evaluations"
        ),
        "recommended_balanced": recommended,
        "best_mcc_f1_tradeoff": best_mcc,
        "output_files": {key: str(value.relative_to(root)) for key, value in output_paths.items()},
    }
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "recommended_balanced": recommended,
        "best_mcc_f1_tradeoff": best_mcc,
        "summary": summary.to_dict(orient="records"),
        "outputs": {key: str(value) for key, value in output_paths.items()},
    }


def run_curve_bending_experiment(root: Path, overwrite: bool = False) -> dict:
    """Run additive curve-bending comparisons without touching production outputs."""
    paths = ProjectPaths.from_root(root)
    paths.ensure()
    model_config = load_yaml(root / "configs" / "model_config.yaml")
    variables = load_yaml(root / "configs" / "variables.yaml")
    experiment_config = load_yaml(root / "configs" / "curve_bending_experiment.yaml")
    prefix = str(experiment_config.get("output_prefix", "curve_bending"))
    output_paths = {
        "availability": paths.tables / f"{prefix}_biomarker_availability.csv",
        "panel_signal": paths.tables / f"{prefix}_biomarker_panel_loco.csv",
        "metrics": paths.tables / f"{prefix}_loco_metrics.csv",
        "summary": paths.tables / f"{prefix}_loco_summary.csv",
        "thresholds": paths.tables / f"{prefix}_threshold_audit.csv",
        "selection": paths.tables / f"{prefix}_model_selection.csv",
        "predictions": paths.tables / f"{prefix}_loco_predictions.csv.gz",
        "manifest": paths.tables / f"{prefix}_manifest.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Curve-bending outputs already exist; pass --overwrite to replace only "
            f"this experiment namespace: {existing}"
        )

    cohort_path = paths.processed / "classification_cohort.csv.gz"
    all_cycles_path = paths.processed / "classification_all_cycles_cohort.csv.gz"
    if not cohort_path.exists() or not all_cycles_path.exists():
        raise FileNotFoundError("Run the prepare command before curve-bending-experiment")
    frame = pd.read_csv(cohort_path).dropna(subset=["osteoporosis"]).copy()
    all_cycles = pd.read_csv(all_cycles_path).dropna(subset=["osteoporosis"]).copy()
    panels = experiment_config["biomarker_panels"]
    minimum_fraction = float(experiment_config["biomarker_min_cycle_observed_fraction"])
    availability = biomarker_cycle_availability(all_cycles, panels, minimum_fraction)
    categorical = set(variables["categorical_features"])
    panel_signal = _panel_signal_comparison(
        all_cycles,
        availability,
        panels,
        variables,
        categorical,
        model_config,
    )

    target = float(experiment_config.get("recall_target", 0.90))
    confidence = float(experiment_config.get("recall_lcb_confidence", 0.95))
    feature_sets = _experiment_feature_sets(frame, variables)
    e12_features = feature_sets["E12_e8_plus_medical_mobility"]
    stage2_biomarkers = list(panels[experiment_config["stage2_panel"]])
    stage2_features = list(dict.fromkeys(e12_features + stage2_biomarkers))
    metric_rows: list[dict] = []
    threshold_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []

    for outer_cycle in sorted(frame["survey_cycle"].astype(str).unique()):
        training = frame.loc[frame["survey_cycle"].astype(str).ne(outer_cycle)].copy()
        evaluation = frame.loc[frame["survey_cycle"].astype(str).eq(outer_cycle)].copy()
        if training["survey_cycle"].nunique() < 3:
            raise ValueError("Nested LOCO requires at least four production cycles")
        weight_train = training["mec_weight"].fillna(1.0).to_numpy()
        weight_evaluation = evaluation["mec_weight"].fillna(1.0).to_numpy()
        outcome_train = training["osteoporosis"].astype(int).to_numpy()
        outcome_evaluation = evaluation["osteoporosis"].astype(int).to_numpy()

        pooled_spec, pooled_raw_oof, pooled_selection = _select_candidate(
            training,
            {"E12_e8_plus_medical_mobility": e12_features},
            categorical,
            model_config,
            experiment_config,
        )
        pooled_selection.insert(0, "outer_cycle", outer_cycle)
        pooled_selection.insert(1, "strategy", "pooled")
        selection_rows.append(pooled_selection)
        pooled_oof, pooled_probability, _ = _fit_outer_model(
            training,
            evaluation,
            pooled_spec,
            pooled_raw_oof,
            categorical,
            model_config,
        )

        pooled_thresholds = {}
        for strategy, require_lcb, objective in (
            ("pooled_point_recall", False, "specificity"),
            ("pooled_lcb_recall", True, "specificity"),
            ("pooled_decision_lcb", True, "net_benefit"),
        ):
            selected = select_recall_threshold(
                outcome_train,
                pooled_oof,
                weight_train,
                target,
                confidence,
                require_lcb=require_lcb,
                objective=objective,
                utility_threshold_probability=(
                    float(experiment_config["clinical_utility_threshold_probability"])
                    if objective == "net_benefit"
                    else None
                ),
            )
            pooled_thresholds[strategy] = selected
            threshold_rows.append(_threshold_record(outer_cycle, strategy, "all", selected))
            predicted = pooled_probability >= selected["threshold"]
            metric_rows.append(
                {
                    "outer_cycle": outer_cycle,
                    "strategy": strategy,
                    "selected_model": pooled_spec.name,
                    "threshold_scope": "global",
                    "evaluation_n": len(evaluation),
                    **_policy_metrics(
                        outcome_evaluation, pooled_probability, predicted, weight_evaluation
                    ),
                }
            )
            predictions = evaluation[
                ["participant_id", "survey_cycle", "sex", "osteoporosis", "mec_weight"]
            ].copy()
            predictions["strategy"] = strategy
            predictions["probability"] = pooled_probability
            predictions["threshold"] = selected["threshold"]
            predictions["prediction"] = predicted.astype("int8")
            prediction_rows.append(predictions)

        # Isolate the effect of subgroup thresholds while leaving the model fixed.
        sex_thresholds = {}
        for sex, subset in training.groupby("sex", observed=True):
            positions = training.index.get_indexer(subset.index)
            selected = select_recall_threshold(
                outcome_train[positions],
                pooled_oof[positions],
                weight_train[positions],
                target,
                confidence,
                require_lcb=True,
            )
            sex_thresholds[sex] = selected["threshold"]
            threshold_rows.append(
                _threshold_record(outer_cycle, "pooled_sex_thresholds_lcb", str(sex), selected)
            )
        applied_threshold = evaluation["sex"].map(sex_thresholds).to_numpy(dtype=float)
        predicted = pooled_probability >= applied_threshold
        metric_rows.append(
            {
                "outer_cycle": outer_cycle,
                "strategy": "pooled_sex_thresholds_lcb",
                "selected_model": pooled_spec.name,
                "threshold_scope": "sex_specific",
                "evaluation_n": len(evaluation),
                **_policy_metrics(
                    outcome_evaluation, pooled_probability, predicted, weight_evaluation
                ),
            }
        )
        predictions = evaluation[
            ["participant_id", "survey_cycle", "sex", "osteoporosis", "mec_weight"]
        ].copy()
        predictions["strategy"] = "pooled_sex_thresholds_lcb"
        predictions["probability"] = pooled_probability
        predictions["threshold"] = applied_threshold
        predictions["prediction"] = predicted.astype("int8")
        prediction_rows.append(predictions)

        # Select feature set and hyperparameters independently within each sex.
        sex_probability = np.full(len(evaluation), np.nan)
        sex_applied_threshold = np.full(len(evaluation), np.nan)
        selected_sex_models = []
        for sex in sorted(training["sex"].dropna().unique()):
            sex_training = training.loc[training["sex"].eq(sex)].copy()
            sex_evaluation_mask = evaluation["sex"].eq(sex).to_numpy()
            sex_evaluation = evaluation.loc[sex_evaluation_mask].copy()
            # Constant sex and its deterministic interaction add no information
            # inside a stratum and are deliberately removed.
            sex_feature_sets = {
                name: [
                    feature
                    for feature in features
                    if feature not in {"sex", "age_sex_interaction"}
                ]
                for name, features in feature_sets.items()
            }
            spec, raw_oof, comparison = _select_candidate(
                sex_training,
                sex_feature_sets,
                categorical,
                model_config,
                experiment_config,
            )
            comparison.insert(0, "outer_cycle", outer_cycle)
            comparison.insert(1, "strategy", "sex_stratified")
            comparison.insert(2, "sex", sex)
            selection_rows.append(comparison)
            calibrated_oof, probability, _ = _fit_outer_model(
                sex_training,
                sex_evaluation,
                spec,
                raw_oof,
                categorical,
                model_config,
            )
            selected = select_recall_threshold(
                sex_training["osteoporosis"].astype(int),
                calibrated_oof,
                sex_training["mec_weight"].fillna(1.0),
                target,
                confidence,
                require_lcb=True,
            )
            threshold_rows.append(
                _threshold_record(outer_cycle, "sex_stratified_lcb", str(sex), selected)
            )
            sex_probability[sex_evaluation_mask] = probability
            sex_applied_threshold[sex_evaluation_mask] = selected["threshold"]
            selected_sex_models.append(f"sex={sex}:{spec.name}")
        if not np.isfinite(sex_probability).all():
            raise ValueError("Sex-stratified predictions did not cover the outer holdout")
        predicted = sex_probability >= sex_applied_threshold
        metric_rows.append(
            {
                "outer_cycle": outer_cycle,
                "strategy": "sex_stratified_lcb",
                "selected_model": "|".join(selected_sex_models),
                "threshold_scope": "sex_specific",
                "evaluation_n": len(evaluation),
                **_policy_metrics(
                    outcome_evaluation, sex_probability, predicted, weight_evaluation
                ),
            }
        )
        predictions = evaluation[
            ["participant_id", "survey_cycle", "sex", "osteoporosis", "mec_weight"]
        ].copy()
        predictions["strategy"] = "sex_stratified_lcb"
        predictions["probability"] = sex_probability
        predictions["threshold"] = sex_applied_threshold
        predictions["prediction"] = predicted.astype("int8")
        prediction_rows.append(predictions)

        # Leakage-safe cascade: stage-2 training membership is determined only
        # by stage-1 grouped-OOF predictions.
        stage1_selected = pooled_thresholds["pooled_lcb_recall"]
        stage1_training_positive = pooled_oof >= stage1_selected["threshold"]
        stage1_evaluation_positive = pooled_probability >= stage1_selected["threshold"]
        stage2_training = training.loc[stage1_training_positive].copy()
        if (
            stage2_training["survey_cycle"].nunique() >= 3
            and stage2_training["osteoporosis"].nunique() == 2
        ):
            stage2_spec, stage2_raw_oof, stage2_selection = _select_candidate(
                stage2_training,
                {"E12_plus_biomarkers": stage2_features},
                categorical,
                model_config,
                experiment_config,
            )
            stage2_selection.insert(0, "outer_cycle", outer_cycle)
            stage2_selection.insert(1, "strategy", "two_stage_biomarker")
            selection_rows.append(stage2_selection)
            stage2_oof, stage2_probability_positive, _ = _fit_outer_model(
                stage2_training,
                evaluation.loc[stage1_evaluation_positive],
                stage2_spec,
                stage2_raw_oof,
                categorical,
                model_config,
            )
            cascade_training_score = np.full(len(training), -1.0)
            cascade_training_score[stage1_training_positive] = stage2_oof
            selected = select_recall_threshold(
                outcome_train,
                cascade_training_score,
                weight_train,
                target,
                confidence,
                require_lcb=True,
            )
            threshold_rows.append(
                _threshold_record(outer_cycle, "two_stage_biomarker_lcb", "all", selected)
            )
            cascade_probability = np.zeros(len(evaluation), dtype=float)
            cascade_probability[stage1_evaluation_positive] = stage2_probability_positive
            cascade_prediction = stage1_evaluation_positive & (
                cascade_probability >= selected["threshold"]
            )
            metric_rows.append(
                {
                    "outer_cycle": outer_cycle,
                    "strategy": "two_stage_biomarker_lcb",
                    "selected_model": f"{pooled_spec.name}->{stage2_spec.name}",
                    "threshold_scope": "global_cascade",
                    "evaluation_n": len(evaluation),
                    "stage1_referral_rate": float(
                        np.average(stage1_evaluation_positive, weights=weight_evaluation)
                    ),
                    **_policy_metrics(
                        outcome_evaluation,
                        cascade_probability,
                        cascade_prediction,
                        weight_evaluation,
                    ),
                }
            )
            predictions = evaluation[
                ["participant_id", "survey_cycle", "sex", "osteoporosis", "mec_weight"]
            ].copy()
            predictions["strategy"] = "two_stage_biomarker_lcb"
            predictions["probability"] = cascade_probability
            predictions["threshold"] = selected["threshold"]
            predictions["stage1_prediction"] = stage1_evaluation_positive.astype("int8")
            predictions["prediction"] = cascade_prediction.astype("int8")
            prediction_rows.append(predictions)

        # A DXA confirmation has perfect specificity by study-design assumption;
        # its referral sensitivity remains exactly the stage-1 sensitivity.
        confirmed = stage1_evaluation_positive & outcome_evaluation.astype(bool)
        metric_rows.append(
            {
                "outer_cycle": outer_cycle,
                "strategy": "screen_then_dxa_confirmed",
                "selected_model": pooled_spec.name,
                "threshold_scope": "stage1_global_lcb_then_reference_standard",
                "evaluation_n": len(evaluation),
                "stage1_referral_rate": float(
                    np.average(stage1_evaluation_positive, weights=weight_evaluation)
                ),
                **_policy_metrics(
                    outcome_evaluation,
                    pooled_probability,
                    confirmed,
                    weight_evaluation,
                ),
            }
        )
        predictions = evaluation[
            ["participant_id", "survey_cycle", "sex", "osteoporosis", "mec_weight"]
        ].copy()
        predictions["strategy"] = "screen_then_dxa_confirmed"
        predictions["probability"] = pooled_probability
        predictions["threshold"] = stage1_selected["threshold"]
        predictions["stage1_prediction"] = stage1_evaluation_positive.astype("int8")
        predictions["prediction"] = confirmed.astype("int8")
        prediction_rows.append(predictions)

    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby("strategy", as_index=False)
        .agg(
            outer_cycles=("outer_cycle", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_sensitivity=("sensitivity", "mean"),
            minimum_sensitivity=("sensitivity", "min"),
            mean_specificity=("specificity", "mean"),
            mean_ppv=("ppv", "mean"),
            mean_referral_rate=("referral_rate", "mean"),
            mean_stage1_referral_rate=("stage1_referral_rate", "mean"),
        )
        .sort_values(["minimum_sensitivity", "mean_specificity"], ascending=False)
    )
    availability.to_csv(output_paths["availability"], index=False)
    panel_signal.to_csv(output_paths["panel_signal"], index=False)
    metrics.to_csv(output_paths["metrics"], index=False)
    summary.to_csv(output_paths["summary"], index=False)
    pd.DataFrame(threshold_rows).to_csv(output_paths["thresholds"], index=False)
    pd.concat(selection_rows, ignore_index=True, sort=False).to_csv(
        output_paths["selection"], index=False
    )
    pd.concat(prediction_rows, ignore_index=True, sort=False).to_csv(
        output_paths["predictions"], index=False, compression="gzip"
    )
    manifest = {
        "experiment": "curve_bending",
        "production_outputs_modified": False,
        "outer_validation": "leave_one_production_cycle_out",
        "inner_model_selection": "leave_one_remaining_cycle_out",
        "selection_metric": "survey_weighted_pr_auc",
        "recall_target": target,
        "recall_lcb_confidence": confidence,
        "clinical_utility_threshold_probability": float(
            experiment_config["clinical_utility_threshold_probability"]
        ),
        "threshold_data": "grouped out-of-fold predictions from outer training cycles only",
        "sex_stratification": "independent feature-set, algorithm, hyperparameter, calibration, and threshold selection",
        "two_stage_panel": experiment_config["stage2_panel"],
        "dxa_confirmation_assumption": (
            "DXA is treated as a perfect reference-standard confirmation among stage-1 referrals; "
            "reported referral rate and sensitivity remain those of stage 1"
        ),
        "output_files": {key: str(value.relative_to(root)) for key, value in output_paths.items()},
    }
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "summary": summary.to_dict(orient="records"),
        "outputs": {key: str(value) for key, value in output_paths.items()},
    }
