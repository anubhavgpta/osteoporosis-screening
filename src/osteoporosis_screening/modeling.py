from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import t as student_t

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover - dependency availability is checked in reports
    XGBClassifier = None
    XGBRegressor = None

from .evaluation import (
    classification_metrics,
    regression_metrics,
    threshold_for_f1,
    threshold_for_precision,
    threshold_for_sensitivity,
)
from .nutrition import NUTRITION_CLUSTER_FEATURES, NutritionClusterAugmenter


set_config(enable_metadata_routing=True)


def _weighted_scorer(name: str):
    return get_scorer(name).set_score_request(sample_weight=True)


@dataclass
class ClassificationResult:
    model: object
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    comparison: pd.DataFrame
    thresholds: pd.DataFrame


@dataclass
class RegressionResult:
    model: object
    metrics: dict
    predictions: pd.DataFrame
    comparison: pd.DataFrame


def feature_experiments(
    frame: pd.DataFrame,
    variables: dict,
    include_targeted_biomarkers: bool = False,
    include_published_lab_panel: bool = False,
    include_medical_mobility_panel: bool = False,
) -> dict:
    groups = variables["feature_groups"]
    demographic = groups["demographic"]
    if "age_sex_interaction" in frame:
        demographic = demographic + ["age_sex_interaction"]
    anthropometric = groups["anthropometric"]
    lifestyle = groups["lifestyle"]
    nutrition = groups["nutrition"]
    clinical = groups.get("clinical", [])
    experiments = {
        "E1_age_sex": (["age", "sex"], False),
        "E2_demographic": (demographic, False),
        "E3_demographic_anthropometric": (demographic + anthropometric, False),
        "E4_demographic_anthropometric_lifestyle": (
            demographic + anthropometric + lifestyle,
            False,
        ),
        "E5_demographic_anthropometric_nutrition": (
            demographic + anthropometric + nutrition,
            False,
        ),
        "E6_all_raw": (demographic + anthropometric + lifestyle + nutrition, False),
        "E7_all_plus_nutrition_cluster": (
            demographic + anthropometric + lifestyle + nutrition,
            True,
        ),
        "E8_e3_plus_clinical": (demographic + anthropometric + clinical, False),
    }
    if include_targeted_biomarkers:
        grip = groups.get("grip", [])
        biomarkers = groups.get("biomarkers", [])
        experiments.update(
            {
                "E9_e3_plus_grip": (
                    demographic + anthropometric + grip,
                    False,
                ),
                "E10_e3_plus_grip_biomarkers": (
                    demographic + anthropometric + grip + biomarkers,
                    False,
                ),
            }
        )
    if include_published_lab_panel and groups.get("published_low_bmd_panel"):
        experiments["E11_published_low_bmd_lab_panel"] = (
            groups["published_low_bmd_panel"],
            False,
        )
    if include_medical_mobility_panel:
        experiments["E12_e8_plus_medical_mobility"] = (
            demographic
            + anthropometric
            + clinical
            + groups.get("medical_mobility", []),
            False,
        )
    return experiments


def run_classification_ablations(
    frame: pd.DataFrame,
    variables: dict,
    categorical: set[str],
    config: dict,
    include_targeted_biomarkers: bool = False,
    include_published_lab_panel: bool = False,
    include_medical_mobility_panel: bool = False,
    target: str = "osteoporosis",
) -> pd.DataFrame:
    experiments = feature_experiments(
        frame,
        variables,
        include_targeted_biomarkers=include_targeted_biomarkers,
        include_published_lab_panel=include_published_lab_panel,
        include_medical_mobility_panel=include_medical_mobility_panel,
    )
    nutrition = set(variables["feature_groups"].get("nutrition", []))
    base = frame.dropna(subset=[target]).copy()
    rows = []
    fold_roc_auc: dict[tuple[str, str], np.ndarray] = {}
    fold_pr_auc: dict[tuple[str, str], np.ndarray] = {}
    reference_features, _ = experiments["E3_demographic_anthropometric"]

    def analysis_data(experiment: str, features: list[str], add_clusters: bool):
        dietary = add_clusters or bool(set(features) & nutrition)
        if experiment == "E11_published_low_bmd_lab_panel":
            usable = base.loc[
                base["survey_cycle"].isin(config["published_panel_cycles"])
            ].copy()
            weights = usable["mec_weight"]
            cohort = "published_panel_cycles"
            weight_source = "WTMEC2YR"
        elif dietary:
            usable = base.loc[base["dietary_recall_status"].eq(1)].copy()
            weights = usable["dietary_day1_weight"].fillna(usable["mec_weight"])
            cohort = "reliable_day1_diet"
            weight_source = "WTDRD1"
        else:
            usable = base
            weights = usable["mec_weight"]
            cohort = "mec_exam"
            weight_source = "WTMEC2YR"
        return usable, weights.fillna(1.0), cohort, weight_source

    def score(features: list[str], add_clusters: bool, usable, weights):
        y = usable[target].astype(int)
        folds = min(config["cv_folds"], int(y.value_counts().min()))
        cv = StratifiedKFold(
            n_splits=max(2, folds), shuffle=True, random_state=config["seed"]
        )
        pipeline = make_pipeline(
            features,
            categorical,
            LogisticRegression(max_iter=3000, random_state=config["seed"]),
            add_clusters,
            config,
        )
        return cross_validate(
            pipeline,
            usable[features],
            y,
            cv=cv,
            scoring={
                "roc_auc": _weighted_scorer("roc_auc"),
                "pr_auc": _weighted_scorer("average_precision"),
            },
            n_jobs=config.get("cv_n_jobs", 1),
            params={"sample_weight": weights.to_numpy()},
        )

    cohort_references = {}
    cohort_e8_references = {}
    cohort_pr_references = {}
    cohort_e8_pr_references = {}
    for experiment, (features, add_clusters) in experiments.items():
        features = list(dict.fromkeys(features))
        usable, weights, cohort, weight_source = analysis_data(
            experiment, features, add_clusters
        )
        scores = score(features, add_clusters, usable, weights)
        fold_roc_auc[(experiment, cohort)] = scores["test_roc_auc"]
        fold_pr_auc[(experiment, cohort)] = scores["test_pr_auc"]
        if cohort not in cohort_references:
            reference_scores = score(
                list(dict.fromkeys(reference_features)), False, usable, weights
            )
            cohort_references[cohort] = reference_scores["test_roc_auc"]
            cohort_pr_references[cohort] = reference_scores["test_pr_auc"]
        if cohort not in cohort_e8_references:
            e8_features, _ = experiments["E8_e3_plus_clinical"]
            e8_reference_scores = score(
                list(dict.fromkeys(e8_features)), False, usable, weights
            )
            cohort_e8_references[cohort] = e8_reference_scores["test_roc_auc"]
            cohort_e8_pr_references[cohort] = e8_reference_scores["test_pr_auc"]
        critical = student_t.ppf(0.975, len(scores["test_roc_auc"]) - 1)
        roc_standard_error = scores["test_roc_auc"].std(ddof=1) / np.sqrt(
            len(scores["test_roc_auc"])
        )
        rows.append(
            {
                "target": target,
                "experiment": experiment,
                "analysis_cohort": cohort,
                "analysis_n": len(usable),
                "outcome_events": int(
                    usable[target].astype(int).sum()
                ),
                "sample_weight": weight_source,
                "feature_count": len(features) + int(add_clusters),
                "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
                "cv_roc_auc_sd": scores["test_roc_auc"].std(),
                "cv_roc_auc_ci_low": scores["test_roc_auc"].mean()
                - critical * roc_standard_error,
                "cv_roc_auc_ci_high": scores["test_roc_auc"].mean()
                + critical * roc_standard_error,
                "cv_pr_auc_mean": scores["test_pr_auc"].mean(),
                "cv_pr_auc_sd": scores["test_pr_auc"].std(),
                "cv_pr_auc_ci_low": scores["test_pr_auc"].mean()
                - critical
                * scores["test_pr_auc"].std(ddof=1)
                / np.sqrt(len(scores["test_pr_auc"])),
                "cv_pr_auc_ci_high": scores["test_pr_auc"].mean()
                + critical
                * scores["test_pr_auc"].std(ddof=1)
                / np.sqrt(len(scores["test_pr_auc"])),
            }
        )
    result = pd.DataFrame(rows)
    for (experiment, cohort), values in fold_roc_auc.items():
        reference = cohort_references[cohort]
        difference = values - reference
        standard_error = difference.std(ddof=1) / np.sqrt(len(difference))
        critical = student_t.ppf(0.975, len(difference) - 1)
        mask = result["experiment"].eq(experiment)
        result.loc[mask, "cv_roc_auc_delta_vs_e3"] = difference.mean()
        result.loc[mask, "cv_roc_auc_delta_vs_e3_ci_low"] = (
            difference.mean() - critical * standard_error
        )
        result.loc[mask, "cv_roc_auc_delta_vs_e3_ci_high"] = (
            difference.mean() + critical * standard_error
        )

    for (experiment, cohort), values in fold_roc_auc.items():
        difference = values - cohort_e8_references[cohort]
        standard_error = difference.std(ddof=1) / np.sqrt(len(difference))
        critical = student_t.ppf(0.975, len(difference) - 1)
        mask = result["experiment"].eq(experiment)
        result.loc[mask, "cv_roc_auc_delta_vs_e8"] = difference.mean()
        result.loc[mask, "cv_roc_auc_delta_vs_e8_ci_low"] = (
            difference.mean() - critical * standard_error
        )
        result.loc[mask, "cv_roc_auc_delta_vs_e8_ci_high"] = (
            difference.mean() + critical * standard_error
        )

    for (experiment, cohort), values in fold_pr_auc.items():
        for reference_name, reference_values in (
            ("e3", cohort_pr_references[cohort]),
            ("e8", cohort_e8_pr_references[cohort]),
        ):
            difference = values - reference_values
            standard_error = difference.std(ddof=1) / np.sqrt(len(difference))
            critical = student_t.ppf(0.975, len(difference) - 1)
            mask = result["experiment"].eq(experiment)
            result.loc[mask, f"cv_pr_auc_delta_vs_{reference_name}"] = difference.mean()
            result.loc[mask, f"cv_pr_auc_delta_vs_{reference_name}_ci_low"] = (
                difference.mean() - critical * standard_error
            )
            result.loc[mask, f"cv_pr_auc_delta_vs_{reference_name}_ci_high"] = (
                difference.mean() + critical * standard_error
            )
    return result


def run_regression_ablations(
    frame: pd.DataFrame, variables: dict, categorical: set[str], config: dict
) -> pd.DataFrame:
    target = "femoral_neck_bmd"
    usable = frame.dropna(subset=[target]).copy()
    indices = np.arange(len(usable))
    train_idx, _ = train_test_split(
        indices, test_size=config["test_size"], random_state=config["seed"]
    )
    training = usable.iloc[train_idx]
    y = training[target]
    weights = training["dietary_day1_weight"].fillna(training["mec_weight"]).fillna(1.0)
    cv = KFold(n_splits=config["cv_folds"], shuffle=True, random_state=config["seed"])
    candidates = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=5,
            n_jobs=config.get("cv_n_jobs", 1),
            random_state=config["seed"],
        ),
    }
    rows = []
    fold_rmse = {}
    for experiment, (features, add_clusters) in feature_experiments(frame, variables).items():
        features = list(dict.fromkeys(features))
        for model_name, estimator in candidates.items():
            pipeline = make_pipeline(
                features, categorical, clone(estimator), add_clusters, config
            )
            scores = cross_validate(
                pipeline,
                training[features],
                y,
                cv=cv,
                scoring={
                    "rmse": _weighted_scorer("neg_root_mean_squared_error"),
                    "r2": _weighted_scorer("r2"),
                },
                n_jobs=config.get("cv_n_jobs", 1),
                params={"sample_weight": weights.to_numpy()},
            )
            rmse = -scores["test_rmse"]
            fold_rmse[(experiment, model_name)] = rmse
            critical = student_t.ppf(0.975, len(rmse) - 1)
            standard_error = rmse.std(ddof=1) / np.sqrt(len(rmse))
            rows.append(
                {
                    "experiment": experiment,
                    "model": model_name,
                    "feature_count": len(features) + int(add_clusters),
                    "cv_rmse_mean": rmse.mean(),
                    "cv_rmse_sd": rmse.std(),
                    "cv_rmse_ci_low": rmse.mean() - critical * standard_error,
                    "cv_rmse_ci_high": rmse.mean() + critical * standard_error,
                    "cv_r2_mean": scores["test_r2"].mean(),
                }
            )
    result = pd.DataFrame(rows)
    for model_name in candidates:
        reference = fold_rmse[("E3_demographic_anthropometric", model_name)]
        for (experiment, candidate_name), values in fold_rmse.items():
            if candidate_name != model_name:
                continue
            difference = values - reference
            critical = student_t.ppf(0.975, len(difference) - 1)
            standard_error = difference.std(ddof=1) / np.sqrt(len(difference))
            mask = result["experiment"].eq(experiment) & result["model"].eq(model_name)
            result.loc[mask, "cv_rmse_delta_vs_e3"] = difference.mean()
            result.loc[mask, "cv_rmse_delta_vs_e3_ci_low"] = (
                difference.mean() - critical * standard_error
            )
            result.loc[mask, "cv_rmse_delta_vs_e3_ci_high"] = (
                difference.mean() + critical * standard_error
            )
    return result.sort_values("cv_rmse_mean")


def _xgb_classifier(config: dict):
    if XGBClassifier is None:
        return None
    return XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_weight=5,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=config.get("cv_n_jobs", 1),
        random_state=config["seed"],
    )


def _xgb_regressor(config: dict):
    if XGBRegressor is None:
        return None
    return XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_weight=5,
        tree_method="hist",
        objective="reg:squarederror",
        n_jobs=config.get("cv_n_jobs", 1),
        random_state=config["seed"],
    )


def _preprocessor(
    features: list[str], categorical: set[str], random_state: int
) -> ColumnTransformer:
    categorical_features = [item for item in features if item in categorical]
    numeric_features = [item for item in features if item not in categorical_features]
    numeric = Pipeline(
        [
            (
                "imputer",
                IterativeImputer(
                    max_iter=15,
                    initial_strategy="median",
                    add_indicator=True,
                    random_state=random_state,
                    skip_complete=True,
                ),
            ),
            ("scale", StandardScaler().set_fit_request(sample_weight=False)),
        ]
    )
    category = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, numeric_features), ("categorical", category, categorical_features)],
        remainder="drop",
    )


def make_pipeline(
    features: list[str], categorical: set[str], estimator, add_clusters: bool, config: dict
) -> Pipeline:
    steps = []
    prepared_features = list(features)
    prepared_categorical = set(categorical)
    if add_clusters:
        absent = set(NUTRITION_CLUSTER_FEATURES) - set(prepared_features)
        if absent:
            raise ValueError(f"Cluster model is missing nutrition features: {sorted(absent)}")
        steps.append(
            (
                "nutrition_clusters",
                NutritionClusterAugmenter(
                    features=NUTRITION_CLUSTER_FEATURES,
                    candidates=tuple(config["nutrition_cluster_candidates"]),
                    random_state=config["seed"],
                ),
            )
        )
        prepared_features.append("nutrition_cluster")
        prepared_categorical.add("nutrition_cluster")
    if hasattr(estimator, "set_fit_request"):
        estimator = estimator.set_fit_request(sample_weight=True)
    steps.extend(
        [
            (
                "preprocess",
                _preprocessor(prepared_features, prepared_categorical, config["seed"]),
            ),
            ("model", estimator),
        ]
    )
    return Pipeline(steps)


def _operating_point_thresholds(
    rows: pd.DataFrame,
    y: pd.Series,
    probability: np.ndarray,
    sample_weight: pd.Series,
    target: float,
    minimum_n: int,
    recall_targets: tuple[float, ...] = (),
    precision_targets: tuple[float, ...] = (),
) -> pd.DataFrame:
    data = rows[["sex", "age", "race_ethnicity"]].copy()
    data["target"] = np.asarray(y)
    data["probability"] = probability
    data["sample_weight"] = np.asarray(sample_weight)
    data["age_group"] = pd.cut(
        data["age"], [49, 64, 74, np.inf], labels=["50-64", "65-74", "75+"]
    )
    data["age_sex_group"] = data["sex"].astype("string") + "|" + data[
        "age_group"
    ].astype("string")
    global_threshold = threshold_for_sensitivity(
        data["target"], probability, target, data["sample_weight"]
    )

    def operating_record(group, threshold, target_sensitivity=np.nan, target_precision=np.nan):
        metrics = classification_metrics(
            data["target"], probability, threshold, data["sample_weight"]
        )
        return {
            "dimension": "global" if group == "all" else "global_tradeoff",
            "group": group,
            "n": len(data),
            "threshold": threshold,
            "target_sensitivity": target_sensitivity,
            "target_precision": target_precision,
            "oof_sensitivity": metrics["sensitivity"],
            "oof_precision": metrics["ppv"],
            "oof_f1": metrics["f1"],
        }

    records = [operating_record("all", global_threshold, target_sensitivity=target)]
    f1_threshold = threshold_for_f1(
        data["target"], probability, data["sample_weight"]
    )
    records.append(operating_record("f1_optimized", f1_threshold))
    for recall_target in sorted(set(recall_targets), reverse=True):
        if np.isclose(recall_target, target):
            continue
        threshold = threshold_for_sensitivity(
            data["target"], probability, recall_target, data["sample_weight"]
        )
        records.append(
            operating_record(
                f"recall_{recall_target:.2f}",
                threshold,
                target_sensitivity=recall_target,
            )
        )
    for precision_target in sorted(set(precision_targets)):
        threshold = threshold_for_precision(
            data["target"], probability, precision_target, data["sample_weight"]
        )
        records.append(
            operating_record(
                f"precision_{precision_target:.2f}",
                threshold,
                target_precision=precision_target,
            )
        )
    for dimension in ("sex", "age_group", "age_sex_group", "race_ethnicity"):
        for group, subset in data.groupby(dimension, observed=True, dropna=False):
            if len(subset) < minimum_n or subset["target"].nunique() < 2:
                continue
            records.append(
                {
                    "dimension": dimension,
                    "group": str(group),
                    "n": len(subset),
                    "threshold": threshold_for_sensitivity(
                        subset["target"],
                        subset["probability"],
                        target,
                        subset["sample_weight"],
                    ),
                    "target_sensitivity": target,
                    "target_precision": np.nan,
                    "oof_sensitivity": np.nan,
                    "oof_precision": np.nan,
                    "oof_f1": np.nan,
                }
            )
    return pd.DataFrame(records)


def run_classification(
    frame: pd.DataFrame,
    features: list[str],
    categorical: set[str],
    config: dict,
    add_clusters: bool = True,
    target: str = "osteoporosis",
    sample_weight_column: str = "mec_weight",
    evaluation_cycle: str | None = None,
    selection_metric: str = "pr_auc",
) -> ClassificationResult:
    usable = frame.dropna(subset=[target]).copy()
    X = usable[features]
    y = usable[target].astype(int)
    weights = usable[sample_weight_column].fillna(usable["mec_weight"]).fillna(1.0)
    indices = np.arange(len(usable))
    if evaluation_cycle is None:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=config["test_size"],
            random_state=config["seed"],
            stratify=y,
        )
    else:
        evaluation_mask = usable["survey_cycle"].eq(evaluation_cycle).to_numpy()
        train_idx, test_idx = indices[~evaluation_mask], indices[evaluation_mask]
        if not len(train_idx) or not len(test_idx):
            raise ValueError(f"Evaluation cycle {evaluation_cycle!r} does not create a split")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    w_train = weights.iloc[train_idx]
    w_test = weights.iloc[test_idx]
    folds = min(config["cv_folds"], int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=max(2, folds), shuffle=True, random_state=config["seed"])

    candidates = {
        "regularized_logistic": LogisticRegression(
            C=1.0, max_iter=3000, random_state=config["seed"]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=5,
            n_jobs=config.get("cv_n_jobs", 1),
            random_state=config["seed"],
        ),
    }
    xgb = _xgb_classifier(config)
    if xgb is not None:
        candidates["xgboost"] = xgb
    rows = []
    pipelines = {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(features, categorical, estimator, add_clusters, config)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "roc_auc": _weighted_scorer("roc_auc"),
                "pr_auc": _weighted_scorer("average_precision"),
            },
            n_jobs=config.get("cv_n_jobs", 1),
            params={"sample_weight": w_train.to_numpy()},
        )
        pipelines[name] = pipeline
        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
                "cv_roc_auc_sd": scores["test_roc_auc"].std(),
                "cv_pr_auc_mean": scores["test_pr_auc"].mean(),
            }
        )
    if selection_metric not in {"roc_auc", "pr_auc"}:
        raise ValueError("selection_metric must be 'roc_auc' or 'pr_auc'")
    selection_column = f"cv_{selection_metric}_mean"
    comparison = pd.DataFrame(rows).sort_values(selection_column, ascending=False)
    comparison["selection_metric"] = selection_metric
    best_name = comparison.iloc[0]["model"]
    selected = pipelines[best_name]
    calibration_cv = min(3, int(y_train.value_counts().min()))
    calibrated_template = CalibratedClassifierCV(
        estimator=selected, method="sigmoid", cv=max(2, calibration_cv)
    ).set_fit_request(sample_weight=True)
    calibrated_out_of_fold = cross_val_predict(
        clone(calibrated_template),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=config.get("cv_n_jobs", 1),
        params={"sample_weight": w_train.to_numpy()},
    )[:, 1]
    thresholds = _operating_point_thresholds(
        usable.iloc[train_idx],
        y_train,
        calibrated_out_of_fold,
        w_train,
        config["screening_sensitivity_target"],
        config.get("subgroup_threshold_min_n", 50),
        tuple(config.get("screening_recall_targets", ())),
        tuple(config.get("screening_precision_targets", ())),
    )
    threshold = float(
        thresholds.loc[thresholds["dimension"].eq("global"), "threshold"].iloc[0]
    )

    calibrated = clone(calibrated_template)
    calibrated.fit(X_train, y_train, sample_weight=w_train.to_numpy())
    probability = calibrated.predict_proba(X_test)[:, 1]

    operating_points = {"default": 0.5}
    for row in thresholds.loc[
        thresholds["dimension"].isin(["global", "global_tradeoff"])
    ].itertuples(index=False):
        label = "sensitivity_oriented" if row.dimension == "global" else str(row.group)
        operating_points[label] = float(row.threshold)

    metric_rows = []
    for label, selected_threshold in operating_points.items():
        metrics = classification_metrics(
            y_test, probability, selected_threshold, sample_weight=w_test
        )
        metric_rows.append(
            {
                "evaluation": label,
                "model": best_name,
                "target": target,
                "evaluation_scope": (
                    f"held_out_cycle_{evaluation_cycle}"
                    if evaluation_cycle is not None
                    else "random_held_out_test"
                ),
                "training_n": len(train_idx),
                "evaluation_n": len(test_idx),
                "sample_weight": sample_weight_column,
                **metrics,
            }
        )

    prediction_columns = [
        "participant_id",
        target,
        "dietary_day1_weight",
        "mec_weight",
        "survey_stratum",
        "survey_psu",
        "sex",
        "age",
        "race_ethnicity",
        "survey_cycle",
    ]
    predictions = usable.iloc[test_idx][prediction_columns].copy()
    predictions["outcome_name"] = target
    predictions["analysis_weight"] = w_test.to_numpy()
    predictions["probability"] = probability
    predictions["screening_prediction"] = (probability >= threshold).astype(int)
    if "f1_optimized" in operating_points:
        predictions["f1_optimized_prediction"] = (
            probability >= operating_points["f1_optimized"]
        ).astype(int)
    return ClassificationResult(
        calibrated, pd.DataFrame(metric_rows), predictions, comparison, thresholds
    )


def run_regression(
    frame: pd.DataFrame,
    features: list[str],
    categorical: set[str],
    config: dict,
    target: str = "femoral_neck_bmd",
    add_clusters: bool = True,
) -> RegressionResult:
    usable = frame.dropna(subset=[target]).copy()
    X = usable[features]
    y = usable[target]
    weights = usable["dietary_day1_weight"].fillna(usable["mec_weight"]).fillna(1.0)
    indices = np.arange(len(usable))
    train_idx, test_idx = train_test_split(
        indices, test_size=config["test_size"], random_state=config["seed"]
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    w_train = weights.iloc[train_idx]
    cv = KFold(n_splits=config["cv_folds"], shuffle=True, random_state=config["seed"])
    candidates = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=5,
            n_jobs=config.get("cv_n_jobs", 1),
            random_state=config["seed"],
        ),
    }
    xgb = _xgb_regressor(config)
    if xgb is not None:
        candidates["xgboost"] = xgb
    rows = []
    pipelines = {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(features, categorical, estimator, add_clusters, config)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "rmse": _weighted_scorer("neg_root_mean_squared_error"),
                "mae": _weighted_scorer("neg_mean_absolute_error"),
            },
            n_jobs=config.get("cv_n_jobs", 1),
            params={"sample_weight": w_train.to_numpy()},
        )
        pipelines[name] = pipeline
        rows.append(
            {
                "model": name,
                "cv_rmse_mean": -scores["test_rmse"].mean(),
                "cv_rmse_sd": scores["test_rmse"].std(),
                "cv_mae_mean": -scores["test_mae"].mean(),
            }
        )
    comparison = pd.DataFrame(rows).sort_values("cv_rmse_mean")
    best_name = comparison.iloc[0]["model"]
    selected = pipelines[best_name].fit(
        X_train, y_train, sample_weight=w_train.to_numpy()
    )
    predicted = selected.predict(X_test)
    metrics = {"model": best_name, **regression_metrics(y_test, predicted, weights.iloc[test_idx])}
    predictions = usable.iloc[test_idx][
        [
            "participant_id",
            target,
            "dietary_day1_weight",
            "mec_weight",
            "survey_stratum",
            "survey_psu",
            "survey_cycle",
        ]
    ].copy()
    predictions["predicted_bmd"] = predicted
    return RegressionResult(selected, metrics, predictions, comparison)
