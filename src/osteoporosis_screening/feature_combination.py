from __future__ import annotations

import json
import hashlib
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

from .config import ProjectPaths, load_yaml
from .curve_bending import (
    _experiment_feature_sets,
    _group_splits,
    _platt_apply,
    _platt_fit,
    _policy_metrics,
    select_recall_threshold,
)
from .modeling import XGBClassifier, _preprocessor


class RoutedCatBoostClassifier(ClassifierMixin, BaseEstimator):
    """Small sklearn-compatible adapter that explicitly accepts survey weights."""

    def __init__(
        self,
        n_estimators: int = 150,
        depth: int = 4,
        learning_rate: float = 0.04,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        if CatBoostClassifier is None:
            raise ImportError("catboost is required for RoutedCatBoostClassifier")
        self.model_ = CatBoostClassifier(
            iterations=self.n_estimators,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=self.random_state,
            thread_count=self.n_jobs,
            verbose=False,
            allow_writing_files=False,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class ProbabilityStackingClassifier(ClassifierMixin, BaseEstimator):
    """Stack base-model probabilities without exposing validation rows to the meta-learner."""

    def __init__(
        self,
        base_estimators: tuple[tuple[str, object], ...],
        cv: int = 3,
        random_state: int = 42,
    ) -> None:
        self.base_estimators = base_estimators
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        y_array = np.asarray(y, dtype=int)
        weight = (
            np.ones(len(y_array), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        folds = max(2, min(int(self.cv), int(pd.Series(y_array).value_counts().min())))
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=self.random_state
        )
        meta_features = np.full((len(y_array), len(self.base_estimators)), np.nan)
        for train_idx, validation_idx in splitter.split(np.zeros(len(y_array)), y_array):
            for column, (_, estimator) in enumerate(self.base_estimators):
                fitted = _fit_raw_estimator(
                    estimator,
                    X[train_idx],
                    y_array[train_idx],
                    weight[train_idx],
                )
                meta_features[validation_idx, column] = fitted.predict_proba(
                    X[validation_idx]
                )[:, 1]
        if not np.isfinite(meta_features).all():
            raise ValueError("Stacking out-of-fold predictions are incomplete")
        self.meta_model_ = LogisticRegression(
            C=1.0, max_iter=1000, random_state=self.random_state
        )
        self.meta_model_.fit(meta_features, y_array, sample_weight=weight)
        self.base_models_ = [
            (name, _fit_raw_estimator(estimator, X, y_array, weight))
            for name, estimator in self.base_estimators
        ]
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, X):
        meta_features = np.column_stack(
            [model.predict_proba(X)[:, 1] for _, model in self.base_models_]
        )
        return self.meta_model_.predict_proba(meta_features)


class RoutedResampledClassifier(ClassifierMixin, BaseEstimator):
    """Optional imbalanced-learn arm for SMOTE, ADASYN, and random oversampling."""

    def __init__(
        self,
        estimator,
        method: str = "smote",
        random_state: int = 42,
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        try:
            from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "imbalanced-learn is required for resampling candidates"
            ) from exc

        samplers = {
            "adasyn": ADASYN(random_state=self.random_state),
            "random_over": RandomOverSampler(random_state=self.random_state),
            "smote": SMOTE(random_state=self.random_state),
        }
        if self.method not in samplers:
            raise ValueError("method must be one of adasyn, random_over, smote")
        sampler = samplers[self.method]
        y_array = np.asarray(y, dtype=int)
        X_resampled, y_resampled = sampler.fit_resample(X, y_array)
        if sample_weight is None:
            weight_resampled = None
        else:
            weight = np.asarray(sample_weight, dtype=float)
            if hasattr(sampler, "sample_indices_"):
                weight_resampled = weight[sampler.sample_indices_]
            else:
                class_weight = {
                    value: float(weight[y_array == value].mean())
                    for value in np.unique(y_array)
                }
                weight_resampled = np.asarray(
                    [class_weight[value] for value in y_resampled], dtype=float
                )
        self.model_ = _fit_raw_estimator(
            self.estimator, X_resampled, y_resampled, weight_resampled
        )
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def _candidate_estimators(config: dict, seed: int, n_jobs: int) -> list[tuple[str, object]]:
    candidates: list[tuple[str, object]] = []
    for value in config.get("logistic_c", [1.0]):
        candidates.append(
            (
                f"logistic_c_{value:g}",
                LogisticRegression(C=float(value), max_iter=3000, random_state=seed),
            )
        )
    for value in config.get("lasso_selection_c", [0.05]):
        selection_model = LogisticRegression(
            C=float(value),
            l1_ratio=1.0,
            solver="liblinear",
            max_iter=3000,
            random_state=seed,
        )
        if hasattr(selection_model, "set_fit_request"):
            selection_model = selection_model.set_fit_request(sample_weight=True)
        selector = SelectFromModel(selection_model, threshold="median")
        if hasattr(selector, "set_fit_request"):
            selector = selector.set_fit_request(sample_weight=True)
        classifier = LogisticRegression(C=1.0, max_iter=3000, random_state=seed)
        if hasattr(classifier, "set_fit_request"):
            classifier = classifier.set_fit_request(sample_weight=True)
        candidates.append(
            (
                f"lasso_selected_logistic_c_{value:g}",
                Pipeline([("select", selector), ("logistic", classifier)]),
            )
        )

    tree_values = [int(value) for value in _as_list(config.get("tree_n_estimators", 150))]
    rf_base = None
    for trees in tree_values:
        for leaf in config.get("random_forest_min_samples_leaf", [10]):
            for max_features in config.get("random_forest_max_features", ["sqrt"]):
                estimator = RandomForestClassifier(
                    n_estimators=trees,
                    min_samples_leaf=int(leaf),
                    max_features=max_features,
                    n_jobs=n_jobs,
                    random_state=seed,
                )
                candidates.append(
                    (
                        f"random_forest_{trees}_leaf_{leaf}_mf_{max_features}",
                        estimator,
                    )
                )
                if rf_base is None:
                    rf_base = estimator

    xgb_base = None
    if XGBClassifier is not None:
        for trees in tree_values:
            for depth in config.get("xgboost_max_depth", [3]):
                for rate in config.get("xgboost_learning_rate", [0.04]):
                    for child_weight in config.get("xgboost_min_child_weight", [5]):
                        estimator = XGBClassifier(
                            n_estimators=trees,
                            max_depth=int(depth),
                            learning_rate=float(rate),
                            subsample=0.8,
                            colsample_bytree=0.8,
                            reg_lambda=2.0,
                            min_child_weight=int(child_weight),
                            tree_method="hist",
                            eval_metric="logloss",
                            n_jobs=n_jobs,
                            random_state=seed,
                        )
                        candidates.append(
                            (
                                f"xgboost_{trees}_depth_{depth}_lr_{rate:g}"
                                f"_child_{child_weight}",
                                estimator,
                            )
                        )
                        if xgb_base is None:
                            xgb_base = estimator

    lgbm_base = None
    if LGBMClassifier is not None:
        for trees in tree_values:
            for leaves in config.get("lightgbm_num_leaves", [15]):
                for rate in config.get("lightgbm_learning_rate", [0.04]):
                    estimator = LGBMClassifier(
                            n_estimators=trees,
                            num_leaves=int(leaves),
                            learning_rate=float(rate),
                            subsample=0.8,
                            colsample_bytree=0.8,
                            reg_lambda=2.0,
                            min_child_samples=20,
                            n_jobs=n_jobs,
                            random_state=seed,
                            verbose=-1,
                    )
                    candidates.append(
                        (
                            f"lightgbm_{trees}_leaves_{leaves}_lr_{rate:g}",
                            estimator,
                        )
                    )
                    if lgbm_base is None:
                        lgbm_base = estimator

    cat_base = None
    if CatBoostClassifier is not None:
        for trees in tree_values:
            for depth in config.get("catboost_depth", [4]):
                for rate in config.get("catboost_learning_rate", [0.04]):
                    estimator = RoutedCatBoostClassifier(
                            n_estimators=trees,
                            depth=int(depth),
                            learning_rate=float(rate),
                            random_state=seed,
                            n_jobs=n_jobs,
                    )
                    candidates.append(
                        (
                            f"catboost_{trees}_depth_{depth}_lr_{rate:g}",
                            estimator,
                        )
                    )
                    if cat_base is None:
                        cat_base = estimator

    for hidden in config.get("mlp_hidden_layer_sizes", [[32]]):
        hidden_tuple = tuple(int(value) for value in hidden)
        for alpha in config.get("mlp_alpha", [0.001]):
            candidates.append(
                (
                    f"mlp_{'_'.join(map(str, hidden_tuple))}_alpha_{alpha:g}",
                    MLPClassifier(
                        hidden_layer_sizes=hidden_tuple,
                        alpha=float(alpha),
                        learning_rate_init=float(
                            config.get("mlp_learning_rate_init", 0.001)
                        ),
                        early_stopping=True,
                        max_iter=int(config.get("mlp_max_iter", 250)),
                        random_state=seed,
                    ),
                )
            )

    if (
        config.get("include_resampling_candidates", False)
        and importlib.util.find_spec("imblearn") is not None
    ):
        resampling_base = xgb_base if xgb_base is not None else rf_base
        if resampling_base is not None:
            for method in config.get("resampling_methods", ["smote", "adasyn"]):
                candidates.append(
                    (
                        f"{method}_{resampling_base.__class__.__name__.lower()}",
                        RoutedResampledClassifier(
                            clone(resampling_base), method=str(method), random_state=seed
                        ),
                    )
                )

    if config.get("include_stacking", True):
        base_estimators = tuple(
            (name, clone(estimator))
            for name, estimator in (
                ("rf", rf_base),
                ("xgb", xgb_base),
                ("lgbm", lgbm_base),
                ("cat", cat_base),
            )
            if estimator is not None
        )
        if len(base_estimators) >= 2:
            candidates.append(
                (
                    "stacked_tree_ensemble",
                    ProbabilityStackingClassifier(
                        base_estimators=base_estimators,
                        cv=int(config.get("stacking_cv_folds", 3)),
                        random_state=seed,
                    ),
                )
            )
    for value in config.get("spline_logistic_c", [1.0]):
        spline = SplineTransformer(
            n_knots=4, degree=2, include_bias=False
        ).set_fit_request(sample_weight=False)
        logistic = LogisticRegression(
            C=float(value), max_iter=3000, random_state=seed
        ).set_fit_request(sample_weight=True)
        candidates.append(
            (
                f"spline_logistic_c_{value:g}",
                Pipeline([("spline", spline), ("logistic", logistic)]),
            )
        )
    return candidates


def _panel_availability(
    frame: pd.DataFrame, panels: dict[str, list[str]], minimum_fraction: float
) -> pd.DataFrame:
    rows = []
    cycles = sorted(frame["survey_cycle"].astype(str).unique())
    for panel, features in panels.items():
        eligible = True
        for cycle in cycles:
            cycle_frame = frame.loc[frame["survey_cycle"].astype(str).eq(cycle)]
            fractions = {
                feature: (
                    float(cycle_frame[feature].notna().mean())
                    if feature in cycle_frame
                    else 0.0
                )
                for feature in features
            }
            minimum = min(fractions.values()) if fractions else 1.0
            eligible &= minimum >= minimum_fraction
            rows.append(
                {
                    "panel": panel,
                    "survey_cycle": cycle,
                    "n": len(cycle_frame),
                    "minimum_observed_fraction": minimum,
                    **{f"observed_fraction__{key}": value for key, value in fractions.items()},
                }
            )
        for row in rows:
            if row["panel"] == panel:
                row["eligible_all_production_cycles"] = bool(eligible)
    return pd.DataFrame(rows)


def _panel_spec_features(spec, variables: dict) -> list[str]:
    if isinstance(spec, dict):
        features = list(spec.get("features", []))
        for group in spec.get("feature_groups", []):
            features.extend(variables["feature_groups"][group])
        return list(dict.fromkeys(features))
    return list(spec)


def _base_features(name: str, e12: list[str], variables: dict) -> list[str]:
    if name == "e12":
        return list(e12)
    if name == "none":
        return []
    if name == "demographic_anthropometric":
        return list(
            dict.fromkeys(
                variables["feature_groups"]["demographic"]
                + variables["feature_groups"]["anthropometric"]
            )
        )
    if name in variables["feature_groups"]:
        return list(variables["feature_groups"][name])
    raise ValueError(f"Unknown feature panel base: {name}")


def _normalize_feature_panels(
    experiment: dict, variables: dict, e12: list[str]
) -> dict[str, list[str]]:
    default_base = str(experiment.get("panel_base", "e12"))
    panels = {}
    for panel, spec in experiment["feature_panels"].items():
        base = str(spec.get("base", default_base)) if isinstance(spec, dict) else default_base
        panels[panel] = list(
            dict.fromkeys(
                _base_features(base, e12, variables) + _panel_spec_features(spec, variables)
            )
        )
    return panels


def _prepare_target_frame(
    frame: pd.DataFrame, target_name: str, experiment: dict
) -> tuple[pd.DataFrame, str, str]:
    if target_name == "osteoporosis":
        return frame.dropna(subset=["osteoporosis"]).copy(), "osteoporosis", (
            "femoral-neck T-score <= -2.5"
        )
    if target_name != "low_bmd":
        raise ValueError("feature-combination target must be 'osteoporosis' or 'low_bmd'")
    cutoff = float(experiment.get("low_bmd_t_score_cutoff", -1.0))
    result = frame.dropna(subset=["femoral_neck_t_score"]).copy()
    result["low_bmd"] = result["femoral_neck_t_score"].lt(cutoff).astype("Int64")
    return result.dropna(subset=["low_bmd"]).copy(), "low_bmd", (
        f"femoral-neck T-score < {cutoff:g}"
    )


def _prefixed_metrics(prefix: str, metrics: dict, skip: set[str] | None = None) -> dict:
    skip = skip or set()
    return {f"{prefix}{key}": value for key, value in metrics.items() if key not in skip}


def _fit_raw_estimator(estimator, X, y, sample_weight):
    """Fit a cloned estimator on already transformed data with survey weights."""
    fitted = clone(estimator)
    if hasattr(fitted, "set_fit_request"):
        fitted = fitted.set_fit_request(sample_weight=True)
    fitted.fit(X, y, sample_weight=sample_weight)
    return fitted


def _shared_panel_predictions(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    candidates: list[tuple[str, object]],
    categorical: set[str],
    model_config: dict,
    outcome_column: str = "osteoporosis",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Generate inner OOF and outer predictions with shared fold preprocessing.

    Imputation and encoding depend on the feature panel and training fold, not on
    the downstream estimator. Reusing those transforms preserves fold isolation
    while avoiding the dominant repeated IterativeImputer work.
    """
    outcome = training[outcome_column].astype(int).to_numpy()
    weight = training["mec_weight"].fillna(1.0).to_numpy()
    raw_oof = {name: np.full(len(training), np.nan) for name, _ in candidates}

    for inner_training, inner_validation in _group_splits(training):
        preprocess = _preprocessor(features, categorical, int(model_config["seed"]))
        X_training = preprocess.fit_transform(
            training.iloc[inner_training][features]
        )
        X_validation = preprocess.transform(
            training.iloc[inner_validation][features]
        )
        for model_name, estimator in candidates:
            fitted = _fit_raw_estimator(
                estimator,
                X_training,
                outcome[inner_training],
                weight[inner_training],
            )
            raw_oof[model_name][inner_validation] = fitted.predict_proba(
                X_validation
            )[:, 1]

    preprocess = _preprocessor(features, categorical, int(model_config["seed"]))
    X_training = preprocess.fit_transform(training[features])
    X_evaluation = preprocess.transform(evaluation[features])
    result = {}
    for model_name, estimator in candidates:
        if not np.isfinite(raw_oof[model_name]).all():
            raise ValueError(
                f"Grouped out-of-fold predictions are incomplete for {model_name}"
            )
        fitted = _fit_raw_estimator(estimator, X_training, outcome, weight)
        result[model_name] = (
            raw_oof[model_name],
            fitted.predict_proba(X_evaluation)[:, 1],
        )
    return result


def _yes_indicator(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.lower()
    return text.isin(["yes", "1", "true"]).astype(float)


def _clinical_score_values(frame: pd.DataFrame, score_name: str) -> pd.Series:
    if score_name == "osta":
        score = -((frame["weight_kg"] - frame["age"]) * 0.2)
    elif score_name == "frax_grade_risk_count":
        risk = pd.Series(0.0, index=frame.index)
        risk += ((frame["age"] - 50).clip(lower=0) / 10).fillna(0)
        risk += ((25 - frame["bmi"]).clip(lower=0) / 5).fillna(0)
        for column in (
            "prior_fracture_history",
            "parental_hip_fracture",
            "long_term_glucocorticoid",
            "current_smoker",
        ):
            if column in frame:
                risk += _yes_indicator(frame[column]).fillna(0)
        if "alcohol_drinks_week" in frame:
            risk += frame["alcohol_drinks_week"].ge(21).fillna(False).astype(float)
        score = risk
    else:
        raise ValueError(f"Unknown clinical score: {score_name}")
    return score.astype(float)


def _fit_score_calibrator(score: pd.Series, outcome, weight) -> LogisticRegression:
    score = score.astype(float)
    fill_value = float(score.median()) if score.notna().any() else 0.0
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(
        score.fillna(fill_value).to_numpy().reshape(-1, 1),
        np.asarray(outcome, dtype=int),
        sample_weight=np.asarray(weight, dtype=float),
    )
    model.fill_value_ = fill_value
    return model


def _apply_score_calibrator(model: LogisticRegression, score: pd.Series) -> np.ndarray:
    return model.predict_proba(
        score.astype(float).fillna(model.fill_value_).to_numpy().reshape(-1, 1)
    )[:, 1]


def _clinical_score_baselines(
    frame: pd.DataFrame,
    target: float,
    confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"age", "weight_kg", "bmi", "osteoporosis", "mec_weight", "survey_cycle"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(), pd.DataFrame()
    metric_rows = []
    prediction_rows = []
    for outer_cycle in sorted(frame["survey_cycle"].astype(str).unique()):
        training_all = frame.loc[frame["survey_cycle"].astype(str).ne(outer_cycle)].copy()
        evaluation_all = frame.loc[frame["survey_cycle"].astype(str).eq(outer_cycle)].copy()
        scopes = [("all", training_all, evaluation_all)]
        female_training = training_all.loc[training_all["sex"].eq(2)].copy()
        female_evaluation = evaluation_all.loc[evaluation_all["sex"].eq(2)].copy()
        if (
            len(female_training)
            and len(female_evaluation)
            and female_training["osteoporosis"].nunique() == 2
            and female_evaluation["osteoporosis"].nunique() == 2
        ):
            scopes.append(("female_only", female_training, female_evaluation))
        for scope, training, evaluation in scopes:
            outcome_train = training["osteoporosis"].astype(int).to_numpy()
            outcome_evaluation = evaluation["osteoporosis"].astype(int).to_numpy()
            weight_train = training["mec_weight"].fillna(1.0).to_numpy()
            weight_evaluation = evaluation["mec_weight"].fillna(1.0).to_numpy()
            try:
                inner_splits = _group_splits(training)
            except ValueError:
                continue
            for score_name in ("osta", "frax_grade_risk_count"):
                oof_probability = np.full(len(training), np.nan)
                for inner_training, inner_validation in inner_splits:
                    train_score = _clinical_score_values(
                        training.iloc[inner_training], score_name
                    )
                    validation_score = _clinical_score_values(
                        training.iloc[inner_validation], score_name
                    )
                    calibrator = _fit_score_calibrator(
                        train_score,
                        outcome_train[inner_training],
                        weight_train[inner_training],
                    )
                    oof_probability[inner_validation] = _apply_score_calibrator(
                        calibrator, validation_score
                    )
                if not np.isfinite(oof_probability).all():
                    raise ValueError(f"Incomplete OOF predictions for {score_name}")
                final_calibrator = _fit_score_calibrator(
                    _clinical_score_values(training, score_name),
                    outcome_train,
                    weight_train,
                )
                probability = _apply_score_calibrator(
                    final_calibrator, _clinical_score_values(evaluation, score_name)
                )
                point = select_recall_threshold(
                    outcome_train,
                    oof_probability,
                    weight_train,
                    target,
                    confidence,
                    require_lcb=False,
                )
                conservative = select_recall_threshold(
                    outcome_train,
                    oof_probability,
                    weight_train,
                    target,
                    confidence,
                    require_lcb=True,
                )
                for policy, selected in (
                    ("point_recall", point),
                    ("lcb_recall", conservative),
                ):
                    predicted = probability >= selected["threshold"]
                    metric_rows.append(
                        {
                            "outer_cycle": outer_cycle,
                            "score": score_name,
                            "scope": scope,
                            "policy": policy,
                            "threshold": selected["threshold"],
                            "evaluation_n": len(evaluation),
                            **_policy_metrics(
                                outcome_evaluation,
                                probability,
                                predicted,
                                weight_evaluation,
                            ),
                            "log_loss": log_loss(
                                outcome_evaluation,
                                probability,
                                sample_weight=weight_evaluation,
                            ),
                            "brier_score": brier_score_loss(
                                outcome_evaluation,
                                probability,
                                sample_weight=weight_evaluation,
                            ),
                        }
                    )
                predictions = evaluation[
                    ["participant_id", "survey_cycle", "sex", "osteoporosis", "mec_weight"]
                ].copy()
                predictions["score"] = score_name
                predictions["scope"] = scope
                predictions["probability"] = probability
                predictions["point_threshold"] = point["threshold"]
                predictions["point_prediction"] = (
                    probability >= point["threshold"]
                ).astype("int8")
                predictions["lcb_threshold"] = conservative["threshold"]
                predictions["lcb_prediction"] = (
                    probability >= conservative["threshold"]
                ).astype("int8")
                prediction_rows.append(predictions)
    predictions = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame()
    )
    return pd.DataFrame(metric_rows), predictions


def _paired_panel_deltas(selected: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = selected.pivot(index="outer_cycle", columns="feature_set", values=metric)
    baseline = pivot["E12_baseline"]
    rows = []
    for feature_set in pivot:
        difference = (pivot[feature_set] - baseline).dropna()
        mean = float(difference.mean())
        if len(difference) > 1:
            critical = float(student_t.ppf(0.975, len(difference) - 1))
            margin = critical * float(difference.std(ddof=1)) / np.sqrt(len(difference))
        else:
            margin = np.nan
        rows.append(
            {
                "feature_set": feature_set,
                "metric": metric,
                "outer_cycles": len(difference),
                "mean_delta_vs_e12": mean,
                "ci_low": mean - margin,
                "ci_high": mean + margin,
            }
        )
    return pd.DataFrame(rows)


def _knhanes_local_recalibration(
    root: Path, target: float, confidence: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / "reports" / "tables" / "external_knhanes_2008_2011_predictions.csv"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    frame = pd.read_csv(path).dropna(
        subset=["osteoporosis", "probability", "analysis_weight", "survey_year"]
    )
    metric_rows = []
    prediction_rows = []
    frozen_threshold = 0.027135471211340634
    for outer_year in sorted(frame["survey_year"].unique()):
        training = frame.loc[frame["survey_year"].ne(outer_year)].copy()
        evaluation = frame.loc[frame["survey_year"].eq(outer_year)].copy()
        calibrated_oof = np.full(len(training), np.nan)
        groups = training["survey_year"].to_numpy()
        for inner_year in sorted(training["survey_year"].unique()):
            inner_fit = groups != inner_year
            inner_validation = groups == inner_year
            calibrator = _platt_fit(
                training.loc[inner_fit, "osteoporosis"].astype(int),
                training.loc[inner_fit, "probability"],
                training.loc[inner_fit, "analysis_weight"],
            )
            calibrated_oof[inner_validation] = _platt_apply(
                calibrator, training.loc[inner_validation, "probability"]
            )
        selected_point = select_recall_threshold(
            training["osteoporosis"].astype(int),
            calibrated_oof,
            training["analysis_weight"],
            target,
            require_lcb=False,
        )
        selected_lcb = select_recall_threshold(
            training["osteoporosis"].astype(int),
            calibrated_oof,
            training["analysis_weight"],
            target,
            confidence=confidence,
            require_lcb=True,
        )
        final_calibrator = _platt_fit(
            training["osteoporosis"].astype(int),
            training["probability"],
            training["analysis_weight"],
        )
        recalibrated = _platt_apply(final_calibrator, evaluation["probability"])
        strategies = {
            "frozen_nhanes_threshold": (
                evaluation["probability"].to_numpy(),
                frozen_threshold,
            ),
            "knhanes_year_nested_recalibration": (
                recalibrated,
                float(selected_point["threshold"]),
            ),
            "knhanes_year_nested_recalibration_lcb": (
                recalibrated,
                float(selected_lcb["threshold"]),
            ),
        }
        for strategy, (probability, threshold) in strategies.items():
            outcome = evaluation["osteoporosis"].astype(int).to_numpy()
            weight = evaluation["analysis_weight"].to_numpy()
            predicted = probability >= threshold
            metric_rows.append(
                {
                    "outer_year": int(outer_year),
                    "strategy": strategy,
                    "threshold": threshold,
                    "evaluation_n": len(evaluation),
                    **_policy_metrics(outcome, probability, predicted, weight),
                    "log_loss": log_loss(outcome, probability, sample_weight=weight),
                    "brier_score": brier_score_loss(
                        outcome, probability, sample_weight=weight
                    ),
                }
            )
            predictions = evaluation[
                ["participant_id", "survey_year", "osteoporosis", "analysis_weight"]
            ].copy()
            predictions["strategy"] = strategy
            predictions["probability"] = probability
            predictions["threshold"] = threshold
            predictions["prediction"] = predicted.astype("int8")
            prediction_rows.append(predictions)
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def _write_report(
    path: Path,
    panel_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    multi_metric_candidates: pd.DataFrame,
    deltas: pd.DataFrame,
    availability: pd.DataFrame,
    clinical_score_summary: pd.DataFrame,
    recalibration_summary: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        def render(value) -> str:
            if pd.isna(value):
                return ""
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.4f}"
            return str(value).replace("|", "\\|")

        header = "| " + " | ".join(map(str, frame.columns)) + " |"
        divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
        rows = [
            "| " + " | ".join(render(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
        return "\n".join([header, divider, *rows])

    best = panel_summary.sort_values(["mean_pr_auc", "mean_roc_auc"], ascending=False).iloc[0]
    best_candidate = candidate_summary.sort_values(
        ["mean_pr_auc", "mean_roc_auc"], ascending=False
    ).iloc[0]
    medication_available = bool(
        availability.loc[
            availability["panel"].eq("E13f_medication_risk"),
            "eligible_all_production_cycles",
        ].all()
    )
    delta = deltas.loc[
        deltas["feature_set"].eq(best["feature_set"])
        & deltas["metric"].eq("pr_auc")
    ].iloc[0]
    lines = [
        "# Feature and model combination results",
        "",
        "This additive experiment did not replace the production model or its outputs.",
        "Feature-set, algorithm, hyperparameter, calibration, and threshold selection were",
        "contained inside each outer leave-one-production-cycle-out split.",
        "",
        "## Leading feature panel",
        "",
        f"The leading panel was `{best['feature_set']}` with mean outer PR-AUC "
        f"{best['mean_pr_auc']:.4f} and ROC-AUC {best['mean_roc_auc']:.4f}.",
        f"Its paired PR-AUC change versus E12 was {delta['mean_delta_vs_e12']:+.4f} "
        f"(95% CI {delta['ci_low']:+.4f} to {delta['ci_high']:+.4f}).",
        (
            "This interval includes zero, so the apparent panel gain is exploratory and "
            "does not justify replacing E12."
            if delta["ci_low"] <= 0 <= delta["ci_high"]
            else "This paired interval excludes zero across the four held-out cycles."
        ),
        "",
        "## Best fixed feature-model combination",
        "",
        f"The highest mean outer score among prespecified fixed candidates was "
        f"`{best_candidate['candidate']}` (PR-AUC {best_candidate['mean_pr_auc']:.4f}; "
        f"ROC-AUC {best_candidate['mean_roc_auc']:.4f}). This is an exploratory ranking "
        "of outer results, not an unbiased estimate after selecting the winner.",
        "",
        "## Panel comparison",
        "",
        markdown_table(panel_summary),
        "",
        "## Multi-metric candidates",
        "",
        markdown_table(
            multi_metric_candidates.loc[
                multi_metric_candidates["dominates_selected_e12_point_policy"]
            ][
                [
                    "candidate",
                    "mean_pr_auc",
                    "mean_roc_auc",
                    "mean_point_sensitivity",
                    "mean_point_specificity",
                    "mean_point_ppv",
                ]
            ]
        )
        if (
            not multi_metric_candidates.empty
            and "dominates_selected_e12_point_policy" in multi_metric_candidates
            and multi_metric_candidates["dominates_selected_e12_point_policy"].any()
        )
        else "No fixed candidate improved PR-AUC, ROC-AUC, sensitivity, specificity, and PPV against the selected E12 panel at the point-recall policy.",
        "",
        "## Paired panel changes versus E12",
        "",
        markdown_table(deltas),
        "",
        "## Clinical-score baselines",
        "",
        (
            markdown_table(clinical_score_summary)
            if not clinical_score_summary.empty
            else "Clinical-score baselines were not run because required columns were unavailable."
        ),
        "",
        "## Medication limitation",
        "",
        (
            "Medication panels were evaluated across all production cycles."
            if medication_available
            else "Medication panels were not scored because prescription files were not "
            "available for every production cycle; partial-cycle missingness was not imputed "
            "into a false model comparison. The aggregation code excludes osteoporosis "
            "treatments and reported indications, and RXDDAYS supplies duration rather than dose."
        ),
    ]
    if not recalibration_summary.empty:
        frozen = recalibration_summary.loc[
            recalibration_summary["strategy"].eq("frozen_nhanes_threshold")
        ].iloc[0]
        local = recalibration_summary.loc[
            recalibration_summary["strategy"].eq(
                "knhanes_year_nested_recalibration"
            )
        ].iloc[0]
        local_lcb = recalibration_summary.loc[
            recalibration_summary["strategy"].eq(
                "knhanes_year_nested_recalibration_lcb"
            )
        ].iloc[0]
        lines.extend(
            [
                "",
                "## KNHANES nested local recalibration",
                "",
                markdown_table(recalibration_summary),
                "",
                f"Nested Korean recalibration reduced mean referral from "
                f"{frozen['mean_referral_rate']:.1%} to "
                f"{local['mean_referral_rate']:.1%}, while mean sensitivity fell from "
                f"{frozen['mean_sensitivity']:.1%} to "
                f"{local['mean_sensitivity']:.1%} (minimum "
                f"{local['minimum_sensitivity']:.1%}). It therefore improves burden and "
                "calibration but does not maintain 90% sensitivity in every held-out year.",
                f"The lower-confidence-bound variant referred "
                f"{local_lcb['mean_referral_rate']:.1%}, with mean sensitivity "
                f"{local_lcb['mean_sensitivity']:.1%} and minimum sensitivity "
                f"{local_lcb['minimum_sensitivity']:.1%}.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_feature_combination_experiment(
    root: Path,
    overwrite: bool = False,
    config_filename: str = "feature_combination_experiment.yaml",
) -> dict:
    paths = ProjectPaths.from_root(root)
    paths.ensure()
    model_config = load_yaml(root / "configs" / "model_config.yaml")
    variables = load_yaml(root / "configs" / "variables.yaml")
    experiment = load_yaml(root / "configs" / config_filename)
    prefix = str(experiment.get("output_prefix", "feature_combination"))
    target_name = str(experiment.get("target", "osteoporosis"))
    output_paths = {
        "availability": paths.tables / f"{prefix}_availability.csv",
        "selection": paths.tables / f"{prefix}_inner_selection.csv",
        "metrics": paths.tables / f"{prefix}_outer_metrics.csv",
        "candidate_summary": paths.tables / f"{prefix}_candidate_summary.csv",
        "panel_summary": paths.tables / f"{prefix}_panel_summary.csv",
        "multi_metric_candidates": paths.tables
        / f"{prefix}_multi_metric_candidates.csv",
        "deltas": paths.tables / f"{prefix}_paired_deltas.csv",
        "predictions": paths.tables / f"{prefix}_selected_predictions.csv.gz",
        "clinical_score_metrics": paths.tables / f"{prefix}_clinical_score_metrics.csv",
        "clinical_score_predictions": paths.tables
        / f"{prefix}_clinical_score_predictions.csv.gz",
        "recalibration_metrics": paths.tables / f"{prefix}_knhanes_recalibration_metrics.csv",
        "recalibration_predictions": paths.tables
        / f"{prefix}_knhanes_recalibration_predictions.csv.gz",
        "manifest": paths.tables / f"{prefix}_manifest.json",
        "report": root
        / "reports"
        / str(experiment.get("report_filename", "FEATURE_COMBINATION_RESULTS.md")),
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Feature-combination outputs already exist; pass --overwrite to replace only "
            f"this additive namespace: {existing}"
        )

    frame = pd.read_csv(paths.processed / "classification_cohort.csv.gz", low_memory=False)
    frame, outcome_column, target_definition = _prepare_target_frame(
        frame, target_name, experiment
    )
    e12 = _experiment_feature_sets(frame, variables)["E12_e8_plus_medical_mobility"]
    panels = _normalize_feature_panels(experiment, variables, e12)
    availability = _panel_availability(
        frame,
        panels,
        float(experiment.get("minimum_cycle_observed_fraction", 0.25)),
    )
    eligible_panels = set(
        availability.loc[
            availability["eligible_all_production_cycles"], "panel"
        ].astype(str)
    )
    feature_sets = {
        panel: features for panel, features in panels.items() if panel in eligible_panels
    }
    if not feature_sets:
        availability.to_csv(output_paths["availability"], index=False)
        raise ValueError(
            "No feature panels met the minimum per-cycle observed-fraction rule; "
            f"availability audit written to {output_paths['availability']}"
        )
    categorical = set(variables["categorical_features"])
    candidates = _candidate_estimators(
        experiment,
        int(model_config["seed"]),
        int(model_config.get("cv_n_jobs", 1)),
    )
    target = float(experiment.get("recall_target", 0.90))
    confidence = float(experiment.get("recall_lcb_confidence", 0.95))
    selection_rows = []
    metric_rows = []
    prediction_rows = []

    checkpoint_signature = hashlib.sha256(
        json.dumps(
            {
                "experiment": experiment,
                "target": target_name,
                "seed": model_config["seed"],
                "data_mtime_ns": (
                    paths.processed / "classification_cohort.csv.gz"
                ).stat().st_mtime_ns,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    checkpoint_dir = paths.root / "tmp" / f"feature_combination_{checkpoint_signature}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for outer_cycle in sorted(frame["survey_cycle"].astype(str).unique()):
        cycle_slug = outer_cycle.replace("-", "_")
        selection_checkpoint = checkpoint_dir / f"{cycle_slug}_selection.csv"
        metrics_checkpoint = checkpoint_dir / f"{cycle_slug}_metrics.csv"
        predictions_checkpoint = checkpoint_dir / f"{cycle_slug}_predictions.csv.gz"
        if all(
            path.exists()
            for path in (
                selection_checkpoint,
                metrics_checkpoint,
                predictions_checkpoint,
            )
        ):
            selection_rows.append(pd.read_csv(selection_checkpoint))
            metric_rows.extend(pd.read_csv(metrics_checkpoint).to_dict("records"))
            prediction_rows.append(pd.read_csv(predictions_checkpoint))
            print(f"Resumed completed outer cycle {outer_cycle}", flush=True)
            continue

        training = frame.loc[frame["survey_cycle"].astype(str).ne(outer_cycle)].copy()
        evaluation = frame.loc[frame["survey_cycle"].astype(str).eq(outer_cycle)].copy()
        outcome_train = training[outcome_column].astype(int).to_numpy()
        outcome_evaluation = evaluation[outcome_column].astype(int).to_numpy()
        weight_train = training["mec_weight"].fillna(1.0).to_numpy()
        weight_evaluation = evaluation["mec_weight"].fillna(1.0).to_numpy()
        cache = {}
        outer_selection = []
        cycle_metric_rows = []
        for feature_set, features in feature_sets.items():
            panel_predictions = _shared_panel_predictions(
                training,
                evaluation,
                features,
                candidates,
                categorical,
                model_config,
                outcome_column=outcome_column,
            )
            for model_name, estimator in candidates:
                candidate = f"{feature_set}__{model_name}"
                raw_oof, raw_evaluation = panel_predictions[model_name]
                inner_pr_auc = average_precision_score(
                    outcome_train, raw_oof, sample_weight=weight_train
                )
                inner_roc_auc = roc_auc_score(
                    outcome_train, raw_oof, sample_weight=weight_train
                )
                calibrator = _platt_fit(outcome_train, raw_oof, weight_train)
                calibrated_oof = _platt_apply(calibrator, raw_oof)
                probability = _platt_apply(calibrator, raw_evaluation)
                point = select_recall_threshold(
                    outcome_train,
                    calibrated_oof,
                    weight_train,
                    target,
                    confidence,
                    require_lcb=False,
                )
                conservative = select_recall_threshold(
                    outcome_train,
                    calibrated_oof,
                    weight_train,
                    target,
                    confidence,
                    require_lcb=True,
                )
                point_metrics = _policy_metrics(
                    outcome_evaluation,
                    probability,
                    probability >= point["threshold"],
                    weight_evaluation,
                )
                lcb_metrics = _policy_metrics(
                    outcome_evaluation,
                    probability,
                    probability >= conservative["threshold"],
                    weight_evaluation,
                )
                selection = {
                    "outer_cycle": outer_cycle,
                    "candidate": candidate,
                    "feature_set": feature_set,
                    "model": model_name,
                    "feature_count": len(features),
                    "inner_oof_pr_auc": inner_pr_auc,
                    "inner_oof_roc_auc": inner_roc_auc,
                }
                outer_selection.append(selection)
                cycle_metric_rows.append(
                    {
                        **selection,
                        "evaluation_n": len(evaluation),
                        "roc_auc": point_metrics["roc_auc"],
                        "pr_auc": point_metrics["pr_auc"],
                        "log_loss": log_loss(
                            outcome_evaluation,
                            probability,
                            sample_weight=weight_evaluation,
                        ),
                        "brier_score": brier_score_loss(
                            outcome_evaluation,
                            probability,
                            sample_weight=weight_evaluation,
                        ),
                        "point_threshold": point["threshold"],
                        **_prefixed_metrics(
                            "point_", point_metrics, {"roc_auc", "pr_auc"}
                        ),
                        "lcb_threshold": conservative["threshold"],
                        **_prefixed_metrics(
                            "lcb_", lcb_metrics, {"roc_auc", "pr_auc"}
                        ),
                    }
                )
                cache[candidate] = (probability, point, conservative)
            print(
                f"Completed {outer_cycle}: {feature_set} ({len(candidates)} models)",
                flush=True,
            )

        selection_frame = pd.DataFrame(outer_selection).sort_values(
            ["inner_oof_pr_auc", "inner_oof_roc_auc", "candidate"],
            ascending=[False, False, True],
        )
        selected_by_panel = set(
            selection_frame.groupby("feature_set", sort=False).head(1)["candidate"]
        )
        selected_overall = str(selection_frame.iloc[0]["candidate"])
        selection_frame["selected_within_feature_set"] = selection_frame[
            "candidate"
        ].isin(selected_by_panel)
        selection_frame["selected_overall"] = selection_frame["candidate"].eq(
            selected_overall
        )
        selection_rows.append(selection_frame)
        for row in cycle_metric_rows:
            row["selected_within_feature_set"] = row["candidate"] in selected_by_panel
            row["selected_overall"] = row["candidate"] == selected_overall
        metric_rows.extend(cycle_metric_rows)
        cycle_predictions = []
        for candidate in selected_by_panel:
            probability, point, conservative = cache[candidate]
            predictions = evaluation[["participant_id", "survey_cycle", "mec_weight"]].copy()
            predictions["target"] = target_name
            predictions["outcome"] = evaluation[outcome_column].astype(int).to_numpy()
            predictions["candidate"] = candidate
            predictions["probability"] = probability
            predictions["point_threshold"] = point["threshold"]
            predictions["point_prediction"] = probability >= point["threshold"]
            predictions["lcb_threshold"] = conservative["threshold"]
            predictions["lcb_prediction"] = probability >= conservative["threshold"]
            prediction_rows.append(predictions)
            cycle_predictions.append(predictions)

        selection_frame.to_csv(selection_checkpoint, index=False)
        pd.DataFrame(cycle_metric_rows).to_csv(metrics_checkpoint, index=False)
        pd.concat(cycle_predictions, ignore_index=True).to_csv(
            predictions_checkpoint, index=False, compression="gzip"
        )
        print(f"Checkpointed outer cycle {outer_cycle}", flush=True)

    selection = pd.concat(selection_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    candidate_summary = (
        metrics.groupby(["candidate", "feature_set", "model"], as_index=False)
        .agg(
            outer_cycles=("outer_cycle", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_point_sensitivity=("point_sensitivity", "mean"),
            minimum_point_sensitivity=("point_sensitivity", "min"),
            mean_point_specificity=("point_specificity", "mean"),
            mean_point_ppv=("point_ppv", "mean"),
            mean_point_f1=("point_f1", "mean"),
            mean_lcb_sensitivity=("lcb_sensitivity", "mean"),
            minimum_lcb_sensitivity=("lcb_sensitivity", "min"),
            mean_lcb_specificity=("lcb_specificity", "mean"),
            mean_lcb_f1=("lcb_f1", "mean"),
        )
        .sort_values(["mean_pr_auc", "mean_roc_auc"], ascending=False)
    )
    selected = metrics.loc[metrics["selected_within_feature_set"]].copy()
    panel_summary = (
        selected.groupby("feature_set", as_index=False)
        .agg(
            outer_cycles=("outer_cycle", "nunique"),
            selected_models=("model", lambda values: "|".join(sorted(set(values)))),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_point_sensitivity=("point_sensitivity", "mean"),
            minimum_point_sensitivity=("point_sensitivity", "min"),
            mean_point_specificity=("point_specificity", "mean"),
            mean_point_ppv=("point_ppv", "mean"),
            mean_point_f1=("point_f1", "mean"),
            mean_lcb_sensitivity=("lcb_sensitivity", "mean"),
            minimum_lcb_sensitivity=("lcb_sensitivity", "min"),
            mean_lcb_specificity=("lcb_specificity", "mean"),
            mean_lcb_f1=("lcb_f1", "mean"),
        )
        .sort_values(["mean_pr_auc", "mean_roc_auc"], ascending=False)
    )
    e12_selected = panel_summary.loc[panel_summary["feature_set"].eq("E12_baseline")]
    if not e12_selected.empty:
        reference = e12_selected.iloc[0]
        multi_metric_candidates = candidate_summary.copy()
        for metric in (
            "mean_pr_auc",
            "mean_roc_auc",
            "mean_point_sensitivity",
            "mean_point_specificity",
            "mean_point_ppv",
            "mean_point_f1",
            "mean_lcb_sensitivity",
            "mean_lcb_specificity",
            "mean_lcb_f1",
        ):
            multi_metric_candidates[f"{metric}_delta_vs_selected_e12"] = (
                multi_metric_candidates[metric] - reference[metric]
            )
        multi_metric_candidates["dominates_selected_e12_point_policy"] = (
            multi_metric_candidates[
                [
                    "mean_pr_auc_delta_vs_selected_e12",
                    "mean_roc_auc_delta_vs_selected_e12",
                    "mean_point_sensitivity_delta_vs_selected_e12",
                    "mean_point_specificity_delta_vs_selected_e12",
                    "mean_point_ppv_delta_vs_selected_e12",
                    "mean_point_f1_delta_vs_selected_e12",
                ]
            ]
            .ge(0)
            .all(axis=1)
        )
        multi_metric_candidates = multi_metric_candidates.sort_values(
            [
                "dominates_selected_e12_point_policy",
                "mean_pr_auc",
                "mean_point_ppv",
            ],
            ascending=[False, False, False],
        )
    else:
        multi_metric_candidates = pd.DataFrame()
    deltas = pd.concat(
        [_paired_panel_deltas(selected, metric) for metric in ("pr_auc", "roc_auc")],
        ignore_index=True,
    )
    if target_name == "osteoporosis":
        clinical_score_metrics, clinical_score_predictions = _clinical_score_baselines(
            frame, target, confidence
        )
    else:
        clinical_score_metrics, clinical_score_predictions = pd.DataFrame(), pd.DataFrame()
    clinical_score_summary = (
        clinical_score_metrics.groupby(["score", "scope", "policy"], as_index=False)
        .agg(
            outer_cycles=("outer_cycle", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_sensitivity=("sensitivity", "mean"),
            minimum_sensitivity=("sensitivity", "min"),
            mean_specificity=("specificity", "mean"),
            mean_ppv=("ppv", "mean"),
            mean_npv=("npv", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_mcc=("mcc", "mean"),
            mean_f1=("f1", "mean"),
            mean_referral_rate=("referral_rate", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier_score=("brier_score", "mean"),
        )
        .sort_values(["mean_pr_auc", "mean_roc_auc"], ascending=False)
        if not clinical_score_metrics.empty
        else pd.DataFrame()
    )
    if target_name == "osteoporosis":
        recalibration_metrics, recalibration_predictions = _knhanes_local_recalibration(
            root, target, confidence
        )
    else:
        recalibration_metrics, recalibration_predictions = pd.DataFrame(), pd.DataFrame()
    recalibration_summary = (
        recalibration_metrics.groupby("strategy", as_index=False)
        .agg(
            years=("outer_year", "nunique"),
            mean_sensitivity=("sensitivity", "mean"),
            minimum_sensitivity=("sensitivity", "min"),
            mean_specificity=("specificity", "mean"),
            mean_ppv=("ppv", "mean"),
            mean_referral_rate=("referral_rate", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier_score=("brier_score", "mean"),
        )
        if not recalibration_metrics.empty
        else pd.DataFrame()
    )

    availability.to_csv(output_paths["availability"], index=False)
    selection.to_csv(output_paths["selection"], index=False)
    metrics.to_csv(output_paths["metrics"], index=False)
    candidate_summary.to_csv(output_paths["candidate_summary"], index=False)
    panel_summary.to_csv(output_paths["panel_summary"], index=False)
    multi_metric_candidates.to_csv(
        output_paths["multi_metric_candidates"], index=False
    )
    deltas.to_csv(output_paths["deltas"], index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output_paths["predictions"], index=False, compression="gzip"
    )
    clinical_score_metrics.to_csv(output_paths["clinical_score_metrics"], index=False)
    if not clinical_score_predictions.empty:
        clinical_score_predictions.to_csv(
            output_paths["clinical_score_predictions"],
            index=False,
            compression="gzip",
        )
    recalibration_metrics.to_csv(output_paths["recalibration_metrics"], index=False)
    if not recalibration_predictions.empty:
        recalibration_predictions.to_csv(
            output_paths["recalibration_predictions"],
            index=False,
            compression="gzip",
        )
    _write_report(
        output_paths["report"],
        panel_summary,
        candidate_summary,
        multi_metric_candidates,
        deltas,
        availability,
        clinical_score_summary,
        recalibration_summary,
    )
    manifest = {
        "experiment": "requested_feature_and_model_combinations",
        "target": target_name,
        "target_definition": target_definition,
        "production_outputs_modified": False,
        "outer_validation": "leave_one_production_cycle_out",
        "inner_selection": "leave_one_remaining_cycle_out",
        "selection_metric": "survey_weighted_pr_auc",
        "eligible_feature_panels": sorted(feature_sets),
        "skipped_feature_panels": sorted(set(panels) - set(feature_sets)),
        "candidate_models": [name for name, _ in candidates],
        "hyperparameter_grid": {
            key: experiment.get(key)
            for key in (
                "logistic_c",
                "lasso_selection_c",
                "tree_n_estimators",
                "random_forest_min_samples_leaf",
                "random_forest_max_features",
                "xgboost_max_depth",
                "xgboost_learning_rate",
                "xgboost_min_child_weight",
                "lightgbm_num_leaves",
                "lightgbm_learning_rate",
                "catboost_depth",
                "catboost_learning_rate",
                "mlp_hidden_layer_sizes",
                "mlp_alpha",
                "spline_logistic_c",
            )
            if key in experiment
        },
        "stacking": (
            "included as an inner-fold probability stack with logistic meta-learner"
            if any(name == "stacked_tree_ensemble" for name, _ in candidates)
            else "not available because fewer than two tree/boosting libraries were importable"
        ),
        "resampling": (
            "included as optional imbalanced-learn candidates"
            if any(name.startswith(("smote_", "adasyn_", "random_over_")) for name, _ in candidates)
            else (
                "requested but skipped because imbalanced-learn is not installed"
                if experiment.get("include_resampling_candidates", False)
                else "not included by default; enable include_resampling_candidates after installing imbalanced-learn"
            )
        ),
        "clinical_score_baselines": (
            clinical_score_summary.to_dict(orient="records")
            if not clinical_score_summary.empty
            else []
        ),
        "medication_definition": {
            "classes": [
                "systemic glucocorticoids",
                "anticonvulsants",
                "aromatase inhibitors",
                "proton-pump inhibitors",
            ],
            "long_term_days": 90,
            "dose_available": False,
            "leakage_exclusions": [
                "osteoporosis treatments",
                "reported medication indications",
                "self-reported osteoporosis",
            ],
        },
        "knhanes_recalibration": (
            "nested leave-one-survey-year-out Platt calibration and threshold selection"
            if not recalibration_metrics.empty
            else "not run; frozen external predictions unavailable"
        ),
        "output_files": {
            key: str(value.relative_to(root)) for key, value in output_paths.items()
        },
    }
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "best_panel": str(panel_summary.iloc[0]["feature_set"]),
        "panel_summary": panel_summary.to_dict(orient="records"),
        "clinical_score_summary": clinical_score_summary.to_dict(orient="records"),
        "recalibration_summary": recalibration_summary.to_dict(orient="records"),
        "outputs": {key: str(value) for key, value in output_paths.items()},
    }
