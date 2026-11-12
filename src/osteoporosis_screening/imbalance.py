from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

from .config import ProjectPaths, load_yaml
from .curve_bending import _platt_apply, _platt_fit, _policy_metrics
from .evaluation import (
    threshold_for_f1,
    threshold_for_precision,
    threshold_for_sensitivity,
)
from .feature_combination import (
    RoutedCatBoostClassifier,
    RoutedResampledClassifier,
    _normalize_feature_panels,
    _panel_availability,
    _shared_panel_predictions,
)
from .modeling import XGBClassifier, feature_experiments


class PositiveWeightClassifier(ClassifierMixin, BaseEstimator):
    """Apply an extra positive-class weight on top of survey sample weights."""

    def __init__(self, estimator, mode: str = "balanced") -> None:
        self.estimator = estimator
        self.mode = mode

    def fit(self, X, y, sample_weight=None):
        y_array = np.asarray(y, dtype=int)
        weight = (
            np.ones(len(y_array), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).copy()
        )
        pos_weight = float(weight[y_array == 1].sum())
        neg_weight = float(weight[y_array == 0].sum())
        ratio = neg_weight / pos_weight if pos_weight else 1.0
        if self.mode == "none":
            multiplier = 1.0
        elif self.mode == "sqrt_balanced":
            multiplier = float(np.sqrt(ratio))
        elif self.mode == "balanced":
            multiplier = ratio
        else:
            raise ValueError("mode must be one of none, sqrt_balanced, balanced")
        fit_weight = weight.copy()
        fit_weight[y_array == 1] *= multiplier
        self.model_ = clone(self.estimator)
        if hasattr(self.model_, "set_fit_request"):
            self.model_ = self.model_.set_fit_request(sample_weight=True)
        self.model_.fit(X, y_array, sample_weight=fit_weight)
        self.classes_ = np.asarray(getattr(self.model_, "classes_", [0, 1]))
        self.positive_weight_multiplier_ = multiplier
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def _base_e12_features(frame: pd.DataFrame, variables: dict) -> list[str]:
    experiments = feature_experiments(
        frame, variables, include_medical_mobility_panel=True
    )
    return list(dict.fromkeys(experiments["E12_e8_plus_medical_mobility"][0]))


def _available_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [feature for feature in features if feature in frame.columns]


def _candidate_estimators(config: dict, seed: int, n_jobs: int) -> tuple[list[tuple[str, object]], dict]:
    base_estimators: list[tuple[str, object]] = []
    if config.get("models", {}).get("logistic", True):
        base_estimators.append(
            (
                "logistic",
                LogisticRegression(C=1.0, max_iter=3000, random_state=seed),
            )
        )
    if config.get("models", {}).get("random_forest", True):
        base_estimators.append(
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=int(config.get("tree_n_estimators", 150)),
                    min_samples_leaf=int(config.get("random_forest_min_samples_leaf", 5)),
                    max_features="sqrt",
                    n_jobs=n_jobs,
                    random_state=seed,
                ),
            )
        )
    if config.get("models", {}).get("xgboost", True) and XGBClassifier is not None:
        base_estimators.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=int(config.get("tree_n_estimators", 150)),
                    max_depth=int(config.get("xgboost_max_depth", 3)),
                    learning_rate=float(config.get("xgboost_learning_rate", 0.03)),
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=2.0,
                    min_child_weight=int(config.get("xgboost_min_child_weight", 3)),
                    tree_method="hist",
                    eval_metric="logloss",
                    n_jobs=n_jobs,
                    random_state=seed,
                ),
            )
        )
    if config.get("models", {}).get("lightgbm", True) and LGBMClassifier is not None:
        base_estimators.append(
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=int(config.get("tree_n_estimators", 150)),
                    num_leaves=int(config.get("lightgbm_num_leaves", 15)),
                    learning_rate=float(config.get("lightgbm_learning_rate", 0.03)),
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=2.0,
                    min_child_samples=20,
                    n_jobs=n_jobs,
                    random_state=seed,
                    verbose=-1,
                ),
            )
        )
    if config.get("models", {}).get("catboost", True):
        base_estimators.append(
            (
                "catboost",
                RoutedCatBoostClassifier(
                    n_estimators=int(config.get("tree_n_estimators", 150)),
                    depth=int(config.get("catboost_depth", 4)),
                    learning_rate=float(config.get("catboost_learning_rate", 0.03)),
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            )
        )

    candidates: list[tuple[str, object]] = []
    for mode in config.get("positive_weight_modes", ["none", "sqrt_balanced", "balanced"]):
        for name, estimator in base_estimators:
            suffix = "survey_weight_only" if mode == "none" else f"positive_{mode}"
            candidates.append(
                (f"{name}__{suffix}", PositiveWeightClassifier(clone(estimator), str(mode)))
            )

    resampling_status = {
        "requested": bool(config.get("include_resampling_candidates", False)),
        "available": importlib.util.find_spec("imblearn") is not None,
        "included": [],
    }
    if resampling_status["requested"] and resampling_status["available"]:
        base = next(
            (
                clone(estimator)
                for name, estimator in base_estimators
                if name in {"xgboost", "lightgbm", "random_forest"}
            ),
            None,
        )
        if base is not None:
            for method in config.get("resampling_methods", ["smote", "adasyn"]):
                name = f"{method}__{base.__class__.__name__.lower()}"
                candidates.append(
                    (
                        name,
                        RoutedResampledClassifier(
                            clone(base), method=str(method), random_state=seed
                        ),
                    )
                )
                resampling_status["included"].append(name)
    return candidates, resampling_status


def _thresholds_from_oof(y, probability, weight, config: dict) -> list[dict]:
    rows = [
        {
            "policy": "default_0.50",
            "threshold": 0.5,
            "target_recall": np.nan,
            "target_precision": np.nan,
        },
        {
            "policy": "f1_optimized",
            "threshold": threshold_for_f1(y, probability, weight),
            "target_recall": np.nan,
            "target_precision": np.nan,
        },
    ]
    for target in config.get("recall_targets", [0.90, 0.80, 0.70]):
        rows.append(
            {
                "policy": f"recall_{float(target):.2f}",
                "threshold": threshold_for_sensitivity(y, probability, float(target), weight),
                "target_recall": float(target),
                "target_precision": np.nan,
            }
        )
    for target in config.get("precision_targets", [0.20, 0.25, 0.30]):
        rows.append(
            {
                "policy": f"precision_{float(target):.2f}",
                "threshold": threshold_for_precision(y, probability, float(target), weight),
                "target_recall": np.nan,
                "target_precision": float(target),
            }
        )
    return rows


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            ["candidate", "feature_set", "model", "imbalance_strategy", "policy"],
            as_index=False,
        )
        .agg(
            outer_cycles=("outer_cycle", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_sensitivity=("sensitivity", "mean"),
            minimum_sensitivity=("sensitivity", "min"),
            mean_specificity=("specificity", "mean"),
            mean_ppv=("ppv", "mean"),
            mean_npv=("npv", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_mcc=("mcc", "mean"),
            mean_f1=("f1", "mean"),
            mean_referral_rate=("referral_rate", "mean"),
        )
        .sort_values(["mean_pr_auc", "mean_roc_auc", "mean_f1"], ascending=False)
    )


def _write_report(path: Path, summary: pd.DataFrame, manifest: dict) -> None:
    best_pr = summary.sort_values(
        ["mean_pr_auc", "mean_roc_auc", "mean_f1"], ascending=False
    ).head(8)
    best_screening = summary.loc[summary["minimum_sensitivity"].ge(0.90)].sort_values(
        ["mean_ppv", "mean_specificity", "mean_pr_auc"], ascending=False
    ).head(8)
    best_f1 = summary.sort_values(
        ["mean_f1", "mean_ppv", "mean_pr_auc"], ascending=False
    ).head(8)

    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return ""
        formatted = frame.copy()
        for column in formatted.columns:
            if pd.api.types.is_numeric_dtype(formatted[column]):
                formatted[column] = formatted[column].map(
                    lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
                )
            else:
                formatted[column] = formatted[column].astype("string").fillna("")
        headers = list(formatted.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in formatted.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    lines = [
        "# Osteoporosis imbalance experiment",
        "",
        "This additive experiment tests osteoporosis-only class-weighting, feature-panel, "
        "calibration, and threshold policies. It does not replace production outputs.",
        "",
        f"Production outputs modified: `{manifest['production_outputs_modified']}`",
        "",
        "## Best PR-AUC rows",
        "",
        markdown_table(best_pr),
        "",
        "## Best high-recall rows",
        "",
        (
            markdown_table(best_screening)
            if not best_screening.empty
            else "No row maintained minimum outer-cycle sensitivity >= 0.90."
        ),
        "",
        "## Best F1 rows",
        "",
        markdown_table(best_f1),
        "",
        "## Notes",
        "",
        "- Positive-class weighting is learned inside each outer/inner training partition.",
        "- Thresholds are selected from calibrated out-of-fold training predictions only.",
        "- SMOTE/ADASYN/random oversampling are included only when `imbalanced-learn` is installed.",
        "- F1-optimized rows are useful as threshold-policy diagnostics, not as a screening default.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_osteoporosis_imbalance_experiment(root: Path, overwrite: bool = False) -> dict:
    paths = ProjectPaths.from_root(root)
    paths.ensure()
    model_config = load_yaml(root / "configs" / "model_config.yaml")
    variables = load_yaml(root / "configs" / "variables.yaml")
    config = load_yaml(root / "configs" / "osteoporosis_imbalance_experiment.yaml")
    prefix = str(config.get("output_prefix", "osteoporosis_imbalance"))
    output_paths = {
        "availability": paths.tables / f"{prefix}_availability.csv",
        "outer_metrics": paths.tables / f"{prefix}_outer_metrics.csv",
        "candidate_summary": paths.tables / f"{prefix}_candidate_summary.csv",
        "thresholds": paths.tables / f"{prefix}_thresholds.csv",
        "predictions": paths.tables / f"{prefix}_predictions.csv.gz",
        "manifest": paths.tables / f"{prefix}_manifest.json",
        "report": root
        / "reports"
        / str(config.get("report_filename", "OSTEOPOROSIS_IMBALANCE_RESULTS.md")),
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Osteoporosis imbalance outputs already exist; pass --overwrite to replace "
            f"only this additive namespace: {existing}"
        )

    frame = pd.read_csv(paths.processed / "classification_cohort.csv.gz", low_memory=False)
    frame = frame.dropna(subset=["osteoporosis"]).copy()
    e12 = _base_e12_features(frame, variables)
    panels = _normalize_feature_panels(config, variables, e12)
    panels = {
        name: _available_features(frame, list(dict.fromkeys(features)))
        for name, features in panels.items()
    }
    availability = _panel_availability(
        frame,
        panels,
        float(config.get("minimum_cycle_observed_fraction", 0.25)),
    )
    eligible_panels = set(
        availability.loc[
            availability["eligible_all_production_cycles"], "panel"
        ].astype(str)
    )
    feature_sets = {
        name: features for name, features in panels.items() if name in eligible_panels
    }
    if not feature_sets:
        availability.to_csv(output_paths["availability"], index=False)
        raise ValueError("No eligible feature panels for imbalance experiment")

    categorical = set(variables["categorical_features"])
    if "dietary_recall_status" in frame:
        categorical.add("dietary_recall_status")
    candidates, resampling_status = _candidate_estimators(
        config,
        int(model_config["seed"]),
        int(model_config.get("cv_n_jobs", 1)),
    )
    metric_rows = []
    threshold_rows = []
    prediction_rows = []

    for outer_cycle in sorted(frame["survey_cycle"].astype(str).unique()):
        training = frame.loc[frame["survey_cycle"].astype(str).ne(outer_cycle)].copy()
        evaluation = frame.loc[frame["survey_cycle"].astype(str).eq(outer_cycle)].copy()
        y_train = training["osteoporosis"].astype(int).to_numpy()
        y_eval = evaluation["osteoporosis"].astype(int).to_numpy()
        w_train = training["mec_weight"].fillna(1.0).to_numpy()
        w_eval = evaluation["mec_weight"].fillna(1.0).to_numpy()
        for feature_set, features in feature_sets.items():
            panel_predictions = _shared_panel_predictions(
                training,
                evaluation,
                features,
                candidates,
                categorical,
                model_config,
                outcome_column="osteoporosis",
            )
            for model_name, _ in candidates:
                candidate = f"{feature_set}__{model_name}"
                if "__positive_" in model_name:
                    model, imbalance_strategy = model_name.split("__positive_", 1)
                    imbalance_strategy = f"positive_{imbalance_strategy}"
                elif "__survey_weight_only" in model_name:
                    model, _ = model_name.split("__survey_weight_only", 1)
                    imbalance_strategy = "survey_weight_only"
                elif "__" in model_name:
                    model, imbalance_strategy = model_name.split("__", 1)
                else:
                    model, imbalance_strategy = model_name, "resampling"
                raw_oof, raw_eval = panel_predictions[model_name]
                calibrator = _platt_fit(y_train, raw_oof, w_train)
                calibrated_oof = _platt_apply(calibrator, raw_oof)
                probability = _platt_apply(calibrator, raw_eval)
                inner_pr_auc = average_precision_score(
                    y_train, calibrated_oof, sample_weight=w_train
                )
                inner_roc_auc = roc_auc_score(
                    y_train, calibrated_oof, sample_weight=w_train
                )
                for threshold in _thresholds_from_oof(
                    y_train, calibrated_oof, w_train, config
                ):
                    predicted = probability >= float(threshold["threshold"])
                    metrics = _policy_metrics(y_eval, probability, predicted, w_eval)
                    record = {
                        "outer_cycle": outer_cycle,
                        "candidate": candidate,
                        "feature_set": feature_set,
                        "model": model,
                        "imbalance_strategy": imbalance_strategy,
                        "policy": threshold["policy"],
                        "evaluation_n": len(evaluation),
                        "inner_oof_pr_auc": inner_pr_auc,
                        "inner_oof_roc_auc": inner_roc_auc,
                        "threshold": threshold["threshold"],
                        "target_recall": threshold["target_recall"],
                        "target_precision": threshold["target_precision"],
                        "log_loss": log_loss(y_eval, probability, sample_weight=w_eval),
                        "brier_score": brier_score_loss(
                            y_eval, probability, sample_weight=w_eval
                        ),
                        **metrics,
                    }
                    metric_rows.append(record)
                    threshold_rows.append(
                        {
                            key: record[key]
                            for key in (
                                "outer_cycle",
                                "candidate",
                                "feature_set",
                                "model",
                                "imbalance_strategy",
                                "policy",
                                "threshold",
                                "target_recall",
                                "target_precision",
                                "inner_oof_pr_auc",
                                "inner_oof_roc_auc",
                            )
                        }
                    )
                predictions = evaluation[
                    ["participant_id", "survey_cycle", "mec_weight"]
                ].copy()
                predictions["target"] = "osteoporosis"
                predictions["outcome"] = y_eval
                predictions["candidate"] = candidate
                predictions["feature_set"] = feature_set
                predictions["model"] = model
                predictions["imbalance_strategy"] = imbalance_strategy
                predictions["probability"] = probability
                prediction_rows.append(predictions)
            print(
                f"Completed imbalance outer cycle {outer_cycle}: {feature_set}",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    summary = _summarize_metrics(metrics)
    manifest = {
        "experiment": "osteoporosis_imbalance",
        "target": "osteoporosis",
        "target_definition": "femoral-neck T-score <= -2.5",
        "production_outputs_modified": False,
        "outer_validation": "leave_one_production_cycle_out",
        "inner_predictions": "leave_one_remaining_cycle_out",
        "calibration": "Platt scaling fit on grouped out-of-fold training predictions",
        "threshold_selection": "calibrated out-of-fold training predictions only",
        "feature_panels": sorted(feature_sets),
        "skipped_feature_panels": sorted(set(panels) - set(feature_sets)),
        "candidate_models": [name for name, _ in candidates],
        "resampling": resampling_status,
        "focal_loss": "not run; no neural-net focal-loss dependency is available in this sklearn tabular pipeline",
        "autoencoder_anomaly_detection": "intentionally skipped; osteoporosis is modeled as graded risk, not reconstruction anomaly",
        "output_files": {
            key: str(value.relative_to(root)) for key, value in output_paths.items()
        },
    }

    availability.to_csv(output_paths["availability"], index=False)
    metrics.to_csv(output_paths["outer_metrics"], index=False)
    summary.to_csv(output_paths["candidate_summary"], index=False)
    pd.DataFrame(threshold_rows).to_csv(output_paths["thresholds"], index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output_paths["predictions"], index=False, compression="gzip"
    )
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(output_paths["report"], summary, manifest)
    best_pr = summary.iloc[0].to_dict()
    high_recall = summary.loc[summary["minimum_sensitivity"].ge(0.90)].sort_values(
        ["mean_ppv", "mean_specificity", "mean_pr_auc"], ascending=False
    )
    best_high_recall = high_recall.iloc[0].to_dict() if not high_recall.empty else None
    best_f1 = summary.sort_values(
        ["mean_f1", "mean_ppv", "mean_pr_auc"], ascending=False
    ).iloc[0].to_dict()
    return {
        "best_pr_auc": best_pr,
        "best_high_recall": best_high_recall,
        "best_f1": best_f1,
        "resampling": resampling_status,
        "outputs": {key: str(value) for key, value in output_paths.items()},
    }
