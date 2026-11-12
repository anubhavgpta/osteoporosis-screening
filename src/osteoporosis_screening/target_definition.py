from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .config import ProjectPaths, load_yaml
from .curve_bending import (
    _experiment_feature_sets,
    _platt_apply,
    _platt_fit,
    _policy_metrics,
    select_recall_threshold,
)
from .evaluation import classification_metrics
from .feature_combination import (
    _candidate_estimators,
    _prefixed_metrics,
    _shared_panel_predictions,
)
from .modeling import _preprocessor, run_classification
from .reporting import classification_bootstrap_intervals


FEATURE_COMBINATION_RIGOROUS_METRIC_COLUMNS = [
    "outer_cycle",
    "candidate",
    "feature_set",
    "model",
    "feature_count",
    "inner_oof_pr_auc",
    "inner_oof_roc_auc",
    "evaluation_n",
    "roc_auc",
    "pr_auc",
    "log_loss",
    "brier_score",
    "point_threshold",
    "point_sensitivity",
    "point_specificity",
    "point_ppv",
    "point_npv",
    "point_balanced_accuracy",
    "point_mcc",
    "point_f1",
    "point_referral_rate",
    "point_weighted_tn",
    "point_weighted_fp",
    "point_weighted_fn",
    "point_weighted_tp",
    "lcb_threshold",
    "lcb_sensitivity",
    "lcb_specificity",
    "lcb_ppv",
    "lcb_npv",
    "lcb_balanced_accuracy",
    "lcb_mcc",
    "lcb_f1",
    "lcb_referral_rate",
    "lcb_weighted_tn",
    "lcb_weighted_fp",
    "lcb_weighted_fn",
    "lcb_weighted_tp",
    "selected_within_feature_set",
    "selected_overall",
]


def rigorous_feature_combination_metric_columns() -> list[str]:
    """Expose the feature-combination metric layout for structural tests."""
    return list(FEATURE_COMBINATION_RIGOROUS_METRIC_COLUMNS)


def derive_low_bmd_label(frame: pd.DataFrame, cutoff: float = -1.0) -> pd.Series:
    """Derive low-BMD labels from the existing femoral-neck T-score."""
    score = frame["femoral_neck_t_score"]
    label = score.lt(float(cutoff)).astype("Int64")
    label.loc[score.isna()] = pd.NA
    return label


def _targeted_frame(frame: pd.DataFrame, target: str, cutoff: float) -> pd.DataFrame:
    result = frame.copy()
    if "low_bmd" not in result:
        result["low_bmd"] = derive_low_bmd_label(result, cutoff)
    if target == "osteoporosis":
        result["analysis_outcome"] = result["osteoporosis"].astype("Int64")
    elif target == "low_bmd":
        result["analysis_outcome"] = result["low_bmd"].astype("Int64")
    else:
        raise ValueError("target must be 'osteoporosis' or 'low_bmd'")
    return result.dropna(subset=["analysis_outcome"]).copy()


def _as_osteoporosis_target(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["osteoporosis"] = result["analysis_outcome"].astype("Int64")
    return result


def _cell_metrics(
    target: str,
    methodology: str,
    evaluation_design: str,
    outcome,
    probability,
    threshold: float,
    sample_weight=None,
    extra: dict | None = None,
) -> dict:
    metrics = classification_metrics(
        outcome, probability, threshold=threshold, sample_weight=sample_weight
    )
    record = {
        "target": target,
        "methodology": methodology,
        "evaluation_design": evaluation_design,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "ppv": metrics["ppv"],
        "f1": metrics["f1"],
        "calibration_slope": metrics["calibration_slope"],
        "calibration_intercept": metrics["calibration_intercept"],
        "brier_score": metrics["brier_score"],
        "threshold": threshold,
    }
    if extra:
        record.update(extra)
    return record


def _run_rigorous_loco(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    categorical: set[str],
    model_config: dict,
    experiment_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis = _as_osteoporosis_target(frame)
    candidates = _candidate_estimators(
        experiment_config,
        int(model_config["seed"]),
        int(experiment_config.get("n_jobs", model_config.get("cv_n_jobs", 1))),
    )
    recall_target = float(experiment_config.get("recall_target", 0.90))
    confidence = float(experiment_config.get("recall_lcb_confidence", 0.95))
    metric_rows = []
    cell_rows = []
    prediction_rows = []
    for outer_cycle in sorted(analysis["survey_cycle"].astype(str).unique()):
        training = analysis.loc[analysis["survey_cycle"].astype(str).ne(outer_cycle)].copy()
        evaluation = analysis.loc[analysis["survey_cycle"].astype(str).eq(outer_cycle)].copy()
        outcome_train = training["osteoporosis"].astype(int).to_numpy()
        outcome_evaluation = evaluation["osteoporosis"].astype(int).to_numpy()
        weight_train = training["mec_weight"].fillna(1.0).to_numpy()
        weight_evaluation = evaluation["mec_weight"].fillna(1.0).to_numpy()
        panel_predictions = _shared_panel_predictions(
            training,
            evaluation,
            features,
            candidates,
            categorical,
            model_config,
        )
        cache = {}
        cycle_rows = []
        for model_name, _ in candidates:
            candidate = f"E12_baseline__{model_name}"
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
                recall_target,
                confidence,
                require_lcb=False,
            )
            conservative = select_recall_threshold(
                outcome_train,
                calibrated_oof,
                weight_train,
                recall_target,
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
            row = {
                "target": target,
                "methodology": "rigorous",
                "outer_cycle": outer_cycle,
                "candidate": candidate,
                "feature_set": "E12_baseline",
                "model": model_name,
                "feature_count": len(features),
                "inner_oof_pr_auc": inner_pr_auc,
                "inner_oof_roc_auc": inner_roc_auc,
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
                **_prefixed_metrics("point_", point_metrics, {"roc_auc", "pr_auc"}),
                "lcb_threshold": conservative["threshold"],
                **_prefixed_metrics("lcb_", lcb_metrics, {"roc_auc", "pr_auc"}),
            }
            cycle_rows.append(row)
            cache[candidate] = (probability, point)
        selection = pd.DataFrame(cycle_rows).sort_values(
            ["inner_oof_pr_auc", "inner_oof_roc_auc", "candidate"],
            ascending=[False, False, True],
        )
        selected_name = str(selection.iloc[0]["candidate"])
        for row in cycle_rows:
            row["selected_within_feature_set"] = row["candidate"] == selected_name
            row["selected_overall"] = row["candidate"] == selected_name
        metric_rows.extend(cycle_rows)
        probability, point = cache[selected_name]
        cell_rows.append(
            _cell_metrics(
                target,
                "rigorous",
                "leave_one_production_cycle_out",
                outcome_evaluation,
                probability,
                float(point["threshold"]),
                sample_weight=weight_evaluation,
                extra={
                    "outer_cycle": outer_cycle,
                    "model": selection.iloc[0]["model"],
                    "evaluation_n": len(evaluation),
                    "sample_weight": "mec_weight",
                    "uncertainty": "PSU-within-stratum bootstrap on temporal output",
                },
            )
        )
        predictions = evaluation[
            [
                "participant_id",
                "survey_cycle",
                "survey_stratum",
                "survey_psu",
                "mec_weight",
                "analysis_outcome",
            ]
        ].copy()
        predictions["target"] = target
        predictions["methodology"] = "rigorous"
        predictions["probability"] = probability
        predictions["threshold"] = float(point["threshold"])
        predictions["prediction"] = (probability >= point["threshold"]).astype("int8")
        predictions["analysis_weight"] = weight_evaluation
        prediction_rows.append(predictions)
        print(f"rigorous_completed_outer_cycle target={target} cycle={outer_cycle}", flush=True)
    return (
        pd.DataFrame(metric_rows)[["target", "methodology", *FEATURE_COMBINATION_RIGOROUS_METRIC_COLUMNS]],
        pd.DataFrame(cell_rows),
        pd.concat(prediction_rows, ignore_index=True),
    )


def _temporal_rigorous_metrics(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    categorical: set[str],
    model_config: dict,
    experiment_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_frame = frame.copy()
    target_frame[target] = target_frame["analysis_outcome"].astype("Int64")
    temporal_cycle = str(
        experiment_config.get(
            "temporal_evaluation_cycle",
            model_config.get("temporal_evaluation_cycle", "2017-2018"),
        )
    )
    result = run_classification(
        target_frame,
        features,
        categorical,
        model_config,
        add_clusters=False,
        target=target,
        sample_weight_column="mec_weight",
        evaluation_cycle=temporal_cycle,
        selection_metric=str(model_config.get("classification_selection_metric", "pr_auc")),
    )
    metrics = result.metrics.copy()
    metrics.insert(0, "methodology", "rigorous")
    threshold = float(
        result.metrics.loc[
            result.metrics["evaluation"].eq("sensitivity_oriented"), "threshold"
        ].iloc[0]
    )
    interval_thresholds = {
        str(row.evaluation): float(row.threshold)
        for row in result.metrics.itertuples(index=False)
        if row.evaluation in {"sensitivity_oriented", "f1_optimized"}
    }
    intervals = classification_bootstrap_intervals(
        result.predictions,
        interval_thresholds,
        int(experiment_config.get("bootstrap_iterations", model_config["bootstrap_iterations"])),
        int(experiment_config.get("seed", model_config["seed"])),
        target=target,
    )
    intervals.insert(0, "target", target)
    intervals.insert(1, "methodology", "rigorous")
    predictions = result.predictions.rename(columns={target: "analysis_outcome"}).copy()
    predictions["target"] = target
    predictions["methodology"] = "rigorous_temporal"
    predictions["threshold"] = threshold
    predictions["prediction"] = predictions["screening_prediction"]
    f1_threshold = interval_thresholds.get("f1_optimized")
    if f1_threshold is not None:
        predictions["f1_optimized_threshold"] = f1_threshold
        predictions["f1_optimized_prediction"] = (
            predictions["probability"] >= f1_threshold
        ).astype(int)
    return metrics, intervals, predictions


def _naive_lasso_full_dataset_selection(
    frame: pd.DataFrame,
    features: list[str],
    categorical: set[str],
    config: dict,
) -> dict:
    outcome = frame["analysis_outcome"].astype(int).to_numpy()
    preprocess = _preprocessor(features, categorical, int(config["seed"]))
    transformed = preprocess.fit_transform(frame[features])
    selector_estimator = LogisticRegression(
        C=float(config.get("naive_lasso_selection_c", 0.1)),
        l1_ratio=1.0,
        solver="liblinear",
        max_iter=3000,
        random_state=int(config["seed"]),
    )
    selector = SelectFromModel(selector_estimator, threshold="median")
    selector.fit(transformed, outcome)
    selected = selector.transform(transformed)
    return {
        "preprocessor": preprocess,
        "selector": selector,
        "selected_features": selected,
        "selector_fit_n": int(len(frame)),
        "selected_feature_count": int(selected.shape[1]),
    }


def _maybe_smote(X, y, seed: int, enabled: bool):
    if not enabled or importlib.util.find_spec("imblearn") is None:
        return X, y, False
    from imblearn.over_sampling import SMOTE

    sampler = SMOTE(random_state=seed)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled, True


def _run_naive(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    categorical: set[str],
    experiment_config: dict,
    use_smote: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = int(experiment_config.get("seed", 42))
    recall_target = float(experiment_config.get("recall_target", 0.90))
    split = float(experiment_config.get("naive_test_size", 0.20))
    audit = _naive_lasso_full_dataset_selection(
        frame, features, categorical, {**experiment_config, "seed": seed}
    )
    X = audit["selected_features"]
    y = frame["analysis_outcome"].astype(int).to_numpy()
    train_idx, test_idx = train_test_split(
        np.arange(len(frame)),
        test_size=split,
        random_state=seed,
        stratify=y,
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_fit, y_fit, smote_applied = _maybe_smote(X_train, y_train, seed, use_smote)
    folds = min(int(experiment_config.get("cv_folds", 5)), int(pd.Series(y_fit).value_counts().min()))
    cv = StratifiedKFold(n_splits=max(2, folds), shuffle=True, random_state=seed)
    estimator = LogisticRegression(C=1.0, max_iter=3000, random_state=seed)
    scores = cross_validate(
        clone(estimator),
        X_fit,
        y_fit,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        n_jobs=int(experiment_config.get("n_jobs", 1)),
    )
    fitted = clone(estimator)
    fitted.fit(X_fit, y_fit)
    train_probability = fitted.predict_proba(X_train)[:, 1]
    test_probability = fitted.predict_proba(X_test)[:, 1]
    threshold = select_recall_threshold(
        y_train,
        train_probability,
        np.ones(len(y_train), dtype=float),
        recall_target,
        require_lcb=False,
    )["threshold"]
    label = "naive_smote" if smote_applied else "naive"
    metrics = _cell_metrics(
        target,
        label,
        "naive_random_80_20_unweighted_leaky_selection",
        y_test,
        test_probability,
        float(threshold),
        sample_weight=None,
        extra={
            "evaluation_n": len(test_idx),
            "training_n": len(train_idx),
            "sample_weight": "none",
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "cv_roc_auc_sd": float(scores["test_roc_auc"].std(ddof=1)),
            "cv_pr_auc_mean": float(scores["test_pr_auc"].mean()),
            "cv_pr_auc_sd": float(scores["test_pr_auc"].std(ddof=1)),
            "selector_fit_n": audit["selector_fit_n"],
            "pre_split_n": len(frame),
            "selected_feature_count": audit["selected_feature_count"],
            "feature_selection_leak": True,
            "smote_applied_before_cv": smote_applied,
            "warning": (
                "naive_non_deployable_bias_diagnostic_only"
            ),
        },
    )
    cv_rows = pd.DataFrame(
        {
            "target": target,
            "methodology": label,
            "fold": np.arange(1, len(scores["test_roc_auc"]) + 1),
            "naive_cv_roc_auc": scores["test_roc_auc"],
            "naive_cv_pr_auc": scores["test_pr_auc"],
            "naive_feature_selection_leak": True,
            "naive_smote_before_cv": smote_applied,
        }
    )
    predictions = frame.iloc[test_idx][["participant_id", "survey_cycle", "analysis_outcome"]].copy()
    predictions["target"] = target
    predictions["methodology"] = label
    predictions["probability"] = test_probability
    predictions["threshold"] = float(threshold)
    predictions["prediction"] = (test_probability >= threshold).astype("int8")
    predictions["analysis_weight"] = 1.0
    return pd.DataFrame([metrics]), cv_rows, predictions


def _summarize_cells(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    grouping = ["target", "methodology", "evaluation_design"]
    numeric = [
        "roc_auc",
        "pr_auc",
        "sensitivity",
        "specificity",
        "ppv",
        "f1",
        "calibration_slope",
        "calibration_intercept",
        "brier_score",
        "threshold",
    ]
    return (
        cell_metrics.groupby(grouping, as_index=False)
        .agg(
            **{metric: (metric, "mean") for metric in numeric},
            evaluations=("target", "size"),
            evaluation_n=("evaluation_n", "sum"),
        )
        .sort_values(["methodology", "target"])
    )


def _decomposition(summary: pd.DataFrame) -> pd.DataFrame:
    primary = summary.loc[summary["methodology"].isin(["rigorous", "naive"])].copy()
    rows = []
    for metric in ("roc_auc", "pr_auc", "f1"):
        values = {
            (row.target, row.methodology): float(getattr(row, metric))
            for row in primary.itertuples(index=False)
        }
        baseline = values.get(("osteoporosis", "rigorous"), np.nan)
        target_effect = values.get(("low_bmd", "rigorous"), np.nan) - baseline
        methodology_effect = values.get(("osteoporosis", "naive"), np.nan) - baseline
        combined_effect = values.get(("low_bmd", "naive"), np.nan) - baseline
        residual = combined_effect - target_effect - methodology_effect
        rows.append(
            {
                "metric": metric,
                "target_effect": target_effect,
                "methodology_effect": methodology_effect,
                "combined_effect": combined_effect,
                "interaction_or_residual": residual,
                "target_fraction_of_combined_gap": (
                    target_effect / combined_effect if combined_effect else np.nan
                ),
                "methodology_fraction_of_combined_gap": (
                    methodology_effect / combined_effect if combined_effect else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    def render(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    if frame.empty:
        return "_No rows._"
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _gap_sentence(decomposition: pd.DataFrame, metric: str) -> str:
    row = decomposition.loc[decomposition["metric"].eq(metric)].iloc[0]
    label = "AUC" if metric == "roc_auc" else "PR-AUC"
    return (
        f"Of the {row['combined_effect']:+.4f}-point typical-paper {label} gap, "
        f"{row['target_effect']:+.4f} points come from the easier low-BMD target and "
        f"{row['methodology_effect']:+.4f} points come from unweighted/leaky methodology; "
        f"the residual interaction is {row['interaction_or_residual']:+.4f}."
    )


def _write_results_report(
    path: Path,
    summary: pd.DataFrame,
    decomposition: pd.DataFrame,
    temporal_metrics: pd.DataFrame,
) -> None:
    lines = [
        "# Target-definition and methodology sensitivity results",
        "",
        "**The naive arm is a non-deployable bias diagnostic only. It must never be used "
        "for deployment, threshold selection, or production promotion.**",
        "",
        "This additive experiment compares the unchanged femoral-neck osteoporosis target "
        "with a low-BMD target, and compares the rigorous pipeline with a "
        "deliberately unweighted, leaky published-paper-style arm.",
        "",
        "## Results",
        "",
        _markdown_table(summary),
        "",
        "## Decomposition",
        "",
        _markdown_table(decomposition),
        "",
        _gap_sentence(decomposition, "roc_auc"),
        "",
        _gap_sentence(decomposition, "pr_auc"),
        "",
        "## Temporal rigorous holdout",
        "",
        _markdown_table(
            temporal_metrics.loc[
                temporal_metrics["evaluation"].isin(
                    ["sensitivity_oriented", "f1_optimized"]
                ),
                [
                    "target",
                    "methodology",
                    "model",
                    "evaluation",
                    "evaluation_scope",
                    "roc_auc",
                    "pr_auc",
                    "sensitivity",
                    "specificity",
                    "ppv",
                    "f1",
                    "calibration_slope",
                    "calibration_intercept",
                    "brier_score",
                ],
            ]
        ),
        "",
        "## Decision",
        "",
        "Diagnostic only, not eligible for production promotion.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_spec(path: Path, cutoff: float) -> None:
    lines = [
        "# Target-definition experiment specification",
        "",
        "This specification is additive and does not change the deployed target, model, "
        "thresholds, or production reports.",
        "",
        "| Field | Threshold | Source | Use |",
        "|---|---|---|---|",
        "| `osteoporosis` | femoral-neck T-score <= -2.5 | WHO international reference standard as summarized in ISCD 2023 Official Positions | Existing deployed outcome, reused unchanged |",
        f"| `low_bmd` | femoral-neck T-score < {cutoff:g} | WHO densitometric classification for low bone mass/osteopenia, with ISCD preferring the term low bone mass/low bone density | Diagnostic sensitivity target only |",
        "",
        "Cut-point sources:",
        "",
        "- ISCD 2023 Official Positions: https://iscd.org/official-positions-2023/",
        "- WHO criteria table via NCBI Bookshelf: https://www.ncbi.nlm.nih.gov/books/NBK543829/table/ch1.Tab1/",
        "",
        "## Promotion Rule",
        "",
        "Never promotable, diagnostic only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _output_paths(paths: ProjectPaths, prefix: str = "target_definition") -> dict[str, Path]:
    report_name = (
        "TARGET_DEFINITION_RESULTS.md"
        if prefix == "target_definition"
        else f"{prefix.upper()}_RESULTS.md"
    )
    spec_name = (
        "TARGET_DEFINITION_SPEC.md"
        if prefix == "target_definition"
        else f"{prefix.upper()}_SPEC.md"
    )
    return {
        "cell_metrics": paths.tables / f"{prefix}_cell_metrics.csv",
        "rigorous_outer_metrics": paths.tables / f"{prefix}_rigorous_outer_metrics.csv",
        "rigorous_temporal_metrics": paths.tables / f"{prefix}_rigorous_temporal_metrics.csv",
        "rigorous_metric_intervals": paths.tables / f"{prefix}_rigorous_metric_intervals.csv",
        "naive_metrics": paths.tables / f"{prefix}_naive_metrics.csv",
        "naive_cv_folds": paths.tables / f"{prefix}_naive_cv_folds.csv",
        "decomposition": paths.tables / f"{prefix}_decomposition.csv",
        "predictions": paths.tables / f"{prefix}_predictions.csv.gz",
        "manifest": paths.tables / f"{prefix}_manifest.json",
        "report": paths.root / "reports" / report_name,
        "spec": paths.root / "reports" / spec_name,
    }


def run_target_definition_experiment(
    root: Path,
    overwrite: bool = False,
    output_prefix: str = "target_definition",
) -> dict:
    paths = ProjectPaths.from_root(root)
    paths.ensure()
    output_paths = _output_paths(paths, output_prefix)
    existing = [
        str(path)
        for path in [
            *paths.tables.glob(f"{output_prefix}_*"),
            output_paths["report"],
            output_paths["spec"],
        ]
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Target-definition outputs already exist; pass --overwrite to replace only "
            f"this additive namespace: {existing}"
        )

    model_config = load_yaml(root / "configs" / "model_config.yaml")
    variables = load_yaml(root / "configs" / "variables.yaml")
    experiment_config = load_yaml(root / "configs" / "target_definition_experiment.yaml")
    cutoff = float(experiment_config.get("low_bmd_t_score_cutoff", -1.0))
    frame = pd.read_csv(
        paths.processed / "classification_cohort.csv.gz", low_memory=False
    ).dropna(subset=["osteoporosis", "femoral_neck_t_score"])
    e12_features = _experiment_feature_sets(frame, variables)["E12_e8_plus_medical_mobility"]
    categorical = set(variables["categorical_features"])

    rigorous_metrics = []
    rigorous_cells = []
    temporal_metrics = []
    temporal_intervals = []
    naive_metrics = []
    naive_cv = []
    predictions = []
    for target in ("osteoporosis", "low_bmd"):
        target_frame = _targeted_frame(frame, target, cutoff)
        outer, cells, pred = _run_rigorous_loco(
            target_frame,
            target,
            e12_features,
            categorical,
            model_config,
            experiment_config,
        )
        rigorous_metrics.append(outer)
        rigorous_cells.append(cells)
        predictions.append(pred)
        temporal, intervals, temporal_pred = _temporal_rigorous_metrics(
            target_frame,
            target,
            e12_features,
            categorical,
            model_config,
            experiment_config,
        )
        temporal_metrics.append(temporal)
        temporal_intervals.append(intervals)
        predictions.append(temporal_pred)
        naive, cv, naive_pred = _run_naive(
            target_frame,
            target,
            e12_features,
            categorical,
            experiment_config,
            use_smote=False,
        )
        naive_metrics.append(naive)
        naive_cv.append(cv)
        predictions.append(naive_pred)
        if (
            bool(experiment_config.get("include_smote_naive_subarm", False))
            and importlib.util.find_spec("imblearn") is not None
        ):
            smote_naive, smote_cv, smote_pred = _run_naive(
                target_frame,
                target,
                e12_features,
                categorical,
                experiment_config,
                use_smote=True,
            )
            naive_metrics.append(smote_naive)
            naive_cv.append(smote_cv)
            predictions.append(smote_pred)
        print(f"naive_completed target={target}", flush=True)

    rigorous_outer = pd.concat(rigorous_metrics, ignore_index=True)
    cell_metrics = pd.concat([*rigorous_cells, *naive_metrics], ignore_index=True)
    summary = _summarize_cells(cell_metrics)
    temporal_frame = pd.concat(temporal_metrics, ignore_index=True)
    decomposition = _decomposition(summary)
    rigorous_outer.to_csv(output_paths["rigorous_outer_metrics"], index=False)
    temporal_frame.to_csv(output_paths["rigorous_temporal_metrics"], index=False)
    pd.concat(temporal_intervals, ignore_index=True).to_csv(
        output_paths["rigorous_metric_intervals"], index=False
    )
    pd.concat(naive_metrics, ignore_index=True).to_csv(output_paths["naive_metrics"], index=False)
    pd.concat(naive_cv, ignore_index=True).to_csv(output_paths["naive_cv_folds"], index=False)
    summary.to_csv(output_paths["cell_metrics"], index=False)
    decomposition.to_csv(output_paths["decomposition"], index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output_paths["predictions"], index=False, compression="gzip"
    )
    _write_results_report(output_paths["report"], summary, decomposition, temporal_frame)
    _write_spec(output_paths["spec"], cutoff)
    manifest = {
        "experiment": "target-definition-experiment",
        "production_outputs_modified": False,
        "target_definitions": {
            "osteoporosis": "femoral-neck T-score <= -2.5",
            "low_bmd": f"femoral-neck T-score < {cutoff:g}",
        },
        "feature_panel": "E12_e8_plus_medical_mobility",
        "methodology_arms": ["rigorous", "naive"],
        "naive_arm_warning": (
            "non-deployable bias diagnostic only; never use for deployment, "
            "threshold selection, or production promotion"
        ),
        "output_files": {
            key: str(value.relative_to(root)) for key, value in output_paths.items()
        },
    }
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "decision": "diagnostic only, not eligible for production promotion",
        "summary": summary.to_dict(orient="records"),
        "decomposition": decomposition.to_dict(orient="records"),
        "outputs": {key: str(value) for key, value in output_paths.items()},
    }
