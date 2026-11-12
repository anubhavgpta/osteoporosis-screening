from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectPaths, load_yaml
from .curve_bending import _experiment_feature_sets
from .modeling import run_classification


TOTAL_FEMUR_REFERENCE = {
    "mean_bmd": 0.942,
    "sd_bmd": 0.122,
    "source": (
        "NHANES III young-adult White female total-femur/total-hip reference"
    ),
    "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2966549/",
}


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


def _derive_site_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["total_femur_t_score"] = (
        result["total_femur_bmd"] - TOTAL_FEMUR_REFERENCE["mean_bmd"]
    ) / TOTAL_FEMUR_REFERENCE["sd_bmd"]
    result["minimum_hip_t_score"] = result[
        ["femoral_neck_t_score", "total_femur_t_score"]
    ].min(axis=1, skipna=True)

    result["low_bmd_femoral_neck"] = (
        result["femoral_neck_t_score"].le(-1.0).astype("Int64")
    )
    result["osteoporosis_femoral_neck"] = (
        result["femoral_neck_t_score"].le(-2.5).astype("Int64")
    )
    result["low_bmd_any_hip"] = result["minimum_hip_t_score"].le(-1.0).astype("Int64")
    result["osteoporosis_any_hip"] = (
        result["minimum_hip_t_score"].le(-2.5).astype("Int64")
    )
    missing_neck = result["femoral_neck_t_score"].isna()
    missing_any = result["minimum_hip_t_score"].isna()
    result.loc[
        missing_neck, ["low_bmd_femoral_neck", "osteoporosis_femoral_neck"]
    ] = pd.NA
    result.loc[missing_any, ["low_bmd_any_hip", "osteoporosis_any_hip"]] = pd.NA
    return result


def _feature_panels(
    frame: pd.DataFrame, variables: dict
) -> dict[str, list[str]]:
    e12 = _experiment_feature_sets(frame, variables)["E12_e8_plus_medical_mobility"]
    return {
        "E12_baseline": list(dict.fromkeys(e12)),
    }


def _target_definitions() -> dict[str, str]:
    return {
        "low_bmd_femoral_neck": "femoral-neck T-score <= -1.0",
        "low_bmd_any_hip": "femoral-neck OR total-femur T-score <= -1.0",
        "osteoporosis_femoral_neck": "femoral-neck T-score <= -2.5",
        "osteoporosis_any_hip": "femoral-neck OR total-femur T-score <= -2.5",
    }


def _collect_confusion(predictions: pd.DataFrame, target: str) -> dict:
    y = predictions[target].astype(int).to_numpy()
    probability = predictions["probability"].to_numpy()
    threshold = float(
        predictions.attrs.get("f1_threshold", predictions["probability"].median())
    )
    predicted = probability >= threshold
    weight = predictions["analysis_weight"].to_numpy()
    raw = {
        "raw_tn": int(((y == 0) & ~predicted).sum()),
        "raw_fp": int(((y == 0) & predicted).sum()),
        "raw_fn": int(((y == 1) & ~predicted).sum()),
        "raw_tp": int(((y == 1) & predicted).sum()),
    }
    weighted = {
        "weighted_tn": float(weight[(y == 0) & ~predicted].sum()),
        "weighted_fp": float(weight[(y == 0) & predicted].sum()),
        "weighted_fn": float(weight[(y == 1) & ~predicted].sum()),
        "weighted_tp": float(weight[(y == 1) & predicted].sum()),
    }
    return {**raw, **weighted}


def _write_report(
    path: Path,
    comparison: pd.DataFrame,
    best_rows: pd.DataFrame,
    confusion: pd.DataFrame,
) -> None:
    lines = [
        "# Site-inclusive target experiment",
        "",
        "This additive experiment does not modify production models or existing "
        "pipeline outputs. It tests whether defining outcomes from femoral neck OR "
        "total femur improves 2017-2018 temporal validation.",
        "",
        "## Best F1 operating points",
        "",
        _markdown_table(best_rows),
        "",
        "## All temporal comparisons",
        "",
        _markdown_table(comparison),
        "",
        "## F1 confusion matrices",
        "",
        _markdown_table(confusion),
        "",
        "## Interpretation",
        "",
        "Use these rows as a diagnostic comparison only. The total-femur T-score uses "
        "a fixed NHANES III young-adult White female reference, so the reference "
        "choice should be cited and reviewed before any publication claim.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_site_target_experiment(root: Path, overwrite: bool = False) -> dict:
    paths = ProjectPaths.from_root(root)
    paths.ensure()
    output_paths = {
        "metrics": paths.tables / "site_target_temporal_metrics.csv",
        "model_comparison": paths.tables / "site_target_model_comparison.csv",
        "thresholds": paths.tables / "site_target_thresholds.csv",
        "confusion": paths.tables / "site_target_confusion_matrices.csv",
        "predictions": paths.tables / "site_target_predictions.csv.gz",
        "manifest": paths.tables / "site_target_manifest.json",
        "report": paths.root / "reports" / "SITE_TARGET_RESULTS.md",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Site-target outputs already exist; pass --overwrite to replace only "
            f"this additive namespace: {existing}"
        )

    model_config = load_yaml(root / "configs" / "model_config.yaml")
    variables = load_yaml(root / "configs" / "variables.yaml")
    frame = pd.read_csv(paths.processed / "classification_cohort.csv.gz", low_memory=False)
    frame = _derive_site_targets(frame)
    panels = _feature_panels(frame, variables)
    categorical = set(variables["categorical_features"])
    temporal_cycle = str(model_config.get("temporal_evaluation_cycle", "2017-2018"))

    metric_rows = []
    comparison_rows = []
    threshold_rows = []
    prediction_rows = []
    confusion_rows = []
    for target, definition in _target_definitions().items():
        target_frame = frame.dropna(subset=[target]).copy()
        for panel_name, features in panels.items():
            result = run_classification(
                target_frame,
                features,
                categorical,
                model_config,
                add_clusters=False,
                target=target,
                sample_weight_column="mec_weight",
                evaluation_cycle=temporal_cycle,
                selection_metric=str(
                    model_config.get("classification_selection_metric", "roc_auc")
                ),
            )
            metrics = result.metrics.copy()
            metrics.insert(0, "target_definition", definition)
            metrics.insert(0, "feature_panel", panel_name)
            metric_rows.append(metrics)

            comparison = result.comparison.copy()
            comparison.insert(0, "target", target)
            comparison.insert(1, "target_definition", definition)
            comparison.insert(2, "feature_panel", panel_name)
            comparison_rows.append(comparison)

            thresholds = result.thresholds.copy()
            thresholds.insert(0, "target", target)
            thresholds.insert(1, "target_definition", definition)
            thresholds.insert(2, "feature_panel", panel_name)
            threshold_rows.append(thresholds)

            f1_threshold = float(
                result.metrics.loc[
                    result.metrics["evaluation"].eq("f1_optimized"), "threshold"
                ].iloc[0]
            )
            predictions = result.predictions.copy()
            predictions["target"] = target
            predictions["target_definition"] = definition
            predictions["feature_panel"] = panel_name
            predictions["f1_threshold"] = f1_threshold
            predictions["f1_prediction"] = (
                predictions["probability"] >= f1_threshold
            ).astype(int)
            prediction_rows.append(predictions)
            predictions.attrs["f1_threshold"] = f1_threshold
            confusion_rows.append(
                {
                    "target": target,
                    "target_definition": definition,
                    "feature_panel": panel_name,
                    "threshold": f1_threshold,
                    **_collect_confusion(predictions, target),
                }
            )
            print(f"completed target={target} panel={panel_name}", flush=True)

    metrics = pd.concat(metric_rows, ignore_index=True)
    model_comparison = pd.concat(comparison_rows, ignore_index=True)
    thresholds = pd.concat(threshold_rows, ignore_index=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    confusion = pd.DataFrame(confusion_rows)
    f1_rows = metrics.loc[metrics["evaluation"].eq("f1_optimized")].copy()
    best_rows = (
        f1_rows.sort_values(["target", "f1", "pr_auc"], ascending=[True, False, False])
        .groupby("target", as_index=False)
        .head(1)
        .loc[
            :,
            [
                "target",
                "target_definition",
                "feature_panel",
                "model",
                "roc_auc",
                "pr_auc",
                "accuracy",
                "sensitivity",
                "specificity",
                "ppv",
                "f1",
                "threshold",
                "evaluation_n",
            ],
        ]
    )
    comparison_table = f1_rows.loc[
        :,
        [
            "target",
            "target_definition",
            "feature_panel",
            "model",
            "roc_auc",
            "pr_auc",
            "accuracy",
            "sensitivity",
            "specificity",
            "ppv",
            "f1",
            "threshold",
        ],
    ].sort_values(["target", "feature_panel"])

    metrics.to_csv(output_paths["metrics"], index=False)
    model_comparison.to_csv(output_paths["model_comparison"], index=False)
    thresholds.to_csv(output_paths["thresholds"], index=False)
    confusion.to_csv(output_paths["confusion"], index=False)
    predictions.to_csv(output_paths["predictions"], index=False, compression="gzip")
    _write_report(output_paths["report"], comparison_table, best_rows, confusion)
    manifest = {
        "experiment": "site-inclusive-target-temporal-comparison",
        "production_outputs_modified": False,
        "evaluation": f"held_out_cycle_{temporal_cycle}",
        "target_definitions": _target_definitions(),
        "total_femur_reference": TOTAL_FEMUR_REFERENCE,
        "feature_panels": {key: value for key, value in panels.items()},
        "skipped_default_panels": {
            "E12_plus_extended_anthropometric": "omitted from the default quick pass",
            "E12_plus_routine_labs": "omitted from the default quick pass",
        },
        "output_files": {
            key: str(value.relative_to(root)) for key, value in output_paths.items()
        },
    }
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "best_rows": best_rows.to_dict(orient="records"),
        "outputs": {key: str(value) for key, value in output_paths.items()},
    }
