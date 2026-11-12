from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectPaths


TRACKED_PREFIXES = ["feature_combination", "low_bmd_feature_combination"]
LOW_BMD_PIPELINE_COMPARISON_PREFIX = "low_bmd_pipeline_comparison"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _preferred_existing_prefix(paths: ProjectPaths, preferred: str, fallback: str) -> str:
    if (paths.tables / f"{preferred}_manifest.json").exists():
        return preferred
    return fallback


def _metric_rows_from_target_definition(paths: ProjectPaths) -> list[dict]:
    rows = []
    baseline = _read_csv(paths.tables / "target_definition_cell_metrics.csv")
    if not baseline.empty:
        for row in baseline.itertuples(index=False):
            rows.append(
                {
                    "source": "target_definition",
                    "target": row.target,
                    "comparison_anchor": "baseline",
                    "scope": row.evaluation_design,
                    "candidate": f"{row.methodology}_{row.evaluation_design}",
                    "roc_auc": row.roc_auc,
                    "pr_auc": row.pr_auc,
                    "sensitivity": row.sensitivity,
                    "specificity": row.specificity,
                    "ppv": row.ppv,
                    "f1": row.f1 if hasattr(row, "f1") else np.nan,
                }
            )
    temporal = _read_csv(paths.tables / "target_definition_rigorous_temporal_metrics.csv")
    if not temporal.empty:
        selected = temporal.loc[
            temporal["evaluation"].isin(["sensitivity_oriented", "f1_optimized"])
        ]
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "source": "target_definition_temporal",
                    "target": row.target,
                    "comparison_anchor": "temporal_baseline",
                    "scope": row.evaluation_scope,
                    "candidate": f"{row.model}_{row.evaluation}",
                    "roc_auc": row.roc_auc,
                    "pr_auc": row.pr_auc,
                    "sensitivity": row.sensitivity,
                    "specificity": row.specificity,
                    "ppv": row.ppv,
                    "f1": row.f1 if hasattr(row, "f1") else np.nan,
                }
            )
    return rows


def _baseline_lookup(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        if (
            row["source"] == "target_definition"
            and row["comparison_anchor"] == "baseline"
            and str(row["candidate"]).startswith("rigorous_")
        ):
            result[str(row["target"])] = row
    return result


def _metric_rows_from_feature_combinations(
    paths: ProjectPaths, baselines: dict[str, dict]
) -> list[dict]:
    rows = []
    for prefix in TRACKED_PREFIXES:
        summary_path = paths.tables / f"{prefix}_panel_summary.csv"
        summary = _read_csv(summary_path)
        if summary.empty:
            continue
        manifest = _read_manifest(paths.tables / f"{prefix}_manifest.json")
        target = str(
            manifest.get(
                "target",
                "low_bmd" if prefix.startswith("low_bmd") else "osteoporosis",
            )
        )
        availability = _read_csv(paths.tables / f"{prefix}_availability.csv")
        if not availability.empty and "eligible_all_production_cycles" in availability:
            panel_status = (
                availability.groupby("panel", as_index=False)
                .agg(eligible_all_production_cycles=("eligible_all_production_cycles", "all"))
            )
            for status in panel_status.itertuples(index=False):
                if not bool(status.eligible_all_production_cycles):
                    rows.append(
                        {
                            "source": prefix,
                            "target": target,
                            "comparison_anchor": "availability_skipped",
                            "scope": "cycle_coverage_audit",
                            "candidate": status.panel,
                            "skip_reason": "did_not_meet_minimum_observed_fraction_in_every_cycle",
                        }
                    )
        baseline = baselines.get(target, {})
        for row in summary.itertuples(index=False):
            record = {
                "source": prefix,
                "target": target,
                "comparison_anchor": "feature_panel_vs_rigorous_baseline",
                "scope": "leave_one_production_cycle_out",
                "candidate": row.feature_set,
                "roc_auc": row.mean_roc_auc,
                "pr_auc": row.mean_pr_auc,
                "sensitivity": row.mean_point_sensitivity,
                "specificity": row.mean_point_specificity,
                "ppv": row.mean_point_ppv,
                "f1": row.mean_point_f1 if hasattr(row, "mean_point_f1") else np.nan,
                "minimum_sensitivity": row.minimum_point_sensitivity,
                "selected_models": row.selected_models,
            }
            for metric in ("roc_auc", "pr_auc", "sensitivity", "specificity", "ppv", "f1"):
                record[f"delta_{metric}_vs_baseline"] = (
                    float(record[metric]) - float(baseline[metric])
                    if metric in baseline and pd.notna(record[metric])
                    else np.nan
                )
            rows.append(record)
    return rows


def _markdown_table(frame: pd.DataFrame) -> str:
    def render(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    if frame.empty:
        return "_No rows available yet._"
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _read_csv_gz(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _comparison_output_paths(paths: ProjectPaths) -> dict[str, Path]:
    prefix = LOW_BMD_PIPELINE_COMPARISON_PREFIX
    return {
        "summary": paths.tables / f"{prefix}_summary.csv",
        "low_bmd_metrics": paths.tables / f"{prefix}_low_bmd_metrics.csv",
        "thresholds": paths.tables / f"{prefix}_thresholds.csv",
        "calibration": paths.tables / f"{prefix}_calibration.csv",
        "predictions": paths.tables / f"{prefix}_predictions.csv.gz",
        "manifest": paths.tables / f"{prefix}_manifest.json",
        "report": paths.root / "reports" / "LOW_BMD_PIPELINE_COMPARISON_RESULTS.md",
    }


def _target_definition_comparison_rows(
    paths: ProjectPaths, prefix: str = "target_definition"
) -> list[dict]:
    rows = []
    cells_filename = f"{prefix}_cell_metrics.csv"
    cells = _read_csv(paths.tables / cells_filename)
    if not cells.empty:
        for row in cells.itertuples(index=False):
            rows.append(
                {
                    "pipeline_family": "E12_baseline",
                    "validation_design": row.evaluation_design,
                    "pipeline": f"{row.methodology}_{row.evaluation_design}",
                    "source_table": cells_filename,
                    "target": row.target,
                    "roc_auc": row.roc_auc,
                    "pr_auc": row.pr_auc,
                    "sensitivity": row.sensitivity,
                    "specificity": row.specificity,
                    "ppv": row.ppv,
                    "f1": row.f1 if hasattr(row, "f1") else np.nan,
                    "threshold": row.threshold,
                    "calibration_slope": (
                        row.calibration_slope if hasattr(row, "calibration_slope") else np.nan
                    ),
                    "calibration_intercept": (
                        row.calibration_intercept
                        if hasattr(row, "calibration_intercept")
                        else np.nan
                    ),
                    "brier_score": row.brier_score if hasattr(row, "brier_score") else np.nan,
                }
            )
    temporal_filename = f"{prefix}_rigorous_temporal_metrics.csv"
    temporal = _read_csv(paths.tables / temporal_filename)
    if not temporal.empty:
        selected = temporal.loc[
            temporal["evaluation"].isin(["sensitivity_oriented", "f1_optimized"])
        ]
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "pipeline_family": "E12_baseline",
                    "validation_design": "2017_2018_temporal",
                    "pipeline": f"{row.model}_{row.evaluation}",
                    "source_table": temporal_filename,
                    "target": row.target,
                    "roc_auc": row.roc_auc,
                    "pr_auc": row.pr_auc,
                    "sensitivity": row.sensitivity,
                    "specificity": row.specificity,
                    "ppv": row.ppv,
                    "f1": row.f1 if hasattr(row, "f1") else np.nan,
                    "threshold": row.threshold if hasattr(row, "threshold") else np.nan,
                    "calibration_slope": (
                        row.calibration_slope if hasattr(row, "calibration_slope") else np.nan
                    ),
                    "calibration_intercept": (
                        row.calibration_intercept
                        if hasattr(row, "calibration_intercept")
                        else np.nan
                    ),
                    "brier_score": row.brier_score if hasattr(row, "brier_score") else np.nan,
                }
            )
    return rows


def _feature_combination_comparison_rows(paths: ProjectPaths) -> list[dict]:
    rows = []
    low_bmd_prefix = _preferred_existing_prefix(
        paths,
        "low_bmd_pipeline_comparison_feature_combination",
        "low_bmd_feature_combination",
    )
    for prefix, target in (
        ("feature_combination", "osteoporosis"),
        (low_bmd_prefix, "low_bmd"),
    ):
        manifest = _read_manifest(paths.tables / f"{prefix}_manifest.json")
        target = str(manifest.get("target", target))
        panel_summary = _read_csv(paths.tables / f"{prefix}_panel_summary.csv")
        if not panel_summary.empty:
            for row in panel_summary.itertuples(index=False):
                rows.append(
                    {
                        "pipeline_family": "feature_combination_panel",
                        "validation_design": "leave_one_production_cycle_out",
                        "pipeline": row.feature_set,
                        "source_table": f"{prefix}_panel_summary.csv",
                        "target": target,
                        "roc_auc": row.mean_roc_auc,
                        "pr_auc": row.mean_pr_auc,
                        "sensitivity": row.mean_point_sensitivity,
                        "specificity": row.mean_point_specificity,
                        "ppv": row.mean_point_ppv,
                        "f1": row.mean_point_f1 if hasattr(row, "mean_point_f1") else np.nan,
                        "brier_score": (
                            row.mean_brier_score if hasattr(row, "mean_brier_score") else np.nan
                        ),
                        "selected_models": (
                            row.selected_models if hasattr(row, "selected_models") else ""
                        ),
                    }
                )
        candidate_summary = _read_csv(paths.tables / f"{prefix}_candidate_summary.csv")
        if not candidate_summary.empty:
            for row in candidate_summary.itertuples(index=False):
                rows.append(
                    {
                        "pipeline_family": "feature_combination_candidate",
                        "validation_design": "leave_one_production_cycle_out",
                        "pipeline": row.candidate,
                        "source_table": f"{prefix}_candidate_summary.csv",
                        "target": target,
                        "roc_auc": row.mean_roc_auc,
                        "pr_auc": row.mean_pr_auc,
                        "sensitivity": row.mean_point_sensitivity,
                        "specificity": row.mean_point_specificity,
                        "ppv": row.mean_point_ppv,
                        "f1": row.mean_point_f1 if hasattr(row, "mean_point_f1") else np.nan,
                        "brier_score": (
                            row.mean_brier_score if hasattr(row, "mean_brier_score") else np.nan
                        ),
                        "selected_models": row.model if hasattr(row, "model") else "",
                    }
                )
    return rows


def _comparison_summary(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    keys = ["pipeline_family", "validation_design", "pipeline"]
    metrics = [
        "roc_auc",
        "pr_auc",
        "sensitivity",
        "specificity",
        "ppv",
        "f1",
        "brier_score",
    ]
    pivot = frame.pivot_table(index=keys, columns="target", values=metrics, aggfunc="first")
    pivot.columns = [f"{target}_{metric}" for metric, target in pivot.columns]
    summary = pivot.reset_index()
    targets = frame.groupby(keys)["target"].agg(lambda values: "|".join(sorted(set(values))))
    sources = frame.groupby(keys)["source_table"].agg(lambda values: "|".join(sorted(set(values))))
    summary = summary.merge(targets.rename("available_targets"), on=keys, how="left")
    summary = summary.merge(sources.rename("source_tables"), on=keys, how="left")
    summary["osteoporosis_available"] = summary["available_targets"].str.contains("osteoporosis")
    summary["low_bmd_available"] = summary["available_targets"].str.contains("low_bmd")
    summary["matched_targets"] = summary["osteoporosis_available"] & summary["low_bmd_available"]
    for metric in metrics:
        low = f"low_bmd_{metric}"
        osteo = f"osteoporosis_{metric}"
        if low in summary and osteo in summary:
            summary[f"delta_{metric}_low_bmd_minus_osteoporosis"] = (
                summary[low] - summary[osteo]
            )
    return summary.sort_values(
        ["matched_targets", "pipeline_family", "validation_design", "pipeline"],
        ascending=[False, True, True, True],
    )


def _add_source(frame: pd.DataFrame, source_table: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    result.insert(0, "source_table", source_table)
    return result


def _low_bmd_detail_tables(paths: ProjectPaths) -> dict[str, pd.DataFrame]:
    detail_frames = []
    thresholds = []
    calibration = []
    target_prefix = _preferred_existing_prefix(
        paths,
        "low_bmd_pipeline_comparison_target_definition",
        "target_definition",
    )
    feature_prefix = _preferred_existing_prefix(
        paths,
        "low_bmd_pipeline_comparison_feature_combination",
        "low_bmd_feature_combination",
    )
    for filename in (
        f"{target_prefix}_cell_metrics.csv",
        f"{target_prefix}_rigorous_outer_metrics.csv",
        f"{target_prefix}_rigorous_temporal_metrics.csv",
        f"{feature_prefix}_outer_metrics.csv",
        f"{feature_prefix}_panel_summary.csv",
        f"{feature_prefix}_candidate_summary.csv",
        f"{feature_prefix}_multi_metric_candidates.csv",
    ):
        frame = _read_csv(paths.tables / filename)
        if frame.empty:
            continue
        if "target" in frame:
            frame = frame.loc[frame["target"].eq("low_bmd")].copy()
        elif filename.startswith("low_bmd_"):
            frame = frame.copy()
            frame.insert(0, "target", "low_bmd")
        if frame.empty:
            continue
        sourced = _add_source(frame, filename)
        detail_frames.append(sourced)
        threshold_columns = [
            column
            for column in frame.columns
            if column == "threshold" or column.endswith("_threshold")
        ]
        if threshold_columns:
            id_columns = [
                column
                for column in ("target", "methodology", "outer_cycle", "candidate", "feature_set", "model")
                if column in frame
            ]
            thresholds.append(_add_source(frame[[*id_columns, *threshold_columns]], filename))
        calibration_columns = [
            column
            for column in (
                "calibration_slope",
                "calibration_intercept",
                "brier_score",
                "log_loss",
                "mean_brier_score",
                "mean_log_loss",
            )
            if column in frame
        ]
        if calibration_columns:
            id_columns = [
                column
                for column in ("target", "methodology", "evaluation", "outer_cycle", "candidate", "feature_set", "model")
                if column in frame
            ]
            calibration.append(_add_source(frame[[*id_columns, *calibration_columns]], filename))
    return {
        "low_bmd_metrics": (
            pd.concat(detail_frames, ignore_index=True, sort=False)
            if detail_frames
            else pd.DataFrame()
        ),
        "thresholds": (
            pd.concat(thresholds, ignore_index=True, sort=False)
            if thresholds
            else pd.DataFrame()
        ),
        "calibration": (
            pd.concat(calibration, ignore_index=True, sort=False)
            if calibration
            else pd.DataFrame()
        ),
    }


def _low_bmd_predictions(paths: ProjectPaths) -> pd.DataFrame:
    frames = []
    target_prefix = _preferred_existing_prefix(
        paths,
        "low_bmd_pipeline_comparison_target_definition",
        "target_definition",
    )
    feature_prefix = _preferred_existing_prefix(
        paths,
        "low_bmd_pipeline_comparison_feature_combination",
        "low_bmd_feature_combination",
    )
    target_predictions = f"{target_prefix}_predictions.csv.gz"
    target_definition = _read_csv_gz(paths.tables / target_predictions)
    if not target_definition.empty and "target" in target_definition:
        frame = target_definition.loc[target_definition["target"].eq("low_bmd")].copy()
        if not frame.empty:
            if "analysis_outcome" in frame:
                frame = frame.rename(columns={"analysis_outcome": "outcome"})
            if "screening_prediction" in frame and "prediction" not in frame:
                frame["prediction"] = frame["screening_prediction"]
            frame["source_table"] = target_predictions
            frame["pipeline"] = frame.get("methodology", "target_definition")
            frames.append(frame)
    feature_predictions = f"{feature_prefix}_selected_predictions.csv.gz"
    feature_combination = _read_csv_gz(paths.tables / feature_predictions)
    if not feature_combination.empty:
        frame = feature_combination.copy()
        frame["source_table"] = feature_predictions
        frame["pipeline"] = frame["candidate"] if "candidate" in frame else "feature_combination"
        if "point_threshold" in frame:
            frame["threshold"] = frame["point_threshold"]
        if "point_prediction" in frame:
            frame["prediction"] = frame["point_prediction"]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _comparison_report(
    path: Path,
    summary: pd.DataFrame,
    low_bmd_metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    matched = summary.loc[summary["matched_targets"]] if "matched_targets" in summary else summary
    low_only = (
        summary.loc[summary["low_bmd_available"] & ~summary["osteoporosis_available"]]
        if {"low_bmd_available", "osteoporosis_available"}.issubset(summary.columns)
        else pd.DataFrame()
    )
    osteo_only = (
        summary.loc[summary["osteoporosis_available"] & ~summary["low_bmd_available"]]
        if {"low_bmd_available", "osteoporosis_available"}.issubset(summary.columns)
        else pd.DataFrame()
    )
    compact_columns = [
        column
        for column in (
            "pipeline_family",
            "validation_design",
            "pipeline",
            "osteoporosis_pr_auc",
            "low_bmd_pr_auc",
            "delta_pr_auc_low_bmd_minus_osteoporosis",
            "osteoporosis_roc_auc",
            "low_bmd_roc_auc",
            "delta_roc_auc_low_bmd_minus_osteoporosis",
        )
        if column in matched
    ]
    lines = [
        "# Low-BMD pipeline comparison",
        "",
        "These low-BMD results are sensitivity and target-definition comparisons. "
        "They do not replace the production osteoporosis model, thresholds, reports, "
        "or saved artifacts.",
        "",
        "## Successfully mirrored pipelines",
        "",
        _markdown_table(
            matched[compact_columns].sort_values(
                ["pipeline_family", "validation_design", "pipeline"]
            )
            if compact_columns and not matched.empty
            else pd.DataFrame()
        ),
        "",
        "## Low-BMD-only detail",
        "",
        f"- Detailed low-BMD metric rows: {len(low_bmd_metrics)}",
        f"- Threshold rows: {len(thresholds)}",
        f"- Calibration rows: {len(calibration)}",
        "",
        "## Not mirrored or target-specific",
        "",
        "Some pipelines are intentionally target-specific or require frozen external "
        "predictions that do not exist for low BMD. Clinical-score baselines and "
        "KNHANES recalibration remain osteoporosis-only unless corresponding low-BMD "
        "external predictions are generated with the same leakage-safe validation design.",
        "",
        "Osteoporosis-only comparison rows:",
        "",
        _markdown_table(
            osteo_only[["pipeline_family", "validation_design", "pipeline"]]
            if not osteo_only.empty
            else pd.DataFrame()
        ),
        "",
        "Low-BMD-only comparison rows:",
        "",
        _markdown_table(
            low_only[["pipeline_family", "validation_design", "pipeline"]]
            if not low_only.empty
            else pd.DataFrame()
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_low_bmd_pipeline_comparison(root: Path, overwrite: bool = False) -> dict:
    paths = ProjectPaths.from_root(root)
    paths.ensure()
    output_paths = _comparison_output_paths(paths)
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Low-BMD pipeline-comparison outputs already exist; pass --overwrite "
            f"to replace only this additive namespace: {existing}"
        )

    target_prefix = _preferred_existing_prefix(
        paths,
        "low_bmd_pipeline_comparison_target_definition",
        "target_definition",
    )
    comparison_rows = [
        *_target_definition_comparison_rows(paths, target_prefix),
        *_feature_combination_comparison_rows(paths),
    ]
    summary = _comparison_summary(comparison_rows)
    details = _low_bmd_detail_tables(paths)
    predictions = _low_bmd_predictions(paths)

    summary.to_csv(output_paths["summary"], index=False)
    details["low_bmd_metrics"].to_csv(output_paths["low_bmd_metrics"], index=False)
    details["thresholds"].to_csv(output_paths["thresholds"], index=False)
    details["calibration"].to_csv(output_paths["calibration"], index=False)
    predictions.to_csv(output_paths["predictions"], index=False, compression="gzip")
    _comparison_report(
        output_paths["report"],
        summary,
        details["low_bmd_metrics"],
        details["thresholds"],
        details["calibration"],
    )
    manifest = {
        "experiment": "low_bmd_pipeline_comparison",
        "production_outputs_modified": False,
        "target_definition": "low_bmd = femoral-neck T-score < -1.0",
        "purpose": (
            "sensitivity and target-definition comparison only; does not replace "
            "the production osteoporosis model"
        ),
        "source_tables": sorted(
            {
                str(row["source_table"])
                for row in comparison_rows
                if row.get("source_table")
            }
        ),
        "matched_pipeline_rows": (
            int(summary["matched_targets"].sum())
            if "matched_targets" in summary
            else 0
        ),
        "output_files": {
            key: str(value.relative_to(root)) for key, value in output_paths.items()
        },
    }
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "matched_pipeline_rows": manifest["matched_pipeline_rows"],
        "low_bmd_metric_rows": len(details["low_bmd_metrics"]),
        "prediction_rows": len(predictions),
        "outputs": {key: str(value) for key, value in output_paths.items()},
    }


def _write_tracking_report(path: Path, tracker: pd.DataFrame) -> None:
    if tracker.empty:
        path.write_text(
            "# Two-target metric tracker\n\n_No tracked result tables were found yet._\n",
            encoding="utf-8",
        )
        return
    compact_columns = [
        "source",
        "target",
        "candidate",
        "roc_auc",
        "pr_auc",
        "sensitivity",
        "specificity",
        "ppv",
        "f1",
        "delta_roc_auc_vs_baseline",
        "delta_pr_auc_vs_baseline",
        "delta_ppv_vs_baseline",
        "delta_f1_vs_baseline",
    ]
    available = [column for column in compact_columns if column in tracker.columns]
    candidate_rows = tracker.loc[
        tracker["comparison_anchor"].eq("feature_panel_vs_rigorous_baseline"),
        available,
    ].copy()
    lines = [
        "# Two-target metric tracker",
        "",
        "This report tracks additive experiments against the frozen rigorous baseline for each target.",
        "Production artifacts are not modified by this tracker.",
        "",
        "## Feature-panel deltas",
        "",
        _markdown_table(candidate_rows.sort_values(["target", "pr_auc"], ascending=[True, False])),
        "",
        "## Baseline rows",
        "",
        _markdown_table(
            tracker.loc[
                tracker["comparison_anchor"].isin(["baseline", "temporal_baseline"]),
                [
                    "source",
                    "target",
                    "scope",
                    "candidate",
                    "roc_auc",
                    "pr_auc",
                    "sensitivity",
                    "specificity",
                    "ppv",
                    "f1",
                ],
            ]
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_result_tracking(root: Path) -> dict:
    paths = ProjectPaths.from_root(root)
    paths.ensure()
    rows = _metric_rows_from_target_definition(paths)
    baselines = _baseline_lookup(rows)
    rows.extend(_metric_rows_from_feature_combinations(paths, baselines))
    tracker = pd.DataFrame(rows)
    output = paths.tables / "two_target_metric_tracker.csv"
    report = paths.root / "reports" / "TWO_TARGET_TRACKING_RESULTS.md"
    manifest = paths.tables / "two_target_metric_tracker_manifest.json"
    tracker.to_csv(output, index=False)
    _write_tracking_report(report, tracker)
    manifest.write_text(
        json.dumps(
            {
                "tracker": "two_target_metric_tracker",
                "production_outputs_modified": False,
                "tracked_prefixes": TRACKED_PREFIXES,
                "output_files": {
                    "tracker": str(output.relative_to(root)),
                    "report": str(report.relative_to(root)),
                    "manifest": str(manifest.relative_to(root)),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "rows": len(tracker),
        "outputs": {
            "tracker": str(output),
            "report": str(report),
            "manifest": str(manifest),
        },
    }
