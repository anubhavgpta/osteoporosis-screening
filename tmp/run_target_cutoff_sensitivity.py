from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from osteoporosis_screening.config import ProjectPaths, load_yaml
from osteoporosis_screening.curve_bending import _experiment_feature_sets
from osteoporosis_screening.target_definition import (
    _run_naive,
    _run_rigorous_loco,
    _targeted_frame,
)


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


def summarize_cell_metrics(cells: pd.DataFrame) -> pd.DataFrame:
    return (
        cells.groupby(["t_score_cutoff", "methodology", "evaluation_design"], as_index=False)
        .agg(
            prevalence=("prevalence", "mean"),
            weighted_prevalence=("weighted_prevalence", "mean"),
            roc_auc=("roc_auc", "mean"),
            pr_auc=("pr_auc", "mean"),
            sensitivity=("sensitivity", "mean"),
            specificity=("specificity", "mean"),
            ppv=("ppv", "mean"),
            calibration_slope=("calibration_slope", "mean"),
            calibration_intercept=("calibration_intercept", "mean"),
            brier_score=("brier_score", "mean"),
            evaluations=("t_score_cutoff", "size"),
            evaluation_n=("evaluation_n", "sum"),
        )
        .sort_values(["methodology", "t_score_cutoff"])
    )


def main() -> None:
    root = Path.cwd()
    paths = ProjectPaths.from_root(root)
    model_config = load_yaml(root / "configs" / "model_config.yaml")
    variables = load_yaml(root / "configs" / "variables.yaml")
    experiment_config = load_yaml(root / "configs" / "target_definition_experiment.yaml")
    output_metrics = paths.tables / "target_definition_cutoff_sensitivity_metrics.csv"
    output_report = root / "reports" / "TARGET_DEFINITION_CUTOFF_SENSITIVITY.md"
    if output_metrics.exists() or output_report.exists():
        raise FileExistsError(
            "target_definition_cutoff_sensitivity outputs already exist; delete them "
            "or archive them before rerunning this ad hoc diagnostic."
        )

    frame = pd.read_csv(
        paths.processed / "classification_cohort.csv.gz", low_memory=False
    ).dropna(subset=["osteoporosis", "femoral_neck_t_score"])
    features = _experiment_feature_sets(frame, variables)["E12_e8_plus_medical_mobility"]
    categorical = set(variables["categorical_features"])
    cutoffs = [-0.8, -0.9, -1.0, -1.1, -1.2, -1.5, -2.0, -2.5]
    cell_frames = []

    for cutoff in cutoffs:
        target_frame = _targeted_frame(frame, "low_bmd", cutoff)
        outcome = target_frame["analysis_outcome"].astype(int)
        weight = target_frame["mec_weight"].fillna(1.0)
        prevalence = float(outcome.mean())
        weighted_prevalence = float(np.average(outcome, weights=weight))
        _, rigorous_cells, _ = _run_rigorous_loco(
            target_frame,
            "low_bmd",
            features,
            categorical,
            model_config,
            experiment_config,
        )
        rigorous_cells["t_score_cutoff"] = cutoff
        rigorous_cells["prevalence"] = prevalence
        rigorous_cells["weighted_prevalence"] = weighted_prevalence
        naive_cells, _, _ = _run_naive(
            target_frame,
            "low_bmd",
            features,
            categorical,
            experiment_config,
            use_smote=False,
        )
        naive_cells["t_score_cutoff"] = cutoff
        naive_cells["prevalence"] = prevalence
        naive_cells["weighted_prevalence"] = weighted_prevalence
        cell_frames.extend([rigorous_cells, naive_cells])
        print(
            f"completed cutoff={cutoff:g} prevalence={prevalence:.3f} "
            f"weighted_prevalence={weighted_prevalence:.3f}",
            flush=True,
        )

    cells = pd.concat(cell_frames, ignore_index=True)
    summary = summarize_cell_metrics(cells)
    summary.to_csv(output_metrics, index=False)
    best_pr = summary.sort_values("pr_auc", ascending=False).iloc[0]
    best_roc = summary.sort_values("roc_auc", ascending=False).iloc[0]
    lines = [
        "# Target-definition cutoff sensitivity",
        "",
        "This ad hoc diagnostic changes only the low-BMD femoral-neck T-score cutoff. "
        "It is not a production model-selection or promotion analysis.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Highest PR-AUC",
        "",
        f"The highest PR-AUC was {best_pr['pr_auc']:.4f} at cutoff "
        f"{best_pr['t_score_cutoff']:.1f} in the `{best_pr['methodology']}` arm.",
        "",
        "## Highest ROC-AUC",
        "",
        f"The highest ROC-AUC was {best_roc['roc_auc']:.4f} at cutoff "
        f"{best_roc['t_score_cutoff']:.1f} in the `{best_roc['methodology']}` arm.",
    ]
    output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
