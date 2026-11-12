from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ProjectPaths, load_yaml
from .curve_bending import (
    run_curve_bending_experiment,
    run_relaxed_operating_experiment,
)
from .external import audit_external_cohort
from .feature_combination import run_feature_combination_experiment
from .imbalance import run_osteoporosis_imbalance_experiment
from .ingest import download_files
from .integration import run_integration_experiment
from .knhanes import harmonize_knhanes, metrics_by_survey_year
from .result_tracking import run_low_bmd_pipeline_comparison, run_result_tracking
from .site_target import run_site_target_experiment
from .target_definition import run_target_definition_experiment
from .workflow import prepare_data, run_all


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="NHANES osteoporosis screening pipeline")
    result.add_argument(
        "command",
        choices=[
            "download",
            "prepare",
            "run",
            "external-audit",
            "knhanes-harmonize",
            "knhanes-validate",
            "integration-experiment",
            "curve-bending-experiment",
            "relaxed-operating-experiment",
            "feature-combination-experiment",
            "low-bmd-feature-combination-experiment",
            "site-target-experiment",
            "osteoporosis-imbalance-experiment",
            "target-definition-experiment",
            "track-two-target-results",
            "low-bmd-pipeline-comparison-experiment",
            "low-bmd-pipeline-comparison",
        ],
    )
    result.add_argument("--root", type=Path, default=Path.cwd())
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--dataset")
    result.add_argument("--input", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    root = args.root.resolve()
    if args.command == "download":
        paths = ProjectPaths.from_root(root)
        variables = load_yaml(root / "configs" / "variables.yaml")
        download_files(paths.raw, variables, overwrite=args.overwrite)
        print(f"Downloaded NHANES components to {paths.raw}")
    elif args.command == "prepare":
        classification, regression, _, _, _ = prepare_data(root)
        print(json.dumps({"classification_n": len(classification), "regression_n": len(regression)}))
    elif args.command == "run":
        print(json.dumps(run_all(root), indent=2))
    elif args.command == "external-audit":
        if not args.dataset or not args.input:
            raise SystemExit("external-audit requires --dataset and --input")
        audit = audit_external_cohort(root, args.dataset, args.input.resolve())
        paths = ProjectPaths.from_root(root)
        paths.ensure()
        prefix = f"external_{args.dataset}"
        audit.metrics.to_csv(paths.tables / f"{prefix}_metrics.csv", index=False)
        audit.predictions.to_csv(
            paths.tables / f"external_{args.dataset}_predictions.csv", index=False
        )
        audit.tradeoffs.to_csv(paths.tables / f"{prefix}_tradeoffs.csv", index=False)
        audit.calibration.to_csv(paths.tables / f"{prefix}_calibration.csv", index=False)
        audit.feature_audit.to_csv(
            paths.tables / f"{prefix}_feature_audit.csv", index=False
        )
        audit.cohort_flow.to_csv(paths.tables / f"{prefix}_cohort_flow.csv", index=False)
        (paths.tables / f"{prefix}_manifest.json").write_text(
            json.dumps(audit.manifest, indent=2), encoding="utf-8"
        )
        print(audit.metrics.to_json(orient="records", indent=2))
    elif args.command in {"knhanes-harmonize", "knhanes-validate"}:
        paths = ProjectPaths.from_root(root)
        paths.ensure()
        harmonized = harmonize_knhanes(root)
        output = paths.processed / "knhanes_2008_2011_harmonized.csv.gz"
        harmonized.frame.to_csv(output, index=False, compression="gzip")
        harmonized.mapping_audit.to_csv(
            paths.tables / "knhanes_2008_2011_mapping_audit.csv", index=False
        )
        harmonized.cohort_flow.to_csv(
            paths.tables / "knhanes_2008_2011_harmonization_flow.csv", index=False
        )
        (paths.tables / "knhanes_2008_2011_harmonization_manifest.json").write_text(
            json.dumps(harmonized.manifest, indent=2), encoding="utf-8"
        )
        if args.command == "knhanes-harmonize":
            print(
                json.dumps(
                    {
                        "output": str(output),
                        "n": len(harmonized.frame),
                        "evaluation_n": harmonized.manifest[
                            "age_50_plus_with_femoral_neck_bmd_n"
                        ],
                    },
                    indent=2,
                )
            )
            return

        audit = audit_external_cohort(root, "knhanes_2008_2011", output)
        prefix = "external_knhanes_2008_2011"
        audit.metrics.to_csv(paths.tables / f"{prefix}_metrics.csv", index=False)
        audit.predictions.to_csv(paths.tables / f"{prefix}_predictions.csv", index=False)
        audit.tradeoffs.to_csv(paths.tables / f"{prefix}_tradeoffs.csv", index=False)
        audit.calibration.to_csv(paths.tables / f"{prefix}_calibration.csv", index=False)
        audit.feature_audit.to_csv(
            paths.tables / f"{prefix}_feature_audit.csv", index=False
        )
        audit.cohort_flow.to_csv(paths.tables / f"{prefix}_cohort_flow.csv", index=False)
        metadata = json.loads(
            (root / "artifacts" / "classification_model_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        metrics_by_survey_year(
            audit.predictions, metadata["operating_points"]
        ).to_csv(paths.tables / f"{prefix}_metrics_by_year.csv", index=False)
        (paths.tables / f"{prefix}_manifest.json").write_text(
            json.dumps(audit.manifest, indent=2), encoding="utf-8"
        )
        print(audit.metrics.to_json(orient="records", indent=2))
    elif args.command == "integration-experiment":
        paths = ProjectPaths.from_root(root)
        paths.ensure()
        experiment = run_integration_experiment(root)
        prefix = "nhanes_knhanes_integration"
        experiment.metrics.to_csv(paths.tables / f"{prefix}_metrics.csv", index=False)
        experiment.ranking.to_csv(paths.tables / f"{prefix}_ranking.csv", index=False)
        experiment.model_selection.to_csv(
            paths.tables / f"{prefix}_model_selection.csv", index=False
        )
        experiment.strategy_audit.to_csv(
            paths.tables / f"{prefix}_strategy_audit.csv", index=False
        )
        experiment.constrained_model_selection.to_csv(
            paths.tables / f"{prefix}_constrained_model_selection.csv", index=False
        )
        experiment.policy_metrics.to_csv(
            paths.tables / f"{prefix}_policy_metrics.csv", index=False
        )
        experiment.policy_comparison.to_csv(
            paths.tables / f"{prefix}_policy_comparison.csv", index=False
        )
        experiment.policy_intervals.to_csv(
            paths.tables / f"{prefix}_policy_intervals.csv", index=False
        )
        experiment.predictions.to_csv(
            paths.tables / f"{prefix}_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
        experiment.policy_predictions.to_csv(
            paths.tables / f"{prefix}_policy_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
        (paths.tables / f"{prefix}_manifest.json").write_text(
            json.dumps(experiment.manifest, indent=2), encoding="utf-8"
        )
        print(
            experiment.policy_comparison.loc[
                experiment.policy_comparison["strategy"].eq("pooled_naive")
            ].to_json(orient="records", indent=2)
        )
    elif args.command == "curve-bending-experiment":
        print(
            json.dumps(
                run_curve_bending_experiment(root, overwrite=args.overwrite),
                indent=2,
            )
        )
    elif args.command == "relaxed-operating-experiment":
        print(
            json.dumps(
                run_relaxed_operating_experiment(root, overwrite=args.overwrite),
                indent=2,
            )
        )
    elif args.command == "feature-combination-experiment":
        print(
            json.dumps(
                run_feature_combination_experiment(root, overwrite=args.overwrite),
                indent=2,
            )
        )
    elif args.command == "low-bmd-feature-combination-experiment":
        print(
            json.dumps(
                run_feature_combination_experiment(
                    root,
                    overwrite=args.overwrite,
                    config_filename="low_bmd_feature_combination_experiment.yaml",
                ),
                indent=2,
            )
        )
    elif args.command == "track-two-target-results":
        print(json.dumps(run_result_tracking(root), indent=2))
    elif args.command == "low-bmd-pipeline-comparison":
        print(
            json.dumps(
                run_low_bmd_pipeline_comparison(root, overwrite=args.overwrite),
                indent=2,
            )
        )
    elif args.command == "low-bmd-pipeline-comparison-experiment":
        target_definition = run_target_definition_experiment(
            root,
            overwrite=args.overwrite,
            output_prefix="low_bmd_pipeline_comparison_target_definition",
        )
        feature_combination = run_feature_combination_experiment(
            root,
            overwrite=args.overwrite,
            config_filename="low_bmd_pipeline_comparison_feature_combination_experiment.yaml",
        )
        comparison = run_low_bmd_pipeline_comparison(root, overwrite=args.overwrite)
        print(
            json.dumps(
                {
                    "target_definition": target_definition,
                    "feature_combination": feature_combination,
                    "comparison": comparison,
                },
                indent=2,
            )
        )
    elif args.command == "site-target-experiment":
        print(
            json.dumps(
                run_site_target_experiment(root, overwrite=args.overwrite),
                indent=2,
            )
        )
    elif args.command == "osteoporosis-imbalance-experiment":
        print(
            json.dumps(
                run_osteoporosis_imbalance_experiment(root, overwrite=args.overwrite),
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                run_target_definition_experiment(root, overwrite=args.overwrite),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
