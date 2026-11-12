import pandas as pd

from osteoporosis_screening.result_tracking import (
    run_low_bmd_pipeline_comparison,
    run_result_tracking,
)


def test_result_tracker_computes_feature_panel_deltas(tmp_path):
    tables = tmp_path / "reports" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "target": "low_bmd",
                "methodology": "rigorous",
                "evaluation_design": "leave_one_production_cycle_out",
                "roc_auc": 0.79,
                "pr_auc": 0.76,
                "sensitivity": 0.90,
                "specificity": 0.42,
                "ppv": 0.56,
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
                "brier_score": 0.18,
                "threshold": 0.25,
                "evaluations": 4,
                "evaluation_n": 100,
            }
        ]
    ).to_csv(tables / "target_definition_cell_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "feature_set": "anthropometric_first",
                "outer_cycles": 4,
                "selected_models": "catboost",
                "mean_roc_auc": 0.81,
                "mean_pr_auc": 0.78,
                "mean_log_loss": 0.5,
                "mean_brier_score": 0.17,
                "mean_point_sensitivity": 0.91,
                "minimum_point_sensitivity": 0.88,
                "mean_point_specificity": 0.50,
                "mean_point_ppv": 0.60,
                "mean_lcb_sensitivity": 0.94,
                "minimum_lcb_sensitivity": 0.90,
                "mean_lcb_specificity": 0.45,
            }
        ]
    ).to_csv(tables / "low_bmd_feature_combination_panel_summary.csv", index=False)

    run_result_tracking(tmp_path)

    tracker = pd.read_csv(tables / "two_target_metric_tracker.csv")
    candidate = tracker.loc[tracker["source"].eq("low_bmd_feature_combination")].iloc[0]
    assert candidate["target"] == "low_bmd"
    assert round(candidate["delta_pr_auc_vs_baseline"], 4) == 0.02
    assert (tmp_path / "reports" / "TWO_TARGET_TRACKING_RESULTS.md").exists()


def test_low_bmd_pipeline_comparison_writes_additive_prefixed_outputs(tmp_path):
    tables = tmp_path / "reports" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "target": "osteoporosis",
                "methodology": "rigorous",
                "evaluation_design": "leave_one_production_cycle_out",
                "roc_auc": 0.70,
                "pr_auc": 0.30,
                "sensitivity": 0.90,
                "specificity": 0.40,
                "ppv": 0.20,
                "f1": 0.33,
                "calibration_slope": 0.9,
                "calibration_intercept": 0.1,
                "brier_score": 0.12,
                "threshold": 0.2,
            },
            {
                "target": "low_bmd",
                "methodology": "rigorous",
                "evaluation_design": "leave_one_production_cycle_out",
                "roc_auc": 0.80,
                "pr_auc": 0.60,
                "sensitivity": 0.91,
                "specificity": 0.45,
                "ppv": 0.50,
                "f1": 0.65,
                "calibration_slope": 1.1,
                "calibration_intercept": -0.1,
                "brier_score": 0.18,
                "threshold": 0.35,
            },
        ]
    ).to_csv(tables / "target_definition_cell_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "feature_set": "E12_baseline",
                "selected_models": "logistic",
                "mean_roc_auc": 0.72,
                "mean_pr_auc": 0.31,
                "mean_brier_score": 0.13,
                "mean_point_sensitivity": 0.9,
                "mean_point_specificity": 0.4,
                "mean_point_ppv": 0.2,
                "mean_point_f1": 0.33,
            }
        ]
    ).to_csv(tables / "feature_combination_panel_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "feature_set": "E12_baseline",
                "selected_models": "logistic",
                "mean_roc_auc": 0.82,
                "mean_pr_auc": 0.62,
                "mean_brier_score": 0.17,
                "mean_point_sensitivity": 0.92,
                "mean_point_specificity": 0.44,
                "mean_point_ppv": 0.52,
                "mean_point_f1": 0.66,
            }
        ]
    ).to_csv(tables / "low_bmd_feature_combination_panel_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "target": "low_bmd",
                "methodology": "rigorous",
                "participant_id": 1,
                "survey_cycle": "2017-2018",
                "analysis_outcome": 1,
                "probability": 0.8,
                "threshold": 0.35,
                "prediction": 1,
            }
        ]
    ).to_csv(tables / "target_definition_predictions.csv.gz", index=False, compression="gzip")

    result = run_low_bmd_pipeline_comparison(tmp_path)

    summary = pd.read_csv(tables / "low_bmd_pipeline_comparison_summary.csv")
    assert result["matched_pipeline_rows"] == 2
    assert summary["matched_targets"].all()
    assert "low_bmd_pr_auc" in summary
    assert (tables / "low_bmd_pipeline_comparison_low_bmd_metrics.csv").exists()
    assert (tables / "low_bmd_pipeline_comparison_thresholds.csv").exists()
    assert (tables / "low_bmd_pipeline_comparison_calibration.csv").exists()
    assert (tables / "low_bmd_pipeline_comparison_predictions.csv.gz").exists()
    report = tmp_path / "reports" / "LOW_BMD_PIPELINE_COMPARISON_RESULTS.md"
    assert "do not replace the production osteoporosis model" in report.read_text(
        encoding="utf-8"
    )
