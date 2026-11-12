import numpy as np
import pandas as pd

from osteoporosis_screening.reporting import (
    classification_bootstrap_intervals,
    missingness_outcome_bias,
)


def test_design_bootstrap_returns_requested_metric_intervals():
    predictions = pd.DataFrame(
        {
            "osteoporosis": [0, 1, 0, 1, 0, 1, 0, 1],
            "probability": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
            "dietary_day1_weight": np.ones(8),
            "mec_weight": np.ones(8),
            "analysis_weight": np.ones(8),
            "survey_stratum": [1, 1, 1, 1, 2, 2, 2, 2],
            "survey_psu": [1, 1, 2, 2, 1, 1, 2, 2],
        }
    )
    result = classification_bootstrap_intervals(
        predictions, {"default": 0.5}, iterations=10, seed=7
    )
    assert set(result["metric"]) == {
        "roc_auc",
        "pr_auc",
        "accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "balanced_accuracy",
        "mcc",
        "log_loss",
        "brier_score",
        "brier_skill_score",
        "calibration_intercept",
        "calibration_slope",
    }
    assert result["ci_low"].notna().all()


def test_missingness_bias_report_contrasts_missing_with_observed():
    frame = pd.DataFrame(
        {
            "osteoporosis": [1, 1, 0, 0],
            "dietary_day1_weight": [1.0] * 4,
            "mec_weight": [1.0] * 4,
            "bmi": [np.nan, np.nan, 20.0, 21.0],
        }
    )
    result = missingness_outcome_bias(frame).set_index("column")
    assert set(result["target"]) == {"osteoporosis"}
    assert result.loc["bmi", "weighted_risk_difference_missing_minus_observed"] == 1
