import numpy as np

from osteoporosis_screening.evaluation import (
    classification_metrics,
    decision_curve,
    threshold_for_f1,
    threshold_for_precision,
)


def test_classification_metrics_include_calibration_and_skill():
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    result = classification_metrics(y, probability, threshold=0.5)
    assert result["accuracy"] == 1.0
    assert result["mcc"] == 1.0
    assert result["brier_skill_score"] > 0
    assert np.isfinite(result["calibration_intercept"])
    assert result["calibration_slope"] > 0


def test_decision_curve_reports_clinical_reference_strategies():
    result = decision_curve(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        thresholds=[0.25, 0.5],
        sample_weight=[1, 1, 1, 1],
    )
    assert set(result.columns) >= {
        "model_net_benefit",
        "test_all_net_benefit",
        "test_none_net_benefit",
        "unnecessary_referrals_avoided_per_100",
    }
    assert (result["model_net_benefit"] > result["test_all_net_benefit"]).all()


def test_precision_threshold_meets_requested_precision_with_best_recall():
    y = np.array([0, 0, 0, 1, 1])
    probability = np.array([0.1, 0.3, 0.6, 0.7, 0.9])
    threshold = threshold_for_precision(y, probability, target=0.8)
    result = classification_metrics(y, probability, threshold=threshold)
    assert result["ppv"] >= 0.8
    assert result["sensitivity"] == 1.0


def test_f1_threshold_maximizes_weighted_f1():
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.45, 0.9])

    threshold = threshold_for_f1(y, probability)
    result = classification_metrics(y, probability, threshold=threshold)

    assert threshold == 0.45
    assert result["f1"] == 1.0
