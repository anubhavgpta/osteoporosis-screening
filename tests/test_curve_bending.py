import numpy as np
import pandas as pd

from osteoporosis_screening.curve_bending import (
    biomarker_cycle_availability,
    select_overall_threshold,
    select_recall_threshold,
    weighted_wilson_lower_bound,
)


def test_lcb_threshold_is_no_higher_than_point_recall_threshold():
    outcome = np.array([1] * 40 + [0] * 60)
    probability = np.r_[np.linspace(0.02, 0.9, 40), np.linspace(0.0, 0.4, 60)]
    weight = np.ones(100)

    point = select_recall_threshold(
        outcome, probability, weight, target=0.9, require_lcb=False
    )
    conservative = select_recall_threshold(
        outcome, probability, weight, target=0.9, require_lcb=True
    )

    assert conservative["threshold"] <= point["threshold"]
    assert conservative["oof_sensitivity"] >= point["oof_sensitivity"]
    assert conservative["oof_sensitivity_lcb"] >= 0.9


def test_weighted_point_threshold_handles_probability_ties():
    outcome = np.array([1, 1, 0, 0])
    probability = np.array([0.9, 0.6, 0.6, 0.2])
    weight = np.array([2.0, 1.0, 3.0, 4.0])

    selected = select_recall_threshold(
        outcome,
        probability,
        weight,
        target=0.6,
        require_lcb=False,
    )

    assert selected["threshold"] == 0.9
    assert selected["oof_sensitivity"] == 2 / 3
    assert selected["oof_specificity"] == 1.0


def test_weighted_wilson_bound_penalizes_small_positive_sample():
    outcome = np.r_[np.ones(10), np.zeros(10)]
    predicted = np.ones(20)
    lower = weighted_wilson_lower_bound(outcome, predicted, confidence=0.95)

    assert 0.7 < lower < 1.0


def test_net_benefit_selection_requires_fixed_clinical_utility():
    outcome = np.array([1, 1, 0, 0])
    probability = np.array([0.8, 0.6, 0.4, 0.2])
    weight = np.ones(4)

    selected = select_recall_threshold(
        outcome,
        probability,
        weight,
        target=0.5,
        require_lcb=False,
        objective="net_benefit",
        utility_threshold_probability=0.05,
    )

    assert selected["utility_threshold_probability"] == 0.05


def test_balanced_accuracy_threshold_is_not_forced_to_high_recall():
    outcome = np.array([1, 1, 0, 0, 0, 0])
    probability = np.array([0.9, 0.45, 0.8, 0.4, 0.2, 0.1])
    selected = select_overall_threshold(
        outcome,
        probability,
        np.ones(len(outcome)),
        objective="balanced_accuracy",
    )

    assert selected["oof_balanced_accuracy"] >= 0.75
    assert selected["recall_constraint"] == "none"


def test_biomarker_availability_prevents_missing_by_cycle_panel():
    frame = pd.DataFrame(
        {
            "survey_cycle": ["A", "A", "B", "B"],
            "calcium": [1.0, 2.0, 1.0, 2.0],
            "folate": [1.0, 2.0, np.nan, np.nan],
        }
    )
    result = biomarker_cycle_availability(
        frame,
        {"core": ["calcium"], "full": ["calcium", "folate"]},
        minimum_observed_fraction=0.25,
    )

    core = result.loc[result["panel"].eq("core")]
    full_b = result.loc[result["panel"].eq("full") & result["survey_cycle"].eq("B")]
    assert core["all_features_available"].all()
    assert not bool(full_b.iloc[0]["all_features_available"])
