import numpy as np
import pandas as pd

from osteoporosis_screening.integration import (
    BASELINE_POLICY,
    FIXED_RECALL_POLICY,
    paired_policy_intervals,
    prevalence_standardized_weights,
    select_precision_constrained_model,
    source_threshold_policy,
    training_weights,
)


def test_prevalence_standardized_weights_reach_requested_prevalence():
    outcome = np.array([0, 0, 0, 1, 1])
    original = np.array([1.0, 2.0, 1.0, 3.0, 1.0])
    standardized = prevalence_standardized_weights(outcome, original, 0.056)
    assert np.isclose(np.average(outcome, weights=standardized), 0.056)
    assert np.isclose(standardized.mean(), 1.0)


def test_dataset_balanced_weights_preserve_within_dataset_ratios():
    frame = pd.DataFrame(
        {
            "source_dataset": ["nhanes", "nhanes", "knhanes", "knhanes"],
            "native_weight": [1.0, 3.0, 2.0, 6.0],
        }
    )
    balanced = training_weights(frame, "equal_dataset_total")
    totals = balanced.groupby(frame["source_dataset"]).sum()
    assert np.isclose(totals["nhanes"], totals["knhanes"])
    assert np.isclose(balanced.iloc[1] / balanced.iloc[0], 3.0)
    assert np.isclose(balanced.iloc[3] / balanced.iloc[2], 3.0)


def test_native_weights_only_change_scale():
    frame = pd.DataFrame(
        {
            "source_dataset": ["nhanes", "knhanes"],
            "native_weight": [2.0, 8.0],
        }
    )
    native = training_weights(frame, "native")
    assert np.isclose(native.mean(), 1.0)
    assert np.isclose(native.iloc[1] / native.iloc[0], 4.0)


def test_source_threshold_policy_meets_each_recall_constraint():
    frame = pd.DataFrame(
        {
            "source_dataset": ["nhanes"] * 5 + ["knhanes"] * 5,
            "osteoporosis": [0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        }
    )
    probability = np.array([0.1, 0.2, 0.8, 0.7, 0.9, 0.1, 0.4, 0.6, 0.5, 0.9])
    thresholds, rows = source_threshold_policy(
        frame,
        probability,
        np.ones(len(frame)),
        {"nhanes": 1.0, "knhanes": 1.0},
    )
    assert set(thresholds) == {"nhanes", "knhanes"}
    assert all(row["oof_recall"] == 1.0 for row in rows)
    assert all(row["recall_constraint_met"] for row in rows)


def test_source_threshold_policy_guards_every_training_cohort():
    frame = pd.DataFrame(
        {
            "source_dataset": ["nhanes"] * 6,
            "cohort_id": ["cycle_a"] * 3 + ["cycle_b"] * 3,
            "osteoporosis": [0, 0, 1, 0, 0, 1],
        }
    )
    probability = np.array([0.1, 0.8, 0.7, 0.1, 0.2, 0.3])
    thresholds, rows = source_threshold_policy(
        frame,
        probability,
        np.ones(len(frame)),
        {"nhanes": 1.0},
    )
    assert thresholds["nhanes"] == 0.3
    assert rows[0]["limiting_training_cohort"] == "cycle_b"
    assert rows[0]["minimum_cohort_oof_recall"] == 1.0


def test_constrained_selection_uses_worst_source_ppv_and_rejects_infeasible_model():
    rows = [
        {
            "model": "stable",
            "source_dataset": source,
            "recall_constraint_met": True,
            "oof_ppv_standardized_0_056": ppv,
            "oof_specificity": specificity,
        }
        for source, ppv, specificity in [
            ("nhanes", 0.20, 0.70),
            ("knhanes", 0.19, 0.68),
        ]
    ]
    rows.extend(
        {
            "model": "uneven",
            "source_dataset": source,
            "recall_constraint_met": feasible,
            "oof_ppv_standardized_0_056": ppv,
            "oof_specificity": specificity,
        }
        for source, feasible, ppv, specificity in [
            ("nhanes", True, 0.30, 0.80),
            ("knhanes", False, 0.25, 0.75),
        ]
    )
    assert select_precision_constrained_model(rows) == "stable"


def test_paired_policy_intervals_compare_matching_holdout_participants():
    base = pd.DataFrame(
        {
            "strategy": "pooled_naive",
            "holdout": "nhanes_test",
            "participant_id": [f"p{i}" for i in range(8)],
            "osteoporosis": [0, 1, 0, 1, 0, 1, 0, 1],
            "analysis_weight": np.ones(8),
            "survey_stratum": [1, 1, 1, 1, 2, 2, 2, 2],
            "survey_psu": [1, 1, 2, 2, 1, 1, 2, 2],
            "prediction": [1, 1, 0, 1, 1, 1, 0, 1],
            "policy": BASELINE_POLICY,
        }
    )
    candidate = base.copy()
    candidate["policy"] = FIXED_RECALL_POLICY
    candidate["prediction"] = [0, 1, 0, 1, 0, 1, 0, 1]
    result = paired_policy_intervals(
        pd.concat([base, candidate], ignore_index=True), iterations=10, seed=7
    )
    accuracy = result.loc[result["metric"].eq("accuracy")].iloc[0]
    assert accuracy["delta"] > 0
    assert accuracy["successful_bootstrap_iterations"] == 10
