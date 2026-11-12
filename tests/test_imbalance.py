from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from osteoporosis_screening.imbalance import PositiveWeightClassifier, _thresholds_from_oof


def test_positive_weight_classifier_records_balanced_multiplier():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1])
    weight = np.array([1, 1, 1, 1, 2, 2], dtype=float)
    model = PositiveWeightClassifier(
        LogisticRegression(max_iter=1000), mode="balanced"
    ).fit(X, y, sample_weight=weight)

    assert model.positive_weight_multiplier_ == 1.0
    assert model.predict_proba(X).shape == (6, 2)


def test_thresholds_from_oof_includes_requested_policies():
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    weight = np.ones(4)

    rows = _thresholds_from_oof(
        y,
        probability,
        weight,
        {"recall_targets": [0.8], "precision_targets": [0.5]},
    )

    assert {row["policy"] for row in rows} == {
        "default_0.50",
        "f1_optimized",
        "recall_0.80",
        "precision_0.50",
    }
