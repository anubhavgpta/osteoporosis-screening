import numpy as np
import pandas as pd

from osteoporosis_screening.feature_combination import (
    _candidate_estimators,
    _clinical_score_baselines,
    _clinical_score_values,
)


def test_candidate_grid_includes_literature_extensions():
    names = [
        name
        for name, _ in _candidate_estimators(
            {
                "tree_n_estimators": [5],
                "logistic_c": [1.0],
                "lasso_selection_c": [0.1],
                "mlp_hidden_layer_sizes": [[4]],
                "mlp_alpha": [0.001],
                "include_stacking": True,
                "include_resampling_candidates": False,
            },
            seed=42,
            n_jobs=1,
        )
    ]

    assert "lasso_selected_logistic_c_0.1" in names
    assert "mlp_4_alpha_0.001" in names
    assert "stacked_tree_ensemble" in names


def test_clinical_score_values_put_higher_osta_risk_on_lower_weight_for_age():
    frame = pd.DataFrame(
        {
            "age": [60, 60],
            "weight_kg": [45.0, 75.0],
            "bmi": [19.0, 28.0],
            "prior_fracture_history": ["yes", "no"],
            "parental_hip_fracture": ["no", "no"],
            "long_term_glucocorticoid": ["no", "no"],
            "current_smoker": ["no", "no"],
            "alcohol_drinks_week": [0.0, 0.0],
        }
    )

    osta = _clinical_score_values(frame, "osta")
    frax_proxy = _clinical_score_values(frame, "frax_grade_risk_count")

    assert osta.iloc[0] > osta.iloc[1]
    assert frax_proxy.iloc[0] > frax_proxy.iloc[1]


def test_clinical_score_baselines_emit_metrics_and_predictions():
    rng = np.random.default_rng(123)
    rows = []
    for cycle_index, cycle in enumerate(["2007-2008", "2009-2010", "2013-2014", "2017-2018"]):
        for row in range(30):
            age = 52 + row % 25
            weight = 48 + (row * 3) % 35
            risk = age - weight + cycle_index
            rows.append(
                {
                    "participant_id": f"{cycle}_{row}",
                    "survey_cycle": cycle,
                    "sex": 2 if row % 2 else 1,
                    "age": age,
                    "weight_kg": float(weight),
                    "bmi": 20.0 + (row % 8),
                    "prior_fracture_history": "yes" if row % 11 == 0 else "no",
                    "parental_hip_fracture": "no",
                    "long_term_glucocorticoid": "no",
                    "current_smoker": "yes" if row % 13 == 0 else "no",
                    "alcohol_drinks_week": float(row % 24),
                    "osteoporosis": int(risk + rng.normal(0, 3) > 18),
                    "mec_weight": 1.0,
                }
            )
    frame = pd.DataFrame(rows)

    metrics, predictions = _clinical_score_baselines(frame, target=0.8, confidence=0.95)

    assert {"osta", "frax_grade_risk_count"}.issubset(set(metrics["score"]))
    assert {"point_recall", "lcb_recall"}.issubset(set(metrics["policy"]))
    assert not predictions.empty
