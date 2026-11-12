import pandas as pd

from osteoporosis_screening.modeling import feature_experiments


def test_targeted_biomarker_experiments_are_opt_in_and_preserve_e8():
    variables = {
        "feature_groups": {
            "demographic": ["age", "sex"],
            "anthropometric": ["bmi"],
            "lifestyle": [],
            "nutrition": [],
            "clinical": ["prior_fracture_history"],
            "grip": ["combined_grip_strength_kg"],
            "biomarkers": ["serum_calcium_mg_dl", "serum_total_folate_nmol_l"],
        }
    }
    frame = pd.DataFrame({"age": [60], "sex": [1], "bmi": [25.0]})

    pooled = feature_experiments(frame, variables)
    targeted = feature_experiments(frame, variables, include_targeted_biomarkers=True)

    assert "E8_e3_plus_clinical" in pooled
    assert "E9_e3_plus_grip" not in pooled
    assert targeted["E9_e3_plus_grip"][0] == [
        "age",
        "sex",
        "bmi",
        "combined_grip_strength_kg",
    ]
    assert targeted["E10_e3_plus_grip_biomarkers"][0][-2:] == [
        "serum_calcium_mg_dl",
        "serum_total_folate_nmol_l",
    ]


def test_medical_mobility_experiment_is_opt_in_and_extends_e8():
    variables = {
        "feature_groups": {
            "demographic": ["age", "sex"],
            "anthropometric": ["bmi"],
            "lifestyle": [],
            "nutrition": [],
            "clinical": ["prior_fracture_history"],
            "medical_mobility": ["arthritis_history", "thyroid_problem"],
        }
    }
    frame = pd.DataFrame({"age": [60], "sex": [2], "bmi": [25.0]})
    experiments = feature_experiments(
        frame, variables, include_medical_mobility_panel=True
    )
    assert experiments["E12_e8_plus_medical_mobility"][0][-2:] == [
        "arthritis_history",
        "thyroid_problem",
    ]
