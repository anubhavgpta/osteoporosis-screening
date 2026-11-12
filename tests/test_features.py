import numpy as np
import pandas as pd

from osteoporosis_screening.features import (
    build_cohorts,
    engineer_features,
    engineer_outcomes,
    replace_special_missing,
)


def test_primary_classification_cohort_does_not_require_diet_recall():
    frame = pd.DataFrame(
        {
            "femur_scan_status": [1, 1],
            "femoral_neck_bmd": [0.7, 0.8],
            "age": [60, 70],
            "dietary_recall_status": [1, 2],
        }
    )
    classification, regression, _ = build_cohorts(frame, 50, 40)
    assert len(classification) == 2
    assert classification["reliable_day1_diet"].tolist() == [True, False]
    assert len(regression) == 1


def test_outcome_thresholds_include_exact_osteoporosis_cutoff():
    reference = {"mean_bmd": 0.858, "sd_bmd": 0.120}
    frame = pd.DataFrame(
        {"femoral_neck_bmd": [0.858, 0.858 - 1.0 * 0.120, 0.858 - 2.5 * 0.120]}
    )
    result = engineer_outcomes(frame, reference)
    assert result["bone_density_class"].astype(str).tolist() == [
        "normal",
        "normal",
        "osteoporosis",
    ]
    assert result["low_bmd_or_osteoporosis"].tolist() == [0, 0, 1]
    assert result["osteoporosis"].tolist() == [0, 0, 1]


def test_energy_adjustment_and_activity_features():
    frame = pd.DataFrame(
        {
            "energy_kcal": [2000.0],
            "calcium_mg": [1000.0],
            "vitamin_d_mcg": [20.0],
            "magnesium_mg": [400.0],
            "phosphorus_mg": [1250.0],
            "protein_g": [80.0],
            "weight_kg": [80.0],
            "vigorous_work_days": [2.0],
            "vigorous_work_minutes": [30.0],
        }
    )
    result = engineer_features(frame)
    assert np.isclose(result.loc[0, "calcium_per_1000kcal"], 500)
    assert np.isclose(result.loc[0, "protein_per_kg"], 1)
    assert np.isclose(result.loc[0, "activity_met_minutes_week"], 480)


def test_activity_skip_is_zero_but_sentinel_nonresponse_stays_missing():
    frame = pd.DataFrame(
        {
            "vigorous_work_days": [np.nan, 77.0, 2.0],
            "vigorous_work_minutes": [np.nan, np.nan, 9999.0],
        }
    )
    cleaned = replace_special_missing(
        frame,
        {
            "vigorous_work_days": [77, 99],
            "vigorous_work_minutes": [7777, 9999],
        },
    )
    result = engineer_features(cleaned)

    assert result.loc[0, "activity_met_minutes_week"] == 0
    assert np.isnan(result.loc[1, "activity_met_minutes_week"])
    assert np.isnan(result.loc[2, "activity_met_minutes_week"])
    assert result["activity_nonresponse"].tolist() == [0, 1, 1]


def test_age_sex_interaction_is_derived_without_using_new_information():
    result = engineer_features(pd.DataFrame({"age": [60, 70, 80], "sex": [1, 2, 1]}))
    assert result["age_sex_interaction"].tolist() == [
        "sex_1__age_under_65",
        "sex_2__age_65_74",
        "sex_1__age_75_plus",
    ]


def test_maximum_grip_strength_is_derived_from_all_recorded_trials():
    frame = pd.DataFrame(
        {
            "grip_hand1_trial1_kg": [20.0, np.nan],
            "grip_hand1_trial2_kg": [22.0, 18.0],
            "grip_hand2_trial1_kg": [21.0, np.nan],
            "grip_hand2_trial3_kg": [23.5, np.nan],
        }
    )
    result = engineer_features(frame)
    assert result["maximum_grip_strength_kg"].tolist() == [23.5, 18.0]


def test_clinical_risk_features_follow_questionnaire_skip_logic():
    frame = pd.DataFrame(
        {
            "sex": [1, 2, 2],
            "fractured_hip_history": [2, 1, 2],
            "fractured_wrist_history": [2, 2, 2],
            "fractured_spine_history": [2, 2, 2],
            "other_adult_fracture_history": [2, 2, 2],
            "maternal_hip_fracture_raw": [2, 1, 2],
            "paternal_hip_fracture_raw": [2, 2, 2],
            "long_term_glucocorticoid_raw": [2, 1, 2],
            "period_past_year_raw": [np.nan, 2, 1],
            "amenorrhea_reason_raw": [np.nan, 7, np.nan],
            "ovaries_removed_raw": [np.nan, 2, 2],
            "hormone_therapy_raw": [np.nan, 1, 2],
        }
    )
    result = engineer_features(frame)
    assert result["prior_fracture_history"].tolist() == ["no", "yes", "no"]
    assert result["parental_hip_fracture"].tolist() == ["no", "yes", "no"]
    assert result["long_term_glucocorticoid"].tolist() == ["no", "yes", "no"]
    assert result["menopausal_status"].tolist() == [
        "not_applicable_male",
        "no_period_past_year",
        "menstruating",
    ]
    assert result["hormone_therapy_ever"].tolist() == [
        "not_applicable_male",
        "yes",
        "no",
    ]


def test_medical_mobility_features_use_non_leaking_questionnaire_fields():
    frame = pd.DataFrame(
        {
            "arthritis_raw": [1, 1, 2, np.nan],
            "arthritis_type_raw": [2, 1, np.nan, np.nan],
            "liver_condition_raw": [1, 2, 2, np.nan],
            "thyroid_problem_raw": [2, 1, 2, np.nan],
            "walking_equipment_raw": [2, 2, 1, np.nan],
        }
    )
    result = engineer_features(frame)
    assert result["arthritis_history"].tolist()[:3] == ["yes", "yes", "no"]
    assert result["rheumatoid_arthritis"].tolist()[:3] == ["yes", "no", "no"]
    assert result["liver_condition"].tolist()[:3] == ["yes", "no", "no"]
    assert result["thyroid_problem"].tolist()[:3] == ["no", "yes", "no"]
    assert result["needs_walking_equipment"].tolist()[:3] == ["no", "no", "yes"]
