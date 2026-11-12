import numpy as np
import pandas as pd

from osteoporosis_screening.feature_combination import (
    _normalize_feature_panels,
    _panel_availability,
    _prepare_target_frame,
)
from osteoporosis_screening.features import engineer_features


def test_albumin_alkaline_phosphatase_ratio_is_engineered_safely():
    frame = pd.DataFrame(
        {
            "serum_albumin_g_dl": [4.0, 3.5],
            "alkaline_phosphatase_iu_l": [80.0, 0.0],
        }
    )

    result = engineer_features(frame)

    assert np.isclose(result.loc[0, "albumin_alkaline_phosphatase_ratio"], 0.05)
    assert np.isnan(result.loc[1, "albumin_alkaline_phosphatase_ratio"])


def test_panel_availability_requires_every_production_cycle():
    frame = pd.DataFrame(
        {
            "survey_cycle": ["A", "A", "B", "B"],
            "albumin": [4.0, 4.1, 3.9, 4.2],
            "medication": ["no", "yes", None, None],
        }
    )

    result = _panel_availability(
        frame,
        {"chemistry": ["albumin"], "medication": ["medication"]},
        minimum_fraction=0.25,
    )

    assert result.loc[result["panel"].eq("chemistry"), "eligible_all_production_cycles"].all()
    assert not result.loc[
        result["panel"].eq("medication"), "eligible_all_production_cycles"
    ].any()


def test_low_bmd_feature_panel_config_can_use_pure_and_e12_bases():
    variables = {
        "feature_groups": {
            "demographic": ["age", "sex"],
            "anthropometric": ["bmi", "weight_kg"],
            "published_low_bmd_panel": ["age", "bmi", "total_cholesterol_mg_dl"],
        }
    }
    experiment = {
        "feature_panels": {
            "published_only": {
                "base": "none",
                "feature_groups": ["published_low_bmd_panel"],
            },
            "e12_plus_lab": {
                "base": "e12",
                "features": ["total_cholesterol_mg_dl"],
            },
        }
    }

    result = _normalize_feature_panels(experiment, variables, ["age", "mobility"])

    assert result["published_only"] == ["age", "bmi", "total_cholesterol_mg_dl"]
    assert result["e12_plus_lab"] == ["age", "mobility", "total_cholesterol_mg_dl"]


def test_prepare_target_frame_derives_low_bmd_outcome():
    frame = pd.DataFrame({"femoral_neck_t_score": [-0.5, -1.0, -1.1, None]})

    result, outcome_column, definition = _prepare_target_frame(
        frame, "low_bmd", {"low_bmd_t_score_cutoff": -1.0}
    )

    assert outcome_column == "low_bmd"
    assert result["low_bmd"].tolist() == [0, 0, 1]
    assert "T-score < -1" in definition
