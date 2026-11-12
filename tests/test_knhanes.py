import numpy as np
import pandas as pd

from osteoporosis_screening.knhanes import (
    DXA_WEIGHT_SPECS,
    NHANES_PIR_QUARTILE_MIDPOINTS,
    _doctor_diagnosis,
    _menopausal_status,
    _normalise_identifier,
    _yes_no_10,
    _yes_no_12,
)


def test_knhanes_binary_codings_preserve_nonresponse_as_missing():
    assert _yes_no_12(pd.Series([1, 2, 8, 9, np.nan])).tolist()[:2] == ["yes", "no"]
    assert _yes_no_12(pd.Series([1, 2, 8, 9, np.nan])).isna().tolist()[2:] == [
        True,
        True,
        True,
    ]
    assert _yes_no_10(pd.Series([1, 0, 8, 9, np.nan])).tolist()[:2] == ["yes", "no"]
    assert _yes_no_10(pd.Series([1, 0, 8, 9, np.nan])).isna().tolist()[2:] == [
        True,
        True,
        True,
    ]
    diagnosis = _doctor_diagnosis(pd.Series([1, 0, 8, 9, np.nan]))
    assert diagnosis.tolist()[:3] == ["yes", "no", "no"]
    assert diagnosis.isna().tolist()[3:] == [True, True]


def test_knhanes_identifier_and_income_transport_are_deterministic():
    identifiers = _normalise_identifier(pd.Series([b"A001", "B002", np.nan]))
    assert identifiers.tolist() == ["A001", "B002", pd.NA]
    assert NHANES_PIR_QUARTILE_MIDPOINTS == {1: 1.17, 2: 2.54, 3: 4.51, 4: 5.0}
    assert np.isclose(sum(factor for _, factor in DXA_WEIGHT_SPECS.values()), 1.0)


def test_knhanes_menopause_mapping_does_not_call_hysterectomy_oophorectomy():
    wave4 = pd.DataFrame(
        {
            "sex": [1, 2, 2, 2],
            "HW_MS": [np.nan, 2, 5, 6],
        }
    )
    assert _menopausal_status(wave4, 2009).tolist() == [
        "not_applicable_male",
        "menstruating",
        "no_period_past_year",
        "no_period_past_year",
    ]

    wave5 = pd.DataFrame(
        {
            "sex": [1, 2, 2, 2],
            "LW_ms": [np.nan, 2, 3, 3],
            "LW_ov": [np.nan, 2, 2, 1],
            "DX_Q_MP": [8, 2, 1, 3],
        }
    )
    assert _menopausal_status(wave5, 2010).tolist() == [
        "not_applicable_male",
        "menstruating",
        "no_period_past_year",
        "surgical_menopause",
    ]
