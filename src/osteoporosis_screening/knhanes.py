from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import classification_metrics


MODEL_FEATURES = [
    "age",
    "sex",
    "race_ethnicity",
    "education",
    "poverty_income_ratio",
    "bmi",
    "height_cm",
    "weight_kg",
    "prior_fracture_history",
    "parental_hip_fracture",
    "long_term_glucocorticoid",
    "menopausal_status",
    "hormone_therapy_ever",
    "arthritis_history",
    "liver_condition",
    "thyroid_problem",
    "needs_walking_equipment",
]

# Outcome-free transport of a KNHANES household-income rank to the NHANES PIR
# scale. These are survey-weighted NHANES PIR values at the midpoints of four
# equally sized groups (12.5th, 37.5th, 62.5th, and 87.5th percentiles).
NHANES_PIR_QUARTILE_MIDPOINTS = {1: 1.17, 2: 2.54, 3: 4.51, 4: 5.00}

# Official KNHANES guide, table 8: DXA sample-PSU proportions for a combined
# 2008-2011 bone-density analysis.
DXA_WEIGHT_SPECS = {
    2008: ("wt_ex1", 108 / 580),
    2009: ("wt_itvex", 200 / 580),
    2010: ("wt_itvex", 192 / 580),
    2011: ("wt_ex1", 80 / 580),
}


@dataclass(frozen=True)
class KNHANESHarmonization:
    frame: pd.DataFrame
    mapping_audit: pd.DataFrame
    cohort_flow: pd.DataFrame
    manifest: dict


def _normalise_identifier(source: pd.Series) -> pd.Series:
    def convert(value):
        if isinstance(value, bytes):
            return value.decode("ascii")
        if pd.isna(value):
            return pd.NA
        return str(value)

    return source.map(convert).astype("string")


def _yes_no_12(source: pd.Series) -> pd.Series:
    return pd.to_numeric(source, errors="coerce").map({1: "yes", 2: "no"})


def _yes_no_10(source: pd.Series) -> pd.Series:
    return pd.to_numeric(source, errors="coerce").map({1: "yes", 0: "no"})


def _doctor_diagnosis(source: pd.Series) -> pd.Series:
    """Map KNHANES disease follow-up coding for an adult evaluation cohort.

    Code 8 means the diagnosis question was skipped because the respondent
    reported never having the disease (or was a child). All evaluated rows are
    at least 50 years old, so code 8 is a documented negative here. Code 9 is
    nonresponse and remains missing.
    """
    return pd.to_numeric(source, errors="coerce").map(
        {1: "yes", 0: "no", 8: "no"}
    )


def _empty(index: pd.Index) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype="object")


def _menopausal_status(frame: pd.DataFrame, year: int) -> pd.Series:
    result = _empty(frame.index)
    sex = pd.to_numeric(frame["sex"], errors="coerce")
    male = sex.eq(1)
    female = sex.eq(2)
    result.loc[male] = "not_applicable_male"

    if year <= 2009:
        menstrual_status = pd.to_numeric(frame["HW_MS"], errors="coerce")
        result.loc[female & menstrual_status.eq(2)] = "menstruating"
        # Natural menopause and hysterectomy both establish no current period.
        # Hysterectomy is not labeled surgical menopause because the frozen
        # NHANES definition specifically requires bilateral oophorectomy.
        result.loc[female & menstrual_status.isin([5, 6])] = "no_period_past_year"
    else:
        menstrual_status = pd.to_numeric(frame["LW_ms"], errors="coerce")
        ovaries_removed = pd.to_numeric(frame["LW_ov"], errors="coerce")
        dxa_menopause = pd.to_numeric(frame["DX_Q_MP"], errors="coerce")
        result.loc[female & menstrual_status.eq(2)] = "menstruating"
        result.loc[female & menstrual_status.eq(3)] = "no_period_past_year"
        result.loc[female & (ovaries_removed.eq(1) | dxa_menopause.eq(3))] = (
            "surgical_menopause"
        )
    return result


def _year_paths(raw: Path, year: int) -> tuple[Path, Path]:
    suffix = str(year)[-2:]
    return (
        raw / "extracted" / f"hn{suffix}_all.sas7bdat",
        raw / f"KNHANES_{year}_DXA_SAS.sas7bdat",
    )


def _read_year(raw: Path, year: int) -> tuple[pd.DataFrame, dict]:
    basic_path, dxa_path = _year_paths(raw, year)
    for path in (basic_path, dxa_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing KNHANES source file: {path}")

    basic = pd.read_sas(basic_path)
    dxa = pd.read_sas(dxa_path)
    basic_id = "ID" if "ID" in basic else "id"
    basic = basic.rename(columns={basic_id: "participant_id_raw"})
    dxa = dxa.rename(columns={"ID": "participant_id_raw"})
    basic["participant_id_raw"] = _normalise_identifier(basic["participant_id_raw"])
    dxa["participant_id_raw"] = _normalise_identifier(dxa["participant_id_raw"])
    if basic["participant_id_raw"].duplicated().any():
        raise ValueError(f"KNHANES {year} basic file contains duplicate participant IDs")
    if dxa["participant_id_raw"].duplicated().any():
        raise ValueError(f"KNHANES {year} DXA file contains duplicate participant IDs")

    basic_columns = [
        "participant_id_raw",
        "sex",
        "age",
        "ho_incm",
        "edu",
        "psu",
        "kstrata",
        "HE_BMI",
        "HE_ht",
        "HE_wt",
        "DM1_dg",
        "DE2_dg",
        "DK4_dg",
    ]
    if year <= 2009:
        basic_columns.extend(["HW_MS", "HW_EST"])
    else:
        basic_columns.extend(["LW_ms", "LW_mp", "LW_wh", "LW_ov"])
    basic_columns = [column for column in basic_columns if column in basic]

    weight_column, weight_factor = DXA_WEIGHT_SPECS[year]
    dxa_columns = [
        "participant_id_raw",
        "DX_NK_BMD",
        "DX_Q_hsty",
        "DX_Q_MP",
        weight_column,
    ]
    if "DX2_dg" in dxa:
        dxa_columns.append("DX2_dg")

    merged = dxa[dxa_columns].merge(
        basic[basic_columns],
        on="participant_id_raw",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int(merged["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(
            f"KNHANES {year} has {unmatched} DXA participant(s) absent from the basic file"
        )
    merged = merged.drop(columns="_merge")

    output = pd.DataFrame(index=merged.index)
    output["participant_id"] = (
        str(year) + "_" + merged["participant_id_raw"].astype("string")
    )
    output["survey_year"] = year
    output["age"] = pd.to_numeric(merged["age"], errors="coerce")
    output["sex"] = pd.to_numeric(merged["sex"], errors="coerce")
    output["race_ethnicity"] = 5.0
    output["education"] = pd.to_numeric(merged["edu"], errors="coerce").map(
        {1: 1.0, 2: 2.0, 3: 3.0, 4: 5.0}
    )
    output["poverty_income_ratio"] = pd.to_numeric(
        merged["ho_incm"], errors="coerce"
    ).map(NHANES_PIR_QUARTILE_MIDPOINTS)
    output["bmi"] = pd.to_numeric(merged["HE_BMI"], errors="coerce")
    output["height_cm"] = pd.to_numeric(merged["HE_ht"], errors="coerce")
    output["weight_kg"] = pd.to_numeric(merged["HE_wt"], errors="coerce")
    output["femoral_neck_bmd"] = pd.to_numeric(
        merged["DX_NK_BMD"], errors="coerce"
    )
    output["external_weight"] = (
        pd.to_numeric(merged[weight_column], errors="coerce") * weight_factor
    )
    output["survey_psu"] = merged.get("psu", np.nan)
    output["survey_stratum"] = merged.get("kstrata", np.nan)

    output["prior_fracture_history"] = (
        _yes_no_12(merged["DX2_dg"]) if "DX2_dg" in merged else _empty(merged.index)
    )
    output["parental_hip_fracture"] = _yes_no_12(merged["DX_Q_hsty"])
    output["long_term_glucocorticoid"] = _empty(merged.index)
    output["menopausal_status"] = _menopausal_status(merged, year)
    hormone_source = "HW_EST" if year <= 2009 else "LW_wh"
    output["hormone_therapy_ever"] = _yes_no_12(merged[hormone_source])
    output.loc[output["sex"].eq(1), "hormone_therapy_ever"] = (
        "not_applicable_male"
    )
    output["arthritis_history"] = _doctor_diagnosis(merged["DM1_dg"])
    output["liver_condition"] = _doctor_diagnosis(merged["DK4_dg"])
    output["thyroid_problem"] = _doctor_diagnosis(merged["DE2_dg"])
    output["needs_walking_equipment"] = _empty(merged.index)

    flow = {
        "survey_year": year,
        "basic_n": len(basic),
        "dxa_file_n": len(dxa),
        "matched_n": len(merged),
        "age_50_plus_n": int(output["age"].ge(50).sum()),
        "age_50_plus_with_femoral_neck_bmd_n": int(
            (output["age"].ge(50) & output["femoral_neck_bmd"].notna()).sum()
        ),
        "dxa_weight_source": weight_column,
        "combined_weight_factor": weight_factor,
    }
    return output, flow


def _mapping_audit(frame: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        ("age", "age", "exact", "attained age", "attained age", "all years", ""),
        ("sex", "sex", "exact", "1 male, 2 female", "1 male, 2 female", "all years", ""),
        (
            "race_ethnicity",
            "fixed Korean population",
            "domain_proxy",
            "national Korean survey population",
            "NHANES code 5: other race including multiracial",
            "all years",
            "No equivalent NHANES Korean category exists.",
        ),
        (
            "education",
            "edu",
            "coarsened_recode",
            "four KNHANES attainment levels",
            "NHANES DMDEDUC2 codes 1, 2, 3, and 5",
            "all years",
            "KNHANES college-or-higher maps to NHANES college graduate-or-higher.",
        ),
        (
            "poverty_income_ratio",
            "ho_incm",
            "rank_proxy",
            "household-income quartile",
            "continuous NHANES poverty-income ratio",
            "all years",
            "Mapped to survey-weighted NHANES PIR quartile midpoints without outcomes.",
        ),
        ("bmi", "HE_BMI", "exact", "kg/m^2", "kg/m^2", "all years", ""),
        ("height_cm", "HE_ht", "exact", "cm", "cm", "all years", ""),
        ("weight_kg", "HE_wt", "exact", "kg", "kg", "all years", ""),
        (
            "prior_fracture_history",
            "DX2_dg",
            "partial_exact",
            "doctor-diagnosed spine, hip, wrist, or other fracture",
            "adult hip, wrist, spine, or other fracture history",
            "2010-2011",
            "Not collected in the 2008-2009 DXA questionnaire.",
        ),
        (
            "parental_hip_fracture",
            "DX_Q_hsty",
            "broader_proxy",
            "parental osteoporosis, kyphosis, or low-trauma fracture",
            "maternal or paternal hip fracture",
            "all years",
            "KNHANES does not identify fracture site or parent separately.",
        ),
        (
            "long_term_glucocorticoid",
            "unavailable",
            "unavailable",
            "not collected",
            "long-term prednisone/cortisone use",
            "all years",
            "Preserved as missing; frozen NHANES imputer is used.",
        ),
        (
            "menopausal_status",
            "HW_MS / LW_ms / LW_ov / DX_Q_MP",
            "compatible_recode",
            "menstrual status and bilateral oophorectomy",
            "menstruating, no period, or surgical menopause",
            "all years",
            "Hysterectomy alone is not called surgical menopause.",
        ),
        (
            "hormone_therapy_ever",
            "HW_EST / LW_wh",
            "exact",
            "ever used female hormone therapy",
            "ever used female hormone therapy",
            "all years",
            "Men are set to the frozen not-applicable category.",
        ),
        (
            "arthritis_history",
            "DM1_dg",
            "exact",
            "doctor-diagnosed osteoarthritis or rheumatoid arthritis",
            "doctor-diagnosed arthritis",
            "all years",
            "",
        ),
        (
            "liver_condition",
            "DK4_dg",
            "narrower_proxy",
            "doctor-diagnosed cirrhosis",
            "doctor-diagnosed liver condition",
            "all years",
            "KNHANES cirrhosis misses less severe liver conditions.",
        ),
        (
            "thyroid_problem",
            "DE2_dg",
            "exact",
            "doctor-diagnosed thyroid disorder",
            "doctor-diagnosed thyroid problem",
            "all years",
            "",
        ),
        (
            "needs_walking_equipment",
            "unavailable",
            "unavailable",
            "not collected compatibly",
            "requires special walking equipment",
            "all years",
            "Preserved as missing; frozen NHANES imputer is used.",
        ),
        (
            "femoral_neck_bmd",
            "DX_NK_BMD",
            "exact_measure_domain_shift",
            "femoral-neck BMD in g/cm^2",
            "femoral-neck BMD in g/cm^2",
            "all years",
            "Scanner, calibration, and population reference may differ from NHANES.",
        ),
        (
            "external_weight",
            "wt_ex1 / wt_itvex",
            "official_combined_weight",
            "KNHANES DXA examination weight",
            "external analysis weight",
            "all years",
            "Combined with official table-8 PSU proportions (108/200/192/80 of 580).",
        ),
    ]
    result = pd.DataFrame(
        definitions,
        columns=[
            "target_variable",
            "source_variable",
            "mapping_quality",
            "source_definition",
            "target_definition",
            "years_available",
            "notes",
        ],
    )
    missing = frame.isna().mean().mul(100)
    result["missing_percent_all_dxa_rows"] = result["target_variable"].map(missing)
    eligible = frame.loc[frame["age"].ge(50) & frame["femoral_neck_bmd"].notna()]
    eligible_missing = eligible.isna().mean().mul(100)
    result["missing_percent_evaluation_cohort"] = result["target_variable"].map(
        eligible_missing
    )
    return result


def harmonize_knhanes(root: Path) -> KNHANESHarmonization:
    root = root.resolve()
    raw = root / "data" / "raw_knhanes"
    frames = []
    flow_rows = []
    for year in DXA_WEIGHT_SPECS:
        frame, flow = _read_year(raw, year)
        frames.append(frame)
        flow_rows.append(flow)
    combined = pd.concat(frames, ignore_index=True)
    if combined["participant_id"].duplicated().any():
        raise ValueError("Harmonized KNHANES participant IDs are not unique")

    missing_columns = sorted(set(MODEL_FEATURES) - set(combined))
    if missing_columns:
        raise ValueError("Harmonizer failed to create: " + ", ".join(missing_columns))
    mapping = _mapping_audit(combined)
    cohort_flow = pd.DataFrame(flow_rows)
    source_paths = [path for year in DXA_WEIGHT_SPECS for path in _year_paths(raw, year)]
    manifest = {
        "dataset": "knhanes_2008_2011",
        "harmonization_mode": "official_codebook_documented_prediction_only",
        "survey_years": list(DXA_WEIGHT_SPECS),
        "input_n": len(combined),
        "age_50_plus_with_femoral_neck_bmd_n": int(
            (combined["age"].ge(50) & combined["femoral_neck_bmd"].notna()).sum()
        ),
        "nhanes_pir_quartile_midpoints": NHANES_PIR_QUARTILE_MIDPOINTS,
        "dxa_weight_specs": {
            str(year): {"column": column, "factor": factor}
            for year, (column, factor) in DXA_WEIGHT_SPECS.items()
        },
        "source_files": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "outcome_used_in_harmonization": False,
        "model_refit": False,
        "threshold_tuned": False,
    }
    return KNHANESHarmonization(combined, mapping, cohort_flow, manifest)


def metrics_by_survey_year(
    predictions: pd.DataFrame, operating_points: dict[str, float]
) -> pd.DataFrame:
    """Apply frozen operating points separately to each KNHANES survey year."""
    required = {"survey_year", "osteoporosis", "probability", "analysis_weight"}
    missing = sorted(required - set(predictions))
    if missing:
        raise ValueError("Year-specific metrics require: " + ", ".join(missing))
    rows = []
    for year, group in predictions.groupby("survey_year", sort=True):
        weight = group["analysis_weight"]
        prevalence = float(np.average(group["osteoporosis"], weights=weight))
        for label, threshold in operating_points.items():
            rows.append(
                {
                    "dataset": "knhanes_2008_2011",
                    "survey_year": int(year),
                    "evaluation": label,
                    "n": len(group),
                    "events": int(group["osteoporosis"].sum()),
                    "weighted_prevalence": prevalence,
                    **classification_metrics(
                        group["osteoporosis"],
                        group["probability"],
                        threshold=float(threshold),
                        sample_weight=weight,
                    ),
                }
            )
    return pd.DataFrame(rows)
