from __future__ import annotations

import numpy as np
import pandas as pd


def replace_special_missing(
    frame: pd.DataFrame, missing_codes_by_column: dict[str, list[int]]
) -> pd.DataFrame:
    """Replace codebook sentinels while retaining their non-response provenance.

    Ordinary nulls in NHANES skip-pattern fields often mean a structurally zero
    follow-up.  Refused/don't-know codes are different: after replacement they
    must remain distinguishable from those ordinary skips.
    """
    result = frame.copy()
    for column, codes in missing_codes_by_column.items():
        if column in result:
            result[f"{column}_nonresponse"] = result[column].isin(codes).astype("int8")
            result[column] = result[column].replace(codes, np.nan)
    return result


def _bounded_product(days: pd.Series, minutes: pd.Series, met: float) -> pd.Series:
    valid_days = days.where(days.between(0, 7))
    valid_minutes = minutes.where(minutes.between(0, 1440))
    return valid_days.fillna(0) * valid_minutes.fillna(0) * met


def _yes_no_category(source: pd.Series) -> pd.Series:
    return source.map({1: "yes", 2: "no"}).astype("object")


def _any_yes_all_no(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame]
    if not available:
        return pd.Series(np.nan, index=frame.index, dtype="object")
    values = frame[available]
    result = pd.Series(np.nan, index=frame.index, dtype="object")
    result.loc[values.eq(1).any(axis=1)] = "yes"
    result.loc[values.eq(2).all(axis=1)] = "no"
    return result


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    grip_trials = [
        column
        for column in (
            "grip_hand1_trial1_kg",
            "grip_hand1_trial2_kg",
            "grip_hand1_trial3_kg",
            "grip_hand2_trial1_kg",
            "grip_hand2_trial2_kg",
            "grip_hand2_trial3_kg",
        )
        if column in result
    ]
    if grip_trials:
        # Retained for sensitivity/audit. The CDC-derived two-hand sum remains
        # the pre-specified E9 predictor because a one-hand maximum is not on
        # the same scale as combined grip strength.
        result["maximum_grip_strength_kg"] = result[grip_trials].max(
            axis=1, skipna=True
        )

    if "diabetes_status_raw" in result:
        result["diabetes_status"] = result["diabetes_status_raw"].map(
            {1: "yes", 2: "no", 3: "borderline"}
        ).astype("object")

    activity_parts = []
    activity_nonresponse = pd.Series(False, index=result.index)
    for days, minutes, met in (
        ("vigorous_work_days", "vigorous_work_minutes", 8.0),
        ("moderate_work_days", "moderate_work_minutes", 4.0),
        ("walk_bike_days", "walk_bike_minutes", 4.0),
        ("vigorous_recreation_days", "vigorous_recreation_minutes", 8.0),
        ("moderate_recreation_days", "moderate_recreation_minutes", 4.0),
    ):
        if days in result and minutes in result:
            part = _bounded_product(result[days], result[minutes], met)
            pair_nonresponse = result.get(
                f"{days}_nonresponse", pd.Series(0, index=result.index)
            ).eq(1) | result.get(
                f"{minutes}_nonresponse", pd.Series(0, index=result.index)
            ).eq(1)
            part = part.mask(pair_nonresponse)
            activity_parts.append(part)
            activity_nonresponse |= pair_nonresponse
    if activity_parts:
        result["activity_met_minutes_week"] = sum(activity_parts)
        result["activity_nonresponse"] = activity_nonresponse.astype("int8")

    if {"alcohol_frequency_quantity", "alcohol_frequency_unit"}.issubset(result.columns):
        multipliers = result["alcohol_frequency_unit"].map({1: 7.0, 2: 1.0, 3: 12 / 52, 4: 1 / 52})
        frequency_week = result["alcohol_frequency_quantity"] * multipliers
        result["alcohol_drinks_week"] = frequency_week * result.get("drinks_per_drinking_day", 1)
        non_drinker = result.get(
            "alcohol_12plus_year", pd.Series(index=result.index, dtype=float)
        ).eq(2)
        result.loc[non_drinker, "alcohol_drinks_week"] = 0
        alcohol_sources = (
            "alcohol_12plus_year",
            "alcohol_frequency_quantity",
            "alcohol_frequency_unit",
            "drinks_per_drinking_day",
        )
        alcohol_nonresponse = pd.concat(
            [
                result.get(f"{column}_nonresponse", pd.Series(0, index=result.index))
                for column in alcohol_sources
            ],
            axis=1,
        ).eq(1).any(axis=1)
        result["alcohol_nonresponse"] = alcohol_nonresponse.astype("int8")
        result.loc[alcohol_nonresponse & ~non_drinker, "alcohol_drinks_week"] = np.nan

    if {"smoked_100_cigarettes", "current_smoking_frequency"}.issubset(result.columns):
        result["current_smoker"] = np.select(
            [
                result["smoked_100_cigarettes"].eq(2),
                result["current_smoking_frequency"].isin([1, 2]),
                result["current_smoking_frequency"].eq(3),
            ],
            ["never", "current", "former"],
            default=None,
        )

    if "household_smoker" in result:
        result["household_smoker"] = result["household_smoker"].map({1: "yes", 2: "no"})

    if {"age", "sex"}.issubset(result.columns):
        age_band = pd.cut(
            result["age"],
            [-np.inf, 64, 74, np.inf],
            labels=["under_65", "65_74", "75_plus"],
        ).astype("string")
        sex = result["sex"].astype("Int64").astype("string")
        result["age_sex_interaction"] = ("sex_" + sex + "__age_" + age_band).where(
            result["age"].notna() & result["sex"].notna(), np.nan
        )

    result["prior_fracture_history"] = _any_yes_all_no(
        result,
        [
            "fractured_hip_history",
            "fractured_wrist_history",
            "fractured_spine_history",
            "other_adult_fracture_history",
        ],
    )
    result["parental_hip_fracture"] = _any_yes_all_no(
        result, ["maternal_hip_fracture_raw", "paternal_hip_fracture_raw"]
    )
    result["long_term_glucocorticoid"] = _yes_no_category(
        result.get(
            "long_term_glucocorticoid_raw",
            pd.Series(np.nan, index=result.index),
        )
    )
    arthritis = result.get("arthritis_raw", pd.Series(np.nan, index=result.index))
    arthritis_type = result.get(
        "arthritis_type_raw", pd.Series(np.nan, index=result.index)
    )
    result["arthritis_history"] = _yes_no_category(arthritis)
    rheumatoid = pd.Series(np.nan, index=result.index, dtype="object")
    rheumatoid.loc[arthritis.eq(2)] = "no"
    rheumatoid.loc[arthritis.eq(1) & arthritis_type.notna()] = "no"
    rheumatoid.loc[arthritis.eq(1) & arthritis_type.eq(2)] = "yes"
    result["rheumatoid_arthritis"] = rheumatoid
    result["liver_condition"] = _yes_no_category(
        result.get("liver_condition_raw", pd.Series(np.nan, index=result.index))
    )
    result["thyroid_problem"] = _yes_no_category(
        result.get("thyroid_problem_raw", pd.Series(np.nan, index=result.index))
    )
    result["needs_walking_equipment"] = _yes_no_category(
        result.get("walking_equipment_raw", pd.Series(np.nan, index=result.index))
    )

    hormone = _yes_no_category(
        result.get("hormone_therapy_raw", pd.Series(np.nan, index=result.index))
    )
    male = result.get("sex", pd.Series(np.nan, index=result.index)).eq(1)
    hormone.loc[male] = "not_applicable_male"
    result["hormone_therapy_ever"] = hormone

    menopause = pd.Series(np.nan, index=result.index, dtype="object")
    female = result.get("sex", pd.Series(np.nan, index=result.index)).eq(2)
    periods = result.get("period_past_year_raw", pd.Series(np.nan, index=result.index))
    ovaries_removed = result.get("ovaries_removed_raw", pd.Series(np.nan, index=result.index))
    menopause.loc[male] = "not_applicable_male"
    menopause.loc[female & periods.eq(1)] = "menstruating"
    menopause.loc[female & ovaries_removed.eq(1)] = "surgical_menopause"
    menopause.loc[female & periods.eq(2) & ~ovaries_removed.eq(1)] = "no_period_past_year"
    result["menopausal_status"] = menopause
    safe_energy = result.get("energy_kcal", pd.Series(np.nan, index=result.index)).where(lambda x: x > 0)
    for source, target in (
        ("calcium_mg", "calcium_per_1000kcal"),
        ("vitamin_d_mcg", "vitamin_d_per_1000kcal"),
        ("magnesium_mg", "magnesium_per_1000kcal"),
    ):
        if source in result:
            result[target] = result[source] / safe_energy * 1000
    if {"protein_g", "weight_kg"}.issubset(result.columns):
        result["protein_per_kg"] = result["protein_g"] / result["weight_kg"].where(result["weight_kg"] > 0)
    if {"calcium_mg", "phosphorus_mg"}.issubset(result.columns):
        result["calcium_phosphorus_ratio"] = result["calcium_mg"] / result["phosphorus_mg"].where(result["phosphorus_mg"] > 0)
    if {"serum_albumin_g_dl", "alkaline_phosphatase_iu_l"}.issubset(result.columns):
        valid_alp = result["alkaline_phosphatase_iu_l"].where(
            result["alkaline_phosphatase_iu_l"] > 0
        )
        result["albumin_alkaline_phosphatase_ratio"] = (
            result["serum_albumin_g_dl"] / valid_alp
        )
    return result


def engineer_outcomes(frame: pd.DataFrame, reference: dict) -> pd.DataFrame:
    result = frame.copy()
    result["femoral_neck_t_score"] = (
        result["femoral_neck_bmd"] - reference["mean_bmd"]
    ) / reference["sd_bmd"]
    score = result["femoral_neck_t_score"]
    at_minus_one = np.isclose(score, -1.0, atol=1e-10, rtol=0)
    at_minus_two_five = np.isclose(score, -2.5, atol=1e-10, rtol=0)
    osteoporosis = score.lt(-2.5) | at_minus_two_five
    low_bmd = score.lt(-1.0) & ~at_minus_one
    category = pd.Series(
        np.select(
            [osteoporosis, low_bmd],
            ["osteoporosis", "low_bone_mass"],
            default="normal",
        ),
        index=result.index,
        dtype="object",
    )
    category.loc[score.isna()] = np.nan
    result["bone_density_class"] = pd.Categorical(
        category, categories=["normal", "low_bone_mass", "osteoporosis"]
    )
    result["low_bmd_or_osteoporosis"] = low_bmd.astype("Int64")
    result["osteoporosis"] = osteoporosis.astype("Int64")
    result.loc[score.isna(), ["low_bmd_or_osteoporosis", "osteoporosis"]] = pd.NA
    return result


def build_cohorts(
    frame: pd.DataFrame, classification_min_age: int, regression_min_age: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use documented status=1 and non-null BMD; retain invalidity codes for audit."""
    flow: list[dict[str, int | str]] = [{"stage": "merged_dxa_anchor", "participants": len(frame)}]
    valid_femur = frame["femur_scan_status"].eq(1) & frame["femoral_neck_bmd"].notna()
    flow.append({"stage": "valid_femoral_neck_dxa", "participants": int(valid_femur.sum())})

    classification = frame.loc[valid_femur & frame["age"].ge(classification_min_age)].copy()
    flow.append({"stage": f"classification_age_{classification_min_age}_plus", "participants": len(classification)})
    classification["reliable_day1_diet"] = classification["dietary_recall_status"].eq(1)
    flow.append(
        {
            "stage": "classification_reliable_day1_diet_available",
            "participants": int(classification["reliable_day1_diet"].sum()),
        }
    )

    regression = frame.loc[valid_femur & frame["age"].ge(regression_min_age)].copy()
    regression = regression.loc[regression["dietary_recall_status"].eq(1)].copy()
    flow.append({"stage": f"regression_age_{regression_min_age}_plus_reliable_diet", "participants": len(regression)})
    return classification, regression, pd.DataFrame(flow)
