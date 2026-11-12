from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import classification_metrics, decision_curve, regression_metrics


def write_json(payload: dict, path: Path) -> None:
    def convert(value):
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if pd.isna(value):
            return None
        raise TypeError(f"Cannot serialize {type(value)}")

    path.write_text(json.dumps(payload, indent=2, default=convert), encoding="utf-8")


def weighted_prevalence(
    frame: pd.DataFrame, target: str = "osteoporosis"
) -> pd.DataFrame:
    outcome = frame[target].astype(float)
    weight = frame["mec_weight"]
    valid = outcome.notna() & weight.notna() & weight.gt(0)
    prevalence = np.average(outcome[valid], weights=weight[valid])
    return pd.DataFrame(
        [
            {
                "target": target,
                "raw_n": int(valid.sum()),
                "raw_prevalence": outcome[valid].mean(),
                "survey_weighted_prevalence": prevalence,
            }
        ]
    )


def missingness_mechanism_summary(frame: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "skip_pattern": {
            "columns": [
                "vigorous_work_days",
                "vigorous_work_minutes",
                "moderate_work_days",
                "moderate_work_minutes",
                "walk_bike_days",
                "walk_bike_minutes",
                "vigorous_recreation_days",
                "vigorous_recreation_minutes",
                "moderate_recreation_days",
                "moderate_recreation_minutes",
            ],
            "strategy": "structural skips become zero in derived activity; sentinel non-response remains missing and flagged",
        },
        "subsample_by_design": {
            "columns": ["household_smoker", "household_smoker_count", "household_smoke_days"],
            "strategy": "excluded from predictive feature sets; retained for audit only",
        },
        "exam_limitation_or_structural": {
            "columns": [
                "total_spine_bmd",
                "spine_invalidity_code",
                "weight_kg",
                "height_cm",
                "bmi",
            ],
            "strategy": "multivariate imputation plus missingness indicator; report outcome association",
        },
        "item_nonresponse": {
            "columns": ["poverty_income_ratio", "alcohol_drinks_week"],
            "strategy": "multivariate iterative imputation plus missingness indicator",
        },
        "cycle_specific_exam_or_lab": {
            "columns": [
                "combined_grip_strength_kg",
                "maximum_grip_strength_kg",
                "alkaline_phosphatase_iu_l",
                "serum_calcium_mg_dl",
                "serum_total_folate_nmol_l",
                "serum_25oh_vitamin_d_nmol_l",
            ],
            "strategy": "analyze only in the pre-specified 2013-2014 cohort; impute within training folds",
        },
        "published_panel_exam_or_lab": {
            "columns": [
                "diabetes_status",
                "total_cholesterol_mg_dl",
                "monocyte_percent",
                "segmented_neutrophils_percent",
                "mean_cell_volume_fl",
                "red_cell_distribution_width_percent",
                "glycohemoglobin_percent",
                "creatine_phosphokinase_iu_l",
                "globulin_g_dl",
                "osmolality_mmol_kg",
                "total_protein_g_dl",
                "uric_acid_mg_dl",
            ],
            "strategy": "evaluate in cycles with complete component coverage; impute within training folds",
        },
    }
    rows = []
    for mechanism, definition in definitions.items():
        for column in definition["columns"]:
            if column in frame:
                rows.append(
                    {
                        "column": column,
                        "missing_percent": frame[column].isna().mean() * 100,
                        "mechanism": mechanism,
                        "pipeline_strategy": definition["strategy"],
                    }
                )
    return pd.DataFrame(rows).sort_values(["mechanism", "missing_percent"], ascending=[True, False])


def missingness_outcome_bias(
    frame: pd.DataFrame, target: str = "osteoporosis"
) -> pd.DataFrame:
    outcome = frame[target].astype(float)
    weight = frame["mec_weight"].fillna(1.0)
    columns = [
        "total_spine_bmd",
        "spine_invalidity_code",
        "weight_kg",
        "height_cm",
        "bmi",
        "poverty_income_ratio",
        "alcohol_drinks_week",
        "combined_grip_strength_kg",
        "maximum_grip_strength_kg",
        "alkaline_phosphatase_iu_l",
        "serum_calcium_mg_dl",
        "serum_total_folate_nmol_l",
        "serum_25oh_vitamin_d_nmol_l",
        "diabetes_status",
        "total_cholesterol_mg_dl",
        "monocyte_percent",
        "segmented_neutrophils_percent",
        "mean_cell_volume_fl",
        "red_cell_distribution_width_percent",
        "glycohemoglobin_percent",
        "creatine_phosphokinase_iu_l",
        "globulin_g_dl",
        "osmolality_mmol_kg",
        "total_protein_g_dl",
        "uric_acid_mg_dl",
    ]
    rows = []
    for column in columns:
        if column not in frame:
            continue
        missing = frame[column].isna()
        valid_target = outcome.notna() & weight.gt(0)

        def prevalence(mask):
            selected = valid_target & mask
            if not selected.any():
                return np.nan
            return np.average(outcome[selected], weights=weight[selected])

        missing_prevalence = prevalence(missing)
        observed_prevalence = prevalence(~missing)
        rows.append(
            {
                "target": target,
                "column": column,
                "missing_n": int((valid_target & missing).sum()),
                "observed_n": int((valid_target & ~missing).sum()),
                "weighted_prevalence_when_missing": missing_prevalence,
                "weighted_prevalence_when_observed": observed_prevalence,
                "weighted_risk_difference_missing_minus_observed": (
                    missing_prevalence - observed_prevalence
                ),
            }
        )
    return pd.DataFrame(rows)


def _resample_psus_within_strata(data: pd.DataFrame, rng) -> pd.DataFrame:
    sampled = []
    valid = data.dropna(subset=["survey_stratum", "survey_psu"])
    for _, stratum in valid.groupby("survey_stratum", observed=True):
        psus = stratum["survey_psu"].dropna().unique()
        if not len(psus):
            continue
        for psu in rng.choice(psus, size=len(psus), replace=True):
            sampled.append(stratum.loc[stratum["survey_psu"].eq(psu)])
    if not sampled:
        return data.iloc[0:0]
    return pd.concat(sampled, ignore_index=True)


def classification_bootstrap_intervals(
    predictions: pd.DataFrame,
    thresholds: dict[str, float],
    iterations: int,
    seed: int,
    target: str = "osteoporosis",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics_to_report = (
        "roc_auc",
        "pr_auc",
        "accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "balanced_accuracy",
        "mcc",
        "log_loss",
        "brier_score",
        "brier_skill_score",
        "calibration_intercept",
        "calibration_slope",
    )
    rows = []
    y = predictions[target].astype(int)
    weight = predictions["analysis_weight"]
    for evaluation, threshold in thresholds.items():
        estimate = classification_metrics(y, predictions["probability"], threshold, weight)
        draws = {metric: [] for metric in metrics_to_report}
        for _ in range(iterations):
            sample = _resample_psus_within_strata(predictions, rng)
            if sample.empty or sample[target].nunique() < 2:
                continue
            sample_weight = sample["analysis_weight"]
            result = classification_metrics(
                sample[target].astype(int),
                sample["probability"],
                threshold,
                sample_weight,
            )
            for metric in metrics_to_report:
                draws[metric].append(result[metric])
        for metric in metrics_to_report:
            values = np.asarray(draws[metric], dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "evaluation": evaluation,
                    "metric": metric,
                    "estimate": estimate[metric],
                    "ci_low": np.quantile(values, 0.025) if len(values) else np.nan,
                    "ci_high": np.quantile(values, 0.975) if len(values) else np.nan,
                    "successful_bootstrap_iterations": len(values),
                    "requested_bootstrap_iterations": iterations,
                    "resampling_design": "PSUs resampled with replacement within survey strata",
                }
            )
    return pd.DataFrame(rows)


def regression_bootstrap_intervals(
    predictions: pd.DataFrame, target: str, iterations: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics_to_report = ("mae", "rmse", "r2")
    weight = predictions["dietary_day1_weight"].fillna(predictions["mec_weight"])
    estimate = regression_metrics(predictions[target], predictions["predicted_bmd"], weight)
    draws = {metric: [] for metric in metrics_to_report}
    for _ in range(iterations):
        sample = _resample_psus_within_strata(predictions, rng)
        if sample.empty:
            continue
        sample_weight = sample["dietary_day1_weight"].fillna(sample["mec_weight"])
        result = regression_metrics(sample[target], sample["predicted_bmd"], sample_weight)
        for metric in metrics_to_report:
            draws[metric].append(result[metric])
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "estimate": estimate[metric],
                "ci_low": np.quantile(draws[metric], 0.025) if draws[metric] else np.nan,
                "ci_high": np.quantile(draws[metric], 0.975) if draws[metric] else np.nan,
                "successful_bootstrap_iterations": len(draws[metric]),
                "requested_bootstrap_iterations": iterations,
                "resampling_design": "PSUs resampled with replacement within survey strata",
            }
            for metric in metrics_to_report
        ]
    )


def subgroup_metrics(
    predictions: pd.DataFrame,
    thresholds: pd.DataFrame,
    target: str = "osteoporosis",
) -> pd.DataFrame:
    data = predictions.copy()
    data["age_group"] = pd.cut(
        data["age"], [49, 64, 74, np.inf], labels=["50-64", "65-74", "75+"]
    )
    data["age_sex_group"] = data["sex"].astype("string") + "|" + data[
        "age_group"
    ].astype("string")
    global_threshold = float(
        thresholds.loc[thresholds["dimension"].eq("global"), "threshold"].iloc[0]
    )
    rows = []
    for dimension in (
        "sex",
        "age_group",
        "age_sex_group",
        "race_ethnicity",
        "survey_cycle",
    ):
        for group, subset in data.groupby(dimension, observed=True, dropna=False):
            if len(subset) < 20 or subset[target].nunique() < 2:
                continue
            match = thresholds.loc[
                thresholds["dimension"].eq(dimension)
                & thresholds["group"].eq(str(group)),
                "threshold",
            ]
            threshold = float(match.iloc[0]) if len(match) else global_threshold
            metrics = classification_metrics(
                subset[target].astype(int),
                subset["probability"],
                threshold,
                subset["analysis_weight"],
            )
            rows.append(
                {
                    "dimension": dimension,
                    "group": str(group),
                    "n": len(subset),
                    "prevalence": np.average(
                        subset[target], weights=subset["analysis_weight"]
                    ),
                    "operating_threshold": threshold,
                    "threshold_source": "group_oof" if len(match) else "global_oof_fallback",
                    "unstable_n_below_100": len(subset) < 100,
                    **{key: metrics[key] for key in ("roc_auc", "sensitivity", "specificity", "brier_score")},
                }
            )
    return pd.DataFrame(rows)


def save_classification_plot(
    predictions: pd.DataFrame,
    path: Path,
    target: str = "osteoporosis",
) -> None:
    from sklearn.metrics import RocCurveDisplay

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    RocCurveDisplay.from_predictions(
        predictions[target].astype(int),
        predictions["probability"],
        sample_weight=predictions["analysis_weight"],
        name="Survey-weighted classifier",
        ax=ax,
    )
    ax.set_title("Held-out low-BMD screening ROC curve")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def calibration_table(
    predictions: pd.DataFrame,
    target: str = "osteoporosis",
    bins: int = 10,
) -> pd.DataFrame:
    data = predictions[[target, "probability", "analysis_weight"]].dropna().copy()
    data["bin"] = pd.qcut(
        data["probability"].rank(method="first"), q=min(bins, len(data)), labels=False
    )
    rows = []
    for bin_number, subset in data.groupby("bin", observed=True):
        weight = subset["analysis_weight"]
        rows.append(
            {
                "bin": int(bin_number) + 1,
                "n": len(subset),
                "weight_sum": weight.sum(),
                "mean_predicted_probability": np.average(
                    subset["probability"], weights=weight
                ),
                "observed_prevalence": np.average(subset[target], weights=weight),
            }
        )
    return pd.DataFrame(rows)


def save_calibration_plot(
    calibration: pd.DataFrame, path: Path, title: str = "Held-out calibration"
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Ideal")
    ax.plot(
        calibration["mean_predicted_probability"],
        calibration["observed_prevalence"],
        marker="o",
        label="Model",
    )
    ax.set(xlabel="Predicted probability", ylabel="Observed prevalence", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def decision_curve_table(
    predictions: pd.DataFrame,
    thresholds,
    target: str = "osteoporosis",
) -> pd.DataFrame:
    return decision_curve(
        predictions[target].astype(int),
        predictions["probability"],
        thresholds,
        predictions["analysis_weight"],
    )


def save_decision_curve_plot(
    curve: pd.DataFrame, path: Path, title: str = "Held-out decision curve"
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(curve["threshold_probability"], curve["model_net_benefit"], label="Model")
    ax.plot(curve["threshold_probability"], curve["test_all_net_benefit"], label="Test all")
    ax.plot(curve["threshold_probability"], curve["test_none_net_benefit"], label="Test none")
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_regression_plot(predictions: pd.DataFrame, target: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(predictions[target], predictions["predicted_bmd"], alpha=0.55, s=18)
    limits = [
        min(predictions[target].min(), predictions["predicted_bmd"].min()),
        max(predictions[target].max(), predictions["predicted_bmd"].max()),
    ]
    ax.plot(limits, limits, "--", color="black", linewidth=1)
    ax.set(xlabel="Observed BMD", ylabel="Predicted BMD", title="Held-out BMD regression")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
