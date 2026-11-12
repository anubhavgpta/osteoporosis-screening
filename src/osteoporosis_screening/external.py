from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .config import load_yaml
from .evaluation import classification_metrics


@dataclass(frozen=True)
class FrozenExternalAudit:
    """Outputs from a prediction-only evaluation of a frozen NHANES model."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    tradeoffs: pd.DataFrame
    calibration: pd.DataFrame
    feature_audit: pd.DataFrame
    cohort_flow: pd.DataFrame
    manifest: dict


def _age_sex_interaction(frame: pd.DataFrame) -> pd.Series:
    age_band = pd.cut(
        frame["age"],
        [-np.inf, 64, 74, np.inf],
        labels=["under_65", "65_74", "75_plus"],
    ).astype("string")
    sex = frame["sex"].astype("Int64").astype("string")
    return ("sex_" + sex + "__age_" + age_band).where(
        frame["age"].notna() & frame["sex"].notna()
    )


def _read_external_frame(input_path: Path) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(f"External input does not exist: {input_path}")
    if input_path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(input_path)
    return pd.read_csv(input_path)


def _validate_reference(root: Path, catalog_reference: dict) -> None:
    """Prevent the external endpoint from silently drifting from model training."""
    config_path = root / "configs" / "model_config.yaml"
    if not config_path.exists():
        return
    model_reference = load_yaml(config_path)["t_score_reference"]
    matches = np.isclose(
        [catalog_reference["reference_mean_bmd"], catalog_reference["reference_sd_bmd"]],
        [model_reference["mean_bmd"], model_reference["sd_bmd"]],
    ).all()
    if not matches:
        raise ValueError(
            "External and training T-score references differ. External validation must "
            "use the frozen NHANES endpoint reference."
        )


def _cohort_flow(frame: pd.DataFrame, minimum_age: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.copy()
    working["age"] = pd.to_numeric(working["age"], errors="coerce")
    working["femoral_neck_bmd"] = pd.to_numeric(
        working["femoral_neck_bmd"], errors="coerce"
    )
    rows = [{"step": "input", "n": len(working), "excluded_at_step": 0}]

    age_eligible = working["age"].ge(minimum_age)
    rows.append(
        {
            "step": f"age_greater_than_or_equal_to_{minimum_age:g}",
            "n": int(age_eligible.sum()),
            "excluded_at_step": int((~age_eligible).sum()),
        }
    )
    working = working.loc[age_eligible].copy()

    has_bmd = working["femoral_neck_bmd"].notna()
    rows.append(
        {
            "step": "observed_femoral_neck_bmd",
            "n": int(has_bmd.sum()),
            "excluded_at_step": int((~has_bmd).sum()),
        }
    )
    working = working.loc[has_bmd].copy()
    rows.append(
        {
            "step": "external_evaluation_cohort",
            "n": len(working),
            "excluded_at_step": 0,
        }
    )
    return working, pd.DataFrame(rows)


def _analysis_weight(frame: pd.DataFrame) -> pd.Series:
    if "external_weight" not in frame:
        return pd.Series(1.0, index=frame.index, name="analysis_weight")
    weight = pd.to_numeric(frame["external_weight"], errors="coerce")
    invalid = weight.isna() | ~np.isfinite(weight) | weight.le(0)
    if invalid.any():
        raise ValueError(
            f"external_weight must be finite and positive for every eligible row; "
            f"found {int(invalid.sum())} invalid value(s)"
        )
    return weight.rename("analysis_weight")


def _model_categorical_levels(model) -> dict[str, set]:
    """Recover category vocabularies from fitted calibrated-pipeline clones."""
    calibrated = getattr(model, "calibrated_classifiers_", None)
    estimators = [item.estimator for item in calibrated] if calibrated else [model]
    expected: dict[str, set] = {}
    for estimator in estimators:
        preprocess = getattr(estimator, "named_steps", {}).get("preprocess")
        for name, transformer, columns in getattr(preprocess, "transformers_", []):
            if name != "categorical":
                continue
            encoder = next(
                (
                    step
                    for step in getattr(transformer, "named_steps", {}).values()
                    if hasattr(step, "categories_")
                ),
                None,
            )
            if encoder is None:
                continue
            for column, levels in zip(columns, encoder.categories_, strict=False):
                expected.setdefault(column, set()).update(
                    level for level in levels if not pd.isna(level)
                )
    return expected


def _feature_audit(
    frame: pd.DataFrame,
    features: list[str],
    expected_categories: dict[str, set],
) -> pd.DataFrame:
    rows = []
    for column in [*features, "femoral_neck_bmd"]:
        series = frame[column]
        observed = set(series.dropna().unique())
        unknown = observed - expected_categories.get(column, observed)
        rows.append(
            {
                "column": column,
                "role": "model_feature" if column in features else "outcome_source",
                "dtype": str(series.dtype),
                "n": len(series),
                "missing_n": int(series.isna().sum()),
                "missing_percent": float(series.isna().mean() * 100),
                "unique_nonmissing": int(series.nunique(dropna=True)),
                "unknown_category_n": len(unknown),
                "unknown_categories": json.dumps(
                    sorted((str(value) for value in unknown)), ensure_ascii=False
                ),
            }
        )
    return pd.DataFrame(rows)


def _calibration_table(
    y_true: pd.Series,
    probability: pd.Series,
    weight: pd.Series,
    bins: int = 10,
) -> pd.DataFrame:
    available_bins = min(bins, int(probability.nunique()))
    if available_bins < 1:
        return pd.DataFrame()
    groups = pd.qcut(probability, q=available_bins, duplicates="drop")
    data = pd.DataFrame(
        {"outcome": y_true.to_numpy(), "probability": probability.to_numpy(),
         "weight": weight.to_numpy(), "bin": groups.astype("string").to_numpy()}
    )
    rows = []
    for label, group in data.groupby("bin", sort=False, observed=True):
        rows.append(
            {
                "bin": label,
                "n": len(group),
                "weight_sum": float(group["weight"].sum()),
                "mean_predicted_probability": float(
                    np.average(group["probability"], weights=group["weight"])
                ),
                "observed_prevalence": float(
                    np.average(group["outcome"], weights=group["weight"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _tradeoff_table(
    y_true: pd.Series,
    probability: pd.Series,
    weight: pd.Series,
    recall_targets: list[float],
    precision_targets: list[float],
    reference_prevalence: float | None = None,
) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(
        y_true, probability, sample_weight=weight
    )
    # The final PR-curve point has no corresponding threshold.
    precision = precision[:-1]
    recall = recall[:-1]
    rows = []

    def add_row(measure: str, target: float, index: int) -> None:
        threshold = float(thresholds[index])
        operating = classification_metrics(
            y_true, probability, threshold=threshold, sample_weight=weight
        )
        row = {
            "measure": measure,
            "target": float(target),
            "achieved_precision": float(precision[index]),
            "achieved_recall": float(recall[index]),
            "achieved_specificity": float(operating["specificity"]),
            "diagnostic_threshold": threshold,
        }
        if reference_prevalence is not None:
            sensitivity = float(operating["sensitivity"])
            specificity = float(operating["specificity"])
            numerator = sensitivity * reference_prevalence
            denominator = numerator + (1 - specificity) * (1 - reference_prevalence)
            row["reference_prevalence"] = reference_prevalence
            row["precision_standardized_to_reference_prevalence"] = (
                numerator / denominator if denominator else np.nan
            )
        rows.append(row)

    for target in recall_targets:
        eligible = np.flatnonzero(recall >= target)
        if len(eligible):
            best_precision = precision[eligible].max()
            candidates = eligible[np.isclose(precision[eligible], best_precision)]
            index = candidates[np.argmin(recall[candidates] - target)]
            add_row("precision_at_recall", target, index)
    for target in precision_targets:
        eligible = np.flatnonzero(precision >= target)
        if len(eligible):
            best_recall = recall[eligible].max()
            candidates = eligible[np.isclose(recall[eligible], best_recall)]
            index = candidates[np.argmax(precision[candidates])]
            add_row("recall_at_precision", target, index)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["threshold_status"] = (
            "external diagnostic only; never use as a deployment threshold"
        )
    return result


def audit_external_cohort(root: Path, dataset: str, input_path: Path) -> FrozenExternalAudit:
    """Evaluate a harmonized cohort without fitting, calibrating, or tuning the model."""
    root = root.resolve()
    input_path = input_path.resolve()
    catalog = load_yaml(root / "configs" / "external_datasets.yaml")
    if dataset not in catalog["datasets"]:
        available = ", ".join(sorted(catalog["datasets"]))
        raise ValueError(f"Unknown external dataset {dataset!r}; choose one of: {available}")

    reference = catalog["harmonized_outcome"]
    _validate_reference(root, reference)
    metadata_path = root / "artifacts" / "classification_model_metadata.json"
    model_path = root / "artifacts" / "classification_model.joblib"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = list(metadata["features"])

    frame = _read_external_frame(input_path)
    required = set(features) | {"age", "sex", "femoral_neck_bmd"}
    # The interaction is deterministically reconstructed from age and sex.
    required.discard("age_sex_interaction")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "External export is not harmonized; missing required columns: "
            + ", ".join(missing)
        )

    frame, cohort_flow = _cohort_flow(frame, minimum_age=50)
    if frame.empty:
        raise ValueError("No participants remain after the age and femoral-neck BMD criteria")
    weight = _analysis_weight(frame)
    sex_numeric = pd.to_numeric(frame["sex"], errors="coerce")
    invalid_sex = sex_numeric.notna() & ~sex_numeric.isin([1, 2])
    if invalid_sex.any():
        raise ValueError("sex must use frozen NHANES coding: 1=male and 2=female")
    frame["sex"] = sex_numeric
    if "age_sex_interaction" not in frame:
        frame["age_sex_interaction"] = _age_sex_interaction(frame)

    frame["femoral_neck_t_score"] = (
        frame["femoral_neck_bmd"] - reference["reference_mean_bmd"]
    ) / reference["reference_sd_bmd"]
    frame["osteoporosis"] = frame["femoral_neck_t_score"].le(-2.5).astype(int)
    if frame["osteoporosis"].nunique() < 2:
        raise ValueError("External cohort must contain both osteoporosis outcome classes")

    model = joblib.load(model_path)
    feature_audit = _feature_audit(frame, features, _model_categorical_levels(model))
    probability = np.asarray(model.predict_proba(frame[features])[:, 1], dtype=float)
    if len(probability) != len(frame) or not np.isfinite(probability).all():
        raise ValueError("Frozen model returned missing or non-finite probabilities")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Frozen model returned probabilities outside [0, 1]")
    frame["probability"] = probability
    frame["analysis_weight"] = weight.to_numpy()

    operating_points = metadata.get(
        "operating_points",
        {"default": 0.5,
         "sensitivity_oriented": metadata["sensitivity_oriented_threshold"]},
    )
    invalid_thresholds = {
        label: threshold
        for label, threshold in operating_points.items()
        if not 0 <= float(threshold) <= 1
    }
    if invalid_thresholds:
        raise ValueError(f"Frozen metadata contains invalid thresholds: {invalid_thresholds}")
    prediction_columns_at_threshold = []
    for label, threshold in operating_points.items():
        column = "prediction__" + str(label).replace(".", "_")
        frame[column] = frame["probability"].ge(float(threshold)).astype("int8")
        prediction_columns_at_threshold.append(column)

    raw_prevalence = float(frame["osteoporosis"].mean())
    weighted_prevalence = float(np.average(frame["osteoporosis"], weights=weight))
    rows = []
    for label, threshold in operating_points.items():
        rows.append(
            {
                "dataset": dataset,
                "evaluation": label,
                "evaluation_mode": "frozen_model_frozen_nhanes_threshold",
                "threshold_origin": "classification_model_metadata.json",
                "target": "osteoporosis",
                "n": len(frame),
                "events": int(frame["osteoporosis"].sum()),
                "raw_prevalence": raw_prevalence,
                "weighted_prevalence": weighted_prevalence,
                **classification_metrics(
                    frame["osteoporosis"],
                    frame["probability"],
                    threshold=float(threshold),
                    sample_weight=weight,
                ),
            }
        )
    metrics = pd.DataFrame(rows)

    model_config_path = root / "configs" / "model_config.yaml"
    model_config = load_yaml(model_config_path) if model_config_path.exists() else {}
    tradeoffs = _tradeoff_table(
        frame["osteoporosis"],
        frame["probability"],
        weight,
        recall_targets=list(model_config.get("screening_recall_targets", [0.9, 0.8, 0.7])),
        precision_targets=list(
            model_config.get("screening_precision_targets", [0.2, 0.25, 0.3])
        ),
        reference_prevalence=reference.get("comparison_prevalence"),
    )
    if not tradeoffs.empty:
        tradeoffs.insert(0, "dataset", dataset)
    calibration = _calibration_table(
        frame["osteoporosis"], frame["probability"], weight
    )
    if not calibration.empty:
        calibration.insert(0, "dataset", dataset)

    prediction_columns = [
        *(column for column in ("participant_id",) if column in frame),
        *(column for column in ("survey_year",) if column in frame),
        "osteoporosis",
        "femoral_neck_bmd",
        "femoral_neck_t_score",
        "probability",
        "analysis_weight",
        *prediction_columns_at_threshold,
    ]
    predictions = frame[prediction_columns].copy()
    manifest = {
        "dataset": dataset,
        "evaluation_mode": "prediction_only_frozen_external_validation",
        "model_artifact": model_path.name,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_metadata": metadata_path.name,
        "model_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "model_feature_set": metadata.get("feature_set"),
        "n_model_features": len(features),
        "input_file": input_path.name,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "input_n": int(cohort_flow.iloc[0]["n"]),
        "evaluation_n": len(frame),
        "raw_prevalence": raw_prevalence,
        "weighted_prevalence": weighted_prevalence,
        "features_with_unknown_categories": int(
            feature_audit["unknown_category_n"].gt(0).sum()
        ),
        "outcome_definition": metadata.get(
            "positive_class_definition", "femoral-neck T-score <= -2.5"
        ),
        "reference_mean_bmd": reference["reference_mean_bmd"],
        "reference_sd_bmd": reference["reference_sd_bmd"],
        "threshold_policy": (
            "All reported operating metrics use thresholds frozen in NHANES metadata. "
            "Tradeoff thresholds are descriptive external-cohort diagnostics only."
        ),
        "model_fitted_on_external_data": False,
        "deployment_threshold_selected_on_external_data": False,
    }
    return FrozenExternalAudit(
        metrics=metrics,
        predictions=predictions,
        tradeoffs=tradeoffs,
        calibration=calibration,
        feature_audit=feature_audit,
        cohort_flow=cohort_flow,
        manifest=manifest,
    )


def evaluate_external_cohort(
    root: Path,
    dataset: str,
    input_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible metrics/predictions interface."""
    audit = audit_external_cohort(root, dataset, input_path)
    return audit.metrics, audit.predictions
