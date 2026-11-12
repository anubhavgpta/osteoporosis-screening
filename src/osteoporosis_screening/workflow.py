from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd

from .attribution import write_shap_attributions
from .config import ProjectPaths, load_yaml
from .evaluation import classification_metrics
from .features import (
    build_cohorts,
    engineer_features,
    engineer_outcomes,
    replace_special_missing,
)
from .ingest import load_and_merge
from .modeling import (
    feature_experiments,
    run_classification,
    run_classification_ablations,
    run_regression,
    run_regression_ablations,
)
from .nutrition import NUTRITION_CLUSTER_FEATURES, NutritionClusterAugmenter
from .reporting import (
    calibration_table,
    classification_bootstrap_intervals,
    decision_curve_table,
    missingness_mechanism_summary,
    missingness_outcome_bias,
    regression_bootstrap_intervals,
    save_classification_plot,
    save_calibration_plot,
    save_decision_curve_plot,
    save_regression_plot,
    subgroup_metrics,
    weighted_prevalence,
    write_json,
)


def prepare_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, ProjectPaths]:
    paths = ProjectPaths.from_root(root)
    paths.ensure()
    variables = load_yaml(paths.root / "configs" / "variables.yaml")
    model_config = load_yaml(paths.root / "configs" / "model_config.yaml")
    merged, merge_flow = load_and_merge(paths.raw, variables)
    cleaned = replace_special_missing(merged, variables["missing_codes_by_column"])
    featured = engineer_features(cleaned)
    featured = engineer_outcomes(featured, model_config["t_score_reference"])
    classification_target = model_config.get("classification_target", "osteoporosis")
    classification_all, regression, eligibility_flow = build_cohorts(
        featured,
        model_config["classification_min_age"],
        model_config["regression_min_age"],
    )
    production_cycles = model_config.get("production_classification_cycles")
    if production_cycles:
        classification = classification_all.loc[
            classification_all["survey_cycle"].isin(production_cycles)
        ].copy()
        pooled_cycle_count = len(variables["cycles"])
        production_cycle_count = len(production_cycles)
        weight_rescale = pooled_cycle_count / production_cycle_count
        classification["mec_weight"] *= weight_rescale
        classification["dietary_day1_weight"] *= weight_rescale
    else:
        classification = classification_all
    candidate_expansion = classification_all.loc[
        ~classification_all.index.isin(classification.index)
    ].copy()
    eligibility_flow = pd.concat(
        [
            eligibility_flow,
            pd.DataFrame(
                [
                    {
                        "stage": "classification_production_cycles",
                        "participants": len(classification),
                    },
                    {
                        "stage": "classification_candidate_expansion_cycles",
                        "participants": len(candidate_expansion),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    merge_flow.to_csv(paths.tables / "merge_audit.csv", index=False)
    eligibility_flow.to_csv(paths.tables / "cohort_flow.csv", index=False)
    missingness = (
        classification.isna().mean().mul(100).sort_values(ascending=False).rename("missing_percent")
    )
    missingness.to_csv(paths.tables / "classification_missingness.csv", header=True)
    missingness_mechanism_summary(classification).to_csv(
        paths.tables / "classification_missingness_mechanisms.csv", index=False
    )
    missingness_outcome_bias(classification, target=classification_target).to_csv(
        paths.tables / "classification_missingness_outcome_bias.csv", index=False
    )
    classification.to_csv(paths.processed / "classification_cohort.csv.gz", index=False)
    classification_all.to_csv(
        paths.processed / "classification_all_cycles_cohort.csv.gz", index=False
    )
    candidate_expansion.to_csv(
        paths.processed / "classification_candidate_expansion_cohort.csv.gz",
        index=False,
    )
    regression.to_csv(paths.processed / "regression_cohort.csv.gz", index=False)
    weighted_prevalence(classification, target=classification_target).to_csv(
        paths.tables / "weighted_prevalence.csv", index=False
    )
    return classification, regression, variables, model_config, paths


def _all_features(variables: dict) -> list[str]:
    groups = variables["feature_groups"]
    ordered = groups["demographic"] + groups["anthropometric"] + groups["lifestyle"] + groups["nutrition"]
    return list(dict.fromkeys(ordered))


def _targeted_feature_columns(variables: dict) -> list[str]:
    groups = variables["feature_groups"]
    return list(dict.fromkeys(groups.get("grip", []) + groups.get("biomarkers", [])))


def _ready_columns(frame: pd.DataFrame, columns: list[str]) -> bool:
    return bool(columns) and all(
        column in frame and frame[column].notna().any() for column in columns
    )


def _write_classification_evaluation(
    result,
    paths: ProjectPaths,
    config: dict,
    prefix: str,
    target: str = "osteoporosis",
) -> None:
    result.metrics.to_csv(paths.tables / f"{prefix}_metrics.csv", index=False)
    result.comparison.to_csv(paths.tables / f"{prefix}_model_comparison.csv", index=False)
    result.predictions.to_csv(paths.tables / f"{prefix}_test_predictions.csv", index=False)
    result.thresholds.to_csv(paths.tables / f"{prefix}_operating_thresholds.csv", index=False)
    operating_points = {"default": 0.5}
    for row in result.thresholds.loc[
        result.thresholds["dimension"].isin(["global", "global_tradeoff"])
    ].itertuples(index=False):
        label = "sensitivity_oriented" if row.dimension == "global" else str(row.group)
        operating_points[label] = float(row.threshold)
    classification_bootstrap_intervals(
        result.predictions,
        operating_points,
        config["bootstrap_iterations"],
        config["seed"],
        target=target,
    ).to_csv(paths.tables / f"{prefix}_metric_intervals.csv", index=False)
    weight = (
        result.predictions["analysis_weight"]
        if "analysis_weight" in result.predictions
        else None
    )
    operating_metrics = []
    for label, operating_threshold in operating_points.items():
        operating_metrics.append(
            {
                "operating_point": label,
                **classification_metrics(
                    result.predictions[target],
                    result.predictions["probability"],
                    threshold=operating_threshold,
                    sample_weight=weight,
                ),
            }
        )
    pd.DataFrame(operating_metrics).to_csv(
        paths.tables / f"{prefix}_operating_point_metrics.csv", index=False
    )
    subgroup_metrics(result.predictions, result.thresholds, target=target).to_csv(
        paths.tables / f"{prefix}_subgroups.csv", index=False
    )
    calibration = calibration_table(result.predictions, target=target)
    calibration.to_csv(paths.tables / f"{prefix}_calibration.csv", index=False)
    save_calibration_plot(
        calibration,
        paths.figures / f"{prefix}_calibration.png",
        title=f"{prefix.replace('_', ' ').title()} calibration",
    )
    curve = decision_curve_table(
        result.predictions,
        config["decision_curve_thresholds"],
        target=target,
    )
    curve.to_csv(paths.tables / f"{prefix}_decision_curve.csv", index=False)
    save_decision_curve_plot(
        curve,
        paths.figures / f"{prefix}_decision_curve.png",
        title=f"{prefix.replace('_', ' ').title()} decision curve",
    )
    save_classification_plot(
        result.predictions,
        paths.figures / f"{prefix}_roc.png",
        target=target,
    )


def run_all(root: Path) -> dict:
    classification, regression, variables, config, paths = prepare_data(root)
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(paths.root / ".joblib_tmp"))
    classification_target = config.get("classification_target", "osteoporosis")
    classification_selection_metric = config.get(
        "classification_selection_metric", "pr_auc"
    )
    categorical = set(variables["categorical_features"])

    published_columns = variables["feature_groups"].get("published_low_bmd_panel", [])
    published_panel_ready = bool(published_columns) and all(
        column in classification and classification[column].notna().any()
        for column in published_columns
    )
    medical_mobility_columns = variables["feature_groups"].get(
        "medical_mobility", []
    )
    medical_mobility_ready = bool(medical_mobility_columns) and all(
        column in classification and classification[column].notna().any()
        for column in medical_mobility_columns
    )
    classification_ablations = run_classification_ablations(
        classification,
        variables,
        categorical,
        config,
        include_published_lab_panel=published_panel_ready,
        include_medical_mobility_panel=medical_mobility_ready,
        target=classification_target,
    )
    classification_ablations.to_csv(
        paths.tables / "classification_ablations.csv", index=False
    )
    comparable_primary = classification_ablations.loc[
        classification_ablations["analysis_cohort"].eq("mec_exam")
        & classification_ablations["experiment"].isin(
            [
                "E1_age_sex",
                "E2_demographic",
                "E3_demographic_anthropometric",
                "E4_demographic_anthropometric_lifestyle",
                "E8_e3_plus_clinical",
            ]
        )
    ].sort_values("cv_pr_auc_mean", ascending=False)
    classification_experiment = str(comparable_primary.iloc[0]["experiment"])
    classification_features, classification_clusters = feature_experiments(
        classification, variables
    )[classification_experiment]
    classification_features = list(dict.fromkeys(classification_features))
    published_row = classification_ablations.loc[
        classification_ablations["experiment"].eq("E11_published_low_bmd_lab_panel")
    ]
    if (
        not published_row.empty
        and published_row.iloc[0]["cv_pr_auc_delta_vs_e8_ci_low"] > 0
        and published_row.iloc[0]["cv_pr_auc_mean"]
        > comparable_primary.iloc[0]["cv_pr_auc_mean"]
    ):
        classification_experiment = "E11_published_low_bmd_lab_panel"
        classification_features = list(
            dict.fromkeys(variables["feature_groups"]["published_low_bmd_panel"])
        )
        classification_clusters = False

    medical_row = classification_ablations.loc[
        classification_ablations["experiment"].eq(
            "E12_e8_plus_medical_mobility"
        )
    ]
    if (
        not medical_row.empty
        and medical_row.iloc[0]["cv_pr_auc_delta_vs_e8_ci_low"] > 0
        and medical_row.iloc[0]["cv_pr_auc_mean"]
        > comparable_primary.iloc[0]["cv_pr_auc_mean"]
    ):
        classification_experiment = "E12_e8_plus_medical_mobility"
        classification_features, classification_clusters = feature_experiments(
            classification,
            variables,
            include_medical_mobility_panel=True,
        )[classification_experiment]
        classification_features = list(dict.fromkeys(classification_features))

    targeted_cycle = config.get("targeted_biomarker_cycle", "2013-2014")
    targeted = classification.loc[classification["survey_cycle"].eq(targeted_cycle)].copy()
    targeted_columns = _targeted_feature_columns(variables)
    availability = pd.DataFrame(
        [
            {
                "survey_cycle": targeted_cycle,
                "column": column,
                "present": column in targeted,
                "observed_n": int(targeted[column].notna().sum()) if column in targeted else 0,
                "missing_percent": (
                    targeted[column].isna().mean() * 100 if column in targeted and len(targeted) else 100.0
                ),
            }
            for column in targeted_columns
        ]
    )
    availability.to_csv(
        paths.tables / "targeted_biomarker_data_availability.csv", index=False
    )
    missingness_mechanism_summary(targeted).to_csv(
        paths.tables / "targeted_biomarker_missingness_mechanisms.csv", index=False
    )
    missingness_outcome_bias(targeted, target=classification_target).to_csv(
        paths.tables / "targeted_biomarker_missingness_outcome_bias.csv", index=False
    )

    targeted_ablations = pd.DataFrame()
    targeted_result = None
    targeted_winner = None
    targeted_ready = bool(len(targeted)) and all(
        column in targeted and targeted[column].notna().any() for column in targeted_columns
    )
    if targeted_ready:
        targeted_ablations = run_classification_ablations(
            targeted,
            variables,
            categorical,
            config,
            include_targeted_biomarkers=True,
            target=classification_target,
        )
        targeted_ablations.to_csv(
            paths.tables / "targeted_biomarker_classification_ablations.csv", index=False
        )
        supported = targeted_ablations.loc[
            targeted_ablations["experiment"].isin(
                ["E9_e3_plus_grip", "E10_e3_plus_grip_biomarkers"]
            )
            & targeted_ablations["cv_pr_auc_delta_vs_e3_ci_low"].gt(0)
            & targeted_ablations["cv_pr_auc_delta_vs_e8_ci_low"].gt(0)
        ].sort_values("cv_pr_auc_mean", ascending=False)
        if not supported.empty:
            targeted_winner = str(supported.iloc[0]["experiment"])
            targeted_features, targeted_clusters = feature_experiments(
                targeted, variables, include_targeted_biomarkers=True
            )[targeted_winner]
            targeted_features = list(dict.fromkeys(targeted_features))
            targeted_result = run_classification(
                targeted,
                targeted_features,
                categorical,
                config,
                add_clusters=targeted_clusters,
                target=classification_target,
                selection_metric=classification_selection_metric,
            )
            _write_classification_evaluation(
                targeted_result,
                paths,
                config,
                "targeted_biomarker_classification",
                target=classification_target,
            )
            joblib.dump(
                targeted_result.model,
                paths.artifacts / "targeted_biomarker_classification_model.joblib",
            )

    regression_ablations = run_regression_ablations(
        regression, variables, categorical, config
    )
    regression_ablations.to_csv(
        paths.tables / "regression_ablations.csv", index=False
    )
    selected_regression_experiment = regression_ablations.iloc[0]["experiment"]
    regression_features, regression_clusters = feature_experiments(
        regression, variables
    )[selected_regression_experiment]
    regression_features = list(dict.fromkeys(regression_features))

    classification_result = run_classification(
        classification,
        classification_features,
        categorical,
        config,
        add_clusters=classification_clusters,
        target=classification_target,
        selection_metric=classification_selection_metric,
    )
    pooled_e3_features, _ = feature_experiments(classification, variables)[
        "E3_demographic_anthropometric"
    ]
    pooled_e3_features = list(dict.fromkeys(pooled_e3_features))
    pooled_e3_classification = run_classification(
        classification,
        pooled_e3_features,
        categorical,
        config,
        add_clusters=False,
        target=classification_target,
        selection_metric=classification_selection_metric,
    )
    pooled_e3_classification.metrics.to_csv(
        paths.tables / "classification_e3_pooled_metrics.csv", index=False
    )
    pooled_e3_classification.comparison.to_csv(
        paths.tables / "classification_e3_pooled_model_comparison.csv", index=False
    )
    _write_classification_evaluation(
        classification_result,
        paths,
        config,
        "classification",
        target=classification_target,
    )
    joblib.dump(classification_result.model, paths.artifacts / "classification_model.joblib")
    shap_status = write_shap_attributions(
        classification_result.model,
        classification,
        classification_features,
        paths,
        config,
        "classification",
        target=classification_target,
    )
    screening_threshold = float(
        classification_result.metrics.loc[
            classification_result.metrics["evaluation"].eq("sensitivity_oriented"), "threshold"
        ].iloc[0]
    )
    screening_metrics = classification_result.metrics.loc[
        classification_result.metrics["evaluation"].eq("sensitivity_oriented")
    ].iloc[0]
    f1_metrics = classification_result.metrics.loc[
        classification_result.metrics["evaluation"].eq("f1_optimized")
    ]
    f1_metrics = f1_metrics.iloc[0] if not f1_metrics.empty else None
    operating_points = {"default": 0.5, "sensitivity_oriented": screening_threshold}
    for row in classification_result.thresholds.loc[
        classification_result.thresholds["dimension"].eq("global_tradeoff")
    ].itertuples(index=False):
        operating_points[str(row.group)] = float(row.threshold)
    write_json(
        {
            "artifact": "classification_model.joblib",
            "target": classification_target,
            "positive_class_definition": "femoral-neck T-score <= -2.5",
            "negative_class_definition": "femoral-neck T-score > -2.5",
            "population": "NHANES MEC/DXA participants age 50 years or older",
            "production_cycles": config.get("production_classification_cycles", []),
            "candidate_expansion_cycles": config.get("candidate_expansion_cycles", []),
            "candidate_expansion_decision": (
                "sensitivity analysis only; require a fixed-holdout gain before deployment"
            ),
            "model": str(classification_result.comparison.iloc[0]["model"]),
            "probability_calibrated": True,
            "selection_metric": classification_selection_metric,
            "feature_set": classification_experiment,
            "features": classification_features,
            "operating_points": operating_points,
            "sensitivity_oriented_threshold": screening_threshold,
            "held_out_roc_auc": float(screening_metrics["roc_auc"]),
            "held_out_pr_auc": float(screening_metrics["pr_auc"]),
            "held_out_sensitivity": float(screening_metrics["sensitivity"]),
            "held_out_specificity": float(screening_metrics["specificity"]),
            "held_out_ppv": float(screening_metrics["ppv"]),
            "held_out_npv": float(screening_metrics["npv"]),
            "f1_optimized_threshold": (
                float(f1_metrics["threshold"]) if f1_metrics is not None else None
            ),
            "f1_optimized_held_out_f1": (
                float(f1_metrics["f1"]) if f1_metrics is not None else None
            ),
            "f1_optimized_held_out_roc_auc": (
                float(f1_metrics["roc_auc"]) if f1_metrics is not None else None
            ),
            "f1_optimized_held_out_pr_auc": (
                float(f1_metrics["pr_auc"]) if f1_metrics is not None else None
            ),
            "f1_optimized_held_out_sensitivity": (
                float(f1_metrics["sensitivity"]) if f1_metrics is not None else None
            ),
            "f1_optimized_held_out_specificity": (
                float(f1_metrics["specificity"]) if f1_metrics is not None else None
            ),
            "f1_optimized_held_out_ppv": (
                float(f1_metrics["ppv"]) if f1_metrics is not None else None
            ),
            "shap_attribution_status": shap_status["status"],
            "intended_use": (
                "research screening to prioritize DXA referral; not standalone diagnosis"
            ),
        },
        paths.artifacts / "classification_model_metadata.json",
    )

    temporal_cycle = config.get("temporal_evaluation_cycle", "2017-2018")
    temporal_result = run_classification(
        classification,
        classification_features,
        categorical,
        config,
        add_clusters=classification_clusters,
        evaluation_cycle=temporal_cycle,
        target=classification_target,
        selection_metric=classification_selection_metric,
    )
    _write_classification_evaluation(
        temporal_result,
        paths,
        config,
        "classification_temporal_validation",
        target=classification_target,
    )

    joblib.dump(
        temporal_result.model,
        paths.artifacts / "classification_temporal_model.joblib",
    )

    regression_result = run_regression(
        regression,
        regression_features,
        categorical,
        config,
        target="femoral_neck_bmd",
        add_clusters=regression_clusters,
    )
    pooled_e3_regression = run_regression(
        regression,
        pooled_e3_features,
        categorical,
        config,
        target="femoral_neck_bmd",
        add_clusters=False,
    )
    pd.DataFrame([pooled_e3_regression.metrics]).to_csv(
        paths.tables / "regression_e3_pooled_metrics.csv", index=False
    )
    pooled_e3_regression.comparison.to_csv(
        paths.tables / "regression_e3_pooled_model_comparison.csv", index=False
    )
    pd.DataFrame([regression_result.metrics]).to_csv(
        paths.tables / "regression_metrics.csv", index=False
    )
    regression_result.comparison.to_csv(
        paths.tables / "regression_model_comparison.csv", index=False
    )
    regression_result.predictions.to_csv(
        paths.tables / "regression_test_predictions.csv", index=False
    )
    regression_bootstrap_intervals(
        regression_result.predictions,
        "femoral_neck_bmd",
        config["bootstrap_iterations"],
        config["seed"],
    ).to_csv(paths.tables / "regression_metric_intervals.csv", index=False)
    joblib.dump(regression_result.model, paths.artifacts / "regression_model.joblib")
    save_regression_plot(
        regression_result.predictions,
        "femoral_neck_bmd",
        paths.figures / "regression_observed_vs_predicted.png",
    )

    # Descriptive full-cohort clusters are separate from fold-local predictive clusters.
    descriptive_clusterer = NutritionClusterAugmenter(
        features=NUTRITION_CLUSTER_FEATURES,
        candidates=tuple(config["nutrition_cluster_candidates"]),
        random_state=config["seed"],
    ).fit(classification)
    clustered = descriptive_clusterer.transform(classification)
    profile_columns = list(NUTRITION_CLUSTER_FEATURES) + [
        "femoral_neck_bmd",
        classification_target,
    ]
    clustered.groupby("nutrition_cluster")[profile_columns].mean().to_csv(
        paths.tables / "descriptive_nutrition_cluster_profiles.csv"
    )
    write_json(
        {
            "selected_clusters": descriptive_clusterer.n_clusters_,
            "selection_scores_negative_average_rank": descriptive_clusterer.cluster_scores_,
            "quality_and_stability_metrics": descriptive_clusterer.cluster_metrics_,
            "warning": "Descriptive full-cohort clusters are not used for held-out predictions.",
        },
        paths.tables / "descriptive_nutrition_cluster_selection.json",
    )

    summary = {
        "survey_cycles": variables["cycles"],
        "classification_n": len(classification),
        "classification_events": int(classification[classification_target].sum()),
        "classification_target": classification_target,
        "production_classification_cycles": config.get(
            "production_classification_cycles", []
        ),
        "candidate_expansion_cycles": config.get("candidate_expansion_cycles", []),
        "regression_n": len(regression),
        "classification_model": classification_result.comparison.iloc[0]["model"],
        "classification_feature_set": classification_experiment,
        "classification_weighted_roc_auc": classification_result.metrics.iloc[0]["roc_auc"],
        "classification_weighted_pr_auc": classification_result.metrics.iloc[0]["pr_auc"],
        "classification_auc_gain_vs_pooled_e3": classification_result.metrics.iloc[0][
            "roc_auc"
        ]
        - pooled_e3_classification.metrics.iloc[0]["roc_auc"],
        "targeted_biomarker_cycle": targeted_cycle,
        "targeted_biomarker_n": len(targeted),
        "targeted_biomarker_data_ready": targeted_ready,
        "targeted_biomarker_ci_supported_winner": targeted_winner,
        "screening_threshold": screening_threshold,
        "screening_sensitivity": screening_metrics["sensitivity"],
        "screening_specificity": screening_metrics["specificity"],
        "screening_ppv": screening_metrics["ppv"],
        "screening_npv": screening_metrics["npv"],
        "shap_attribution_status": shap_status["status"],
        "temporal_evaluation_cycle": temporal_cycle,
        "temporal_model": temporal_result.comparison.iloc[0]["model"],
        "temporal_weighted_roc_auc": temporal_result.metrics.iloc[0]["roc_auc"],
        "temporal_weighted_pr_auc": temporal_result.metrics.iloc[0]["pr_auc"],
        "regression_model": regression_result.metrics["model"],
        "regression_feature_set": selected_regression_experiment,
        "regression_weighted_rmse": regression_result.metrics["rmse"],
        "regression_weighted_r2": regression_result.metrics["r2"],
        "artifacts_directory": str(paths.artifacts),
        "reports_directory": str(paths.root / "reports"),
    }
    baseline = {
        "regression_weighted_rmse": 0.11368331840199414,
        "regression_weighted_r2": 0.35475625657783016,
        "regression_n": 2884,
    }
    comparison = pd.DataFrame(
        [
            {
                "metric": "classification_auc_selected_vs_pooled_e3",
                "previous": pooled_e3_classification.metrics.iloc[0]["roc_auc"],
                "updated": classification_result.metrics.iloc[0]["roc_auc"],
                "improvement": classification_result.metrics.iloc[0]["roc_auc"]
                - pooled_e3_classification.metrics.iloc[0]["roc_auc"],
            },
            {
                "metric": "regression_rmse_vs_previous_single_cycle",
                "previous": baseline["regression_weighted_rmse"],
                "updated": regression_result.metrics["rmse"],
                "improvement": baseline["regression_weighted_rmse"]
                - regression_result.metrics["rmse"],
            },
            {
                "metric": "regression_r2_vs_previous_single_cycle",
                "previous": baseline["regression_weighted_r2"],
                "updated": regression_result.metrics["r2"],
                "improvement": regression_result.metrics["r2"]
                - baseline["regression_weighted_r2"],
            },
            {
                "metric": "regression_rmse_e8_vs_pooled_e3",
                "previous": pooled_e3_regression.metrics["rmse"],
                "updated": regression_result.metrics["rmse"],
                "improvement": pooled_e3_regression.metrics["rmse"]
                - regression_result.metrics["rmse"],
            },
            {
                "metric": "regression_r2_e8_vs_pooled_e3",
                "previous": pooled_e3_regression.metrics["r2"],
                "updated": regression_result.metrics["r2"],
                "improvement": regression_result.metrics["r2"]
                - pooled_e3_regression.metrics["r2"],
            },
            {
                "metric": "regression_n",
                "previous": baseline["regression_n"],
                "updated": len(regression),
                "improvement": len(regression) - baseline["regression_n"],
            },
        ]
    )
    comparison.to_csv(paths.tables / "performance_comparison.csv", index=False)
    write_json(summary, paths.root / "reports" / "run_summary.json")
    return summary
