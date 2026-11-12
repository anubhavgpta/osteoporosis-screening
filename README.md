# Non-Invasive Osteoporosis Screening Pipeline

The separate native R implementation is available in
[OsteoScreen R](https://github.com/anubhavgpta/OsteoScreen-R-Survey-Weighted-Osteoporosis-Screening-with-NHANES-and-KNHANES).

## Reproducing the archived report

This repository contains the research source, configurations, tests, and historical
result summaries. Raw survey data, processed participant records, fitted models,
and generated result tables are excluded from Git. Download NHANES using the CLI;
obtain approved KNHANES exports separately for external validation.

Use the [reproducibility guide](reproducibility/README.md) for the pinned Python
environment, reference input hashes, isolated runs, and verification evidence.
The current configuration uses `pr_auc`, matching the report, and new runs save
the temporal model. Exact reproduction of historical metrics remains unverified:
the original generating revision/environment is unknown and the archived model
and metadata differ from their old external-validation manifest hashes. Those
historical discrepancies are preserved rather than silently corrected. See
`manuscript/README.md` for the original provenance and evaluation limitations.

This repository implements an end-to-end NHANES 2005–2018 research pipeline for
screening current femoral-neck osteoporosis from routinely collectible demographic,
anthropometric, lifestyle, and dietary variables. DXA is used only to construct the
research outcome. The output is a screening estimate—not a diagnosis or treatment
recommendation.

## Implemented design

- Femoral-neck DXA anchors the cohort; other components are left-joined by `SEQN`.
- Primary classification uses all eligible MEC/DXA participants aged 50+; only
  nutrition ablations require a reliable Day 1 dietary recall.
- NHANES 2005–2006 is harmonized as a candidate expansion cycle. It remains outside
  the production model because it did not improve the fixed 2017–2018 temporal test.
- Regression uses reliable Day 1 recalls in adults aged 40+.
- The sole classification outcome is osteoporosis (`T-score <= -2.5`). Osteopenia and
  normal BMD are both negative for this classifier.
- Continuous femoral-neck BMD is retained for a secondary regression task.
- Dietary-pattern clusters are fitted inside each training fold to prevent leakage.
- Regularized logistic regression, random forest, and XGBoost classifiers are compared
  by cross-validation. Ridge, random forest, and XGBoost regressors are compared likewise.
- E1–E8 ablations quantify demographics, anthropometrics, lifestyle, nutrition,
  nutrition clusters, and FRAX-grade clinical-history variables. E12 adds common-cycle
  arthritis, liver, thyroid, and walking-equipment variables. E11 independently
  replicates a published 19-variable low-BMD panel in cycles with complete coverage.
- A separate 2013-14 targeted analysis adds E9 (E3 + CDC combined grip strength) and
  E10 (E9 + alkaline phosphatase, serum calcium, serum total folate, and serum 25(OH)D).
  Keeping this analysis cycle-specific prevents grip-test availability from acting as
  a survey-cycle proxy in the four-cycle pooled model.
- Cluster count selection combines silhouette, Calinski–Harabasz, Davies–Bouldin,
  repeated-subsample adjusted-Rand stability, and a minimum cluster-size guardrail.
- The deployed classifier uses E12: E8 plus arthritis history, liver condition,
  thyroid problem, and need for walking equipment. Promotion required a positive
  paired PR-AUC confidence-interval lower bound against E8.
- Regression E1–E8 ablations run only on the training partition; the best feature set
  is then evaluated once on the held-out test partition.
- The chosen classifier is probability-calibrated, and its operating threshold is
  selected from calibrated out-of-fold training probabilities.
- Numeric item non-response uses iterative multivariate imputation. Structural activity
  skips, sentinel non-response, survey-subsample missingness, and exam limitations are
  handled and audited separately.
- Model fitting, cross-validation scoring, and held-out metrics use MEC weights for
  non-diet analyses and Day 1 dietary weights for nutrition analyses.
  PSU-within-stratum bootstrap intervals,
  raw and weighted prevalence,
  cohort attrition, missingness, subgroup checks, predictions, plots, and fitted models
  are written to disk.
- Subgroup operating thresholds are learned from out-of-fold training predictions and
  held-out subgroup results with n < 100 are explicitly flagged as unstable.
- A 2017–2018 temporal holdout tests transport over time for the same exact-
  osteoporosis endpoint.
- Evaluation includes ROC-AUC, PR-AUC, sensitivity, specificity, PPV, NPV, MCC,
  log loss, Brier score/skill, calibration intercept/slope and plots, and decision
  curves. Threshold-dependent metrics are reported at both 0.5 and a training-only
  sensitivity-oriented threshold.
- SHAP attribution is explicitly disabled in the report configuration. Set
  `shap_enabled: true` to generate it with the optional `shap` dependency; a
  manifest records disabled or unavailable attribution without failing the run.
- Training-only operating points are also supplied for recall targets of 90%, 80%,
  and 70%, and precision targets of 20%, 25%, and 30%.
- The NHANES-KNHANES integration experiment additionally compares a source-specific
  recall-preserving policy and a source-specific fixed-90%-recall policy. Candidate
  models maximize worst-source prevalence-standardized PPV only after meeting the
  recall constraint in every held-out training cycle/year. Paired PSU-within-stratum
  intervals compare each policy with the unchanged global-threshold incumbent.
- A harmonized external-audit command supports approved KNHANES, MrOS, and UK Biobank
  exports without fitting or tuning on the external cohort.

The configurable T-score reference is in
[`configs/model_config.yaml`](configs/model_config.yaml). It must be cited and reviewed
before publication or clinical interpretation. The current code uses a fixed
femoral-neck reference for both sexes, consistent with the intended ISCD-style
classification design, but does not turn the model into a clinical diagnostic device.

## Run

PowerShell:

```powershell
py -3.13 -m venv .venv-repro
.\.venv-repro\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv-repro\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv-repro\Scripts\python.exe -m osteoporosis_screening.reproduce --output runs/core --download --expected-inputs reproducibility/reference_inputs.json
```

The download step retrieves official 2005–06, 2007–08, 2009–10, 2013–14, and
2017–18 XPT files listed in
[`configs/variables.yaml`](configs/variables.yaml). Raw files are ignored by Git and
never modified.

The examples below use the original `.venv` path; substitute `.venv-repro` when
using the locked setup above, and pass `--root runs/core` to work with its outputs.
Use Python 3.13.3 on Windows x86_64 for the validated environment. KNHANES data
must be obtained separately; see the guide before adding `--external`.

Native harmonization and frozen external validation of the approved KNHANES
2008-2011 SAS files in `data/raw_knhanes/`:

```powershell
.\.venv\Scripts\osteoporosis-pipeline.exe knhanes-validate --root .
```

This produces the harmonized cohort, a variable-mapping audit, an official-weight
cohort flow, pooled metrics, metrics by survey year, calibration, and descriptive
precision-recall trade-offs. Use `knhanes-harmonize` instead when only the harmonized
cohort is required.

Compare NHANES-only, KNHANES-only, native-weight pooled, dataset-balanced pooled, and
reduced consistently measured training strategies:

```powershell
.\.venv\Scripts\osteoporosis-pipeline.exe integration-experiment --root .
```

This command selects models and 90%-recall thresholds using leave-one-cycle/year-out
training predictions, then evaluates unchanged models on NHANES 2017-2018 and KNHANES
2011. It does not overwrite the production model artifact.

Run the additive curve-bending experiment without replacing any production result or
model artifact:

```powershell
.\.venv\Scripts\osteoporosis-pipeline.exe curve-bending-experiment --root .
```

This compares pooled biomarkers without grip, conservative lower-confidence-bound
recall thresholds, sex-specific models with independently selected features and
hyperparameters, a biomarker confirmation cascade, and a screen-then-DXA pathway.
Outer leave-one-cycle-out evaluation contains all model, calibration, and threshold
selection inside each training partition. Outputs use the `curve_bending_` prefix;
the command refuses to replace even those additive outputs unless `--overwrite` is
given.

After that model-selection run exists, compare less restrictive operating points
without repeating the full hyperparameter grid:

```powershell
.\.venv\Scripts\osteoporosis-pipeline.exe relaxed-operating-experiment --root .
```

This consolidates the already training-selected 80% and 70% recall policies and the
20%, 25%, and 30% precision policies across both random and temporal holdouts. It
ranks the pre-specified choices by balanced accuracy, MCC, and F1 without fitting or
tuning on either evaluation set. Its independent outputs use the
`curve_bending_relaxed_` prefix.

Run the additive feature/model combination experiment without replacing production
artifacts:

```powershell
.\.venv\Scripts\osteoporosis-pipeline.exe feature-combination-experiment --root .
```

This evaluates albumin/alkaline-phosphatase chemistry, additional anthropometry,
leakage-safe high-risk medication classes when all cycle files are available,
expanded nested hyperparameter grids, LightGBM, CatBoost, MLP, LASSO-selected
logistic regression, spline logistic regression, and a stacked tree/boosting
ensemble. It also writes OSTA and non-proprietary FRAX-grade risk-count clinical
score baselines so the ML model can be compared with simple clinical tools.
Feature, model, calibration, and threshold selection are nested inside
leave-one-cycle-out evaluation. It also performs nested leave-one-year-out KNHANES
recalibration using the existing frozen-model external predictions. Outputs use the
independent `feature_combination_` prefix. Optional SMOTE/ADASYN/random-oversampling
arms are available by installing `.[literature]` and setting
`include_resampling_candidates: true`.

## Target-definition and methodology sensitivity analysis

Run `.\.venv\Scripts\osteoporosis-pipeline.exe target-definition-experiment --root .`
to compare the deployed osteoporosis target with an inclusive low-BMD target under both
the rigorous pipeline and a deliberately naive literature-style diagnostic arm. This
experiment exists to explain literature comparisons, not to change the deployed model.

Run the additive low-BMD feature/model experiment:

```powershell
.\.venv\Scripts\osteoporosis-pipeline.exe low-bmd-feature-combination-experiment --root .
```

This reuses the nested feature-combination machinery for the inclusive low-BMD target,
including anthropometric-first, E12, extended-anthropometric, routine-lab, and
published low-BMD panel candidates where cycle coverage permits. Outputs use the
`low_bmd_feature_combination_` prefix and do not replace production osteoporosis
artifacts.

Refresh the two-target metric tracker after any additive experiment:

```powershell
.\.venv\Scripts\osteoporosis-pipeline.exe track-two-target-results --root .
```

The tracker writes `reports/tables/two_target_metric_tracker.csv` and
`reports/TWO_TARGET_TRACKING_RESULTS.md`, comparing each available experiment against
the frozen rigorous baseline for its target.

External validation of another already-harmonized CSV or Parquet export:

```powershell
.\.venv\Scripts\osteoporosis-pipeline.exe external-audit `
  --dataset knhanes_2008_2011 `
  --input C:\path\to\harmonized_knhanes.csv
```

The required dataset coding and access notes are in
[`configs/external_datasets.yaml`](configs/external_datasets.yaml). The command derives
the identical femoral-neck osteoporosis endpoint and never retrains or recalibrates the
model. All operating metrics use thresholds stored with the NHANES model. The generated
`external_knhanes_2008_2011_manifest.json` records the input hash, frozen-model policy,
endpoint reference, sample size, and prevalence. Separate cohort-flow, feature-audit,
calibration, and precision/recall trade-off tables support the transportability review.
Thresholds in the trade-off table are descriptive diagnostics only and must not replace
the frozen NHANES screening threshold. Precision is also standardized to the 5.6%
NHANES comparison prevalence because observed KNHANES prevalence is substantially higher.

## Outputs

- `data/processed/`: reproducible classification and regression cohorts
- `data/processed/classification_all_cycles_cohort.csv.gz`: production plus candidate
  expansion cycles; `classification_cohort.csv.gz` contains production cycles only
- `artifacts/`: serialized calibrated classifier and BMD regressor
- `reports/run_summary.json`: compact run summary
- `reports/tables/`: cohort flow, merge audit, missingness mechanisms and bias checks,
  prevalence, test metrics and confidence intervals, operating thresholds, model
  comparisons, subgroup checks, predictions, and dietary cluster profiles
- `reports/tables/performance_comparison.csv`: historical and same-cohort E3/E8 comparisons
- `reports/tables/targeted_biomarker_classification_ablations.csv`: paired 2013-14
  E3/E9/E10 comparison; downstream targeted deployment outputs are created only when
  an E9/E10 PR-AUC gain has a positive paired 95% CI lower bound
- `reports/figures/`: held-out ROC, calibration, decision-curve, and
  observed-versus-predicted plots
- `reports/DATASET_EXPANSION_RESULTS.md`: fixed-holdout evaluation of NHANES 2005–2006
  and the external-dataset access/compatibility decision
- `reports/MEDICAL_MOBILITY_RESULTS.md`: E12 promotion evidence and held-out
  precision–recall operating-point trade-offs
- `reports/PRECISION_RECALL_CONSTRAINED_RESULTS.md`: constrained NHANES-KNHANES
  precision/recall analysis and the retain-incumbent decision
- `reports/tables/nhanes_knhanes_integration_policy_*.csv`: policy metrics,
  paired comparisons and intervals; the corresponding gzip file contains participant-
  level policy predictions
- `reports/tables/curve_bending_*.csv`: additive biomarker availability/signal,
  nested leave-one-cycle-out model and policy comparisons, threshold audits, model
  selections, and participant-level predictions; these never replace production files

## Validate

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
```

## Interpretation limits

NHANES is cross-sectional, so the model estimates concurrent osteoporosis risk and cannot
predict future disease onset or establish causal effects. It is a screening/triage
model, not a replacement for DXA diagnosis. The weighted metrics use
participant weights and a PSU-within-stratum bootstrap, but a target publication may
still require replicate-weight or Taylor-series variance estimation. The pipeline uses
a held-out split plus inner cross-validation and temporal validation; a later
study-grade release should add nested repeated cross-validation, Day 1/Day 2 and
supplement sensitivity analyses, and independent geographic validation. See
[`reports/IMPLEMENTATION_ROADMAP.md`](reports/IMPLEMENTATION_ROADMAP.md) for the tracked
rework status.
