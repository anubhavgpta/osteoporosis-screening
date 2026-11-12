# Osteoporosis Screening

**Survey-weighted machine learning for femoral-neck osteoporosis screening using NHANES and KNHANES.**

[![Reproducibility checks](https://github.com/anubhavgpta/osteoporosis-screening/actions/workflows/tests.yml/badge.svg)](https://github.com/anubhavgpta/osteoporosis-screening/actions/workflows/tests.yml)

## Project overview

This repository provides a configuration-driven research pipeline for estimating concurrent femoral-neck osteoporosis from routinely collected demographic, anthropometric, and medical-history information. Model development uses the US National Health and Nutrition Examination Survey (NHANES); external evaluation uses approved exports from the Korea National Health and Nutrition Examination Survey (KNHANES).

The analytical objective is to assess risk discrimination, probability calibration, and screening-threshold transportability across survey periods and populations. Dual-energy X-ray absorptiometry (DXA) defines the research outcome and is excluded from the screening predictors. The software supports research on prioritization for DXA assessment; it is not a clinically validated diagnostic system.

The independent [R implementation](https://github.com/anubhavgpta/OsteoScreen-R-Survey-Weighted-Osteoporosis-Screening-with-NHANES-and-KNHANES) is maintained in a separate repository. Its outputs are not assumed to reproduce the Python results exactly.

## Study design

| Component | Specification |
|---|---|
| Primary population | NHANES participants aged 50 years or older with eligible femoral-neck DXA measurements |
| Development cycles | 2007–2008, 2009–2010, 2013–2014, and 2017–2018 |
| Expansion analysis | 2005–2006, retained as a sensitivity analysis |
| Primary endpoint | Femoral-neck T-score ≤ −2.5 |
| Reference calculation | T-score = (BMD − 0.858) / 0.120, with BMD in g/cm² |
| Candidate classifiers | Regularized logistic regression, random forest, and XGBoost |
| Selection metric | Survey-weighted average precision (`pr_auc`) |
| Calibration | Sigmoid calibration within training data |
| Primary operating policy | Threshold selected from calibrated out-of-fold predictions to target 90% training sensitivity |
| Evaluation | Random holdout, 2017–2018 temporal holdout, frozen KNHANES evaluation, and separate nested experiments |
| Secondary analyses | Continuous BMD regression, predictor-panel comparisons, and alternative outcome definitions |

The historical report selected the E12 panel, comprising 17 routine predictors and one derived age–sex category. The current workflow retains panel comparisons; a new run is not guaranteed to select the same panel or algorithm.

Preprocessing addresses questionnaire nonresponse, structural skips, missing values, and categorical encoding. Numeric imputation, scaling, and categorical transformations are fitted within model pipelines. MEC or dietary survey weights are applied according to the analysis population. Evaluation includes discrimination, calibration, sensitivity, specificity, predictive values, and PSU-within-stratum bootstrap intervals.

The outcome constants and analysis settings are defined in [model_config.yaml](configs/model_config.yaml). Variable mappings are specified in [variables.yaml](configs/variables.yaml).

## Reproducibility status

| Verification component | Status |
|---|---|
| Dependency environment | Exact package versions recorded in `requirements-lock.txt`; target platform is Windows x86_64 with CPython 3.13.3 |
| Automated validation | 58 tests passed locally; the clean-install GitHub Actions workflow passed |
| NHANES cohort reconstruction | 9,367 rows and 161 columns reproduced from raw inputs, including 562 osteoporosis cases |
| KNHANES harmonization | 21,303 rows and 23 columns reproduced; the external evaluation population comprises 9,387 eligible adults |
| Repeated training and serialization | Synthetic-data tests verify repeated probabilities, thresholds, and saved-model predictions within 1e-12 |
| Full-data model verification | Pending completion of the new full training and external-evaluation run |
| Historical report metrics | Retained as reference values; exact regeneration has not been established |

Cohort comparisons used relative and absolute tolerances of 1e-12, with storage-dtype differences ignored. Verification records are available in [cohort_verification.json](reproducibility/cohort_verification.json). Historical model results are preserved in [historical_reference.json](reproducibility/historical_reference.json).

The original model-generating revision and environment are not fully recoverable. Archived model and metadata hashes also differ from their historical external-validation manifest. These discrepancies remain documented; successful software tests should not be interpreted as proof that every published metric has been reproduced.

## Installation

The supported validation environment is **CPython 3.13.3 on Windows x86_64**. Execute the following commands from the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv-repro
.\.venv-repro\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv-repro\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv-repro\Scripts\python.exe -m pip check
.\.venv-repro\Scripts\python.exe -m pytest -q
```

The `py -3.13` launcher must resolve to Python 3.13.3. The isolated runner checks Python and installed package versions against the committed environment specification. Other platforms and interpreter versions have not been validated.

## Data requirements

NHANES files are retrieved from official CDC endpoints. KNHANES external evaluation requires separately obtained, approved basic-survey and DXA exports for 2008–2011. Required filenames and extraction paths are documented in the [reproducibility guide](reproducibility/README.md#data).

Raw and processed participant data, fitted model binaries, generated tables, and local run directories are excluded from version control. The repository includes a [reference input manifest](reproducibility/reference_inputs.json) containing SHA-256 hashes for 100 NHANES files and eight KNHANES SAS files. Hashes establish input identity; they do not provide access to the datasets.

## Execution

### NHANES development and evaluation

The following command creates a new isolated directory, downloads NHANES components, verifies reference inputs, and executes the core workflow:

```powershell
.\.venv-repro\Scripts\python.exe -m osteoporosis_screening.reproduce `
  --output runs/core `
  --download `
  --expected-inputs reproducibility/reference_inputs.json
```

The core workflow includes classification, temporal evaluation, and secondary regression analyses. It can require substantial CPU time. Existing run directories are rejected to protect prior outputs.

### External evaluation

After placing approved KNHANES files in the documented data directories, execute a separate run:

```powershell
.\.venv-repro\Scripts\python.exe -m osteoporosis_screening.reproduce `
  --output runs/core-external `
  --external `
  --expected-inputs reproducibility/reference_inputs.json
```

This command uses existing local raw files. Add `--download` if NHANES files have not been obtained. External predictions use the fitted NHANES model and stored thresholds without refitting on KNHANES.

### Cohort-only verification

```powershell
.\.venv-repro\Scripts\python.exe -m osteoporosis_screening.reproduce `
  --output runs/cohorts `
  --prepare-only `
  --external `
  --expected-inputs reproducibility/reference_inputs.json
```

Use `--data-root PATH` to select an alternative directory containing `raw_xpt` and `raw_knhanes`. The runner copies inputs into the isolated run; sufficient storage for that copy is required.

### Integrity and prediction verification

After a full run completes:

```powershell
.\.venv-repro\Scripts\python.exe -m osteoporosis_screening.reproduce --output runs/core-external --verify
.\.venv-repro\Scripts\python.exe -m osteoporosis_screening.verify_results --root runs/core-external
```

The first command verifies recorded source, configuration, environment-file, input, and output hashes. The second reloads locally trained models and compares recomputed internal, temporal, and available external probabilities with exported predictions. A failed or incomplete run does not pass integrity verification.

## Supplementary analyses

Add `--experiments` to a new full run to execute the nested feature-combination and target-definition analyses. Additional commands are available through the CLI:

| Command | Analytical purpose |
|---|---|
| `feature-combination-experiment` | Nested evaluation of extended predictor panels and model families |
| `target-definition-experiment` | Comparison of osteoporosis and inclusive low-BMD endpoints |
| `low-bmd-feature-combination-experiment` | Extended model and panel comparisons for low BMD |
| `curve-bending-experiment` | Biomarker, subgroup, and screening-policy analyses |
| `relaxed-operating-experiment` | Alternative recall and precision operating points from existing experiment outputs |
| `integration-experiment` | NHANES-only, KNHANES-only, and pooled training strategies |
| `osteoporosis-imbalance-experiment` | Class-imbalance sensitivity analyses |
| `site-target-experiment` | Alternative skeletal-site endpoint analyses |
| `external-audit` | Frozen evaluation of a compatible harmonized external dataset |
| `track-two-target-results` | Consolidation of available osteoporosis and low-BMD results |

For example, the following command writes additive experiment outputs into an existing run:

```powershell
.\.venv-repro\Scripts\osteoporosis-pipeline.exe feature-combination-experiment --root runs/core-external
```

Direct CLI commands do not refresh the isolated runner's manifest. Use a new isolated run with `--experiments` when a single manifest must cover all generated outputs. Some supplementary commands require prior experiment outputs or external data. Refer to their configuration files and result summaries before execution.

SHAP attribution is disabled explicitly in the default configuration. Set `shap_enabled: true` for a separate attribution analysis. The implementation restores categorical labels before model prediction and records whether attribution was completed or skipped.

## Repository structure

| Location | Contents |
|---|---|
| `src/osteoporosis_screening/` | Ingestion, harmonization, modeling, evaluation, experiment, and reproducibility modules |
| `configs/` | Variable mappings, endpoint definitions, and experiment parameters |
| `tests/` | Unit, integration, provenance, and repeatability tests |
| `reproducibility/` | Execution guidance, reference hashes, and verification records |
| `reports/` | Historical analytical summaries |
| `manuscript/` | Research text, methodological reviews, and provenance records |
| `.github/workflows/` | Automated installation and validation checks |
| `runs/` | Local isolated runs; excluded from Git |

Each isolated run contains a source snapshot, copied configuration, environment details, stage logs, and `manifest.json`. Generated outputs include harmonized cohorts, internal and temporal models, operating thresholds, participant predictions, metric tables, and figures. Historical outputs at the repository root are not overwritten by the isolated runner.

## Methodological limitations

- The surveys are cross-sectional. The endpoint represents concurrent femoral-neck osteoporosis, not future fracture risk or disease onset.
- The historical core workflow selects predictor panels before final random and temporal splits. Consequently, those evaluations should not be described as fully independent estimates of the complete selection procedure. Separate nested experiments address different evaluation designs.
- Survey weighting and the implemented bootstrap do not establish equivalence to a validated replicate-weight or Taylor-series variance procedure.
- KNHANES mapping includes proxies and unavailable predictors. Differences in calibration and screening performance cannot be attributed to geography alone.
- The fixed T-score reference, age eligibility, skeletal-site scope, and scanner comparability require consideration when interpreting results.
- Prospective clinical effectiveness, fairness, and cost-effectiveness have not been established.

Detailed provenance and methodological qualifications are retained in the [manuscript documentation](manuscript/README.md). The [implementation roadmap](reports/IMPLEMENTATION_ROADMAP.md) records additional research work.

## Research use and attribution

For research using this software, report the repository commit, Python and package versions, input-manifest identity, configuration, evaluation design, and run-manifest status. Cite the underlying NHANES and KNHANES data sources and the relevant methodological references. Repository availability does not establish a software license or confer redistribution rights for external datasets; review applicable terms before reuse or redistribution.
