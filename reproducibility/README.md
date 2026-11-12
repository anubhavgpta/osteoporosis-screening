# Reproducing an analysis

## Supported environment

The validated environment is CPython 3.13.3 on Windows x86_64.
`requirements-lock.txt` pins runtime, optional experiment, test, and build packages.
Other operating systems or Python versions are not yet validated. This is a new
reproducible environment, not a reconstruction of the report's lost environment.

From the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv-repro
.\.venv-repro\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv-repro\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv-repro\Scripts\python.exe -m pip check
.\.venv-repro\Scripts\python.exe -m pytest -q
```

Use Python 3.13.3 specifically for the validated environment. An existing copied
`.venv` referencing a missing interpreter must not be reused. GitHub Actions runs
the same installation and tests without access to the research datasets.

## Data

NHANES downloads are public. On a clean checkout:

```powershell
.\.venv-repro\Scripts\osteoporosis-pipeline.exe download --root .
```

The downloader records source URLs and SHA-256 hashes in
`data/raw_xpt/download_manifest.json`, including absent optional components.
`reference_inputs.json` records the 100 NHANES XPT files and eight KNHANES SAS
inputs present during the November 12, 2026 audit. It contains hashes, not records.
With `--expected-inputs`, a missing or changed reference file stops the run. A
future revised data release should receive its own reviewed manifest; do not
silently replace the reference hashes merely to pass verification.

KNHANES is optional for core training but required for the report's external test.
Obtain approved exports through KDCA, extract the basic-file ZIP archives, and use:

```text
data/raw_knhanes/extracted/hn08_all.sas7bdat
data/raw_knhanes/extracted/hn09_all.sas7bdat
data/raw_knhanes/extracted/hn10_all.sas7bdat
data/raw_knhanes/extracted/hn11_all.sas7bdat
data/raw_knhanes/KNHANES_2008_DXA_SAS.sas7bdat
data/raw_knhanes/KNHANES_2009_DXA_SAS.sas7bdat
data/raw_knhanes/KNHANES_2010_DXA_SAS.sas7bdat
data/raw_knhanes/KNHANES_2011_DXA_SAS.sas7bdat
```

The repository does not distribute participant records or fitted historical models.
Source-file hashes alone do not grant access to the underlying data.

## Isolated execution

Rebuild and audit both cohorts without training:

```powershell
.\.venv-repro\Scripts\python.exe -m osteoporosis_screening.reproduce --output runs/cohorts --prepare-only --external --expected-inputs reproducibility/reference_inputs.json
```

Run training, random and temporal evaluation, then frozen KNHANES validation:

```powershell
.\.venv-repro\Scripts\python.exe -m osteoporosis_screening.reproduce --output runs/core --external --expected-inputs reproducibility/reference_inputs.json
```

For NHANES only, omit `--external`. Add `--download` to download into the new run
directory. Add `--experiments` to run the report's nested feature-combination and
target-definition analyses after core training; these are computationally expensive.
Use a new output directory for each execution; existing directories are refused.
`--data-root PATH` selects another directory containing `raw_xpt` and `raw_knhanes`.
The runner copies data, so allow disk space for a second copy.

Each run contains a source/configuration snapshot, exact installed package versions,
Python/platform details, input and output hashes, a stage log, and a manifest whose
status is `complete` or `failed`. Source hashes identify uncommitted changes even
when a Git revision is available. Training uses seed 42; the runner fixes Python
hash randomization and numerical-library thread counts. Models, predictions,
thresholds, metrics, and the temporal fitted model are saved under the run root.
The ordinary CLI still writes to its `--root`; use the isolated runner to preserve
historical results.

Verify that a completed run's recorded source, inputs, and outputs remain intact:

```powershell
.\.venv-repro\Scripts\python.exe -m osteoporosis_screening.reproduce --output runs/core --verify
```

This checks integrity against that run's manifest, not statistical validity or
agreement with the historical report. Failed runs are rejected.

Independently reload the locally trained internal and temporal models and re-score
their exported test rows, plus KNHANES when its external predictions exist:

```powershell
.\.venv-repro\Scripts\python.exe -m osteoporosis_screening.verify_results --root runs/core
```

This first verifies the run hashes and then requires probability agreement within
1e-12. Only use it with trusted, locally trained models: loading a joblib artifact
executes Python serialization code.

## What has and has not been reproduced

The archived classification cohort has 9,367 rows, 161 columns, and 562 cases.
The harmonized KNHANES file has 21,303 rows and 23 columns; the external evaluation
restricts this to 9,387 eligible adults. `cohort_verification.json` records the
fresh rebuild comparison. Values were compared with relative/absolute tolerance
1e-12 and dtype differences ignored. Compressed-file bytes may differ because
gzip headers contain timestamps; compare decoded data as well as recording hashes.

Tests exercise two real weighted training runs, calibration, operating thresholds,
and model save/reload agreement to tolerance 1e-12, alongside the existing unit
suite. They do not establish identical numerical results across platforms.

The full core workflow and frozen KNHANES validation completed on November 13,
2026. `full_run_verification.json` records the audit. All 1,874 internal, 2,286
temporal, and 9,387 external probabilities matched their historical values exactly.
Reloading the new models also reproduced their exports with zero maximum absolute
difference. Internal and temporal confidence-interval tables matched within 1e-12.
Supplementary nested experiments were not rerun. Iterative-imputer convergence
warnings occurred and remain in the run log.

The primary selection setting is now `pr_auc`, matching the report. Historical
model/metadata hashes still disagree with the old external manifest; that manifest
has not been rewritten to hide the discrepancy. The original trained model's
exact generating revision and environment remain unknown. The observed numerical
agreement does not resolve those provenance gaps or guarantee cross-platform
equivalence.

SHAP is disabled explicitly with `shap_enabled: false`, since it is not required
for the report's main results. Set it to `true` for an additional attribution run.
Its masker now encodes text categories and restores their labels before prediction;
the test suite covers text categories and missing values. Installing an optional
package no longer changes whether this stage runs in the default configuration.

Reproducibility is separate from evaluation validity. The historical core workflow
selects feature panels on the whole cohort before final splits. This behavior is
preserved and documented; use the nested experiments for their stated evaluation
design. Correcting the study design requires a new analysis and new reported
results, not relabeling historical results as unbiased.
