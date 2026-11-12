# Osteoporosis pipeline rework tracker

Updated: 2026-07-19

## Completed

- [x] Select the screening threshold from calibrated out-of-fold probabilities.
- [x] Promote E12 after its paired PR-AUC gain over E8 had a positive 95% CI lower
  bound; select the final algorithm by weighted CV PR-AUC and retain calibration.
- [x] Exclude the SMQFAM household-smoke variables from predictive feature groups;
  retain them in the cohort and missingness audit.
- [x] Preserve refusal/don't-know sentinel provenance before null replacement.
- [x] Treat ordinary activity follow-up skips as structural zero while leaving true
  non-response missing and flagged.
- [x] Use iterative multivariate numeric imputation and missingness indicators.
- [x] Report missingness mechanism and outcome-association audits.
- [x] Use dietary/MEC survey weights for model fitting, CV scoring, and held-out metrics.
- [x] Wire `bootstrap_iterations` into PSU-within-stratum bootstrap confidence intervals
  for held-out classification and regression metrics.
- [x] Add training-OOF age, sex, age-by-sex, and race/ethnicity operating thresholds;
  evaluate these thresholds on held-out subgroups and flag results with n < 100.
- [x] Add confidence intervals for CV ablations and their paired AUC deltas from E3.
- [x] Pool 2007–08, 2009–10, 2013–14, and 2017–18 with cycle-specific strata and
  four-cycle dietary/MEC weight scaling.
- [x] Add prior fracture, parental hip fracture, long-term glucocorticoid use,
  menopausal status, and hormone therapy from the OSQ/RHQ modules.
- [x] Add training-only regression E1–E8 ablations for ridge and random forest.
- [x] Add XGBoost as a weighted candidate for classification and regression.
- [x] Produce same-pooled-cohort E3 versus E8 comparisons and historical comparisons.
- [x] Remove the unnecessary diet-recall restriction from non-diet classification
  experiments and use MEC weights for their target population.
- [x] Replicate a published 19-variable low-BMD panel in compatible NHANES cycles.
- [x] Make exact osteoporosis (`T-score <= -2.5`) the sole classification target and
  add a 2017–2018 temporal holdout for that endpoint.
- [x] Add PR-AUC, MCC, log loss, Brier skill, calibration intercept/slope and plots,
  and decision-curve analysis alongside ROC-AUC and operating-point metrics.
- [x] Harmonize NHANES 2005–2006 as a candidate expansion and reject its deployment
  after performance decreased on the unchanged 2017–2018 test cycle.
- [x] Add frozen-model external-audit adapters for approved KNHANES, MrOS, and UK
  Biobank exports.
- [x] Add training-only recall and precision target thresholds and evaluate every
  operating point on held-out data.

## Deferred: requires added data or a separate study-design change

- [ ] Implement a best-valid-site WHO/ISCD outcome. A spine reference population and
  a pre-specified cross-site fallback rule are required; the current configured
  reference is femoral-neck only.
- [x] Evaluate decision-curve net benefit prospectively at a pre-specified 5% clinical
  utility threshold inside the additive `curve_bending` experiment. It selected an
  operating point effectively identical to the LCB-specificity policy and did not
  improve the result; production decision curves remain post-model evaluation outputs.
- [x] Add nested leave-one-cycle-out comparison in the additive `curve_bending`
  experiment: feature/hyperparameter selection, calibration, and threshold selection
  occur inside each outer training partition. Production fitting remains unchanged.
- [ ] Replace the PSU bootstrap approximation with a publication-specific survey
  variance analysis (for example replicate weights or a pre-specified Taylor-series
  design) if required by the target journal or protocol.

## New primary outputs

- `classification_operating_thresholds.csv`
- `classification_calibration.csv`
- `classification_decision_curve.csv`
- `classification_temporal_validation_metrics.csv`
- `classification_metric_intervals.csv`
- `regression_metric_intervals.csv`
- `classification_missingness_mechanisms.csv`
- `classification_missingness_outcome_bias.csv`
- `regression_ablations.csv`
- `regression_e3_pooled_metrics.csv`
- `performance_comparison.csv`

All output tables are written under `reports/tables/`.
