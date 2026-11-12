# Four-cycle clinical-feature expansion results

> **Superseded historical analysis.** This document used the former combined
> low-BMD/osteoporosis endpoint and is not a production classification result. The
> active classifier now targets only osteoporosis; see
> `RESEARCH_INTEGRATION_RESULTS.md` and `tables/classification_metrics.csv`.

Run date: 2026-07-19

## Data expansion

NHANES 2007–08, 2009–10, 2013–14, and 2017–18 were harmonized. Survey strata are
cycle-specific and two-year MEC/dietary weights are divided by four for pooled analysis.

- Classification cohort: 2,069 → 8,930
- Regression cohort: 2,884 → 11,452

Clinical predictors are derived from the official Osteoporosis (OSQ) and Reproductive
Health (RHQ) modules: prior fracture, maternal/paternal hip fracture, long-term
prednisone/cortisone use, menstrual/menopausal status, and hormone therapy.

## Classification

On the identical pooled split, E8 improves held-out weighted AUC from 0.7831 to 0.7867
(+0.0036), PR-AUC from 0.7280 to 0.7360, and Brier score from 0.1871 to 0.1857.
Paired five-fold CV gives an E8-minus-E3 AUC difference of +0.0057 (95% CI +0.0018 to
+0.0096).

The historical single-cycle AUC was 0.8138. The pooled held-out AUC is lower at 0.7867;
this is not a like-for-like deterioration because the test population, race coding,
cycles, and weights changed. The same-cohort E3/E8 comparison above isolates the feature
gain and shows that it is real but modest.

Model CV AUC on E8:

- Regularized logistic regression: 0.7959
- XGBoost: 0.7900
- Random forest: 0.7849

## Regression

Training-only E1–E8 selection chose E8 with ridge. Against E3 on the identical pooled
held-out split:

- RMSE: 0.11044 → 0.10890 (improvement 0.00154)
- R²: 0.33290 → 0.35133 (improvement 0.01843)

Against the historical single-cycle result, RMSE improves from 0.11368 to 0.10890,
while R² changes from 0.35476 to 0.35133 because R² also depends on the outcome variance
of the changed test population.

Final E8 CV RMSE:

- Ridge: 0.11140
- XGBoost: 0.11188
- Random forest: 0.11545

XGBoost did not improve either task.

## Sources

- [CDC NHANES 2013–14 Osteoporosis codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/OSQ_H.htm)
- [CDC NHANES 2013–14 Reproductive Health codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/RHQ_H.htm)
- [CDC NHANES 2017–18 data listing](https://wwwn.cdc.gov/nchs/nhanes/search/DataPage.aspx?Cycle=2017-2018)
- [CDC 2007–10 survey-design combination note](https://wwwn.cdc.gov/nchs/data/nhanes/analyticguidelines/07-10-analytic-note.pdf)
