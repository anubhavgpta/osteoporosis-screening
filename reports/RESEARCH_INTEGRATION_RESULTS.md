# Research-guided osteoporosis screening update

Run date: 2026-07-19

## Conclusion

The production model is a calibrated, survey-weighted random-forest classifier using
E12. It combines E8 demographic, anthropometric, and clinical-history predictors with
arthritis history, liver condition, thyroid problems, and need for walking equipment.
It screens only for current femoral-neck osteoporosis (`T-score <= -2.5`) in adults
aged 50 years or older. It is not a diagnostic replacement for DXA and does not predict
future fracture or disease.

The literature-backed grip/biomarker hypotheses and a close replication of a published
19-variable laboratory panel were tested, but neither produced a CI-supported gain.
The E12 medical/mobility panel did: its cross-validated PR-AUC gain over E8 was 0.0123
(paired 95% CI 0.0030 to 0.0215). NHANES 2005–2006 was also harmonized, but was retained
only for sensitivity analysis because it reduced performance on the unchanged future-
cycle test.

## Evidence translated into pipeline changes

| Evidence | Implemented response |
|---|---|
| Xu et al. used 19 LASSO-selected clinical/laboratory variables; logistic regression was their best model, with test AUC 0.785 and external AUC 0.78. | Added E11 with the corresponding non-leaking variables and evaluated it on 2013–2014 plus 2017–2018, the cycles with complete component coverage. |
| Suh et al. reported femoral-neck micro-AUC 0.851 using 89 NHANES variables and five-fold CV. | Treated this as a richer, multiclass internal-validation result rather than a like-for-like target. Kept simpler models as candidates and added a temporal test. |
| Grip strength is associated with BMD in NHANES, but association does not prove incremental predictive value. | Tested E9 (E3 + grip) and E10 (E9 + ALP, calcium, folate, and serum 25(OH)D) with paired fold-level CIs. |
| TRIPOD+AI calls for discrimination, calibration, and clinical-utility evaluation. | Added PR-AUC, log loss, Brier score/skill, calibration intercept/slope and plots, and decision curves, while retaining operating-point and subgroup metrics. |
| Prediction-model reviews identify missing-data handling, overfitting, and weak external evaluation as common bias sources. | Kept imputation inside folds, avoided SMOTE, used a held-out test plus 500 PSU-within-stratum bootstrap repetitions, and added a 2017–2018 temporal holdout. |
| Common comorbidity and mobility variables may add signal without requiring laboratory testing. | Added E12 and promoted it only after a positive paired PR-AUC CI against E8. |
| Larger development samples do not guarantee transportability. | Added NHANES 2005–2006 as a candidate cycle, tested it against the unchanged 2017–2018 holdout, and did not deploy it after performance decreased. |

Primary sources: [Xu et al., 2024](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2024.1347219/full),
[Suh et al., 2023](https://www.jmir.org/2023/1/e40179/),
[TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378), and
[PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505).

## Local results

| Analysis | Cohort / endpoint | Main result | Decision |
|---|---|---|---|
| E12 production test | 9,367 total; 562 events; 1,874 held out | ROC-AUC 0.8533 (95% CI 0.8208–0.8903); PR-AUC 0.3721 (0.3021–0.4481); Brier 0.0467; calibration slope 1.146 | Deploy as the sole classifier. |
| E12 temporal test | Train earlier cycles n=7,081; test 2017–2018 n=2,286 | ROC-AUC 0.8335 (0.7911–0.8809); PR-AUC 0.3695 (0.2903–0.4766); Brier 0.0543; calibration slope 1.206 | Modest PR-AUC improvement over E8; transport still requires external validation. |
| E12 vs E8 paired CV | Same 9,367-person cohort | PR-AUC 0.3509 vs 0.3386; delta +0.0123 (paired 95% CI +0.0030 to +0.0215) | Promotion rule passed. |
| NHANES 2005–2006 expansion | Adds 1,614 people and 100 events | Fixed temporal PR-AUC decreased from 0.3598 to 0.3516 and recall from 0.8654 to 0.8093 | Keep as sensitivity data; do not deploy. |
| E11 published panel | 2013–2014 and 2017–2018, n=4,516 | CV ROC-AUC 0.8278; PR-AUC 0.3251; paired PR-AUC CI versus E8 crosses zero | Reject replacement. |
| E9 grip | 2013–2014, n=2,230 | CV ROC-AUC 0.8193; PR-AUC 0.3028; delta vs E3 +0.0016 (-0.0192 to +0.0224) | Reject replacement. |
| E10 grip + biomarkers | 2013–2014, n=2,230 | CV ROC-AUC 0.8060; PR-AUC 0.3587; delta vs E3 +0.0575 (-0.0159 to +0.1309) | Promising point estimate but too uncertain; reject deployment. |

At the high-recall production threshold of 0.0271, held-out sensitivity was 0.9140,
specificity 0.5600, PPV 0.1166, and NPV 0.9903. Raising the threshold improves precision
but necessarily reduces recall. For example, the training-selected precision-20%
operating point (threshold 0.0495) achieved held-out precision 0.2085 and recall 0.7045.
The training-selected precision-25% point (threshold 0.0824) achieved held-out precision
0.2773 and recall 0.5573. These alternatives are reported rather than implying one
threshold can maximize both metrics.

Decision curves show that the osteoporosis model begins to improve over test-all
around 3% threshold probability and remains positive over the evaluated
range. Thresholds still require prospective selection based on the clinical cost of a
missed case versus an unnecessary DXA referral.

## Metric set and interpretation

- **Discrimination:** ROC-AUC plus PR-AUC; PR-AUC drives model selection.
- **Probability accuracy:** log loss, Brier score, Brier skill, calibration intercept,
  calibration slope, and calibration plots.
- **Operating point:** sensitivity, specificity, PPV, NPV, F1, balanced accuracy, and
  MCC at both 0.5 and a threshold learned only from training predictions.
- **Clinical utility:** survey-weighted decision-curve net benefit across plausible
  referral thresholds.
- **Uncertainty and robustness:** 500 survey-design bootstrap repetitions, subgroup
  checks, and a temporally separated NHANES cycle.

Accuracy is intentionally not the primary metric. For screening, sensitivity and NPV
matter at the chosen referral threshold; for a rare exact-osteoporosis endpoint, PR-AUC,
PPV, and calibration prevent an apparently strong ROC-AUC from being overinterpreted.

## Remaining limitations

NHANES is cross-sectional, and the 2017–2018 test is temporal rather than a genuinely
independent health system or country. The tempting 2011–2012 grip-strength validation
cannot be performed for this endpoint because femoral-neck BMD was not available in
that cycle, as documented by [Suh et al.](https://www.jmir.org/2023/1/e40179/) and the
[CDC NHANES osteoporosis brief](https://www.cdc.gov/nchs/products/databriefs/db405.htm).
Frozen-model adapters are now ready for approved KNHANES, MrOS, and UK Biobank exports,
but those controlled datasets are not locally present. Before clinical use, the model
still needs completed independent geographic validation, prespecified referral
costs/thresholds, prospective workflow testing, and local recalibration.
