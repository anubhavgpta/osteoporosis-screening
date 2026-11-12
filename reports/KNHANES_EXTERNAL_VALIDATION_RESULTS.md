# KNHANES 2008-2011 frozen external validation

## What was run

The official KNHANES 2008-2011 basic and DXA SAS files were harmonized to the frozen
NHANES model's 18-input contract (17 supplied fields plus the deterministically derived
age-sex interaction). The classifier was not refit or recalibrated, and all reported
operating-point metrics use thresholds already stored in
`classification_model_metadata.json`.

The combined analysis uses the KNHANES guide's DXA examination weights and table-8
survey-year factors: 108/580 for 2008, 200/580 for 2009, 192/580 for 2010, and 80/580
for 2011.

## Cohort and endpoint

- Evaluation participants: 9,387 age 50 or older with observed femoral-neck BMD
- Raw osteoporosis events: 1,766
- Raw prevalence: 18.81%
- Survey-weighted prevalence: 16.81%
- Endpoint: femoral-neck T-score <= -2.5 using the frozen NHANES reference mean 0.858
  g/cm2 and SD 0.120 g/cm2

This fixed reference is required for a comparable endpoint, but it is not a Korean
clinical reference. Scanner, calibration, and reference-population differences remain
part of the external domain shift.

## Pooled performance

Discrimination was strong: ROC-AUC 0.8839 and PR-AUC 0.6471. The PR-AUC prevalence
baseline is 0.1681, giving a 3.85-fold lift over prevalence. The weighted mean predicted
probability was 11.53%, below the observed 16.81%, indicating systematic underprediction;
the calibration intercept was 0.974 and slope 1.177.

At the frozen NHANES screening threshold of 0.027135:

| Metric | NHANES held-out | KNHANES external | Change |
|---|---:|---:|---:|
| ROC-AUC | 0.8533 | 0.8839 | +0.0306 |
| PR-AUC | 0.3721 | 0.6471 | +0.2751 |
| Recall | 91.40% | 98.82% | +7.41 points |
| Specificity | 56.00% | 27.83% | -28.17 points |
| Precision | 11.66% | 21.67% | +10.02 points |
| NPV | 99.03% | 99.15% | +0.11 points |

The higher observed precision does not mean the target was solved: weighted KNHANES
prevalence is 16.81%, versus 5.97% in the NHANES held-out cohort. When KNHANES
sensitivity and specificity at the frozen threshold are standardized to 5.6%
prevalence, expected precision is only 7.51%.

As an external-cohort diagnostic only, the point nearest 90% recall achieved 90.13%
recall, 66.30% specificity, and 35.08% observed precision at threshold 0.04827. At a
5.6% prevalence, those sensitivity and specificity imply 13.69% precision, still well
below the 25% target. This threshold must not replace the frozen deployment threshold.

## Transportability limitations

- `long_term_glucocorticoid` and `needs_walking_equipment` were unavailable and remain
  missing for 100% of the cohort; the frozen model's fitted imputer handles them.
- Prior fracture history is available only in 2010-2011, so it is missing in 55.66% of
  the pooled evaluation cohort.
- Parental bone history is broader than parental hip fracture.
- KNHANES cirrhosis is narrower than the NHANES any-liver-condition question.
- Household-income quartiles were mapped without outcomes to survey-weighted NHANES PIR
  quartile midpoints; this is a rank proxy, not a direct measurement.
- All Korean participants use the only compatible frozen NHANES category (code 5,
  other race including multiracial), which is a domain proxy rather than an ethnic match.

## Conclusion

The model transports with good ranking performance, but the frozen screening threshold
is poorly specific in KNHANES and the model underpredicts absolute risk. The apparent
precision improvement is largely prevalence-driven. These results support further
transportability work, not pooled retraining or clinical deployment yet.
