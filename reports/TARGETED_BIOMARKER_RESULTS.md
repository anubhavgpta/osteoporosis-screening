# Targeted 2013–2014 grip and biomarker results

Run date: 2026-07-19

The analysis used the same 2,230-person MEC/DXA cohort for E3, E9, and E10, with
osteoporosis (`T-score <= -2.5`) as the only classification target. There were 155
osteoporosis events. Missing combined grip strength was outcome-associated:
survey-weighted osteoporosis prevalence was 0.1905 when missing versus 0.0535 when
observed. Predictors remain imputed only inside training folds.

## Paired five-fold cross-validation

| Experiment | ROC-AUC | PR-AUC | PR-AUC delta vs E3 | 95% CI |
|---|---:|---:|---:|---:|
| E3 demographic + anthropometric | 0.8209 | 0.3012 | 0.0000 | reference |
| E9 E3 + combined grip | 0.8193 | 0.3028 | +0.0016 | -0.0192 to +0.0224 |
| E10 E9 + four biomarkers | 0.8060 | 0.3587 | +0.0575 | -0.0159 to +0.1309 |

Neither candidate met the pre-specified rule that the paired PR-AUC-difference 95% CI
must have a lower bound greater than zero. The E10 point estimate is interesting but
too uncertain in only one cycle to deploy. The pooled E8 model remains the production
classifier, and no targeted-model artifact was written.
