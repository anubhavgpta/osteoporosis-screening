# NHANES-KNHANES training integration experiment

## Decision

The native-survey-weight pooled model is the best **screening** candidate. It was the
only full-feature strategy that achieved at least 90% recall on both held-out cohorts:

- NHANES 2017-2018: 90.28% recall, 54.27% specificity, 12.57% observed precision
- KNHANES 2011: 97.97% recall, 33.57% specificity, 22.09% observed precision

However, pooling did not produce the best prevalence-standardized precision-recall
ranking. The NHANES-only model retained the highest mean PR-AUC standardized to 5.6%
prevalence (0.404 versus 0.389 for native-weight pooling). Therefore, integration
improved high-recall detection, but not discrimination uniformly.

No strategy reached the original target of approximately 90% recall and 25% precision
at 5.6% prevalence. The native-weight pooled model's precision standardized to 5.6%
was 10.48% on NHANES and 8.05% on KNHANES at its training-selected threshold.

## Leakage-safe design

Training excluded all NHANES 2017-2018 and KNHANES 2011 participants. Within each
training strategy, candidate estimators were selected from regularized logistic
regression, random forest, and XGBoost using leave-one-survey-cycle/year-out predictions.
The selection metric was mean PR-AUC standardized to 5.6% prevalence. Calibration and
threshold selection also used only grouped out-of-fold training predictions.

For pooled strategies, the threshold was the more sensitive of the source-specific
NHANES and KNHANES thresholds, ensuring at least 90% out-of-fold training recall in each
source. Numeric variables used fold-local median imputation; categorical missingness was
represented explicitly. Neither holdout was used for model fitting, calibration, or
threshold selection.

## Strategies tested

| Strategy | Training participants | Events | Features | Weight contribution |
|---|---:|---:|---:|---|
| NHANES only | 7,081 | 407 | 18 | 100% NHANES |
| KNHANES only | 8,113 | 1,532 | 18 | 100% KNHANES |
| Native-weight pooled | 15,194 | 1,939 | 18 | 84.2% NHANES / 15.8% KNHANES |
| Dataset-balanced pooled | 15,194 | 1,939 | 18 | 50% / 50% |
| Reduced consistently measured pooled | 15,194 | 1,939 | 11 | 50% / 50% |

The reduced feature set retained age, sex, education, the age-sex interaction, BMI,
height, weight, menopausal status, hormone therapy, arthritis, and thyroid history. It
excluded race and income proxies plus fracture, glucocorticoid, parental fracture,
liver, and walking-equipment fields that were unavailable or inconsistently defined.

## Overall comparison

| Discrimination rank | Screening rank | Strategy | Selected model | Mean standardized PR-AUC | Minimum recall | Mean specificity | Mean standardized precision |
|---:|---:|---|---|---:|---:|---:|---:|
| 1 | 4 | NHANES only | XGBoost | 0.404 | 84.50% | 44.58% | 9.10% |
| 2 | 1 | Native-weight pooled | Random forest | 0.389 | 90.28% | 43.92% | 9.26% |
| 3 | 2 | Reduced consistent pooled | XGBoost | 0.377 | 93.91% | 38.36% | 8.59% |
| 4 | 5 | Dataset-balanced pooled | Random forest | 0.374 | 88.39% | 43.01% | 8.94% |
| 5 | 3 | KNHANES only | XGBoost | 0.366 | 61.97% | 78.29% | 17.18% |

Screening ranks prioritize meeting 90% recall on both holdouts, then standardized
precision. Ranks for strategies that fail the recall requirement are not deployment
recommendations; for example, KNHANES-only has higher precision because it detects only
61.97% of NHANES cases at its training-selected threshold.

## Holdout results at training-selected thresholds

| Strategy | Holdout | ROC-AUC | Standardized PR-AUC | Recall | Specificity | Observed precision | Precision at 5.6% prevalence |
|---|---|---:|---:|---:|---:|---:|---:|
| NHANES only | NHANES 2017-2018 | 0.832 | 0.363 | 84.50% | 57.71% | 12.70% | 10.60% |
| NHANES only | KNHANES 2011 | 0.875 | 0.444 | 95.20% | 31.46% | 21.07% | 7.61% |
| KNHANES only | NHANES 2017-2018 | 0.846 | 0.304 | 61.97% | 82.51% | 20.51% | 17.37% |
| KNHANES only | KNHANES 2011 | 0.896 | 0.427 | 89.40% | 74.07% | 39.86% | 16.98% |
| Native-weight pooled | NHANES 2017-2018 | 0.837 | 0.346 | 90.28% | 54.27% | 12.57% | 10.48% |
| Native-weight pooled | KNHANES 2011 | 0.895 | 0.433 | 97.97% | 33.57% | 22.09% | 8.05% |
| Dataset-balanced pooled | NHANES 2017-2018 | 0.838 | 0.341 | 88.39% | 50.92% | 11.59% | 9.65% |
| Dataset-balanced pooled | KNHANES 2011 | 0.894 | 0.407 | 97.97% | 35.10% | 22.49% | 8.22% |
| Reduced consistent pooled | NHANES 2017-2018 | 0.836 | 0.304 | 93.91% | 47.38% | 11.50% | 9.57% |
| Reduced consistent pooled | KNHANES 2011 | 0.897 | 0.450 | 97.97% | 29.35% | 21.04% | 7.60% |

## Change from the previous temporal result

The previous NHANES 2017-2018 temporal model achieved 83.15% recall, 60.47%
specificity, 13.28% precision, ROC-AUC 0.834, and PR-AUC 0.370. The native-weight pooled
candidate achieved 90.28% recall, 54.27% specificity, 12.57% precision, ROC-AUC 0.837,
and observed PR-AUC 0.381 on the same temporal cohort.

Thus integration detected approximately 7.1 percentage points more cases, but lost 6.2
points of specificity and 0.7 points of precision. This is an improved high-recall
screening operating point, not a simultaneous improvement in recall and precision.

## Recommendation and limitation

If the immediate objective is a single high-recall research screening model across both
surveys, carry forward the native-weight pooled random forest as the candidate. Do not
replace the current production model yet: these two holdouts were used to compare and
rank strategies, so a third independent cohort is still required for an unbiased final
claim. MrOS could provide male-only confirmation; an approved UK Biobank or another
population DXA cohort would be needed for broader confirmation.
