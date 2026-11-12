# Osteoporosis imbalance experiment

This additive experiment tests osteoporosis-only class-weighting, feature-panel, calibration, and threshold policies. It does not replace production outputs.

Production outputs modified: `False`

## Best PR-AUC rows

| candidate | feature_set | model | imbalance_strategy | policy | outer_cycles | mean_roc_auc | mean_pr_auc | mean_log_loss | mean_brier_score | mean_sensitivity | minimum_sensitivity | mean_specificity | mean_ppv | mean_npv | mean_balanced_accuracy | mean_mcc | mean_f1 | mean_referral_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E12_baseline__lightgbm__survey_weight_only | E12_baseline | lightgbm | survey_weight_only | precision_0.30 | 4.0000 | 0.8579 | 0.3880 | 0.1608 | 0.0427 | 0.5327 | 0.4612 | 0.9314 | 0.3121 | 0.9708 | 0.7320 | 0.3598 | 0.3878 | 0.0938 |
| E12_baseline__lightgbm__survey_weight_only | E12_baseline | lightgbm | survey_weight_only | f1_optimized | 4.0000 | 0.8579 | 0.3880 | 0.1608 | 0.0427 | 0.4105 | 0.2468 | 0.9610 | 0.3911 | 0.9643 | 0.6857 | 0.3535 | 0.3765 | 0.0586 |
| E12_baseline__lightgbm__survey_weight_only | E12_baseline | lightgbm | survey_weight_only | precision_0.25 | 4.0000 | 0.8579 | 0.3880 | 0.1608 | 0.0427 | 0.6319 | 0.5793 | 0.8921 | 0.2563 | 0.9759 | 0.7620 | 0.3467 | 0.3603 | 0.1366 |
| E12_baseline__lightgbm__survey_weight_only | E12_baseline | lightgbm | survey_weight_only | precision_0.20 | 4.0000 | 0.8579 | 0.3880 | 0.1608 | 0.0427 | 0.7355 | 0.6774 | 0.8314 | 0.2038 | 0.9810 | 0.7835 | 0.3217 | 0.3160 | 0.1996 |
| E12_baseline__lightgbm__survey_weight_only | E12_baseline | lightgbm | survey_weight_only | recall_0.70 | 4.0000 | 0.8579 | 0.3880 | 0.1608 | 0.0427 | 0.7476 | 0.6774 | 0.8214 | 0.1961 | 0.9817 | 0.7845 | 0.3167 | 0.3086 | 0.2097 |
| E12_baseline__lightgbm__survey_weight_only | E12_baseline | lightgbm | survey_weight_only | recall_0.80 | 4.0000 | 0.8579 | 0.3880 | 0.1608 | 0.0427 | 0.8251 | 0.7727 | 0.7061 | 0.1408 | 0.9851 | 0.7656 | 0.2575 | 0.2394 | 0.3230 |
| E12_baseline__lightgbm__survey_weight_only | E12_baseline | lightgbm | survey_weight_only | default_0.50 | 4.0000 | 0.8579 | 0.3880 | 0.1608 | 0.0427 | 0.1259 | 0.0892 | 0.9974 | 0.7668 | 0.9509 | 0.5616 | 0.2938 | 0.2137 | 0.0093 |
| E12_baseline__lightgbm__survey_weight_only | E12_baseline | lightgbm | survey_weight_only | recall_0.90 | 4.0000 | 0.8579 | 0.3880 | 0.1608 | 0.0427 | 0.9237 | 0.8434 | 0.5094 | 0.0990 | 0.9905 | 0.7166 | 0.1962 | 0.1781 | 0.5143 |

## Best high-recall rows

| candidate | feature_set | model | imbalance_strategy | policy | outer_cycles | mean_roc_auc | mean_pr_auc | mean_log_loss | mean_brier_score | mean_sensitivity | minimum_sensitivity | mean_specificity | mean_ppv | mean_npv | mean_balanced_accuracy | mean_mcc | mean_f1 | mean_referral_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E12_plus_extended_anthropometric__xgboost__survey_weight_only | E12_plus_extended_anthropometric | xgboost | survey_weight_only | recall_0.90 | 4.0000 | 0.8616 | 0.3801 | 0.1600 | 0.0427 | 0.9475 | 0.9115 | 0.5399 | 0.1073 | 0.9941 | 0.7437 | 0.2215 | 0.1924 | 0.4868 |
| E13e_prior_best_chemistry_anthropometric__xgboost__positive_balanced | E13e_prior_best_chemistry_anthropometric | xgboost | positive_balanced | recall_0.90 | 4.0000 | 0.8625 | 0.3724 | 0.1615 | 0.0433 | 0.9374 | 0.9097 | 0.5303 | 0.1048 | 0.9927 | 0.7339 | 0.2128 | 0.1878 | 0.4954 |
| E12_baseline__xgboost__positive_sqrt_balanced | E12_baseline | xgboost | positive_sqrt_balanced | recall_0.90 | 4.0000 | 0.8624 | 0.3789 | 0.1606 | 0.0430 | 0.9304 | 0.9076 | 0.5340 | 0.1043 | 0.9919 | 0.7322 | 0.2106 | 0.1870 | 0.4914 |
| E13e_prior_best_chemistry_anthropometric__lightgbm__survey_weight_only | E13e_prior_best_chemistry_anthropometric | lightgbm | survey_weight_only | recall_0.90 | 4.0000 | 0.8613 | 0.3795 | 0.1601 | 0.0426 | 0.9398 | 0.9031 | 0.4967 | 0.0980 | 0.9922 | 0.7183 | 0.1977 | 0.1772 | 0.5271 |
| E13e_prior_best_chemistry_anthropometric__lightgbm__positive_balanced | E13e_prior_best_chemistry_anthropometric | lightgbm | positive_balanced | recall_0.90 | 4.0000 | 0.8581 | 0.3773 | 0.1628 | 0.0434 | 0.9469 | 0.9063 | 0.4390 | 0.0898 | 0.9926 | 0.6930 | 0.1775 | 0.1638 | 0.5819 |

## Best F1 rows

| candidate | feature_set | model | imbalance_strategy | policy | outer_cycles | mean_roc_auc | mean_pr_auc | mean_log_loss | mean_brier_score | mean_sensitivity | minimum_sensitivity | mean_specificity | mean_ppv | mean_npv | mean_balanced_accuracy | mean_mcc | mean_f1 | mean_referral_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E12_baseline__xgboost__survey_weight_only | E12_baseline | xgboost | survey_weight_only | precision_0.30 | 4.0000 | 0.8626 | 0.3814 | 0.1599 | 0.0427 | 0.5433 | 0.4736 | 0.9345 | 0.3284 | 0.9715 | 0.7389 | 0.3754 | 0.4022 | 0.0914 |
| E12_plus_extended_anthropometric__lightgbm__survey_weight_only | E12_plus_extended_anthropometric | lightgbm | survey_weight_only | f1_optimized | 4.0000 | 0.8581 | 0.3765 | 0.1615 | 0.0430 | 0.4875 | 0.4261 | 0.9459 | 0.3444 | 0.9689 | 0.7167 | 0.3674 | 0.4010 | 0.0778 |
| E12_plus_extended_anthropometric__xgboost__positive_sqrt_balanced | E12_plus_extended_anthropometric | xgboost | positive_sqrt_balanced | f1_optimized | 4.0000 | 0.8618 | 0.3751 | 0.1606 | 0.0430 | 0.5020 | 0.4555 | 0.9408 | 0.3296 | 0.9697 | 0.7214 | 0.3634 | 0.3965 | 0.0835 |
| E12_baseline__random_forest__positive_balanced | E12_baseline | random_forest | positive_balanced | f1_optimized | 4.0000 | 0.8584 | 0.3584 | 0.1622 | 0.0433 | 0.4638 | 0.4068 | 0.9484 | 0.3459 | 0.9676 | 0.7061 | 0.3584 | 0.3938 | 0.0742 |
| E12_baseline__xgboost__positive_sqrt_balanced | E12_baseline | xgboost | positive_sqrt_balanced | f1_optimized | 4.0000 | 0.8624 | 0.3789 | 0.1606 | 0.0430 | 0.5384 | 0.4453 | 0.9328 | 0.3212 | 0.9710 | 0.7356 | 0.3669 | 0.3928 | 0.0926 |
| E13e_prior_best_chemistry_anthropometric__lightgbm__survey_weight_only | E13e_prior_best_chemistry_anthropometric | lightgbm | survey_weight_only | precision_0.30 | 4.0000 | 0.8613 | 0.3795 | 0.1601 | 0.0426 | 0.5124 | 0.4788 | 0.9372 | 0.3234 | 0.9701 | 0.7248 | 0.3616 | 0.3927 | 0.0875 |
| E12_plus_extended_anthropometric__xgboost__positive_sqrt_balanced | E12_plus_extended_anthropometric | xgboost | positive_sqrt_balanced | precision_0.30 | 4.0000 | 0.8618 | 0.3751 | 0.1606 | 0.0430 | 0.5580 | 0.5122 | 0.9263 | 0.3050 | 0.9722 | 0.7422 | 0.3647 | 0.3905 | 0.1001 |
| E13e_prior_best_chemistry_anthropometric__xgboost__positive_sqrt_balanced | E13e_prior_best_chemistry_anthropometric | xgboost | positive_sqrt_balanced | f1_optimized | 4.0000 | 0.8633 | 0.3795 | 0.1599 | 0.0427 | 0.4527 | 0.3529 | 0.9535 | 0.3693 | 0.9666 | 0.7031 | 0.3627 | 0.3905 | 0.0682 |

## Notes

- Positive-class weighting is learned inside each outer/inner training partition.
- Thresholds are selected from calibrated out-of-fold training predictions only.
- SMOTE/ADASYN/random oversampling are included only when `imbalanced-learn` is installed.
- F1-optimized rows are useful as threshold-policy diagnostics, not as a screening default.
