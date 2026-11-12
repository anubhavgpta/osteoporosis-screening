# Feature and model combination results

This additive experiment did not replace the production model or its outputs.
Feature-set, algorithm, hyperparameter, calibration, and threshold selection were
contained inside each outer leave-one-production-cycle-out split.

## Leading feature panel

The leading panel was `E13e_chemistry_plus_anthropometric` with mean outer PR-AUC 0.3745 and ROC-AUC 0.8598.
Its paired PR-AUC change versus E12 was +0.0088 (95% CI -0.0296 to +0.0472).
This interval includes zero, so the apparent panel gain is exploratory and does not justify replacing E12.

## Best fixed feature-model combination

The highest mean outer score among prespecified fixed candidates was `E12_baseline__lightgbm_150_leaves_15_lr_0.03` (PR-AUC 0.3880; ROC-AUC 0.8579). This is an exploratory ranking of outer results, not an unbiased estimate after selecting the winner.

## Panel comparison

| feature_set | outer_cycles | selected_models | mean_roc_auc | mean_pr_auc | mean_log_loss | mean_brier_score | mean_point_sensitivity | minimum_point_sensitivity | mean_point_specificity | mean_point_ppv | mean_lcb_sensitivity | minimum_lcb_sensitivity | mean_lcb_specificity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E13e_chemistry_plus_anthropometric | 4 | lightgbm_150_leaves_7_lr_0.03\|random_forest_150_leaf_5_mf_sqrt\|xgboost_150_depth_3_lr_0.03_child_3 | 0.8598 | 0.3745 | 0.1605 | 0.0427 | 0.9126 | 0.8352 | 0.5516 | 0.1068 | 0.9422 | 0.8783 | 0.4750 |
| E13c_albumin_alp_ratio | 4 | mlp_64_32_alpha_0.001\|random_forest_150_leaf_5_mf_sqrt\|stacked_tree_ensemble\|xgboost_150_depth_3_lr_0.03_child_3 | 0.8573 | 0.3683 | 0.1622 | 0.0431 | 0.8933 | 0.8208 | 0.5995 | 0.1157 | 0.9420 | 0.9129 | 0.4956 |
| E12_baseline | 4 | lightgbm_150_leaves_15_lr_0.03\|lightgbm_150_leaves_7_lr_0.03\|logistic_c_3\|xgboost_150_depth_3_lr_0.03_child_3 | 0.8609 | 0.3656 | 0.1615 | 0.0432 | 0.9210 | 0.8630 | 0.5545 | 0.1087 | 0.9552 | 0.9352 | 0.4488 |
| E13b_alp_only | 4 | lasso_selected_logistic_c_0.03\|lightgbm_150_leaves_15_lr_0.03\|xgboost_150_depth_3_lr_0.03_child_3 | 0.8607 | 0.3642 | 0.1608 | 0.0429 | 0.9127 | 0.8806 | 0.5708 | 0.1105 | 0.9300 | 0.8892 | 0.4841 |
| E13d_extended_anthropometric | 4 | lightgbm_150_leaves_7_lr_0.03\|logistic_c_0.3\|mlp_32_alpha_0.01\|stacked_tree_ensemble | 0.8589 | 0.3636 | 0.1625 | 0.0431 | 0.9229 | 0.8541 | 0.5510 | 0.1069 | 0.9492 | 0.8765 | 0.4505 |
| E13a_albumin_only | 4 | lasso_selected_logistic_c_0.03\|lightgbm_150_leaves_15_lr_0.03\|lightgbm_150_leaves_7_lr_0.03\|xgboost_150_depth_3_lr_0.03_child_3 | 0.8611 | 0.3623 | 0.1618 | 0.0433 | 0.9320 | 0.9009 | 0.5367 | 0.1054 | 0.9482 | 0.9254 | 0.4688 |

## Multi-metric candidates

| candidate | mean_pr_auc | mean_roc_auc | mean_point_sensitivity | mean_point_specificity | mean_point_ppv |
| --- | --- | --- | --- | --- | --- |
| E13c_albumin_alp_ratio__stacked_tree_ensemble | 0.3797 | 0.8666 | 0.9289 | 0.5860 | 0.1157 |
| E13a_albumin_only__stacked_tree_ensemble | 0.3684 | 0.8641 | 0.9267 | 0.5809 | 0.1141 |

## Paired panel changes versus E12

| feature_set | metric | outer_cycles | mean_delta_vs_e12 | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| E12_baseline | pr_auc | 4 | 0.0000 | 0.0000 | 0.0000 |
| E13a_albumin_only | pr_auc | 4 | -0.0033 | -0.0157 | 0.0091 |
| E13b_alp_only | pr_auc | 4 | -0.0015 | -0.0344 | 0.0315 |
| E13c_albumin_alp_ratio | pr_auc | 4 | 0.0027 | -0.0257 | 0.0311 |
| E13d_extended_anthropometric | pr_auc | 4 | -0.0020 | -0.0425 | 0.0384 |
| E13e_chemistry_plus_anthropometric | pr_auc | 4 | 0.0088 | -0.0296 | 0.0472 |
| E12_baseline | roc_auc | 4 | 0.0000 | 0.0000 | 0.0000 |
| E13a_albumin_only | roc_auc | 4 | 0.0002 | -0.0062 | 0.0066 |
| E13b_alp_only | roc_auc | 4 | -0.0002 | -0.0076 | 0.0071 |
| E13c_albumin_alp_ratio | roc_auc | 4 | -0.0036 | -0.0095 | 0.0024 |
| E13d_extended_anthropometric | roc_auc | 4 | -0.0020 | -0.0279 | 0.0239 |
| E13e_chemistry_plus_anthropometric | roc_auc | 4 | -0.0011 | -0.0180 | 0.0158 |

## Clinical-score baselines

| score | scope | policy | outer_cycles | mean_roc_auc | mean_pr_auc | mean_sensitivity | minimum_sensitivity | mean_specificity | mean_ppv | mean_npv | mean_balanced_accuracy | mean_mcc | mean_f1 | mean_referral_rate | mean_log_loss | mean_brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| osta | female_only | lcb_recall | 4 | 0.8027 | 0.3566 | 0.9400 | 0.8998 | 0.3356 | 0.1222 | 0.9821 | 0.6378 | 0.1689 | 0.2155 | 0.6890 | 0.2540 | 0.0707 |
| osta | female_only | point_recall | 4 | 0.8027 | 0.3566 | 0.9283 | 0.8782 | 0.4104 | 0.1344 | 0.9825 | 0.6693 | 0.1984 | 0.2338 | 0.6200 | 0.2540 | 0.0707 |
| osta | all | lcb_recall | 4 | 0.8383 | 0.3100 | 0.9390 | 0.9083 | 0.4511 | 0.0915 | 0.9918 | 0.6950 | 0.1798 | 0.1661 | 0.5706 | 0.1713 | 0.0453 |
| osta | all | point_recall | 4 | 0.8383 | 0.3100 | 0.9139 | 0.8938 | 0.5372 | 0.1037 | 0.9905 | 0.7256 | 0.2053 | 0.1857 | 0.4877 | 0.1713 | 0.0453 |
| frax_grade_risk_count | female_only | lcb_recall | 4 | 0.7769 | 0.2762 | 0.9655 | 0.9456 | 0.2682 | 0.1148 | 0.9870 | 0.6169 | 0.1533 | 0.2047 | 0.7526 | 0.2642 | 0.0748 |
| frax_grade_risk_count | female_only | point_recall | 4 | 0.7769 | 0.2762 | 0.9172 | 0.8699 | 0.3924 | 0.1303 | 0.9780 | 0.6548 | 0.1825 | 0.2268 | 0.6353 | 0.2642 | 0.0748 |
| frax_grade_risk_count | all | lcb_recall | 4 | 0.7715 | 0.1838 | 0.9440 | 0.9121 | 0.2811 | 0.0716 | 0.9880 | 0.6126 | 0.1155 | 0.1327 | 0.7313 | 0.1908 | 0.0495 |
| frax_grade_risk_count | all | point_recall | 4 | 0.7715 | 0.1838 | 0.9009 | 0.8382 | 0.4088 | 0.0826 | 0.9853 | 0.6548 | 0.1445 | 0.1506 | 0.6082 | 0.1908 | 0.0495 |

## Medication limitation

Medication panels were not scored because prescription files were not available for every production cycle; partial-cycle missingness was not imputed into a false model comparison. The aggregation code excludes osteoporosis treatments and reported indications, and RXDDAYS supplies duration rather than dose.

## KNHANES nested local recalibration

| strategy | years | mean_sensitivity | minimum_sensitivity | mean_specificity | mean_ppv | mean_referral_rate | mean_log_loss | mean_brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_nhanes_threshold | 4 | 0.9870 | 0.9638 | 0.2783 | 0.2175 | 0.7664 | 0.3216 | 0.0982 |
| knhanes_year_nested_recalibration | 4 | 0.8957 | 0.8769 | 0.6666 | 0.3537 | 0.4284 | 0.3026 | 0.0913 |
| knhanes_year_nested_recalibration_lcb | 4 | 0.9128 | 0.9023 | 0.6293 | 0.3343 | 0.4623 | 0.3026 | 0.0913 |

Nested Korean recalibration reduced mean referral from 76.6% to 42.8%, while mean sensitivity fell from 98.7% to 89.6% (minimum 87.7%). It therefore improves burden and calibration but does not maintain 90% sensitivity in every held-out year.
The lower-confidence-bound variant referred 46.2%, with mean sensitivity 91.3% and minimum sensitivity 90.2%.
