# Feature and model combination results

This additive experiment did not replace the production model or its outputs.
Feature-set, algorithm, hyperparameter, calibration, and threshold selection were
contained inside each outer leave-one-production-cycle-out split.

## Leading feature panel

The leading panel was `E12_plus_extended_anthropometric` with mean outer PR-AUC 0.7591 and ROC-AUC 0.7889.
Its paired PR-AUC change versus E12 was +0.0001 (95% CI -0.0101 to +0.0103).
This interval includes zero, so the apparent panel gain is exploratory and does not justify replacing E12.

## Best fixed feature-model combination

The highest mean outer score among prespecified fixed candidates was `E12_baseline__spline_logistic_c_0.3` (PR-AUC 0.7650; ROC-AUC 0.7925). This is an exploratory ranking of outer results, not an unbiased estimate after selecting the winner.

## Panel comparison

| feature_set | outer_cycles | selected_models | mean_roc_auc | mean_pr_auc | mean_log_loss | mean_brier_score | mean_point_sensitivity | minimum_point_sensitivity | mean_point_specificity | mean_point_ppv | mean_lcb_sensitivity | minimum_lcb_sensitivity | mean_lcb_specificity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E12_plus_extended_anthropometric | 4 | catboost_150_depth_4_lr_0.06\|spline_logistic_c_0.3\|xgboost_150_depth_2_lr_0.06_child_3\|xgboost_150_depth_3_lr_0.03_child_3 | 0.7889 | 0.7591 | 0.5560 | 0.1873 | 0.9087 | 0.8881 | 0.4215 | 0.5693 | 0.9185 | 0.9042 | 0.3906 |
| E12_baseline | 4 | lasso_selected_logistic_c_0.03\|random_forest_150_leaf_10_mf_sqrt\|stacked_tree_ensemble\|xgboost_150_depth_2_lr_0.06_child_3 | 0.7881 | 0.7590 | 0.5575 | 0.1878 | 0.8997 | 0.8765 | 0.4183 | 0.5655 | 0.9113 | 0.8941 | 0.3822 |
| anthropometric_first | 4 | spline_logistic_c_1\|xgboost_150_depth_2_lr_0.03_child_3\|xgboost_150_depth_2_lr_0.06_child_3\|xgboost_150_depth_3_lr_0.06_child_3 | 0.7823 | 0.7546 | 0.5627 | 0.1901 | 0.9016 | 0.8652 | 0.3946 | 0.5564 | 0.9113 | 0.8853 | 0.3560 |

## Multi-metric candidates

| candidate | mean_pr_auc | mean_roc_auc | mean_point_sensitivity | mean_point_specificity | mean_point_ppv |
| --- | --- | --- | --- | --- | --- |
| E12_baseline__spline_logistic_c_0.3 | 0.7650 | 0.7925 | 0.9061 | 0.4212 | 0.5685 |
| E12_baseline__spline_logistic_c_1 | 0.7647 | 0.7926 | 0.9057 | 0.4203 | 0.5680 |
| E12_baseline__logistic_c_0.3 | 0.7638 | 0.7923 | 0.9088 | 0.4251 | 0.5708 |
| E12_baseline__lasso_selected_logistic_c_0.03 | 0.7637 | 0.7923 | 0.9099 | 0.4252 | 0.5712 |
| E12_baseline__lasso_selected_logistic_c_0.1 | 0.7637 | 0.7923 | 0.9099 | 0.4252 | 0.5712 |
| E12_baseline__logistic_c_1 | 0.7637 | 0.7923 | 0.9099 | 0.4252 | 0.5712 |
| E12_baseline__logistic_c_3 | 0.7637 | 0.7923 | 0.9088 | 0.4227 | 0.5698 |
| E12_plus_extended_anthropometric__spline_logistic_c_0.3 | 0.7633 | 0.7921 | 0.9063 | 0.4233 | 0.5693 |
| E12_plus_extended_anthropometric__spline_logistic_c_1 | 0.7630 | 0.7919 | 0.9071 | 0.4238 | 0.5698 |
| E12_plus_extended_anthropometric__lasso_selected_logistic_c_0.1 | 0.7623 | 0.7931 | 0.9106 | 0.4283 | 0.5727 |
| E12_plus_extended_anthropometric__stacked_tree_ensemble | 0.7623 | 0.7897 | 0.9023 | 0.4230 | 0.5683 |
| E12_plus_extended_anthropometric__logistic_c_0.3 | 0.7620 | 0.7927 | 0.9088 | 0.4287 | 0.5724 |
| E12_plus_extended_anthropometric__lasso_selected_logistic_c_0.03 | 0.7620 | 0.7927 | 0.9088 | 0.4289 | 0.5724 |
| E12_plus_extended_anthropometric__logistic_c_1 | 0.7620 | 0.7927 | 0.9088 | 0.4289 | 0.5724 |
| E12_plus_extended_anthropometric__logistic_c_3 | 0.7620 | 0.7927 | 0.9088 | 0.4289 | 0.5725 |
| E12_baseline__stacked_tree_ensemble | 0.7613 | 0.7901 | 0.9029 | 0.4230 | 0.5683 |
| E12_plus_extended_anthropometric__mlp_32_alpha_0.01 | 0.7612 | 0.7911 | 0.9004 | 0.4284 | 0.5698 |
| E12_plus_extended_anthropometric__catboost_150_depth_4_lr_0.06 | 0.7607 | 0.7898 | 0.9050 | 0.4187 | 0.5670 |
| E12_baseline__mlp_64_32_alpha_0.001 | 0.7596 | 0.7886 | 0.9019 | 0.4203 | 0.5670 |
| E12_baseline__catboost_150_depth_3_lr_0.06 | 0.7594 | 0.7886 | 0.9020 | 0.4216 | 0.5676 |

## Paired panel changes versus E12

| feature_set | metric | outer_cycles | mean_delta_vs_e12 | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| E12_baseline | pr_auc | 4 | 0.0000 | 0.0000 | 0.0000 |
| E12_plus_extended_anthropometric | pr_auc | 4 | 0.0001 | -0.0101 | 0.0103 |
| anthropometric_first | pr_auc | 4 | -0.0044 | -0.0187 | 0.0099 |
| E12_baseline | roc_auc | 4 | 0.0000 | 0.0000 | 0.0000 |
| E12_plus_extended_anthropometric | roc_auc | 4 | 0.0008 | 0.0003 | 0.0014 |
| anthropometric_first | roc_auc | 4 | -0.0059 | -0.0082 | -0.0035 |

## Clinical-score baselines

Clinical-score baselines were not run because required columns were unavailable.

## Medication limitation

Medication panels were evaluated across all production cycles.
