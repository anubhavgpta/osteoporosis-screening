# Feature and model combination results

This additive experiment did not replace the production model or its outputs.
Feature-set, algorithm, hyperparameter, calibration, and threshold selection were
contained inside each outer leave-one-production-cycle-out split.

## Leading feature panel

The leading panel was `E12_baseline` with mean outer PR-AUC 0.7594 and ROC-AUC 0.7900.
Its paired PR-AUC change versus E12 was +0.0000 (95% CI +0.0000 to +0.0000).
This interval includes zero, so the apparent panel gain is exploratory and does not justify replacing E12.

## Best fixed feature-model combination

The highest mean outer score among prespecified fixed candidates was `E12_baseline__spline_logistic_c_1` (PR-AUC 0.7618; ROC-AUC 0.7915). This is an exploratory ranking of outer results, not an unbiased estimate after selecting the winner.

## Panel comparison

| feature_set | outer_cycles | selected_models | mean_roc_auc | mean_pr_auc | mean_log_loss | mean_brier_score | mean_point_sensitivity | minimum_point_sensitivity | mean_point_specificity | mean_point_ppv | mean_point_f1 | mean_lcb_sensitivity | minimum_lcb_sensitivity | mean_lcb_specificity | mean_lcb_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E12_baseline | 4 | catboost_150_depth_4_lr_0.06\|logistic_c_0.3\|spline_logistic_c_0.3\|xgboost_150_depth_2_lr_0.06_child_3 | 0.7900 | 0.7594 | 0.5556 | 0.1871 | 0.8995 | 0.8745 | 0.4252 | 0.5665 | 0.6942 | 0.9145 | 0.8946 | 0.3855 | 0.6893 |
| E12_plus_extended_anthropometric | 4 | catboost_150_depth_4_lr_0.06\|logistic_c_3\|random_forest_150_leaf_5_mf_sqrt\|xgboost_150_depth_2_lr_0.06_child_3 | 0.7861 | 0.7520 | 0.5596 | 0.1885 | 0.8990 | 0.8721 | 0.4141 | 0.5618 | 0.6904 | 0.9126 | 0.8954 | 0.3756 | 0.6851 |
| anthropometric_first | 4 | catboost_150_depth_4_lr_0.06\|spline_logistic_c_1\|stacked_tree_ensemble\|xgboost_150_depth_2_lr_0.06_child_3 | 0.7831 | 0.7515 | 0.5624 | 0.1900 | 0.8983 | 0.8658 | 0.4058 | 0.5582 | 0.6873 | 0.9089 | 0.8827 | 0.3645 | 0.6799 |

## Multi-metric candidates

| candidate | mean_pr_auc | mean_roc_auc | mean_point_sensitivity | mean_point_specificity | mean_point_ppv |
| --- | --- | --- | --- | --- | --- |
| E12_baseline__logistic_c_0.3 | 0.7609 | 0.7917 | 0.9093 | 0.4252 | 0.5692 |
| E12_baseline__logistic_c_3 | 0.7608 | 0.7917 | 0.9093 | 0.4259 | 0.5695 |
| E12_baseline__lasso_selected_logistic_c_0.03 | 0.7608 | 0.7917 | 0.9093 | 0.4258 | 0.5694 |
| E12_baseline__lasso_selected_logistic_c_0.1 | 0.7608 | 0.7917 | 0.9093 | 0.4258 | 0.5694 |
| E12_baseline__logistic_c_1 | 0.7608 | 0.7917 | 0.9093 | 0.4258 | 0.5694 |

## Paired panel changes versus E12

| feature_set | metric | outer_cycles | mean_delta_vs_e12 | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| E12_baseline | pr_auc | 4 | 0.0000 | 0.0000 | 0.0000 |
| E12_plus_extended_anthropometric | pr_auc | 4 | -0.0074 | -0.0227 | 0.0080 |
| anthropometric_first | pr_auc | 4 | -0.0078 | -0.0182 | 0.0026 |
| E12_baseline | roc_auc | 4 | 0.0000 | 0.0000 | 0.0000 |
| E12_plus_extended_anthropometric | roc_auc | 4 | -0.0038 | -0.0132 | 0.0055 |
| anthropometric_first | roc_auc | 4 | -0.0069 | -0.0115 | -0.0022 |

## Clinical-score baselines

Clinical-score baselines were not run because required columns were unavailable.

## Medication limitation

Medication panels were evaluated across all production cycles.
