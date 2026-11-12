# Low-BMD pipeline comparison

These low-BMD results are sensitivity and target-definition comparisons. They do not replace the production osteoporosis model, thresholds, reports, or saved artifacts.

## Successfully mirrored pipelines

| pipeline_family | validation_design | pipeline | osteoporosis_pr_auc | low_bmd_pr_auc | delta_pr_auc_low_bmd_minus_osteoporosis | osteoporosis_roc_auc | low_bmd_roc_auc | delta_roc_auc_low_bmd_minus_osteoporosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E12_baseline | 2017_2018_temporal | xgboost_f1_optimized | 0.3695 | 0.7961 | 0.4265 | 0.8335 | 0.7974 | -0.0361 |
| E12_baseline | 2017_2018_temporal | xgboost_sensitivity_oriented | 0.3695 | 0.7961 | 0.4265 | 0.8335 | 0.7974 | -0.0361 |
| E12_baseline | leave_one_production_cycle_out | rigorous_leave_one_production_cycle_out | 0.3547 | 0.7615 | 0.4067 | 0.8571 | 0.7917 | -0.0653 |
| E12_baseline | naive_random_80_20_unweighted_leaky_selection | naive_naive_random_80_20_unweighted_leaky_selection | 0.3661 | 0.7588 | 0.3928 | 0.8204 | 0.7940 | -0.0264 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__catboost_150_depth_3_lr_0.03 | 0.3342 | 0.7524 | 0.4183 | 0.8600 | 0.7845 | -0.0755 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__catboost_150_depth_3_lr_0.06 | 0.3380 | 0.7582 | 0.4203 | 0.8612 | 0.7898 | -0.0714 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__catboost_150_depth_4_lr_0.03 | 0.3429 | 0.7564 | 0.4135 | 0.8560 | 0.7872 | -0.0688 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__catboost_150_depth_4_lr_0.06 | 0.3407 | 0.7594 | 0.4186 | 0.8585 | 0.7908 | -0.0677 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__lasso_selected_logistic_c_0.03 | 0.3568 | 0.7608 | 0.4040 | 0.8600 | 0.7917 | -0.0683 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__lasso_selected_logistic_c_0.1 | 0.3568 | 0.7608 | 0.4040 | 0.8600 | 0.7917 | -0.0683 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__lightgbm_150_leaves_15_lr_0.03 | 0.3880 | 0.7510 | 0.3630 | 0.8579 | 0.7846 | -0.0734 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__lightgbm_150_leaves_15_lr_0.06 | 0.3689 | 0.7462 | 0.3773 | 0.8515 | 0.7802 | -0.0713 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__lightgbm_150_leaves_7_lr_0.03 | 0.3780 | 0.7583 | 0.3802 | 0.8578 | 0.7873 | -0.0705 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__lightgbm_150_leaves_7_lr_0.06 | 0.3830 | 0.7556 | 0.3726 | 0.8578 | 0.7866 | -0.0713 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__logistic_c_0.3 | 0.3567 | 0.7609 | 0.4042 | 0.8600 | 0.7917 | -0.0683 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__logistic_c_1 | 0.3568 | 0.7608 | 0.4040 | 0.8600 | 0.7917 | -0.0683 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__logistic_c_3 | 0.3564 | 0.7608 | 0.4044 | 0.8600 | 0.7917 | -0.0683 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__mlp_32_alpha_0.001 | 0.3424 | 0.7584 | 0.4159 | 0.8599 | 0.7895 | -0.0705 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__mlp_32_alpha_0.01 | 0.3470 | 0.7585 | 0.4115 | 0.8600 | 0.7895 | -0.0705 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__mlp_64_32_alpha_0.001 | 0.3455 | 0.7554 | 0.4099 | 0.8673 | 0.7892 | -0.0781 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__mlp_64_32_alpha_0.01 | 0.3438 | 0.7605 | 0.4167 | 0.8667 | 0.7911 | -0.0756 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__random_forest_150_leaf_10_mf_sqrt | 0.3697 | 0.7556 | 0.3860 | 0.8577 | 0.7873 | -0.0704 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__random_forest_150_leaf_20_mf_sqrt | 0.3649 | 0.7580 | 0.3931 | 0.8579 | 0.7873 | -0.0707 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__random_forest_150_leaf_5_mf_sqrt | 0.3664 | 0.7515 | 0.3850 | 0.8523 | 0.7860 | -0.0663 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__spline_logistic_c_0.3 | 0.3527 | 0.7617 | 0.4090 | 0.8586 | 0.7915 | -0.0671 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__spline_logistic_c_1 | 0.3522 | 0.7618 | 0.4097 | 0.8587 | 0.7915 | -0.0672 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__stacked_tree_ensemble | 0.3718 | 0.7579 | 0.3861 | 0.8635 | 0.7894 | -0.0741 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__xgboost_150_depth_2_lr_0.03_child_3 | 0.3686 | 0.7591 | 0.3905 | 0.8589 | 0.7875 | -0.0714 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__xgboost_150_depth_2_lr_0.03_child_5 | 0.3686 | 0.7591 | 0.3905 | 0.8589 | 0.7875 | -0.0714 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__xgboost_150_depth_2_lr_0.06_child_3 | 0.3748 | 0.7613 | 0.3864 | 0.8610 | 0.7901 | -0.0709 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__xgboost_150_depth_2_lr_0.06_child_5 | 0.3748 | 0.7613 | 0.3864 | 0.8610 | 0.7901 | -0.0709 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__xgboost_150_depth_3_lr_0.03_child_3 | 0.3814 | 0.7617 | 0.3803 | 0.8626 | 0.7898 | -0.0728 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__xgboost_150_depth_3_lr_0.03_child_5 | 0.3814 | 0.7617 | 0.3803 | 0.8626 | 0.7898 | -0.0728 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__xgboost_150_depth_3_lr_0.06_child_3 | 0.3786 | 0.7600 | 0.3814 | 0.8628 | 0.7878 | -0.0750 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_baseline__xgboost_150_depth_3_lr_0.06_child_5 | 0.3786 | 0.7600 | 0.3814 | 0.8628 | 0.7878 | -0.0750 |
| feature_combination_panel | leave_one_production_cycle_out | E12_baseline | 0.3656 | 0.7594 | 0.3937 | 0.8609 | 0.7900 | -0.0709 |

## Low-BMD-only detail

- Detailed low-BMD metric rows: 603
- Threshold rows: 414
- Calibration rows: 603

## Not mirrored or target-specific

Some pipelines are intentionally target-specific or require frozen external predictions that do not exist for low BMD. Clinical-score baselines and KNHANES recalibration remain osteoporosis-only unless corresponding low-BMD external predictions are generated with the same leakage-safe validation design.

Osteoporosis-only comparison rows:

| pipeline_family | validation_design | pipeline |
| --- | --- | --- |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__catboost_150_depth_3_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__catboost_150_depth_3_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__catboost_150_depth_4_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__catboost_150_depth_4_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__lasso_selected_logistic_c_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__lasso_selected_logistic_c_0.1 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__lightgbm_150_leaves_15_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__lightgbm_150_leaves_15_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__lightgbm_150_leaves_7_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__lightgbm_150_leaves_7_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__logistic_c_0.3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__logistic_c_1 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__logistic_c_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__mlp_32_alpha_0.001 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__mlp_32_alpha_0.01 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__mlp_64_32_alpha_0.001 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__mlp_64_32_alpha_0.01 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__random_forest_150_leaf_10_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__random_forest_150_leaf_20_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__random_forest_150_leaf_5_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__spline_logistic_c_0.3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__spline_logistic_c_1 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__stacked_tree_ensemble |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__xgboost_150_depth_2_lr_0.03_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__xgboost_150_depth_2_lr_0.03_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__xgboost_150_depth_2_lr_0.06_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__xgboost_150_depth_2_lr_0.06_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__xgboost_150_depth_3_lr_0.03_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__xgboost_150_depth_3_lr_0.03_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__xgboost_150_depth_3_lr_0.06_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13a_albumin_only__xgboost_150_depth_3_lr_0.06_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__catboost_150_depth_3_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__catboost_150_depth_3_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__catboost_150_depth_4_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__catboost_150_depth_4_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__lasso_selected_logistic_c_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__lasso_selected_logistic_c_0.1 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__lightgbm_150_leaves_15_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__lightgbm_150_leaves_15_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__lightgbm_150_leaves_7_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__lightgbm_150_leaves_7_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__logistic_c_0.3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__logistic_c_1 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__logistic_c_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__mlp_32_alpha_0.001 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__mlp_32_alpha_0.01 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__mlp_64_32_alpha_0.001 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__mlp_64_32_alpha_0.01 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__random_forest_150_leaf_10_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__random_forest_150_leaf_20_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__random_forest_150_leaf_5_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__spline_logistic_c_0.3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__spline_logistic_c_1 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__stacked_tree_ensemble |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__xgboost_150_depth_2_lr_0.03_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__xgboost_150_depth_2_lr_0.03_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__xgboost_150_depth_2_lr_0.06_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__xgboost_150_depth_2_lr_0.06_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__xgboost_150_depth_3_lr_0.03_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__xgboost_150_depth_3_lr_0.03_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__xgboost_150_depth_3_lr_0.06_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13b_alp_only__xgboost_150_depth_3_lr_0.06_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__catboost_150_depth_3_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__catboost_150_depth_3_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__catboost_150_depth_4_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__catboost_150_depth_4_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__lasso_selected_logistic_c_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__lasso_selected_logistic_c_0.1 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__lightgbm_150_leaves_15_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__lightgbm_150_leaves_15_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__lightgbm_150_leaves_7_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__lightgbm_150_leaves_7_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__logistic_c_0.3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__logistic_c_1 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__logistic_c_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__mlp_32_alpha_0.001 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__mlp_32_alpha_0.01 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__mlp_64_32_alpha_0.001 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__mlp_64_32_alpha_0.01 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__random_forest_150_leaf_10_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__random_forest_150_leaf_20_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__random_forest_150_leaf_5_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__spline_logistic_c_0.3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__spline_logistic_c_1 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__stacked_tree_ensemble |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__xgboost_150_depth_2_lr_0.03_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__xgboost_150_depth_2_lr_0.03_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__xgboost_150_depth_2_lr_0.06_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__xgboost_150_depth_2_lr_0.06_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__xgboost_150_depth_3_lr_0.03_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__xgboost_150_depth_3_lr_0.03_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__xgboost_150_depth_3_lr_0.06_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13c_albumin_alp_ratio__xgboost_150_depth_3_lr_0.06_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__catboost_150_depth_3_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__catboost_150_depth_3_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__catboost_150_depth_4_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__catboost_150_depth_4_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__lasso_selected_logistic_c_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__lasso_selected_logistic_c_0.1 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__lightgbm_150_leaves_15_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__lightgbm_150_leaves_15_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__lightgbm_150_leaves_7_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__lightgbm_150_leaves_7_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__logistic_c_0.3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__logistic_c_1 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__logistic_c_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__mlp_32_alpha_0.001 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__mlp_32_alpha_0.01 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__mlp_64_32_alpha_0.001 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__mlp_64_32_alpha_0.01 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__random_forest_150_leaf_10_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__random_forest_150_leaf_20_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__random_forest_150_leaf_5_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__spline_logistic_c_0.3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__spline_logistic_c_1 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__stacked_tree_ensemble |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__xgboost_150_depth_2_lr_0.03_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__xgboost_150_depth_2_lr_0.03_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__xgboost_150_depth_2_lr_0.06_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__xgboost_150_depth_2_lr_0.06_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__xgboost_150_depth_3_lr_0.03_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__xgboost_150_depth_3_lr_0.03_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__xgboost_150_depth_3_lr_0.06_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13d_extended_anthropometric__xgboost_150_depth_3_lr_0.06_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__catboost_150_depth_3_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__catboost_150_depth_3_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__catboost_150_depth_4_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__catboost_150_depth_4_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__lasso_selected_logistic_c_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__lasso_selected_logistic_c_0.1 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__lightgbm_150_leaves_15_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__lightgbm_150_leaves_15_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__lightgbm_150_leaves_7_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__lightgbm_150_leaves_7_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__logistic_c_0.3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__logistic_c_1 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__logistic_c_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__mlp_32_alpha_0.001 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__mlp_32_alpha_0.01 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__mlp_64_32_alpha_0.001 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__mlp_64_32_alpha_0.01 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__random_forest_150_leaf_10_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__random_forest_150_leaf_20_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__random_forest_150_leaf_5_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__spline_logistic_c_0.3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__spline_logistic_c_1 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__stacked_tree_ensemble |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__xgboost_150_depth_2_lr_0.03_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__xgboost_150_depth_2_lr_0.03_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__xgboost_150_depth_2_lr_0.06_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__xgboost_150_depth_2_lr_0.06_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__xgboost_150_depth_3_lr_0.03_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__xgboost_150_depth_3_lr_0.03_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__xgboost_150_depth_3_lr_0.06_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric__xgboost_150_depth_3_lr_0.06_child_5 |
| feature_combination_panel | leave_one_production_cycle_out | E13a_albumin_only |
| feature_combination_panel | leave_one_production_cycle_out | E13b_alp_only |
| feature_combination_panel | leave_one_production_cycle_out | E13c_albumin_alp_ratio |
| feature_combination_panel | leave_one_production_cycle_out | E13d_extended_anthropometric |
| feature_combination_panel | leave_one_production_cycle_out | E13e_chemistry_plus_anthropometric |

Low-BMD-only comparison rows:

| pipeline_family | validation_design | pipeline |
| --- | --- | --- |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__catboost_150_depth_3_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__catboost_150_depth_3_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__catboost_150_depth_4_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__catboost_150_depth_4_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__lasso_selected_logistic_c_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__lasso_selected_logistic_c_0.1 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__lightgbm_150_leaves_15_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__lightgbm_150_leaves_15_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__lightgbm_150_leaves_7_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__lightgbm_150_leaves_7_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__logistic_c_0.3 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__logistic_c_1 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__logistic_c_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__mlp_32_alpha_0.001 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__mlp_32_alpha_0.01 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__mlp_64_32_alpha_0.001 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__mlp_64_32_alpha_0.01 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__random_forest_150_leaf_10_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__random_forest_150_leaf_20_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__random_forest_150_leaf_5_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__spline_logistic_c_0.3 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__spline_logistic_c_1 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__stacked_tree_ensemble |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__xgboost_150_depth_2_lr_0.03_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__xgboost_150_depth_2_lr_0.03_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__xgboost_150_depth_2_lr_0.06_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__xgboost_150_depth_2_lr_0.06_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__xgboost_150_depth_3_lr_0.03_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__xgboost_150_depth_3_lr_0.03_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__xgboost_150_depth_3_lr_0.06_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | E12_plus_extended_anthropometric__xgboost_150_depth_3_lr_0.06_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__catboost_150_depth_3_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__catboost_150_depth_3_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__catboost_150_depth_4_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__catboost_150_depth_4_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__lasso_selected_logistic_c_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__lasso_selected_logistic_c_0.1 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__lightgbm_150_leaves_15_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__lightgbm_150_leaves_15_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__lightgbm_150_leaves_7_lr_0.03 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__lightgbm_150_leaves_7_lr_0.06 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__logistic_c_0.3 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__logistic_c_1 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__logistic_c_3 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__mlp_32_alpha_0.001 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__mlp_32_alpha_0.01 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__mlp_64_32_alpha_0.001 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__mlp_64_32_alpha_0.01 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__random_forest_150_leaf_10_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__random_forest_150_leaf_20_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__random_forest_150_leaf_5_mf_sqrt |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__spline_logistic_c_0.3 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__spline_logistic_c_1 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__stacked_tree_ensemble |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__xgboost_150_depth_2_lr_0.03_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__xgboost_150_depth_2_lr_0.03_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__xgboost_150_depth_2_lr_0.06_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__xgboost_150_depth_2_lr_0.06_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__xgboost_150_depth_3_lr_0.03_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__xgboost_150_depth_3_lr_0.03_child_5 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__xgboost_150_depth_3_lr_0.06_child_3 |
| feature_combination_candidate | leave_one_production_cycle_out | anthropometric_first__xgboost_150_depth_3_lr_0.06_child_5 |
| feature_combination_panel | leave_one_production_cycle_out | E12_plus_extended_anthropometric |
| feature_combination_panel | leave_one_production_cycle_out | anthropometric_first |
