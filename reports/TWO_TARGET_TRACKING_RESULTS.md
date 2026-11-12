# Two-target metric tracker

This report tracks additive experiments against the frozen rigorous baseline for each target.
Production artifacts are not modified by this tracker.

## Feature-panel deltas

| source | target | candidate | roc_auc | pr_auc | sensitivity | specificity | ppv | f1 | delta_roc_auc_vs_baseline | delta_pr_auc_vs_baseline | delta_ppv_vs_baseline | delta_f1_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low_bmd_feature_combination | low_bmd | E12_plus_extended_anthropometric | 0.7889 | 0.7591 | 0.9087 | 0.4215 | 0.5693 | 0.6992 | -0.0012 | -0.0009 | 0.0031 | 0.0034 |
| low_bmd_feature_combination | low_bmd | E12_baseline | 0.7881 | 0.7590 | 0.8997 | 0.4183 | 0.5655 | 0.6937 | -0.0020 | -0.0010 | -0.0007 | -0.0021 |
| low_bmd_feature_combination | low_bmd | anthropometric_first | 0.7823 | 0.7546 | 0.9016 | 0.3946 | 0.5564 | 0.6868 | -0.0079 | -0.0054 | -0.0098 | -0.0090 |
| feature_combination | osteoporosis | E13e_chemistry_plus_anthropometric | 0.8598 | 0.3745 | 0.9126 | 0.5516 | 0.1068 | 0.1906 | 0.0027 | 0.0197 | -0.0013 | -0.0014 |
| feature_combination | osteoporosis | E13c_albumin_alp_ratio | 0.8573 | 0.3683 | 0.8933 | 0.5995 | 0.1157 | 0.2044 | 0.0002 | 0.0136 | 0.0076 | 0.0124 |
| feature_combination | osteoporosis | E12_baseline | 0.8609 | 0.3656 | 0.9210 | 0.5545 | 0.1087 | 0.1934 | 0.0038 | 0.0109 | 0.0006 | 0.0015 |
| feature_combination | osteoporosis | E13b_alp_only | 0.8607 | 0.3642 | 0.9127 | 0.5708 | 0.1105 | 0.1964 | 0.0036 | 0.0094 | 0.0024 | 0.0045 |
| feature_combination | osteoporosis | E13d_extended_anthropometric | 0.8589 | 0.3636 | 0.9229 | 0.5510 | 0.1069 | 0.1911 | 0.0018 | 0.0089 | -0.0012 | -0.0009 |
| feature_combination | osteoporosis | E13a_albumin_only | 0.8611 | 0.3623 | 0.9320 | 0.5367 | 0.1054 | 0.1889 | 0.0040 | 0.0076 | -0.0027 | -0.0031 |

## Baseline rows

| source | target | scope | candidate | roc_auc | pr_auc | sensitivity | specificity | ppv | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target_definition | low_bmd | naive_random_80_20_unweighted_leaky_selection | naive_naive_random_80_20_unweighted_leaky_selection | 0.7976 | 0.7726 | 0.8910 | 0.4809 | 0.5891 | 0.7093 |
| target_definition | osteoporosis | naive_random_80_20_unweighted_leaky_selection | naive_naive_random_80_20_unweighted_leaky_selection | 0.8204 | 0.3661 | 0.8393 | 0.6646 | 0.1372 | 0.2359 |
| target_definition | low_bmd | leave_one_production_cycle_out | rigorous_leave_one_production_cycle_out | 0.7901 | 0.7601 | 0.9052 | 0.4164 | 0.5662 | 0.6958 |
| target_definition | osteoporosis | leave_one_production_cycle_out | rigorous_leave_one_production_cycle_out | 0.8571 | 0.3547 | 0.9141 | 0.5486 | 0.1081 | 0.1919 |
| target_definition_temporal | osteoporosis | held_out_cycle_2017-2018 | xgboost_sensitivity_oriented | 0.8335 | 0.3695 | 0.8315 | 0.6047 | 0.1328 | 0.2290 |
| target_definition_temporal | osteoporosis | held_out_cycle_2017-2018 | xgboost_f1_optimized | 0.8335 | 0.3695 | 0.5078 | 0.9369 | 0.3693 | 0.4276 |
| target_definition_temporal | low_bmd | held_out_cycle_2017-2018 | regularized_logistic_sensitivity_oriented | 0.7974 | 0.7936 | 0.8985 | 0.4345 | 0.5946 | 0.7156 |
| target_definition_temporal | low_bmd | held_out_cycle_2017-2018 | regularized_logistic_f1_optimized | 0.7974 | 0.7936 | 0.7959 | 0.6347 | 0.6679 | 0.7263 |
