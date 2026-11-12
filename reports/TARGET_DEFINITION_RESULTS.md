# Target-definition and methodology sensitivity results

**The naive arm is a non-deployable bias diagnostic only. It must never be used for deployment, threshold selection, or production promotion.**

This additive experiment compares the unchanged femoral-neck osteoporosis target with an inclusive low-BMD target, and compares the rigorous pipeline with a deliberately unweighted, leaky published-paper-style arm.

## Results

| target | methodology | evaluation_design | roc_auc | pr_auc | sensitivity | specificity | ppv | f1 | calibration_slope | calibration_intercept | brier_score | threshold | evaluations | evaluation_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low_bmd | naive | naive_random_80_20_unweighted_leaky_selection | 0.7976 | 0.7726 | 0.8910 | 0.4809 | 0.5891 | 0.7093 | 0.9838 | 0.0005 | 0.1823 | 0.2784 | 1 | 1874 |
| osteoporosis | naive | naive_random_80_20_unweighted_leaky_selection | 0.8204 | 0.3661 | 0.8393 | 0.6646 | 0.1372 | 0.2359 | 0.7970 | -0.4016 | 0.0467 | 0.0337 | 1 | 1874 |
| low_bmd | rigorous | leave_one_production_cycle_out | 0.7901 | 0.7601 | 0.9052 | 0.4164 | 0.5662 | 0.6958 | 1.0256 | -0.0163 | 0.1867 | 0.2629 | 4 | 9367 |
| osteoporosis | rigorous | leave_one_production_cycle_out | 0.8571 | 0.3547 | 0.9141 | 0.5486 | 0.1081 | 0.1919 | 1.0971 | 0.1329 | 0.0435 | 0.0249 | 4 | 9367 |

## Decomposition

| metric | target_effect | methodology_effect | combined_effect | interaction_or_residual | target_fraction_of_combined_gap | methodology_fraction_of_combined_gap |
| --- | --- | --- | --- | --- | --- | --- |
| roc_auc | -0.0669 | -0.0367 | -0.0595 | 0.0441 | 1.1260 | 0.6165 |
| pr_auc | 0.4053 | 0.0113 | 0.4178 | 0.0012 | 0.9701 | 0.0271 |
| f1 | 0.5038 | 0.0439 | 0.5173 | -0.0304 | 0.9739 | 0.0849 |

Of the -0.0595-point typical-paper AUC gap, -0.0669 points come from the easier low-BMD target and -0.0367 points come from unweighted/leaky methodology; the residual interaction is +0.0441.

Of the +0.4178-point typical-paper PR-AUC gap, +0.4053 points come from the easier low-BMD target and +0.0113 points come from unweighted/leaky methodology; the residual interaction is +0.0012.

## Temporal rigorous holdout

| target | methodology | model | evaluation | evaluation_scope | roc_auc | pr_auc | sensitivity | specificity | ppv | f1 | calibration_slope | calibration_intercept | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| osteoporosis | rigorous | xgboost | sensitivity_oriented | held_out_cycle_2017-2018 | 0.8335 | 0.3695 | 0.8315 | 0.6047 | 0.1328 | 0.2290 | 1.2064 | 0.9579 | 0.0543 |
| osteoporosis | rigorous | xgboost | f1_optimized | held_out_cycle_2017-2018 | 0.8335 | 0.3695 | 0.5078 | 0.9369 | 0.3693 | 0.4276 | 1.2064 | 0.9579 | 0.0543 |
| low_bmd | rigorous | regularized_logistic | sensitivity_oriented | held_out_cycle_2017-2018 | 0.7974 | 0.7936 | 0.8985 | 0.4345 | 0.5946 | 0.7156 | 1.0575 | 0.0938 | 0.1837 |
| low_bmd | rigorous | regularized_logistic | f1_optimized | held_out_cycle_2017-2018 | 0.7974 | 0.7936 | 0.7959 | 0.6347 | 0.6679 | 0.7263 | 1.0575 | 0.0938 | 0.1837 |

## Decision

Diagnostic only, not eligible for production promotion.
