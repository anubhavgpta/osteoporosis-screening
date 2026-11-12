# Target-definition and methodology sensitivity results

**The naive arm is a non-deployable bias diagnostic only. It must never be used for deployment, threshold selection, or production promotion.**

This additive experiment compares the unchanged femoral-neck osteoporosis target with a low-BMD target, and compares the rigorous pipeline with a deliberately unweighted, leaky published-paper-style arm.

## Results

| target | methodology | evaluation_design | roc_auc | pr_auc | sensitivity | specificity | ppv | f1 | calibration_slope | calibration_intercept | brier_score | threshold | evaluations | evaluation_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low_bmd | naive | naive_random_80_20_unweighted_leaky_selection | 0.7940 | 0.7588 | 0.9022 | 0.4634 | 0.5821 | 0.7076 | 0.9477 | -0.0430 | 0.1840 | 0.2738 | 1 | 1874 |
| osteoporosis | naive | naive_random_80_20_unweighted_leaky_selection | 0.8204 | 0.3661 | 0.8393 | 0.6646 | 0.1372 | 0.2359 | 0.7970 | -0.4016 | 0.0467 | 0.0337 | 1 | 1874 |
| low_bmd | rigorous | leave_one_production_cycle_out | 0.7917 | 0.7615 | 0.9052 | 0.4243 | 0.5676 | 0.6969 | 1.0507 | -0.0077 | 0.1860 | 0.2670 | 4 | 9367 |
| osteoporosis | rigorous | leave_one_production_cycle_out | 0.8571 | 0.3547 | 0.9141 | 0.5486 | 0.1081 | 0.1919 | 1.0971 | 0.1329 | 0.0435 | 0.0249 | 4 | 9367 |

## Decomposition

| metric | target_effect | methodology_effect | combined_effect | interaction_or_residual | target_fraction_of_combined_gap | methodology_fraction_of_combined_gap |
| --- | --- | --- | --- | --- | --- | --- |
| roc_auc | -0.0653 | -0.0367 | -0.0631 | 0.0389 | 1.0357 | 0.5812 |
| pr_auc | 0.4067 | 0.0113 | 0.4041 | -0.0140 | 1.0065 | 0.0280 |
| f1 | 0.5049 | 0.0439 | 0.5157 | -0.0332 | 0.9792 | 0.0852 |

Of the -0.0631-point typical-paper AUC gap, -0.0653 points come from the easier low-BMD target and -0.0367 points come from unweighted/leaky methodology; the residual interaction is +0.0389.

Of the +0.4041-point typical-paper PR-AUC gap, +0.4067 points come from the easier low-BMD target and +0.0113 points come from unweighted/leaky methodology; the residual interaction is -0.0140.

## Temporal rigorous holdout

| target | methodology | model | evaluation | evaluation_scope | roc_auc | pr_auc | sensitivity | specificity | ppv | f1 | calibration_slope | calibration_intercept | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| osteoporosis | rigorous | xgboost | sensitivity_oriented | held_out_cycle_2017-2018 | 0.8335 | 0.3695 | 0.8315 | 0.6047 | 0.1328 | 0.2290 | 1.2064 | 0.9579 | 0.0543 |
| osteoporosis | rigorous | xgboost | f1_optimized | held_out_cycle_2017-2018 | 0.8335 | 0.3695 | 0.5078 | 0.9369 | 0.3693 | 0.4276 | 1.2064 | 0.9579 | 0.0543 |
| low_bmd | rigorous | xgboost | sensitivity_oriented | held_out_cycle_2017-2018 | 0.7974 | 0.7961 | 0.9118 | 0.3981 | 0.5824 | 0.7108 | 1.1534 | 0.0872 | 0.1832 |
| low_bmd | rigorous | xgboost | f1_optimized | held_out_cycle_2017-2018 | 0.7974 | 0.7961 | 0.8686 | 0.5212 | 0.6255 | 0.7272 | 1.1534 | 0.0872 | 0.1832 |

## Decision

Diagnostic only, not eligible for production promotion.
