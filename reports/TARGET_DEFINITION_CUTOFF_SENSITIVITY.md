# Target-definition cutoff sensitivity

This ad hoc diagnostic changes only the low-BMD femoral-neck T-score cutoff. It is not a production model-selection or promotion analysis.

## Summary

| t_score_cutoff | methodology | evaluation_design | prevalence | weighted_prevalence | roc_auc | pr_auc | sensitivity | specificity | ppv | calibration_slope | calibration_intercept | brier_score | evaluations | evaluation_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -2.5000 | naive | naive_random_80_20_unweighted_leaky_selection | 0.0600 | 0.0561 | 0.8204 | 0.3661 | 0.8393 | 0.6646 | 0.1372 | 0.7970 | -0.4016 | 0.0467 | 1 | 1874 |
| -2.0000 | naive | naive_random_80_20_unweighted_leaky_selection | 0.1486 | 0.1509 | 0.8422 | 0.4978 | 0.9101 | 0.5708 | 0.2697 | 0.9956 | 0.0290 | 0.0974 | 1 | 1874 |
| -1.5000 | naive | naive_random_80_20_unweighted_leaky_selection | 0.2846 | 0.2878 | 0.8041 | 0.6355 | 0.8987 | 0.4855 | 0.4098 | 0.9460 | -0.0847 | 0.1527 | 1 | 1874 |
| -1.2000 | naive | naive_random_80_20_unweighted_leaky_selection | 0.3904 | 0.3911 | 0.8051 | 0.7454 | 0.8839 | 0.5009 | 0.5316 | 1.0161 | 0.0525 | 0.1719 | 1 | 1874 |
| -1.1000 | naive | naive_random_80_20_unweighted_leaky_selection | 0.4228 | 0.4268 | 0.8082 | 0.7614 | 0.9116 | 0.4603 | 0.5528 | 1.0077 | -0.0442 | 0.1746 | 1 | 1874 |
| -1.0000 | naive | naive_random_80_20_unweighted_leaky_selection | 0.4551 | 0.4592 | 0.7976 | 0.7726 | 0.8910 | 0.4809 | 0.5891 | 0.9838 | 0.0005 | 0.1823 | 1 | 1874 |
| -0.9000 | naive | naive_random_80_20_unweighted_leaky_selection | 0.4868 | 0.4944 | 0.8001 | 0.7944 | 0.9079 | 0.4678 | 0.6179 | 0.9723 | -0.0118 | 0.1825 | 1 | 1874 |
| -0.8000 | naive | naive_random_80_20_unweighted_leaky_selection | 0.5202 | 0.5319 | 0.7937 | 0.8037 | 0.9015 | 0.4572 | 0.6430 | 0.9908 | 0.0122 | 0.1849 | 1 | 1874 |
| -2.5000 | rigorous | leave_one_production_cycle_out | 0.0600 | 0.0561 | 0.8571 | 0.3547 | 0.9141 | 0.5486 | 0.1081 | 1.0971 | 0.1329 | 0.0435 | 4 | 9367 |
| -2.0000 | rigorous | leave_one_production_cycle_out | 0.1486 | 0.1509 | 0.8361 | 0.4853 | 0.9198 | 0.5434 | 0.2603 | 1.0842 | 0.0661 | 0.1001 | 4 | 9367 |
| -1.5000 | rigorous | leave_one_production_cycle_out | 0.2846 | 0.2878 | 0.8118 | 0.6370 | 0.9046 | 0.4848 | 0.4132 | 1.0586 | 0.0139 | 0.1523 | 4 | 9367 |
| -1.2000 | rigorous | leave_one_production_cycle_out | 0.3904 | 0.3911 | 0.7969 | 0.7140 | 0.9060 | 0.4389 | 0.5071 | 1.0296 | -0.0201 | 0.1781 | 4 | 9367 |
| -1.1000 | rigorous | leave_one_production_cycle_out | 0.4228 | 0.4268 | 0.7960 | 0.7456 | 0.9093 | 0.4209 | 0.5367 | 1.0501 | -0.0059 | 0.1818 | 4 | 9367 |
| -1.0000 | rigorous | leave_one_production_cycle_out | 0.4551 | 0.4592 | 0.7901 | 0.7601 | 0.9052 | 0.4164 | 0.5662 | 1.0256 | -0.0163 | 0.1867 | 4 | 9367 |
| -0.9000 | rigorous | leave_one_production_cycle_out | 0.4868 | 0.4944 | 0.7944 | 0.7997 | 0.9049 | 0.4125 | 0.5988 | 1.0471 | -0.0035 | 0.1856 | 4 | 9367 |
| -0.8000 | rigorous | leave_one_production_cycle_out | 0.5202 | 0.5319 | 0.7953 | 0.8182 | 0.9110 | 0.4142 | 0.6359 | 1.0600 | -0.0181 | 0.1858 | 4 | 9367 |

## Highest PR-AUC

The highest PR-AUC was 0.8182 at cutoff -0.8 in the `rigorous` arm.

## Highest ROC-AUC

The highest ROC-AUC was 0.8571 at cutoff -2.5 in the `rigorous` arm.
