# Site-inclusive target experiment

This additive experiment does not modify production models or existing pipeline outputs. It tests whether defining outcomes from femoral neck OR total femur improves 2017-2018 temporal validation.

## Best F1 operating points

| target | target_definition | feature_panel | model | roc_auc | pr_auc | accuracy | sensitivity | specificity | ppv | f1 | threshold | evaluation_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low_bmd_any_hip | femoral-neck OR total-femur T-score <= -1.0 | E12_baseline | regularized_logistic | 0.8064 | 0.8101 | 0.7161 | 0.7983 | 0.6381 | 0.6768 | 0.7325 | 0.4074 | 2286 |
| low_bmd_femoral_neck | femoral-neck T-score <= -1.0 | E12_baseline | regularized_logistic | 0.7974 | 0.7936 | 0.7121 | 0.7959 | 0.6347 | 0.6679 | 0.7263 | 0.3971 | 2286 |
| osteoporosis_any_hip | femoral-neck OR total-femur T-score <= -2.5 | E12_baseline | regularized_logistic | 0.8230 | 0.3865 | 0.9270 | 0.2966 | 0.9754 | 0.4816 | 0.3671 | 0.2458 | 2286 |
| osteoporosis_femoral_neck | femoral-neck T-score <= -2.5 | E12_baseline | xgboost | 0.8335 | 0.3695 | 0.9077 | 0.5078 | 0.9369 | 0.3693 | 0.4276 | 0.0732 | 2286 |

## All temporal comparisons

| target | target_definition | feature_panel | model | roc_auc | pr_auc | accuracy | sensitivity | specificity | ppv | f1 | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low_bmd_any_hip | femoral-neck OR total-femur T-score <= -1.0 | E12_baseline | regularized_logistic | 0.8064 | 0.8101 | 0.7161 | 0.7983 | 0.6381 | 0.6768 | 0.7325 | 0.4074 |
| low_bmd_femoral_neck | femoral-neck T-score <= -1.0 | E12_baseline | regularized_logistic | 0.7974 | 0.7936 | 0.7121 | 0.7959 | 0.6347 | 0.6679 | 0.7263 | 0.3971 |
| osteoporosis_any_hip | femoral-neck OR total-femur T-score <= -2.5 | E12_baseline | regularized_logistic | 0.8230 | 0.3865 | 0.9270 | 0.2966 | 0.9754 | 0.4816 | 0.3671 | 0.2458 |
| osteoporosis_femoral_neck | femoral-neck T-score <= -2.5 | E12_baseline | xgboost | 0.8335 | 0.3695 | 0.9077 | 0.5078 | 0.9369 | 0.3693 | 0.4276 | 0.0732 |

## F1 confusion matrices

| target | target_definition | feature_panel | threshold | raw_tn | raw_fp | raw_fn | raw_tp | weighted_tn | weighted_fp | weighted_fn | weighted_tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low_bmd_femoral_neck | femoral-neck T-score <= -1.0 | E12_baseline | 0.3971 | 770 | 429 | 249 | 838 | 7468441.8388 | 4297782.5465 | 2217161.7697 | 8645265.3667 |
| low_bmd_any_hip | femoral-neck OR total-femur T-score <= -1.0 | E12_baseline | 0.4074 | 763 | 419 | 248 | 856 | 7406955.6869 | 4200997.0069 | 2223187.6413 | 8797511.1866 |
| osteoporosis_femoral_neck | femoral-neck T-score <= -2.5 | E12_baseline | 0.0732 | 1962 | 169 | 75 | 80 | 19761058.2167 | 1331819.3122 | 755935.5101 | 779838.4828 |
| osteoporosis_any_hip | femoral-neck OR total-femur T-score <= -2.5 | E12_baseline | 0.2458 | 2056 | 65 | 112 | 53 | 20496637.5241 | 515901.4782 | 1136837.4912 | 479275.0282 |

## Interpretation

Use these rows as a diagnostic comparison only. The total-femur T-score uses a fixed NHANES III young-adult White female reference, so the reference choice should be cited and reviewed before any publication claim.
