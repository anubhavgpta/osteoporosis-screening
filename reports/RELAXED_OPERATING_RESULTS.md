# Relaxed operating-point results

Run date: 2026-07-20

This analysis leaves all existing results unchanged. It consolidates thresholds that
were already selected from calibrated training out-of-fold predictions, then compares
their performance on the random held-out test set and the untouched 2017-2018 temporal
holdout. No evaluation outcome was used to choose or adjust a threshold.

## Comparison across both held-out evaluations

| Operating point | Mean recall | Minimum recall | Mean specificity | Mean PPV | Balanced accuracy | MCC | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 80% training-recall target | 79.57% | 74.47% | 73.49% | 16.99% | **0.7653** | 0.2829 | 0.2793 |
| 20% training-precision target | 67.91% | 65.37% | 83.84% | 22.27% | 0.7588 | 0.3192 | 0.3348 |
| 70% training-recall target | 66.51% | 64.98% | 84.75% | 22.93% | 0.7563 | 0.3223 | 0.3405 |
| 25% training-precision target | 57.83% | 55.73% | 89.76% | 27.83% | 0.7379 | 0.3431 | 0.3757 |
| Original sensitivity-oriented point | 87.28% | 83.15% | 58.23% | 12.47% | 0.7276 | 0.2235 | 0.2179 |
| 30% training-precision target | 49.10% | 47.15% | **93.61%** | **34.39%** | 0.7136 | **0.3628** | **0.4044** |

## Interpretation

The 80%-recall policy is the best balanced default. Relative to the original
sensitivity-oriented point, it gains about 15.3 specificity points and 4.5 PPV points
while giving up about 7.7 recall points. It also has the highest mean balanced accuracy
and retains at least 74.5% recall in the temporal evaluation.

The 20%-precision and 70%-recall policies form a second reasonable region. They deliver
approximately 84% specificity and 22-23% PPV while retaining roughly two-thirds of
cases. Their balanced accuracy is only about 0.006-0.009 below the 80%-recall policy.

The 30%-precision policy is best if “overall” means MCC or F1 and false positives are
substantially more costly. It reaches 93.6% specificity and 34.4% PPV, but recall falls
to 49.1%. It should therefore be treated as a confirmatory/high-risk operating point,
not the main screening threshold.

## Recommendation

Use the 80%-recall operating point as the leading general-purpose candidate. Keep the
30%-precision point as an explicitly labeled high-specificity option. Do not describe
either as universally superior: the former is the best balanced screen, while the
latter is the best precision/MCC trade-off.

Detailed results are in `reports/tables/curve_bending_relaxed_*.csv`.
