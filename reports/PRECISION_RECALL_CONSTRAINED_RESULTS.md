# Precision-at-recall constrained integration experiment

## Decision

Retain the native-weight pooled random forest with its existing global threshold.
Neither experimental policy improved osteoporosis-detection precision without a loss
of recall, and neither improved weighted accuracy, precision, and recall together.
The experiment did not overwrite the production model artifact.

## Methods

The existing integration experiment was extended without using either final holdout
for model or threshold selection. Candidate logistic-regression, random-forest, and
XGBoost models were trained with the existing native survey weights and
leave-one-cycle/year-out predictions. Two policies were compared with the incumbent:

- `source_specific_recall_preserving`: maximize the worst-source PPV standardized to
  5.6% prevalence while meeting or exceeding each source's incumbent training recall.
- `source_specific_fixed_90`: maximize the same precision criterion subject to at
  least 90% recall in every held-out training cycle/year.

For each source, the deployed experimental threshold was the lowest threshold required
by any training cycle/year. This conservative guard prevents a weak cycle from being
hidden by stronger cycles. Final changes were evaluated on unchanged NHANES 2017-2018
and KNHANES 2011 participants. Paired 95% confidence intervals resampled PSUs within
survey strata.

Weighted accuracy is reported, but balanced accuracy is preferred for interpretation
because osteoporosis is uncommon and ordinary accuracy is dominated by the negative
class. Raw PPV is also prevalence-dependent; the machine-readable output additionally
contains PPV standardized to 5.6% prevalence.

## Native-weight pooled results

| Holdout / policy | Model | Accuracy | Recall | Precision (PPV) | Specificity | Balanced accuracy |
|---|---|---:|---:|---:|---:|---:|
| NHANES incumbent | Random forest | 56.71% | 90.28% | 12.57% | 54.27% | 72.27% |
| NHANES recall-preserving | XGBoost | 38.55% | 95.49% | 9.58% | 34.41% | 64.95% |
| NHANES fixed-90 | Random forest | 49.03% | 91.25% | 10.95% | 45.95% | 68.60% |
| KNHANES incumbent | Random forest | 43.95% | 97.97% | 22.09% | 33.57% | 65.77% |
| KNHANES recall-preserving | XGBoost | 33.04% | 99.08% | 19.30% | 20.35% | 59.71% |
| KNHANES fixed-90 | Random forest | 74.96% | 87.45% | 37.99% | 72.57% | 80.01% |

### Paired changes versus the incumbent

| Holdout / policy | Accuracy delta | Recall delta | Precision delta | Interpretation |
|---|---:|---:|---:|---|
| NHANES recall-preserving | -18.16 pp (-19.74, -16.20) | +5.21 pp (+0.29, +9.90) | -2.98 pp (-3.72, -2.21) | More cases detected, substantially more false positives |
| NHANES fixed-90 | -7.68 pp (-9.07, -5.99) | +0.97 pp (+0.34, +1.60) | -1.62 pp (-2.02, -1.15) | Recall rises slightly, precision and accuracy fall |
| KNHANES recall-preserving | -10.91 pp (-12.35, -9.33) | +1.11 pp (0.00, +2.50) | -2.79 pp (-3.34, -2.21) | Recall rises slightly, precision and accuracy fall |
| KNHANES fixed-90 | +31.01 pp (+27.89, +34.10) | -10.53 pp (-14.54, -6.84) | +15.90 pp (+13.38, +18.73) | Large precision gain purchased by a clear recall loss |

`pp` denotes percentage points. Intervals are paired PSU-within-stratum bootstrap
95% intervals.

## Interpretation

The recall-preserving policy behaved as designed on the unseen holdouts: recall rose in
both surveys. It did not improve the ranking frontier, however; precision, specificity,
balanced accuracy, and weighted accuracy all declined. The fixed-90 policy exposed the
opposite trade-off in KNHANES: it removed many false positives but missed materially
more true osteoporosis cases, and its 87.45% recall did not transport at the 90% target
despite satisfying that target in every 2008-2010 training year.

Therefore thresholding and the current three candidate algorithms cannot deliver the
requested Pareto improvement. A future model should be promoted only if paired
evaluation shows non-decreasing recall and a positive precision change in both surveys;
if weighted accuracy is also a stated endpoint, it must rise as well. The next useful
work is new, consistently available discriminatory information or a confirmatory DXA
triage pathway—not further tuning of the present score threshold.

## Reproducible outputs

- `reports/tables/nhanes_knhanes_integration_constrained_model_selection.csv`
- `reports/tables/nhanes_knhanes_integration_policy_metrics.csv`
- `reports/tables/nhanes_knhanes_integration_policy_comparison.csv`
- `reports/tables/nhanes_knhanes_integration_policy_intervals.csv`
- `reports/tables/nhanes_knhanes_integration_policy_predictions.csv.gz`

