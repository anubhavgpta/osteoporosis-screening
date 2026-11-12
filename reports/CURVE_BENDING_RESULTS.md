# Curve-bending experiment results

Run date: 2026-07-20

This additive experiment did not overwrite any production result or model artifact.
It used four outer NHANES production-cycle holdouts. Within every outer training
partition, the code repeated leave-one-cycle-out model selection, fitted calibration
from grouped out-of-fold predictions, and selected all thresholds without seeing the
outer cycle.

## Result

The conservative pooled E12 policy is the only learned screening strategy that kept
weighted recall above 90% in every outer cycle. It achieved mean recall 94.37%,
minimum-cycle recall 91.26%, mean specificity 45.36%, and mean PPV 9.22%.

| Strategy | Mean recall | Minimum recall | Mean specificity | Mean PPV | Mean referral rate |
|---|---:|---:|---:|---:|---:|
| Pooled E12, point-estimate 90% recall | 91.76% | 86.22% | 53.91% | 10.56% | 48.62% |
| Pooled E12, 95% LCB recall | 94.37% | 91.26% | 45.36% | 9.22% | 56.83% |
| Fixed-utility decision policy + LCB | 94.37% | 91.26% | 45.26% | 9.20% | 56.93% |
| Pooled E12, sex-specific LCB thresholds | 94.67% | 89.73% | 25.62% | 6.95% | 75.48% |
| Separate sex-specific models + LCB thresholds | 91.44% | 88.18% | 30.13% | 7.11% | 71.04% |
| E12 screen + three-biomarker confirmation | 93.30% | 90.65% | 46.96% | 9.37% | 55.26% |

The three-biomarker cascade gained only 1.60 percentage points of mean specificity
over the conservative pooled screen, reduced mean recall by 1.07 points, reduced mean
PR-AUC from 0.3649 to 0.3514, and avoided only about 1.57 referrals per 100 people.
That is too small to justify the added blood-test stage on these data.

The sex-stratified models selected genuinely different feature sets, algorithms, and
hyperparameters by sex and outer fold. Their mean PR-AUC was slightly higher (0.3693
versus 0.3649), but rank discrimination and recall transport were worse. They therefore
do not solve the harder subgroup's operating-point problem.

## Biomarkers without grip

Availability supports three distinct analyses rather than one structurally missing
pooled panel:

| Panel | Cycles | N / events | PR-AUC delta vs E3 | 95% CI |
|---|---|---:|---:|---:|
| Alkaline phosphatase + calcium | All five | 10,981 / 662 | +0.0064 | -0.0367 to +0.0494 |
| Above + vitamin D | Four production cycles | 9,367 / 562 | +0.0118 | -0.0360 to +0.0595 |
| Above + folate | 2013-14 and 2017-18 | 4,516 / 310 | -0.0423 | -0.9964 to +0.9118 |

None has a positive confidence-interval lower bound. The four-biomarker estimate is
especially unstable because only two cycles are compatible and 2017-18 folate is
partially observed. The earlier E10 gain does not reproduce when grip is removed and
the panel is transported across cycles.

## Two-stage interpretation

A screen followed by actual DXA confirmation retains the pooled screen's referral
recall and gives confirmed positives 100% specificity/PPV by reference-standard
assumption. It still sends 56.83% of the screened population to DXA, so it improves the
meaning of a final positive result rather than reducing screening burden.

## Decision

Retain the existing pooled model. If the operational requirement is a transported
90%-recall floor, use the conservative lower-confidence-bound threshold policy as the
leading candidate for a prospective validation study. Do not promote the biomarker
cascade or sex-stratified models from this experiment.

Detailed files are `reports/tables/curve_bending_*.csv`; participant-level predictions
are in `curve_bending_loco_predictions.csv.gz`.
