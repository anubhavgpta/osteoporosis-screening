# Medical/mobility feature integration

Updated: 2026-07-19

## Promotion decision

E12 adds four routinely collectable, non-laboratory predictors to E8: arthritis
history, liver condition, thyroid problem, and need for walking equipment. All are
derived from questionnaire or physical-function components and are kept inside the
same imputation, weighting, calibration, and validation workflow as the original
features.

On the 9,367-person production cohort (562 osteoporosis cases), weighted five-fold CV
PR-AUC increased from 0.3386 for E8 to 0.3509 for E12. The paired improvement was
+0.0123 (95% CI +0.0030 to +0.0215), satisfying the pre-specified rule that the lower
confidence bound must be positive. E12 was therefore promoted.

## Held-out results

The final random-split model selected random forest by CV PR-AUC. On 1,874 held-out
participants it achieved ROC-AUC 0.8533, PR-AUC 0.3721, Brier score 0.0467, and
calibration slope 1.146. On the fully separated 2017–2018 cycle, the independently
selected temporal model achieved ROC-AUC 0.8335 and PR-AUC 0.3695.

This is a modest improvement, not a jump to 86% diagnostic accuracy. ROC-AUC measures
ranking across all thresholds. PR-AUC and the threshold-specific precision and recall
are more informative for this uncommon exact-osteoporosis endpoint.

## Precision–recall choices

Every threshold below was selected only from training out-of-fold predictions and then
applied unchanged to the random held-out test.

| Training objective | Threshold | Held-out recall | Held-out precision | Held-out specificity | Held-out NPV |
|---|---:|---:|---:|---:|---:|
| Approximately 90% recall | 0.0271 | 0.9140 | 0.1166 | 0.5600 | 0.9903 |
| Approximately 80% recall | 0.0344 | 0.8466 | 0.1587 | 0.7149 | 0.9866 |
| Approximately 70% recall | 0.0528 | 0.6804 | 0.2136 | 0.8409 | 0.9764 |
| At least 20% precision | 0.0495 | 0.7045 | 0.2085 | 0.8302 | 0.9779 |
| At least 25% precision | 0.0824 | 0.5573 | 0.2773 | 0.9078 | 0.9700 |
| At least 30% precision | 0.1286 | 0.4715 | 0.3396 | 0.9418 | 0.9656 |

The 90%-recall point is appropriate when missing osteoporosis is considered much more
costly than an unnecessary DXA referral. The 20%-precision point is the strongest
balanced alternative observed here: compared with the high-recall setting it roughly
doubles precision, but recall falls to about 70%. The threshold must ultimately be
chosen prospectively from referral capacity and the relative cost of false negatives.

## Dataset decision

NHANES 2005–2006 is available in the harmonized all-cycle cohort, but it is not used by
the production artifact. Adding it to development reduced fixed 2017–2018 PR-AUC from
0.3598 to 0.3516 and recall from 0.8654 to 0.8093. KNHANES is the preferred next
geographic validation dataset; MrOS is useful for older men, and UK Biobank is a
larger but more shifted scanner/population validation.
