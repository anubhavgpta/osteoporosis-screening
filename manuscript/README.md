# Diagnostics manuscript working files

The standalone `diagnostics_manuscript.tex` is the editable manuscript and compiles without external figures or bibliography files. It follows Diagnostics' structured-abstract and research-article organization, but uses the standard article class for the built-in editor. It is not an official MDPI submission package or submission-ready paper.

Journal instructions: https://www.mdpi.com/journal/diagnostics/instructions
Official LaTeX package: https://www.mdpi.com/authors/latex

## Evidence and provenance

All paths below are relative to the project root containing `src/`, `configs/`, and `reports/`.

| Manuscript material | Repository source |
|---|---|
| Cohort counts | reports/tables/cohort_flow.csv; reports/run_summary.json |
| Frozen feature contract and model | artifacts/classification_model_metadata.json |
| Internal metrics and intervals | reports/tables/classification_metric_intervals.csv |
| Temporal metrics and intervals | reports/tables/classification_temporal_validation_metrics.csv; classification_temporal_validation_metric_intervals.csv |
| Operating points | reports/MEDICAL_MOBILITY_RESULTS.md |
| External estimates and mapping limits | reports/tables/external_knhanes_2008_2011_metrics.csv; reports/KNHANES_EXTERNAL_VALIDATION_RESULTS.md |
| Nested panel comparison | reports/FEATURE_COMBINATION_RESULTS.md |
| Target comparison | reports/TARGET_DEFINITION_RESULTS.md |
| Ordering of feature-panel selection and splitting | src/osteoporosis_screening/workflow.py: run_all; src/osteoporosis_screening/modeling.py: run_classification_ablations and run_classification |
| Average precision definition | src/osteoporosis_screening/evaluation.py |

## Required research decisions before submission

1. Recover the exact generating commit/configuration. Current `classification_selection_metric: roc_auc` conflicts with archived metadata specifying PR-AUC. Package version constraints do not establish the historical installed versions.
2. Resolve selection independence. Current main code evaluates feature panels across the full classification cohort before creating final random/temporal splits. Do not call these fully untouched tests. Either rerun the full selection procedure within isolated development data or make audited nested evaluation the primary analysis. No rerun or code change was performed during drafting.
3. Document chronology. The 2017–2018 cycle influenced the expansion decision and was repeatedly examined; do not call all later work prespecified or independently confirmed. Confirm how external results influenced later model decisions.
4. Audit the nested experiments end-to-end and freeze the paper's primary estimand, panel-selection procedure, operating policy, and evaluation design. Do not choose the best outer-evaluated candidate and report its score as an unbiased selected-model estimate.
5. Review survey strata/PSU handling, sparse/single-PSU strata, missing/nonpositive weights, dietary weight fallbacks, and confidence-interval methodology with an appropriate statistical reviewer.
6. Establish the exact source of the T-score constants and discuss the age/menopause eligibility, skeletal-site limitation, and Korean reference/scanner differences.
7. Add a descriptive Table 1 by development/evaluation cohort, cycle-level events, missingness summaries, and cohort flow. Existing overall cohort flow includes the candidate expansion cycle; do not label it four-cycle-only.
8. Add external uncertainty estimates using an appropriate design. Do not invent confidence intervals. Internal threshold-independent intervals use the saved default rows; repeated rows differ slightly because of bootstrap draws.
9. Review existing ROC, calibration, decision-curve, and subgroup outputs before choosing figures. This draft deliberately does not embed unreviewed figures. Add precision–recall and referral-burden displays from the frozen final analysis.
10. Expand related work with verified osteoporosis prediction studies and clinical screening comparators. Five foundational references are included; this is not a completed literature review.
11. Supply authors, affiliations, CRediT roles, funding, ethics/consent documentation, conflicts, data permissions, and code-archive details. Review the AI-use disclosure.
12. Complete an item-level TRIPOD+AI checklist and adapt the final text to the current official MDPI class. The built-in compiler accepts a standalone document, not a multi-file MDPI class project.

## Scope

The paper centers on exact femoral-neck osteoporosis screening and transportability. Broader low-BMD results are a sensitivity analysis. BMD regression, all exploratory tuning variants, and pooled international model experiments are not treated as co-primary claims. Findings describe existing outputs, not newly executed analyses.
