# Author verification audit — 3 November 2026

This report supersedes earlier statements that the 2009 KNHANES weight was inconsistent with its documentation. It is an evidence audit, not certification of author declarations or submission readiness. The Google Doc was corrected; the older diagnostics_manuscript.tex remains a separate, unsynchronized draft.

## Completed checks and corrections

- Recomputed metrics from saved internal, temporal, and external prediction files. ROC-AUC values were 0.853260, 0.833514, and 0.883885. Counts were 1,874/112, 2,286/155, and 9,387/1,766 participants/cases. These checks reproduce predictions' summary statistics, not the historical model fit.
- Verified full cohorts: NHANES 9,367/562 and KNHANES 9,387/1,766; neither had duplicate participant IDs. Cycle demographics and subgroup denominators are supplied in verification_outputs.
- All eight KNHANES source hashes and the harmonized-input hash match the archived manifests.
- Resolved the KNHANES weight issue: guide_107.pdf, Table II-1, printed p. 12 (PDF p. 22), specifies wt_itvex for joint BMD/health-questionnaire analysis in 2009. The BMD-only row in guide_103.pdf was insufficient for this predictor set. All 3,440 eligible 2009 records match wt_itvex × 200/580. The four pooling factors are 108/580, 200/580, 192/580, and 80/580. No numerical results changed.
- Corrected the abstract outcome inequality to T-score ≤ −2.5, NPV to 99.0%, and threshold-selection description to maximum specificity subject to ≥90% weighted training sensitivity.
- Verified the reference mean/SD 0.858/0.120 g/cm² in Looker Table 3. Both surveys use g/cm². KNHANES native Japanese-reference T-scores are intentionally not the study endpoint. The study recomputes the endpoint from BMD.
- NHANES femur documentation identifies QDR-4500A instruments for the included cycles. Acquisition/analysis software changed: Discovery 12.4 in 2007–2010; Apex 3.2 acquisition and Apex 4.0 analysis in 2013–2018. No study-specific US/Korean cross-calibration data were found, so scanner equivalence is not established.
- Subgroup files use group-specific out-of-fold thresholds. Separate global-threshold checks show internal male/female case counts of 23/89 and sensitivities approximately 64.6%/96.1%. Some race/ethnicity groups have six or seven cases. These descriptive checks are not fairness certification or precise population estimates.

## Findings that cannot be resolved by editorial verification

1. Historical artifact identity: the external manifest records model SHA-256 14bdbb98aaf5c65704adbfbadbe730950514de3da89749d5043e39fd8536da7b; the current file has c824cdfedd6c93be8dbc4c01cd40a7d0f20aeadff2e598a5eaaf9b62afb73693. Metadata also differ (details in verification_checks.json). Different hashes do not prove different predictions, but prevent certification of the original fit. Recover the historical artifacts or conduct and clearly label a new reproducible analysis.
2. Generating configuration: archived selection uses weighted average precision, while the current configuration specifies ROC-AUC. A dated configuration/code record was not located. The saved virtual environment points to a missing Python executable, and dependencies are not fully locked.
3. Selection independence: the main feature-panel selection precedes the final split. Nested routines isolate selection inside outer training sets, but retrospective choice among outer results remains exploratory. A chronology of decisions and untouched confirmatory evaluation are unavailable.
4. Survey uncertainty: the bootstrap resamples observed PSUs within retained evaluation strata with original weights and fixed predictions. It does not establish validated survey replicate-weight/domain variance or account for the full model-selection process. Do not describe the existing intervals as certified design-based 95% coverage. Resolving this requires statistical reanalysis.
5. Clinical evidence: no prospective clinical utility, scanner cross-calibration, robust fairness assessment, or definitive same-population validated clinical-score comparison was established. A custom risk-factor count must not be described as FRAX.

## Access, ethics, and declarations

User statements: no KNHANES access conditions were received; the user thinks the files were downloaded from the website. Record this as recollection, not verified provenance or completed pledge acceptance. The downloader, download date, release, and any accepted terms remain unconfirmed. No claim of a separate approval letter is necessary without evidence.

The KNHANES V guide's introductory instructions (PDF p. 4, item 5) describe data-use/security pledges during download. Public-use access is distinct from restricted-data approval. Do not infer compliance or noncompliance from the absence of a letter. Do not redistribute raw data until the relevant terms are confirmed.

Source-survey ethics can be documented independently: the guide lists 2008-04EXP-01-C, 2009-01CON-03-2C, 2010-02CON-21-C, and 2011-02CON-06-C. NHANES source protocols are listed at https://www.cdc.gov/nchs/nhanes/about/erb.html. These approvals do not establish the authors' institution's determination for this secondary analysis.

Still required from the authors: names, affiliations, corresponding-author details, actual contributions and final approval; funding/support; conflicts; institutional ethics determination or the applicable basis for no additional review; confirmation of data provenance/terms; registration/protocol information if any. Do not substitute 'none', 'exempt', 'not applicable', or 'all authors approved' without confirmation.

AI-assistance draft for author review: 'OpenAI Codex assisted with manuscript editing, literature checks, and code/output auditing.' Add a statement about human review and responsibility only after authors have performed and accepted it. Do not invent a model version.

## TRIPOD+AI reporting audit

The table paraphrases item topics and assesses available evidence; it is not a declaration of compliance. Source: Collins et al., BMJ 2024;385:e078378, https://doi.org/10.1136/bmj-2023-078378.

| Item | Study status / remaining work |
|---|---|
| 1 Title | Outcome, surveys, and modeling identified. |
| 2 Abstract | Corrected outcome, threshold, NPV; limitations disclosed. |
| 3 Background | Rationale and related models included; target users and inequalities need fuller specification. |
| 4 Objectives | Screening evaluation stated. |
| 5 Data sources | Cycles identified; precise access provenance remains unconfirmed. |
| 6 Participants | Age/BMD eligibility described; treatment handling needs explicit discussion. |
| 7 Preparation | Harmonization and quality checks documented; cross-calibration unavailable. |
| 8 Outcome | Formula/reference verified; assessment/blinding details need source-based completion. |
| 9 Predictors | Feature panels documented; measurement timing/assessors incompletely reported. |
| 10 Sample size | Available-survey sample; no formal adequacy justification established. |
| 11 Missing data | Imputation and external missing predictors described; provide complete missingness table. |
| 12 Analysis | Methods described; exact historical code/configuration/artifact unresolved. |
| 13 Imbalance | Weighted metrics/thresholds described; explicit handling rationale needed. |
| 14 Fairness | Sparse subgroup audit completed; fairness not demonstrated. |
| 15 Outputs | Probability and screening thresholds described. |
| 16 Population differences | Harmonization limitations and prevalence differences documented. |
| 17 Ethics | Source approvals available; institutional determination/access confirmation pending. |
| 18 Open science | Funding/conflicts/protocol/registration need author facts; historical executable release absent. |
| 19 Public involvement | No evidence located; authors must confirm actual involvement or absence. |
| 20 Participant reporting | Counts checked; full flow and weighted characteristics should accompany submission. |
| 21 Analysis denominators | Main and subgroup counts exported. |
| 22 Model availability | Current artifact exists; historically evaluated artifact identity unverified. |
| 23 Performance | Point metrics reproduced; interval coverage and subgroup precision unresolved. |
| 24 Updating | Recalibration experiments separated from archived external evaluation. |
| 25 Interpretation | Discrimination, calibration, and threshold behavior distinguished. |
| 26 Limitations | Selection, transportability, provenance, and variance limitations retained. |
| 27 Implementation | Further independent/prospective evaluation required; input-quality/user requirements not validated. |

## Reproduction

Run `python osteoporosis/manuscript/verify_author_items.py` from the workspace root. It reads existing data/artifacts and writes verification_outputs; it does not refit models. Outputs: verification_checks.json, recomputed_metrics.csv, cohort_by_cycle.csv, subgroup_denominators.csv. Run completed successfully on Python 3.13.3, NumPy 2.3.3, pandas 2.2.3; one pandas fragmentation performance warning does not affect computed results.

Primary documentation consulted: CDC femur codebooks DXXFEM_E/F/H/J; official KNHANES guide_107.pdf and guide_103.pdf in data/raw_knhanes/guides; Looker 1998 Table 3; CDC survey variance guidance. Corrections were applied to the native Google Doc with a pre-edit backup and revision control. Historical local drafts should not be treated as the corrected submission manuscript.
