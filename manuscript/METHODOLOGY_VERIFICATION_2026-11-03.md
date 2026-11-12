 > Superseded in part by AUTHOR_VERIFICATION_2026-11-03.md: the 2009 KNHANES joint-analysis weight is supported by guide_107 Table II-1. Historical model hash mismatches are separately documented in the new audit.

# Methodology verification and change audit

Reviewed 3 November 2026. This review checked source code, saved model attributes, archived comparison/metric tables, processed cohorts, the 2009 source SAS file, local official KNHANES guide pages, and the external sources listed below. No models were retrained, no study outputs were recalculated, and no numerical findings were replaced.

## Document changes

- The supplied Google Doc contained a title, abstract, introduction, and Study Design text only. There were no Sections 2.2–2.6, equations, tables, or bibliography to overwrite. The requested sections were added after Study Design, with a native 6-row, 2-column Table 1 in Section 2.3 and seven numbered methodological references.
- All original content through Study Design was preserved exactly, including paragraph formatting, as verified by connector readback. Existing headings were unnumbered; new requested subsection numbers are 2.2–2.6.
- The equations T = (BMD − 0.858) / 0.120 and Y = I(T ≤ −2.5) were retained from the task specification. The low-BMD sensitivity endpoint remains T < −1.0.
- Original Google Doc: https://docs.google.com/document/d/1JrKpC3FK83lAd74EaGqjyL0Am0nN_pDL6b2oCO4VHF4
- Backup created before editing: https://docs.google.com/document/d/1d7ps-UO0kMBj0v8254P8Dfn7svqTROOROtr3Dc-ycPs/edit
- Backup location: ChatGPT folder in My Drive. No sharing settings were intentionally changed.

## Verification findings

| Claim | Evidence checked | Finding |
|---|---|---|
| Exact T-score constants | User-supplied original Looker PDF, Table 3, printed p.476 / PDF p.9, visually inspected | RESOLVED: female, non-Hispanic White, age 20–29, n=409; femoral-neck mean 0.858 g/cm², SD 0.120 g/cm². Table footnote confirms population-weighted estimates. The numerical-source action is resolved; across-survey instrument compatibility is a separate open question. |
| Both-sex reference | ISCD Reference Database and Central DXA sections | Uniform White female reference supported for men and women. ISCD does not independently establish the exact constants. Menopause qualifications retained. |
| Measurement scale | Official DXXFEM_J documentation | 2017–2018 Hologic QDR-4500A, BMD in g/cm² verified. Complete across-release/scanner compatibility is not established. |
| Predictor count | Saved artifact feature_names_in_; JSON metadata; knhanes.py MODEL_FEATURES | 5 demographic + 3 anthropometric + 5 history + 4 medical/mobility = 17 supplied; age–sex category gives 18 total. |
| Saved preprocessing | Direct attribute inspection of classification_model.joblib | Numeric IterativeImputer: median initialization, max_iter=15, indicators enabled. Categorical most-frequent imputation with indicators and one-hot handle_unknown='ignore'. StandardScaler present. |
| Saved classifier | Direct stored attributes | CalibratedClassifierCV, sigmoid, cv=3; random forest 350 trees, minimum leaf size 5. |
| Artifact inspection limit | Installed scikit-learn 1.7.2; serialized estimator warning reports 1.9.0 | Attributes were inspected; predictions were not recomputed. Generic estimator repr failed due to a cross-version attribute difference; subsequent direct attribute inspection succeeded. Do not infer historical runtime reproducibility. |
| NHANES pooled weights | ingest.py and workflow.py | Initial division by all harmonized cycles followed by production-cohort rescaling yields WTMEC2YR/4 and WTDRD1/4 for four production cycles. Cycle-specific stratum offsets and retained PSU IDs are separate from scaling. |
| KNHANES factors | Guide 103, Table 8, printed p.20 / PDF p.28, rendered and inspected; knhanes.py DXA_WEIGHT_SPECS | 108/580, 200/580, 192/580, 80/580 agree. Table distinguishes BMD denominator 580 from body-fat denominator 579. |
| KNHANES source weights | Guide 103, Table 4-1, printed p.17 / PDF p.25, rendered and inspected | 2008 wt_ex1, 2009 wt_ex, 2010 wt_itvex, 2011 wt_ex1. Implementation instead uses wt_itvex in 2009. This is unresolved and material. |
| 2009 source-weight discrepancy | hn09_all.sas7bdat joined by participant ID to eligible harmonized cohort | All 3,440 eligible 2009 participants have differing wt_ex and wt_itvex values. wt_itvex/wt_ex ranges approximately 0.2555–2.3958. Archived pooled weights match wt_itvex × 200/580. This is not a simple constant normalization. |
| Weight fallbacks | modeling.py and external.py; processed cohorts | Main code fills missing requested weight from MEC then 1.0; zero/negative/non-finite values are not uniformly rejected in that routine. External code defaults to unit weights if the entire weight column is absent; if present it rejects invalid values. |
| Fallback occurrence in available cohorts | Saved production and external cohorts | All 9,367 MEC weights finite and positive. All 8,930 reliable-diet participants have non-missing positive dietary weights. All 9,387 eligible external weights finite and positive. No missing-weight fallback was needed in these observed cohorts. This does not repair the 2009 source-variable mismatch. |
| Selection metric | classification_model_metadata.json; classification_model_comparison.csv; temporal comparison CSV | Archived evidence consistently says pr_auc, implemented as average precision. RF CV AP 0.348906 exceeds XGBoost 0.345980 and logistic 0.340941; XGBoost has higher ROC-AUC than RF. Thus RF selection aligns with AP, not ROC-AUC. |
| Generating version | Current YAML and available Git history | YAML says roc_auc. Project directory is untracked in the parent repository and no project-specific historical commit was available. Archived selection criterion is supported; exact generating version remains unresolved. |
| Evaluation separation | workflow.py run_all; modeling.py; experiment reports and nested source paths | Main feature selection precedes final split. Final algorithm/calibration/threshold fitting is training-restricted. Nested outer-cycle experiments are separate. Temporal XGBoost is a different fit from frozen external RF. |
| Metrics | evaluation.py | pr_auc uses average_precision_score; calibration uses weighted logistic regression of outcome on clipped probability logits; Brier is weighted squared probability error. |
| Intervals | reporting.py; internal and temporal prediction tables | 500 PSU-within-stratum bootstrap draws, fixed predictions/thresholds, percentile limits. Missing design IDs are dropped by bootstrap code; none are missing in audited prediction files. Neither file has singleton observed PSU strata (61 internal strata, 15 temporal strata). |
| Survey variance validity | Implementation vs CDC variance guidance | No Taylor domain variance or validated replicate-weight procedure demonstrated. Evaluation-subset bootstrap and development-selection uncertainty remain limitations. |

## Reference audit

Numbering was newly established because the supplied document had no numbered bibliography. Existing unnumbered literature mentions in the Introduction were not altered or claimed to have been reconciled into a nonexistent bibliography.

1. **“Development and application of a Japanese model of the WHO fracture risk assessment tool (FRAX™)” by Fujiwara et al.** Authors: S. Fujiwara, T. Nakamura, H. Orimo, T. Hosoi, I. Gorai, A. Oden, H. Johansson, J. A. Kanis. Osteoporosis International, 2008;19:429–435. DOI: https://doi.org/10.1007/s00198-007-0544-4. Full text checked: https://pmc.ncbi.nlm.nih.gov/articles/PMC2267486/. Explicit numerical constants, age, sex, anatomical site, and attribution to Looker verified. This is primary fracture-model research reproducing the reference values, not the original NHANES reference table.
2. **“Updated Data on Proximal Femur Bone Mineral Levels of US Adults” by Looker et al.** A. C. Looker, H. W. Wahner, W. L. Dunn, M. S. Calvo, T. B. Harris, S. P. Heyse, C. C. Johnston Jr, R. Lindsay. Osteoporosis International, 1998;8:468–489. DOI: https://doi.org/10.1007/s001980050093. Original PDF subsequently supplied by the user at `C:/Users/Anubhav Gupta/Downloads/s001980050093.pdf`. Table 3, printed p.476 / PDF p.9, was visually inspected and confirms the exact numerical reference. Methods on printed p.469 identify Hologic QDR-1000 instruments. The article's printed first-page pagination is 468–489; the manuscript citation now follows the supplied article rather than the publisher landing page's inconsistent 468–490 metadata. Text extraction misaligned some age labels, so the rendered table, not extracted row labels, was used to verify the population.
3. **Official Adult Positions.** International Society for Clinical Densitometry. https://iscd.org/text-based-official-adult-positions/. Current web text accessed 3 November 2026. Supports reference population, both-sex use, skeletal-site distinctions, and menopause-specific interpretation. Does not supply the exact mean and SD.
4. **NHANES 2017–2018 Data Documentation, Codebook, and Frequencies: Dual-Energy X-ray Absorptiometry—Femur (DXXFEM_J).** National Center for Health Statistics, first published May 2020. https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DXXFEM_J.htm. Supports site, units, Hologic instrument, and examination-weight use.
5. **NHANES Tutorials: Weighting.** National Center for Health Statistics. https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx. Supports appropriate component weights and pooling construction; does not validate every study-specific decision or establish that nonconsecutive cycles describe a continuous period.
6. **국민건강영양조사 원시자료 이용지침서 제5기 (2010–2012)** [KNHANES Raw Data User Guide, Fifth Cycle (2010–2012)]. Korea Disease Control and Prevention Agency. Cover states first release March 2012; revision February 2026. Local official guide inspected: `data/raw_knhanes/guides/guide_103.pdf`, Table 4-1 p.17 and Table 8 p.20. Official portal verified: https://knhanes.kdca.go.kr/knhanes/main.do. Direct legacy PDF URL could not be fetched; do not treat the portal as a direct PDF download link. Local SHA-256: `28bdeba4014a12229821aa5e9d9970d18fdd1afb3a5d4139fba22c457ef50617`. Retain a citable archived copy consistent with data permissions.
7. **NHANES Tutorials: Variance Estimation.** National Center for Health Statistics. https://wwwn.cdc.gov/nchs/nhanes/tutorials/varianceestimation.aspx. Supports distinction between sampling weights and variance design, masked PSU/strata handling, and domain-analysis concerns. It does not validate the custom bootstrap used in this project.

## Unresolved author actions

1. **Highest priority: KNHANES 2009 weight.** Establish applicable release guidance and reconcile wt_ex versus wt_itvex. Review/recalculate external weighted results if required. No existing numerical findings were changed in this editorial task.
2. **Cross-survey measurement compatibility only.** The exact numerical reference and original-table verification are now resolved from the supplied Looker PDF. Confirm measurement-scale compatibility across the study's NHANES releases and Korean data; confirmation of the original Hologic QDR-1000 reference does not establish equivalence of every later instrument or release.
3. **Run provenance.** Recover the AP-generating configuration/code snapshot and compatible environment. Current ROC-AUC configuration cannot be presented as the historical configuration.
4. **Selection independence and chronology.** Retain main-pipeline optimism limitation; establish a dated account of feature-panel, expansion, temporal, and external-result review. Audit nested claims before using them as confirmatory evidence.
5. **Survey variance.** Review domain handling and bootstrap coverage. No claim of full survey-design validity is warranted solely from positive weights and nonmissing design IDs.
6. **Outside requested scope: abstract errors.** It says osteoporosis is a T-score “greater than −2.5”; the study endpoint is ≤ −2.5. It gives NPV 98.6% beside the 90%-recall operating point, whereas saved NPV at that threshold is approximately 99.0%; 98.6% is associated with another operating point. It also describes the threshold as maximizing sensitivity, whereas the code maximizes specificity subject to the sensitivity constraint. These sentences were preserved per the instruction not to rewrite unrelated sections and need correction before submission.
7. **Incomplete full manuscript.** The supplied Google Doc has no Results, Discussion, or full reference list for its Introduction studies. The local older manuscript was evidence/context, not substituted for the supplied document. Reference numbering will need global reconciliation when remaining sections and citations are added.

## Corrected or clarified methodological descriptions

- Replaced a generic missing-reference action with direct original-table support after the user supplied Looker et al.; only the separate cross-survey measurement-scale action remains.
- Replaced generic KNHANES weighting uncertainty with verified pooling factors and a specific base-weight discrepancy.
- Distinguished evidence for AP-based archived selection from unresolved version provenance.
- Stated that preprocessing is within model pipelines while feature-panel selection remains upstream of splitting.
- Defined AP explicitly and retained fixed-prediction/bootstrap limitations.
- Kept the Korean frozen-model audit separate from later recalibration/integration experiments.

## Quality checks

### Follow-up: original reference supplied

The user supplied the original Looker PDF and https://wwwn.cdc.gov/nchs/nhanes/nhanes3/datafiles.aspx. The CDC page was opened and its Core Data examination-documentation link was followed to https://wwwn.cdc.gov/nchs/data/nhanes3/1a/exam-acc.pdf (NHANES III Examination Data File Documentation, Catalog 76200, December 1996). The portal provides survey provenance and data/documentation access; it is not substituted for the exact numerical table citation. No NHANES III data were newly fitted or used to recompute the reference values. Table 3 supplies the published reference directly. Section 2.2 and reference [2] were updated in place and read back. Equations and numerical study findings were unchanged. Backup before this follow-up: https://docs.google.com/document/d/14CXzaSWhcBIbVGDTbxumA6dN4r-Aa_Uc1PAYsyiLN9w/edit.

Connector readback confirmed five numbered method headings, both equations, one native Table 1 with all 18 predictors, and seven numbered references. Original document content before the insertion was byte-equivalent at the structured paragraph level at that verification point. A subsequent concurrent revision changed “Study Design” to “2.1 Study Design”; that change was preserved, and the table-formatting update was rebased onto the current revision. The PDF export was inspected and Table 1 was moved with its caption to start a page so all six rows remain together. The manuscript is improved but is not scientifically submission-ready while the unresolved items above remain.

