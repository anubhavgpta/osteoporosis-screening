 > Superseded in part by AUTHOR_VERIFICATION_2026-11-03.md: the 2009 KNHANES joint-analysis weight is supported by guide_107 Table II-1. Historical model hash mismatches are separately documented in the new audit.

# Reference expansion — 3 November 2026

Added nine verified references to the Google Docs manuscript, increasing the bibliography from 7 to 16. References were renumbered in order of first citation, and existing in-text citations were updated consistently. Tables and study estimates were preserved. Backup: https://docs.google.com/document/d/1EXpYQifs6aPtrCbrUlCHHxowdwZ8wPBSYv9kGVoRrEY

## Verification and interpretation

- Tang: verified against the original abstract indexed at https://pubmed.ncbi.nlm.nih.gov/35122626/. The C-statistic 0.849 is explicitly labelled development-cohort performance. Full-text methods were not independently audited.
- Suh: verified publisher Table 2 and citation metadata at https://www.jmir.org/2023/1/e40179/. The quoted 0.851 and 0.827 are multiclass microaverages, not binary osteoporosis-specific AUCs. The Introduction now identifies this distinction and gives the full paper title.
- Xu: verified publisher Table 2 at https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2024.1347219/full. The 0.785 test AUC belongs to logistic regression and a low-bone-density endpoint. The source has a separate RF discrepancy (0.761 in narrative versus 0.768 in Table 2); no RF result from that paper was imported.
- Saito and Rehmsmeier: publisher article supports precision–recall interpretation in imbalanced data.
- Van Calster and colleagues: publisher article supports distinguishing calibration from discrimination.
- Kweon and colleagues: original resource-profile abstract and PubMed metadata support the KNHANES source description; this does not resolve the release-specific 2009 weight discrepancy.
- Varma and Simon: original methodological paper supports separation of selection and performance estimation.
- Vickers and Elkin: original decision-curve paper supports a future net-benefit analysis. No claim was made that this analysis was performed here.
- Collins and colleagues: original TRIPOD+AI guidance supports reporting and fairness review. The manuscript explicitly states that a completed checklist remains an author task.

The prior abstract inconsistencies, KNHANES weighting discrepancy, model-selection provenance, and variance-review issues remain unresolved. The prior editorial-review statement that Tang/Suh/Xu had not been verified is superseded only to the extent specified above. No local LaTeX manuscript was synchronized in this reference-editing task.

## Current bibliography

[1] International Society for Clinical Densitometry. Official Adult Positions. 2023 positions, current web text accessed 3 November 2026. https://iscd.org/text-based-official-adult-positions/
[2] Saito T, Rehmsmeier M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE. 2015;10(3):e0118432. https://doi.org/10.1371/journal.pone.0118432
[3] Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW; STRATOS initiative. Calibration: the Achilles heel of predictive analytics. BMC Medicine. 2019;17:230. https://doi.org/10.1186/s12916-019-1466-7
[4] National Center for Health Statistics. NHANES Tutorials: Weighting. Accessed 3 November 2026. https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx
[5] National Center for Health Statistics. NHANES Tutorials: Variance Estimation. Accessed 3 November 2026. https://wwwn.cdc.gov/nchs/nhanes/tutorials/varianceestimation.aspx
[6] Kweon S, Kim Y, Jang MJ, Kim Y, Kim K, Choi S, Chun C, Khang YH, Oh K. Data Resource Profile: The Korea National Health and Nutrition Examination Survey (KNHANES). International Journal of Epidemiology. 2014;43(1):69–77. https://doi.org/10.1093/ije/dyt228
[7] Tang Y, Liu Z, Wang S, Yi Q, Xia Y, Geng B. Development and validation of a novel screening tool for osteoporosis in older US adults: The NHANES cross-sectional study. Endocrine. 2022;76(2):446–456. https://doi.org/10.1007/s12020-022-03001-2
[8] Suh B, Yu H, Kim H, Lee S, Kong S, Kim JW, Choi J. Interpretable Deep-Learning Approaches for Osteoporosis Risk Screening and Individualized Feature Analysis Using Large Population-Based Data: Model Development and Performance Evaluation. Journal of Medical Internet Research. 2023;25:e40179. https://doi.org/10.2196/40179
[9] Xu R, Chen Y, Yao Z, Wu W, Cui J, Wang R, Diao Y, Jin C, Hong Z, Li X. Application of machine learning algorithms to identify people with low bone density. Frontiers in Public Health. 2024;12:1347219. https://doi.org/10.3389/fpubh.2024.1347219
[10] Fujiwara S, Nakamura T, Orimo H, Hosoi T, Gorai I, Oden A, Johansson H, Kanis JA. Development and application of a Japanese model of the WHO fracture risk assessment tool (FRAX™). Osteoporosis International. 2008;19:429–435. https://doi.org/10.1007/s00198-007-0544-4
[11] Looker AC, Wahner HW, Dunn WL, Calvo MS, Harris TB, Heyse SP, Johnston CC Jr, Lindsay R. Updated Data on Proximal Femur Bone Mineral Levels of US Adults. Osteoporosis International. 1998;8:468–489. Table 3, p. 476. https://doi.org/10.1007/s001980050093
[12] National Center for Health Statistics. NHANES 2017–2018 Data Documentation, Codebook, and Frequencies: Dual-Energy X-ray Absorptiometry—Femur (DXXFEM_J). 2020. https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DXXFEM_J.htm
[13] Korea Disease Control and Prevention Agency. 국민건강영양조사 원시자료 이용지침서 제5기 (2010–2012) [KNHANES Raw Data User Guide, Fifth Cycle (2010–2012)]. First released March 2012; revised February 2026. Table 4-1, printed p. 17; Table 8, printed p. 20. Official portal: https://knhanes.kdca.go.kr/knhanes/main.do. Consulted project copy: guide_103.pdf.
[14] Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics. 2006;7:91. https://doi.org/10.1186/1471-2105-7-91
[15] Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Medical Decision Making. 2006;26(6):565–574. https://doi.org/10.1177/0272989X06295361
[16] Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378


