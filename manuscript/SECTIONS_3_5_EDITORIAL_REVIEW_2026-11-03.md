 > Superseded in part by AUTHOR_VERIFICATION_2026-11-03.md: the 2009 KNHANES joint-analysis weight is supported by guide_107 Table II-1. Historical model hash mismatches are separately documented in the new audit.

# Editorial review: Sections 3–5

## Scope and document versions

The complete seven-page supplied main.pdf and the current Google Doc were read. The PDF supplied Results, Discussion, Conclusions, Tables 2–3, and declarations. The Google Doc contained the abstract, Introduction, Methods, Table 1, and seven methodological references, but no Sections 3–5 or Tables 2–3. Drive search found the main document and earlier backups, not another complete native manuscript.

The revised Sections 3–5 were added immediately before the existing References heading. Tables 2–3 were reconstructed as editable native tables from the PDF; all cell values, units, captions, and confidence intervals were preserved. No external confidence intervals were invented. The abstract, Sections 1–2, Table 1, and existing reference entries were preserved. PDF declarations were not imported or edited because they were outside the requested scope and absent from the Google Doc.

Main document: https://docs.google.com/document/d/1JrKpC3FK83lAd74EaGqjyL0Am0nN_pDL6b2oCO4VHF4

Backup before this edit: https://docs.google.com/document/d/1YLKQwSpNsxTv1xEKYLKg8XDt98haCyCT-A9RtWdR0pc

## Editorial changes

- Simplified sentence structure while retaining all requested numeric findings and methodological limitations.
- Distinguished the internal random forest, separately trained temporal XGBoost, frozen external random forest, nested feature-panel comparison, and separate outcome-definition experiment.
- Explained discrimination, calibration, and threshold performance separately, including low PPV under high-sensitivity screening and the prevalence dependence of predictive values.
- Preserved the distinction between the illustrative 5.6% standardized prevalence and the actual 5.97% weighted internal-test prevalence.
- Explained why a confidence interval spanning zero does not establish equivalence, and why higher average precision or PPV for low BMD does not establish superiority over osteoporosis classification.
- Retained limitations concerning selection optimism, chronology, configuration provenance, fixed-prediction bootstrap intervals, survey variance, external harmonization, fairness, and absent prospective assessment.
- Carried the existing Methods caveats about the 2009 KNHANES weight and measurement compatibility into the Discussion, without recalculating or changing results.
- Kept Conclusions to one paragraph without deployment, future-fracture prediction, or DXA-replacement claims.

## Numerical and methodological issues requiring author review

1. **Abstract outcome inequality:** The Google Doc abstract says T-score greater than −2.5; the Methods and source PDF define the research outcome as T ≤ −2.5. The abstract was left unchanged under the scope restriction.
2. **Abstract NPV:** It reports 98.6% alongside the high-sensitivity policy. Table 2 and the supplied instructions report 99.0% for that policy. Resolve the policy mismatch in a separately authorized abstract revision.
3. **Abstract threshold wording:** It says the threshold maximized sensitivity. Methods specify maximizing specificity subject to at least 90% training sensitivity. The abstract was preserved.
4. **External source weight:** Current Methods document wt_itvex for 2009 in the archived analysis versus wt_ex in the consulted KNHANES guide. Reconcile the applicable release documentation and reassess external estimates if necessary. The archived external values, including prevalence and calibration, remain unchanged.
5. **Selection provenance:** Archived metadata identify weighted average precision, but the current configuration specifies ROC-AUC. Recover the generating code and configuration before describing the analysis as fully reproducible.
6. **Evaluation independence:** Main-workflow feature-panel selection preceded the random/temporal splits; repeated review of temporal/external analyses also limits stronger independence claims. Recover the development chronology. Independent confirmation requires appropriately isolated development decisions.
7. **Uncertainty:** Bootstrap intervals condition on fixed predictions and thresholds. Review survey-design coverage and whether the evaluation subset is handled appropriately. External intervals are absent from the source table, not estimated in this editing task.
8. **Outcome comparability:** Verify scanner/measurement-scale compatibility across included releases. Preserve the distinction between the research label for all women aged at least 50 and menopause-specific clinical interpretation.
9. **Nested difference rounding is consistent:** The displayed 0.3745 − 0.3656 equals 0.0089 after rounding, but full-precision saved means are 0.3744564567021309 and 0.3656411674032063. Their difference is 0.0088152892989246, correctly reported as +0.0088. The paired table gives CI −0.029610538368551424 to +0.04724111696640073. No correction is needed.
10. **Threshold rounding is consistent:** Table 3's 0.0271 is the rounded presentation of the narrative threshold 0.027135. No correction is needed.
11. **Different prevalence denominators are intentional:** 6.0% is the unweighted full NHANES cohort proportion; 5.97% is the weighted internal-test prevalence; 5.6% is illustrative standardization; 16.81% is the archived weighted external prevalence. These should not be interchanged.

## Claims and references requiring verification

- The existing Introduction names Tang, Suh, and Xu studies and gives performance estimates, but these are not reconciled with the current seven-entry numbered bibliography. This task did not independently verify those studies, alter the Introduction, or repeat their findings in Discussion. Verify titles, original endpoints, populations, evaluation designs, numerical claims, and numbered citations before submission.
- The supplied references.docx lists three background reviews on osteoporosis risk factors, smoking, and alcohol, followed by a blank [4]. It does not supply a verified comparison of predictive models. Its numbers are a separate list, not the Google Doc's numbering. No entries were inserted or renumbered.
- The PDF's five-entry bibliography and the Google Doc's seven-entry bibliography differ. Sections 3–5 of the PDF contain no numbered literature citations to transfer. The existing Google Doc references were preserved; a full-manuscript bibliography reconciliation remains necessary.
- A verified comparison with validated clinical screening tools and contemporary prediction studies remains an author task. No unsupported claim of superiority or novelty was added.
- Clinical utility, cost-effectiveness, prospective referral benefit, future-fracture prediction, and demographic fairness have not been established. Subgroup claims require sample-size and uncertainty review.
- The PDF declarations contain unresolved author, funding, ethics, consent, data-access, conflict-of-interest, and AI-disclosure items. They were outside this edit and are not present in the Google Doc; they still require author completion before submission.

## Validation

Connector readback confirmed exact agreement of all 66 Table 2–3 cells with the transcribed PDF source. The entire pre-existing document prefix through Methods was structurally unchanged, including Table 1. The existing seven reference entries were textually unchanged. Revised prose was checked against the requested values and source tables. The paired average-precision rounding check used feature_combination_panel_summary.csv and feature_combination_paired_deltas.csv. No model was refitted and no research estimate was changed.

The final Google Doc export was rendered and pages 8–12 were visually inspected. Both results tables fit intact on their respective pages; headings, intervals, symbols, and the conclusion were legible without clipping or overlap.

