# 3 Results

## 3.1 Internal and Temporal Performance

The unweighted proportion of participants with femoral-neck osteoporosis in the NHANES cohort was 6.0%. In the internal random-split evaluation, the archived random forest achieved a ROC-AUC of 0.853 and an average precision of 0.372. At the training-derived threshold of 0.027135, sensitivity was 91.4%, specificity was 56.0%, positive predictive value (PPV) was 11.7%, and negative predictive value (NPV) was 99.0% (Table 2). The Brier score was 0.0467. The calibration intercept was 0.484, and the calibration slope was 1.146.

The temporal evaluation used a separately developed XGBoost model, rather than the saved random forest used for external evaluation. This model achieved a ROC-AUC of 0.834 and an average precision of 0.370 in the 2017–2018 survey cycle. At its training-derived threshold of 0.029249, sensitivity was 83.2%, specificity was 60.5%, and PPV was 13.3%. The calibration intercept and slope were 0.958 and 1.206, respectively. Thus, meeting the 90% sensitivity target during development did not ensure the same sensitivity in a later survey cycle. Both evaluations remain subject to the feature-selection limitations described in Methods, and the temporal analysis should not be treated as a fully independent confirmation study.

Table 2: Archived evaluation results. Percentages are weighted. Internal and temporal analyses retain the feature-selection caveat described in Methods. Parentheses show 95% bootstrap intervals. Threshold-independent intervals use the saved default-evaluation rows consistently. External intervals were not available in the source table used here.

| Measure | Internal random split | Temporal 2017–2018 | External KNHANES |
| --- | --- | --- | --- |
| Participants | 1874 | 2286 | 9387 |
| Algorithm | Random forest | XGBoost | Frozen random forest |
| ROC-AUC | 0.853 (0.821–0.890) | 0.834 (0.791–0.881) | 0.884 |
| Average precision | 0.372 (0.302–0.448) | 0.370 (0.290–0.477) | 0.647 |
| Sensitivity, % | 91.4 (86.9–96.1) | 83.2 (76.3–91.6) | 98.8 |
| Specificity, % | 56.0 (52.9–59.5) | 60.5 (58.1–63.0) | 27.8 |
| PPV, % | 11.7 (9.4–13.9) | 13.3 (11.7–15.2) | 21.7 |
| NPV, % | 99.0 (98.4–99.6) | 98.0 (97.1–99.0) | 99.1 |

## 3.2 Operating-Point Trade-offs

Table 3 presents the internal random-test results at thresholds selected from training data. Increasing the threshold raised specificity and PPV but lowered sensitivity. A higher threshold therefore reduced false-positive predictions and the number of people who would be referred under a screening policy, while missing more participants with osteoporosis. For example, at a threshold of 0.0824, sensitivity was 55.7%, specificity was 90.8%, and PPV was 27.7%, compared with 91.4%, 56.0%, and 11.7%, respectively, at the high-sensitivity threshold.

These results show the trade-off between detecting more osteoporosis cases and limiting referrals among people without the defined femoral-neck outcome. A positive screen would require further assessment; it would not establish a diagnosis. The thresholds illustrate different screening choices and should not be interpreted as clinically optimal or ready for use. Referral consequences were not evaluated prospectively.

Table 3: Training-derived operating points applied to the archived random test set.

| Training objective | Threshold | Sensitivity (%) | PPV (%) | Specificity (%) |
| --- | --- | --- | --- | --- |
| 90% recall | 0.0271 | 91.4 | 11.7 | 56.0 |
| 80% recall | 0.0344 | 84.7 | 15.9 | 71.5 |
| 70% recall | 0.0528 | 68.0 | 21.4 | 84.1 |
| 25% precision | 0.0824 | 55.7 | 27.7 | 90.8 |
| 30% precision | 0.1286 | 47.2 | 34.0 | 94.2 |

## 3.3 External Performance and Harmonization

The weighted prevalence of femoral-neck osteoporosis in KNHANES was 16.81%. The frozen NHANES random forest achieved a ROC-AUC of 0.884 and an average precision of 0.647. At the unchanged NHANES threshold, sensitivity was 98.8%, specificity was 27.8%, and PPV was 21.7% (Table 2). The model identified a high proportion of participants with osteoporosis, but it also classified many participants without osteoporosis as positive.

Discrimination, calibration, and performance at a chosen threshold describe different aspects of a model. ROC-AUC describes how well the model distinguishes cases from non-cases across thresholds; it does not establish that predicted probabilities match observed outcomes or that a particular threshold gives acceptable screening results. In KNHANES, the mean predicted probability was 11.53%, below the observed weighted prevalence of 16.81%, indicating underprediction on average. The calibration intercept was 0.974, and the calibration slope was 1.177.

The higher external PPV should be interpreted alongside the higher osteoporosis prevalence. In the saved external report, standardization to an illustrative prevalence of 5.6% yielded a PPV of 7.51%. The 5.6% value was a chosen comparison prevalence, not the actual weighted prevalence of 5.97% in the NHANES internal test set. This calculation illustrates the influence of prevalence on PPV and is not a new observed test-set result.

Several predictors were unavailable or differed between surveys. Long-term glucocorticoid use and need for walking equipment were unavailable throughout KNHANES, and prior fracture history was unavailable for 55.66% of participants in the pooled cohort. Parental bone history and liver-condition variables were not directly equivalent to the NHANES variables. Imputing missing inputs does not make them equivalent to measured inputs. These differences limit direct comparisons between populations and prevent attribution of the performance changes solely to geographic or ethnic differences. The external values remain archived estimates pending resolution of the 2009 KNHANES source-weight discrepancy described in Section 2.4.

## 3.4 Nested Experiments and Target Sensitivity

In the nested feature-panel experiment, the leading panel combining chemistry measurements with additional anthropometric variables had a mean outer ROC-AUC of 0.8598 and an average precision of 0.3745. The E12 comparator in the same experiment had an average precision of 0.3656. The reported paired difference in average precision was +0.0088 (95% confidence interval, −0.0296 to +0.0472). Because the interval included zero, the experiment did not establish a clear improvement from the additional predictors. It also did not establish that the two approaches were equivalent. The paired comparison used the E12 results from this experiment, rather than a baseline from a different evaluation.

A separate leave-one-cycle-out experiment compared outcome definitions. For femoral-neck osteoporosis, ROC-AUC was 0.8571, average precision was 0.3547, sensitivity was 91.4%, and PPV was 10.8%. For the broader low-BMD outcome, defined as T < −1.0, ROC-AUC was 0.7901, average precision was 0.7601, sensitivity was 90.5%, and PPV was 56.6%. The low-BMD task included a broader group of participants as cases and had a different outcome prevalence. These differences affect performance measures, particularly average precision and PPV. Their higher values for low BMD therefore do not establish that this model was inherently superior to the osteoporosis model.

# 4 Discussion

Routine demographic, anthropometric, and medical-history information helped distinguish participants with femoral-neck osteoporosis from those without the defined outcome. However, a relatively high ROC-AUC did not ensure reliable screening decisions at a chosen threshold. At the internal high-sensitivity threshold, only 11.7% of positive screens corresponded to femoral-neck osteoporosis. Most positive screens therefore occurred in participants without that outcome. The high NPV must also be interpreted in the context of the relatively low osteoporosis prevalence; it does not, by itself, establish that further assessment can safely be withheld.

The external results show why discrimination, calibration, and screening performance should be evaluated separately. ROC-AUC remained relatively high in KNHANES, but the model underestimated average risk, and specificity fell substantially when the original NHANES threshold was applied. The same threshold therefore did not produce consistent screening performance across the two populations. Differences in prevalence, demographic composition, missing predictors, variable definitions, and measurement procedures could have contributed. The analysis did not determine the individual contribution of any of these factors, and the changes cannot be attributed solely to geography or ethnicity. Local validation, calibration assessment, and threshold evaluation would be necessary before any clinical application, followed by prospective assessment of referral decisions.

Adding laboratory measurements and further body measurements did not clearly improve performance in the nested feature-panel comparison. This is relevant to the practical aim of screening with routinely available information, because additional tests increase data-collection requirements. However, the confidence interval for the paired difference in average precision ranged from −0.0296 to +0.0472. This uncertainty does not establish equivalence or show that laboratory variables are unhelpful. Further independent evaluation would be needed to determine whether particular additional predictors offer a useful improvement.

Identifying low BMD is a different classification task from identifying osteoporosis. The broader definition changes which participants are considered cases and increases outcome prevalence. These changes can raise average precision and PPV even when ROC-AUC is lower. The results should therefore be interpreted within each outcome definition, rather than used to claim superiority of one model over the other. Comparisons with other studies also need to account for the outcome, age range, sampling design, and evaluation procedure.

Several limitations affect interpretation. The cross-sectional design assessed current measured status and cannot establish prediction of future fractures or osteoporosis onset. The outcome was restricted to the femoral neck and did not capture osteoporosis at other skeletal sites or provide a comprehensive clinical diagnosis. As described in Methods, including women aged at least 50 years without requiring documented postmenopausal status also limits direct clinical interpretation of the research classification. Measurement-scale compatibility across the included NHANES and KNHANES releases requires verification.

In the main workflow, feature-panel selection occurred before the final internal or temporal split and could therefore use information from participants later included in evaluation. This creates a risk of optimistic performance estimates despite training-only fitting, calibration, and threshold selection in the final routine. Stronger confirmatory claims would require a rerun with all development decisions isolated from evaluation data or a primary analysis based on audited nested evaluation. The archived model-selection metadata also differ from the current configuration, so the generating code and configuration need to be recovered. Repeated examination of the temporal survey cycle during expansion and threshold experiments further limits its independence. The sequence of development and external analyses also needs to be documented, because repeated review of external results may influence later research decisions even when a particular evaluated model remains frozen.

The bootstrap confidence intervals describe uncertainty for fixed predictions and thresholds; they do not account for all feature-selection and model-development decisions. The survey variance procedure also requires further review before its design-based coverage can be assumed. In addition, the 2009 KNHANES source weight used in the archived analysis differs from the weight specified in the consulted guide, as detailed in Section 2.4. This discrepancy must be reconciled with the applicable release documentation, and the external estimates must be reassessed if necessary. The numerical results reported here have been retained rather than silently changed.

External harmonization was limited by missing predictors and variables that were not directly equivalent across surveys. These limitations reduce confidence that performance differences reflect population differences alone. No prospective study of referral decisions, cost-effectiveness, or patient outcomes was conducted. Consequently, the present analyses cannot establish the clinical benefit or resource implications of using the model to guide screening.

Fairness across demographic groups has not been established. Although subgroup outputs are available, their sample sizes, uncertainty, and handling of small groups require a dedicated assessment before comparable performance can be claimed. The use of race/ethnicity and socioeconomic variables also makes evaluation of portability and potential disparities important. A focused comparison with validated clinical screening tools and verified contemporary prediction studies remains necessary to place these findings in context. The study's contribution should be assessed through its evaluation of thresholds, external calibration, population differences, and practical screening limitations; the use of machine learning or national survey data alone does not establish novelty.

# 5 Conclusions

Routine demographic, anthropometric, and medical-history variables helped identify individuals more likely to have the study's femoral-neck osteoporosis outcome. However, high-sensitivity screening produced many false-positive results, and the same threshold performed differently in an external population. Adding clinical and laboratory measurements did not clearly improve performance in the tested nested comparison. Further independent validation, resolution of the identified methodological issues, and prospective assessment of screening and referral outcomes are needed before clinical implementation. The model should not be considered a replacement for DXA testing or clinical diagnosis.
