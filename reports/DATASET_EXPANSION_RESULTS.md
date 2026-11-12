# Dataset expansion and external-validation integration

Updated: 2026-07-19

## NHANES 2005–2006

The official `DXXFEM_D` cycle is now downloaded and harmonized. It adds 1,614 eligible
participants and 100 osteoporosis cases, increasing the all-cycle research cohort to
10,981 participants and 662 cases. E8 clinical-feature availability in this cycle is
94.3%–99.4%.

It was evaluated by keeping 2017–2018 completely fixed as the future-cycle test:

| Training data | Temporal ROC-AUC | Temporal PR-AUC | Sensitivity | Specificity | PPV | Decision |
|---|---:|---:|---:|---:|---:|---|
| Existing earlier cycles | 0.8325 | 0.3598 | 0.8654 | 0.6056 | 0.1378 | Production |
| Existing cycles + 2005–2006 | 0.8217 | 0.3516 | 0.8093 | 0.6605 | 0.1479 | Reject |

Although five-cycle development CV PR-AUC increased, performance on the unchanged
future cohort did not. The cycle is therefore integrated as a sensitivity dataset and
stored in `classification_all_cycles_cohort.csv.gz`, but the saved production artifact
continues to use 2007–2008, 2009–2010, and 2013–2014 for development when 2017–2018 is
the temporal test. This prevents a larger but empirically worse dataset from being
deployed. The source is the official [CDC femur DXA component](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/DXXFEM_D.htm).

## External datasets

| Dataset | Best role | Availability | Integration status |
|---|---|---|---|
| KNHANES 2008–2011 | Geographic external validation in a national Korean survey | Registration/data-use agreement | Adapter ready; data not locally present |
| MrOS | Sex-specific validation in older men | Account and data-use agreement | Adapter ready; data not locally present |
| UK Biobank | Large scanner and geographic domain-shift validation | Approved application and Research Analysis Platform | Adapter ready; data not locally present |

KNHANES is the highest-priority next dataset because it is structurally closest to
NHANES and includes femoral-neck BMD. MrOS is valuable for the model's weakly
represented male osteoporosis cases. UK Biobank offers much greater scale but has a
different scanner, a selected imaging population, and controlled access.

Sources: [KNHANES BMD population study](https://pubmed.ncbi.nlm.nih.gov/24954336/),
[MrOS Online](https://mrosonline.ucsf.edu/), and
[UK Biobank imaging data](https://community.ukbiobank.ac.uk/hc/en-gb/articles/24618819821981-Imaging-Data).

## External-audit contract

`osteoporosis-pipeline external-audit` accepts a harmonized CSV or Parquet export,
checks all frozen production inputs, derives the same femoral-neck `T-score <= -2.5`
outcome, loads the
frozen production artifact, and writes external metrics and predictions. It does not
fit, tune, choose a threshold, or recalibrate on the external cohort. Dataset-specific
access and harmonization requirements are recorded in `configs/external_datasets.yaml`.
