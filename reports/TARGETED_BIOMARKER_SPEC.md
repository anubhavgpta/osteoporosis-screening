# NHANES 2013–2014 grip and biomarker specification

The targeted experiment is deliberately restricted to the 2013–2014 classification
cohort. The primary pooled analysis remains unchanged because grip strength was not
measured in every pooled cycle.

| Component | NHANES field | Pipeline field | Unit / coding | Use |
|---|---|---|---|---|
| `MGX_H` | `MGDCGSZ` | `combined_grip_strength_kg` | kg; sum of the largest reading from each hand | E9/E10 predictor |
| `MGX_H` | `MGDEXSTS` | `grip_test_status` | 1 both hands, 2 one hand, 3 not performed | missingness audit |
| `MGX_H` | `MGXH1T1`–`MGXH2T3` | six trial fields | kg | derive `maximum_grip_strength_kg` for sensitivity/audit |
| `BIOPRO_H` | `LBXSAPSI` | `alkaline_phosphatase_iu_l` | IU/L | E10 predictor |
| `BIOPRO_H` | `LBXSCA` | `serum_calcium_mg_dl` | mg/dL | E10 predictor |
| `FOLFMS_H` | `LBDFOTSI` | `serum_total_folate_nmol_l` | nmol/L | E10 predictor |
| `VID_H` | `LBXVIDMS` | `serum_25oh_vitamin_d_nmol_l` | nmol/L; D2 + D3, excluding epi-25OHD3 | E10 predictor |
| `VID_H` | `LBDVIDLC` | `serum_25oh_vitamin_d_lod_code` | 0 at/above LOD, 1 below LOD | audit only |

`FOLATE_H` is not used: it contains RBC folate (`LBDRFO`/`LBDRFOSI`), whereas the
requested serum total folate is `LBDFOTSI` in `FOLFMS_H`. CDC already inserts
LLOD/sqrt(2) values for below-detection-limit vitamin D components and folate forms;
these values must not be recoded as missing. Total vitamin D had no below-LOD records
in the released 2013–2014 file.

E9 and E10 are evaluated with the existing weighted, fold-identical, paired-CI
procedure against E3. The pipeline only fits and writes a targeted calibrated model,
thresholds, bootstrap intervals, subgroup results, and predictions when the paired
AUC-difference 95% CI lower bound is greater than zero.

Official CDC documentation:

- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/MGX_H.htm
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BIOPRO_H.htm
- https://wwwn.cdc.gov/nchs/data/nhanes/public/2013/datafiles/FOLFMS_H.htm
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/VID_H.htm
