# Target-definition experiment specification

This specification is additive and does not change the deployed target, model, thresholds, or production reports.

| Field | Threshold | Source | Use |
|---|---|---|---|
| `osteoporosis` | femoral-neck T-score <= -2.5 | WHO international reference standard as summarized in ISCD 2023 Official Positions | Existing deployed outcome, reused unchanged |
| `low_bmd` | femoral-neck T-score < -1 | WHO densitometric classification for low bone mass/osteopenia, with ISCD preferring the term low bone mass/low bone density | Diagnostic sensitivity target only |

Cut-point sources:

- ISCD 2023 Official Positions: https://iscd.org/official-positions-2023/
- WHO criteria table via NCBI Bookshelf: https://www.ncbi.nlm.nih.gov/books/NBK543829/table/ch1.Tab1/

## Promotion Rule

Never promotable, diagnostic only.
