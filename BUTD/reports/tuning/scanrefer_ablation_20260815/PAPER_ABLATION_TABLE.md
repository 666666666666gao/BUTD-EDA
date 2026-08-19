# ScanRefer ablation table

Generated: `2026-08-19T17:28:20+0800`

Results are percentages. `‡` denotes the external BUTD-DETR paper result; `†` denotes an interim strict-best checkpoint; `–` denotes no result yet.

| Group | Setting | SACR | RAPF | QAHNL | Quality | Gate Sup. | Relation | U@0.25 | U@0.50 | M@0.25 | M@0.50 | O@0.25 | O@0.50 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| module | BUTD-DETR (paper)‡ | × | × | × | × | × | × | 84.20 | 66.30 | 46.60 | 35.10 | 52.20 | 39.80 |
| module | SACR only | ✓ | × | × | × | × | ✓ | – | – | – | – | – | – |
| module | QAHNL only (base source)† | × | × | ✓ | × | × | × | 77.31 | 47.43 | 34.52 | 19.16 | 40.90 | 23.38 |
| module | SACR + QAHNL (structured source) | ✓ | × | ✓ | × | × | ✓ | – | – | – | – | – | – |
| module | SACR + RAPF | ✓ | ✓ | × | ✓ | ✓ | ✓ | – | – | – | – | – | – |
| module | Full model | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 85.98 | 65.89 | 48.10 | 35.20 | 53.75 | 39.78 |
| internal | Full w/o Quality | ✓ | ✓ | ✓ | × | ✓ | ✓ | – | – | – | – | – | – |
| internal | Full w/o Gate supervision | ✓ | ✓ | ✓ | ✓ | × | ✓ | – | – | – | – | – | – |
| internal | Full w/o Relation | ✓ | ✓ | ✓ | ✓ | ✓ | × | – | – | – | – | – | – |
| internal | Full with QAHNL base source | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | – | – | – | – |

The BUTD-DETR baseline is taken from Jain et al. (ECCV 2022), Table 1 and Supplementary Table 8: https://arxiv.org/abs/2112.08879. The original paper used ground-truth text labels. All trained ablation variants use ScanRefer only with the repository spaCy parsing protocol, seed 0, exactly 65 epochs, validation every 5 epochs, and the same step learning-rate schedule with 0.1x decays after epochs 55 and 60, from the same verified official detector initialization. Each completed trained row retains only the strict-best official Overall Acc@0.25 checkpoint.

RAPF is structurally dependent on SACR; therefore RAPF-only and RAPF+QAHNL-without-SACR are invalid configurations and are intentionally excluded.
