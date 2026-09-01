# ScanRefer ablation tables — live record

`-` denotes a pending result. All values are percentages. Stage154 is recorded
with its actual fixed calibration provenance.

## Table 3. Main modules

| ID | Setting | SACR | RAPF | QAHNL | Overall@0.25 | Overall@0.50 | Evidence |
|---|---|:---:|:---:|:---:|---:|---:|---|
| M0 | BUTD-DETR paper baseline | | | | 50.42 | 38.60 | external paper value |
| M1 | + SACR | yes | | | 50.8309 | 37.4632 | audited E65 checkpoint |
| M2 | + RAPF | yes | yes | | 53.5654 | 40.1662 | audited strict-best E60 checkpoint |
| M3 | + QAHNL + fixed calibration | yes | yes | yes | 54.9011 | 42.3538 | Stage154, two sources + selector |

## Table 4. SACR internal design

| ID | Variant | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---|---|---:|---:|---:|---:|
| S0 | w/o target-attribute | - | - | - | - |
| S1 | w/o relation-anchor | 47.8057 | 35.2207 | 53.2814 | 39.7665 |
| S2 | w/o pairwise geometry | - | - | - | - |
| S3 | hard top-1 anchor | - | - | - | - |
| S4 | matched-protocol Full | 48.1024 | 35.1959 | 53.7547 | 39.7770 |

## Table 5. RAPF internal design

| ID | Variant | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---|---|---:|---:|---:|---:|
| R0 | fixed fusion, g=0.1 | - | - | - | - |
| R1 | w/o query-quality cue | - | - | - | - |
| R2 | w/o parser/anchor cues | - | - | - | - |
| R3 | w/o gate supervision | - | - | - | - |
| R4 | matched-protocol Full | 48.1024 | 35.1959 | 53.7547 | 39.7770 |

Negative and non-monotonic results remain in this record. A checkpoint or epoch
is not replaced after seeing its metric unless the same preregistered selection
rule requires it.
