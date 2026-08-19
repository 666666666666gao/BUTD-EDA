# Experiment Report Template

## Run Metadata

- Dataset:
- Stage / family:
- Script:
- Commit / patch id:
- Log directory:
- Epochs reviewed: early, mid, best, last
- Primary metric: `bbs`
- Diagnostic contrastive metric: `bbf`

## Result Groups

Use separate tables for these result families:

- ScanRefer two-stage ablations: scripts under `scripts/new_method_v2/scanrefer/two_stage/`.
- ScanRefer single-stage main results: baseline / quality-only / full under `scripts/new_method_v2/scanrefer/single_stage/`.
- NR3D/SR3D main results: baseline / quality-only / full under `scripts/new_method_v2/nr3d/` and `scripts/new_method_v2/sr3d/`.

Historical NR3D/SR3D runs that used `--butd_gt --detect_intermediate` are legacy-only. Do not compare them directly against the active `--butd_cls --self_attend` mainline without rerunning.

| result family | baseline | quality-only | full | ablations | notes |
| --- | --- | --- | --- | --- | --- |
| ScanRefer two-stage | | | | | |
| ScanRefer single-stage | | | | n/a | |
| NR3D | | | | n/a | |
| SR3D | | | | n/a | |

## Score Source Contract

- `bbs` is the primary metric family and may use `base`, `structured`, `quality`, `fused`, or `acd` for the target row.
- Anchor and intermediate rows keep baseline span scores.
- `bbf` remains diagnostic contrastive base and is not the primary model-selection metric.
- Record `eval_primary_score_source`, `eval_bbs_score_source`, `eval_bbf_score_source`, `eval_target_row_uses_primary_score`, and `eval_anchor_rows_use_baseline_score`.

## Trend Review

Fill this table for early, mid, best, and last checkpoints.

| area | early | mid | best | last | interpretation |
| --- | --- | --- | --- | --- | --- |
| decomposition stats: ok / repaired / weak / global | | | | | |
| positive map source: explicit / entity / lexical / missing | | | | | |
| metadata conflict / global-only target-empty warnings | | | | | |
| SACR valid / relation / global / weak ratios | | | | | |
| RAPF gate mean / right-to-wrong / wrong-to-right | | | | | |
| Quality IoU corr / @0.50 positives / top1 improvement | | | | | |
| QA-HNL valid / ambiguous-as-negative / margin / violation | | | | | |
| eval score source and primary/diagnostic split | | | | | |

## Decision Notes

- Best checkpoint rationale:
- Regression risks:
- Diagnostics that require follow-up:
- Baseline parity status:
- Short-training status:
