# Optuna ScanRefer Two-Stage Full RAPF Summary

Exploratory short-run tuning report. These trials are not final paper results.

- Objective: `0.5 * Acc@0.25 + 0.5 * Acc@0.5`
- Study name: `scanrefer_two_stage_full_rapf_quick5`
- Storage: `postgresql://optuna@127.0.0.1:5432/optuna_scanrefer`
- Successful trials: 20
- Failed trials: 24
- Recommendation: run top-k 25/30 epoch rerun before any long training.

## Best Trial by Balanced Objective

| rank | trial | worker | objective | Acc@0.25 | Acc@0.5 | clip | quality_weight | gate_loss_weight | gate_bias | generic_gate_cap | quality_anchor_enabled | checkpoint_status | log_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | worker04 | 0.2758 | 0.3638 | 0.1878 | 0.25 | 0.75 | 0.1 | -2.5 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker04/trial_0032 |

## Top-5 by balanced objective

| rank | trial | worker | objective | Acc@0.25 | Acc@0.5 | clip | quality_weight | gate_loss_weight | gate_bias | generic_gate_cap | quality_anchor_enabled | checkpoint_status | log_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | worker04 | 0.2758 | 0.3638 | 0.1878 | 0.25 | 0.75 | 0.1 | -2.5 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker04/trial_0032 |
| 2 | 37 | worker01 | 0.27465 | 0.3588 | 0.1905 | 0.25 | 0.5 | 0.2 | -2 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker01/trial_0037 |
| 3 | 43 | worker03 | 0.27125 | 0.3575 | 0.185 | 0 | 1 | 0.05 | -2.5 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker03/trial_0043 |
| 4 | 11 | worker03 | 0.2655 | 0.3599 | 0.1711 | 0 | 0.5 | 0 | -2.5 | 0.2 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker03/trial_0011 |
| 5 | 45 | worker04 | 0.2617 | 0.3468 | 0.1766 | 0.25 | 1 | 0.2 | -2.5 | 0.2 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker04/trial_0045 |

## Top-5 by Acc@0.25

| rank | trial | worker | objective | Acc@0.25 | Acc@0.5 | clip | quality_weight | gate_loss_weight | gate_bias | generic_gate_cap | quality_anchor_enabled | checkpoint_status | log_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 32 | worker04 | 0.2758 | 0.3638 | 0.1878 | 0.25 | 0.75 | 0.1 | -2.5 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker04/trial_0032 |
| 2 | 11 | worker03 | 0.2655 | 0.3599 | 0.1711 | 0 | 0.5 | 0 | -2.5 | 0.2 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker03/trial_0011 |
| 1 | 37 | worker01 | 0.27465 | 0.3588 | 0.1905 | 0.25 | 0.5 | 0.2 | -2 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker01/trial_0037 |
| 3 | 43 | worker03 | 0.27125 | 0.3575 | 0.185 | 0 | 1 | 0.05 | -2.5 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker03/trial_0043 |
| 5 | 44 | worker02 | 0.25445 | 0.3469 | 0.162 | 0.5 | 1 | 0.1 | -2 | 0.35 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker02/trial_0044 |

## Top-5 by Acc@0.5

| rank | trial | worker | objective | Acc@0.25 | Acc@0.5 | clip | quality_weight | gate_loss_weight | gate_bias | generic_gate_cap | quality_anchor_enabled | checkpoint_status | log_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 37 | worker01 | 0.27465 | 0.3588 | 0.1905 | 0.25 | 0.5 | 0.2 | -2 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker01/trial_0037 |
| 2 | 32 | worker04 | 0.2758 | 0.3638 | 0.1878 | 0.25 | 0.75 | 0.1 | -2.5 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker04/trial_0032 |
| 3 | 43 | worker03 | 0.27125 | 0.3575 | 0.185 | 0 | 1 | 0.05 | -2.5 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker03/trial_0043 |
| 4 | 45 | worker04 | 0.2617 | 0.3468 | 0.1766 | 0.25 | 1 | 0.2 | -2.5 | 0.2 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker04/trial_0045 |
| 5 | 35 | worker03 | 0.25975 | 0.3458 | 0.1737 | 0.5 | 0.5 | 0.2 | -2 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker03/trial_0035 |

## Top-5 Optuna Parameter Sets

| rank | trial | worker | objective | Acc@0.25 | Acc@0.5 | clip | quality_weight | gate_loss_weight | gate_bias | generic_gate_cap | quality_anchor_enabled | checkpoint_status | log_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | worker04 | 0.2758 | 0.3638 | 0.1878 | 0.25 | 0.75 | 0.1 | -2.5 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker04/trial_0032 |
| 2 | 37 | worker01 | 0.27465 | 0.3588 | 0.1905 | 0.25 | 0.5 | 0.2 | -2 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker01/trial_0037 |
| 3 | 43 | worker03 | 0.27125 | 0.3575 | 0.185 | 0 | 1 | 0.05 | -2.5 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker03/trial_0043 |
| 4 | 11 | worker03 | 0.2655 | 0.3599 | 0.1711 | 0 | 0.5 | 0 | -2.5 | 0.2 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker03/trial_0011 |
| 5 | 45 | worker04 | 0.2617 | 0.3468 | 0.1766 | 0.25 | 1 | 0.2 | -2.5 | 0.2 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker04/trial_0045 |

## Recommended Top-K Rerun Candidates

| rank | trial | worker | objective | Acc@0.25 | Acc@0.5 | clip | quality_weight | gate_loss_weight | gate_bias | generic_gate_cap | quality_anchor_enabled | checkpoint_status | log_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | worker04 | 0.2758 | 0.3638 | 0.1878 | 0.25 | 0.75 | 0.1 | -2.5 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker04/trial_0032 |
| 2 | 37 | worker01 | 0.27465 | 0.3588 | 0.1905 | 0.25 | 0.5 | 0.2 | -2 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker01/trial_0037 |
| 3 | 43 | worker03 | 0.27125 | 0.3575 | 0.185 | 0 | 1 | 0.05 | -2.5 | 0.1 | 0 | exists | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker03/trial_0043 |

Default rerun uses top 3 by balanced objective; top 5 are retained for manual review.

## Acc@0.25 / Acc@0.5 Trade-Off Notes

No threshold-based trade-off labels are computed. Review the metric ranks below.

| rank_balanced | trial | worker | objective | Acc@0.25 | Acc@0.25 rank | Acc@0.5 | Acc@0.5 rank | log_dir |
|---|---|---|---|---|---|---|---|---|
| 1 | 32 | worker04 | 0.2758 | 0.3638 | 1 | 0.1878 | 2 | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker04/trial_0032 |
| 2 | 37 | worker01 | 0.27465 | 0.3588 | 3 | 0.1905 | 1 | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker01/trial_0037 |
| 3 | 43 | worker03 | 0.27125 | 0.3575 | 4 | 0.185 | 3 | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker03/trial_0043 |
| 4 | 11 | worker03 | 0.2655 | 0.3599 | 2 | 0.1711 | 6 | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker03/trial_0011 |
| 5 | 45 | worker04 | 0.2617 | 0.3468 | 6 | 0.1766 | 4 | logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/worker04/trial_0045 |

## Parameter Distribution in Top 5

- `rapf_struct_residual_clip`: 0: 2, 0.25: 3
- `rapf_quality_weight`: 0.5: 2, 0.75: 1, 1: 2
- `rapf_gate_loss_weight`: 0: 1, 0.05: 1, 0.1: 1, 0.2: 2
- `rapf_initial_gate_bias`: -2: 1, -2.5: 4
- `rapf_generic_gate_cap`: 0.1: 3, 0.2: 2

## Failed Trials

Failed trial count: 24