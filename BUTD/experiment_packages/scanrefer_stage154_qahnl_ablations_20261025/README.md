# Stage154 QAHNL internal ablations (ScanRefer, BUTD)

Single-switch rows relative to Full42 (Overall Acc@0.5 = 42.3538%), run through
the same Stage154 chain as S0-S3 / R0-R3.

| Row | Setting | Config change |
|---|---|---|
| Q1 | w/o Hungarian matching | `qahnl_disable_hungarian_pos_rescue: false -> true` |
| Q2 | IoU-only positives | Q1 + `qahnl_disable_top_iou_pos: false -> true` |
| Q3 | Softplus -> Hinge | `qahnl_loss_form: hinge` (all QAHNL losses, incl. Stage150 tiered loss; new runtime) |

QAHNL as trained in Full42: positives = {IoU >= 0.25} U TopIoU_3 U Hungarian-matched,
negatives = IoU <= 0.10 (16 hardest), adaptive margin
`m = clamp(0.2 + 0.5 (IoU+ - IoU-), 0.05, 0.5)`, T = 1.0, loss weight 0.2 in
base/continuation/microtune/14c/16/18. Stage150 uses the tiered QAHNL variant
(weight 1.0), on which Q1/Q2 have no effect.

## Files

- `compile_stage.py` - Stage154 compiler with rows Q1/Q2/Q3 added to `apply_row`.
- `execute_stage.py`, `smoke_base.py` - unchanged from `scanrefer_stage154_ablations_20260907`.
- `start_q_base.sh` - compile -> preflight -> 32-batch smoke -> formal 65-epoch base.

## Start command (server 35608)

```bash
cd /home/gb/butd_stage154_qahnl_ablations_20261025
screen -dmS q1_base bash -c "./package/start_q_base.sh Q1 > q1_base_launcher.log 2>&1"
```

Progress: `control/Q1_base/status.log`; training logs under `Q1/base/`.
Runs go one at a time (base uses ~29 GB GPU): Q1 -> Q2 -> Q3, ~4.6 days each.
