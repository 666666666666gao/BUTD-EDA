# BUTD ScanRefer ablation live status

Updated: 2026-08-14T09:01+08:00

## Queue

- Formal queue: `scanrefer_ablation_retrain_20260814_v2`
- Watchdog: `scanrefer_ablation_watchdog_20260814`
- Active row: `01_baseline`
- Training state at audit: epoch 7, batch 1400/2027
- Error scan: no Traceback, RuntimeError, CUDA OOM, killed process, or NaN

## First official validation receipt (preliminary)

- Validation epoch: 5
- Selection metric: `last__bbs_acc0.25_top1`
- Selection score: `0.35254522507362224`
- Official BBS Overall Acc@0.25 / Acc@0.50: `35.2545 / 19.0787`
- Official BBS Unique Acc@0.25 / Acc@0.50: `70.8245 / 37.9140` (n=1419)
- Official BBS Multiple Acc@0.25 / Acc@0.50: `29.0147 / 15.7745` (n=8089)
- Best checkpoint: `ckpt_best_primary.pth` (756,736,122 bytes)
- Checkpoint SHA256 at epoch 5: `48e42f5e063ddc12897dbc1eb6fed6b0bf3e23c142bc3e2753af29fab9d8953b`

The official BBS Unique/Multiple numbers above come from `last__bbs_unique_*` and `last__bbs_multiple_*`. The legacy `unique`/`multi` analysis fields are BBF-derived and are not used in the paper-facing table.

This receipt is preliminary. `ckpt_best_primary.pth` is replaced only when a later validation score is strictly greater; the completed-row queue receipt will hash and register the final retained best checkpoint.
