# SUPERSEDED

This protocol was superseded on 2026-08-17 by FIXED_LR55_60_E65_PROTOCOL.md.

# ScanRefer ablation early-stopping protocol

Status: deployed and monitoring.

All nine trained ablation rows start independently from the verified official initialization. Epoch 100 is a ceiling, not a required endpoint.

- Monitor: official Overall Acc@0.25 (last__bbs_acc0.25_top1), maximized.
- Validation cadence: every 5 epochs.
- Warm-up: no stale count before epoch 35.
- Saturation: stop after 4 consecutive validation events (20 epochs) without an improvement greater than 0.001, i.e. 0.10 percentage point.
- Best checkpoint: independent strict-maximum rule with zero minimum delta. Any positive gain replaces ckpt_best_primary.pth.
- Retention: exactly one best model-weight file per trained row; final evaluation reloads that file.
- Fairness: the same early-stop rule is locked for every trained ablation.
- Current live full row: an external bridge observes completed validation logs because the process predates the native trainer patch. It does not resume or restart training. When saturation triggers, it stops the exact process tree, independently reloads and evaluates the strict-best checkpoint, records SHA256 and the terminal receipt, then resumes the queue.
- Future rows: native distributed-safe early stopping in main_utils.py.

At deployment the full row improved at epoch 35 to 0.5071518721, so it was not saturated. With patience reset at epoch 35, the earliest possible stop is epoch 55.
