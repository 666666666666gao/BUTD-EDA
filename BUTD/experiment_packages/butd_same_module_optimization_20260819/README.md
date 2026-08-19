# BUTD same-module optimization pipeline

This package keeps the paper-facing boundary identical to EDA: SACR + RAPF + QAHNL only.
It explicitly forbids Source-Choice, detector-policy adapters, and SemanticSupport.

Execution order:

1. Wait for the active ScanRefer ablation row03 to save a resumable best checkpoint at epoch 10 or later; poll every 120 seconds.
2. Stop the row03 queue and watchdog only after the checkpoint passes a load audit. The executable resume entry is `resume_ablation_row03.sh`.
3. Continue the formal BUTD ScanRefer Full checkpoint from epoch 65 with the same optimizer, scheduler, module flags, and fused official score. Validate every epoch (`val_freq=1`) and stop only after a durable checkpoint strictly exceeds Acc@0.25 = 0.5391, or after epoch 80 if the target is not reached.
4. Independently reload and verify the accepted ScanRefer checkpoint.
5. Mirror the EDA domain-transfer sequence with one Nr3D epoch followed by one Sr3D epoch, retaining one strict-best checkpoint per dataset.

All logs and machine-readable receipts are written under `state/` and `logs/butd_universal_target/main_results_20260819/`.

If the pipeline has already paused row03, use `resume_from_scanrefer_valfreq1.sh` to restart directly from ScanRefer optimization without replaying the ablation pause stage.
