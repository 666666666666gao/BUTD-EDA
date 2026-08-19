# BUTD-DETR ScanRefer independent-retraining ablation plan

Date: 2026-08-14 CST

Scope: BUTD-DETR on `scanrefer_spacy` only. The paper-facing additions are SACR, RAPF, and QAHNL; Source-Choice is excluded.

## Reproducibility contract

- Every row, including baseline and full, starts independently from `/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth`.
- No run may pass `--checkpoint_path`; no full-model or other ablation checkpoint is reused.
- Shared seed/schedule: seed 0, batch 24, epochs 1–100 inclusive, validation every 5 epochs, LR decay at epoch 65.
- Shared randomly initialized parameters are also controlled, not merely the seed: an actual-model parity audit over baseline/full/no-quality/no-relation found all 1,005 shared tensors bit-identical (`99a3e905a320edee9f93ee2267b9da7685ddd472dda2fc6ce2bf474b37bff2f5`). Optional modules use isolated deterministic initialization streams so changing an ablation switch cannot shift the base-transformer initialization.
- Selection metric: official `last__bbs_acc0.25_top1`, strict best-only retention.
- Per row retain `ckpt_best_primary.pth`, `best_primary.json`, `config.json`, final re-evaluation of the reloaded best model, launcher log, and SHA256.
- Report overall official BBS Acc@0.25/0.50 and official-BBS Unique/Multiple Acc@0.25/0.50. Legacy `Analysis unique/multi` is BBF-based and must not be used for those columns.

## Reviewed minimal matrix

| ID | Configuration/change | Reviewer question | Expected if useful | Priority |
|---|---|---|---|---:|
| 01 baseline | no SACR/RAPF/QAHNL | Does the stack beat the underlying BUTD protocol? | Full improves overall and/or Multiple | 1 |
| 02 full | SACR+RAPF+quality+QAHNL, fused primary | Main control | strongest or near-strongest | 1 |
| 03 no SACR/RAPF | QAHNL trained on BBS base; no SACR/RAPF/quality | Does QAHNL alone help, and what is the joint value of structure+fusion? | below full if SACR/RAPF matter | 1 |
| 04 no QAHNL | full architecture without QAHNL loss | QAHNL marginal contribution | drop particularly on Multiple | 1 |
| 05 no quality | full minus quality head/path; QAHNL remains on fused | quality-path contribution | drop if quality calibration matters | 1 |
| 06 no gate supervision | full but `rapf_gate_loss_weight=0`; architecture/fused source unchanged | learned reliability supervision vs extra parameters | weaker or less stable than full | 2 |
| 07 no relation | full with only `sacr_disable_relation`; everything else unchanged | contribution of relation-anchor branch | drop on relational/Multiple cases | 2 |

The matrix deliberately avoids the old `03_sacr_only` and `09_sacr_no_relation` scripts because they changed the primary score source and removed several components simultaneously. Old `04_rapf_quality` duplicated no-QAHNL. Quality-primary is a diagnostic score-source comparison, not a component ablation.

## Full-control hyperparameters

The full and matched leave-one-out rows use the prior ScanRefer candidate-A settings: `rapf_quality_weight=0.75`, `rapf_struct_residual_clip=0.25`, `rapf_gate_loss_weight=0.1`, `rapf_initial_gate_bias=-2.5`, and `rapf_generic_gate_cap=0.1`. These settings are held fixed except when the removed component makes a parameter inapplicable.

## Compute and run order

Sequential on one A100 40GB: baseline, full, no-SACR/RAPF, no-QAHNL, no-quality, no-gate-supervision, no-relation. Historical training is about 0.55 GPU-hour/epoch, so seven 100-epoch rows plus validation are approximately 385–400 GPU-hours (about 16 days wall time). No early stopping is used because it would make training budgets unequal.

## Remote artifacts

- Launchers: `scripts/ablations/scanrefer_20260814/`
- Queue: `scripts/run_scanrefer_ablation_retrain_queue_20260814_v2.sh`
- Queue evidence: `logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_queue/`
- Training root: `logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_from_official_init/`

## Formal-run preflight and start

- Final formal queue start: `2026-08-14T05:39:09+08:00`, detached screen `scanrefer_ablation_retrain_20260814_v2`.
- Official initializer SHA256: `9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2`.
- Focused preflight tests: 8 passed (`preflight_pytest.log`).
- Actual-model initialization parity: PASS over 1,005 shared tensors (`model_init_parity.log`).
- Frozen code/launcher hashes: `code_and_launchers.sha256` in the queue evidence directory.
- The queue reruns both focused tests and the actual-model parity audit before starting any training row.

The first reviewer-generated queue was stopped during baseline epoch 1 after protocol audit found mixed-variable rows. Its directory is historical/invalid and must not be included in result aggregation.
An additional v2 preflight attempt stopped before completing epoch 1 while exact shared-RNG parity was added; its failed marker and partial launcher log are isolated under `scanrefer_ablation_retrain_20260814_v2_queue/invalid_preflight_attempt_20260814_0420/` and are likewise excluded.

## Runtime performance correction (scientific protocol unchanged)

The first v2 formal runtime attempt was stopped at epoch 1 batch 1000/2027, before any validation or checkpoint, after external profiling showed severe CPU thread oversubscription: every one of eight DataLoader workers had 66 threads, creating roughly 528 runnable threads on 80 logical CPUs. Disk I/O was negligible and GPU utilization was bursty/low, while throughput was about 3.52 seconds/step. This attempt is isolated at `scanrefer_ablation_retrain_20260814_v2_queue/invalid_performance_thread_oversubscription_20260814_0538/` and is excluded from all results.

All seven launchers now enforce `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, and `BLIS_NUM_THREADS=1`. These are CPU scheduling controls only: dataset, augmentation, model, loss, optimizer, seed, official initialization, epoch budget, and selection metric are unchanged. Logging frequency was changed uniformly from 1000 to 100 batches for observability; this also has no scientific effect.

The final formal restart verified 3 threads per DataLoader worker instead of 66. Stable batch intervals were 82, 84, and 81 seconds per 100 steps (about 0.82 seconds/step), a roughly 4.3x throughput improvement. The new run remains an independent epoch-1 start from the official initializer, not a resume.
