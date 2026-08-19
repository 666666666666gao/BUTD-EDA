# ScanRefer baseline-to-paper transition receipt

- Applied: 2026-08-15 07:33--07:41 +08:00.
- User directive: do not retrain the BUTD-DETR baseline; use the original paper result.
- External source: Jain et al., ECCV 2022, arXiv:2112.08879v5, Table 1 and Supplementary Table 8.
- Paper metrics (Unique / Multiple / Overall, Acc@0.25 and Acc@0.50): 84.2/66.3, 46.6/35.1, 52.2/39.8.
- Protocol disclosure: the paper baseline used ground-truth text labels; all independently trained ablation variants use the repository ScanRefer spaCy parsing protocol.
- Paper payload SHA256: `3b4da886e479c649208f9dc25933a8e925e35e6089011299708f8d6ab9a9c622`.
- Aborted partial baseline logs are preserved under `logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_queue/aborted_runs/invalid_aborted_baseline_retrain_user_requested_paper_20260815` and are excluded by the renderer, validation-history collector, and all completion audits.
- Deleted unused partial baseline weight: `ckpt_best_primary.pth`, 756,737,338 bytes, SHA256 `698ad46823894438401318a64ef8c8d50ce401a6932576c681226d2ed93e58e8`. The deletion is irreversible; audit logs and the reason file remain.
- Pre-transition source backup: `reports/tuning/scanrefer_ablation_20260815/transition_backup_external_paper_baseline_20260815_0732`.
- Patched queue / original auditor / master auditor / renderer SHA256: `335f4cf3246d202b55c23d195a791c495be018903623a34d3e96c098c01ae8de`, `829ffcc90905315d37814196401f383ef410806f92555cb90af8da70048c36cb`, `c53c49b9c5837aca8a007651c4c2db81c7e089f25981330e101564df0325a87d`, `7f5b1472317e2ca03582c89eddeb157abb75f859eeedd70e781dfa301125c9da`.
- Validation: Bash syntax PASS, Python compile PASS, 8/8 preflight tests PASS, model-init parity PASS (1,005 tensors), paper-baseline audit unit PASS.
- Paper table transition snapshot SHA256 (Markdown / LaTeX / TSV): `d7b0962d0cdafe8fcab0c76b2f00d9e0656fcc520b14ac5afd8e057dfe6409de`, `ef0702e1ef5f3da94c42fb1dd86f8f5327efe01404eea0adeb4ca1f777f0f01f`, `ce0f62343955fdbf18aacfa72656dd38708586b5dd3f314867f5115567150779`.
- Queue resumed directly with `02_full_sacr_rapf_qahnl` at 2026-08-15T07:33:56+08:00 from the verified official detector initialization, seed 0, with no resume checkpoint.
- Finalizer was restarted at 2026-08-15T07:36:42+08:00 so it loaded the new external-baseline-aware auditor; its initial count is 1/7.
- Follow-up hygiene audit at 2026-08-15T07:41+08:00 moved the aborted logs outside the training root and atomically regenerated validation history to 0 valid rows before the first Full-model validation. The training root then contained only `02_full_sacr_rapf_qahnl`, with 0 receipts and 0 weights until its first validation.
