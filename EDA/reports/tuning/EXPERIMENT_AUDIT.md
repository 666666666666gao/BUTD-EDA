# EDA SACR/RAPF/QAHNL Experiment Integrity Audit

**Date:** 2026-08-13  
**Auditor:** independent GPT-5.5 xhigh reviewer (same-family Type-A review; not a cross-family Type-B acquittal)  
**Project:** `/home/gb/new butd/butd_detr-main/EDA-master`  
**Overall verdict:** **PASS**, with scope qualifiers  
**Integrity status:** `pass`

## Claim scope audited

The audited claim is that the same paper-facing **SACR + RAPF + QAHNL** module, with a separately identified spatial-backbone compatibility adapter and no Source-Choice runtime branch, reaches the requested official primary thresholds on the complete ScanRefer, Nr3D, and Sr3D test sets. The official primary metric is `last_ position alignment` Top-1. `semantic alignment` is diagnostic only.

## A. Ground-truth provenance — PASS

`src/grounding_evaluator.py:1355-1369` builds ground-truth boxes from batch label fields `center_label` and `size_gts`; predicted boxes are built separately from model outputs at `src/grounding_evaluator.py:1067-1070`. The official position evaluator compares the selected predicted boxes with those batch ground-truth boxes at `src/grounding_evaluator.py:1141-1147`. Ground truth is therefore dataset-provided, not generated from model predictions.

## B. Score normalization — PASS

`src/grounding_evaluator.py:1126-1155` selects/sorts the score source, computes IoU, applies the benchmark thresholds, and accumulates detections divided by the ground-truth count. No primary accuracy normalization by a prediction's own max/min/mean was found. Threshold comparison uses strict IoU greater-than semantics.

## C. Result existence and exact-number traceability — PASS

- ScanRefer: `/root/autodl-tmp/eda_target/eval_mcln_spatial_threeinnov_launcher.log` records 9,508 samples, completed 149/149 batches, and official primary `0.57015 / 0.45236` at IoU 0.25/0.50.
- Nr3D: `/root/autodl-tmp/eda_target/train_nr3d_spacy_threeinnov_e1_b14_launcher.log` records 7,899 samples, completed 565/565 batches, and official primary `0.49804 / 0.35486`.
- Sr3D: `/root/autodl-tmp/eda_target/train_sr3d_spacy_threeinnov_e1_b14_launcher.log` records 17,678 samples, completed 1,263/1,263 batches, and official primary `0.62066 / 0.48060`.
- The three logs contain no traceback, RuntimeError, OOM, killed-process, or exception signal. The semantic-alignment lines are separate diagnostics and are not used to pass the requested thresholds.
- Protocol/config evidence is in the corresponding `config.json` files and summarized by `/root/autodl-tmp/eda_target/final_audit/final_acceptance.json`.

## D. Primary evaluator call path — PASS

`src/grounding_evaluator.py:666-668` calls the position-alignment evaluator for the supplied prediction prefix; `src/grounding_evaluator.py:164-177` prints `<prefix> position alignment Acc...`. For `prefix == "last_"`, the configured fused-score override is selected at `src/grounding_evaluator.py:980-995`. The claimed primary metric is therefore live in the executed evaluation path, not a defined-but-unused metric.

## E. Scope — PASS with qualifiers

Evidence covers three complete official test sets and one visible `rng_seed=0` main run/config per dataset. The current report does not claim multi-seed robustness or an extensive robustness sweep. The supported scope is full-test, single-seed/single-main-run evidence. Nr3D and Sr3D are sequential one-epoch domain adaptations, not training from scratch.

## F. Evaluation type — PASS (`real_gt`)

- ScanRefer: `real_gt`, complete `scanrefer_spacy` test set, established `butd` protocol, fused-score `last_ position alignment`.
- Nr3D: `real_gt`, complete `nr3d_spacy` test set, `joint_det + butd_cls`, fused-score `last_ position alignment`.
- Sr3D: `real_gt`, complete `sr3d_spacy` test set, `joint_det + butd_cls`, fused-score `last_ position alignment`.

## G. Innovation boundary — PASS

- `models/bdetr.py:27-29`, `:289-316`, and `:853-931` import, conditionally instantiate, and execute SACR, reliability fusion (RAPF), and the quality dependency used by QAHNL.
- QAHNL configuration/objective wiring is explicit and default-off at `main_utils.py:229-266`, `:1371-1394`, and `train_dist_mod.py:182-183`.
- The spatial adapter is separately declared and default-off at `main_utils.py:51-54`; its conditional encoder/decoder branches are at `models/encoder_decoder_layers.py:303-342` and `:414-501`. It is a compatibility adapter, not a fourth paper-facing innovation.
- The scoped runtime-code scan found no Source-Choice, source selector, selector head, source mask, or SWA branch. ScanRefer migration audit reports `source_choice_used=false`, `mask_branch_used=false`, and no forbidden output keys. Nr3D/Sr3D audits each report 1,115 model tensors, exact tensor-copy verification, optimizer state removed, SACR 18 tensors, RAPF 4, QAHNL quality dependency 6, spatial adapter 72, and forbidden tensor count 0.
- Nr3D/Sr3D checkpoint hashes match their audit JSON and the final acceptance receipt. The independent reviewer inspected the audit metadata and bounded code paths rather than directly enumerating every checkpoint tensor; the executor-side exact-copy/hash receipts mitigate that stated evidence limit.

## Claim impact

- **ScanRefer thresholds:** supported — official primary 57.015% @0.25 and 45.236% @0.50 exceed 55.68% and 44.03%.
- **Nr3D threshold:** supported — official primary 49.804% @0.25 exceeds 46.49%.
- **Sr3D threshold:** supported — official primary 62.066% @0.25 exceeds 57.95%.
- **Cross-baseline/general-module boundary:** supported at the demonstrated scope: the same paper-facing SACR/RAPF/QAHNL boundary is active across the three EDA dataset protocols, the spatial adapter is compatibility-only, and Source-Choice is absent from promoted artifacts/scoped runtime code.

## Scope qualifiers to preserve

- Do not substitute semantic diagnostics for the official primary position metric.
- Do not call Nr3D/Sr3D training-from-scratch; each is a one-epoch sequential domain adaptation.
- Do not claim per-component causal contribution without ablation experiments.
- Do not claim multi-seed robustness from the present single-seed/single-main-run evidence.

## Authoritative receipts

- Final acceptance JSON: `/root/autodl-tmp/eda_target/final_audit/final_acceptance.json`
- Unit-test receipt: `/root/autodl-tmp/eda_target/final_audit/unittest_126.log`
- Compile receipt: `/root/autodl-tmp/eda_target/final_audit/py_compile.log`
- Canonical tuning/handoff report: `/home/gb/new butd/butd_detr-main/EDA-master/reports/tuning/eda_scanrefer_sacr_rapf_qahnl.md`
- Full independent-review trace: `/home/gb/new butd/butd_detr-main/EDA-master/.aris/traces/experiment-audit/2026-08-13_run01/`