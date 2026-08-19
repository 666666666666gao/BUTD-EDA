# MCLN Contrastive Text Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the weak MCLN `mask_text` source with a stronger deployable `contrastive_text` source and launch a continued training run from the preserved epoch70 checkpoint.

**Architecture:** Keep the existing source-choice selector unchanged. Add a new adapter score source computed from normalized `last_proj_queries`, `proj_tokens`, and the first-row language maps, then train/evaluate with `default,contrastive_text`.

**Tech Stack:** Python, PyTorch, pytest, MCLN ScanRefer training scripts.

---

### Task 1: Add Adapter Test

**Files:**
- Modify: `MCLN-main/tests/test_source_choice_adapter.py`

- [ ] **Step 1:** Add a failing test requiring `contrastive_text` to appear in `source_scores`.
- [ ] **Step 2:** Run `pytest MCLN-main/tests/test_source_choice_adapter.py -q` and confirm it fails because the source is missing.

### Task 2: Implement Source

**Files:**
- Modify: `MCLN-main/models/source_choice_adapter.py`

- [ ] **Step 1:** Add `compute_contrastive_text_source_scores`.
- [ ] **Step 2:** Wire it into `build_mcln_source_choice_batch`.
- [ ] **Step 3:** Run the adapter test and full MCLN tests.

### Task 3: Launch Training

**Files:**
- Use: `MCLN-main/scripts/train_scanrefer_mcln_sp.sh`

- [ ] **Step 1:** Start from `/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_source_choice/current_best_rec_acc025_epoch70_0.57920.pth`.
- [ ] **Step 2:** Use `--source_choice_selector_sources default,contrastive_text`.
- [ ] **Step 3:** Start with a large batch size on the 40GB A100; if CUDA OOM occurs, reduce one step and restart.
