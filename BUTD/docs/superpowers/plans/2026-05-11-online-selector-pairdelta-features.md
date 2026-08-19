# Online Selector Pairdelta Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in online selector feature path that exposes the same source-pair delta signals that improved offline selector probes.

**Architecture:** Keep the existing selector and loss contracts intact. Add a default-off CLI/model flag, compute source-pair deltas inside `SourcePoolSelectorHead` from existing `source_scores` and `pred_boxes`, and append those features only when the flag is enabled.

**Tech Stack:** PyTorch, existing BDETR selector modules, `unittest`.

---

### Task 1: Add Pairdelta Head Tests

**Files:**
- Modify: `tests/test_source_pool_selector.py`

- [ ] **Step 1: Write failing tests**

Add tests that instantiate `SourcePoolSelectorHead` with `pairdelta_features=True` and verify:

```python
def test_direct_choice_pairdelta_features_have_stable_shape(self):
    head = SourcePoolSelectorHead(
        d_model=4,
        hidden_dim=8,
        direct_choice=True,
        pairdelta_features=True,
    )
    self.assertEqual(head.direct_choice_pairdelta_dim, 21)
    self.assertEqual(head.choice_mlp[0].in_features, 224)
```

```python
def test_candidate_aware_pairdelta_features_have_stable_shape(self):
    head = SourcePoolSelectorHead(
        d_model=4,
        hidden_dim=8,
        candidate_aware=True,
        pairdelta_features=True,
    )
    self.assertEqual(head.candidate_pairdelta_dim, 21)
    self.assertEqual(head.candidate_mlp[0].in_features, 47)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
OMP_NUM_THREADS=1 conda run --no-capture-output -n bdetr python -m unittest tests.test_source_pool_selector.TestSourcePoolSelector.test_direct_choice_pairdelta_features_have_stable_shape tests.test_source_pool_selector.TestSourcePoolSelector.test_candidate_aware_pairdelta_features_have_stable_shape
```

Expected: fails because `pairdelta_features` is not accepted or pairdelta dims are missing.

### Task 2: Implement Selector Pairdelta Features

**Files:**
- Modify: `models/source_pool_selector.py`

- [ ] **Step 1: Add constructor flag and dimensions**

Add a default-off `pairdelta_features` argument. When enabled, set:

```python
self.direct_choice_pairdelta_dim = len(self.candidate_sources) * 7
self.candidate_pairdelta_dim = len(self.candidate_sources) * 7
```

Append the relevant dim to `choice_feature_dim` and `candidate_pre_context_dim`.

- [ ] **Step 2: Add helper methods**

Add helpers that compute per-source deltas against every candidate source:

```python
_direct_choice_pairdelta_features(
    top_indices, top_scores, top_margins, present, choice_boxes,
    source_score_stack,
)
```

and:

```python
_candidate_pairdelta_features(source_score_stack, present, pred_boxes)
```

Each source row uses fixed per-other-source fields:
`same_query_or_present`, `score_delta`, `rank_delta_or_margin_delta`,
`delta_to_top_or_score_at_top_delta`, `top_score_delta`,
`center_l1_delta`, `size_l1_delta`.

- [ ] **Step 3: Wire helpers into forward**

Append the new feature tensors to `candidate_feature_parts` and `choice_feature_parts` only when `pairdelta_features` is true.

- [ ] **Step 4: Verify GREEN for new tests**

Run the two focused tests from Task 1. Expected: pass.

### Task 3: Add CLI / Model Plumbing

**Files:**
- Modify: `main_utils.py`
- Modify: `models/bdetr.py`
- Modify: `train_dist_mod.py`
- Modify: `tests/test_source_pool_selector.py`

- [ ] **Step 1: Write parser/plumbing tests**

Add a parser test that passes:

```bash
--use_source_pool_selector --source_pool_selector_direct_choice --source_pool_selector_pairdelta_features
```

and asserts `args.source_pool_selector_pairdelta_features` is true.

- [ ] **Step 2: Verify RED**

Run the new parser test. Expected: fails because the flag does not exist.

- [ ] **Step 3: Add the flag and pass it through**

Add:

```python
parser.add_argument(
    '--source_pool_selector_pairdelta_features',
    action='store_true',
    default=False,
    help='Append source-pair delta features to online source-pool selector candidates (default: False)',
)
```

Add a guard requiring either `--source_pool_selector_direct_choice` or `--source_pool_selector_candidate_aware`.

Pass the value into `BDETR(...)` and then into `SourcePoolSelectorHead(...)`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
OMP_NUM_THREADS=1 conda run --no-capture-output -n bdetr python -m unittest tests.test_source_pool_selector
```

Expected: all tests pass.

### Task 4: Record and Prepare Targeted Run

**Files:**
- Modify: `reports/tuning/scanrefer_epoch85_targeted_decision.md`

- [ ] **Step 1: Record the implementation**

Add a short section noting the new `--source_pool_selector_pairdelta_features` flag, the default-off compatibility choice, and the intended next run from the protected epoch85 checkpoint.

- [ ] **Step 2: Syntax verification**

Run:

```bash
python3 -m py_compile models/source_pool_selector.py models/bdetr.py main_utils.py train_dist_mod.py
```

Expected: exit code 0.

### Self-Review

- Scope is limited to online selector feature plumbing; no loss rewrite or offline calibrator rewrite.
- The flag is default-off to avoid breaking old selector checkpoints.
- Existing reports and source-choice dump work remain unchanged.
- This plan does not mark the 0.56 / 0.44 goal complete; it creates the next trainable online path toward that target.
