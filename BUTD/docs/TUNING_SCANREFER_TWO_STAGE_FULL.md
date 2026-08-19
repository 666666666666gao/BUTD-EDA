# ScanRefer Two-Stage Full RAPF Tuning

This document describes exploratory tuning utilities for the ScanRefer
two-stage full SACR/RAPF/Quality/QA-HNL model. These utilities are not official
experiment entrypoints and do not produce final paper results by themselves.

## Current Recommended Flow

Eval-only sweep is optional. It is only a checkpoint-based diagnostic when the
user can provide an explicit existing checkpoint. If no checkpoint exists, skip
eval-only sweep and start with Optuna short-run tuning.

Recommended flow:

1. If an explicit checkpoint exists, optionally run eval-only fusion sweep.
2. If no checkpoint exists, run Optuna short-run tuning directly.
3. Export and review the top5 balanced-objective parameter sets from the study.
4. Rerun top3 or top5 candidates to 15, then 25/30 epochs from scratch by
   default.
5. Fix parameters and run same-length formal baseline, quality-only, and full
   training before making final claims.

The epoch 50 result remains historical evidence only. Because the epoch 50
checkpoint is unavailable, it cannot be used for reproducible eval-only sweep or
future tuning.

## Optional Eval-Only Diagnostic

The current failure mode points to fused ranking being pulled down by the
structured residual path. Eval-only sweep can test forward-time fusion
parameters on an already trained checkpoint, but it must not infer or search for
a checkpoint.

Eval-only sweep only treats parameters that affect checkpoint inference as
effective:

- `rapf_struct_residual_clip`
- `rapf_quality_weight`
- `rapf_generic_gate_cap`
- `rapf_quality_anchor_structured_residual`

It does not treat `rapf_gate_loss_weight` or `rapf_initial_gate_bias` as
effective eval-only variables for an already trained checkpoint.

Use `--checkpoint` as the canonical argument. `--checkpoint_path` is accepted
only as a backward-compatible alias. If both are provided, they must resolve to
the same path. Real runs fail when the checkpoint is missing:

```text
Eval-only sweep requires an existing checkpoint. No fallback checkpoint is used.
```

Dry-run may use a nonexistent path and prints:

```text
dry_run = true
checkpoint existence not required
```

## Why Only RAPF/Fusion Parameters

The latest diagnostics show that base and quality diagnostic ranking are
stronger than fused ranking on the same final boxes. This points to RAPF fusion
composition rather than base detector capacity. The first tuning pass therefore
keeps the training structure fixed and searches only RAPF/fusion parameters.

The tuning scripts do not search:

- learning rate or backbone learning rate
- batch size
- encoder or decoder layer count
- number of target queries
- QA-HNL IoU thresholds
- dataset or stage flags

## Why Not Tune Decoder Layers

The two-stage full script keeps `--num_decoder_layers 6`. The issue is score
source ranking on the same last-layer boxes, not evidence that fewer decoder
layers are needed. Changing decoder depth would create a different model
capacity comparison and is outside this tuning pass.

## Primary Metrics

Acc@0.25 and Acc@0.5 are both primary metrics. The default tuning objective is:

```text
0.5 * last__bbs_acc0.25_top1 + 0.5 * last__bbs_acc0.50_top1
```

Reports also show top rankings by Acc@0.25 and by Acc@0.5 separately. Do not
select parameters using only Acc@0.5.

## Short-Run and Rerun Policy

Optuna 15-epoch runs are only for screening parameter candidates. The exported
top5 candidates must be reviewed and the top-k candidates should be rerun to 25
or 30 epochs before choosing a fixed setting. Missing Optuna trial checkpoints
do not invalidate top5 selection because top-k rerun trains from scratch by
default.

Because one 15-epoch trial is slow, the current recommended multi-machine
strategy is:

1. First round quick search: `max_epoch=5`, 2-4 trials per worker, 8-12 total
   trials across workers.
2. Second round: rerun the top3 candidates to 15 epochs.
3. Third round: rerun the top2 or top3 candidates to 30 epochs.
4. Final: fix one parameter set and run same-length baseline, quality-only, and
   full training.

Do not present short-run, Optuna, or rerun screening numbers as final paper
performance.

Single-machine SQLite debug command:

```bash
python scripts/new_method_v2/tuning/optuna_scanrefer_two_stage_full.py \
  --study-name dryrun_single \
  --storage sqlite:///reports/tuning/dryrun_single.db \
  --n-trials 1 \
  --max-epoch 1 \
  --dry-run
```

SQLite is for single-machine local debugging only. Multi-worker mode requires
PostgreSQL or MySQL storage. If `--multi-worker` is combined with SQLite, the
script fails with:

```text
SQLite storage is only supported for single-machine debugging. Use PostgreSQL/MySQL for multi-machine Optuna.
```

## Multi-Machine Optuna

All workers must use the same `--study-name` and `--storage`. Optuna allocates
different trials from the shared study with:

```python
optuna.create_study(
    study_name=args.study_name,
    storage=args.storage,
    direction="maximize",
    load_if_exists=True,
)
```

Each worker runs only its own `--n-trials` count. Use PostgreSQL/MySQL for the
shared storage:

```bash
createdb optuna_scanrefer

python scripts/new_method_v2/tuning/optuna_scanrefer_two_stage_full.py \
  --multi-worker \
  --study-name scanrefer_two_stage_full_rapf_quick5 \
  --storage postgresql://USER:PASSWORD@DB_HOST:5432/optuna_scanrefer \
  --worker-id worker01 \
  --n-trials 3 \
  --max-epoch 5 \
  --data-root /root/autodl-tmp/DATA_ROOT \
  --output-root logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5
```

Other servers should keep the same `--study-name` and `--storage`, changing only
`--worker-id` such as `worker02` or `worker03`. The helper launcher can be used
the same way:

```bash
STORAGE=postgresql://USER:PASSWORD@DB_HOST:5432/optuna_scanrefer \
WORKER_ID=worker01 \
N_TRIALS=3 \
MAX_EPOCH=5 \
DATA_ROOT=/root/autodl-tmp/DATA_ROOT \
OUTPUT_ROOT=logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5 \
bash scripts/new_method_v2/tuning/run_optuna_worker.sh
```

Worker output directories are isolated as:

```text
logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna_quick5/
  worker01/
    trial_0000/
  worker02/
    trial_0001/
```

Each trial records `worker_id`, `hostname`, `gpu`, `study_name`,
`storage_url_masked`, code version, Python executable, conda environment,
`effective_command`, config path, exact eval log path, exact checkpoint path,
`checkpoint_status`, objective, Acc@0.25, Acc@0.5, status, return code, and
error message. Passwords in storage URLs are masked in reports.

Top5 for multi-machine runs must be exported from the shared Optuna study, not
from one worker's local CSV:

```bash
python scripts/new_method_v2/tuning/export_optuna_top5.py \
  --study-name scanrefer_two_stage_full_rapf_quick5 \
  --storage postgresql://USER:PASSWORD@DB_HOST:5432/optuna_scanrefer \
  --output-dir reports/tuning
```

The export writes:

- `reports/tuning/optuna_scanrefer_two_stage_full_trials.csv`
- `reports/tuning/optuna_scanrefer_two_stage_full_best.json`
- `reports/tuning/optuna_scanrefer_two_stage_full_top5.json`
- `reports/tuning/optuna_scanrefer_two_stage_full_top5.csv`
- `reports/tuning/optuna_scanrefer_two_stage_full_summary.md`

The summary includes top5 by balanced objective, Acc@0.25, Acc@0.5, and
recommended top-k rerun candidates.

Default top-k rerun command:

```bash
python scripts/new_method_v2/tuning/rerun_top_optuna_scanrefer_two_stage.py \
  --top-k 3 \
  --max-epoch 30 \
  --top5-json reports/tuning/optuna_scanrefer_two_stage_full_top5.json \
  --output-root logs/new_method_v2/tuning/scanrefer_two_stage_top3_30epoch
```

The default rerun command does not include `--checkpoint_path`, `--resume`,
`--eval`, or `--eval_only`. Resume is allowed only when
`--resume-from-trial-checkpoint` is explicitly set.

Final claims require same-length formal training with fixed parameters for:

- baseline
- quality-only
- full model

SR3D and NR3D should not receive Optuna tuning or ablation sweeps in this pass.
After ScanRefer parameters are fixed, run SR3D/NR3D baseline, quality-only, and
full under the final schedule.

## Quality-Anchor RAPF Variant

`--rapf_quality_anchor_structured_residual` is an experimental variant and is
disabled by default. When enabled, RAPF anchors the structured residual on
`base + quality` instead of base only. It is still gated residual fusion, not a
standard MoE.

Any use of this flag must be reported as an experimental tuning variant unless
it is later promoted through full matched training.
