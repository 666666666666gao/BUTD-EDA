# Tuning Reports

Files in this directory are exploratory tuning artifacts.

They are not official ablations, do not change large-scale readiness status, and
must not be used as final paper performance unless the selected parameters are
rerun with the formal training schedule.

Acc@0.25 and Acc@0.5 are both primary metrics. Tuning reports must show them
together and must not rank candidates only by Acc@0.5.

The Optuna objective is:

```text
0.5 * last__bbs_acc0.25_top1 + 0.5 * last__bbs_acc0.50_top1
```

Eval-only sweep is optional and checkpoint-based. It only tests forward-time
parameters on an explicitly provided existing checkpoint. `rapf_gate_loss_weight`
and `rapf_initial_gate_bias` must not be interpreted as effective eval-only
variables for an already trained checkpoint.

Epoch 50 remains historical evidence only. Its checkpoint is unavailable, so it
cannot be used for reproducible eval-only sweep or future tuning.

Recommended flow:

1. If an explicit checkpoint exists, optionally run eval-only fusion sweep.
2. If no checkpoint exists, run Optuna short-run directly.
3. Review the top5 balanced-objective short-run candidates exported from the
   study.
4. Rerun top3 or top5 to 15, then 25/30 epochs from scratch by default.
5. Fix parameters and run same-length formal baseline, quality-only, and full
   training before making final claims.

Current multi-machine recommendation:

- First quick search: `max_epoch=5`, 2-4 trials per worker, 8-12 total trials.
- Second round: top3 rerun to 15 epochs.
- Third round: top2 or top3 rerun to 30 epochs.
- Final: fixed-parameter same-length baseline, quality-only, and full training.

SQLite storage is only for single-machine local debugging. Multi-worker Optuna
must use PostgreSQL/MySQL storage and the same `study_name` on every worker.
Combining `--multi-worker` with SQLite fails with:

```text
SQLite storage is only supported for single-machine debugging. Use PostgreSQL/MySQL for multi-machine Optuna.
```

Single-machine debug entry:

```bash
python scripts/new_method_v2/tuning/optuna_scanrefer_two_stage_full.py \
  --study-name dryrun_single \
  --storage sqlite:///reports/tuning/dryrun_single.db \
  --n-trials 1 \
  --max-epoch 1 \
  --dry-run
```

PostgreSQL multi-worker entry:

```bash
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

Each worker changes only `--worker-id` and keeps the same study/storage. Worker
logs are isolated under `output-root/<worker_id>/trial_0000`.

Top5 export must read the shared study, not a single worker CSV:

```bash
python scripts/new_method_v2/tuning/export_optuna_top5.py \
  --study-name scanrefer_two_stage_full_rapf_quick5 \
  --storage postgresql://USER:PASSWORD@DB_HOST:5432/optuna_scanrefer \
  --output-dir reports/tuning
```

Default rerun entry:

```bash
python scripts/new_method_v2/tuning/rerun_top_optuna_scanrefer_two_stage.py \
  --top-k 3 \
  --max-epoch 30 \
  --top5-json reports/tuning/optuna_scanrefer_two_stage_full_top5.json \
  --output-root logs/new_method_v2/tuning/scanrefer_two_stage_top3_30epoch
```

SR3D/NR3D should not use Optuna or ablation sweeps in this workflow. After
ScanRefer parameters are fixed, run only baseline, quality-only, and full.
