# ScanRefer three-table ablations (2026-08-21)

This package is the active ablation plan. It supersedes the earlier two-table
queue while preserving every completed or already-running compatible row.

Paper tables:

1. Main modules: M0 baseline, M1 SACR, M2 SACR+RAPF, M3 Full.
2. SACR internals: S0 no target-attribute, S1 no relation-anchor, S2 no
   pairwise geometry, S3 hard top-1 anchor, S4 Full.
3. RAPF internals: R0 fixed fusion, R1 no quality cue, R2 no parser/anchor
   gate cues, R3 no gate supervision, R4 Full.

M0 is the published BUTD-DETR result and is not retrained. The main table uses
the accepted optimized M3 checkpoint (Unique 87.46/67.02, Multiple 48.60/35.15,
Overall 54.40/39.90, `rapf_quality_weight=0.25`). The SACR/RAPF internal tables
use the separate matched-protocol official-init 65-epoch Full control (Unique
85.98/65.89, Multiple 48.10/35.20, Overall 53.75/39.78,
`rapf_quality_weight=0.75`) as S4/R4. Both weights already exist and neither is
retrained. This split prevents the optimized M3 micro-tune from being mixed with
65-epoch one-factor internal rows while preserving the user-confirmed M3=54.40
in the main module table.

There are ten trainable seed-0 configurations. The already-running compatible
queue provides S1. The monotonic-main package independently trains the corrected
M1 and M2, copies their audited receipts here, and then this package trains S2,
S0, S3, R1, R3, R0, and R2 in that order. No seed-1/2 replications are scheduled.

All trainable rows use the official detector initialization and batch size 24.
M1/M2 use a 75-epoch cap, decay at 60/70, and preregistered dense validation at
epochs 55--62; the seven internal rows use 65 epochs, decay at 55/60, and
validation every 5 epochs. Every formal row reports all metrics from one selected
checkpoint, verifies it by reload, and retains exactly one best weight.

Start the waiting takeover queue:

```bash
bash experiment_packages/scanrefer_three_table_ablations_20260821/start_in_screen.sh
```
