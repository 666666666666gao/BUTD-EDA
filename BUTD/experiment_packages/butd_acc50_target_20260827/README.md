# BUTD Acc@0.50 target branch (2026-08-27)

Objective: keep ScanRefer Overall Acc@0.25 >= 53.91 while increasing the same
checkpoint's Overall Acc@0.50 strictly above 42.41.

First stage is a read-only evaluation of the accepted Full/M3 checkpoint.  It
uses utterance-derived `text_target_cid` only and reports every already-defined
deployable detector-policy score source on the full validation denominator.
The temporary evaluator patch only records metrics; it does not change scores.

Accepted Full checkpoint:

`/home/gb/new butd/butd_detr-main/logs/butd_universal_target/three_targets_20260820/scanrefer_microtune_lr2e5_e6/scanrefer_spacy/1787171156/ckpt_best_primary.pth`

Accepted Full metrics: Overall Acc@0.25=54.3963, Acc@0.50=39.9032.

Safety controls:

- Existing ablation queue parent PID 114992 must remain SIGSTOP state `T`.
- The watcher waits for the current warm-start child to finish and for GPU 0 to
  become idle.
- The evaluator backup SHA is checked before evaluation and restored on every
  exit path.
- This stage creates no checkpoint.

Stage 1 starts automatically after the text-only diagnostic. It retains the
same SACR, RAPF, QAHNL, and QualityHead architecture and freezes the BUTD
backbone. Only the shared universal-module parameters are optimized from the
accepted Full checkpoint.

High-IoU controls:

- A CPU contract test runs before Stage 1 and verifies that IoU 0.50/0.25
  define the positive/negative sets, the ambiguous interval is gradient-free,
  and both positive-rescue paths are disabled.
- QAHNL positives require IoU >= 0.50 and negatives require IoU <= 0.25.
- Top-IoU and Hungarian positive rescue are disabled so low-IoU queries cannot
  enter the positive set.
- Quality classification uses the 0.50 threshold and top-5 rerank supervision.
- Validation runs after every epoch from E4 through E10.
- Selection first requires the same checkpoint to satisfy Overall Acc@0.25
  strictly above 53.91 and Acc@0.50 strictly above 42.41, then maximizes
  Overall Acc@0.25 to preserve the original 54.40 result as closely as possible.
- `best_checkpoint_only` retains exactly one training checkpoint.

After training, `verify_stage1_reload.sh` starts a separate full-validation
process, reloads the retained checkpoint, computes SHA256, and writes
`stage1_reload_verify/goal_receipt.json`. Only that receipt can establish that
the two strict thresholds are simultaneously met.

Before Stage 1, the already-scheduled Full-checkpoint diagnostic evaluates the
same 15 RAPF quality coefficients at no extra forward-pass cost. A feasible
coefficient is independently reloaded with the original evaluator into
`stage0_reload_verify/goal_receipt.json`; a positive receipt skips all longer
training stages, while a negative receipt releases Stage 1 normally.

Stage 2 runs only when the independent Stage-1 receipt is negative. It starts
from the closest-to-feasible Stage-1 checkpoint, unfreezes the full model, and
adds six low-LR epochs with stronger high-IoU QAHNL and quality supervision.

If Stage 2 also fails, Stage 3 keeps the exact same SACR/RAPF/QAHNL model and
calibrates only RAPF's existing quality coefficient. One patched diagnostic
evaluation measures 15 preregistered coefficients from 0.00 through 2.00 in a
single forward pass, then restores the evaluator byte-for-byte.

Stage 3 applies the same strict dual-threshold gate and, among feasible values,
maximizes Overall Acc@0.25. Any selected coefficient is then evaluated again in
a separate full reload with the original evaluator. Only
`stage3_reload_verify/goal_receipt.json` can establish a Stage-3 success.

The quality-grid reconstruction passed `RAPF_QUALITY_GRID_CONTRACT_PASS`; patch
application, Python compilation, SHA-verified restoration, shell syntax, and
actual argument parsing also passed before the watcher was deployed.
