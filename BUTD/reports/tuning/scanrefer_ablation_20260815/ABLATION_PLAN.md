# ScanRefer dependency-aware ablation plan

Dataset scope: ScanRefer only. All rows independently retrain epochs 1--100 from the same verified official detector initialization, seed 0, validation every 5 epochs. Model selection uses official `last__bbs_acc0.25_top1`; only the strict-best checkpoint is retained and reloaded for the final evaluation.

## Module-level ablations (primary evidence)

| Logical configuration | Queue row | Question |
|---|---|---|
| BUTD baseline | `01_baseline` | Official same-protocol reference |
| SACR | `08_sacr_only` | Does SACR improve the baseline by itself? |
| QAHNL(base) | `03_no_sacr_rapf_qahnl_base` | Does QAHNL help base scores without SACR/RAPF? |
| SACR + QAHNL(structured) | `09_sacr_qahnl` | What is RAPF's marginal contribution in the full dependency graph? |
| SACR + RAPF | `04_no_qahnl` | What is QAHNL's marginal contribution on top of SACR+RAPF? |
| SACR + RAPF + QAHNL | `02_full_sacr_rapf_qahnl` | Total method gain |

RAPF requires SACR `structured_scores` in the implemented method. Therefore RAPF-only and RAPF+QAHNL-without-SACR are invalid, not omitted results. The six valid configurations form a dependency-aware lattice rather than an invalid full `2^3` factorial.

## Module-internal ablations (secondary evidence)

| Row | Question |
|---|---|
| `05_no_quality` | Does RAPF benefit from the learned quality signal? |
| `06_no_gate_supervision` | Does explicit reliability-gate supervision matter when the gate architecture is retained? |
| `07_no_relation` | Does SACR's relation branch contribute beyond target attributes? |
| `10_full_qahnl_base_source` | Does QAHNL benefit from fused evidence rather than raw base scores? |

## Interpretation constraints

- `04_no_qahnl` is identical to the logical SACR+RAPF row and is run only once.
- `09_sacr_qahnl` uses `qahnl_score_source=structured` and structured scores for evaluation, so removing RAPF does not accidentally remove QAHNL.
- Negative and null effects remain in the final table; no row may be deleted based on outcome.
- The final claim gate requires 10 independent run directories, 20 validation points per row, one retained best weight per row, exact checkpoint SHA256 receipts, and best-checkpoint reload parity.

Estimated serial compute: approximately 20 A100 days for ten rows at the observed roughly two-day cost per row.

## Execution order

The GPU queue enforces module-level priority: rows 01--04, then extension rows 08--09, then internal rows 05--07, and finally row 10. A temporary fail-closed gate wraps row 05 while rows 08--09 run; the queue restores the byte-identical frozen row-05 launcher (SHA256 89af902e...) before releasing that gate.

## Runtime smoke gate

A CPU-only synthetic runtime smoke passed before the extension rows were eligible to run. It exercised SACR forward, RAPF fusion, QAHNL base-source loss, QAHNL structured-source loss, and the fail-closed RAPF ordering test. GPU memory was 29069 MiB both before and after, so the active training job was not disturbed. Evidence files are module_runtime_smoke.log, module_runtime_smoke.receipt, module_runtime_smoke.sha256, and MODULE_RUNTIME_SMOKE_PASS in the extension queue directory.
