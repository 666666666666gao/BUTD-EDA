# ScanRefer dual-machine ablation execution (2026-09-01)

This package executes the seven pending seed-0 internal ablations on two A100
servers. It reuses the audited M1, M2, S1 and matched-protocol Full results.
The published BUTD-DETR baseline is not retrained.

The audited launcher package is vendored under
`scanrefer_three_table_ablations_20260821/`; the default execution path no
longer depends on an experiment package outside this directory. An explicit
`BASE_PACKAGE` environment override remains available for reproducing the
already-running 2026-09-01 queues against their original frozen remote copy.

The main-result value registered as M3 is Stage154 (`54.9011/42.3538`). Stage154
is a fixed Stage142/Stage150 source-selection calibration result, not a single
SACR/RAPF/QAHNL checkpoint. This provenance must remain in the experiment record.

Machine assignment:

- `machine35608`: S2, S0, S3
- `machine50630`: R1, R3, R0, R2

Every newly trained row uses the same official detector initialization, seed 0,
batch size 24, 65 epochs, validation every 5 epochs, LR decays at epochs 55 and
60, and selects exactly one checkpoint by Overall Acc@0.25. Non-best weights are
not retained.

Validate without launching:

```bash
bash validate.sh
```

Launch one machine queue:

```bash
bash start_machine_queue.sh machine35608 S2 S0 S3
bash start_machine_queue.sh machine50630 R1 R3 R0 R2
```

The queues write machine-local status and SHA256 receipts under
`/root/autodl-tmp/logs/butd_scanrefer_dual_machine_ablations_20260901/`.
