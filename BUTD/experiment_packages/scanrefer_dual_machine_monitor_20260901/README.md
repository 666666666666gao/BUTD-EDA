# Server-local E20 milestone watcher

This monitor is deliberately separate from the frozen training package. It
checks the machine-local run directory every five minutes and writes a marker
only after the first assigned row has produced `eval_epoch_20.log`. It does not
read metrics, touch checkpoints, use a GPU, or poll through SSH.

```bash
bash start_e20_watcher.sh machine35608
bash start_e20_watcher.sh machine50630
```

After a watcher has written `E20_READY`, collect that milestone once with the
fail-closed collector:

```bash
/root/miniconda3/envs/bdetr/bin/python collect_e20_once.py \
  /root/autodl-tmp/logs/butd_scanrefer_dual_machine_ablations_20260901/<machine>/control/E20_READY \
  /root/autodl-tmp/logs/butd_scanrefer_dual_machine_ablations_20260901/<machine>/control/E20_SUMMARY.json
```

The collector verifies the declared evaluation-log path and SHA256, requires
all ten official BBS keys, and checks that the Unique/Multiple counts are
internally consistent with the 9,508-sample validation split. It exits without
writing a summary if any gate fails. It must not be run before `E20_READY`
exists.

For unattended one-shot collection, start the dependent server-local watcher:

```bash
bash start_e20_collector_watcher.sh machine35608
bash start_e20_collector_watcher.sh machine50630
```

It sleeps until `E20_READY`, invokes the same fail-closed collector exactly
once, and writes `control/E20_SUMMARY_READY` with the summary SHA256. Before
the milestone it does not open a training or evaluation log.

After both one-shot summaries are available, merge them without touching the
queues:

```bash
python merge_e20_summaries.py combined_e20.json combined_e20.md \
  machine35608_E20_SUMMARY.json machine50630_E20_SUMMARY.json
```

The merger accepts exactly the S2/machine35608 and R1/machine50630 E20
summaries, validates their sample contracts, metric/percentage identities and
distinct evaluation-log SHA256 values, then emits one explicitly provisional
JSON/Markdown snapshot. It never stops, restarts, or changes a queue.

Verified locally with:

```bash
python -W error::ResourceWarning -m unittest -v test_collect_e20_once.py
python -W error::ResourceWarning -m unittest -v test_merge_e20_summaries.py
python -m py_compile collect_e20_once.py test_collect_e20_once.py \
  merge_e20_summaries.py test_merge_e20_summaries.py
```

- `collect_e20_once.py` SHA256: `b00ec96a4535490b6c83a03f78632bb0170365e08049d9b9d9a6cd1de8d99727`
- `test_collect_e20_once.py` SHA256: `f80a2d6e879ed1010cd4e307d9933b3e88eb21308c60fb148d9e0db388d72922`
- `merge_e20_summaries.py` SHA256: `79a9526a8e32089d1beef074ea9a3588266c9f42ef9dc2fb59f92b86ff09140b`
- `test_merge_e20_summaries.py` SHA256: `1d3d55fe929548b1e3e3be797a35782af86f98bf9d3f2c6ca347106a4acf19a7`
- `wait_and_collect_e20_summary.sh` SHA256: `5094fe6f90589a4e47b508e4ad88094346d548b1c5e3549420c81a3b012ae2d8`
- `start_e20_collector_watcher.sh` SHA256: `f9e60cfa609e0a95539515fb7bd1f656a1576e9e1f025a52c10e77254faf3dd4`

## Final row collection

`collect_completed_row.py` audits one completed row before it can enter the
paper tables. It verifies the single retained checkpoint and SHA256, fixed
training protocol, strict-best score, full official BBS metric set, 9,508-row
Unique/Multiple contract, weighted Overall identity, and best-epoch versus
reload parity.

The machine-local completion watchers wait for `ALL_COMPLETE` and only then
collect the assigned rows:

```bash
bash start_machine_result_watcher.sh machine35608 S2 S0 S3
bash start_machine_result_watcher.sh machine50630 R1 R3 R0 R2
```

They write audited row JSON files under each machine's
`control/audited_rows/` and a SHA256 manifest named `MACHINE_RESULTS_READY`.
`assemble_final_tables.py` accepts exactly those seven audited JSON files and
produces the final three-table JSON and Markdown; with
`--output-latex-dir <directory>` it also creates three standalone paper tables
and one combined LaTeX include. It rejects protocol drift, missing or duplicate
rows, false reload parity, duplicate checkpoint identities, missing M0--M2
provenance, and any M3 record that is not the audited non-single-checkpoint
Stage154 fixed calibration result.

`finalize_dual_machine_tables.py` is the final consolidation gate. Given both
copied `MACHINE_RESULTS_READY` markers and the seven row JSON files, it verifies
the fixed machine-to-row assignments and every marker-declared SHA256, then
atomically publishes JSON, Markdown, the four LaTeX files and
`FINAL_TABLES_RECEIPT.json`. It refuses an existing output directory and leaves
no partial bundle after a failed check.

- `collect_completed_row.py` SHA256: `ad5da503a1bc91ef1fdffd85ebdb563b36d412b2b4e242cb8c935e70d82cdcfc`
- `test_collect_completed_row.py` SHA256: `5de71d67b4c72e8c3efed2d217d8b5165fcf6b7b94c24026b0548611b4e52fab`
- `assemble_final_tables.py` SHA256: `d4d33737bd9abf761665fe2837eeb508c81b99ef9b6070cb906ae84c935d8e43`
- `test_assemble_final_tables.py` SHA256: `6a66ca52ffecd0d2e9065d7eb629355b54edfd2f9fbe6f935842deedf74f94db`
- `finalize_dual_machine_tables.py` SHA256: `fe2094c130586eddad5e8619a06b4d199407d44c2a1f5b653ab1b91ae6842f2e`
- `test_finalize_dual_machine_tables.py` SHA256: `9a2f80ed383173651ae8b1230fc5309c560f027ad0b629e0700980f7243f5706`
- `wait_and_collect_machine_rows.sh` SHA256: `4bbc1def7a5ec869035032248a7e98de43a46f53d69769da3f004ca511e6cda3`
- `start_machine_result_watcher.sh` SHA256: `e0e0129658b8ee9e9f258d8525e7f37895237f3bd4d1219fc982c93b52bb2974`
