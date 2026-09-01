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
  /root/autodl-tmp/logs/butd_scanrefer_dual_machine_ablations_20260901/<machine>/E20_READY \
  /root/autodl-tmp/logs/butd_scanrefer_dual_machine_ablations_20260901/<machine>/E20_SUMMARY.json
```

The collector verifies the declared evaluation-log path and SHA256, requires
all ten official BBS keys, and checks that the Unique/Multiple counts are
internally consistent with the 9,508-sample validation split. It exits without
writing a summary if any gate fails. It must not be run before `E20_READY`
exists.

Verified locally with:

```bash
python -W error::ResourceWarning -m unittest -v test_collect_e20_once.py
python -m py_compile collect_e20_once.py test_collect_e20_once.py
```

- `collect_e20_once.py` SHA256: `b00ec96a4535490b6c83a03f78632bb0170365e08049d9b9d9a6cd1de8d99727`
- `test_collect_e20_once.py` SHA256: `f80a2d6e879ed1010cd4e307d9933b3e88eb21308c60fb148d9e0db388d72922`

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
produces the final three-table JSON and Markdown; it rejects protocol drift,
missing or duplicate rows, false reload parity, and duplicate checkpoint
identities.

- `collect_completed_row.py` SHA256: `ad5da503a1bc91ef1fdffd85ebdb563b36d412b2b4e242cb8c935e70d82cdcfc`
- `test_collect_completed_row.py` SHA256: `5de71d67b4c72e8c3efed2d217d8b5165fcf6b7b94c24026b0548611b4e52fab`
- `assemble_final_tables.py` SHA256: `065d3988a0b55a20f151e1a6398fa239ced01889360e7695df41c62604fc1b17`
- `test_assemble_final_tables.py` SHA256: `df36c21393c0e435177fefd825b7bcb0ffbef88971fb59775195459788b96e08`
- `wait_and_collect_machine_rows.sh` SHA256: `4bbc1def7a5ec869035032248a7e98de43a46f53d69769da3f004ca511e6cda3`
- `start_machine_result_watcher.sh` SHA256: `e0e0129658b8ee9e9f258d8525e7f37895237f3bd4d1219fc982c93b52bb2974`
