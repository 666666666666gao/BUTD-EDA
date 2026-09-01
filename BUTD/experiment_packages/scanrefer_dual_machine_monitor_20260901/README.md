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
