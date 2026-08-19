# MLCN Consistent Source-Choice Module Transfer Plan

日期：2026-06-14

## 目标

本路线把 BUTD-DETR 中的训练时 oracle 监督 source gate / source-choice selector 包装成通用模块，并移植到 MLCN。核心要求是：BUTD 和 MLCN 使用同一个方法定义、同一个训练/推理协议、同一类消融口径；不同 backbone 只通过 adapter 暴露候选框、候选特征和可部署 score source。

目标不是把 BUTD 的所有工程细节照搬到 MLCN，而是证明该模块可以作为通用 source arbitration 层迁移：

```text
Backbone-specific outputs -> Common SourceChoiceAdapter -> Generic SourceChoiceSelector -> selected deployable score source
```

若 BUTD 继续冲不到 `Acc@0.56`，MLCN 路线应成为论文主线备选：在更强或不同结构的 baseline 上验证文本分解 + source-choice selector + 困难负样本是否仍有稳定收益。

## 当前 MLCN/MCLN source-choice 最好指标

截至 `2026-06-20 11:05 UTC / 19:05 local`，文档中可复核的最好 MCLN source-choice 指标来自旧长训：

- run: `MCLN_source_choice_from_scratch_joint_seed0_long`
- log: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_from_scratch_joint_seed0_long/1781376797/log.txt`
- 口径：`batch_size=12`、full joint training、`use_source_choice_selector=true`、`source_choice_selector_train_only=false`、`sources=default,mask_text`。

当前最好可部署 source-choice / 正常 eval 口径是 epoch 13：

| metric source | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
| fixed_default / learned_selector | 0.51062 | 0.37390 |
| mask@kiou overall | 0.534497 | 0.421014 |

诊断 upper bound 口径：

- oracle 最好 `Acc@0.25`: epoch 14 `0.54985`。
- oracle 最好 `Acc@0.50`: epoch 13 `0.40082`。
- oracle 只作 source-choice headroom / upper-bound，不是可部署指标。

恢复状态：

- 旧长训最后日志停在 epoch 15 的 `Train: [15][2400/4054]` 左右。
- 旧 run 原配置 `save_freq=50`，epoch 1-14 没有保存 checkpoint。
- 已复查旧 run 目录和 `/root/autodl-tmp` / `/home/gb` 下的 `.pth/.pt/.ckpt`，没有发现 epoch 13/14/15 可恢复 checkpoint。
- 因此除非后续找到外部 checkpoint，否则不能从 epoch 13 精确续训；后续只能按旧配置重启，并把保存频率和 checkpoint 清理策略改为省盘且可恢复。

注意：`MCLN_source_choice_ratio_smoke_b1_seed0_20260620_002129` 的 `0.000xx` 指标只是 batch-1 selector-only ratio smoke 的工程闭环结果，不是当前最好指标，也不能与上面的 full joint 旧长训口径直接比较。

## 2026-06-14 执行记录

已确认 `MCLN-main` 中存在 MLCN 迁移实现：

- `models/source_choice_adapter.py`
- `models/source_choice_selector.py`
- `models/mcln.py`
- `models/losses.py`
- `src/grounding_evaluator.py`
- `scripts/audit_source_choice_adapter.py`
- `tests/test_*source_choice*.py`

新鲜复验：

```bash
cd /home/gb/new\ butd/butd_detr-main/MCLN-main
conda run -n bdetr python -m pytest -q \
  tests/test_source_choice_adapter.py \
  tests/test_source_choice_selector.py \
  tests/test_grounding_evaluator_source_choice.py \
  tests/test_main_utils_source_choice_checkpoint.py
# 8 passed in 2.01s

conda run -n bdetr python -m py_compile \
  models/source_choice_adapter.py \
  models/source_choice_selector.py \
  models/mcln.py \
  models/losses.py \
  src/grounding_evaluator.py \
  main_utils.py \
  train_dist_mod.py \
  scripts/audit_source_choice_adapter.py
# exit 0
```

运行中长训：

- PID file: `/root/autodl-tmp/DATA_ROOT/output/run_control/mcln_source_choice_from_scratch_joint_seed0_long.pid`
- main PID: `56614`; python worker: `56664`
- log: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_from_scratch_joint_seed0_long/1781376797/log.txt`
- 最近状态：epoch 10 eval 完成，epoch 11 training 进行中。

epoch 10 source-choice diagnostics：

| source | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
| fixed default | 0.48885 | 0.35044 |
| fixed mask_text | 0.23580 | 0.10255 |
| learned selector | 0.48885 | 0.35044 |
| oracle | 0.53040 | 0.37316 |

结论：2-source oracle 有明显 headroom，但 learned selector 当前仍等同 fixed default。长训可以继续观察，但这个 run 不能替代阶段 4/5 要求的带完整 ratio diagnostics 的 selector smoke/short gate。

已修复的诊断缺口：

- `models/source_choice_selector.py` 现在额外输出 `source_choice_false_override_ratio`。
- `main_utils.py` 现在会在 train/eval print 点输出 `[source_choice]` 诊断行，包含 `source_choice_*acc*` 和 `source_choice_*ratio*` 标量。
- 该补丁只影响后续新启动的进程；当前 PID `56664` 已加载旧代码，不会在现有 log 中出现新 `[source_choice]` 行。

下一步建议优先启动一个新的 1 epoch selector smoke 或 selector-only 2-3 epoch short run，确认 `target_non_default_ratio`、`selected_non_default_ratio`、`false_override_ratio` 和 `target_acc` 后，再决定是否保留或重启 long train。

### 2026-06-14 15:48 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `20:55:06`，CPU `114%`
- log mtime: `2026-06-14 23:45:33 +0800`
- 最新日志位置：`Train: [11][1600/4054]`
- 最新完整 source-choice eval 仍是 epoch 10；尚无 epoch 11 eval 结果。

最新 train print：

```text
[06/14 23:45:33] root INFO: Train: [11][1600/4054]
source_choice_loss 0.0365
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 旧长训日志仍没有新的 `[source_choice]` ratio 行，这是预期行为；该进程已在诊断补丁前加载旧代码。
- 当前代码中已存在 `source_choice_target_non_default_ratio`、`source_choice_selected_non_default_ratio`、`source_choice_false_override_ratio` 和 `[source_choice]` 文本日志逻辑，下一次新启动 smoke/short run 才能验证这些诊断的真实运行输出。

当前判断不变：epoch 10 oracle headroom 成立，但 learned selector 仍等同 fixed default；在没有 ratio/false-override 诊断的新 smoke/short run 之前，不应把当前 long run 作为阶段 4/5 验收证据。

### 2026-06-14 15:52 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `20:58:19`，CPU `114%`，MEM `3.3%`
- log mtime: `2026-06-14 23:50:46 +0800`
- log size: `162329` bytes
- 最新日志位置：`Train: [11][1800/4054]`
- 最新完整 source-choice eval 仍是 epoch 10；尚无 epoch 11 eval 结果。

最新 train print：

```text
[06/14 23:50:46] root INFO: Train: [11][1800/4054]
source_choice_loss 0.0365
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，这是预期行为。
- epoch 10 source-choice diagnostics 仍是最新完整证据：learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：该 long run 可继续观察 epoch 11 之后的 eval，但不能作为阶段 4/5 selector smoke/short gate 的验收证据。下一步仍应以新启动的 1 epoch smoke 或 selector-only 2-3 epoch short run 验证 ratio/false-override 诊断。

### 2026-06-14 15:57 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Sl`，elapsed `21:04:05`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-14 23:56:34 +0800`
- log size: `162775` bytes
- 最新稳定日志位置：`Train: [11][2000/4054]`
- 最新完整 source-choice eval 仍是 epoch 10；尚无 epoch 11 eval 结果。

最新 train print：

```text
[06/14 23:56:34] root INFO: Train: [11][2000/4054]
source_choice_loss 0.0367
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- 目前没有新的 source-choice diagnostics；epoch 10 仍是最新完整证据，learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：该 long run 可继续跑到 epoch 11 eval 以观察趋势，但阶段 4/5 验收仍必须依赖后续新启动的带 ratio/false-override 诊断的 smoke/short run。

### 2026-06-14 16:02 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `21:08:32`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 00:01:47 +0800`
- log size: `163221` bytes
- 最新稳定日志位置：`Train: [11][2200/4054]`
- 最新完整 source-choice eval 仍是 epoch 10；尚无 epoch 11 eval 结果。

最新 train print：

```text
[06/15 00:01:47] root INFO: Train: [11][2200/4054]
source_choice_loss 0.0370
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- 目前没有新的 source-choice diagnostics；epoch 10 仍是最新完整证据，learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：该 long run 可继续观察 epoch 11 eval，但阶段 4/5 smoke/short gate 仍必须依赖后续新启动的带 ratio/false-override 诊断的 run。

### 2026-06-14 16:08 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Sl`，elapsed `21:14:29`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 00:06:50 +0800`
- log size: `163667` bytes
- 最新稳定日志位置：`Train: [11][2400/4054]`
- 最新完整 source-choice eval 仍是 epoch 10；尚无 epoch 11 eval 结果。

最新 train print：

```text
[06/15 00:06:50] root INFO: Train: [11][2400/4054]
source_choice_loss 0.0369
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- 目前没有新的 source-choice diagnostics；epoch 10 仍是最新完整证据，learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：该 long run 可继续跑到 epoch 11 eval 以观察趋势，但不能作为阶段 4/5 selector smoke/short gate 的验收证据；带 `target_non_default_ratio`、`selected_non_default_ratio`、`false_override_ratio` 的新 smoke/short run 仍是必要下一步。

### 2026-06-14 16:13 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `21:19:32`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 00:11:33 +0800`
- log size: `164113` bytes
- 最新稳定日志位置：`Train: [11][2600/4054]`
- 最新完整 source-choice eval 仍是 epoch 10；尚无 epoch 11 eval 结果。

最新 train print：

```text
[06/15 00:11:33] root INFO: Train: [11][2600/4054]
source_choice_loss 0.0371
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- 目前没有新的 source-choice diagnostics；epoch 10 仍是最新完整证据，learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：该 long run 可继续观察 epoch 11 eval，但阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 16:16 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `21:23:22`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 00:16:10 +0800`
- log size: `164559` bytes
- 最新稳定日志位置：`Train: [11][2800/4054]`
- 最新完整 source-choice eval 仍是 epoch 10；尚无 epoch 11 eval 结果。

最新 train print：

```text
[06/15 00:16:10] root INFO: Train: [11][2800/4054]
source_choice_loss 0.0374
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- 目前没有新的 source-choice diagnostics；epoch 10 仍是最新完整证据，learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：该 long run 可继续观察 epoch 11 eval，但阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 16:21 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Sl`，elapsed `21:28:14`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 00:20:35 +0800`
- log size: `165005` bytes
- 最新稳定日志位置：`Train: [11][3000/4054]`
- 最新完整 source-choice eval 仍是 epoch 10；尚无 epoch 11 eval 结果。

最新 train print：

```text
[06/15 00:20:35] root INFO: Train: [11][3000/4054]
source_choice_loss 0.0375
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- 目前没有新的 source-choice diagnostics；epoch 10 仍是最新完整证据，learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：该 long run 可继续观察 epoch 11 eval，但阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 16:26 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `21:32:50`，CPU `114%`，MEM `3.3%`
- log mtime: `2026-06-15 00:24:55 +0800`
- log size: `165451` bytes
- 最新稳定日志位置：`Train: [11][3200/4054]`
- 最新完整 source-choice eval 仍是 epoch 10；尚无 epoch 11 eval 结果。

最新 train print：

```text
[06/15 00:24:55] root INFO: Train: [11][3200/4054]
source_choice_loss 0.0377
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- 目前没有新的 source-choice diagnostics；epoch 10 仍是最新完整证据，learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：该 long run 可继续观察 epoch 11 eval，但阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 16:31 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Sl`，elapsed `21:38:35`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 00:29:14 +0800`
- log size: `165897` bytes
- 最新稳定日志位置：`Train: [11][3400/4054]`
- stdout 进度已到约 `3519/4054`，但稳定日志尚未打印 `3600/4054`
- 最新完整 source-choice eval 仍是 epoch 10；尚无 epoch 11 eval 结果。

最新 train print：

```text
[06/15 00:29:14] root INFO: Train: [11][3400/4054]
source_choice_loss 0.0378
```

最新完整 source-choice diagnostics 仍为 epoch 10：

```text
fixed_default Acc0.25 Top-1: 0.48885, Acc0.50 Top-1: 0.35044
fixed_mask_text Acc0.25 Top-1: 0.23580, Acc0.50 Top-1: 0.10255
learned_selector Acc0.25 Top-1: 0.48885, Acc0.50 Top-1: 0.35044
oracle Acc0.25 Top-1: 0.53040, Acc0.50 Top-1: 0.37316
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- learned selector 仍与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：该 long run 可继续观察 epoch 11 eval，但阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 16:34 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `21:40:56`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 00:33:32 +0800`
- log size: `166343` bytes
- 最新稳定日志位置：`Train: [11][3600/4054]`
- stdout 进度已到约 `3629/4054`
- 最新完整 source-choice eval 仍是 epoch 10；尚无 epoch 11 eval 结果。

最新 train print：

```text
[06/15 00:33:32] root INFO: Train: [11][3600/4054]
source_choice_loss 0.0378
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- 目前没有新的 source-choice diagnostics；epoch 10 仍是最新完整证据，learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：该 long run 可继续观察 epoch 11 eval，但阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 16:38 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Sl`，elapsed `21:45:17`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 00:37:51 +0800`
- log size: `166789` bytes
- 最新稳定日志位置：`Train: [11][3800/4054]`
- stdout 进度已到约 `3831/4054`
- 最新完整 source-choice eval 仍是 epoch 10；尚无 epoch 11 eval 结果。

最新 train print：

```text
[06/15 00:37:51] root INFO: Train: [11][3800/4054]
source_choice_loss 0.0377
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- 目前没有新的 source-choice diagnostics；epoch 10 仍是最新完整证据，learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：该 long run 可继续观察 epoch 11 eval，但阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 16:44 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `21:51:06`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 00:43:18 +0800`
- log size: `167330` bytes
- 最新稳定日志位置：`epoch 11, total time 6089.57`
- epoch 11 最后一个 train print：`Train: [11][4000/4054]`
- stdout 显示 eval 已启动，进度约 `59/793`；稳定日志尚未打印 `Eval: [200/793]`
- 最新完整 source-choice eval 仍是 epoch 10；尚无 epoch 11 eval metrics / diagnostics。

最新 train / epoch-end print：

```text
[06/15 00:42:08] root INFO: Train: [11][4000/4054]
source_choice_loss 0.0374
[06/15 00:43:18] root INFO: epoch 11, total time 6089.57, lr_base 0.00020, lr_pointnet 0.00200
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- 目前没有新的 source-choice diagnostics；epoch 10 仍是最新完整证据，learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：epoch 11 eval 已开始但尚无稳定日志结果；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 16:47 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Sl`，elapsed `21:54:27`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 00:46:44 +0800`
- log size: `167771` bytes
- 最新稳定日志位置：epoch 11 eval `Eval: [200/793]`
- stdout eval 进度约 `252/793`
- 尚无 epoch 11 完整 eval metrics / source-choice diagnostics。

最新 eval print：

```text
[06/15 00:46:44] root INFO: Eval: [200/793]
source_choice_loss 0.0356
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- 目前没有新的 source-choice diagnostics；epoch 10 仍是最新完整 source-choice 证据，learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：epoch 11 eval 正在推进但尚无完整结果；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 16:51 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `21:58:27`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 00:50:15 +0800`
- log size: `168212` bytes
- 最新稳定日志位置：epoch 11 eval `Eval: [400/793]`
- stdout eval 进度约 `480/793`
- 尚无 epoch 11 完整 eval metrics / source-choice diagnostics。

最新 eval print：

```text
[06/15 00:50:15] root INFO: Eval: [400/793]
source_choice_loss 0.0332
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- 目前没有新的 source-choice diagnostics；epoch 10 仍是最新完整 source-choice 证据，learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：epoch 11 eval 正在推进但尚无完整结果；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 16:55 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `22:02:09`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 00:53:43 +0800`
- log size: `168653` bytes
- 最新稳定日志位置：epoch 11 eval `Eval: [600/793]`
- stdout eval 进度约 `696/793`
- 尚无 epoch 11 完整 eval metrics / source-choice diagnostics。

最新 eval print：

```text
[06/15 00:53:43] root INFO: Eval: [600/793]
source_choice_loss 0.0346
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- 目前没有新的 source-choice diagnostics；epoch 10 仍是最新完整 source-choice 证据，learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：epoch 11 eval 已过 `600/793`，完整 metrics 与 source-choice diagnostics 尚未落盘；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 17:00 UTC 追加跟踪

epoch 11 完整 eval / source-choice diagnostics 已落盘，长训继续进入下一轮训练，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `22:07:20`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 00:57:00 +0800`
- log size: `173928` bytes
- 最新稳定日志位置：epoch 11 `Source choice diagnostics`
- stdout 显示已进入 epoch 12 训练，进度约 `143/4054`；稳定日志尚未打印 `Train: [12][200/4054]`

epoch 11 主要 grounding / mask 指标：

```text
last_ position Acc0.25 Top-1: 0.49758
last_ position Acc0.50 Top-1: 0.34518
last_ semantic Acc0.25 Top-1: 0.50536
last_ semantic Acc0.50 Top-1: 0.34876
mask@kiou overall25: 0.5269246949936895
mask@kiou overall50: 0.42248632730332353
```

epoch 11 source-choice diagnostics：

```text
fixed_default Acc0.25 Top-1: 0.49758, Acc0.50 Top-1: 0.34518
fixed_mask_text Acc0.25 Top-1: 0.23454, Acc0.50 Top-1: 0.11149
learned_selector Acc0.25 Top-1: 0.49758, Acc0.50 Top-1: 0.34518
oracle Acc0.25 Top-1: 0.53965, Acc0.50 Top-1: 0.36864
```

与 epoch 10 最新完整 source-choice eval 对比：

- fixed default / learned selector：Acc@0.25 从 `0.48885` 升到 `0.49758`，Acc@0.50 从 `0.35044` 降到 `0.34518`。
- fixed mask_text：Acc@0.25 从 `0.23580` 降到 `0.23454`，Acc@0.50 从 `0.10255` 升到 `0.11149`。
- oracle：Acc@0.25 从 `0.53040` 升到 `0.53965`，Acc@0.50 从 `0.37316` 降到 `0.36864`。
- epoch 11 oracle 相对 learned/default 仍有 headroom：Acc@0.25 `+0.04207`，Acc@0.50 `+0.02346`。

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- learned selector 仍与 fixed default 完全持平，没有显示出 selector 学到 source 切换收益。

当前判断继续不变：该 long run 已给出 epoch 11 完整 eval，结论仍是 oracle 有 headroom，但 learned selector 未超过 fixed default；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 17:03 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `22:10:09`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 01:01:53 +0800`
- log size: `174373` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][200/4054]`
- stdout 显示 epoch 12 训练继续推进到约 `254/4054`

最新 train print：

```text
[06/15 01:01:53] root INFO: Train: [12][200/4054]
source_choice_loss 0.0389
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：长训已进入 epoch 12，但 gate 所需的 selector ratio / false-override 诊断仍不在该旧进程中；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 17:06 UTC 追加跟踪

长训仍在运行，未手动中断：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `22:12:35`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 01:01:53 +0800`
- log size: `174373` bytes
- 最新稳定日志位置仍是 epoch 12 train `Train: [12][200/4054]`
- stdout 显示 epoch 12 训练继续推进到约 `358/4054`

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：当前只是 epoch 12 训练继续推进，尚无新的稳定 train print / eval / source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 17:08 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 17:08:06 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `22:14:52`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 01:06:51 +0800`
- log size: `174818` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][400/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `452/4054`

最新 train print：

```text
[06/15 01:06:51] root INFO: Train: [12][400/4054]
loss 9.3907
source_choice_loss 0.0387
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：epoch 12 train 已稳定推进到 `400/4054`，但尚无新的 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 17:13 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 17:13:17 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `22:20:03`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 01:11:49 +0800`
- log size: `175263` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][600/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `659/4054`

最新 train print：

```text
[06/15 01:11:49] root INFO: Train: [12][600/4054]
loss 9.3627
source_choice_loss 0.0385
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：epoch 12 train 已稳定推进到 `600/4054`，但尚无新的 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 17:17 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 17:17:26 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `22:24:11`，CPU `114%`，MEM `3.3%`
- log mtime: `2026-06-15 01:16:41 +0800`
- log size: `175708` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][800/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `833/4054`

最新 train print：

```text
[06/15 01:16:41] root INFO: Train: [12][800/4054]
loss 9.3561
source_choice_loss 0.0383
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle 仍有 headroom。

当前判断继续不变：epoch 12 train 已稳定推进到 `800/4054`，但尚无新的 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 17:21 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 17:21:40 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `22:28:25`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 01:21:13 +0800`
- log size: `176154` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][1000/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `1019/4054`

最新 train print：

```text
[06/15 01:21:13] root INFO: Train: [12][1000/4054]
loss 9.3331
source_choice_loss 0.0379
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定推进到 `1000/4054`，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 17:26 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 17:26:01 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Sl`，elapsed `22:32:46`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 01:25:36 +0800`
- log size: `176600` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][1200/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `1218/4054`

最新 train print：

```text
[06/15 01:25:36] root INFO: Train: [12][1200/4054]
loss 9.3257
source_choice_loss 0.0380
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定推进到 `1200/4054`，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 17:30 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 17:30:32 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `22:37:17`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 01:30:04 +0800`
- log size: `177046` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][1400/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `1420/4054`

最新 train print：

```text
[06/15 01:30:04] root INFO: Train: [12][1400/4054]
loss 9.3321
source_choice_loss 0.0381
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定推进到 `1400/4054`，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 17:34 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 17:34:53 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Sl`，elapsed `22:41:38`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 01:34:25 +0800`
- log size: `177492` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][1600/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `1621/4054`

最新 train print：

```text
[06/15 01:34:25] root INFO: Train: [12][1600/4054]
loss 9.3330
source_choice_loss 0.0379
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定推进到 `1600/4054`，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 17:39 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 17:39:10 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Sl`，elapsed `22:45:55`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 01:38:43 +0800`
- log size: `177938` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][1800/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `1820/4054`

最新 train print：

```text
[06/15 01:38:43] root INFO: Train: [12][1800/4054]
loss 9.3285
source_choice_loss 0.0375
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定推进到 `1800/4054`，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 17:43 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 17:43:28 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `22:50:13`，CPU `115%`，MEM `3.2%`
- log mtime: `2026-06-15 01:43:00 +0800`
- log size: `178384` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][2000/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `2021/4054`

最新 train print：

```text
[06/15 01:43:00] root INFO: Train: [12][2000/4054]
loss 9.3393
source_choice_loss 0.0374
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定推进到 `2000/4054`，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 17:47 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 17:47:54 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `22:54:39`，CPU `115%`，MEM `3.2%`
- log mtime: `2026-06-15 01:47:21 +0800`
- log size: `178830` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][2200/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `2224/4054`

最新 train print：

```text
[06/15 01:47:21] root INFO: Train: [12][2200/4054]
loss 9.3369
source_choice_loss 0.0375
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定推进到 `2200/4054`，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 17:52 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 17:52:20 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `22:59:05`，CPU `115%`，MEM `3.2%`
- log mtime: `2026-06-15 01:51:44 +0800`
- log size: `179276` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][2400/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `2428/4054`

最新 train print：

```text
[06/15 01:51:44] root INFO: Train: [12][2400/4054]
loss 9.3218
source_choice_loss 0.0376
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定推进到 `2400/4054`，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 17:58 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 17:58:47 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Sl`，elapsed `23:05:32`，CPU `115%`，MEM `3.3%`
- log mtime: `2026-06-15 01:55:59 +0800`
- log size: `179722` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][2600/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `2703/4054`

最新 train print：

```text
[06/15 01:55:59] root INFO: Train: [12][2600/4054]
loss 9.3068
source_choice_loss 0.0375
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定推进到 `2600/4054`，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 18:01 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 18:01:52 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `23:08:37`，CPU `115%`，MEM `3.2%`
- log mtime: `2026-06-15 02:01:39 +0800`
- log size: `180168` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][2800/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `2809/4054`

最新 train print：

```text
[06/15 02:01:39] root INFO: Train: [12][2800/4054]
loss 9.2889
source_choice_loss 0.0377
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定推进到 `2800/4054`，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 18:08 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 18:08:58 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Sl`，elapsed `23:15:43`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 02:07:16 +0800`
- log size: `180614` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][3000/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `3061/4054`

最新 train print：

```text
[06/15 02:07:16] root INFO: Train: [12][3000/4054]
loss 9.2905
source_choice_loss 0.0377
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定推进到 `3000/4054`，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 18:13 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 18:13:15 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `23:20:00`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 02:12:47 +0800`
- log size: `181060` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][3200/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `3220/4054`

最新 train print：

```text
[06/15 02:12:47] root INFO: Train: [12][3200/4054]
loss 9.3052
source_choice_loss 0.0379
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定推进到 `3200/4054`，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 18:19 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 18:19:24 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `23:26:09`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 02:18:14 +0800`
- log size: `181506` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][3400/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `3437/4054`

最新 train print：

```text
[06/15 02:18:14] root INFO: Train: [12][3400/4054]
loss 9.3134
source_choice_loss 0.0379
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定推进到 `3400/4054`，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 18:24 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 18:24:43 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Sl`，elapsed `23:31:28`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 02:23:46 +0800`
- log size: `181952` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][3600/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `3635/4054`

最新 train print：

```text
[06/15 02:23:46] root INFO: Train: [12][3600/4054]
loss 9.3077
source_choice_loss 0.0376
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定推进到 `3600/4054`，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 18:30 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 18:30:04 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `23:36:49`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 02:29:19 +0800`
- log size: `182398` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][3800/4054]`
- stdout 显示 epoch 12 训练已至少推进到 `3824/4054`

最新 train print：

```text
[06/15 02:29:19] root INFO: Train: [12][3800/4054]
loss 9.3180
source_choice_loss 0.0374
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定推进到 `3800/4054`，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 18:37 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 18:37:55 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Sl`，elapsed `23:44:41`，CPU `114%`，MEM `3.3%`
- log mtime: `2026-06-15 02:36:23 +0800`
- log size: `182939` bytes
- 最新稳定日志位置：epoch 12 train `Train: [12][4000/4054]`；stable log 随后写入 epoch 12 完成行
- stdout 显示 epoch 12 训练已完成并进入 `Test evaluation`，评估至少推进到 `74/793`

最新 train print：

```text
[06/15 02:34:51] root INFO: Train: [12][4000/4054]
loss 9.3076
source_choice_loss 0.0375
```

稳定日志中的 epoch 完成行：

```text
[06/15 02:36:23] root INFO: epoch 12, total time 5963.74, lr_base 0.00020, lr_pointnet 0.00200
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 train 已稳定完成并进入 eval，但尚无新的 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 18:41 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 18:41:04 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `23:47:49`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 02:40:21 +0800`
- log size: `183380` bytes
- 最新稳定日志位置：epoch 12 eval `Eval: [200/793]`
- stdout 显示 epoch 12 eval 已至少推进到 `241/793`

最新 eval print：

```text
[06/15 02:40:21] root INFO: Eval: [200/793]
loss 15.8773
source_choice_loss 0.0370
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 eval 已稳定推进到 `200/793`，但尚无完整 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 18:44 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 18:44:43 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `23:51:28`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 02:44:04 +0800`
- log size: `183821` bytes
- 最新稳定日志位置：epoch 12 eval `Eval: [400/793]`
- stdout 显示 epoch 12 eval 已至少推进到 `431/793`

最新 eval print：

```text
[06/15 02:44:04] root INFO: Eval: [400/793]
loss 15.5636
source_choice_loss 0.0362
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 eval 已稳定推进到 `400/793`，但尚无完整 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 18:48 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 18:48:37 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `23:55:23`，CPU `115%`，MEM `3.2%`
- log mtime: `2026-06-15 02:47:54 +0800`
- log size: `184262` bytes
- 最新稳定日志位置：epoch 12 eval `Eval: [600/793]`
- stdout 显示 epoch 12 eval 已至少推进到 `639/793`

最新 eval print：

```text
[06/15 02:47:54] root INFO: Eval: [600/793]
loss 15.4726
source_choice_loss 0.0368
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程。
- epoch 11 仍是最新完整 source-choice diagnostics；learned selector 与 fixed default 完全持平，oracle headroom 仍为 Acc@0.25 `+0.04207`、Acc@0.50 `+0.02346`。

当前判断继续不变：epoch 12 eval 已稳定推进到 `600/793`，但尚无完整 epoch 12 eval/source-choice diagnostics；阶段 4/5 gate 仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 18:53 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 18:53:37 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Sl`，elapsed `1-00:00:22`，CPU `115%`，MEM `3.2%`
- log mtime: `2026-06-15 02:51:39 +0800`
- log size: `189534` bytes
- 最新稳定日志位置：epoch 12 eval 已完成，并写出完整 source-choice diagnostics
- stdout 显示 epoch 12 eval 已完成 `793/793`，并已进入下一轮 train，进度条至少到 `60/4054`

最新 epoch 12 eval/source-choice diagnostics：

```text
[06/15 02:51:39] root INFO: last_ position alignment Acc0.25: Top-1: 0.50831
[06/15 02:51:39] root INFO: last_ position alignment Acc0.50: Top-1: 0.37042
[06/15 02:51:39] root INFO: last_ semantic alignment Acc0.25: Top-1: 0.51325
[06/15 02:51:39] root INFO: last_ semantic alignment Acc0.50: Top-1: 0.37190
[06/15 02:51:39] root INFO: overall25 0.5309213294068154
[06/15 02:51:39] root INFO: overall50 0.4212242322254943

Source choice diagnostics
[06/15 02:51:39] root INFO: fixed_default Acc0.25 Top-1: 0.50831, Acc0.50 Top-1: 0.37042
[06/15 02:51:39] root INFO: fixed_mask_text Acc0.25 Top-1: 0.25137, Acc0.50 Top-1: 0.12631
[06/15 02:51:39] root INFO: learned_selector Acc0.25 Top-1: 0.50831, Acc0.50 Top-1: 0.37042
[06/15 02:51:39] root INFO: oracle Acc0.25 Top-1: 0.54491, Acc0.50 Top-1: 0.39167
```

细分指标同步记录：

- iou@0.25: easy `0.7294164668265388`，hard `0.4360548101627177`，vd `0.4831135994220697`，vid `0.5552757491815663`，unique `0.821000704721635`，multi `0.45926566942761776`
- iou@0.50: easy50 `0.5675459632294164`，hard50 `0.30202683414216386`，vd50 `0.3527180783817952`，vid50 `0.39864014102241246`，unique50 `0.6412966878083157`，multi50 `0.3246383978242057`
- mask@kiou: unique25 `0.8653981677237491`，unique50 `0.6828752642706131`，multi25 `0.4722462603535666`，multi50 `0.3753245147731487`，overall25 `0.5309213294068154`，overall50 `0.4212242322254943`
- mask@identity: vd25 `0.49394979230630304`，vd50 `0.391547769550298`，vid25 `0.5824729287333166`，vid50 `0.4626038781163435`，easy25 `0.7621902478017586`，easy50 `0.6027178257394085`，hard25 `0.448330002854696`，hard50 `0.35640879246360263`

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断更新为：epoch 12 已有完整稳定 eval/source-choice diagnostics，但 learned selector 尚未相对 fixed default 产生收益；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 18:58 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 18:58:06 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `1-00:04:51`，CPU `115%`，MEM `3.2%`
- log mtime: `2026-06-15 02:57:17 +0800`
- log size: `189979` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][200/4054]`
- stdout 显示 epoch 13 train 已至少推进到 `229/4054`

最新 train print：

```text
[06/15 02:57:17] root INFO: Train: [13][200/4054]
adaptive_weight_loss_dice 0.2804
adaptive_weight_loss_mask 0.0104
corresponding_loss_dice 0.2681
corresponding_loss_mask 0.0141
loss 8.9789
loss_bbox 1.8551
loss_ce 15.3259
loss_dice 0.8351
loss_giou 3.3344
loss_mask 0.0123
loss_sem_align 35.6643
query_points_generation_loss 0.0041
source_choice_loss 0.0342
sp_loss_dice 0.2798
sp_loss_mask 0.0106
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 已有稳定 train checkpoint，`source_choice_loss` 从 epoch 12 eval print 的 `0.0368` 降到本次 train print 的 `0.0342`，但这不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 19:03 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 19:02:56 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `1-00:09:41`，CPU `114%`，MEM `3.3%`
- log mtime: `2026-06-15 03:02:49 +0800`
- log size: `190424` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][400/4054]`
- stdout 显示 epoch 13 train 已至少推进到 `406/4054`

最新 train print：

```text
[06/15 03:02:49] root INFO: Train: [13][400/4054]
adaptive_weight_loss_dice 0.2820
adaptive_weight_loss_mask 0.0107
corresponding_loss_dice 0.2713
corresponding_loss_mask 0.0143
loss 9.2065
loss_bbox 1.8940
loss_ce 16.1029
loss_dice 0.8285
loss_giou 3.3439
loss_mask 0.0126
loss_sem_align 37.5681
query_points_generation_loss 0.0040
source_choice_loss 0.0360
sp_loss_dice 0.2814
sp_loss_mask 0.0108
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `400/4054`，但 `source_choice_loss 0.0360` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 19:09 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 19:08:50 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `1-00:14:58`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 03:08:09 +0800`
- log size: `190869` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][600/4054]`
- stdout 显示 epoch 13 train 已至少推进到 `623/4054`

最新 train print：

```text
[06/15 03:08:09] root INFO: Train: [13][600/4054]
adaptive_weight_loss_dice 0.2805
adaptive_weight_loss_mask 0.0107
corresponding_loss_dice 0.2727
corresponding_loss_mask 0.0145
loss 9.2679
loss_bbox 1.8808
loss_ce 16.4176
loss_dice 0.8256
loss_giou 3.3442
loss_mask 0.0127
loss_sem_align 38.3177
query_points_generation_loss 0.0040
source_choice_loss 0.0358
sp_loss_dice 0.2798
sp_loss_mask 0.0108
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `600/4054`，但 `source_choice_loss 0.0358` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 19:14 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 19:14:20 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `1-00:20:33`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 03:13:44 +0800`
- log size: `191314` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][800/4054]`
- stdout 显示 epoch 13 train 已至少推进到 `818/4054`

最新 train print：

```text
[06/15 03:13:44] root INFO: Train: [13][800/4054]
adaptive_weight_loss_dice 0.2787
adaptive_weight_loss_mask 0.0106
corresponding_loss_dice 0.2716
corresponding_loss_mask 0.0145
loss 9.2145
loss_bbox 1.8748
loss_ce 16.2301
loss_dice 0.8278
loss_giou 3.3308
loss_mask 0.0126
loss_sem_align 37.8922
query_points_generation_loss 0.0040
source_choice_loss 0.0371
sp_loss_dice 0.2780
sp_loss_mask 0.0107
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `800/4054`，但 `source_choice_loss 0.0371` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 19:19 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 19:19:44 UTC`）：

- main PID: `56614`
- python worker: `56664`
- worker 状态：`Rl`，elapsed `1-00:26:30`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 03:19:16 +0800`
- log size: `191760` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][1000/4054]`
- stdout 显示 epoch 13 train 已至少推进到 `1022/4054`

最新 train print：

```text
[06/15 03:19:16] root INFO: Train: [13][1000/4054]
adaptive_weight_loss_dice 0.2777
adaptive_weight_loss_mask 0.0105
corresponding_loss_dice 0.2708
corresponding_loss_mask 0.0145
loss 9.2095
loss_bbox 1.8682
loss_ce 16.2516
loss_dice 0.8264
loss_giou 3.3231
loss_mask 0.0126
loss_sem_align 37.9749
query_points_generation_loss 0.0040
source_choice_loss 0.0384
sp_loss_dice 0.2770
sp_loss_mask 0.0107
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `1000/4054`，但 `source_choice_loss 0.0384` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 19:26 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 19:26:34 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- child workers: `528082`, `528094`
- worker 状态：`Rl`，elapsed `1-00:33:03`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 03:24:33 +0800`
- log size: `192206` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][1200/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准。

最新稳定 train print：

```text
[06/15 03:24:33] root INFO: Train: [13][1200/4054]
adaptive_weight_loss_dice 0.2766
adaptive_weight_loss_mask 0.0105
corresponding_loss_dice 0.2714
corresponding_loss_mask 0.0145
loss 9.1673
loss_bbox 1.8510
loss_ce 16.1446
loss_dice 0.8281
loss_giou 3.3099
loss_mask 0.0126
loss_sem_align 37.6581
query_points_generation_loss 0.0040
source_choice_loss 0.0386
sp_loss_dice 0.2758
sp_loss_mask 0.0107
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `1200/4054`，但 `source_choice_loss 0.0386` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 19:31 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 19:31:12 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- child workers: `528082`, `528094`
- worker 状态：`Rl`，elapsed `1-00:37:57`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 03:30:04 +0800`
- log size: `192652` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][1400/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout 已至少推进到 `1445/4054`）。

最新稳定 train print：

```text
[06/15 03:30:04] root INFO: Train: [13][1400/4054]
adaptive_weight_loss_dice 0.2774
adaptive_weight_loss_mask 0.0106
corresponding_loss_dice 0.2712
corresponding_loss_mask 0.0145
loss 9.1650
loss_bbox 1.8589
loss_ce 16.1063
loss_dice 0.8292
loss_giou 3.3144
loss_mask 0.0126
loss_sem_align 37.5023
query_points_generation_loss 0.0040
source_choice_loss 0.0392
sp_loss_dice 0.2767
sp_loss_mask 0.0107
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `1400/4054`，但 `source_choice_loss 0.0392` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 19:36 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 19:36:33 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- child workers: `528082`, `528094`
- worker 状态：`Sl`，elapsed `1-00:43:18`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 03:35:27 +0800`
- log size: `193098` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][1600/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout 已至少推进到 `1638/4054`）。

最新稳定 train print：

```text
[06/15 03:35:27] root INFO: Train: [13][1600/4054]
adaptive_weight_loss_dice 0.2758
adaptive_weight_loss_mask 0.0105
corresponding_loss_dice 0.2710
corresponding_loss_mask 0.0146
loss 9.1401
loss_bbox 1.8487
loss_ce 16.0544
loss_dice 0.8293
loss_giou 3.3125
loss_mask 0.0126
loss_sem_align 37.3939
query_points_generation_loss 0.0040
source_choice_loss 0.0391
sp_loss_dice 0.2751
sp_loss_mask 0.0107
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `1600/4054`，但 `source_choice_loss 0.0391` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 19:42 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 19:42:20 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- child workers: `528082`, `528094`
- worker 状态：`Rl`，elapsed `1-00:49:05`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 03:40:52 +0800`
- log size: `193544` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][1800/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout 已至少推进到 `1852/4054`）。

最新稳定 train print：

```text
[06/15 03:40:52] root INFO: Train: [13][1800/4054]
adaptive_weight_loss_dice 0.2743
adaptive_weight_loss_mask 0.0105
corresponding_loss_dice 0.2713
corresponding_loss_mask 0.0147
loss 9.1509
loss_bbox 1.8371
loss_ce 16.1096
loss_dice 0.8280
loss_giou 3.3031
loss_mask 0.0127
loss_sem_align 37.6899
query_points_generation_loss 0.0040
source_choice_loss 0.0393
sp_loss_dice 0.2735
sp_loss_mask 0.0107
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `1800/4054`，但 `source_choice_loss 0.0393` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 19:47 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 19:47:34 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- child workers: `528082`, `528094`
- worker 状态：`Rl`，elapsed `1-00:54:19`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 03:46:21 +0800`
- log size: `193990` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][2000/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout 已至少推进到 `2050/4054`）。

最新稳定 train print：

```text
[06/15 03:46:21] root INFO: Train: [13][2000/4054]
adaptive_weight_loss_dice 0.2742
adaptive_weight_loss_mask 0.0105
corresponding_loss_dice 0.2709
corresponding_loss_mask 0.0147
loss 9.1361
loss_bbox 1.8392
loss_ce 16.0757
loss_dice 0.8280
loss_giou 3.3018
loss_mask 0.0127
loss_sem_align 37.5192
query_points_generation_loss 0.0040
source_choice_loss 0.0395
sp_loss_dice 0.2734
sp_loss_mask 0.0106
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `2000/4054`，但 `source_choice_loss 0.0395` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 19:52 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 19:52:48 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- child workers: `528082`, `528094`
- worker 状态：`Rl`，elapsed `1-00:59:33`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 03:51:36 +0800`
- log size: `194436` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][2200/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout 已至少推进到 `2239/4054`）。

最新稳定 train print：

```text
[06/15 03:51:36] root INFO: Train: [13][2200/4054]
adaptive_weight_loss_dice 0.2747
adaptive_weight_loss_mask 0.0105
corresponding_loss_dice 0.2709
corresponding_loss_mask 0.0147
loss 9.1454
loss_bbox 1.8424
loss_ce 16.0867
loss_dice 0.8278
loss_giou 3.3034
loss_mask 0.0127
loss_sem_align 37.5836
query_points_generation_loss 0.0040
source_choice_loss 0.0395
sp_loss_dice 0.2739
sp_loss_mask 0.0106
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `2200/4054`，但 `source_choice_loss 0.0395` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 19:58 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 19:58:03 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- child workers: `528082`, `528094`
- worker 状态：`Rl`，elapsed `1-01:04:48`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 03:57:08 +0800`
- log size: `194882` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][2400/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout 已至少推进到 `2435/4054`）。

最新稳定 train print：

```text
[06/15 03:57:08] root INFO: Train: [13][2400/4054]
adaptive_weight_loss_dice 0.2750
adaptive_weight_loss_mask 0.0105
corresponding_loss_dice 0.2711
corresponding_loss_mask 0.0146
loss 9.1447
loss_bbox 1.8400
loss_ce 16.0597
loss_dice 0.8273
loss_giou 3.3035
loss_mask 0.0127
loss_sem_align 37.6220
query_points_generation_loss 0.0040
source_choice_loss 0.0400
sp_loss_dice 0.2743
sp_loss_mask 0.0106
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `2400/4054`，但 `source_choice_loss 0.0400` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 20:03 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 20:03:10 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- child workers: `528082`, `528094`
- worker 状态：`Sl`，elapsed `1-01:09:55`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 04:02:32 +0800`
- log size: `195328` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][2600/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout 已至少推进到 `2628/4054`）。

最新稳定 train print：

```text
[06/15 04:02:32] root INFO: Train: [13][2600/4054]
adaptive_weight_loss_dice 0.2748
adaptive_weight_loss_mask 0.0105
corresponding_loss_dice 0.2714
corresponding_loss_mask 0.0146
loss 9.1548
loss_bbox 1.8369
loss_ce 16.1001
loss_dice 0.8266
loss_giou 3.3023
loss_mask 0.0127
loss_sem_align 37.7735
query_points_generation_loss 0.0040
source_choice_loss 0.0405
sp_loss_dice 0.2740
sp_loss_mask 0.0106
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `2600/4054`，但 `source_choice_loss 0.0405` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 20:08 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 20:08:17 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- child workers: `528082`, `528094`
- worker 状态：`Rl`，elapsed `1-01:15:02`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 04:07:54 +0800`
- log size: `195774` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][2800/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout 已至少推进到 `2811/4054`）。

最新稳定 train print：

```text
[06/15 04:07:54] root INFO: Train: [13][2800/4054]
adaptive_weight_loss_dice 0.2750
adaptive_weight_loss_mask 0.0105
corresponding_loss_dice 0.2715
corresponding_loss_mask 0.0146
loss 9.1635
loss_bbox 1.8368
loss_ce 16.1369
loss_dice 0.8260
loss_giou 3.3024
loss_mask 0.0127
loss_sem_align 37.8652
query_points_generation_loss 0.0040
source_choice_loss 0.0407
sp_loss_dice 0.2743
sp_loss_mask 0.0106
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `2800/4054`，但 `source_choice_loss 0.0407` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 20:14 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 20:14:30 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- child workers: `528082`, `528094`
- worker 状态：`Rl`，elapsed `1-01:21:15`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 04:13:23 +0800`
- log size: `196220` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][3000/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout 已至少推进到 `3043/4054`）。

最新稳定 train print：

```text
[06/15 04:13:23] root INFO: Train: [13][3000/4054]
adaptive_weight_loss_dice 0.2749
adaptive_weight_loss_mask 0.0105
corresponding_loss_dice 0.2716
corresponding_loss_mask 0.0146
loss 9.1662
loss_bbox 1.8349
loss_ce 16.1424
loss_dice 0.8257
loss_giou 3.3037
loss_mask 0.0127
loss_sem_align 37.9326
query_points_generation_loss 0.0040
source_choice_loss 0.0407
sp_loss_dice 0.2742
sp_loss_mask 0.0106
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `3000/4054`，但 `source_choice_loss 0.0407` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 20:20 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 20:20:02 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- child workers: `528082`, `528094`
- worker 状态：`Rl`，elapsed `1-01:26:47`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 04:18:45 +0800`
- log size: `196666` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][3200/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout 已至少推进到 `3242/4054`）。

最新稳定 train print：

```text
[06/15 04:18:45] root INFO: Train: [13][3200/4054]
adaptive_weight_loss_dice 0.2751
adaptive_weight_loss_mask 0.0104
corresponding_loss_dice 0.2715
corresponding_loss_mask 0.0145
loss 9.1537
loss_bbox 1.8368
loss_ce 16.0923
loss_dice 0.8262
loss_giou 3.3062
loss_mask 0.0126
loss_sem_align 37.7724
query_points_generation_loss 0.0040
source_choice_loss 0.0409
sp_loss_dice 0.2743
sp_loss_mask 0.0106
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `3200/4054`，但 `source_choice_loss 0.0409` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 20:25 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 20:25:36 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- child workers: `528082`, `528094`
- worker 状态：`Sl`，elapsed `1-01:32:23`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 04:24:16 +0800`
- log size: `197112` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][3400/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout 已至少推进到 `3445/4054`）。

最新稳定 train print：

```text
[06/15 04:24:16] root INFO: Train: [13][3400/4054]
adaptive_weight_loss_dice 0.2750
adaptive_weight_loss_mask 0.0104
corresponding_loss_dice 0.2717
corresponding_loss_mask 0.0146
loss 9.1560
loss_bbox 1.8356
loss_ce 16.0893
loss_dice 0.8257
loss_giou 3.3064
loss_mask 0.0127
loss_sem_align 37.8188
query_points_generation_loss 0.0040
source_choice_loss 0.0412
sp_loss_dice 0.2743
sp_loss_mask 0.0106
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `3400/4054`，但 `source_choice_loss 0.0412` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 20:31 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 20:31:13 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- child workers: `528082`, `528094`
- worker 状态：`Rl`，elapsed `1-01:37:58`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 04:29:45 +0800`
- log size: `197558` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][3600/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout 已至少推进到 `3657/4054`）。

最新稳定 train print：

```text
[06/15 04:29:45] root INFO: Train: [13][3600/4054]
adaptive_weight_loss_dice 0.2745
adaptive_weight_loss_mask 0.0104
corresponding_loss_dice 0.2719
corresponding_loss_mask 0.0146
loss 9.1518
loss_bbox 1.8317
loss_ce 16.0710
loss_dice 0.8256
loss_giou 3.3054
loss_mask 0.0127
loss_sem_align 37.8478
query_points_generation_loss 0.0040
source_choice_loss 0.0409
sp_loss_dice 0.2738
sp_loss_mask 0.0106
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `3600/4054`，但 `source_choice_loss 0.0409` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 20:35 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 20:35:27 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- child workers: `528082`, `528094`
- worker 状态：`Rl`，elapsed `1-01:42:36`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 04:35:03 +0800`
- log size: `198004` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][3800/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout 已至少推进到 `3828/4054`）。

最新稳定 train print：

```text
[06/15 04:35:03] root INFO: Train: [13][3800/4054]
adaptive_weight_loss_dice 0.2750
adaptive_weight_loss_mask 0.0104
corresponding_loss_dice 0.2719
corresponding_loss_mask 0.0145
loss 9.1580
loss_bbox 1.8339
loss_ce 16.0921
loss_dice 0.8254
loss_giou 3.3072
loss_mask 0.0127
loss_sem_align 37.8800
query_points_generation_loss 0.0040
source_choice_loss 0.0408
sp_loss_dice 0.2743
sp_loss_mask 0.0106
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `3800/4054`，但 `source_choice_loss 0.0408` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 20:41 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 20:41:15 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- child workers: `528082`, `528094`
- worker 状态：`Rl`，elapsed `1-01:48:16`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 04:40:32 +0800`
- log size: `198450` bytes
- 最新稳定日志位置：epoch 13 train `Train: [13][4000/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout 已至少推进到 `4038/4054`）。

最新稳定 train print：

```text
[06/15 04:40:32] root INFO: Train: [13][4000/4054]
adaptive_weight_loss_dice 0.2752
adaptive_weight_loss_mask 0.0104
corresponding_loss_dice 0.2719
corresponding_loss_mask 0.0146
loss 9.1555
loss_bbox 1.8372
loss_ce 16.0665
loss_dice 0.8257
loss_giou 3.3080
loss_mask 0.0127
loss_sem_align 37.8133
query_points_generation_loss 0.0040
source_choice_loss 0.0409
sp_loss_dice 0.2745
sp_loss_mask 0.0106
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 没有新的完整 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：epoch 13 stable train checkpoint 已推进到 `4000/4054`，但 `source_choice_loss 0.0409` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 20:42 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 20:42:27 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- 原 train child workers `528082`, `528094` 在本次 `ps` 快照中已不再出现
- worker 状态：`Rl`，elapsed `1-01:49:12`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 04:42:02 +0800`
- log size: `198545` bytes
- 最新稳定日志位置：epoch 13 train 已完成，稳定日志写入 `epoch 13, total time 6623.17, lr_base 0.00020, lr_pointnet 0.00200`
- 最近稳定 train print 仍是 `Train: [13][4000/4054]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout 已进入 eval 进度，至少到 `25/793`）。

最新稳定日志片段：

```text
[06/15 04:40:32] root INFO: Train: [13][4000/4054]
[06/15 04:40:32] root INFO: adaptive_weight_loss_dice 0.2752  adaptive_weight_loss_mask 0.0104  corresponding_loss_dice 0.2719  corresponding_loss_mask 0.0146  loss 9.1555  loss_bbox 1.8372  loss_ce 16.0665  loss_dice 0.8257  loss_giou 3.3080  loss_mask 0.0127  loss_sem_align 37.8133  query_points_generation_loss 0.0040  source_choice_loss 0.0409  sp_loss_dice 0.2745  sp_loss_mask 0.0106
[06/15 04:42:02] root INFO: epoch 13, total time 6623.17, lr_base 0.00020, lr_pointnet 0.00200
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- 还没有新的完整 epoch 13 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：稳定日志已证明 epoch 13 train 完成并进入后续评估阶段，但尚无新的完整 selector diagnostics；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 20:46 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 20:46:40 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- 原 train child workers `528082`, `528094` 在本次 `ps` 快照中仍未出现
- worker 状态：`Rl`，elapsed `1-01:53:40`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 04:45:47 +0800`
- log size: `198986` bytes
- 最新稳定日志位置：epoch 13 eval `Eval: [200/793]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout eval 已至少推进到 `258/793`）。

最新稳定 eval print：

```text
[06/15 04:45:47] root INFO: Eval: [200/793]
adaptive_weight_loss_dice 0.2546
adaptive_weight_loss_mask 0.0123
corresponding_loss_dice 0.3254
corresponding_loss_mask 0.0241
loss 15.9216
loss_bbox 1.5207
loss_ce 43.0490
loss_dice 0.5503
loss_giou 2.9242
loss_mask 0.0254
loss_sem_align 113.4029
query_points_generation_loss 0.0012
source_choice_loss 0.0375
sp_loss_dice 0.2539
sp_loss_mask 0.0124
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- epoch 13 eval 已出现首个稳定进度点，但还没有新的完整 epoch 13 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：稳定日志已证明 epoch 13 eval 推进到 `200/793`，但 `source_choice_loss 0.0375` 仍只是 eval loss 快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 20:50 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 20:50:34 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- 原 train child workers `528082`, `528094` 在本次 `ps` 快照中仍未出现
- worker 状态：`Rl`，elapsed `1-01:57:34`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 04:49:43 +0800`
- log size: `199427` bytes
- 最新稳定日志位置：epoch 13 eval `Eval: [400/793]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout eval 已至少推进到 `462/793`）。

最新稳定 eval print：

```text
[06/15 04:49:43] root INFO: Eval: [400/793]
adaptive_weight_loss_dice 0.2525
adaptive_weight_loss_mask 0.0123
corresponding_loss_dice 0.3207
corresponding_loss_mask 0.0234
loss 15.6828
loss_bbox 1.4173
loss_ce 42.9212
loss_dice 0.5392
loss_giou 2.8254
loss_mask 0.0255
loss_sem_align 112.0755
query_points_generation_loss 0.0013
source_choice_loss 0.0366
sp_loss_dice 0.2518
sp_loss_mask 0.0123
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- epoch 13 eval 已推进到第二个稳定进度点，但还没有新的完整 epoch 13 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：稳定日志已证明 epoch 13 eval 推进到 `400/793`，但 `source_choice_loss 0.0366` 仍只是 eval loss 快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 20:54 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 20:54:36 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- 原 train child workers `528082`, `528094` 在本次 `ps` 快照中仍未出现
- worker 状态：`Rl`，elapsed `1-02:01:37`，CPU `114%`，MEM `3.3%`
- log mtime: `2026-06-15 04:53:25 +0800`
- log size: `199868` bytes
- 最新稳定日志位置：epoch 13 eval `Eval: [600/793]`
- stdout 仅作 liveness 参考；本次进度仍以稳定日志为准（stdout eval 已至少推进到 `670/793`）。

最新稳定 eval print：

```text
[06/15 04:53:25] root INFO: Eval: [600/793]
adaptive_weight_loss_dice 0.2424
adaptive_weight_loss_mask 0.0119
corresponding_loss_dice 0.3279
corresponding_loss_mask 0.0241
loss 15.5643
loss_bbox 1.3688
loss_ce 42.0104
loss_dice 0.5460
loss_giou 2.7696
loss_mask 0.0250
loss_sem_align 111.9626
query_points_generation_loss 0.0013
source_choice_loss 0.0384
sp_loss_dice 0.2419
sp_loss_mask 0.0120
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- epoch 13 eval 已推进到第三个稳定进度点，但还没有新的完整 epoch 13 source-choice diagnostics；最新完整 diagnostics 仍是 epoch 12 的 `[06/15 02:51:39]`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。
- epoch 12 learned selector 仍与 fixed default 完全持平；oracle headroom 仍为 Acc@0.25 `+0.03660`、Acc@0.50 `+0.02125`。

当前判断继续不变：稳定日志已证明 epoch 13 eval 推进到 `600/793`，但 `source_choice_loss 0.0384` 仍只是 eval loss 快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 20:58 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 20:58:36 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- 原 train child workers `528082`, `528094` 在本次 `ps` 快照中仍未出现
- worker 状态：`Rl`，elapsed `1-02:05:22`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 04:57:04.478181839 +0800`
- log size: `205142` bytes
- 最新稳定日志位置：epoch 13 eval 已完成，并在 `[06/15 04:57:04]` 写入完整 alignment、analysis 与 source-choice diagnostics
- stdout 仅作 liveness 参考；stdout 进度条已显示新一轮 train 约 `54/4054`，但稳定日志尚未写入 `Train: [14]`，因此不把 epoch 14 记为稳定进度。

epoch 13 最新稳定 eval diagnostics：

```text
[06/15 04:57:04] root INFO: last_ position alignment Acc0.25: Top-1: 0.51062, Top-5: 0.66470, Top-10: 0.72854
[06/15 04:57:04] root INFO: last_ position alignment Acc0.50: Top-1: 0.37390, Top-5: 0.53891, Top-10: 0.60307
[06/15 04:57:04] root INFO: last_ semantic alignment Acc0.25: Top-1: 0.51062, Top-5: 0.66565, Top-10: 0.73202
[06/15 04:57:04] root INFO: last_ semantic alignment Acc0.50: Top-1: 0.37358, Top-5: 0.53944, Top-10: 0.60402
```

epoch 13 analysis 摘要：

```text
iou@0.25: easy 0.7402078337330136, hard 0.4286326006280331, vd 0.47950153512732524, vid 0.554016620498615, unique 0.8414376321353065, multi 0.45258993695141553
iou@0.50: easy50 0.5651478816946442, hard50 0.30516699971453043, vd50 0.3532598880260069, vid50 0.4019138755980861, unique50 0.6575052854122622, multi50 0.32377302509580913
mask@mean iou: mask_pos 0.3635837574908506, mask_sem 0.3638826236894052
mask@kiou: unique25 0.8682170542635659, unique50 0.6835799859055673, multi25 0.4759550006181234, multi50 0.374953640746693, overall25 0.5344972654606647, overall50 0.42101388304585613
mask@identity: vd25 0.4973812533863103, vd50 0.39281199205345857, vid25 0.5862503147821707, vid50 0.46033744648703095, easy25 0.7569944044764189, easy50 0.5987210231814548, hard25 0.4550385383956609, hard50 0.3575506708535541
```

epoch 13 source-choice diagnostics：

```text
[06/15 04:57:04] root INFO: fixed_default Acc0.25 Top-1: 0.51062, Acc0.50 Top-1: 0.37390
[06/15 04:57:04] root INFO: fixed_mask_text Acc0.25 Top-1: 0.30111, Acc0.50 Top-1: 0.16870
[06/15 04:57:04] root INFO: learned_selector Acc0.25 Top-1: 0.51062, Acc0.50 Top-1: 0.37390
[06/15 04:57:04] root INFO: oracle Acc0.25 Top-1: 0.54870, Acc0.50 Top-1: 0.40082
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- epoch 13 learned selector 仍与 fixed default 完全持平：Acc@0.25 `0.51062`、Acc@0.50 `0.37390`。
- epoch 13 oracle 仍高于 fixed default：Acc@0.25 headroom `+0.03808`，Acc@0.50 headroom `+0.02692`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。

当前判断继续不变：稳定日志已证明 epoch 13 eval 完成并产出完整 source-choice diagnostics，但 learned selector 没有超过 fixed default，且当前旧进程仍无法提供 ratio/false-override 诊断；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-14 21:03 UTC 追加跟踪

长训仍在运行，未手动中断（核查时间 `2026-06-14 21:03:02 UTC`）：

- main PID: `56614`
- launcher PID: `56618`
- python worker: `56664`
- 原 train child workers `528082`, `528094` 在本次 `ps` 快照中仍未出现
- worker 状态：`Sl`，elapsed `1-02:09:48`，CPU `114%`，MEM `3.2%`
- log mtime: `2026-06-15 05:02:35.229155274 +0800`
- log size: `205587` bytes
- 最新稳定日志位置：epoch 14 train `Train: [14][200/4054]`
- stdout 仅作 liveness 参考；stdout 进度条已显示约 `212/4054`，但本次记录以稳定日志 `200/4054` 为准。

最新稳定 train print：

```text
[06/15 05:02:35] root INFO: Train: [14][200/4054]
adaptive_weight_loss_dice 0.2761
adaptive_weight_loss_mask 0.0103
corresponding_loss_dice 0.2669
corresponding_loss_mask 0.0142
loss 9.0663
loss_bbox 1.8453
loss_ce 15.6407
loss_dice 0.8255
loss_giou 3.3323
loss_mask 0.0124
loss_sem_align 37.1475
query_points_generation_loss 0.0038
source_choice_loss 0.0385
sp_loss_dice 0.2754
sp_loss_mask 0.0104
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`。
- epoch 14 已出现首个稳定 train checkpoint，但还没有新的 epoch 14 eval/source-choice diagnostics；最新完整 diagnostics 仍是 epoch 13 的 `[06/15 04:57:04]`。
- epoch 13 learned selector 仍与 fixed default 完全持平：Acc@0.25 `0.51062`、Acc@0.50 `0.37390`。
- epoch 13 oracle 仍高于 fixed default：Acc@0.25 headroom `+0.03808`，Acc@0.50 headroom `+0.02692`。
- 仍未出现新的 `[source_choice]` ratio 行；当前长训进程是在诊断补丁前启动的旧进程，因此缺少 ratio/false-override 诊断属于预期。

当前判断继续不变：稳定日志已证明 epoch 14 train 推进到 `200/4054`，但 `source_choice_loss 0.0385` 只是训练损失快照，不是 selector 生效的验收证据；阶段 4/5 gate 仍不放行，仍必须由后续新启动、带 ratio/false-override 诊断的 smoke/short run 验收。

### 2026-06-19 17:00 UTC 追加跟踪

按阶段 4 要求重启了带 `[source_choice]` ratio / false-override 诊断的新 1 epoch smoke。目标仍是验证 selector 诊断、loss 稳定性和 eval 四组 source-choice 输出；若 OOM，遵循阶段 5 策略优先降 batch，不改变 source-choice 方法。

已完成的单元测试复验：

```bash
cd /home/gb/new\ butd/butd_detr-main/MCLN-main
conda run -n bdetr python -m pytest -q \
  tests/test_source_choice_selector.py \
  tests/test_source_choice_adapter.py \
  tests/test_main_utils_source_choice_checkpoint.py \
  tests/test_grounding_evaluator_source_choice.py
# 8 passed in 5.21s
```

batch 12 smoke：

- run: `MCLN_source_choice_ratio_smoke_seed0_20260619_163015`
- stdout: `/root/autodl-tmp/DATA_ROOT/output/run_control/MCLN_source_choice_ratio_smoke_seed0_20260619_163015.stdout.log`
- log: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_ratio_smoke_seed0_20260619_163015/1781886624/log.txt`
- 已出现 `[source_choice]` 诊断行：

```text
source_choice_false_override_ratio 0.0050
source_choice_selected_non_default_ratio 0.0050
source_choice_target_acc 0.9917
source_choice_target_non_default_ratio 0.0033
```

- 结果：失败，未完成 epoch 1 / eval。train step 237 OOM：

```text
RuntimeError: CUDA out of memory. Tried to allocate 74.71 GiB
models/losses.py:560, loss_masks:
distances = torch.norm(selected_xyz - selected_xyz.permute(0, 2, 1, 3), dim=3)
```

batch 6 smoke：

- run: `MCLN_source_choice_ratio_smoke_b6_seed0_20260619_164518`
- stdout: `/root/autodl-tmp/DATA_ROOT/output/run_control/MCLN_source_choice_ratio_smoke_b6_seed0_20260619_164518.stdout.log`
- log: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_ratio_smoke_b6_seed0_20260619_164518/1781887522/log.txt`
- config highlights: `batch_size=6`, `selector_train_only=True`, sources `default,mask_text`, target `precision_gain_default_sourcewise_focal_bce`, `skip_missing_superpoints=True`
- step 200 `[source_choice]`：

```text
source_choice_false_override_ratio 0.0000
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9983
source_choice_target_non_default_ratio 0.0017
```

- step 400 `[source_choice]`：

```text
source_choice_false_override_ratio 0.0000
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9992
source_choice_target_non_default_ratio 0.0008
```

- 结果：失败，未完成 epoch 1 / eval。train step 463 OOM，根因仍是 `models/losses.py:560` 的 `loss_masks` selected-point pairwise distance matrix：

```text
RuntimeError: CUDA out of memory. Tried to allocate 7.22 GiB
GPU 0; 39.38 GiB total capacity; 29.86 GiB already allocated; 3.97 GiB free
```

batch 3 smoke 已按同一方法启动：

- run: `MCLN_source_choice_ratio_smoke_b3_seed0_20260619_170024`
- wrapper PID: `25126`
- launcher PID: `25127`
- worker PID: `25159`
- pid file: `/root/autodl-tmp/DATA_ROOT/output/run_control/MCLN_source_choice_ratio_smoke_b3_seed0_20260619_170024.pid`
- stdout: `/root/autodl-tmp/DATA_ROOT/output/run_control/MCLN_source_choice_ratio_smoke_b3_seed0_20260619_170024.stdout.log`
- log: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_ratio_smoke_b3_seed0_20260619_170024/1781888428/log.txt`
- config: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_ratio_smoke_b3_seed0_20260619_170024/1781888428/config.json`
- config 已确认 `batch_size=3`，`use_source_choice_selector=True`，`source_choice_selector_train_only=True`，`eval_use_selector_choice_scores=True`
- 当前状态：已完成数据加载并进入训练；截至本次核查，stdout 进度已超过 `436/16218`，没有 OOM。
- 数据集长度：train `48655`，test `9508`。
- 注意：`batch_size=3` 后每 epoch train step 变为 `16218`，完整 1 epoch smoke 会比 batch 6 更久。

batch 3 已出现的 `[source_choice]` 诊断：

```text
[06/20 01:10:15] Train: [1][200/16218]
source_choice_false_override_ratio 0.0033
source_choice_selected_non_default_ratio 0.0033
source_choice_target_acc 0.9950
source_choice_target_non_default_ratio 0.0017

[06/20 01:11:42] Train: [1][400/16218]
source_choice_false_override_ratio 0.0017
source_choice_selected_non_default_ratio 0.0017
source_choice_target_acc 0.9967
source_choice_target_non_default_ratio 0.0017

[06/20 01:13:10] Train: [1][600/16218]
source_choice_false_override_ratio 0.0011
source_choice_selected_non_default_ratio 0.0011
source_choice_target_acc 0.9961
source_choice_target_non_default_ratio 0.0028

[06/20 01:14:37] Train: [1][800/16218]
source_choice_false_override_ratio 0.0008
source_choice_selected_non_default_ratio 0.0008
source_choice_target_acc 0.9958
source_choice_target_non_default_ratio 0.0033

[06/20 01:16:07] Train: [1][1000/16218]
source_choice_false_override_ratio 0.0007
source_choice_selected_non_default_ratio 0.0007
source_choice_target_acc 0.9963
source_choice_target_non_default_ratio 0.0030
```

当前 gate 判断：

- 阶段 4 仍未通过，因为 batch 12 / batch 6 都未完成 eval，尚无 fixed default、fixed mask_text、learned selector、oracle 四组完整输出。
- ratio logging 本身已被新进程验证可用，且 batch 3 已越过 batch 6 的 OOM step 463 并稳定到 step 800；但 batch 3 仍需要继续观察到至少完成 epoch 1 + eval，才能作为 smoke gate 证据。
- 若 batch 3 仍因同一 `loss_masks` distance matrix OOM，下一步按计划继续降到 batch 2 或 batch 1；在重复 OOM 得到充分证据前，不改 source-choice 方法。

### 2026-06-20 00:15 UTC / 08:15 local 追加跟踪

先回答当前文档中记录的 MLCN 最好指标：

- 当前最高可部署 / 正常 eval 口径：epoch 13 `fixed_default` / `learned_selector`，Acc@0.25 `0.51062`，Acc@0.50 `0.37390`。
- 当前最高诊断上限：epoch 13 `oracle`，Acc@0.25 `0.54870`，Acc@0.50 `0.40082`。该值只作 source-choice upper bound，不是可部署指标。
- epoch 13 `last_ position alignment` 同为 Acc@0.25 `0.51062`，Acc@0.50 `0.37390`；`last_ semantic alignment` 为 Acc@0.25 `0.51062`，Acc@0.50 `0.37358`。

磁盘清理：

- 启动新实验前检查 `/root/autodl-tmp`：`45G/50G` 已用，剩余 `5.5G`，使用率 `90%`。
- 已删除 `/root/autodl-tmp/log_backups` 下 19 个 2026-01 旧 `.pth` checkpoint 备份，合计约 `14.15 GiB`；保留日志文本、配置和当前 MCLN output 目录。
- 清理后 `/root/autodl-tmp` 为 `31G/50G` 已用，剩余 `20G`，使用率 `61%`。

batch 3 smoke 最终状态：

- run: `MCLN_source_choice_ratio_smoke_b3_seed0_20260619_170024`
- stdout: `/root/autodl-tmp/DATA_ROOT/output/run_control/MCLN_source_choice_ratio_smoke_b3_seed0_20260619_170024.stdout.log`
- log: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_ratio_smoke_b3_seed0_20260619_170024/1781888428/log.txt`
- 稳定日志推进到 `Train: [1][2000/16218]`，`source_choice_loss 0.0051`，无 NaN。
- step 2000 `[source_choice]`：

```text
source_choice_false_override_ratio 0.0003
source_choice_selected_non_default_ratio 0.0003
source_choice_target_acc 0.9972
source_choice_target_non_default_ratio 0.0025
```

- 结果：失败，未完成 epoch 1 / eval。stdout 进度在 `2012/16218` 处 OOM，根因仍是 `models/losses.py:560` 的 `loss_masks` selected-point pairwise distance matrix：

```text
RuntimeError: CUDA out of memory. Tried to allocate 30.05 GiB
models/losses.py:560, loss_masks:
distances = torch.norm(selected_xyz - selected_xyz.permute(0, 2, 1, 3), dim=3)
```

当前判断：

- ratio / false-override logging 已在 batch 12、6、3 中验证可用。
- batch 3 仍因同一非 source-choice 的 `loss_masks` pairwise distance matrix OOM，阶段 4 仍未通过；尚无本轮新进程的 fixed default、fixed mask_text、learned selector、oracle 四组 eval 输出。
- 下一步按计划继续降到 batch 1，不改变 source-choice 方法；若 batch 1 仍在同一位置 OOM，则 batch-size fallback 基本耗尽，后续需要专门处理 smoke 中 `loss_masks` 的显存峰值，同时保持 source-choice 协议不变。

### 2026-06-20 00:45 UTC / 08:45 local 追加跟踪

batch 1 smoke 已按同一 source-choice 方法启动，并确认能越过前两次 OOM 区间：

- run: `MCLN_source_choice_ratio_smoke_b1_seed0_20260620_002129`
- wrapper PID: `92736`
- launcher PID: `92742`
- worker PID: `92764`
- stdout: `/root/autodl-tmp/DATA_ROOT/output/run_control/MCLN_source_choice_ratio_smoke_b1_seed0_20260620_002129.stdout.log`
- log: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_ratio_smoke_b1_seed0_20260620_002129/1781914892/log.txt`
- config: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_ratio_smoke_b1_seed0_20260620_002129/1781914892/config.json`
- config 已确认：`batch_size=1`，`use_source_choice_selector=True`，`source_choice_selector_train_only=True`，`eval_use_selector_choice_scores=True`，sources `default,mask_text`，target `precision_gain_default_sourcewise_focal_bce`，`skip_missing_superpoints=True`。
- 数据集长度：train `48655`，test `9508`。
- 进入训练时 stdout 确认 selector-only：`source_choice_selector_train_only: trainable selector parameters 104353`。

稳定日志已推进到 `Train: [1][2200/48655]`，这已经越过：

- batch 6 的 OOM step `463`
- batch 3 的 OOM step `2012`

已记录的 batch 1 `[source_choice]` 诊断：

```text
[06/20 08:30:51] Train: [1][200/48655]
source_choice_false_override_ratio 0.0050
source_choice_selected_non_default_ratio 0.0050
source_choice_target_acc 0.9800
source_choice_target_non_default_ratio 0.0150

[06/20 08:31:52] Train: [1][400/48655]
source_choice_false_override_ratio 0.0025
source_choice_selected_non_default_ratio 0.0025
source_choice_target_acc 0.9850
source_choice_target_non_default_ratio 0.0125

[06/20 08:32:53] Train: [1][600/48655]
source_choice_false_override_ratio 0.0017
source_choice_selected_non_default_ratio 0.0017
source_choice_target_acc 0.9883
source_choice_target_non_default_ratio 0.0100

[06/20 08:33:51] Train: [1][800/48655]
source_choice_false_override_ratio 0.0013
source_choice_selected_non_default_ratio 0.0013
source_choice_target_acc 0.9900
source_choice_target_non_default_ratio 0.0088

[06/20 08:34:52] Train: [1][1000/48655]
source_choice_false_override_ratio 0.0010
source_choice_selected_non_default_ratio 0.0010
source_choice_target_acc 0.9900
source_choice_target_non_default_ratio 0.0090

[06/20 08:35:50] Train: [1][1200/48655]
source_choice_false_override_ratio 0.0008
source_choice_selected_non_default_ratio 0.0008
source_choice_target_acc 0.9908
source_choice_target_non_default_ratio 0.0083

[06/20 08:36:51] Train: [1][1400/48655]
source_choice_false_override_ratio 0.0007
source_choice_selected_non_default_ratio 0.0007
source_choice_target_acc 0.9907
source_choice_target_non_default_ratio 0.0086

[06/20 08:37:52] Train: [1][1600/48655]
source_choice_false_override_ratio 0.0006
source_choice_selected_non_default_ratio 0.0006
source_choice_target_acc 0.9906
source_choice_target_non_default_ratio 0.0088

[06/20 08:38:50] Train: [1][1800/48655]
source_choice_false_override_ratio 0.0006
source_choice_selected_non_default_ratio 0.0006
source_choice_target_acc 0.9917
source_choice_target_non_default_ratio 0.0078

[06/20 08:39:51] Train: [1][2000/48655]
source_choice_false_override_ratio 0.0005
source_choice_selected_non_default_ratio 0.0005
source_choice_target_acc 0.9915
source_choice_target_non_default_ratio 0.0080

[06/20 08:40:53] Train: [1][2200/48655]
source_choice_false_override_ratio 0.0005
source_choice_selected_non_default_ratio 0.0005
source_choice_target_acc 0.9918
source_choice_target_non_default_ratio 0.0077
```

资源状态：

- GPU memory around step 2200: about `19.4 GiB / 40 GiB` used.
- `/root/autodl-tmp`: still about `31G/50G` used, `20G` available, `61%` usage.

当前判断：

- batch 1 已证明 batch-size fallback 可以避开目前观测到的 `loss_masks` early OOM 区间。
- 阶段 4 仍未通过，因为 batch 1 尚未完成 epoch 1 和 eval，仍缺 fixed default、fixed mask_text、learned selector、oracle 四组完整输出。
- 继续让 batch 1 跑到 epoch 1 eval；如果完整 epoch 时间过长，可以后续考虑专门加一个 smoke-only step cap / eval-only 验证，但这会是新的工程决策，当前 run 仍保持原方法不变。

### 2026-06-20 01:25 UTC / 09:25 local 追加跟踪

batch 1 smoke 继续运行，仍未出现失败信号：

- wrapper PID: `92736`
- launcher PID: `92742`
- worker PID: `92764`
- worker 状态：`Rl`，elapsed `01:04:16`，CPU `110%`，MEM `2.8%`
- 稳定日志最新进度：`Train: [1][11200/48655]`
- stdout 进度条已超过 `11199/48655`，但本次记录以稳定日志 `11200/48655` 为准。
- GPU memory: `19397 MiB / 40960 MiB`
- `/root/autodl-tmp`: `31G/50G` used, `20G` available, `61%`
- `/`: `16G/30G` used, `15G` available, `51%`

最新稳定 `[source_choice]` 诊断：

```text
[06/20 09:24:38] Train: [1][11000/48655]
source_choice_default_acc025 0.0002
source_choice_false_override_ratio 0.0001
source_choice_oracle_acc025 0.0004
source_choice_selected_non_default_ratio 0.0001
source_choice_target_acc 0.9935
source_choice_target_non_default_ratio 0.0065

[06/20 09:25:36] Train: [1][11200/48655]
source_choice_default_acc025 0.0002
source_choice_false_override_ratio 0.0001
source_choice_oracle_acc025 0.0004
source_choice_selected_non_default_ratio 0.0001
source_choice_target_acc 0.9935
source_choice_target_non_default_ratio 0.0064
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`、`Killed`。
- batch 1 已越过 batch 6 OOM step `463` 和 batch 3 OOM step `2012` 很远，说明单纯降 batch 已经明显缓解当前 `loss_masks` 显存峰值。
- 仍未进入 epoch 1 eval，尚无 `fixed_default`、`fixed_mask_text`、`learned_selector`、`oracle` 四组完整输出。

当前判断：

- 阶段 4 仍未通过；当前证据只能说明 batch 1 训练段稳定推进、ratio diagnostics 正常记录、磁盘空间暂时安全。
- 继续监控到 epoch 1 完成并进入 eval；只有拿到四组 source-choice eval 输出后，才能决定是否进入阶段 5 short run。

### 2026-06-20 01:38 UTC / 09:38 local 追加跟踪

batch 1 smoke 继续运行，10 分钟监控窗口内从 `Train: [1][11600/48655]` 推进到稳定日志 `Train: [1][13800/48655]`：

- wrapper PID: `92736`
- launcher PID: `92742`
- worker PID: `92764`
- worker 状态：`Rl`，elapsed `01:17:13`，CPU `111%`，MEM `2.8%`
- GPU memory: `19399 MiB / 40960 MiB`
- `/root/autodl-tmp`: `31G/50G` used, `20G` available, `61%`
- `/`: `16G/30G` used, `15G` available, `51%`

最新稳定 `[source_choice]` 诊断：

```text
[06/20 09:37:35] Train: [1][13600/48655]
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0001
source_choice_oracle_acc025 0.0004
source_choice_selected_non_default_ratio 0.0001
source_choice_target_acc 0.9935
source_choice_target_non_default_ratio 0.0064

[06/20 09:38:33] Train: [1][13800/48655]
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0001
source_choice_oracle_acc025 0.0004
source_choice_selected_non_default_ratio 0.0001
source_choice_target_acc 0.9936
source_choice_target_non_default_ratio 0.0063
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`、`Killed`。
- 仍未进入 epoch 1 eval，尚无 `fixed_default`、`fixed_mask_text`、`learned_selector`、`oracle` 四组完整输出。
- 当前速度下完整 batch 1 epoch 仍需数小时；继续长间隔监控，避免用训练中间点替代阶段 4 smoke gate。

当前判断：

- batch 1 训练段已稳定超过 `28%`，但阶段 4 仍未通过。
- 磁盘暂时安全，不需要继续清理；后续若 `/root/autodl-tmp` 回到高水位，再优先清理旧 checkpoint backup，不删当前 run 和数据集。

### 2026-06-20 01:56 UTC / 09:56 local 追加跟踪

batch 1 smoke 继续运行，稳定日志已从上一记录的 `Train: [1][13800/48655]` 推进到 `Train: [1][17200/48655]`：

- wrapper PID: `92736`
- launcher PID: `92742`
- worker PID: `92764`
- worker 状态：`Rl`，elapsed `01:34:30`，CPU `111%`，MEM `2.8%`
- GPU memory: `19399 MiB / 40960 MiB`
- `/root/autodl-tmp`: `31G/50G` used, `20G` available, `61%`
- `/`: `16G/30G` used, `15G` available, `51%`

最新稳定 `[source_choice]` 诊断：

```text
[06/20 09:54:33] Train: [1][17000/48655]
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0001
source_choice_oracle_acc025 0.0004
source_choice_selected_non_default_ratio 0.0001
source_choice_target_acc 0.9939
source_choice_target_non_default_ratio 0.0060

[06/20 09:55:33] Train: [1][17200/48655]
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0001
source_choice_oracle_acc025 0.0003
source_choice_selected_non_default_ratio 0.0001
source_choice_target_acc 0.9940
source_choice_target_non_default_ratio 0.0059
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`、`Killed`。
- 仍未进入 epoch 1 eval，尚无 `fixed_default`、`fixed_mask_text`、`learned_selector`、`oracle` 四组完整输出。

当前判断：

- batch 1 训练段已稳定超过 `35%`，ratio diagnostics 继续正常记录，但阶段 4 仍未通过。
- 继续长间隔监控到 epoch 1 完成；只有 eval 四组 source-choice 输出落盘后，才能判断是否进入阶段 5 short run。

### 2026-06-20 02:10 UTC / 10:10 local 追加跟踪

batch 1 smoke 继续运行，稳定日志已推进到 `Train: [1][20000/48655]`，并继续到 `Train: [1][20200/48655]`：

- wrapper PID: `92736`
- launcher PID: `92742`
- worker PID: `92764`
- worker 状态：`Rl`，elapsed `01:49:52`，CPU `111%`，MEM `2.8%`
- GPU memory: `19399 MiB / 40960 MiB`
- `/root/autodl-tmp`: `31G/50G` used, `20G` available, `61%`
- `/`: `16G/30G` used, `15G` available, `51%`

最新稳定训练与 `[source_choice]` 诊断：

```text
[06/20 10:09:28] Train: [1][20000/48655]
loss 116.2425
source_choice_loss 0.0120
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0001
source_choice_oracle_acc025 0.0003
source_choice_selected_non_default_ratio 0.0001
source_choice_target_acc 0.9938
source_choice_target_non_default_ratio 0.0062

[06/20 10:10:26] Train: [1][20200/48655]
loss 116.1994
source_choice_loss 0.0120
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0003
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9938
source_choice_target_non_default_ratio 0.0062
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`、`Killed`。
- 仍未进入 epoch 1 eval，尚无 `fixed_default`、`fixed_mask_text`、`learned_selector`、`oracle` 四组完整输出。

当前判断：

- batch 1 训练段已稳定超过 `41%`，且 `source_choice_loss` 与 ratio diagnostics 仍为有限值。
- 阶段 4 仍未通过；继续让当前 batch 1 run 跑到 epoch 1 eval。
- 磁盘仍安全，暂不清理；保持不删除当前 run artifacts 和数据集。

### 2026-06-20 02:37 UTC / 10:37 local 追加跟踪

batch 1 smoke 继续运行，稳定日志已从 `Train: [1][20200/48655]` 推进到 `Train: [1][25600/48655]`：

- wrapper PID: `92736`
- launcher PID: `92742`
- worker PID: `92764`
- worker 状态：`Rl`，elapsed `02:16:21`，CPU `111%`，MEM `2.8%`
- GPU memory: `19399 MiB / 40960 MiB`
- `/root/autodl-tmp`: `31G/50G` used, `20G` available, `61%`
- `/`: `16G/30G` used, `15G` available, `51%`

最新稳定训练与 `[source_choice]` 诊断：

```text
[06/20 10:34:15] Train: [1][25000/48655]
loss 116.0666
source_choice_loss 0.0116
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0003
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9940
source_choice_target_non_default_ratio 0.0059

[06/20 10:37:12] Train: [1][25600/48655]
loss 116.0396
source_choice_loss 0.0116
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0003
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9941
source_choice_target_non_default_ratio 0.0059
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`、`Killed`。
- 仍未进入 epoch 1 eval，尚无 `fixed_default`、`fixed_mask_text`、`learned_selector`、`oracle` 四组完整输出。

当前判断：

- batch 1 训练段已稳定超过 `52%`，ratio diagnostics 持续正常。
- 阶段 4 仍未通过；继续让当前 batch 1 run 跑到 epoch 1 eval。
- 磁盘仍安全，暂不清理；保持不删除当前 run artifacts 和数据集。

### 2026-06-20 03:05 UTC / 11:05 local 追加跟踪

batch 1 smoke 继续运行，稳定日志已从 `Train: [1][25600/48655]` 推进到 `Train: [1][31000/48655]`：

- wrapper PID: `92736`
- launcher PID: `92742`
- worker PID: `92764`
- worker 状态：`Sl`，elapsed `02:43:34`，CPU `111%`，MEM `2.8%`
- GPU memory: `19399 MiB / 40960 MiB`
- `/root/autodl-tmp`: `31G/50G` used, `20G` available, `61%`
- `/`: `16G/30G` used, `15G` available, `51%`

最新稳定训练与 `[source_choice]` 诊断：

```text
[06/20 10:59:01] Train: [1][30000/48655]
loss 116.0302
source_choice_loss 0.0119
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0003
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9939
source_choice_target_non_default_ratio 0.0061

[06/20 11:01:02] Train: [1][30400/48655]
loss 116.0616
source_choice_loss 0.0118
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0003
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9939
source_choice_target_non_default_ratio 0.0060

[06/20 11:04:00] Train: [1][31000/48655]
loss 116.0592
source_choice_loss 0.0117
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0003
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9940
source_choice_target_non_default_ratio 0.0060
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`、`Killed`。
- 仍未进入 epoch 1 eval，尚无 `fixed_default`、`fixed_mask_text`、`learned_selector`、`oracle` 四组完整输出。

当前判断：

- batch 1 训练段已稳定超过 `63%`，`source_choice_loss` 仍为有限值，ratio diagnostics 持续正常。
- 阶段 4 仍未通过；继续让当前 batch 1 run 跑到 epoch 1 eval。
- 磁盘仍安全，暂不清理；保持不删除当前 run artifacts 和数据集。

### 2026-06-20 03:24 UTC / 11:24 local 追加跟踪

batch 1 smoke 继续运行，稳定日志已从 `Train: [1][31000/48655]` 推进到 `Train: [1][35000/48655]`，并继续到 `Train: [1][35200/48655]`：

- wrapper PID: `92736`
- launcher PID: `92742`
- worker PID: `92764`
- worker 状态：`Rl`，elapsed `03:03:03`，CPU `111%`，MEM `2.8%`
- GPU memory: `19399 MiB / 40960 MiB`
- `/root/autodl-tmp`: `31G/50G` used, `20G` available, `61%`
- `/`: `16G/30G` used, `15G` available, `51%`

最新稳定训练与 `[source_choice]` 诊断：

```text
[06/20 11:18:56] Train: [1][34000/48655]
loss 115.9736
source_choice_loss 0.0116
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0003
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9941
source_choice_target_non_default_ratio 0.0058

[06/20 11:23:53] Train: [1][35000/48655]
loss 115.9674
source_choice_loss 0.0116
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0003
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9942
source_choice_target_non_default_ratio 0.0058

[06/20 11:24:51] Train: [1][35200/48655]
loss 115.9650
source_choice_loss 0.0116
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0003
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9942
source_choice_target_non_default_ratio 0.0058
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`、`Killed`。
- 未出现 `Eval:`，也未出现 `fixed_default`、`fixed_mask_text`、`learned_selector`、`oracle` 四组完整输出。

当前判断：

- batch 1 训练段已稳定超过 `72%`，`source_choice_loss` 与 ratio diagnostics 仍为有限值。
- 阶段 4 仍未通过；继续让当前 batch 1 run 跑到 epoch 1 eval。
- 磁盘仍安全，暂不清理；保持不删除当前 run artifacts 和数据集。

### 2026-06-20 03:50 UTC / 11:50 local 追加跟踪

batch 1 smoke 继续运行，稳定日志已从 `Train: [1][35200/48655]` 推进到 `Train: [1][40000/48655]`，并继续到 `Train: [1][40200/48655]`：

- wrapper PID: `92736`
- launcher PID: `92742`
- worker PID: `92764`
- worker 状态：`Rl`，elapsed `03:28:27`，CPU `111%`，MEM `2.8%`
- GPU memory: `19399 MiB / 40960 MiB`
- `/root/autodl-tmp`: `31G/50G` used, `20G` available, `61%`
- `/`: `16G/30G` used, `15G` available, `51%`

最新稳定训练与 `[source_choice]` 诊断：

```text
[06/20 11:38:48] Train: [1][38000/48655]
loss 115.9339
source_choice_loss 0.0117
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0002
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9942
source_choice_target_non_default_ratio 0.0058

[06/20 11:48:49] Train: [1][40000/48655]
loss 115.9649
source_choice_loss 0.0117
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0002
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9941
source_choice_target_non_default_ratio 0.0059

[06/20 11:49:49] Train: [1][40200/48655]
loss 115.9513
source_choice_loss 0.0117
source_choice_default_acc025 0.0000
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0002
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9941
source_choice_target_non_default_ratio 0.0058
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`、`Killed`。
- 未出现 `Eval:`，也未出现 `fixed_default`、`fixed_mask_text`、`learned_selector`、`oracle` 四组完整输出。

当前判断：

- batch 1 训练段已稳定超过 `82%`，`source_choice_loss` 与 ratio diagnostics 仍为有限值。
- 阶段 4 仍未通过；继续让当前 batch 1 run 跑到 epoch 1 eval。
- 磁盘仍安全，暂不清理；保持不删除当前 run artifacts 和数据集。

### 2026-06-20 04:15 UTC / 12:15 local 追加跟踪

batch 1 smoke 继续运行，稳定日志已从 `Train: [1][40200/48655]` 推进到 `Train: [1][45000/48655]`，并继续到 `Train: [1][45200/48655]`：

- wrapper PID: `92736`
- launcher PID: `92742`
- worker PID: `92764`
- worker 状态：`Rl`，elapsed `03:54:07`，CPU `112%`，MEM `2.8%`
- GPU memory: `19399 MiB / 40960 MiB`
- `/root/autodl-tmp`: `31G/50G` used, `20G` available, `61%`
- `/`: `16G/30G` used, `15G` available, `51%`

最新稳定训练与 `[source_choice]` 诊断：

```text
[06/20 12:08:44] Train: [1][44000/48655]
loss 115.8318
source_choice_loss 0.0117
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0003
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9940
source_choice_target_non_default_ratio 0.0060

[06/20 12:13:41] Train: [1][45000/48655]
loss 115.8386
source_choice_loss 0.0118
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0003
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9940
source_choice_target_non_default_ratio 0.0060

[06/20 12:14:40] Train: [1][45200/48655]
loss 115.8381
source_choice_loss 0.0118
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0003
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9939
source_choice_target_non_default_ratio 0.0061
```

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`、`Killed`。
- 未出现 `Eval:`，也未出现 `fixed_default`、`fixed_mask_text`、`learned_selector`、`oracle` 四组完整输出。

当前判断：

- batch 1 训练段已稳定超过 `92%`，`source_choice_loss` 与 ratio diagnostics 仍为有限值。
- 阶段 4 仍未通过；继续让当前 batch 1 run 跑完 epoch 1 并进入 eval。
- 磁盘仍安全，暂不清理；保持不删除当前 run artifacts 和数据集。

### 2026-06-20 04:33 UTC / 12:33 local 追加跟踪

batch 1 smoke 已完成 epoch 1 训练并进入 test evaluation：

- wrapper PID: `92736`
- launcher PID: `92742`
- worker PID: `92764`
- log: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_ratio_smoke_b1_seed0_20260620_002129/1781914892/log.txt`
- stdout: `/root/autodl-tmp/DATA_ROOT/output/run_control/MCLN_source_choice_ratio_smoke_b1_seed0_20260620_002129.stdout.log`
- checkpoint: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_ratio_smoke_b1_seed0_20260620_002129/1781914892/ckpt_epoch_1.pth` (`573M`)
- GPU memory: `19399 MiB / 40960 MiB`
- `/root/autodl-tmp`: `31G/50G` used, `20G` available, `62%`
- `/`: `16G/30G` used, `15G` available, `53%`

epoch 1 训练完成证据：

```text
[06/20 12:31:32] Train: [1][48600/48655]
loss 115.8309
source_choice_loss 0.0117
source_choice_default_acc025 0.0001
source_choice_false_override_ratio 0.0000
source_choice_oracle_acc025 0.0003
source_choice_selected_non_default_ratio 0.0000
source_choice_target_acc 0.9941
source_choice_target_non_default_ratio 0.0059

[06/20 12:31:48] epoch 1, total time 14519.20, lr_base 0.00100, lr_pointnet 0.00200
Saved in /root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_ratio_smoke_b1_seed0_20260620_002129/1781914892/ckpt_epoch_1.pth
Test evaluation.......
```

eval 当前状态：

- stdout 已进入 `Test evaluation.......`。
- eval 进度最近约 `300/9508`，速度约 `3.4-3.7 it/s`，预计还需约 40-45 分钟完成。
- 稳定 `log.txt` 尚未落盘 eval metrics；当前 eval 进度主要在 stdout 的 tqdm 中。

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`、`Killed`。
- 尚未出现 `fixed_default`、`fixed_mask_text`、`learned_selector`、`oracle` 四组完整输出。

当前判断：

- batch 1 已证明训练段可以完整通过，并且 ratio / false-override diagnostics 正常记录。
- 阶段 4 仍未通过；必须等待 eval 完成并产出 fixed default、fixed mask_text、learned selector、oracle 四组输出后才能判定。
- 磁盘仍安全，暂不清理；保留当前 run artifacts 和数据集。

### 2026-06-20 06:05 UTC / 14:05 local smoke 最终结果

batch 1 smoke 已完成 epoch 1 train + full eval，wrapper 正常退出 `0`：

- run: `MCLN_source_choice_ratio_smoke_b1_seed0_20260620_002129`
- log: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_ratio_smoke_b1_seed0_20260620_002129/1781914892/log.txt`
- stdout: `/root/autodl-tmp/DATA_ROOT/output/run_control/MCLN_source_choice_ratio_smoke_b1_seed0_20260620_002129.stdout.log`
- exit: `[wrapper] exit 0 2026-06-20T14:05:43+08:00`

最终 source-choice eval 输出以 `14:05:36` block 为准：

| source | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
| fixed_default | 0.00000 | 0.00000 |
| fixed_mask_text | 0.00011 | 0.00000 |
| learned_selector | 0.00000 | 0.00000 |
| oracle | 0.00011 | 0.00000 |

同期整体 eval 仍接近全零：

- `last_ position alignment Acc0.25 Top-1: 0.00000`
- `last_ semantic alignment Acc0.25 Top-1: 0.00000`
- `overall25 0.001472444257467396`
- `overall50 0.0`

日志复查：

- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`nan`、`inf`、`Killed`。
- 四组 source-choice 输出完整出现，说明阶段 4 的工程闭环已跑通。
- 指标上 learned selector 没有优于 fixed source；oracle 也只与 `fixed_mask_text` 持平到 `0.00011 / 0.00000`。因此该 smoke 只能证明可运行，不能证明方法有效。

### 2026-06-20 10:49 UTC / 18:49 local selector-only 长训误启动记录

先前尝试的 `MCLN_source_choice_ratio_long_b1_seed0_20260620_103711` 已进入训练并打印首条 `Train: [1][200/48655]`，但命令继承脚本默认 `--save_freq 1`。按 smoke checkpoint 约 `573M` 估算，100 epoch 会产生约 57G checkpoint，超过 `/root/autodl-tmp` 当前可用空间。因此已主动停止该刚开始的进程组，并用省盘保存频率重启。

更正：下面这个 run 不是要延续的 epoch 13 full joint 长训口径，而是 batch-1 selector-only ratio run。用户确认目标是继续文档中 epoch 13 达到 `Acc@0.25 0.51062` 的旧 full joint 长训口径后，该 run 已停止，不再作为当前活跃长训或最好指标来源。

- run: `MCLN_source_choice_ratio_long_b1_save10_seed0_20260620_184918`
- wrapper PID: `240616`
- launcher PID: `240620`
- stdout: `/root/autodl-tmp/DATA_ROOT/output/run_control/MCLN_source_choice_ratio_long_b1_save10_seed0_20260620_184918.stdout.log`
- cmd: `/root/autodl-tmp/DATA_ROOT/output/run_control/MCLN_source_choice_ratio_long_b1_save10_seed0_20260620_184918.cmd.sh`
- pid file: `/root/autodl-tmp/DATA_ROOT/output/run_control/MCLN_source_choice_ratio_long_b1_save10_seed0_20260620_184918.pid`
- log: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_ratio_long_b1_save10_seed0_20260620_184918/1781952583/log.txt`
- config: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_ratio_long_b1_save10_seed0_20260620_184918/1781952583/config.json`

关键配置：

- `max_epoch: 100`
- `batch_size: 1`
- `save_freq: 10`
- `print_freq: 200`
- `skip_missing_superpoints: true`
- `use_source_choice_selector: true`
- `source_choice_selector_train_only: true`
- `source_choice_selector_lr: 0.001`
- `source_choice_selector_loss_weight: 1.0`
- `source_choice_selector_sources: default,mask_text`
- `source_choice_selector_choice_target: precision_gain_default_sourcewise_focal_bce`
- `eval_use_selector_choice_scores: true`
- `rng_seed: 0`
- `MASTER_PORT=4461`
- `CUDA_VISIBLE_DEVICES=0`

最终状态：

- 已停止；后续不再监控该 run。
- 该 run 的 `batch_size=1`、`source_choice_selector_train_only=true`，与旧最佳 full joint 训练配置不一致。
- `/root/autodl-tmp`: `32G/50G` used, `19G` available, `63%`。
- `/`: `16G/30G` used, `15G` available, `54%`。
- 未删除数据集或当前 run artifacts；`ckpt_epoch_1.pth` 与 `ckpt_epoch_last.pth` 字节不相同，暂不当作重复文件清理。

### 2026-06-20 11:03 UTC / 19:03 local checkpoint 复查与长训策略更正

复查结论：

- 当前没有 MCLN 训练进程，A100 空闲。
- 旧最佳 run `MCLN_source_choice_from_scratch_joint_seed0_long/1781376797` 目录只有 `config.json` 和 `log.txt`。
- 扩大搜索 `/root/autodl-tmp` 与 `/home/gb` 后，只发现 `MCLN_source_choice_ratio_smoke_b1_seed0_20260620_002129` 的 `ckpt_epoch_1.pth` 和 `ckpt_epoch_last.pth`，没有旧最佳 run 的 epoch 13/14/15 checkpoint。
- 因此不能从 epoch 13 精确恢复；下一步按旧最佳 full joint 配置重启长训。

重启配置必须保持旧最佳口径：

- `batch_size=12`
- `num_workers=2`
- `max_epoch=100`
- `skip_missing_superpoints=true`
- `use_source_choice_selector=true`
- `source_choice_selector_train_only=false`，即不要传 `--source_choice_selector_train_only`
- `source_choice_selector_sources=default,mask_text`
- `source_choice_selector_choice_target=precision_gain_default_sourcewise_focal_bce`
- `eval_use_selector_choice_scores=true`
- `rng_seed=0`
- `save_freq=1`，并用外部清理脚本只保留最近少量 checkpoint，避免再次因为 `save_freq=50` 丢失十几个 epoch 的进度。

### 2026-06-20 11:22 UTC / 19:22 local 正确 full joint 长训已重启

已按旧最佳 epoch 13 同口径重启 full joint 长训；这不是 batch-1 selector-only run。

- run: `MCLN_source_choice_full_joint_restart_save1_keep3_seed0_20260620_110332`
- wrapper PID: `249737`
- launcher PID: `249747`
- train PID: `249800`
- stdout: `/root/autodl-tmp/DATA_ROOT/output/run_control/MCLN_source_choice_full_joint_restart_save1_keep3_seed0_20260620_110332.stdout.log`
- cmd: `/root/autodl-tmp/DATA_ROOT/output/run_control/MCLN_source_choice_full_joint_restart_save1_keep3_seed0_20260620_110332.cmd.sh`
- pid file: `/root/autodl-tmp/DATA_ROOT/output/run_control/MCLN_source_choice_full_joint_restart_save1_keep3_seed0_20260620_110332.pid`
- cleanup log: `/root/autodl-tmp/DATA_ROOT/output/run_control/MCLN_source_choice_full_joint_restart_save1_keep3_seed0_20260620_110332.cleanup.log`
- log: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_full_joint_restart_save1_keep3_seed0_20260620_110332/1781953653/log.txt`
- config: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_full_joint_restart_save1_keep3_seed0_20260620_110332/1781953653/config.json`

核对配置：

- `batch_size=12`
- `num_workers=2`
- `max_epoch=100`
- `save_freq=1`
- `val_freq=1`
- `use_source_choice_selector=true`
- `source_choice_selector_train_only=false`
- `source_choice_selector_sources=default,mask_text`
- `source_choice_selector_choice_target=precision_gain_default_sourcewise_focal_bce`
- `eval_use_selector_choice_scores=true`
- `rng_seed=0`
- `MASTER_PORT=4463`
- `CUDA_VISIBLE_DEVICES=0`

首条训练证据：

```text
[06/20 19:15:48] root INFO: length of training dataset: 48655
[06/20 19:15:48] root INFO: length of testing dataset: 9508
[06/20 19:21:38] root INFO: Train: [1][200/4054]
source_choice_loss 0.0306
[source_choice] source_choice_default_acc025 0.0058  source_choice_false_override_ratio 0.0075  source_choice_oracle_acc025 0.0100  source_choice_selected_non_default_ratio 0.0075  source_choice_target_acc 0.9367  source_choice_target_non_default_ratio 0.0558
```

资源状态：

- A100 正在训练，约 `25.7G/40G` 显存，占用和旧 batch-12 full joint 口径一致。
- `/root/autodl-tmp`: `31G/50G` used, `20G` available, `61%`。
- 已删除 batch-1 smoke 的两个 `573M` checkpoint 释放空间；保留日志/config 作为证据。
- 当前 run 用外部 cleanup loop 保留最近 `3` 个 `ckpt_epoch_N.pth`，避免 100 epoch 保存撑满磁盘；`ckpt_epoch_last.pth` 不在自动清理范围内。

### 2026-06-20 11:31 UTC / 19:31 local 正确长训继续推进

当前 full joint restart 仍在运行：

- wrapper PID: `249737`
- launcher PID: `249747`
- train PID: `249800`
- dataloader workers: `253164`, `253244`
- log: `/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_full_joint_restart_save1_keep3_seed0_20260620_110332/1781953653/log.txt`

最新稳定训练日志已经从 `[1][200/4054]` 推进到 `[1][400/4054]`：

```text
[06/20 19:21:38] root INFO: Train: [1][200/4054]
source_choice_loss 0.0306
[source_choice] source_choice_default_acc025 0.0058  source_choice_false_override_ratio 0.0075  source_choice_oracle_acc025 0.0100  source_choice_selected_non_default_ratio 0.0075  source_choice_target_acc 0.9367  source_choice_target_non_default_ratio 0.0558

[06/20 19:27:10] root INFO: Train: [1][400/4054]
source_choice_loss 0.0302
[source_choice] source_choice_default_acc025 0.0088  source_choice_false_override_ratio 0.0038  source_choice_oracle_acc025 0.0158  source_choice_selected_non_default_ratio 0.0038  source_choice_target_acc 0.9229  source_choice_target_non_default_ratio 0.0733
```

状态判断：

- 训练已持续推进，不是只启动到 data/text decoupling。
- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`Killed`、`nan` 或 `inf`。
- GPU 仍在跑，约 `26253MiB / 40960MiB`，util 约 `70%`。
- `/root/autodl-tmp`: `31G/50G` used, `20G` available, `61%`。
- 还未完成 epoch 1，因此尚未生成本 run 的 checkpoint；cleanup loop 已在监控，当前 checkpoint 数为 `0`。

### 2026-06-20 11:33 UTC / 19:33 local 继续推进到 step 600

当前 full joint restart 仍在运行，最新稳定训练日志推进到 `[1][600/4054]`：

```text
[06/20 19:21:38] root INFO: Train: [1][200/4054]
source_choice_loss 0.0306
[source_choice] source_choice_default_acc025 0.0058  source_choice_false_override_ratio 0.0075  source_choice_oracle_acc025 0.0100  source_choice_selected_non_default_ratio 0.0075  source_choice_target_acc 0.9367  source_choice_target_non_default_ratio 0.0558

[06/20 19:27:10] root INFO: Train: [1][400/4054]
source_choice_loss 0.0302
[source_choice] source_choice_default_acc025 0.0088  source_choice_false_override_ratio 0.0038  source_choice_oracle_acc025 0.0158  source_choice_selected_non_default_ratio 0.0038  source_choice_target_acc 0.9229  source_choice_target_non_default_ratio 0.0733

[06/20 19:32:41] root INFO: Train: [1][600/4054]
source_choice_loss 0.0298
[source_choice] source_choice_default_acc025 0.0110  source_choice_false_override_ratio 0.0025  source_choice_oracle_acc025 0.0194  source_choice_selected_non_default_ratio 0.0025  source_choice_target_acc 0.9199  source_choice_target_non_default_ratio 0.0776
```

状态判断：

- 训练仍在持续推进，`source_choice_loss` 从 `0.0306` 降到 `0.0298`。
- `false_override_ratio` 从 `0.0075` 降到 `0.0025`，当前早期训练段没有异常升高。
- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`Killed`、`nan` 或 `inf`。
- A100 显存约 `26257MiB / 40960MiB`；`/root/autodl-tmp` 仍为 `31G/50G` used、`20G` available、`61%`。
- 尚未完成 epoch 1，因此仍未生成本 run checkpoint；cleanup loop 继续监控，当前 checkpoint 数为 `0`。

### 2026-06-20 11:42 UTC / 19:42 local 继续推进到 step 800

当前 full joint restart 仍在运行，最新稳定训练日志推进到 `[1][800/4054]`：

```text
[06/20 19:32:41] root INFO: Train: [1][600/4054]
source_choice_loss 0.0298
[source_choice] source_choice_default_acc025 0.0110  source_choice_false_override_ratio 0.0025  source_choice_oracle_acc025 0.0194  source_choice_selected_non_default_ratio 0.0025  source_choice_target_acc 0.9199  source_choice_target_non_default_ratio 0.0776

[06/20 19:38:14] root INFO: Train: [1][800/4054]
source_choice_loss 0.0295
[source_choice] source_choice_default_acc025 0.0128  source_choice_false_override_ratio 0.0019  source_choice_oracle_acc025 0.0235  source_choice_selected_non_default_ratio 0.0019  source_choice_target_acc 0.9173  source_choice_target_non_default_ratio 0.0808
```

状态判断：

- 训练仍在持续推进，`source_choice_loss` 从 `0.0306` 继续降到 `0.0295`。
- `false_override_ratio` 从 `0.0075` 降到 `0.0019`，当前早期训练段没有异常升高。
- 未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`Killed`、`nan` 或 `inf`。
- train PID `249800` 仍在运行，elapsed 约 `34:39`，CPU 约 `110%`。
- A100 显存约 `26945MiB / 40960MiB`；`/root/autodl-tmp` 仍为 `31G/50G` used、`20G` available、`61%`。
- 尚未完成 epoch 1，因此仍未生成本 run checkpoint；下一关键节点是 `[1][1000/4054]` 与 epoch 1 保存/eval。

## 一致性原则

BUTD 与 MLCN 必须一致的是方法协议，不是张量名字：

| 维度 | 必须一致 | 允许不同 |
|---|---|---|
| 训练标签 | 用 GT IoU 判断哪个 deployable source 更好 | 具体 GT 字段读取方式 |
| 推理方式 | 不使用 GT，只用 selector 输出选 source | backbone 原生 score 名称 |
| selector 输入 | candidate features、candidate boxes、source scores、文本/分解上下文 | feature 维度和 adapter projection |
| loss | source-choice CE / sourcewise focal BCE / precision-first target | loss 接入位置 |
| 消融 | fixed source、oracle source、learned source | 每个模型可用 source 数量 |

不能为了“看起来一致”而伪造 BUTD 的 source 名称。MLCN 如果没有 `detector_jointtight`，就使用 MLCN 原生 default source，并在 adapter 中映射为 canonical `default`；论文中说明 default source 在 BUTD 对应 detector-primary，在 MLCN 对应 MLCN native grounding score。

## 通用模块定义

建议抽象为四个文件级单元：

```text
source_choice/
  selector.py
  losses.py
  adapter_base.py
  diagnostics.py
```

最小公共接口：

```python
class SourceChoiceBatch:
    candidate_boxes: Tensor      # [B, Q, 6], cxcyczwhd 或统一后的 xyzxyz
    candidate_feats: Tensor      # [B, Q, D]
    source_scores: Dict[str, Tensor]  # 每个 source: [B, Q]
    valid_mask: Tensor           # [B, Q]
    gt_boxes: Optional[Tensor]   # [B, G, 6], 仅训练/验证构造标签
    gt_mask: Optional[Tensor]    # [B, G]
    text_feats: Optional[Tensor]
    text_mask: Optional[Tensor]
    meta: Dict[str, Any]
```

通用 selector 输出：

```python
{
    "selector_choice_scores": Tensor,      # [B, S]
    "selector_choice_source_names": List[str],
    "selected_source_scores": Tensor,      # [B, Q]
    "selected_source_id": Tensor,          # [B]
}
```

训练期：

1. 对每个 source 取 top-1 candidate。
2. 计算 top-1 candidate 与 GT 的 IoU。
3. 用 threshold bucket / precision-first 规则生成 source-choice target。
4. 用 source-choice loss 监督 selector。

推理期：

1. adapter 生成同样的 deployable source scores。
2. selector 输出 source logits。
3. 选择 source 后，用该 source 的 `[B, Q]` score 排 candidate。
4. 不读取 GT、oracle IoU 或验证集 row dump 标签。

## MLCN 当前可接入位置

MLCN 相关文件：

| 文件 | 作用 |
|---|---|
| `E:\butd_detr-main\MCLN-main\models\mcln.py` | backbone forward，已有 `text_feats`、`text_attention_mask`、`query_last`、last prediction heads |
| `E:\butd_detr-main\MCLN-main\models\losses.py` | Hungarian / detection / grounding losses 接入点 |
| `E:\butd_detr-main\MCLN-main\src\grounding_evaluator.py` | 验证与 primary score source 切换 |
| `E:\butd_detr-main\MCLN-main\src\joint_det_dataset.py` | 数据字段、GT box、检测框输入 |
| `E:\butd_detr-main\MCLN-main\scripts\train_scanrefer_mcln_sp.sh` | ScanRefer MLCN 训练入口 |

`models/mcln.py` 中已有可用信号：

- `end_points['text_feats']`
- `end_points['text_attention_mask']`
- `end_points['query_points_feature']`
- decoder 中的 `query_last`
- prediction heads 输出的 `last_center` / `last_pred_size` / 语义与 mask 分支相关字段
- `end_points['last_pred_masks']`
- `end_points['adaptive_weights']`

第一步要先 audit 这些字段在一次 forward 后的真实 shape，再决定 adapter 使用哪一个作为 `candidate_feats` 和默认 source。

## MLCN source 设计

第一轮只做 2-source，避免 source 过多导致 harmful override：

| canonical source | MLCN 对应 | 说明 |
|---|---|---|
| `default` | MLCN 原生 grounding / contrastive / final query score | 对应 BUTD 中的 detector-primary/default fallback |
| `mask_text` | MLCN text decoder + mask 分支得到的 query/mask score | 对应 BUTD 中的 quality-like 辅助 source |

第二轮再扩展到 3-source：

| canonical source | MLCN 对应 | 使用条件 |
|---|---|---|
| `default` | 原生 grounding score | 必选 |
| `mask_text` | mask-text score | 必选 |
| `det_prior` 或 `box_prior` | BUTD 输入检测框/box objectness/semantic prior | 只有当 MLCN 已稳定暴露该 source 时加入 |

注意：不要在第一轮加入离线文本分解数据集产生的强耦合 source。用户当前观察到离线分解数据与数据增强有冲突，EDA 中指标甚至从约 `0.54` 降到 `0.51`；MLCN 迁移应先证明 source-choice selector 自身不破坏 baseline，再叠加文本分解增强。

## 执行阶段

### 阶段 0：复验 MLCN baseline

先不加模块，跑一次短验证，确认本机可复现 baseline。

参考 README 指标：

| 模型 | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
| MLCN ScanRefer reported | 57.17 | 45.53 |

如果本机复现低于 README，以本机 reproduced baseline 为后续对比基线；论文表格中同时报告 official 和 reproduced。

### 阶段 1：adapter audit

新增只读审计脚本：

```text
E:\butd_detr-main\MCLN-main\scripts\audit_source_choice_adapter.py
```

审计内容：

- 打印 `last_center`、`last_pred_size`、`query_last`、`text_feats`、`last_pred_masks`、`adaptive_weights` 的 shape。
- 取一个 batch 计算 default source top-1 Acc@0.25/Acc@0.50。
- 取一个 batch 计算 mask_text source top-1 Acc@0.25/Acc@0.50。
- 计算 2-source oracle，确认是否存在 headroom。

只有 oracle 明显高于 fixed best source，才进入 selector 训练；如果 oracle 都没有提升，说明 source 设计不成立。

### 阶段 2：接入通用 selector

建议文件变更：

| 文件 | 动作 |
|---|---|
| `E:\butd_detr-main\MCLN-main\models\source_choice_selector.py` | 从 BUTD 的 `models/source_pool_selector.py` 提取通用 selector，去掉 BUTD 专属 source 名称依赖 |
| `E:\butd_detr-main\MCLN-main\models\source_choice_adapter.py` | 新增 MLCN adapter，输出公共接口 |
| `E:\butd_detr-main\MCLN-main\models\mcln.py` | forward 末尾调用 adapter/selector，写入 `end_points` |
| `E:\butd_detr-main\MCLN-main\models\losses.py` | 增加 source-choice loss，训练期用 GT IoU 构造 target |
| `E:\butd_detr-main\MCLN-main\src\grounding_evaluator.py` | 增加 `selector_choice` primary score |
| `E:\butd_detr-main\MCLN-main\scripts\train_scanrefer_mcln_sp.sh` | 增加 selector 开关和长训配置 |

参数名建议与 BUTD 保持一致，便于论文和脚本对齐：

```bash
--use_source_choice_selector
--source_choice_selector_train_only
--source_choice_selector_lr
--source_choice_selector_loss_weight
--source_choice_selector_sources default,mask_text
--source_choice_selector_default_source default
--source_choice_selector_choice_target precision_gain_default_sourcewise_focal_bce
--eval_use_selector_choice_scores
```

MLCN 内部如果沿用 BUTD 代码，也可以继续保留 `source_pool_selector_*` 参数名；但论文和文档统一称为 `SourceChoiceSelector`。

### 阶段 3：单元测试

新增测试：

```text
E:\butd_detr-main\MCLN-main\tests\test_source_choice_selector.py
E:\butd_detr-main\MCLN-main\tests\test_source_choice_adapter.py
```

至少覆盖：

- adapter 输出 shape 正确。
- source scores 均为 `[B, Q]`。
- 训练 loss 使用 GT 构造 target，但 inference 分支不需要 GT。
- `precision_gain_default_sourcewise_focal_bce` 在小 IoU gap 时保持 default，在跨阈值且足够 gap 时选择 non-default。
- evaluator 使用 selected source score，而不是 oracle top IoU。

### 阶段 4：1 epoch smoke

先跑 1 epoch，目标不是涨点，而是验证：

- loss 不为 NaN。
- selector target ratio、selected non-default ratio、false override ratio 都能记录。
- eval 能输出 fixed default、fixed mask_text、oracle、learned selector 四组结果。

建议脚本：

```bash
cd /root/autodl-tmp/butd_detr-main/MCLN-main
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_scanrefer_mcln_sp.sh \
  --max_epoch 1 \
  --use_source_choice_selector \
  --source_choice_selector_train_only \
  --source_choice_selector_lr 0.001 \
  --source_choice_selector_loss_weight 1.0 \
  --source_choice_selector_sources default,mask_text \
  --source_choice_selector_default_source default \
  --source_choice_selector_choice_target precision_gain_default_sourcewise_focal_bce \
  --eval_use_selector_choice_scores \
  --rng_seed 0
```

### 阶段 5：2-3 epoch short run

短训配置：

- seed 固定为 `0`。
- 先 selector-only，冻结 MLCN 主干。
- 2-source：`default,mask_text`。
- loss target：`precision_gain_default_sourcewise_focal_bce`。
- batch size 使用 baseline 可承受的最大值；若 OOM，优先降 batch，不改方法。

短训验收：

| 条件 | 决策 |
|---|---|
| learned selector >= fixed default + 0.2 Acc@0.25 点 | 进入 long train |
| learned selector 与 fixed default 持平，但 false override 明显低 | 可调 1 轮 LR/loss weight |
| learned selector 明显低于 fixed default | 停止长训，回到 source 设计 |
| oracle 不高于 fixed default | 停止 selector，说明 source pool 不成立 |

### 阶段 6：long train

长训只在 short run 满足条件后启动。

建议命名：

```text
output/logs/mcln_source_choice_selector_default_masktext_seed0_long/
```

长训设置：

- `--rng_seed 0`
- `--max_epoch 100`
- `--val_freq 1`
- `--save_freq 1`
- 先 selector-only 到稳定，再决定是否低 LR joint fine-tune。

若 selector-only 有提升，第二阶段可以小心尝试 joint fine-tune：

```bash
--lr 1e-5
--lr_backbone 1e-5
--text_encoder_lr 1e-6
```

但 joint fine-tune 必须作为单独 run，不要和 selector-only 混在一个主结果里。

## 与 BUTD 三个基础创新点的关系

该路线可以包装为通用模块，但要避免把所有创新都塞进 MLCN：

1. 文本分解：作为 source/context 的输入增强，而不是强制替换 MLCN 原始在线解析。
2. 分解感知增强：MLCN 第一轮不直接使用离线分解数据增强，避免复现 EDA 中的冲突。
3. 困难负样本：适合包装成通用模块，但应该作为 selector 的训练样本策略，而不是推理规则。

困难负样本的通用化建议：

- 定义 hard negative 为：default 与 non-default top candidate 不同，且 non-default 看似高分但低于 default 的 GT threshold bucket。
- 训练时提高这类样本的 negative weight，抑制 false override。
- BUTD 与 MLCN 都使用同一规则，只是 candidate/source 来自各自 adapter。

这样第三个创新点可以和 source-choice selector 自然合并：不是单独“造负样本 trick”，而是为了训练通用 source arbitration module 的 precision-first hard-negative supervision。

## 论文消融表建议

MLCN 表格至少包含：

| 设置 | Acc@0.25 | Acc@0.50 | 说明 |
|---|---:|---:|---|
| MLCN reproduced baseline |  |  | 本机复现 |
| fixed `default` source |  |  | 不使用 selector |
| fixed `mask_text` source |  |  | 辅助 source 单独表现 |
| oracle over sources |  |  | 只作上限，不作主结果 |
| learned SourceChoiceSelector |  |  | 主结果 |
| learned selector w/o hard negatives |  |  | 证明困难负样本作用 |
| learned selector w/o decomposition context |  |  | 证明文本分解上下文作用 |

报告时必须写：

```text
Oracle source labels are used only for training supervision and diagnostic upper bound. During inference, the selector uses no ground-truth boxes or oracle IoU.
```

## 成功标准

MLCN 路线的成功标准建议比 BUTD 更务实：

- 若 MLCN reproduced baseline 约 `0.5717 / 0.4553`，learned selector 稳定提升 `+0.002` 到 `+0.004` Acc@0.25，就足够作为通用模块证据。
- 若同时 `Acc@0.50` 不下降，并且 hard-negative ablation 能解释 false override 降低，可以支撑论文。
- 如果 MLCN 涨幅低于 `+0.001`，但 BUTD 能稳定涨 `+0.004`，则 source-choice selector 作为 BUTD 内部优化写法更合适，不宜强行宣称通用。

## 停止条件

满足任一条件，暂停 MLCN 长训：

- 2-source oracle 不高于 fixed default。
- learned selector 连续两个 short run 低于 fixed default 超过 `0.002`。
- selected non-default ratio 远高于 oracle non-default ratio，且 false override 增加。
- 离线文本分解数据集再次导致 baseline 从约 `0.54/0.57` 降到明显低位，说明数据管线与增强冲突未解决。

下一步优先级：

1. 先做 adapter audit 和 source oracle。
2. oracle 有 headroom 再做 selector smoke。
3. smoke 稳定后短训。
4. 短训有正收益才长训。
