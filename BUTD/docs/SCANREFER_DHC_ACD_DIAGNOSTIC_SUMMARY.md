# ScanRefer DHC/ACD 当前问题与排查总结

更新日期：2026-04-28

本文只总结当前 ScanRefer two-stage、S2S、ACD、DHC 相关问题和实验建议。已弃用的旧文本侧方案不纳入本文。

## 1. 当前检查的训练

日志目录：

```text
logs/scanrefer_spacy/two-stage/block5_s2s_acd_dhc_scanrefer/scanrefer_spacy/1777282307
```

关键配置：

| 项目 | 当前值 | 说明 |
|---|---:|---|
| `dataset` | `scanrefer_spacy` | ScanRefer spaCy 数据 |
| `joint_det` | `true` | 训练中混入 detection-only 样本 |
| `use_structured_slots` | `true` | 启用结构化槽 |
| `use_late_acd` | `true` | 启用 ACD |
| `use_dhc` | `true` | 启用 DHC |
| `use_s2s_aux_loss` | `false` | 当前 run 没有验证 S2S aux |
| `acd_rank_weight` | `1.0` | ACD rank loss 权重偏强，需消融 |
| `eval_use_acd_scores` | `false` | 当前 eval 默认不用 ACD final score |
| `dhc_margin_min` | `0.0` | 没有 margin floor |
| `dhc_temperature_max` | `0.0` | 没有 temperature cap |

## 2. 当前指标表现

用户给定参考：

| 模型/目标 | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
| 基准模型 | 49.8 | 37.1 |
| 目标 | 54.3 | 42.2 |

当前 run 的 eval 曲线：

| Epoch | bbf@0.25 | bbf@0.50 | bbs@0.25 | bbs@0.50 |
|---:|---:|---:|---:|---:|
| 5 | 35.37 | 17.59 | 34.35 | 17.37 |
| 10 | 41.74 | 24.61 | 41.34 | 24.39 |
| 15 | 44.40 | 29.01 | 43.89 | 28.69 |
| 20 | 46.38 | 31.47 | 46.01 | 31.26 |
| 25 | 47.65 | 33.06 | 47.38 | 32.77 |
| 30 | 48.74 | 33.11 | 48.23 | 32.73 |
| 35 | 50.00 | 35.80 | 49.68 | 35.46 |

结论：

- @0.25 已接近或略过基准，但 @0.50 仍低于基准约 1.3。
- 距离目标 54.3 / 42.2 仍有明显差距。
- 曲线到 epoch 35 仍在上涨，尤其 @0.50 在 epoch 35 有一次明显提升。

## 3. 已确认并修复的问题

### 3.1 ScanNet detection-only 样本污染结构化分支

问题：

- `joint_det=true` 时，训练 batch 中约 24%-25% 是 detection-only 样本。
- 这些样本不应进入结构化分支的监督和重排。
- 旧逻辑可能让这类样本影响 ACD/DHC，造成结构化分支被污染。

当前状态：

- 日志中 `dbg_struct_scannet_batch_ratio ~= 0.24-0.25`。
- `dbg_struct_valid_batch_ratio ~= 0.75`。
- `dbg_acd_top1_changed_unstructured_ratio=0.0000`。

判断：

- 当前 run 中 unstructured 样本没有被 ACD 改变 top1。
- 这个污染问题目前看已经被挡住。
- 后续实验仍应持续监控 `dbg_acd_top1_changed_unstructured_ratio`，它应保持为 0。

### 3.2 DHC loss 聚合

问题：

- 之前 DHC 聚合需要确保只聚合真正的 `loss_dhc_*` 项。
- debug 项不能混入训练 loss。

当前状态：

- 已按 loss key 过滤。
- 当前日志中 `dbg_dhc_total_loss` 只作为诊断输出，不应参与反向传播。

后续检查点：

- 新增 debug 指标时，必须避免命名成 `loss_*`，除非它确实要参与训练。
- 聚合逻辑继续只匹配真实 DHC loss 项。

### 3.3 ScanRefer spaCy `target_slot` 没有传入 sample

问题：

- ScanRefer spaCy loader 里构造 `entity_spans` 时读取了 `anno.get("target_slot")`。
- 但 sample 曾经没有保留 `target_slot` 字段。
- 下游 batch 读取 `anno.get("target_slot", [])` 时总是空，导致 S2S target aux 退回到 `positive_map` fallback。

当前状态：

- 已补充 `target_slot` 字段传递。
- 但当前 run 配置 `use_s2s_aux_loss=false`，所以这次没有验证该路径的实际训练效果。

后续检查点：

- 单独跑开启 S2S aux 的实验，确认：
  - 显式 `target_slot` 使用比例。
  - fallback 使用比例。
  - `loss_s2s_aux` 是否为非零且稳定。

### 3.4 优先实验脚本与实际计划不一致

问题：

- 旧优先脚本和建议实验顺序不完全一致。
- 多进程继承同一 `MASTER_PORT` 时可能冲突。
- 用户已明确当前只有单卡，不应按三卡并行设计。

当前状态：

- 优先脚本已调整为单卡顺序执行。
- ACD 降权消融已放入优先顺序。
- 子实验端口按 base port 派生，避免显式设置同一端口时冲突。

## 4. 当前仍存在的主要问题

### 4.1 DHC objective 近似失效

当前 run 后期的典型现象：

| 指标 | 后期数值 | 解释 |
|---|---:|---|
| `dbg_dhc_total_loss` | 约 0.0012 | DHC loss 已非常小 |
| `dbg_dhc_margin_acd_rank` | 约 0.0009-0.0011 | ACD rank margin 塌缩 |
| `dbg_dhc_margin_attr` | 约 0.0003 | 属性 margin 塌缩 |
| `dbg_dhc_margin_entity` | 约 0.0004-0.0005 | 实体 margin 塌缩 |
| `dbg_dhc_margin_rel` | 约 0.0003 | 关系 margin 塌缩 |
| `dbg_dhc_temperature` | 约 10-12 | temperature 过高 |
| `dbg_dhc_ent_gap` | 负值 | 正样本没有高于 hard negative |
| `dbg_dhc_attr_gap` | 负值 | 正样本没有高于 hard negative |
| `dbg_dhc_rel_gap` | 负值 | 正样本没有高于 hard negative |
| violation ratio | 约 0.89-0.93+ | 大量样本仍违反约束 |

判断：

- DHC 没有真正把正样本拉开。
- loss 变小主要来自 learned margin 塌到接近 0，以及 temperature 拉高。
- 这是当前最重要的问题。

可能原因：

- `dhc_margin_min=0.0`，模型可以通过压低 margin 规避约束。
- `dhc_temperature_max=0.0`，没有上限，temperature 可以变大。
- DHC 的 hard negative 约束太容易通过参数缩放绕开。
- 正样本监督极稀疏，`dbg_dhc_positive_query_ratio ~= 0.0039`，约等于每 256 query 只有 1 个 target positive。

建议优先消融：

- `--dhc_margin_min 0.02`
- `--dhc_margin_min 0.05`
- `--dhc_temperature_max 4.0`
- `--dhc_temperature_max 6.0`

### 4.2 DHC consistency 基本没有贡献

现象：

- `loss_dhc_consistency=0.0000`。
- `dbg_dhc_consistency_js_proxy` 接近 0。
- `dbg_dhc_consistency_top1_agreement` 后期大约 0.20-0.26，仍偏低。

判断：

- consistency loss 没有形成有效约束。
- top1 agreement 很低，说明不同 score/source 之间的一致性并不好。
- 需要确认 consistency 计算是否因为分布过平、detach、mask 或 score 尺度导致几乎无梯度。

### 4.3 ACD rank 没有产生正向分离

现象：

- `dbg_acd_rank_gap` 基本为 `-0.0000`。
- `dbg_acd_rank_pos_mean` 与 hard negative 几乎相等。
- `dbg_acd_top1_changed_structured_ratio` 约 0.76-0.81。
- `dbg_acd_top1_changed_unstructured_ratio=0.0000`。

判断：

- ACD 现在不会污染 unstructured 样本，这是好信号。
- 但 structured 样本上 ACD 大量改变 top1，却没有证据表明它在朝正确候选重排。
- ACD rank loss 权重 `1.0` 可能过强或方向不稳定。

建议优先消融：

- `--acd_rank_weight 0.25`
- `--acd_rank_weight 0.0`

需要新增或优化诊断：

- ACD 改变 top1 后，正确样本比例是上升还是下降。
- top1 changed 的样本中，原本正确变错误、原本错误变正确分别占多少。
- ACD final score 与 base score 的 top1 IoU 差异。

### 4.4 ACD 结构化贡献过小

现象：

- `dbg_acd_struct_contrib_abs_mean` 后期接近 0。
- `dbg_acd_alpha_mean` 后期约 0.006。
- `dbg_acd_s_ea_mean` 约 0.011，明显大于 base 均值。
- `dbg_acd_s_base_mean` 后期缩小到约 0.001-0.002。

判断：

- ACD final score 的变化主要来自 EA 项，而不是结构化 tuple 贡献。
- 结构化分支虽然启用，但实际影响可能非常弱。
- base score 尺度后期明显变小，容易被小的附加 score 主导。

潜在风险：

- ACD 可能变成 score scale 调整器，而不是可靠的结构化推理模块。
- 如果 eval 使用 ACD final score，可能带来不稳定提升或下降。

### 4.5 当前 eval 默认没有使用 ACD final score

现象：

- config 中 `eval_use_acd_scores=false`。

判断：

- 当前 reported ScanRefer eval 主要反映训练正则带来的间接影响。
- 如果论文或实验目标需要证明 ACD 推理贡献，需要单独对比：
  - eval 不使用 ACD final score。
  - eval 使用 ACD final score。

注意：

- 当前 ACD rank gap 不好，直接打开 eval ACD 可能有风险。
- 建议只作为对照评估，不先作为主线。

### 4.6 当前 run 没有验证 S2S aux

现象：

- `use_s2s_aux_loss=false`。
- `loss_s2s_aux=0.0000`。

判断：

- 这次训练不能说明 `target_slot` 修复后 S2S aux 是否有效。
- 需要单独实验验证，不能和 DHC 参数修改混在一起。

### 4.7 训练进程可能已停止

观察：

- 目录已有 epoch 35 checkpoint 和 eval。
- `log.txt` 最新记录停在 epoch 36 中段。
- 通过进程名查找未看到包含当前 run id 的进程。

判断：

- 如果预期仍在训练，需要在服务器上确认 GPU 进程。
- 若已停止，应优先确认是否手动停止、报错退出、磁盘空间问题或外部调度中断。

## 5. 潜在代码与实验风险

### 5.1 数据字段传递风险

需要持续确认：

- ScanRefer spaCy sample 中保留：
  - `entity_spans`
  - `attr_spans`
  - `rel_spans`
  - `target_slot`
  - `dataset`
- batch collate 后这些字段仍能被 loss 侧读取。
- 不同 dataset 的空字段不会触发错误 fallback。

### 5.2 fallback 逻辑风险

风险：

- `target_slot` 为空时 fallback 到 `positive_map` 是必要兜底。
- 但如果字段传递丢失，fallback 会掩盖 bug。

建议诊断：

- 输出显式 `target_slot` 命中比例。
- 输出 fallback 比例。
- 输出空 target 但非 ScanRefer 样本比例。

### 5.3 结构化 mask 风险

当前有效 mask 逻辑应满足：

- 必须有 parsed target。
- 必须不是 detection-only 样本。

建议持续监控：

- `dbg_struct_valid_batch_ratio`
- `dbg_struct_scannet_batch_ratio`
- `dbg_acd_rank_valid_ratio`
- `dbg_dhc_valid_batch_ratio`
- `dbg_acd_top1_changed_unstructured_ratio`

其中 `dbg_acd_top1_changed_unstructured_ratio` 应保持 0。

### 5.4 loss 命名风险

风险：

- debug 项如果命名成 `loss_*`，可能被总 loss 聚合。
- DHC 聚合必须只包含真实反向传播项。

建议：

- debug 一律使用 `dbg_*`。
- warning 一律使用 `dbg_warn_*`。
- 新增 loss 必须明确对应权重和是否加入总 loss。

### 5.5 旧脚本和旧文档误用风险

风险：

- 仓库里仍有历史方案文档和脚本。
- 如果在另一台服务器误用旧入口，可能跑出和当前主线不一致的实验。

建议：

- 当前 ScanRefer 主线优先使用 `scripts/scanrefer/two-stage/` 和 `run_priority_experiments.sh`。
- 另一台服务器同步文件后，先检查 `config.json` 中关键 flags。
- 实验表中记录完整命令和 log dir，避免只看目录名判断实验内容。

### 5.6 单卡实验排队风险

风险：

- 当前只有单卡，不能按三卡并行计划推进。
- 多实验同时启动会互相抢显存或端口。

建议：

- 使用顺序执行。
- 每个实验固定独立 log dir。
- 每次只跑一个主实验。
- 优先在 epoch 10/20/30/40 做中间评估，不等 300 epoch。

## 6. 建议的单卡实验顺序

### 优先级 1：确认当前 run 是否能继续

目的：

- 当前曲线到 epoch 35 仍上涨。
- @0.50 在 epoch 35 有明显提升。

建议：

- 若训练已停，先确认原因。
- 如果没有异常，可从 epoch 35 checkpoint 继续到 epoch 45 或 50。
- 继续观察是否能超过基准 @0.50=37.1。

### 优先级 2：干净 S2S-only 参照

目的：

- 确认结构化 mask 修复后，block5 是否真的优于 S2S-only。
- 避免把自然训练收益误判为 DHC/ACD 收益。

建议：

- 跑 `block1_s2s_only`。
- 与当前 block5 使用同一数据、同一训练设置、同一 eval 口径。

### 优先级 3：ACD rank 降权

目的：

- 验证 `acd_rank_weight=1.0` 是否过强。
- 当前 ACD rank gap 没有正向分离，但 structured top1 changed 很高。

建议：

```text
--acd_rank_weight 0.25
```

若仍无改善，再跑：

```text
--acd_rank_weight 0.0
```

### 优先级 4：DHC 稳定参数

目的：

- 阻止 margin 塌缩。
- 限制 temperature 过大。

建议先跑一个轻量组合：

```text
--dhc_margin_min 0.02 --dhc_temperature_max 6.0
```

若 DHC loss 仍快速消失，再考虑：

```text
--dhc_margin_min 0.05 --dhc_temperature_max 4.0
```

观察重点：

- gap 是否从负值接近 0 或转正。
- violation ratio 是否下降。
- DHC loss 是否保持有效但不过大。
- @0.50 是否改善。

### 优先级 5：单独验证 S2S aux

目的：

- 验证 `target_slot` 字段修复后是否带来收益。

建议：

- 单独开启 `--use_s2s_aux_loss`。
- 暂时不要同时改 DHC margin/temperature。
- 观察 `loss_s2s_aux` 和显式 target 命中比例。

### 优先级 6：ACD eval 对照

目的：

- 判断 ACD final score 在推理阶段是否有正收益。

建议：

- 对同一个 checkpoint 分别评估：
  - `eval_use_acd_scores=false`
  - `eval_use_acd_scores=true`

注意：

- 当前 ACD rank gap 不好，建议只作为诊断，不先作为主结果。

## 7. 建议补充的诊断指标

### 7.1 ACD 改 top1 的真实收益

新增指标建议：

- `dbg_acd_top1_base_correct_ratio`
- `dbg_acd_top1_final_correct_ratio`
- `dbg_acd_changed_wrong_to_right_ratio`
- `dbg_acd_changed_right_to_wrong_ratio`
- `dbg_acd_changed_iou_delta_mean`

目的：

- 直接判断 ACD 改 top1 是帮忙还是破坏。

### 7.2 DHC margin 和 temperature 有效性

新增指标建议：

- raw margin
- clipped/floored margin
- raw temperature
- clipped temperature
- weighted DHC loss
- unweighted DHC loss

目的：

- 区分 loss 变小是模型学好了，还是参数绕开了约束。

### 7.3 target source 诊断

新增指标建议：

- 显式 `target_slot` 使用比例。
- fallback 到 `positive_map` 比例。
- 空 target 比例。
- 非 detection-only 且 target 为空的比例。

目的：

- 防止字段传递问题被 fallback 掩盖。

### 7.4 按样本来源分组

新增指标建议：

- ScanRefer 样本上的 ACD/DHC 有效比例。
- detection-only 样本上的 ACD/DHC 有效比例。
- unstructured 样本 top1 changed ratio。

目的：

- 防止后续改动重新引入样本污染。

## 8. 近期需同步文件清单

以下为近期源码和脚本变更清单，不含 `__pycache__`。另一台服务器建议同步这些文件后再跑实验：

```text
main_utils.py
train_dist_mod.py
visualize_grounding.py
run_priority_experiments.sh
models/acd_head.py
models/bdetr.py
models/encoder_decoder_layers.py
models/losses.py
models/structured_losses.py
src/grounding_evaluator.py
src/joint_det_dataset.py
docs/SCANREFER_DHC_ACD_DIAGNOSTIC_SUMMARY.md
```

同步后建议先运行：

```text
python -m py_compile src/joint_det_dataset.py main_utils.py models/losses.py models/structured_losses.py models/acd_head.py visualize_grounding.py train_dist_mod.py
bash -n run_priority_experiments.sh
```

## 9. 当前结论

当前最可信的判断：

1. detection-only 样本污染结构化分支的问题已经被当前 mask 挡住。
2. 当前 block5 相比旧污染 run 明显更正常，bbs 不再崩。
3. 主要瓶颈已经转移到 DHC/ACD objective 本身：
   - DHC margin 塌缩。
   - temperature 过大。
   - gap 仍为负。
   - violation ratio 仍高。
   - ACD rank 没有正向分离。
4. 当前 run 尚未验证 S2S aux 的 `target_slot` 修复效果。
5. 下一步不应盲目叠模块，应先用单卡顺序消融确定：
   - S2S-only 的干净上限。
   - ACD rank 是否副作用。
   - DHC margin/temperature 稳定后是否有效。
   - S2S aux 单独打开是否有增益。
