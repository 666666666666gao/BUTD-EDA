# BUTD Run225 Source-Choice Selector Optuna Retuning Plan

日期：2026-06-14

## 目标

本路线继续优化 BUTD-DETR 中目前最像论文方法点的额外优化：训练时 oracle 监督的 source gate / source-choice selector。它不是测试时偷看 GT，而是在训练阶段用 GT IoU 判断哪个可部署 score source 更好，再把这个选择蒸馏给一个小 selector；测试阶段只使用模型学到的 selector logits 做 source 选择。

当前目标分两级：

1. 短期目标：复现并超过 Run225 的 documented best learned result。
2. 冲刺目标：把 ScanRefer `Acc@0.25` 从当前约 `0.543` 推到 `0.56` 附近，同时不能明显牺牲 `Acc@0.50`。

## 当前基线与约束

| 项目 | Acc@0.25 | Acc@0.50 | 说明 |
|---|---:|---:|---|
| Run225 documented best learned reference | 0.5431 | 0.4226 | 当前记录中最好的可部署 learned selector；checkpoint 目前缺失 |
| best loadable detector-primary reference | 0.5425 | 0.4221 | 当前可加载、可复验的强参考 |
| Run331 same-run detector fallback | 0.5397 | 0.4205 | Run297 epoch89 审计环境下的 detector fallback |
| Run331 selector_choice | 0.5369 | 0.4171 | 证明当前 selector 仍会做有害切换 |
| Run331 oracle over selector sources | 0.5715 | 0.4575 | 说明 source-choice 仍有上限，但不是可部署结果 |

关键风险：

- Run225 的 checkpoint 缺失。若无法找回，只能把 Run225 当作指标参考，从 Run297 epoch89 或最近可加载的兼容 checkpoint 复刻同类配置。
- 近期 Run314/315/317/329/331/332 说明：简单增加 margin、继续长训、加入更多 source 或只改 augmentation，均没有稳定转化为可部署提升。
- 当前问题更像 source-choice separability：selector 要减少 false override，同时抓住少量高置信 non-default switch。

## 方法定义

论文/实验口径建议统一为：

**Oracle-Supervised Source Choice Selector**

给定同一模型产生的多个可部署 score source，训练阶段计算每个 source 的 top-1 candidate 与 GT 的 IoU，并构造 source-choice 标签；selector 学习从候选框、候选特征、source 分数与文本分解上下文中预测应该信任哪个 source。推理阶段不使用 GT，只按 selector 输出选择 source，再用被选 source 的分数完成 grounding。

必须固定的边界：

- GT 只用于训练监督和验证统计。
- 测试时不可使用 GT IoU、oracle source id、oracle top IoU、验证集回放阈值。
- source 名称和数量可以作为超参搜索，但 long train 的最终配置必须是训练/验证/测试一致的 deployable selector。

## 固定种子与可复现设置

所有 short trial 和 long train 固定同一随机种子：

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NMV2_RNG_SEED=0
```

训练参数中显式传入：

```bash
--rng_seed 0
```

如果代码路径允许，建议同时打开：

```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(False)
```

最后一项保持 `False`，避免 3D/分布式算子直接报错；但要在文档中记录 CUDA、PyTorch、GPU 数量、batch size、checkpoint sha 或 run id。

## Optuna 复调设计

### 搜索预算

- Optuna trials：20 组。
- 每组训练：2 个 epoch。
- 每组至少保存：`config.json`、`log.txt`、epoch eval log、`eval_results_latest.json`、source-choice row dump 或 summary。
- 每组使用相同 seed；如果机器时间允许，top 3 配置再用 seed `0/1/2` 做短复验，但本轮主任务先固定 seed。

### 起点 checkpoint

优先级：

1. 找回 Run225 checkpoint，并从 Run225 继续复调。
2. 如果 Run225 checkpoint 不可恢复，使用 Run297 epoch89：
   `/root/autodl-tmp/butd_logs/new_method_v2/scanrefer/two_stage/297_candidateaware_3src_loosegap003_from_run261_epoch88_to_90_b112/scanrefer_spacy/1779169753/ckpt_epoch_89.pth`
3. 如果本机没有上述路径，使用当前最接近 Run225 配置且可加载的 detector-primary / candidate-aware checkpoint。

若从 epoch89 开始，2-epoch trial 的 `max_epoch` 设为 `91`；若从 epoch85 开始，设为 `87`。文档和 Optuna trial name 必须写清楚实际起点。

### 搜索空间

建议单独新建或改造：

```text
scripts/new_method_v2/tuning/optuna_run225_source_choice.py
```

不要直接复用 `optuna_scanrefer_two_stage_full.py` 的 RAPF 搜索空间；那个脚本主要调 RAPF/fusion 参数，不是 Run225/source-choice selector 的核心变量。

本轮搜索空间建议：

| 参数 | 候选值 | 说明 |
|---|---|---|
| `batch_size` | `24`, `48`, `80`, `112` | 先按显存过滤；OOM 记为 failed trial |
| `source_pool_selector_lr` | `3e-4`, `1e-3`, `3e-3` | selector-only 优先；full fine-tune 不放进第一轮 |
| `source_pool_selector_loss_weight` | `0.5`, `1.0`, `2.0` | 避免过小导致 logits 不动 |
| `source_pool_selector_choice_target` | `threshold_gain_default_diffquery_sourcewise_focal_bce`, `precision_gain_default_sourcewise_focal_bce`, `base_override_sourcewise_focal_bce` | 以 precision-first 和 diff-query 为主 |
| `source_pool_selector_override_default_source` | `detector_jointtight` | 默认 source 固定为最强 detector fallback |
| `source_pool_selector_candidate_sources` | `detector_jointtight,base,quality`; `detector_jointtight,quality` | 三源看上限，两源看精度 |
| `source_pool_selector_min_iou_gap` | `0.02`, `0.03`, `0.05` | 控制弱增益标签 |
| `source_pool_selector_false_base_weight` | `1.0`, `1.5`, `2.0` | 抑制该切却不切 |
| `source_pool_selector_false_override_weight` | `1.0`, `2.0`, `3.0` | 抑制 harmful override，建议重点调 |
| `source_pool_selector_sourcewise_negative_weight` | `1.0`, `1.5`, `2.0` | sourcewise focal BCE 的负样本权重 |
| `eval_selector_choice_min_margin` | `0.0`, `0.25`, `0.5` | 推理时保守切换阈值；不能用 GT 调验证集后验阈值 |

固定打开：

```bash
--use_source_pool_selector
--source_pool_selector_direct_choice
--source_pool_selector_train_only
--source_pool_selector_source source_choice
--eval_use_selector_choice_scores
--eval_target_cid_source text
--text_target_alias_policy strict
--disable_box_jitter
--spacy_relation_free_yaw_only_aug
--spacy_relation_free_view_guard_aug
--spacy_relation_free_compass_guard_aug
--spacy_direction_sensitive_no_jitter_aug
```

原则上第一轮不打开 `--augment_det`。近期记录显示 augmentation cleanup 是必要控制项，但单靠它不是涨点来源。

## Optuna 目标函数

主目标：

```text
objective = 0.45 * Acc@0.25 + 0.45 * Acc@0.50 + 0.10 * mean_iou
```

并加入硬性筛选：

- `Acc@0.25 < detector_fallback - 0.001` 的 trial 不进入 top list。
- `Acc@0.50 < detector_fallback - 0.001` 的 trial 不进入 top list。
- `false_override_ratio > useful_override_ratio` 且指标低于 detector fallback 的 trial 标记为 reject。
- selector 选择 non-default 的比例若大于 oracle non-default 比例 1.5 倍，优先判为过切换风险。

排序时以 `Acc@0.25` 为第一指标，`Acc@0.50` 为第二指标，`false_override_ratio` 为第三指标。

## 执行流程

### 1. 先补 Run225 checkpoint 审计

```bash
cd /root/autodl-tmp/butd_detr-main
find /root/autodl-tmp/butd_logs -iname '*225*' -o -iname 'ckpt_epoch_*.pth' | sort
```

若找到 Run225 checkpoint，先只跑 eval，确认能否复现 `0.5431 / 0.4226`。若找不到，记录为：

```text
Run225 checkpoint unavailable; retuning starts from Run297 epoch89 compatible checkpoint.
```

### 2. 实现 Run225 Optuna launcher

从 `scripts/new_method_v2/tuning/optuna_scanrefer_two_stage_full.py` 拷贝基础框架，但修改三处：

- `SEARCH_SPACE` 改成 source-choice selector 参数。
- `trial_override_args()` 改成传入 selector 参数、checkpoint、batch size 和 eval source。
- parser 增加 `--start-checkpoint`、`--start-epoch`、`--trial-epochs`、`--objective-metric`。

建议输出目录：

```text
logs/new_method_v2/tuning/run225_source_choice_optuna_20x2_seed0/
```

### 3. 先 dry-run 一个 trial

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
conda run -n bdetr python scripts/new_method_v2/tuning/optuna_run225_source_choice.py \
  --dry-run \
  --n-trials 1 \
  --trial-epochs 2 \
  --seed 0 \
  --start-epoch 89 \
  --start-checkpoint /root/autodl-tmp/butd_logs/new_method_v2/scanrefer/two_stage/297_candidateaware_3src_loosegap003_from_run261_epoch88_to_90_b112/scanrefer_spacy/1779169753/ckpt_epoch_89.pth \
  --output-root logs/new_method_v2/tuning/run225_source_choice_optuna_20x2_seed0
```

检查 dry-run 输出中必须包含：

```text
--use_source_pool_selector
--source_pool_selector_direct_choice
--source_pool_selector_train_only
--source_pool_selector_source source_choice
--eval_use_selector_choice_scores
--rng_seed 0
```

### 4. 跑 20 组短训

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
conda run -n bdetr python scripts/new_method_v2/tuning/optuna_run225_source_choice.py \
  --n-trials 20 \
  --trial-epochs 2 \
  --seed 0 \
  --start-epoch 89 \
  --start-checkpoint /root/autodl-tmp/butd_logs/new_method_v2/scanrefer/two_stage/297_candidateaware_3src_loosegap003_from_run261_epoch88_to_90_b112/scanrefer_spacy/1779169753/ckpt_epoch_89.pth \
  --output-root logs/new_method_v2/tuning/run225_source_choice_optuna_20x2_seed0 \
  --study-name run225_source_choice_seed0
```

如果使用 Run225 recovered checkpoint，把 `--start-checkpoint` 和 `--start-epoch` 改成实际值，并在 study name 中写 `from_run225`。

### 5. 选 top 配置并复验

短训结束后导出表格：

```bash
conda run -n bdetr python scripts/new_method_v2/tuning/parse_eval_metric.py \
  --root logs/new_method_v2/tuning/run225_source_choice_optuna_20x2_seed0 \
  --sort-by Acc@0.25 \
  --top-k 5
```

选择标准：

- 首选同时超过 detector fallback 和 Run225 reference 的配置。
- 如果没有超过 Run225，但超过 detector fallback 且 false override 更低，可以进入 top 3 复验。
- 如果 top 配置仍低于 detector fallback，不启动长训，改走 MLCN 通用模块路线。

### 6. 长训

长训只选 1 个主配置，最多加 2 个备选配置。建议：

- 从 short trial 的最佳 checkpoint 继续，而不是重新从老 checkpoint 训练。
- 若 short trial 是 epoch89 到 epoch91，long train 可继续到 epoch100 或 epoch110。
- `save_freq=1`，至少保存最近 3 个 epoch 的 checkpoint 与 eval。

命名建议：

```text
logs/new_method_v2/scanrefer/two_stage/225R_optuna_best_source_choice_seed0_from_epoch91_to100/
```

长训启动前必须把完整配置写入本报告或单独 `run225R_config.md`。

## 失败/停止条件

满足任一条件，停止 BUTD 继续堆 selector：

- 20 组短训没有任何配置超过 `0.5425 / 0.4221`。
- top 配置只提高 `Acc@0.25` 但 `Acc@0.50` 下降超过 `0.003`。
- long train 连续 3 个 epoch 没有超过 Run225 `0.5431 / 0.4226`。
- selector 的提升主要来自验证集阈值回放，而不是训练出的 logits。

满足任一条件，可继续 BUTD 长训：

- short trial 达到或超过 `0.545 / 0.424`，且 false override 不高于 detector fallback 审计。
- long train 中 `Acc@0.25` 稳定超过 `0.55`，且 `Acc@0.50` 不低于 `0.422`。
- source-choice row dump 显示 useful switch 增加，同时 harmful switch 没有同步增加。

## 论文记录方式

如果该路线有效，可以写成方法点，而不是 trick：

- 名称：Oracle-Supervised Source Choice Selector。
- 方法核心：训练期以 GT IoU 构造 source-level preference，推理期由 learned selector 决策。
- 消融必须包括：fixed best source、oracle source choice、learned source choice、without decomposition context、without precision-first target。
- 必须报告：是否使用 GT at test time = no。

如果最终只涨 `0.1-0.2` 个点，建议不要把它作为第三或第四创新点主打；可以作为通用 source arbitration module 的一个组成部分，并把主要论文贡献转到跨 backbone 迁移和困难负样本构造上。

