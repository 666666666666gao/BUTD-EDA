# 通用 SACR/RAPF/QAHNL/Source-Choice 模块实验与写作交接

日期：2026-06-28

> 2026-08-05 scope correction: this document preserves the historical UCRA/source-choice exploration. For the current EDA transfer and paper packaging, the only innovations are `SACR`, `RAPF`, and `QAHNL`. Source-Choice is an optional historical arbitration layer, not a fourth innovation. The canonical EDA experiment record and handoff is `EDA-master/reports/tuning/eda_scanrefer_sacr_rapf_qahnl.md`, which records the final `scanrefer_spacy` result `55.048 / 43.269` plus Unique/Multiple metrics.

本文档是后续论文写作和新 agent 接手项目时的主入口。当前主线不再展开旧的 `S2S/ACD/DHC` 叙事；正式包装为一个跨 backbone 的通用模块：

**Universal Compositional Reliability Arbitration, UCRA，通用组合可靠性仲裁模块。**

UCRA 的核心思想是：不同 3D visual grounding backbone 都会产生多个可部署的候选评分来源，例如语言匹配分数、结构组合分数、质量分数、mask-text 分数、检测器策略分数。训练阶段可以用 GT IoU 判断哪个评分来源在当前样本上更可靠，但推理阶段不能看 GT。历史 UCRA 探索把系统组织为三个核心创新加一个可选仲裁层：

1. **SACR**：Structured Anchor-Compositional Reasoning，用目标、属性、关系和 anchor 语义构造结构化评分来源。
2. **RAPF**：Reliability-Aware Probabilistic Fusion，用可靠性门控决定结构化评分何时能影响基础评分。
3. **QAHNL**：Quality-Aware Hard Negative Learning，用质量感知的困难负样本训练让模型区分高混淆候选。
4. **Source-Choice（历史可选工程层，不计创新点）**：训练监督的评分来源仲裁器，训练时用 GT IoU 生成 source 标签，推理时只根据模型输出选择可部署 source。

在 BUTD-DETR 上，三个核心创新和可选仲裁层都有工程实现或诊断链路；在 MCLN 上，历史实验完成了同一通用接口下的 source-choice 迁移，用 `default` 和 `mask_text` 两个可部署来源验证模块可移植性。当前 EDA 写作只包装 SACR/RAPF/QAHNL，source-choice 仅保留为历史工程背景。

## 当前结论

当前可作为主结果的最高 MCLN ScanRefer REC 指标为：

| Backbone | 模块/口径 | 任务 | Acc@0.25 | Acc@0.50 | epoch | 状态 |
|---|---|---|---:|---:|---:|---|
| MCLN | learned source-choice selector | REC | **0.57920** | **0.45877** | 70 | 当前采用结果，权重已保留 |
| MCLN | Optuna trial0000 | REC | 0.57699 | 0.46066 | 72 | 调参结果，未超过 0.57920 |
| MCLN | 旧长训后期 | REC | 0.57215 | 0.46088 | 86 | 后期回落，不作为最好结果 |
| MCLN | 旧长训最佳 Acc@0.50 日志 | REC | 0.57289 | **0.46487** | 76 | Acc@0.50 较高，但不是当前主 REC@0.25 权重 |
| MCLN | mask@kiou | RES | 0.59403 | 0.48843 | 71 | RES 指标，不能和 REC 混写 |

当前 BUTD-DETR 结果需要分三类写：

| Backbone | 模块/口径 | Acc@0.25 | Acc@0.50 | 状态 | 写作处理 |
|---|---|---:|---:|---|---|
| BUTD-DETR | Run225 documented learned selector | 0.5431 | 0.4226 | 文档记录最好 learned selector，checkpoint 缺失 | 可作为 documented reference，不能说已新鲜复现 |
| BUTD-DETR | best loadable detector-primary | 0.5425 | 0.4221 | 可加载强参考 | 可作为可复验参考 |
| BUTD-DETR | Run331 detector fallback | 0.5397 | 0.4205 | fallback baseline | 用来对比 selector 是否有害切换 |
| BUTD-DETR | Run331 selector_choice | 0.5369 | 0.4171 | 可部署 selector 低于 fallback | 负结果，说明 source-choice separability 仍需优化 |
| BUTD-DETR | Run331 oracle over sources | 0.5715 | 0.4575 | 诊断上界 | 只能说明 headroom，不能当最终结果 |
| BUTD-DETR | two-stage full fused primary, best @0.50 | 0.462663 | 0.333403 | SACR/RAPF/QAHNL full 诊断 | 作为融合失败诊断，不放主表 |
| BUTD-DETR | two-stage full fused primary, best @0.25 | 0.470551 | 0.333088 | SACR/RAPF/QAHNL full 诊断 | 作为融合失败诊断，不放主表 |
| BUTD-DETR | same checkpoint quality diagnostic, epoch 50 | 0.462663 primary | 0.372949 quality@0.50 | 诊断评分强于 fused | 支撑“质量源有效、融合需要仲裁” |

主论文口径建议：MCLN 的 `0.57920 / 0.45877` 是当前最稳主结果；BUTD-DETR 提供完整方法构件、诊断证据和 documented source-choice reference。Source-choice 的 oracle 上界和 BUTD 负结果应作为“为什么需要可靠性仲裁和保守切换”的论证材料，而不是夸大成最终部署性能。

## 结果来源与保留权重

MCLN 当前采用结果：

```text
log:
/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_full_joint_restart_save1_keep3_seed0_20260620_110332/1781953653/log.txt

best checkpoint:
/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_full_joint_restart_save1_keep3_seed0_20260620_110332/1781953653/best_rec_acc025_epoch70.pth

preserved checkpoint:
/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_source_choice/current_best_rec_acc025_epoch70_0.57920.pth

metadata:
/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_source_choice/current_best_rec_acc025_epoch70_0.57920.json
```

epoch 70 的关键日志行：

```text
last_ position alignment Acc0.25: Top-1: 0.57920, Top-5: 0.65303, Top-10: 0.68006
last_ position alignment Acc0.50: Top-1: 0.45877, Top-5: 0.56121, Top-10: 0.59497
fixed_default Acc0.25 Top-1: 0.57920, Acc0.50 Top-1: 0.45877
fixed_mask_text Acc0.25 Top-1: 0.01115, Acc0.50 Top-1: 0.00841
learned_selector Acc0.25 Top-1: 0.57920, Acc0.50 Top-1: 0.45877
oracle Acc0.25 Top-1: 0.58004, Acc0.50 Top-1: 0.45993
```

重要解释：

- 论文主结果使用 `learned_selector` 行的 `0.57920 / 0.45877`。
- 同一 epoch 中 `learned_selector` 与 `fixed_default` 相等，说明当前最好点的 selector 基本选择 default；不能声称 selector 在 epoch 70 明显超过 default。
- `oracle 0.58004 / 0.45993` 是同源集合的诊断上界，不是推理结果。
- RES 的 `mask@kiou overall25/overall50` 是 segmentation/mask 口径，不是 REC；写表时必须单独列任务。

MCLN Optuna trial0000：

```text
report:
/root/autodl-tmp/DATA_ROOT/output/tuning/mcln_source_choice_continue_optuna20_20260628_005200/reports/best.json

best trial checkpoint:
/root/autodl-tmp/DATA_ROOT/output/tuning/mcln_source_choice_continue_optuna20_20260628_005200/logs/scanrefer/trial_0000/1782579127/best_trial_acc025_epoch72.pth
```

trial0000 指标为 `0.57699 / 0.46066`，未超过当前采用的 `0.57920 / 0.45877`。该 trial 还暴露过 Python 3.7 的 `Path.unlink(missing_ok=...)` 兼容问题，修复在：

```text
MCLN-main/scripts/tuning/optuna_mcln_source_choice_continue.py
```

## 任务口径

REC 和 RES 必须分开：

| 任务 | 主指标 | 代码/日志表现 | 当前写法 |
|---|---|---|---|
| REC, referring expression comprehension | `Acc@0.25`, `Acc@0.50` | `position alignment`, `semantic alignment`, `fixed_default`, `learned_selector` | 主表优先写 REC |
| RES, referring expression segmentation | `mask@kiou overall25`, `mask@kiou overall50` | `mask@kiou` | 单独放附表或补充结果 |

当前论文主线先看 REC。MCLN 的 `mask@kiou overall25=0.59403` 很高，但它是 RES，不应被写成 REC 的 `Acc@0.25`。

## 通用模块定义

### 统一接口

UCRA 不依赖某个特定 backbone。每个 backbone 只需要通过 adapter 暴露以下字段：

```python
{
    "candidate_boxes": Tensor[B, Q, 6],
    "candidate_feats": Tensor[B, Q, D],
    "source_scores": {
        "base": Tensor[B, Q],
        "structured": Tensor[B, Q],
        "quality": Tensor[B, Q],
        "fused": Tensor[B, Q],
        "mask_text": Tensor[B, Q],
        "...": Tensor[B, Q],
    },
    "valid_mask": Tensor[B, Q],
    "text_feats": Optional[Tensor[B, T, D]],
    "meta": Optional[dict],
}
```

训练时额外使用 GT boxes 计算每个 source 的 top-1 IoU，用于监督 selector；推理时只使用上面的可部署字段。

### 统一流程

```text
Backbone outputs
  -> Adapter normalizes candidates and score sources
  -> SACR builds structured compositional scores
  -> RAPF estimates reliability and produces fused scores
  -> QAHNL trains quality/ranking behavior on hard candidate pools
  -> Source-Choice chooses one deployable score source per sample
  -> selected source ranks boxes for final grounding
```

### SACR

SACR 的目标是把语言中的目标、属性、关系、anchor 语义转为一个结构化评分来源，而不是直接替换基础 grounding 分数。

代码入口：

```text
models/sacr_head.py
models/bdetr.py
```

核心实现：

- `SACRHead` 输入 `query_feats`, `pred_boxes`, `base_scores`, `slot_dict`。
- `target_attr_mlp` 对候选 query、target slot、attribute slot、global slot 组合打分。
- `anchor_mlp` 为 relation anchor 选候选 anchor。
- `rel_pair_mlp` 对 target-anchor-relation 组合和几何特征打分。
- 输出 `structured_scores`, `target_attr_scores`, `relation_anchor_scores`。

写作解释：

SACR 将语言分解后的结构语义转为一个可排序 source。目标/属性负责“这个候选像不像被描述对象”，关系/anchor 负责“这个候选和参照物之间是否满足空间关系”。它的输出不是最终答案，而是交给 RAPF 或 source-choice 仲裁。

需要避免的说法：

- 不要说 SACR 一定提升最终指标；BUTD two-stage full 诊断显示结构分数直接融合会伤害排名。
- 不要把旧 `S2S` 名称作为当前主创新点；如果必须提，只写“早期结构语义探索被收敛为 SACR 的结构评分接口”。

### RAPF

RAPF 的目标是解决结构化分数不稳定的问题。它不盲目把 `structured_scores` 加到 base，而是先估计结构信号是否可靠。

代码入口：

```text
models/reliability_fusion.py
models/bdetr.py
```

核心实现：

- 标准化 `base_scores`, `structured_scores`, `quality_scores`。
- 构造可靠性特征，包括 base entropy、top-1 margin、base/structured top-1 disagreement、JS divergence、parse confidence、anchor entropy、anchor top1 mass、global-only/generic mask。
- `gate_mlp` 输出每个 query 的 gate。
- 根据 gate 将结构残差注入 base 或 quality-anchored score：

```text
delta = clip(norm(structured) - anchor)
fused = norm(base) + gate * delta + optional quality term
```

写作解释：

RAPF 是“可靠性控制器”：当结构解析明确且结构分数与基础分数一致时允许结构信息参与；当文本是 generic/global-only 或结构源和基础源冲突时压低 gate。这样比固定加权更符合不同句子难度和解析质量差异。

BUTD 诊断结论：

- full fused primary 在 epoch 50 的 `Acc@0.50=0.333403`，弱于同 checkpoint 的 quality diagnostic `0.372949`。
- 这说明当前 RAPF/structured residual 版本没有稳定转化为最终收益，但它提供了清楚的失败信号：质量源有价值，结构源需要更保守的可靠性仲裁。

写作时建议把 RAPF 放在方法章节，实验中诚实报告 ablation/diagnosis：简单融合不够，source-choice 是更稳的外层仲裁。

### QAHNL

QAHNL 的目标是把质量估计训练在“真正容易混淆的候选集合”上，而不是只做全局平均回归。

代码入口：

```text
models/quality_head.py
models/losses.py
models/bdetr.py
```

关键函数：

```text
_quality_losses
_quality_topk_candidate_mask
_quality_topk_rerank_losses
```

核心机制：

- `QualityHead` 预测候选 box 的 IoU/quality。
- `_quality_losses` 用 GT IoU 做质量回归和二分类。
- `_quality_topk_rerank_losses` 从指定 source 的 top-k 候选中找正样本和 hard negative。
- hard negative 需要与正样本有明确 IoU gap，再用 margin ranking 训练质量分数。

写作解释：

QAHNL 让质量分数学习“在模型自己最可能选错的一小组候选里，哪个更接近 GT”。这与 REC 任务很匹配，因为错误通常不是随机候选，而是同类物体、相邻物体或关系混淆物体。

当前证据：

- BUTD two-stage full 诊断中 quality-only ranking 在多个 checkpoint 上强于 fused primary，说明质量源具有有效 ranking 信息。
- 因此 QAHNL 可以作为 source-choice 的重要 source generator 和训练支撑，而不是孤立的辅助 loss。

### Source-Choice

Source-choice 是当前最适合包装为通用模块的部分。它把不同 scoring source 看成多个专家，然后学习“当前样本应该信任哪个专家”。

通用训练定义：

```text
For each training sample:
  For each source s:
    q_s = top1 candidate under source s
    u_s = IoU(q_s, GT)
  y = precision-first source target from {u_s}
  train selector to predict y from deployable features only

Inference:
  selected_source = argmax selector_logits
  final_scores = scores[selected_source]
  output top1(final_scores)
```

关键边界：

- GT IoU 只用于训练 target 和诊断 oracle。
- 推理时不使用 GT IoU、oracle source id、验证集后验阈值。
- 所有 source 必须是推理时可部署的模型输出。

BUTD-DETR 代码入口：

```text
models/source_pool_selector.py
models/detector_policy_sources.py
models/losses.py
models/bdetr.py
src/grounding_evaluator.py
```

MCLN 代码入口：

```text
MCLN-main/models/source_choice_adapter.py
MCLN-main/models/source_choice_selector.py
MCLN-main/models/losses.py
MCLN-main/models/mcln.py
MCLN-main/src/grounding_evaluator.py
```

MCLN 当前 source：

- `default`：从 `last_sem_cls_scores` 和 text positive maps 计算默认 grounding 分数。
- `mask_text`：从 text mask logits、query mask logits、adaptive weight 构造 mask-text 分数。

MCLN 当前 selector：

- `SourceChoiceSelector` 对每个 source 的 top candidate 提取 query feature、box、top score、margin、source embedding 和 text context。
- 输出 `selector_choice_scores`。
- 推理时根据 `selected_source_id` 选择该 source 的完整分数排序候选。

BUTD 当前 selector：

- `SourcePoolSelectorHead` 支持 candidate-aware/direct-choice/rank/pairdelta/context features。
- 可选 source 包括 `base`, `fused`, `quality`, `contrastive_base` 和 detector-policy sources。
- evaluator 中 `selector_choice`、`selector_choice_hybrid`、`selector_choice_quality_override` 是评估入口。

## 代码阅读顺序

新 agent 接手时按这个顺序读，不要从旧文档里随机找指标。

### 1. 先读本文档和结果报告

```text
docs/UNIVERSAL_SACR_RAPF_QAHNL_SOURCE_CHOICE_MODULE.md
reports/tuning/mcln_consistent_source_choice_transfer_plan.md
reports/tuning/butd_run225_source_choice_optuna_plan.md
reports/scanrefer_two_stage_full_eval_diagnosis.md
reports/tuning/optuna_scanrefer_two_stage_full_summary.md
```

注意：`docs/S2S_ACD_DHC_THREE_INNOVATIONS.md`、`docs/PAPER_FRAMEWORK.md` 是旧探索记录，不作为当前主线展开。

### 2. 读 BUTD-DETR 方法实现

```text
models/bdetr.py
```

重点看：

- 构造函数里的 `use_sacr`, `use_rapf`, `use_qahnl`, `use_source_pool_selector` 参数。
- forward 中 quality head、SACR、RAPF、detector-policy source、source-pool selector 的连接。
- end_points 如何写入 `structured_scores`, `fused_scores`, `pred_iou`, `selector_choice_scores`。

然后读：

```text
models/sacr_head.py
models/reliability_fusion.py
models/quality_head.py
models/source_pool_selector.py
models/detector_policy_sources.py
models/losses.py
src/grounding_evaluator.py
```

阅读目标：

- 搞清每个 source 的 shape 都是 `[B, Q]`。
- 搞清哪些 source 是 deployable，哪些是 oracle/diagnostic。
- 搞清 `src/grounding_evaluator.py` 里 `selector_choice` 指标怎么统计。

### 3. 读 MCLN 迁移实现

```text
MCLN-main/models/source_choice_adapter.py
MCLN-main/models/source_choice_selector.py
MCLN-main/models/mcln.py
MCLN-main/models/losses.py
MCLN-main/src/grounding_evaluator.py
MCLN-main/scripts/tuning/optuna_mcln_source_choice_continue.py
```

阅读目标：

- `source_choice_adapter.py` 如何把 MCLN 输出转成通用接口。
- `source_choice_selector.py` 如何计算训练 target 和 selector loss。
- `mcln.py` 如何接入 adapter/selector。
- `losses.py` 如何把 source-choice loss 加入总 loss。
- `grounding_evaluator.py` 如何打印 `fixed_default`, `fixed_mask_text`, `learned_selector`, `oracle`。

### 4. 查指标的命令

MCLN 主结果：

```bash
rg -n "learned_selector|fixed_default|oracle|mask@kiou|overall25|overall50|position alignment Acc0.25" \
  /root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_full_joint_restart_save1_keep3_seed0_20260620_110332/1781953653/log.txt
```

MCLN preserved metadata：

```bash
cat /root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_source_choice/current_best_rec_acc025_epoch70_0.57920.json
```

Optuna trial0000：

```bash
cat /root/autodl-tmp/DATA_ROOT/output/tuning/mcln_source_choice_continue_optuna20_20260628_005200/reports/best.json
```

BUTD 结果先读文档来源：

```bash
sed -n '1,220p' reports/tuning/butd_run225_source_choice_optuna_plan.md
sed -n '1,220p' reports/scanrefer_two_stage_full_eval_diagnosis.md
```

## 写作建议

### 推荐标题

可以从下面几类选一个：

| 风格 | 标题 |
|---|---|
| 方法主导 | Universal Compositional Reliability Arbitration for 3D Visual Grounding |
| 仲裁主导 | Learning to Choose Reliable Grounding Sources for 3D Visual Grounding |
| 结构语义主导 | Reliability-Aware Compositional Source Arbitration for 3D Visual Grounding |

### Abstract 主线

摘要建议按这个逻辑写：

1. 3D visual grounding 中同一句文本可能需要依赖目标类别、属性、空间关系、mask/quality 等不同证据。
2. 单一 score 或固定融合无法适应不同样本，结构化分数在解析错误或 generic 表达中可能伤害排名。
3. 提出 UCRA：SACR 生成结构组合 source，RAPF 做可靠性门控融合，QAHNL 增强质量源对 hard negative 的辨别，source-choice 在训练时用 GT IoU 学习选择可部署 source。
4. 在 MCLN 和 BUTD-DETR 两个 backbone 上验证该统一接口；MCLN 当前达到 `Acc@0.25=0.57920`，BUTD 诊断显示 oracle source-choice 上界到 `0.5715 / 0.4575`，说明可靠 source 仲裁存在明确 headroom。

### Contributions

建议写成 3 点，不要写成 6 个零散 trick：

1. 提出一个跨 backbone 的 UCRA 框架，把多个可部署 grounding score sources 统一为候选、source、selector 三元接口。
2. 设计结构组合与可靠性建模组件：SACR 从目标/属性/关系/anchor 产生结构 source，RAPF 用不确定性和一致性特征控制结构残差，QAHNL 强化质量源对 hard negatives 的排序能力。
3. 提出训练监督的 source-choice 仲裁策略：训练时用 GT IoU 生成 precision-first source 标签，推理时只用模型输出选择 source，并在 MCLN/BUTD-DETR 上给出迁移和诊断结果。

### Method 章节结构

建议章节：

```text
3. Method
3.1 Problem Formulation
3.2 Universal Source Interface
3.3 SACR: Structured Anchor-Compositional Reasoning
3.4 RAPF: Reliability-Aware Probabilistic Fusion
3.5 QAHNL: Quality-Aware Hard Negative Learning
3.6 Training-Supervised Source-Choice Arbitration
3.7 Instantiation on BUTD-DETR and MCLN
```

### 实验章节结构

建议章节：

```text
4. Experiments
4.1 Datasets and Metrics
4.2 Main Results on ScanRefer REC
4.3 Backbone Transfer: BUTD-DETR and MCLN
4.4 Ablation and Diagnostic Results
4.5 Oracle Headroom and Failure Analysis
4.6 RES/Mask Results as Auxiliary Evaluation
```

主表放 deployable 结果，诊断表放 oracle/quality/fused diagnosis。

### 安全表述

可以写：

- “The selector is trained with oracle source labels derived from GT IoU, while inference uses only deployable model scores.”
- “The oracle source-choice result is used only to quantify headroom.”
- “The MCLN instantiation demonstrates that the source-choice interface can be transferred by implementing a lightweight adapter.”
- “BUTD-DETR diagnostics show that quality scores contain useful ranking information, while naive structured fusion may hurt top-1 ranking.”

不要写：

- “Oracle is our final test result.”
- “MCLN source-choice improves over fixed_default at epoch 70.” 这个 epoch 的两者相等。
- “SACR/RAPF full fusion already improves BUTD final result.” 当前诊断不支持。
- “mask@kiou overall25 is REC Acc@0.25.” 这是 RES。
- “Run225 checkpoint is available and reproduced.” 当前记录是 documented reference，checkpoint 缺失。
- “S2S/ACD/DHC are the current three innovations.” 这是旧叙事。

## 推荐图示

论文里建议画一张总图：

```text
Language + 3D candidates
        |
        v
Backbone encoder/decoder
        |
        +--> base source
        +--> SACR structured source
        +--> quality source trained by QAHNL
        +--> mask/detector policy source if backbone provides it
        |
        v
RAPF optional reliability fusion
        |
        v
Source-Choice Arbitration
        |
        v
Selected deployable score source -> final REC prediction
```

图中要标清：

- GT IoU 只连到 training target，不连到 inference。
- Adapter 是 backbone-specific。
- Selector 是 generic。

## 消融与诊断怎么写

推荐把实验分成四组：

1. **source availability**：base / quality / structured / fused / mask_text / detector-policy。
2. **fusion reliability**：fixed fusion vs RAPF vs source-choice。
3. **training target**：default CE vs precision-gain focal BCE vs threshold bucket。
4. **oracle headroom**：selector vs oracle，说明还有多少可学习空间。

已有证据可以支持的诊断：

- BUTD quality diagnostic 强于 fused primary，说明质量源值得保留。
- BUTD oracle over selector sources 达到 `0.5715 / 0.4575`，说明 source 集合有 headroom。
- BUTD selector_choice 低于 detector fallback，说明不受控切换会伤害结果。
- MCLN epoch70 selector 基本回退 default，说明保守 selector 能避免明显伤害，但 mask_text source 当前太弱。
- MCLN Optuna trial0000 没有超过 0.57920，说明简单继续调 LR/selector loss 不一定突破，需要增强 source 本身或引入更有信息的 source。

## 后续实验建议

短期不建议再盲目长训。更有价值的是：

1. 在 MCLN 中增加一个更强的可部署 source，而不是继续依赖很弱的 `mask_text`。
2. 对 BUTD 的 selector 做 conservative switching：默认 source 必须强，non-default 需要足够 margin。
3. 把 QAHNL 的 candidate source 从单一 fused 改为 source pool，让 quality 直接学习“多个 source 的 top-k 混淆候选”。
4. 对 source-choice 记录 false override、useful override、target non-default ratio、selected non-default ratio，避免只看最终 Acc。
5. 所有长训都使用 `save_freq=1` 并保留 best checkpoint metadata，避免再次丢失 epoch 级最好权重。

## 磁盘和权重保留规则

必须保留：

```text
/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_source_choice/current_best_rec_acc025_epoch70_0.57920.pth
/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_source_choice/current_best_rec_acc025_epoch70_0.57920.json
```

可以清理：

- 非 best 的 trial checkpoint。
- 已确认不超过 `0.57920` 的中间 epoch 权重。
- 重复硬链接以外的大文件副本。

清理前必须先确认：

```bash
stat /root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_source_choice/current_best_rec_acc025_epoch70_0.57920.pth
cat /root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_source_choice/current_best_rec_acc025_epoch70_0.57920.json
```

## 新 agent 接手检查清单

1. 先确认当前要写的是 UCRA，即 SACR/RAPF/QAHNL/source-choice，不是旧的 S2S/ACD/DHC。
2. 先确认 MCLN 主结果是 `learned_selector Acc@0.25=0.57920, Acc@0.50=0.45877, epoch 70`。
3. 查 REC 时读 `learned_selector`、`fixed_default`、`position alignment`；查 RES 时读 `mask@kiou`。
4. 所有 oracle 指标只能写成 diagnostic upper bound。
5. BUTD Run225 `0.5431 / 0.4226` 是 documented reference；checkpoint 缺失时不能写“复现成功”。
6. BUTD full SACR/RAPF/QAHNL fused primary 是诊断/负结果，不是最终主结果。
7. 写方法时把 MCLN 和 BUTD-DETR 统一到 adapter/source/selector 接口；写实现时再区分各自文件。
8. 写实验表时区分 deployable、diagnostic、oracle、log-only、checkpoint-preserved。
