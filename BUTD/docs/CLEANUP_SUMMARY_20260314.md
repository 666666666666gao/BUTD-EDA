# 代码清理总结 - 2026-03-14

## 概述

移除 TextFeatureRouter（多路径文本特征）和 Gated Fusion 相关的所有代码路径，
只保留当前活跃的两条创新线：**Token Type Embeddings** 和 **Span-Direct Contrastive Supervision**。

---

## 一、清理范围

### 已移除的模块/概念

| 模块 | 说明 | 状态 |
|------|------|------|
| TextFeatureRouter | 多路径文本特征路由（global/entity/attr/rel 四路编码） | 已移除 |
| GatedTextFusion | 门控融合模块（2-way, type4, span-residual, adaptive） | 已移除 |
| Multi-path text encoding | 四路独立 RoBERTa 编码 + 拼接/融合 | 已移除 |

### 保留的模块

| 模块 | 说明 | 文件 |
|------|------|------|
| Token Type Embeddings | 给 tokens 添加语义类型嵌入（entity/attr/rel） | `models/bdetr.py` |
| Span-Direct Contrastive Loss | 基于 span 的细粒度对比监督 | `models/losses.py` |
| Span Warmup | span loss 线性预热 | `main_utils.py` |

---

## 二、文件修改明细

### 1. `models/bdetr.py`

**改动**：移除 TextFeatureRouter 和 GatedTextFusion 的使用代码

- 移除 `TextFeatureRouter` 实例化和调用
- 移除 `GatedTextFusion` 实例化和调用
- 移除多路径文本编码逻辑（四路 span 切分、独立编码、拼接）
- 保留构造函数中的 legacy 参数（`use_global_text`, `use_gated_text_fusion` 等），标记为 checkpoint 兼容
- 保留 `use_token_type_embed` 和 `span_aux_warmup_steps` 活跃参数

**构造函数签名（清理后）**：
```python
def __init__(self, ...,
             use_token_type_embed=False,
             token_type_embed_init='zeros',
             # Legacy params (kept for checkpoint compat, unused)
             use_global_text=True, use_entity_text=False, ...,
             span_aux_warmup_steps=0):
```

### 2. `models/bidecoder_layer.py`（如适用）

**改动**：移除 gated fusion 代码路径

### 3. `train_dist_mod.py`

**改动**：

- 移除传递给 `BeaUTyDETR` 的 TextFeatureRouter/GatedFusion 参数（共 14 个）
- 补回遗漏的 `span_aux_warmup_steps` 传参
- 更新 span 输入注释

**清理前**：
```python
model = BeaUTyDETR(
    ...,
    use_global_text=args.use_global_text,
    use_entity_text=args.use_entity_text,
    use_attr_text=args.use_attr_text,
    use_rel_text=args.use_rel_text,
    max_entity=args.max_entity,
    max_attr=args.max_attr,
    max_rel=args.max_rel,
    text_fusion=args.text_fusion,
    use_gated_text_fusion=args.use_gated_text_fusion,
    gated_text_fusion_mode=args.gated_text_fusion_mode,
    gated_text_fusion_init=args.gated_text_fusion_init,
    gated_text_fusion_log=args.gated_text_fusion_log,
    span_residual_scale=args.span_residual_scale,
    span_residual_last_k=args.span_residual_last_k,
    span_aux_warmup_steps=args.span_aux_warmup_steps,
    use_token_type_embed=args.use_token_type_embed,
    token_type_embed_init=args.token_type_embed_init
)
```

**清理后**：
```python
model = BeaUTyDETR(
    ...,
    use_token_type_embed=args.use_token_type_embed,
    token_type_embed_init=args.token_type_embed_init,
    span_aux_warmup_steps=args.span_aux_warmup_steps
)
```

### 4. `main_utils.py`

**改动**：

- 移除 gate 相关统计收集（`_accumulate_stats` 中的 `gate_` 条件）
- 移除 gate_keys 日志输出（`train_one_epoch` 中的 gate 日志块）
- 更新 span 输入注释（"multi-path text features" → "span contrastive loss"）
- argparse 中无需改动（TextFeatureRouter/GatedFusion 参数已在之前移除）

**保留的 argparse 参数组**：

```
# Token-type embeddings
--use_token_type_embed
--token_type_embed_init {zeros,small,normal}

# Span-direct contrastive loss
--use_span_contrastive_direct
--lambda_con_ent_span    (default: 0.1)
--lambda_con_attr_span   (default: 0.2)
--lambda_con_rel_span    (default: 0.2)
--span_positive_source   {span_direct, intersect}
--span_contrastive_last_k (default: 3)
--span_contrastive_weight (default: 1.0)
--span_aux_warmup_steps  (default: 10000)
--span_loss_debug
```

---

## 三、参数一致性验证

| 参数 | argparse | train_dist_mod.py | bdetr.py | 状态 |
|------|----------|-------------------|----------|------|
| `use_token_type_embed` | ✅ | ✅ 传递 | ✅ 活跃 | OK |
| `token_type_embed_init` | ✅ | ✅ 传递 | ✅ 活跃 | OK |
| `span_aux_warmup_steps` | ✅ (default=10000) | ✅ 传递 | ✅ 活跃 (default=0) | OK |
| span contrastive 系列 | ✅ | N/A (loss 层) | N/A | OK |
| TextFeatureRouter 系列 | ❌ 已移除 | ❌ 已移除 | ⚠️ Legacy | OK |
| GatedFusion 系列 | ❌ 已移除 | ❌ 已移除 | ⚠️ Legacy | OK |

---

## 四、消融实验设计

基于清理后的代码，可运行以下消融实验：

| # | 实验 | 脚本 | 关键开关 |
|---|------|------|----------|
| A | Baseline | `train_sr3d_ablation_baseline.sh` | 无额外开关 |
| B | + Type Embed only | `train_sr3d_ablation_type_embed.sh` | `--use_token_type_embed` |
| C | + Span-Direct only | `train_sr3d_ablation_span_direct.sh` | `--use_span_contrastive_direct` |
| D | + Both (conservative + warmup) | `train_sr3d_type_embed_span_conservative_warmup.sh` | 两者都开 |

---

## 五、注意事项

1. `bdetr.py` 构造函数保留 legacy 参数是为了兼容旧 checkpoint 的 `config.json`
2. 如果确认不再需要加载旧 checkpoint，可以安全移除 legacy 参数
3. `span_aux_warmup_steps` 在 argparse 中默认 10000，在模型中默认 0，务必确保传递
