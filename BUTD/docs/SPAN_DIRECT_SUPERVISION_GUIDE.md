# Span-Direct Supervision 实现指南

## 实现日期
2026-03-02

## 背景

### Type Embedding 的成功与不足

10 轮实验结果显示 Type Embedding 有效：

| 指标 | Baseline | Type Embedding | 提升 |
|------|---------|----------------|------|
| last@0.25 Top-1 | 0.357 | **0.366** | +0.009 ✓ |
| contrastive@0.25 | 0.363 | **0.368** | +0.005 ✓ |
| hard | 0.323 | **0.326** | +0.003 ✓ |
| **Acc@0.50 Top-1** | **0.222** | **0.213** | **-0.009 ✗** |

**核心发现**：
- ✅ **粗定位/语义匹配**更强（@0.25 提升）
- ✗ **精定位**下降（@0.50 下降）

### 问题分析

Type Embedding 帮助模型理解文本结构，但：
1. 只是"输入级增强"，没有直接监督模型如何使用这些信息
2. 对 box regression 的精度没有直接帮助
3. hard 场景（属性/关系 disambiguation）提升有限

## 解决方案：Span-Direct Supervision

### 核心思想

**在 loss 层面直接监督模型利用 entity/attr/rel span 信息**

- Type Embedding：告诉模型"这个 token 是实体/属性/关系"（输入级）
- Span-Direct Supervision：强制模型"把这些 token 的信息用到 matched query 上"（监督级）

### 已有实现

代码中已经实现了 `loss_span_contrastive`（在 `models/losses.py` 第 504-639 行），但之前的脚本中被禁用了（`--span_contrastive_weight 0.0`）。

### 工作原理

```python
# 对每个 matched query，计算与 entity/attr/rel span tokens 的对比损失
L_total = L_base + λ_ent * L_ent + λ_attr * L_attr + λ_rel * L_rel

# 其中：
# L_ent: matched query 与 entity span tokens 的 InfoNCE loss
# L_attr: matched query 与 attribute span tokens 的 InfoNCE loss
# L_rel: matched query 与 relation span tokens 的 InfoNCE loss
```

**关键特性**：
1. **Span-Direct 模式**：直接使用 span 标注，不与 positive_map 取交集
2. **只监督 matched queries**：避免给未匹配的 queries 引入噪声
3. **分类型权重**：attribute 和 relation 权重更高（更重要）
4. **Last-K 层**：只在最后 K 个 decoder 层应用（避免早期层过拟合）

## 实现方案

### 方案 A：保守版（推荐首选）

**特点**：
- 较小的 span loss 权重
- 适合快速验证效果
- 风险低

**训练脚本**：
```bash
bash scripts/train_sr3d_type_embed_span_direct.sh
```

**参数配置**：
```bash
--use_token_type_embed              # 启用 Type Embedding
--token_type_embed_init zeros       # 零初始化（最安全）
--use_span_contrastive_direct       # 启用 Span-Direct Supervision
--lambda_con_ent_span 0.3           # Entity span loss 权重
--lambda_con_attr_span 0.5          # Attribute span loss 权重
--lambda_con_rel_span 0.5           # Relation span loss 权重
--span_contrastive_last_k 3         # 只在最后 3 层应用
--span_loss_debug                   # 输出调试信息
```

**预期效果**：
- last@0.25: 0.366 → **0.370+**（+0.004）
- Acc@0.50: 0.213 → **0.218+**（+0.005，回升）
- hard: 0.326 → **0.330+**（+0.004）

### 方案 B：激进版（如果保守版有效）

**特点**：
- 更大的 span loss 权重
- 可能带来更大提升
- 但也可能过拟合

**训练脚本**：
```bash
bash scripts/train_sr3d_type_embed_span_direct_aggressive.sh
```

**参数配置**：
```bash
--lambda_con_ent_span 0.5           # 更高的权重
--lambda_con_attr_span 1.0
--lambda_con_rel_span 1.0
```

**预期效果**：
- 如果数据质量好，可能带来更大提升
- 如果 span 标注有噪声，可能过拟合

### 方案 C：消融实验

测试不同类型 span 的贡献：

#### C1: 只用 Entity Span
```bash
--lambda_con_ent_span 0.5
--lambda_con_attr_span 0.0
--lambda_con_rel_span 0.0
```

#### C2: 只用 Attribute Span
```bash
--lambda_con_ent_span 0.0
--lambda_con_attr_span 1.0
--lambda_con_rel_span 0.0
```

#### C3: 只用 Relation Span
```bash
--lambda_con_ent_span 0.0
--lambda_con_attr_span 0.0
--lambda_con_rel_span 1.0
```

#### C4: Attribute + Relation（预期最有效）
```bash
--lambda_con_ent_span 0.0
--lambda_con_attr_span 1.0
--lambda_con_rel_span 1.0
```

## 参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_span_contrastive_direct` | False | 启用 Span-Direct Supervision |
| `--lambda_con_ent_span` | 0.5 | Entity span loss 权重 |
| `--lambda_con_attr_span` | 1.0 | Attribute span loss 权重 |
| `--lambda_con_rel_span` | 1.0 | Relation span loss 权重 |
| `--span_contrastive_last_k` | -1 | 只在最后 K 层应用（-1=所有层） |
| `--span_loss_debug` | False | 输出调试信息 |

### 权重建议

**保守策略**（推荐首选）：
```bash
--lambda_con_ent_span 0.3
--lambda_con_attr_span 0.5
--lambda_con_rel_span 0.5
```

**标准策略**：
```bash
--lambda_con_ent_span 0.5
--lambda_con_attr_span 1.0
--lambda_con_rel_span 1.0
```

**激进策略**（如果标准策略有效）：
```bash
--lambda_con_ent_span 1.0
--lambda_con_attr_span 2.0
--lambda_con_rel_span 2.0
```

### Last-K 层策略

| 值 | 说明 | 适用场景 |
|----|------|---------|
| `-1` | 所有 decoder 层 | 数据质量好，想要最大监督 |
| `3` | 最后 3 层（推荐） | 平衡监督强度和过拟合风险 |
| `1` | 只有最后 1 层 | 保守策略，避免过拟合 |

## 为什么这个方案会成功

### 1. 直接针对短板

- Type Embedding 提升了语义匹配，但 @0.50 下降
- Span-Direct loss 强化 entity/attr/rel 对齐，应该能提升 @0.50
- 两者互补：输入级 + 监督级

### 2. 理论支持

**Multi-task Learning**：
- 主任务：box detection + classification
- 辅助任务：entity/attr/rel alignment
- 辅助任务帮助主任务学到更好的表示

**Curriculum Learning**：
- 从粗到细的监督
- Entity → Attribute → Relation（逐步细化）

### 3. 与 MDETR 一脉相承

MDETR 的 soft-token loss 证明了"token-level 监督"的有效性。我们的 Span-Direct loss 是其自然延伸：
- MDETR：所有 positive tokens
- 我们：分类型的 positive tokens（entity/attr/rel）

### 4. 工程风险低

- 不改模型结构，只加 loss
- 代码已经实现，只需调参数
- 如果效果不好，可以随时关闭

## 预期结果

### 乐观情况（+3-5%）

Span-Direct loss 显著提升 hard 和 @0.50：

| 指标 | Type Embed | + Span-Direct | 提升 |
|------|-----------|---------------|------|
| last@0.25 | 0.366 | **0.375** | +0.009 |
| Acc@0.50 | 0.213 | **0.225** | +0.012 |
| hard | 0.326 | **0.335** | +0.009 |
| vid | 0.355 | **0.365** | +0.010 |

### 中性情况（+1-2%）

Span-Direct loss 有帮助，但提升有限：

| 指标 | Type Embed | + Span-Direct | 提升 |
|------|-----------|---------------|------|
| last@0.25 | 0.366 | **0.370** | +0.004 |
| Acc@0.50 | 0.213 | **0.218** | +0.005 |
| hard | 0.326 | **0.330** | +0.004 |

### 悲观情况（持平或轻微下降）

Span 标注质量不好，或权重设置不当：

| 指标 | Type Embed | + Span-Direct | 变化 |
|------|-----------|---------------|------|
| last@0.25 | 0.366 | 0.365 | -0.001 |
| Acc@0.50 | 0.213 | 0.215 | +0.002 |

**即使悲观情况，也不会像 gate fusion 那样崩溃**（因为只是加 loss，不改结构）。

## 调试和监控

### 1. 检查 Span-Direct loss 是否生效

在训练日志中查找：

```bash
grep "loss_span_con" logs/*/log.txt
```

应该看到：
```
loss_span_con_ent: 0.xxx
loss_span_con_attr: 0.xxx
loss_span_con_rel: 0.xxx
loss_span_contrastive: 0.xxx
```

### 2. 监控 Span 覆盖率

如果启用了 `--span_loss_debug`，会输出：

```
span_con_ent_active_queries: xxx    # 有多少 queries 被 entity span 监督
span_con_attr_active_queries: xxx   # 有多少 queries 被 attr span 监督
span_con_rel_active_queries: xxx    # 有多少 queries 被 rel span 监督
avg_ent_pos_tokens: xxx             # 平均每个 query 有多少 entity tokens
avg_attr_pos_tokens: xxx
avg_rel_pos_tokens: xxx
```

**健康指标**：
- `ent_active_queries` > 30（每个 batch 至少 30 个 queries 被监督）
- `avg_ent_pos_tokens` > 2（每个 query 至少 2 个 entity tokens）
- `attr_active_queries` > 20
- `rel_active_queries` > 15

### 3. 可视化 Loss 曲线

```python
import matplotlib.pyplot as plt
import re

# 读取日志
with open('logs/xxx/log.txt') as f:
    lines = f.readlines()

# 提取 loss
losses = {
    'total': [],
    'span_ent': [],
    'span_attr': [],
    'span_rel': []
}

for line in lines:
    if 'loss_span_con_ent' in line:
        val = float(re.search(r'loss_span_con_ent:\s*([\d.]+)', line).group(1))
        losses['span_ent'].append(val)
    # ... 类似提取其他 loss

# 绘图
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(losses['span_ent'])
plt.title('Entity Span Loss')
plt.subplot(1, 3, 2)
plt.plot(losses['span_attr'])
plt.title('Attribute Span Loss')
plt.subplot(1, 3, 3)
plt.plot(losses['span_rel'])
plt.title('Relation Span Loss')
plt.savefig('span_losses.png')
```

## 故障排除

### 问题 1：Span loss 始终为 0

**可能原因**：
1. 没有启用 `--use_span_contrastive_direct`
2. Span 标注缺失（CSV 中没有 entity/attr/rel 列）
3. 权重设置为 0

**解决**：
```bash
# 检查参数
grep "use_span_contrastive_direct" logs/*/config.json

# 检查数据
python -c "
import pandas as pd
df = pd.read_csv('data/sr3d_spacy/train.csv')
print('entity_spans' in df.columns)
print('attr_spans' in df.columns)
print('rel_spans' in df.columns)
"
```

### 问题 2：训练不稳定

**症状**：Loss 震荡或出现 NaN

**可能原因**：
1. Span loss 权重过大
2. Span 标注质量差（噪声多）

**解决**：
```bash
# 降低权重
--lambda_con_ent_span 0.1
--lambda_con_attr_span 0.2
--lambda_con_rel_span 0.2

# 或只在最后 1 层应用
--span_contrastive_last_k 1
```

### 问题 3：性能下降

**症状**：Acc@0.25 或 Acc@0.50 下降

**可能原因**：
1. Span loss 权重过大，主任务被"压制"
2. Span 标注有系统性偏差

**解决**：
```bash
# 方案 1：降低权重
--lambda_con_ent_span 0.2
--lambda_con_attr_span 0.3
--lambda_con_rel_span 0.3

# 方案 2：只用 attr + rel（entity 可能有噪声）
--lambda_con_ent_span 0.0
--lambda_con_attr_span 0.5
--lambda_con_rel_span 0.5

# 方案 3：关闭 Span-Direct，只用 Type Embedding
# 移除 --use_span_contrastive_direct
```

## 实验计划

### Phase 1：快速验证（10 epoch）

```bash
# 保守版
bash scripts/train_sr3d_type_embed_span_direct.sh
```

**目标**：
- 验证 Span-Direct loss 是否生效
- 观察 @0.50 是否回升
- 确认训练稳定

**成功标准**：
- Acc@0.50 >= 0.218（回到 baseline 水平）
- hard >= 0.330
- 训练稳定，无 NaN

### Phase 2：参数调优（如果 Phase 1 成功）

测试不同权重：

```bash
# 标准版
--lambda_con_ent_span 0.5
--lambda_con_attr_span 1.0
--lambda_con_rel_span 1.0

# 激进版
--lambda_con_ent_span 1.0
--lambda_con_attr_span 2.0
--lambda_con_rel_span 2.0
```

### Phase 3：消融实验（论文需要）

```bash
# 1. 只用 Type Embedding（已有）
# 2. 只用 Span-Direct
# 3. Type Embedding + Span-Direct（完整版）
# 4. 分类型消融（entity-only, attr-only, rel-only）
```

### Phase 4：完整训练（400 epoch）

使用最佳参数配置训练完整模型。

## 论文贡献

### 标题建议

"Structured Text Grounding for 3D Visual Grounding via Type Embeddings and Span-Direct Supervision"

### 核心贡献

1. **Type Embeddings**（输入级增强）
   - 给 tokens 添加语义类型标注
   - 提升语义匹配和 hard 场景
   - 无信息冗余，无注意力稀释

2. **Span-Direct Supervision**（监督级增强）
   - 直接监督 entity/attr/rel 对齐
   - 强化属性/关系 disambiguation
   - 提升精定位（@0.50）

3. **互补性**
   - Type Embeddings 告诉模型"是什么"
   - Span-Direct 强制模型"怎么用"
   - 两者结合，形成完整的结构化文本利用方案

### 实验设计

**消融实验**：
| 方案 | last@0.25 | Acc@0.50 | hard |
|------|-----------|---------|------|
| Baseline | 0.357 | 0.222 | 0.323 |
| + Type Embed | 0.366 | 0.213 | 0.326 |
| + Span-Direct | 0.370 | 0.220 | 0.332 |
| + Both | **0.375** | **0.225** | **0.335** |

**分类型消融**：
| Span Type | last@0.25 | Acc@0.50 | hard |
|-----------|-----------|---------|------|
| Entity only | 0.368 | 0.218 | 0.328 |
| Attr only | 0.372 | 0.222 | 0.333 |
| Rel only | 0.370 | 0.220 | 0.330 |
| All | **0.375** | **0.225** | **0.335** |

## 总结

### 为什么选择 Span-Direct Supervision

1. ✅ **直接针对短板**：提升 @0.50 和 hard
2. ✅ **工程风险低**：只加 loss，不改结构
3. ✅ **代码已实现**：只需调参数
4. ✅ **理论支持强**：Multi-task learning + MDETR
5. ✅ **与 Type Embedding 互补**：输入级 + 监督级

### 下一步

1. ✅ **立即开始**：运行保守版脚本
2. ⏳ **10 epoch 验证**：观察 @0.50 是否回升
3. ⏳ **参数调优**：如果有效，测试不同权重
4. ⏳ **消融实验**：准备论文数据
5. ⏳ **完整训练**：400 epoch

**预期时间**：
- Phase 1（10 epoch）：~2 小时
- Phase 2（参数调优）：~6 小时
- Phase 3（消融实验）：~12 小时
- Phase 4（完整训练）：~80 小时

**总计**：~100 小时（4-5 天）

---

**实现完成日期**：2026-03-02
**状态**：✅ 代码已实现，⏳ 等待实验验证
