# 方案 A：保守 + Warmup 实现方案

## 问题分析

### Aggressive 版本失败原因

**10 轮结果对比**：
- Baseline (Type Embedding only): last@0.25=0.366, last@0.50=0.213, hard=0.326
- Aggressive (lambda=0.5/1.0/1.0): last@0.25=0.347 (-0.019), last@0.50=0.217 (+0.004), hard=0.306 (-0.020)

**Loss 量级分析**：
```
Baseline:
  loss_constrastive_align: 30.4467 → 30.5852
  loss_span_contrastive:    0.0000  (未启用)

Aggressive:
  loss_constrastive_align: 30.6762 → 30.8923  (+0.3 相比 baseline)
  loss_span_contrastive:    2.4332 → 2.4369   (新增)
```

**核心问题**：
1. **监督信号冲突**：
   - Contrastive align: query 对齐所有 positive tokens
   - Span direct: query 只对齐特定类型的 tokens
   - 两个信号在拉扯，导致模型困惑

2. **过拟合到细粒度监督**：
   - @0.50 微升：span loss 帮助精定位
   - @0.25 下降：过度关注细节损害粗定位

3. **Lambda 权重过大**：
   - 虽然 span loss 数值小（~2.4），但梯度可能很大
   - 干扰了主任务的学习

## 方案 A：保守 + Warmup

### 核心思想

1. **降低 Lambda 权重**：让 span loss 作为"辅助信号"而不是"主导信号"
2. **添加 Warmup**：前期逐渐增加 span loss 权重，让模型先学好主任务

### 参数设置

```bash
# Lambda 权重（降低 5-10 倍）
--lambda_con_ent_span 0.1    # 原 0.5 → 0.1
--lambda_con_attr_span 0.2   # 原 1.0 → 0.2
--lambda_con_rel_span 0.2    # 原 1.0 → 0.2

# Warmup 步数
--span_aux_warmup_steps 10000  # 约 5 轮（77836 samples / 12 batch = 6486 steps/epoch）
```

### Warmup 实现

**线性 Warmup**：
```python
if global_step < warmup_steps:
    warmup_factor = global_step / warmup_steps
else:
    warmup_factor = 1.0

# 应用到 span loss
span_loss_weighted = span_loss * warmup_factor
```

**效果**：
- Epoch 1-5: span loss 权重从 0 逐渐增加到设定值
- Epoch 6+: span loss 权重保持设定值

### 预期效果

**10 轮结果预期**：
- last@0.25: 0.366 → 0.368-0.370 (+0.002-0.004)
- last@0.50: 0.213 → 0.218-0.222 (+0.005-0.009)
- hard: 0.326 → 0.330-0.334 (+0.004-0.008)

**理由**：
1. 更小的 lambda 避免监督信号冲突
2. Warmup 让模型先学好主任务，再逐渐引入辅助监督
3. 两者结合，既保留 span loss 的好处，又避免其副作用

## 实现步骤

### Step 1: 修改 SetCriterion

在 `models/losses.py` 中：

1. 添加 `global_step` 参数到 `forward()` 方法
2. 添加 `warmup_steps` 参数到 `__init__()` 方法
3. 在计算 span loss 时应用 warmup factor

### Step 2: 修改训练循环

在 `main_utils.py` 中：

1. 维护 `global_step` 计数器
2. 将 `global_step` 传递给 criterion

### Step 3: 创建训练脚本

已创建：`scripts/train_sr3d_type_embed_span_conservative_warmup.sh`

## 备选方案

如果方案 A 仍然不理想，可以尝试：

### 方案 B：Intersect 模式

```bash
--lambda_con_ent_span 0.3
--lambda_con_attr_span 0.5
--lambda_con_rel_span 0.5
--span_positive_source intersect  # 改用交集模式
```

**Intersect vs Span-Direct**：
- Span-Direct: 所有 type tokens 都是 positive
- Intersect: 只有 positive_map ∩ type_mask 的交集是 positive
- Intersect 更保守，避免监督信号冲突

### 方案 C：更激进的降权

```bash
--lambda_con_ent_span 0.05
--lambda_con_attr_span 0.1
--lambda_con_rel_span 0.1
--span_aux_warmup_steps 15000  # 更长的 warmup
```

## 实验计划

1. **训练方案 A**（保守 + Warmup）
2. **评估 10 轮 checkpoint**
3. **根据结果决定**：
   - 如果有效 → 继续训练到 400 轮
   - 如果不理想 → 尝试方案 B 或 C
   - 如果仍然下降 → 放弃 Span-Direct，只用 Type Embedding

## 成功标准

**10 轮结果**：
- ✅ last@0.25 >= 0.366 (不下降)
- ✅ last@0.50 >= 0.215 (提升 +0.002)
- ✅ hard >= 0.326 (不下降)

**最终结果（400 轮）**：
- 🎯 last@0.25 >= 0.370 (提升 +0.004)
- 🎯 last@0.50 >= 0.225 (提升 +0.012)
- 🎯 hard >= 0.335 (提升 +0.012)
