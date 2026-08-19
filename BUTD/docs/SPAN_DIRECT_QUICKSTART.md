# Span-Direct Supervision 快速开始

## 一句话总结

**在 Type Embedding 基础上，添加 Span-Direct loss 直接监督 entity/attr/rel 对齐，提升 @0.50 和 hard 场景。**

## 为什么需要这个？

### Type Embedding 的成功与不足

10 轮实验显示：
- ✅ last@0.25: 0.357 → **0.366** (+0.009)
- ✅ hard: 0.323 → **0.326** (+0.003)
- ✗ **Acc@0.50: 0.222 → 0.213 (-0.009)**

**问题**：Type Embedding 提升了语义匹配，但精定位下降。

### Span-Direct Supervision 的作用

- Type Embedding：告诉模型"这个 token 是实体/属性/关系"（输入级）
- Span-Direct：强制模型"把这些信息用到 matched query 上"（监督级）

**预期**：
- 保持 @0.25 的提升
- **回升 @0.50**（0.213 → 0.220+）
- **进一步提升 hard**（0.326 → 0.330+）

## 立即开始

### 方案 1：保守版（推荐首选）

```bash
cd /home/gb/new\ butd/butd_detr-main
bash scripts/train_sr3d_type_embed_span_direct.sh
```

**参数配置**：
- Entity span loss: 0.3
- Attribute span loss: 0.5
- Relation span loss: 0.5
- 只在最后 3 层应用

**预期效果**：
- last@0.25: 0.366 → **0.370+**
- Acc@0.50: 0.213 → **0.218+**
- hard: 0.326 → **0.330+**

### 方案 2：激进版（如果保守版有效）

```bash
bash scripts/train_sr3d_type_embed_span_direct_aggressive.sh
```

**参数配置**：
- Entity span loss: 0.5
- Attribute span loss: 1.0
- Relation span loss: 1.0

**预期效果**：
- 可能带来更大提升
- 但也可能过拟合

## 核心参数

| 参数 | 保守版 | 激进版 | 说明 |
|------|--------|--------|------|
| `--use_token_type_embed` | ✓ | ✓ | 启用 Type Embedding |
| `--use_span_contrastive_direct` | ✓ | ✓ | 启用 Span-Direct |
| `--lambda_con_ent_span` | 0.3 | 0.5 | Entity loss 权重 |
| `--lambda_con_attr_span` | 0.5 | 1.0 | Attribute loss 权重 |
| `--lambda_con_rel_span` | 0.5 | 1.0 | Relation loss 权重 |
| `--span_contrastive_last_k` | 3 | 3 | 只在最后 K 层 |

## 检查是否生效

### 方法 1：查看训练日志

```bash
tail -f /root/autodl-tmp/logs/sr3d_type_embed_span_direct/*/log.txt | grep span_con
```

应该看到：
```
loss_span_con_ent: 0.xxx
loss_span_con_attr: 0.xxx
loss_span_con_rel: 0.xxx
loss_span_contrastive: 0.xxx
```

### 方法 2：查看配置

```bash
cat /root/autodl-tmp/logs/sr3d_type_embed_span_direct/*/config.json | grep span
```

应该看到：
```json
"use_span_contrastive_direct": true,
"lambda_con_ent_span": 0.3,
"lambda_con_attr_span": 0.5,
"lambda_con_rel_span": 0.5
```

## 预期结果对比

| 方案 | last@0.25 | Acc@0.50 | hard | 说明 |
|------|-----------|---------|------|------|
| Baseline | 0.357 | 0.222 | 0.323 | 原始模型 |
| Type Embed | 0.366 | 0.213 | 0.326 | 输入级增强 |
| + Span-Direct | **0.370+** | **0.220+** | **0.330+** | 输入+监督 |

## 故障排除

### 问题：Span loss 始终为 0

**检查**：
```bash
# 1. 确认参数
grep "use_span_contrastive_direct" logs/*/config.json

# 2. 确认数据有 span 标注
python -c "
import pandas as pd
df = pd.read_csv('/root/autodl-tmp/DATA_ROOT/sr3d_spacy/train.csv')
print('Has entity_spans:', 'entity_spans' in df.columns)
print('Has attr_spans:', 'attr_spans' in df.columns)
print('Has rel_spans:', 'rel_spans' in df.columns)
"
```

### 问题：性能下降

**解决**：降低权重
```bash
--lambda_con_ent_span 0.1
--lambda_con_attr_span 0.2
--lambda_con_rel_span 0.2
```

## 消融实验（论文需要）

### 实验 1：只用 Type Embedding
```bash
# 已有结果：last@0.25 = 0.366
```

### 实验 2：只用 Span-Direct（不用 Type Embedding）
```bash
# 移除 --use_token_type_embed
# 保留 --use_span_contrastive_direct
```

### 实验 3：Type Embedding + Span-Direct（完整版）
```bash
bash scripts/train_sr3d_type_embed_span_direct.sh
```

### 实验 4：分类型消融

#### 只用 Entity
```bash
--lambda_con_ent_span 0.5
--lambda_con_attr_span 0.0
--lambda_con_rel_span 0.0
```

#### 只用 Attribute
```bash
--lambda_con_ent_span 0.0
--lambda_con_attr_span 1.0
--lambda_con_rel_span 0.0
```

#### 只用 Relation
```bash
--lambda_con_ent_span 0.0
--lambda_con_attr_span 0.0
--lambda_con_rel_span 1.0
```

## 为什么会成功

### 1. 互补性

| 方案 | 作用 | 效果 |
|------|------|------|
| Type Embedding | 告诉模型"是什么" | 提升语义匹配 |
| Span-Direct | 强制模型"怎么用" | 提升精定位 |

### 2. 理论支持

- **Multi-task Learning**：辅助任务帮助主任务
- **MDETR**：token-level 监督的有效性
- **Curriculum Learning**：从粗到细的监督

### 3. 工程安全

- 不改模型结构，只加 loss
- 代码已实现，只需调参数
- 最坏情况：持平（不会像 gate fusion 那样崩溃）

## 下一步

1. ✅ **立即开始**：运行保守版脚本
2. ⏳ **10 epoch 验证**：~2 小时
3. ⏳ **参数调优**：如果有效，测试不同权重
4. ⏳ **消融实验**：准备论文数据
5. ⏳ **完整训练**：400 epoch

## 文档索引

- **详细指南**：[SPAN_DIRECT_SUPERVISION_GUIDE.md](SPAN_DIRECT_SUPERVISION_GUIDE.md)
- **Type Embedding**：[TYPE_EMBEDDING_QUICKSTART.md](TYPE_EMBEDDING_QUICKSTART.md)
- **实现总结**：[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

**关键优势**：
1. ✅ 直接针对短板（@0.50 下降）
2. ✅ 工程风险低（只加 loss）
3. ✅ 代码已实现（只需调参数）
4. ✅ 与 Type Embedding 互补
5. ✅ 有理论支持（Multi-task + MDETR）

**立即开始训练，预计 2 小时后看到初步结果！**
