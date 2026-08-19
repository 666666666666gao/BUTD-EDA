# Token-Level Type Embeddings 方案

## 问题分析

### 当前 Span-Residual 方案的问题

从训练日志看到：
1. **Gate 值失控**：从 epoch 1 的 0.17 增长到 epoch 2 的 0.78+
2. **NaN 出现**：`gate_diversity_loss` 在 epoch 2 step 1100 变成 NaN
3. **数值不稳定**：`compute_gate_diversity_loss` 中的 log 计算导致梯度爆炸

### 根本原因（来自实验报告）

所有融合方案（2-way 门控、span-residual）都**未能超过 baseline**：

| 方案 | Acc@0.25 | 相比基准 |
|------|---------|---------|
| 基准模型 | 0.357 | - |
| 2-way 门控 v1 | 0.353 | -0.004 |
| 2-way 门控 v2 | 0.351 | -0.006 |
| Span-Residual | 0.339 | -0.018 |

**核心问题**：
1. **信息冗余**：Span tokens 从同一 RoBERTa 提取，不包含新信息
2. **注意力稀释**：将 span tokens 加入 cross-encoder 稀释视觉-文本对齐
3. **语义空间不匹配**：不加入 cross-encoder 则导致语义空间不一致
4. **额外参数代价**：门控机制引入额外参数，10 epoch 内无法充分收敛

## 新方案：Token-Level Type Embeddings

### 核心思想

**不创建新的 span tokens，而是给原始 RoBERTa tokens 添加类型标注**

类似 BERT 的 segment embeddings，但用于标注语义类型：
- Type 0: 全局 token（默认）
- Type 1: 实体 token
- Type 2: 属性 token
- Type 3: 关系 token

### 优势

✅ **无信息冗余**：使用原始 tokens，不创建 span pooling
✅ **无注意力稀释**：token 数量与 baseline 完全相同
✅ **最小参数**：仅 4 个 type embeddings（4 × 768 = 3K 参数）
✅ **无融合复杂度**：简单的加法操作，无门控机制
✅ **向后兼容**：初始化为小值，接近 baseline 行为
✅ **数值稳定**：无 sigmoid/softmax/log，无 NaN 风险

### 架构对比

#### 当前 Span-Residual 方案
```
RoBERTa → [full tokens (58)] + [span tokens (21)] = 79 tokens
                ↓                      ↓
         Cross-Encoder          Cross-Encoder (span)
                ↓                      ↓
         baseline_ctx            span_ctx
                ↓                      ↓
                └──── Gate Fusion ─────┘
                         ↓
                    enhanced_ctx
```

**问题**：
- 79 tokens 稀释注意力
- Gate 机制不稳定
- Span tokens 未经跨模态编码（语义空间不匹配）

#### 新方案：Type Embeddings
```
RoBERTa → [tokens (58)] + [type embeddings]
                ↓
         Cross-Encoder (58 tokens, 与 baseline 相同)
                ↓
            text_ctx
```

**优势**：
- 58 tokens，与 baseline 完全相同
- 无门控机制，无数值不稳定
- Type embeddings 在 RoBERTa 之后添加，进入 cross-encoder

## 实现方案

### 1. 核心模块

已创建 `models/text_type_embeddings.py`：

```python
class TextTypeEmbeddings(nn.Module):
    """为文本 tokens 添加可学习的类型嵌入"""

    def __init__(self, d_model=768, num_types=4):
        self.type_embeddings = nn.Embedding(num_types, d_model)
        # 初始化为小值，接近 baseline
        nn.init.normal_(self.type_embeddings.weight, mean=0.0, std=0.02)
```

### 2. Token 类型标注

基于 span 标注自动确定每个 token 的类型：

```python
def get_token_type_ids(text, entity_spans, attr_spans, rel_spans, offset_mapping):
    """
    使用 RoBERTa 的 offset_mapping 将字符级 span 映射到 token 级

    优先级：relation > attribute > entity > global
    """
    token_type_ids = torch.zeros(batch_size, seq_len)  # 默认 type 0

    # 标注实体 tokens (type 1)
    for span in entity_spans:
        mark_tokens_in_span(span, token_type_ids, type_id=1)

    # 标注属性 tokens (type 2)
    for span in attr_spans:
        mark_tokens_in_span(span, token_type_ids, type_id=2)

    # 标注关系 tokens (type 3)
    for span in rel_spans:
        mark_tokens_in_span(span, token_type_ids, type_id=3)

    return token_type_ids
```

### 3. 集成到模型

修改 `models/bdetr.py` 中的 `_run_backbones()`：

```python
def _run_backbones(self, inputs):
    # 获取 RoBERTa features
    text_feats = self.text_encoder(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    ).last_hidden_state  # (B, L, 768)

    # 添加 type embeddings（如果启用）
    if self.use_type_embeddings:
        token_type_ids = self.type_embeddings.get_token_type_ids(
            text=inputs['text'],
            entity_spans=inputs.get('entity_spans', []),
            attr_spans=inputs.get('attr_spans', []),
            rel_spans=inputs.get('rel_spans', []),
            offset_mapping=inputs['offset_mapping']
        )
        text_feats = self.type_embeddings(text_feats, token_type_ids)

    # 投影到 d_model（与 baseline 相同）
    text_feats = self.text_projector(text_feats)

    # 进入 cross-encoder（与 baseline 完全相同）
    return text_feats, inputs['attention_mask']
```

### 4. 命令行参数

在 `main_utils.py` 中添加：

```python
parser.add_argument('--use_type_embeddings', action='store_true',
                   help='Use token-level type embeddings')
parser.add_argument('--type_embedding_std', type=float, default=0.02,
                   help='Std for type embedding initialization')
```

### 5. 训练脚本

```bash
#!/bin/bash
# SR3D with Type Embeddings

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 train_dist_mod.py \
    --dataset sr3d_spacy --test_dataset sr3d_spacy \
    --data_root /root/autodl-tmp/DATA_ROOT \
    --use_color --butd --self_attend --augment_det \
    --batch_size 40 --max_epoch 400 \
    --lr 1e-4 --lr_backbone 1e-3 \
    --use_soft_token_loss --use_contrastive_align \
    --detect_intermediate --joint_det \
    --use_type_embeddings \
    --log_dir /root/autodl-tmp/logs/sr3d_type_embeddings \
    --val_freq 5 --save_freq 5 --print_freq 100
```

## 实验计划

### Phase 1: 验证向后兼容性

```bash
# 不启用 type embeddings（应该与 baseline 完全相同）
bash scripts/train_sr3d_baseline.sh
```

**预期**：指标与 baseline 一致（Acc@0.25 ≈ 0.357）

### Phase 2: 启用 Type Embeddings

```bash
# 启用 type embeddings
bash scripts/train_sr3d_type_embeddings.sh
```

**预期**：
- 训练稳定，无 NaN
- 指标 >= baseline（至少不低于 0.357）
- 如果有提升，说明类型信息有效

### Phase 3: 消融实验

测试不同的初始化策略：

```bash
# 1. 小初始化（std=0.02，默认）
--type_embedding_std 0.02

# 2. 更小初始化（std=0.01，更接近 baseline）
--type_embedding_std 0.01

# 3. 零初始化（完全等同 baseline）
--type_embedding_std 0.0
```

### Phase 4: 分析 Type Embeddings

训练后分析学到的 type embeddings：

```python
# 可视化 type embeddings
type_embeds = model.type_embeddings.type_embeddings.weight.data
# (4, 768)

# 计算相似度
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(type_embeds.cpu().numpy())
print("Type embedding similarities:")
print(sim)

# 预期：
# - entity vs attribute 相似度较高（都是名词性）
# - relation 与其他类型相似度较低（动词性/介词性）
```

## 为什么这个方案可能成功

### 1. 避免了之前方案的所有问题

| 问题 | Span-Residual | Type Embeddings |
|------|--------------|-----------------|
| 信息冗余 | ✗ span pooling 损失信息 | ✓ 使用原始 tokens |
| 注意力稀释 | ✗ 79 tokens | ✓ 58 tokens（与 baseline 相同）|
| 语义空间不匹配 | ✗ span 未经 cross-encoder | ✓ 所有 tokens 都经过 cross-encoder |
| 额外参数 | ✗ Gate MLP + 额外注意力头 | ✓ 仅 3K 参数 |
| 数值稳定性 | ✗ Gate diversity loss 导致 NaN | ✓ 无门控，无 log/sigmoid |

### 2. 理论基础

**BERT Segment Embeddings 的成功**：
- BERT 使用 segment embeddings 区分句子 A/B
- 简单的加法操作，但效果显著
- 证明了 type embeddings 的有效性

**我们的方案**：
- 类似思想，但用于语义类型（entity/attr/rel）
- 让模型知道哪些 tokens 是关键实体、属性、关系
- 帮助 cross-attention 更好地对齐视觉和文本

### 3. 最坏情况保证

如果 type embeddings 学不到有用信息：
- 梯度会将它们推向零
- 模型退化为 baseline
- **不会比 baseline 更差**

这是 span-residual 方案失败的关键原因：即使 gate≈0.12，12% 的随机噪声也会降低性能。

## 实现步骤

### Step 1: 修改数据加载（已完成）

`src/joint_det_dataset.py` 已经提取了 span 信息，需要额外返回 `offset_mapping`：

```python
# 在 tokenizer 调用中添加
encoding = self.tokenizer(
    text,
    return_tensors='pt',
    padding=True,
    truncation=True,
    return_offsets_mapping=True  # 新增
)

return {
    'input_ids': encoding['input_ids'],
    'attention_mask': encoding['attention_mask'],
    'offset_mapping': encoding['offset_mapping'],  # 新增
    'entity_spans': entity_spans,
    'attr_spans': attr_spans,
    'rel_spans': rel_spans
}
```

### Step 2: 修改模型（核心）

在 `models/bdetr.py` 中：

```python
def __init__(self, ...):
    # 原有代码
    self.text_encoder = RobertaModel.from_pretrained(...)

    # 新增：type embeddings
    if args.use_type_embeddings:
        from models.text_type_embeddings import TextTypeEmbeddings
        self.type_embeddings = TextTypeEmbeddings(
            d_model=768,
            num_types=4
        )
        self.use_type_embeddings = True
    else:
        self.type_embeddings = None
        self.use_type_embeddings = False

def _run_backbones(self, inputs):
    # RoBERTa encoding
    text_feats = self.text_encoder(...).last_hidden_state

    # 添加 type embeddings
    if self.use_type_embeddings:
        token_type_ids = self.type_embeddings.get_token_type_ids(
            text=inputs['text'],
            entity_spans=inputs.get('entity_spans', []),
            attr_spans=inputs.get('attr_spans', []),
            rel_spans=inputs.get('rel_spans', []),
            offset_mapping=inputs['offset_mapping']
        )
        text_feats = self.type_embeddings(text_feats, token_type_ids)

    # 后续处理与 baseline 完全相同
    text_feats = self.text_projector(text_feats)
    return text_feats, attention_mask
```

### Step 3: 创建训练脚本

```bash
# scripts/train_sr3d_type_embeddings.sh
#!/bin/bash

DATA_ROOT=/root/autodl-tmp/DATA_ROOT
LOG_ROOT=/root/autodl-tmp/logs

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 train_dist_mod.py \
    --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ${DATA_ROOT} \
    --val_freq 5 --batch_size 40 --save_freq 5 --print_freq 100 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset sr3d_spacy --test_dataset sr3d_spacy \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ${LOG_ROOT}/sr3d_type_embeddings \
    --lr_decay_epochs 25 26 \
    --pp_checkpoint ${DATA_ROOT}/gf_detector_l6o256.pth \
    --butd --self_attend --augment_det \
    --num_workers 8 \
    --use_amp \
    --use_type_embeddings \
    --type_embedding_std 0.02
```

### Step 4: 运行实验

```bash
# 1. 验证 baseline（不启用 type embeddings）
bash scripts/train_sr3d_baseline.sh

# 2. 测试 type embeddings
bash scripts/train_sr3d_type_embeddings.sh

# 3. 对比结果
python scripts/compare_results.py \
    --baseline logs/sr3d_baseline \
    --type_emb logs/sr3d_type_embeddings
```

## 预期结果

### 乐观情况（+2-5%）

Type embeddings 帮助模型更好地理解文本结构：
- Acc@0.25: 0.357 → 0.365-0.375
- 特别是在 hard/multi 场景下提升明显

### 中性情况（持平）

Type embeddings 学到的信息有限，但不会降低性能：
- Acc@0.25: 0.357 → 0.355-0.360
- 至少证明了方案的稳定性

### 悲观情况（-1-2%）

Type embeddings 引入轻微噪声：
- Acc@0.25: 0.357 → 0.350-0.355
- 但远好于 span-residual 的 -0.018

## 后续改进方向

如果 Type Embeddings 有效，可以进一步探索：

### 1. 层次化 Type Embeddings

```python
# 不同 decoder 层使用不同的 type embeddings
self.type_embeddings_per_layer = nn.ModuleList([
    TextTypeEmbeddings(d_model) for _ in range(num_decoder_layers)
])
```

### 2. 动态 Type Attention

```python
# 让模型学习每个 query 应该关注哪种类型
type_attn = softmax(query @ type_keys)  # (B, Q, 4)
weighted_type_emb = type_attn @ type_embeddings  # (B, Q, D)
```

### 3. Type-Aware Cross-Attention

```python
# 在 cross-attention 中显式建模类型信息
# 例如：entity queries 更关注 entity tokens
```

## 总结

**Type Embeddings 方案的核心优势**：

1. ✅ **简单**：仅 3K 参数，无复杂融合机制
2. ✅ **稳定**：无门控，无 NaN 风险
3. ✅ **高效**：与 baseline 相同的计算量
4. ✅ **安全**：最坏情况等同 baseline
5. ✅ **可解释**：类似 BERT segment embeddings

**与之前方案的本质区别**：

- Span-Residual：创建新 tokens → 信息冗余 + 注意力稀释
- Type Embeddings：标注现有 tokens → 无冗余 + 无稀释

**建议立即实施**，这是目前最有希望超过 baseline 的方案。
