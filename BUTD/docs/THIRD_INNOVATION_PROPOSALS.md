# 第三创新点：模型结构级改进方案

## 背景

当前两个创新点的问题：
- Type Embed：只加了嵌入（3K参数），太轻量
- Span-Direct：只加了loss，没改推理过程
- NR3D 最好 38.2 < baseline 38.7，SR3D 最好 50.6 仅比 baseline 50.1 高 0.5
- SOTA（TSP3D）：NR3D 48.7，SR3D 57.1

根本原因：我们分解了文本（entity/attr/rel），但 decoder 的 cross-attention 仍然把所有 tokens 混在一起处理，分解信息没有真正影响推理路径。

---

## 方向 A：Type-Decomposed Cross-Attention（类型分解交叉注意力）

### 核心思想

把 decoder 每层的语言交叉注意力从"一条路"拆成"三条专用路"。

### 现状 vs 改进

```
现状（BiDecoderLayer.cross_l）：
  query → cross_attn(query, [the, red, chair, left, of, the, table]) → output
  所有 tokens 混在一起，模型自己猜哪些重要

改进：
  query → cross_attn_ent(query, [chair, table])     → feat_ent    # "是什么"
  query → cross_attn_attr(query, [red])              → feat_attr   # "什么样的"
  query → cross_attn_rel(query, [left, of])          → feat_rel    # "在哪里"
  output = adaptive_fusion(feat_ent, feat_attr, feat_rel)
```

### 具体实现

#### 修改文件：`models/encoder_decoder_layers.py`

在 `BiDecoderLayer.__init__` 中新增三个独立的 cross-attention 头：

```python
# 现有
self.cross_l = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

# 新增三条类型专用路径
self.cross_l_ent = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
self.cross_l_attr = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
self.cross_l_rel = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

# 自适应融合模块
self.type_fusion = TypeAdaptiveFusion(d_model)
```

在 `BiDecoderLayer.forward` 中替换语言交叉注意力部分：

```python
# ---- 原始代码（第 614-622 行）----
query2 = self.cross_l(
    query=query + query_pos,
    key=lang_feats.transpose(0, 1),
    value=lang_feats.transpose(0, 1),
    key_padding_mask=text_key_padding_mask
)[0]

# ---- 替换为 ----
if use_decomposed_cross_attn and token_is_ent is not None:
    max_len = lang_feats.shape[1]

    # 构造三种类型的 key_padding_mask
    ent_mask = token_is_ent[:, :max_len].bool()
    attr_mask = token_is_attr[:, :max_len].bool()
    rel_mask = token_is_rel[:, :max_len].bool()

    ent_kpm = text_key_padding_mask | (~ent_mask)   # 只保留 entity tokens
    attr_kpm = text_key_padding_mask | (~attr_mask)  # 只保留 attr tokens
    rel_kpm = text_key_padding_mask | (~rel_mask)    # 只保留 rel tokens

    # 处理某类 tokens 为空的情况
    ent_valid = (~ent_kpm).any(dim=1)
    attr_valid = (~attr_kpm).any(dim=1)
    rel_valid = (~rel_kpm).any(dim=1)

    # 三条独立路径（共享 key/value 来源，但 mask 不同）
    feat_ent = self.cross_l_ent(
        query=query + query_pos,
        key=lang_feats.transpose(0, 1),
        value=lang_feats.transpose(0, 1),
        key_padding_mask=ent_kpm
    )[0]

    feat_attr = self.cross_l_attr(
        query=query + query_pos,
        key=lang_feats.transpose(0, 1),
        value=lang_feats.transpose(0, 1),
        key_padding_mask=attr_kpm
    )[0]

    feat_rel = self.cross_l_rel(
        query=query + query_pos,
        key=lang_feats.transpose(0, 1),
        value=lang_feats.transpose(0, 1),
        key_padding_mask=rel_kpm
    )[0]

    # 自适应融合
    query2 = self.type_fusion(
        query, feat_ent, feat_attr, feat_rel,
        ent_valid, attr_valid, rel_valid
    )
else:
    # Fallback 到原始单路径
    query2 = self.cross_l(...)[0]
```

#### 新增文件：`models/type_adaptive_fusion.py`

```python
class TypeAdaptiveFusion(nn.Module):
    """自适应融合三条类型路径的输出"""

    def __init__(self, d_model):
        super().__init__()
        # 方案 1：Query-Dependent Gating
        # 每个 query 根据自身特征动态决定三条路径的权重
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3),  # 3 个权重
        )

    def forward(self, query, feat_ent, feat_attr, feat_rel,
                ent_valid, attr_valid, rel_valid):
        """
        Args:
            query: (L, B, D) - 当前 query 特征
            feat_ent/attr/rel: (L, B, D) - 三条路径输出
            *_valid: (B,) - 该类型是否有有效 tokens
        Returns:
            fused: (L, B, D)
        """
        # 计算门控权重 (L, B, 3)
        gates = self.gate_proj(query.transpose(0, 1))  # (B, L, 3)
        gates = torch.softmax(gates, dim=-1)
        g_ent = gates[..., 0:1].transpose(0, 1)   # (L, B, 1)
        g_attr = gates[..., 1:2].transpose(0, 1)
        g_rel = gates[..., 2:3].transpose(0, 1)

        # 无效类型的权重置零
        if not ent_valid.all():
            feat_ent = torch.where(ent_valid[None, :, None], feat_ent, torch.zeros_like(feat_ent))
        if not attr_valid.all():
            feat_attr = torch.where(attr_valid[None, :, None], feat_attr, torch.zeros_like(feat_attr))
        if not rel_valid.all():
            feat_rel = torch.where(rel_valid[None, :, None], feat_rel, torch.zeros_like(feat_rel))

        fused = g_ent * feat_ent + g_attr * feat_attr + g_rel * feat_rel
        return fused
```

### 参数量分析

```
每层新增：
  cross_l_ent:  288 * 288 * 4 = 331,776  (Q/K/V/Out projections)
  cross_l_attr: 331,776
  cross_l_rel:  331,776
  gate_proj:    288 * 144 + 144 * 3 = 41,904
  小计：~1.04M

6 层 decoder 总计：~6.2M
原始模型参数量：~50M
增加比例：~12%
```

### 优势

1. 每条路径有独立的注意力权重，可以学到不同的对齐模式
2. Entity path 学习"对象是什么"
3. Attribute path 学习"对象什么样"
4. Relation path 学习"对象在哪里"
5. 门控融合让模型自适应决定每条路径的重要性

### 风险

1. 参数量增加 12%，训练时间增加
2. 某些样本可能只有 entity 没有 attr/rel，需要处理空 mask
3. 三个独立 attention 头可能过拟合

### 变体：共享权重版（轻量）

如果担心参数量，可以三条路径共享 cross-attention 权重，只用不同的 mask：

```python
# 共享同一个 cross_l，只改 mask
feat_ent = self.cross_l(query + query_pos, lang_feats, lang_feats, key_padding_mask=ent_kpm)[0]
feat_attr = self.cross_l(query + query_pos, lang_feats, lang_feats, key_padding_mask=attr_kpm)[0]
feat_rel = self.cross_l(query + query_pos, lang_feats, lang_feats, key_padding_mask=rel_kpm)[0]
```

参数量增加仅 ~42K（只有 gate_proj），但需要 3 次前向传播。

---

## 方向 B：Hierarchical Decomposed Decoding（层次化分解解码）

### 核心思想

不同 decoder 层专注处理不同类型的语义信息，模拟人类理解语言的层次化过程：
1. 先识别"是什么"（entity）
2. 再判断"什么样的"（attribute）
3. 最后推理"在哪里"（relation）

### 现状 vs 改进

```
现状（6 层 decoder，每层都一样）：
  Layer 1: cross_attn(query, ALL tokens) → 粗定位
  Layer 2: cross_attn(query, ALL tokens) → 精化
  Layer 3: cross_attn(query, ALL tokens) → 精化
  Layer 4: cross_attn(query, ALL tokens) → 精化
  Layer 5: cross_attn(query, ALL tokens) → 精化
  Layer 6: cross_attn(query, ALL tokens) → 最终预测

改进（层次化分解）：
  Layer 1-2: Entity-Focused Decoding
    cross_attn 主要关注 entity tokens（通过 soft mask 加权）
    → 粗定位：找到 "chair"、"table" 等候选对象

  Layer 3-4: Attribute-Focused Decoding
    cross_attn 主要关注 attribute tokens
    → 属性筛选：区分 "red chair" vs "blue chair"

  Layer 5-6: Relation-Focused Decoding
    cross_attn 主要关注 relation tokens
    → 空间推理：确定 "left of the table"
```

### 具体实现

#### 修改文件：`models/encoder_decoder_layers.py`

在 `BiDecoderLayer.forward` 中，根据层号调整 attention mask：

```python
def forward(self, query, vis_feats, lang_feats, query_pos,
            padding_mask, text_key_padding_mask,
            token_is_ent=None, token_is_attr=None, token_is_rel=None,
            layer_idx=0, num_layers=6,
            hierarchical_mode='soft', hierarchical_ratio=0.7,
            **kwargs):

    # ... self-attention 不变 ...

    # 层次化语言交叉注意力
    if hierarchical_mode != 'none' and token_is_ent is not None:
        max_len = lang_feats.shape[1]
        ent_mask = token_is_ent[:, :max_len].bool()
        attr_mask = token_is_attr[:, :max_len].bool()
        rel_mask = token_is_rel[:, :max_len].bool()

        # 确定当前层的阶段
        stage_size = num_layers // 3  # 每阶段 2 层
        if layer_idx < stage_size:
            focus_mask = ent_mask       # Layer 0-1: Entity
        elif layer_idx < 2 * stage_size:
            focus_mask = attr_mask      # Layer 2-3: Attribute
        else:
            focus_mask = rel_mask       # Layer 4-5: Relation

        if hierarchical_mode == 'hard':
            # 硬掩码：完全屏蔽非焦点 tokens
            modified_kpm = text_key_padding_mask | (~focus_mask)
            # 安全处理：如果焦点类型为空，fallback 到全部
            if not (~modified_kpm).any(dim=1).all():
                modified_kpm = text_key_padding_mask

        elif hierarchical_mode == 'soft':
            # 软掩码：焦点 tokens 权重更高，非焦点 tokens 权重降低
            # 通过 attention bias 实现
            attn_bias = torch.zeros(query.shape[1], query.shape[0], max_len,
                                    device=query.device)
            # 焦点 tokens 加正偏置，非焦点 tokens 加负偏置
            focus_weight = hierarchical_ratio  # 0.7
            non_focus_weight = (1.0 - focus_weight) / 2.0  # 0.15

            # 构造 bias
            attn_bias[:, :, :] = math.log(non_focus_weight + 1e-8)
            attn_bias = attn_bias.masked_fill(
                focus_mask.unsqueeze(1).expand(-1, query.shape[0], -1),
                math.log(focus_weight + 1e-8)
            )
            # 注意：需要修改 MultiheadAttention 支持 attn_bias

        query2 = self.cross_l(
            query=query + query_pos,
            key=lang_feats.transpose(0, 1),
            value=lang_feats.transpose(0, 1),
            key_padding_mask=modified_kpm  # hard mode
            # 或 attn_mask=attn_bias       # soft mode
        )[0]
    else:
        # 原始行为
        query2 = self.cross_l(query + query_pos, lang_feats.T, lang_feats.T,
                              key_padding_mask=text_key_padding_mask)[0]
```

#### 修改文件：`models/bdetr.py`

在 decoder 循环中传递 `layer_idx`：

```python
for i, layer in enumerate(self.decoder):
    query = layer(
        query, vis_feats, lang_feats, query_pos,
        padding_mask, text_key_padding_mask,
        token_is_ent=token_is_ent,
        token_is_attr=token_is_attr,
        token_is_rel=token_is_rel,
        layer_idx=i,
        num_layers=len(self.decoder),
        hierarchical_mode=self.hierarchical_mode,
        hierarchical_ratio=self.hierarchical_ratio,
    )
```

### 参数量分析

```
Hard 模式：0 额外参数（只改 mask）
Soft 模式：0 额外参数（只加 attention bias）
```

### 优势

1. 零额外参数，纯结构改进
2. 模拟人类理解过程，论文故事好讲
3. 实现简单，改动集中
4. 可以和方向 A 结合使用

### 风险

1. 硬性分层可能太死板（某些样本不需要三阶段）
2. 某些层只看部分 tokens，信息可能不够
3. 需要仔细调 soft mask 的比例

### 变体：渐进式层次化

不是硬性分三段，而是逐层渐进：

```
Layer 1: 80% entity + 10% attr + 10% rel
Layer 2: 60% entity + 20% attr + 20% rel
Layer 3: 30% entity + 40% attr + 30% rel
Layer 4: 20% entity + 40% attr + 40% rel
Layer 5: 10% entity + 20% attr + 70% rel
Layer 6: 10% entity + 10% attr + 80% rel
```

这样更平滑，避免信息断层。

---

## 两个方向的对比

| 维度 | 方向 A：Decomposed Cross-Attn | 方向 B：Hierarchical Decoding |
|------|-------------------------------|-------------------------------|
| 改动位置 | decoder 每层的 cross_l | decoder 每层的 attention mask |
| 额外参数 | ~6.2M（独立头）或 ~42K（共享头） | 0 |
| 实现复杂度 | 中等 | 简单 |
| 论文故事 | "分解后分路处理" | "分解后分层推理" |
| 预期提升 | 较大（独立参数学不同模式） | 中等（只改信息流向） |
| 训练稳定性 | 需要注意空 mask | 需要注意 soft ratio |
| 与现有创新点关系 | Type Embed 标注类型 → Decomposed Attn 分路处理 → Span-Direct 监督对齐 | Type Embed 标注类型 → Hierarchical 分层推理 → Span-Direct 监督对齐 |

---

## 推荐方案

### 首选：方向 A（共享权重版）+ 方向 B 结合

```
Layer 1-2: Entity-Focused + Decomposed Cross-Attn
  三条路径，但 entity 路径权重更高

Layer 3-4: Attribute-Focused + Decomposed Cross-Attn
  三条路径，但 attribute 路径权重更高

Layer 5-6: Relation-Focused + Decomposed Cross-Attn
  三条路径，但 relation 路径权重更高
```

这样：
- 每层都有三条路径（方向 A），保证信息完整
- 不同层有不同的焦点（方向 B），引导层次化推理
- 额外参数少（共享权重 + gate）
- 论文故事最完整

### 论文三个创新点

1. **Type Embeddings**（编码层）：告诉模型每个 token 的语义角色
2. **Type-Decomposed Hierarchical Decoding**（推理层）：分路处理 + 分层推理
3. **Span-Direct Contrastive Supervision**（监督层）：细粒度对齐监督

形成完整的"编码 → 推理 → 监督"闭环。

### 预期效果

- NR3D: 38.7 → 40-42（超过 baseline 1-3 个点）
- SR3D: 50.1 → 52-54（超过 baseline 2-4 个点）
- 特别是 Hard 样本和 Relation 相关样本应该有显著提升

---

## 实现优先级

1. 先实现方向 A 共享权重版（最快验证，改动最小）
2. 如果有效，加入方向 B 的层次化
3. 最后调参优化

## 需要修改的文件

1. `models/encoder_decoder_layers.py` - BiDecoderLayer（核心改动）
2. `models/bdetr.py` - 传递参数
3. `models/type_adaptive_fusion.py` - 新增融合模块
4. `main_utils.py` - 新增命令行参数
5. `train_dist_mod.py` - 传递参数
