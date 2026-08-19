# S2S-ACD-DHC 三个创新点详解

本文档只讨论当前这套方法里真正作为“最终三创新点”的部分：

1. **S2S: Span-to-Slot Structured Decomposition**
2. **ACD: Anchor-Conditioned Compositional Decoder**
3. **DHC: Decomposition-Guided Hard Negative & Consistency Learning**

这里**不把** `token type embedding`、`type-decomposed cross-attention`、`span-direct contrastive` 视为最终三创新点本体。它们更适合作为过渡方案、增强模块或对比实验。

---

## 0. 一句话总览

这套方法的核心不是“给文本加几个类型标签”，而是把文本分解结果真正变成一个可计算的结构化推理链：

- **S2S** 负责把文本中的 `entity / attribute / relation` 解析成结构化 slot memory。
- **ACD** 负责让 decoder 在候选 query 上做“目标-属性-关系-锚点”组合推理，而不是只看一句整体文本。
- **DHC** 负责把这种分解结构变成训练监督，让模型不仅能“用结构推理”，还能“被结构监督拉住”。

可以把整体流程理解为：

```text
spacy spans
    -> S2S structured slots
    -> ACD compositional scoring
    -> DHC structured supervision
```

---

## 1. 创新点一：S2S

### 1.1 核心思想

传统 3DVG 模型即使拿到了文本，也往往还是把整句当作一串 token 做全局匹配。这样会有两个问题：

- 模型知道“句子里有 target / attribute / relation”，但不知道它们之间的结构角色。
- 文本分解结果只停留在辅助信息，不能真正进入模型的推理主链路。

S2S 的思路是：

- 从 `sr3d_spacy / nr3d_spacy` 标注里读出 `entity / attribute / relation spans`
- 把这些 span 映射到 RoBERTa token 上
- 再构造成一组显式结构槽位：
  - `global_slot`
  - `target_slot`
  - `attr_slot`
  - `rel_slots`
  - `anchor_slots`
  - `parse_confidence`

也就是说，文本不再只是“token 序列”，而是被重写成一个**结构化记忆**。

### 1.2 这个创新点解决了什么

- 它把“文本分解”从标注层提升到了模型表示层。
- 它显式保留了“目标是谁、属性是什么、关系依附哪个 anchor”的结构。
- 它为后面的 ACD 提供了可组合的结构输入，而不是只提供一堆 token mask。

### 1.3 对应代码入口

- `src/joint_det_dataset.py`
- `models/structured_slots.py`
- `models/bdetr.py`

### 1.4 代码一：从 CSV 解析 entity / attribute / relation spans

文件：`src/joint_det_dataset.py`

```python
@staticmethod
def _parse_spans_from_csv(line, headers):
    """
    Parse entity/attribute/relation spans from CSV columns.
    Returns empty lists if columns don't exist or parsing fails.
    """
    entity_spans = []
    attr_spans = []
    rel_spans = []

    try:
        if 'entities' in headers and line[headers['entities']]:
            entities = json.loads(line[headers['entities']])
            entity_spans = [
                {'start': e.get('start', 0), 'end': e.get('end', 0), 'text': e.get('text', '')}
                for e in entities if isinstance(e, dict)
            ]
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    try:
        if 'attributes' in headers and line[headers['attributes']]:
            attributes = json.loads(line[headers['attributes']])
            attr_spans = [
                {'start': a.get('start', 0), 'end': a.get('end', 0), 'text': a.get('text', '')}
                for a in attributes if isinstance(a, dict)
            ]
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    try:
        if 'relations' in headers and line[headers['relations']]:
            relations = json.loads(line[headers['relations']])
            rel_spans = [
                {'start': r.get('start', 0), 'end': r.get('end', 0),
                 'text': r.get('text', ''), 'head': r.get('head', ''), 'tail': r.get('tail', '')}
                for r in relations if isinstance(r, dict)
            ]
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    return entity_spans, attr_spans, rel_spans
```

这个步骤的意义是：把文本分解信息正式带进训练样本，而不是靠运行时临时猜。

### 1.5 代码二：把 span 构造成 structured slot memory

文件：`models/structured_slots.py`

```python
class StructuredSlotBuilder(nn.Module):
    """Build structured slot memory from parsed spans."""

    def __init__(self, d_model=288, pooling='attention', max_pairs=3):
        super().__init__()
        self.d_model = d_model
        self.pooling = pooling
        self.max_pairs = max_pairs

        if pooling == 'attention':
            self.global_attn = nn.Linear(d_model, 1)
            self.target_attn = nn.Linear(d_model, 1)
            self.attr_attn = nn.Linear(d_model, 1)
            self.rel_attn = nn.Linear(d_model, 1)
            self.anchor_attn = nn.Linear(d_model, 1)

        self.target_select = nn.Linear(d_model, d_model)

        self.confidence_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, token_feats, tokenized, entity_spans=None,
                attr_spans=None, rel_spans=None, anchor_types=None,
                anchor_ids=None, utterances=None):
        attn_mask = tokenized['attention_mask']

        global_slot = self._pool_tokens(
            token_feats, attn_mask,
            self.global_attn if self.pooling == 'attention' else None
        )

        target_slot, has_target = self._build_target_slot(
            token_feats, entity_spans, global_slot
        )
        attr_slot, num_attrs = self._build_attr_slot(token_feats, attr_spans)
        rel_slots, anchor_slots, slot_mask, num_pairs = \
            self._build_rel_anchor_pairs(
                token_feats, rel_spans, entity_spans, anchor_ids
            )

        parse_confidence = self._compute_parse_confidence(
            has_target, num_attrs, num_pairs
        )

        return {
            'global_slot': global_slot,
            'target_slot': target_slot,
            'attr_slot': attr_slot,
            'rel_slots': rel_slots,
            'anchor_slots': anchor_slots,
            'parse_confidence': parse_confidence,
            'slot_mask': slot_mask,
            'coverage_stats': {
                'has_target': has_target,
                'num_attrs': num_attrs,
                'num_pairs': num_pairs,
            }
        }
```

这个模块的关键点不是 pooling 本身，而是它把文本拆成了**不同语义职责的 slot**。  
后续 ACD 不再需要自己从整句里猜“哪个 token 是目标、哪个 token 是 anchor”。

### 1.6 代码三：relation-anchor tuple 的构造

文件：`models/structured_slots.py`

```python
def _build_rel_anchor_pairs(self, token_feats, rel_spans, entity_spans, anchor_ids):
    """Build relation-anchor tuple slots. Vectorized."""
    B, L, D = token_feats.shape
    device = token_feats.device
    K = self.max_pairs

    rel_slots = torch.zeros(B, K, D, device=device)
    anchor_slots = torch.zeros(B, K, D, device=device)
    slot_mask = torch.zeros(B, K, device=device, dtype=torch.bool)
    num_pairs = torch.zeros(B, device=device, dtype=torch.long)

    if rel_spans is None:
        return rel_slots, anchor_slots, slot_mask, num_pairs

    N_rel = min(rel_spans.shape[1], K)

    rel_masks, rel_valid = self._span_to_mask(rel_spans[:, :N_rel], L, device)
    rel_pooled = self._pool_spans(token_feats, rel_masks, self.rel_attn)

    if anchor_ids is not None and entity_spans is not None:
        aid = anchor_ids[:, :N_rel].clamp(0, entity_spans.shape[1] - 1)
        aid_exp = aid.unsqueeze(-1).expand(B, N_rel, 2)
        anc_spans = torch.gather(entity_spans, 1, aid_exp)

        anc_masks, anc_valid = self._span_to_mask(anc_spans, L, device)
        anc_pooled = self._pool_spans(token_feats, anc_masks, self.anchor_attn)

        aid_in_range = (
            (anchor_ids[:, :N_rel] >= 0) &
            (anchor_ids[:, :N_rel] < entity_spans.shape[1])
        )
        pair_valid = rel_valid & anc_valid & aid_in_range
    else:
        anc_pooled = torch.zeros(B, N_rel, D, device=device)
        pair_valid = torch.zeros(B, N_rel, device=device, dtype=torch.bool)

    rel_slots[:, :N_rel] = rel_pooled * pair_valid.unsqueeze(-1).float()
    anchor_slots[:, :N_rel] = anc_pooled * pair_valid.unsqueeze(-1).float()
    slot_mask[:, :N_rel] = pair_valid
    num_pairs = pair_valid.long().sum(dim=1)

    return rel_slots, anchor_slots, slot_mask, num_pairs
```

这是 S2S 最关键的一步：  
它不是只保留 relation span，而是把 relation 和它指向的 anchor entity 绑成了一组 tuple，这为 ACD 做 anchor-conditioned 推理提供了基础。

### 1.7 代码四：在主模型 forward 中接入 structured slots

文件：`models/bdetr.py`

```python
if self.structured_slot_builder is not None:
    device = inputs['point_clouds'].device
    entity_spans_tensor = build_token_span_tensors(
        tokenized, entity_spans or [[] for _ in range(tokenized['input_ids'].shape[0])], device
    )
    attr_spans_tensor = build_token_span_tensors(
        tokenized, attr_spans or [[] for _ in range(tokenized['input_ids'].shape[0])], device
    )
    rel_spans_tensor = build_token_span_tensors(
        tokenized, rel_spans or [[] for _ in range(tokenized['input_ids'].shape[0])], device
    )
    anchor_ids = inputs.get('anchor_ids', None)

    slot_dict = self.structured_slot_builder(
        token_feats=end_points['text_feats'],
        tokenized=tokenized,
        entity_spans=entity_spans_tensor,
        attr_spans=attr_spans_tensor,
        rel_spans=rel_spans_tensor,
        anchor_types=inputs.get('anchors', None),
        anchor_ids=anchor_ids,
        utterances=inputs.get('text', None)
    )
    end_points['slot_dict'] = slot_dict
```

这说明 S2S 不是离线预处理，而是**前向图的一部分**。  
也就是说，结构化文本表示是直接参与网络推理的。

---

## 2. 创新点二：ACD

### 2.1 核心思想

S2S 解决的是“把句子拆开”。  
ACD 解决的是“拆开以后，怎么真的做组合推理”。

传统 grounding 往往是：

- query 和整句文本对齐
- 输出一个 query score

ACD 做的是一个更细粒度的多阶段组合打分：

1. 先算一个 baseline grounding score
2. 再引入 `target + attribute` 的 coarse score
3. 用它挑出 top-M target candidates
4. 再对每个 relation-anchor tuple 做 anchor-conditioned reasoning
5. 最后把 baseline、EA、structured score 融合成 `acd_final_scores`

所以 ACD 的本质不是“又加了一个 MLP”，而是把 grounding score 从**单路匹配**改成了**组合式结构打分**。

### 2.2 这个创新点解决了什么

- 它让模型在 query 上显式区分“目标候选筛选”和“关系约束重排”。
- 它把 anchor 引入到了 query 级别的推理里。
- 它支持几何关系编码、置信度融合、warmup 残差融合，以及你现在在试的 `pool_ea_multiplier / final_ea_multiplier`。

### 2.3 对应代码入口

- `models/acd_head.py`
- `models/bdetr.py`
- `main_utils.py`
- `train_dist_mod.py`

### 2.4 代码一：ACD Head 的定义

文件：`models/acd_head.py`

```python
class LateACDHead(nn.Module):
    """Late anchor-conditioned reasoning head."""

    def __init__(self, d_model=288, geo_dim=16, hidden_dim=288,
                 top_m_targets=32, top_k_anchors=16,
                 use_confidence_fusion=False,
                 global_residual_alpha=0.5,
                 warmup_steps=5000,
                 initial_alpha=0.05,
                 ea_scale=1.0,
                 pool_ea_multiplier=1.0,
                 final_ea_multiplier=1.0,
                 proj_dim=64):
        super().__init__()
        self.top_m_targets = top_m_targets
        self.top_k_anchors = top_k_anchors
        self.use_confidence_fusion = use_confidence_fusion
        self.global_residual_alpha = global_residual_alpha
        self.warmup_steps = warmup_steps
        self.initial_alpha = initial_alpha
        self.ea_scale = ea_scale
        self.pool_ea_multiplier = pool_ea_multiplier
        self.final_ea_multiplier = final_ea_multiplier

        self.base_score_attn = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 1)
        )
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

        self.target_attr_mlp = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.anchor_mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        rel_input_dim = d_model * 3 + (geo_dim if geo_dim > 0 else 0)
        self.rel_pair_mlp = nn.Sequential(
            nn.Linear(rel_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
```

这个定义里已经能看出 ACD 的结构性：

- `target_attr_mlp` 管目标和属性
- `anchor_mlp` 管 anchor 候选
- `rel_pair_mlp` 管 relation pair score
- `pool_ea_multiplier / final_ea_multiplier` 控制 EA 分数在候选筛选和最终融合中的不同作用

### 2.5 代码二：ACD 的主打分流程

文件：`models/acd_head.py`

```python
def forward(self, query_feats, pred_boxes, base_scores, slot_dict,
            end_points=None, global_step=None):
    with torch.cuda.amp.autocast(enabled=False):
        query_feats = query_feats.float()
        pred_boxes = pred_boxes.float()
        base_scores = base_scores.float()

        target_slot = slot_dict['target_slot'].float()
        attr_slot = slot_dict['attr_slot'].float()
        rel_slots = slot_dict['rel_slots'].float()
        anchor_slots = slot_dict['anchor_slots'].float()
        slot_mask = slot_dict['slot_mask']

        # Step A: target-attribute coarse score
        s_ea = self._compute_target_attr_score(
            query_feats, target_slot, attr_slot
        )
        scaled_s_ea = self.ea_scale * s_ea

        # Step B: candidate pool selection
        combined_scores = base_scores + self.pool_ea_multiplier * scaled_s_ea
        M = min(self.top_m_targets, query_feats.shape[1])
        top_m_indices = torch.topk(combined_scores, M, dim=1).indices

        # Step C & D: relation-anchor reasoning
        s_struct = self._compute_structured_score_batched(
            query_feats, pred_boxes, top_m_indices,
            rel_slots, anchor_slots, slot_mask
        )

        # Step E: residual fusion
        alpha = self._get_warmup_alpha(global_step, self.global_residual_alpha)
        final_scores = (
            base_scores
            + self.final_ea_multiplier * scaled_s_ea
            + alpha * s_struct
        )

        return {
            'structured_scores': s_struct,
            'final_scores': final_scores,
        }
```

这是整套方法最像“论文主创新”的地方。  
因为这里已经不是一个普通注意力模块，而是**围绕 target / attr / rel / anchor 四类语义组件重写了 query 排序逻辑**。

### 2.6 代码三：anchor-conditioned relation scoring

文件：`models/acd_head.py`

```python
def _compute_structured_score_batched(self, query_feats, pred_boxes, top_m_indices,
                                       rel_slots, anchor_slots, slot_mask,
                                       return_anchor_stats=False):
    B, Q, D = query_feats.shape
    K = rel_slots.shape[1]
    M = top_m_indices.shape[1]
    K_anc = min(self.top_k_anchors, Q)

    anc_expanded = anchor_slots.unsqueeze(2).expand(B, K, Q, D)
    q_expanded = query_feats.unsqueeze(1).expand(B, K, Q, D)
    anchor_input = torch.cat([q_expanded, anc_expanded], dim=-1)
    anchor_scores_all = self.anchor_mlp(
        anchor_input.reshape(B * K * Q, 2 * D)
    ).reshape(B, K, Q)

    top_anc_indices = torch.topk(anchor_scores_all, K_anc, dim=2).indices
    top_anc_scores = torch.gather(anchor_scores_all, 2, top_anc_indices)
    p_anchor = F.softmax(top_anc_scores, dim=2)

    ...

    rel_input = torch.cat(rel_inputs, dim=-1)
    rel_scores = self.rel_pair_mlp(flat_input).reshape(B, K, M, K_anc)

    weighted = (rel_scores * p_anchor.unsqueeze(2)).sum(dim=3)
    mask_f = slot_mask.float().unsqueeze(2)
    s_struct_m = (weighted * mask_f).sum(dim=1)

    one_hot = torch.zeros(B, M, Q, device=query_feats.device)
    one_hot.scatter_(2, top_m_indices.unsqueeze(2), 1.0)
    s_struct = torch.einsum('bm,bmq->bq', s_struct_m, one_hot)
```

这里的关键不是“top-k”本身，而是：

- ACD 先为每个 relation tuple 找 anchor 候选
- 再在 target pool 上做 relation-pair 打分
- 最后再 scatter 回全 query 空间

这是一条真正的 **anchor-conditioned compositional reasoning path**。

### 2.7 代码四：把 ACD 接到主模型最后一层

文件：`models/bdetr.py`

```python
# Apply late ACD head if enabled
if self.acd_head is not None and 'slot_dict' in end_points:
    last_query = query
    end_points['last_queries'] = last_query
    last_boxes = torch.cat([base_xyz, base_size], dim=-1)

    proj_tokens = end_points.get('proj_tokens', None)
    proj_queries = end_points.get('last_proj_queries', None)
    assert proj_tokens is not None and proj_queries is not None

    last_base_scores = self.acd_head.compute_base_scores(
        proj_queries, proj_tokens
    )

    acd_out = self.acd_head(
        query_feats=last_query,
        pred_boxes=last_boxes,
        base_scores=last_base_scores,
        slot_dict=end_points['slot_dict'],
        end_points=end_points if self.structured_debug else None,
        global_step=getattr(self, '_acd_global_step', None)
    )

    end_points['acd_structured_scores'] = acd_out['structured_scores']
    end_points['acd_final_scores'] = acd_out['final_scores']
```

这段代码说明 ACD 不是替代基础 decoder，而是**接在最终 query 上做 structured re-ranking**。  
这也让它在工程上更稳，方便和 baseline 对照。

---

## 3. 创新点三：DHC

### 3.1 核心思想

如果只有 S2S + ACD，模型虽然有了结构推理能力，但训练监督仍然主要来自原始 grounding loss。  
这样会导致一个问题：

- 模型知道怎样“用结构算分”
- 但没有足够强的机制保证它真的把结构信息学扎实

DHC 的思想是：  
既然我们已经把文本拆成了 `target / attr / rel-anchor`，那就应该基于这种分解结果去构造监督：

- 对 target slot 做 entity hard negative
- 对 attr slot 做 attribute hard negative
- 对 rel slot 做 relation hard negative
- 对 ACD final score 和 baseline score 做 consistency regularization

所以 DHC 的定位是：  
把“文本分解”从推理层继续推进到监督层。

### 3.2 这个创新点解决了什么

- 它让结构信息参与 loss，而不是只参与 forward。
- 它提供了更难、更有针对性的 hard negative。
- 它约束 ACD 分数不要完全漂离基础 grounding 分布。

### 3.3 对应代码入口

- `models/structured_losses.py`
- `models/losses.py`
- `models/bdetr.py`

### 3.4 代码一：可学习 margin 和 temperature

文件：`models/structured_losses.py`

```python
class DHCLossModule(nn.Module):
    """Learned parameters for DHC losses."""

    def __init__(self):
        super().__init__()
        self.log_margin_entity = nn.Parameter(torch.tensor(-1.6))
        self.log_margin_attr = nn.Parameter(torch.tensor(-1.9))
        self.log_margin_rel = nn.Parameter(torch.tensor(-1.9))
        self.log_margin_acd_rank = nn.Parameter(torch.tensor(-0.7))
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def margin_entity(self):
        return F.softplus(self.log_margin_entity)

    def margin_attr(self):
        return F.softplus(self.log_margin_attr)

    def margin_rel(self):
        return F.softplus(self.log_margin_rel)

    def margin_acd_rank(self):
        return F.softplus(self.log_margin_acd_rank)

    def temperature(self):
        return self.log_temperature.exp() + 0.01
```

这里不是把 margin 写死，而是做成可学习参数。  
这使 DHC 不是“硬编码规则”，而是一个可适配数据分布的 structured supervision 包。

### 3.5 代码二：consistency loss

文件：`models/structured_losses.py`

```python
def loss_dhc_consistency(end_points, weight=0.2):
    """Global-structured consistency loss. Detached contrastive base."""
    if 'acd_final_scores' not in end_points or 'dhc_temperature' not in end_points:
        return torch.tensor(0.0, device=end_points['seed_xyz'].device)

    proj_tokens = end_points.get('proj_tokens', None)
    proj_queries = end_points.get('last_proj_queries', None)
    if proj_tokens is None or proj_queries is None:
        return torch.tensor(0.0, device=end_points['seed_xyz'].device)

    proj_queries_d = proj_queries.detach()
    proj_tokens_d = proj_tokens.detach()

    temperature = end_points['dhc_temperature']
    sim = torch.matmul(proj_queries_d, proj_tokens_d.transpose(-1, -2)) / temperature
    base_scores = sim.logsumexp(dim=-1)  # (B, Q)
    struct_scores = end_points['acd_final_scores']  # (B, Q)

    kl_loss = F.kl_div(
        F.log_softmax(struct_scores, dim=-1),
        F.softmax(base_scores, dim=-1),
        reduction='batchmean'
    )
    return weight * kl_loss
```

这部分的含义是：

- baseline grounding 分布代表模型的原始语义匹配能力
- ACD final score 代表结构推理后的分布
- DHC consistency 强迫两者保持一致性，而不是完全分叉

### 3.6 代码三：entity / attribute / relation hard negative

文件：`models/structured_losses.py`

```python
def loss_dhc_entity_hardneg(end_points, indices, weight=0.2):
    slot_dict = end_points['slot_dict']
    target_slot = slot_dict['target_slot']      # (B, D)
    query_feats = end_points['last_queries']    # (B, Q, D)

    target_expanded = target_slot.unsqueeze(1).expand(B, Q, D)
    similarities = F.cosine_similarity(query_feats, target_expanded, dim=-1)

    pos_mask, valid = _build_pos_mask(indices, B, Q, query_feats.device)
    margin = end_points['dhc_margin_entity']
    loss = _batched_hardneg_contrastive(similarities, pos_mask, valid, margin)
    return weight * loss


def loss_dhc_attribute_hardneg(end_points, indices, weight=0.2):
    attr_slot = end_points['slot_dict']['attr_slot']
    num_attrs = end_points['slot_dict']['coverage_stats']['num_attrs']
    query_feats = end_points['last_queries']

    attr_expanded = attr_slot.unsqueeze(1).expand(B, Q, D)
    similarities = F.cosine_similarity(query_feats, attr_expanded, dim=-1)

    pos_mask, valid = _build_pos_mask(indices, B, Q, query_feats.device)
    valid = valid & (num_attrs > 0)
    margin = end_points['dhc_margin_attr']
    loss = _batched_hardneg_contrastive(similarities, pos_mask, valid, margin)
    return weight * loss


def loss_dhc_relation_hardneg(end_points, indices, weight=0.2):
    rel_slots = end_points['slot_dict']['rel_slots']   # (B, K, D)
    slot_mask = end_points['slot_dict']['slot_mask']   # (B, K)
    query_feats = end_points['last_queries']           # (B, Q, D)

    sims = F.cosine_similarity(
        query_feats.unsqueeze(1).expand(B, K, Q, D),
        rel_slots.unsqueeze(2).expand(B, K, Q, D),
        dim=-1
    )  # (B, K, Q)

    ...
```

这三类 hard negative 都不是随机加出来的，而是由 S2S 分解结果决定的。  
因此它们比普通 negative 更“结构敏感”。

### 3.7 代码四：把 DHC 和 ACD ranking loss 接入总 loss

文件：`models/losses.py`

```python
# ACD ranking loss: trains ACD head even without DHC.
loss_acd_rank = torch.tensor(0.0, device=gt_center.device)
if 'acd_final_scores' in end_points and last_indices is not None and 'dhc_margin_acd_rank' in end_points:
    acd_scores = end_points['acd_final_scores']
    margin = end_points['dhc_margin_acd_rank']
    pos_mask, valid = _build_pos_mask(last_indices, B_acd, Q_acd, acd_scores.device)
    ...
    loss_acd_rank = per_batch.sum() / max(B_acd, 1)
end_points['loss_acd_rank'] = loss_acd_rank

# DHC losses
loss_dhc = 0
if use_dhc and dhc_config is not None and last_indices is not None and 'dhc_margin_entity' in end_points:
    dhc_losses = compute_dhc_losses(end_points, last_indices, dhc_config)
    for k, v in dhc_losses.items():
        end_points[k] = v
        loss_dhc += v

loss = (
    8 * query_points_generation_loss
    + 1.0 / (num_decoder_layers + 1) * (
        loss_ce
        + 5 * loss_bbox
        + loss_giou
        + loss_contrastive_align
        + effective_span_contrastive_weight * loss_span_contrastive
    )
    + loss_acd_rank
    + loss_dhc
)
```

这一步很重要，因为它说明：

- ACD 本身有 ranking supervision
- DHC 是一个额外叠加的 structured supervision 包
- 整套系统形成了“推理路径 + 监督路径”的闭环

### 3.8 代码五：把 DHC 参数从 forward 导出到 loss

文件：`models/bdetr.py`

```python
if self.dhc_loss_module is not None:
    end_points['dhc_margin_entity'] = self.dhc_loss_module.margin_entity()
    end_points['dhc_margin_attr'] = self.dhc_loss_module.margin_attr()
    end_points['dhc_margin_rel'] = self.dhc_loss_module.margin_rel()
    end_points['dhc_margin_acd_rank'] = self.dhc_loss_module.margin_acd_rank()
    end_points['dhc_temperature'] = self.dhc_loss_module.temperature()
```

这一步的工程意义是避免 DDP 图外参数访问问题，同时保证 DHC 的学习参数真正属于模型前向图的一部分。

---

## 4. 三个创新点之间的关系

这三点不是并列堆模块，而是严格的前后依赖关系：

### 4.1 S2S 是输入结构化

它回答的是：

- 文本中的 target 是谁？
- attribute 是哪些？
- relation 对应哪个 anchor？

没有 S2S，后面的结构推理就没有稳定输入。

### 4.2 ACD 是推理结构化

它回答的是：

- 哪些 query 是 target 候选？
- 哪些 query 能作为 relation 的 anchor？
- 哪些 query 在 target / attribute / relation / anchor 四种因素联合下最合理？

没有 ACD，S2S 只是一种更好的文本表示，但还没有真正变成结构推理。

### 4.3 DHC 是监督结构化

它回答的是：

- 怎么让模型真的学会用这些结构信息？
- 怎么让 target / attr / rel 的区分不只停留在 forward 里？

没有 DHC，模型会“能用结构”，但不一定“学得牢结构”。

---

## 5. 对论文写法的建议

如果写成论文，建议三条贡献写成下面这种口径：

### Contribution 1

We propose a **Span-to-Slot Structured Decomposition (S2S)** module that converts parsed textual spans into structured slot memory, including target, attribute, and relation-anchor tuple representations.

### Contribution 2

We design an **Anchor-Conditioned Compositional Decoder (ACD)** that performs compositional grounding by combining target-attribute scoring, anchor-conditioned relation reasoning, and residual fusion over decoder queries.

### Contribution 3

We introduce **Decomposition-Guided Hard Negative & Consistency Learning (DHC)**, which turns structured decomposition into supervision through entity / attribute / relation hard negatives and consistency regularization.

---

## 6. 训练开关与实验入口

配置入口在 `main_utils.py`：

```python
# S2S-ACD-DHC: Structured slot / decoder flags
parser.add_argument('--use_structured_slots', action='store_true', default=False)
parser.add_argument('--use_late_acd', action='store_true', default=False)
parser.add_argument('--use_dhc', action='store_true', default=False)

# ACD settings
parser.add_argument('--acd_top_m_targets', type=int, default=32)
parser.add_argument('--acd_top_k_anchors', type=int, default=16)
parser.add_argument('--acd_geo_dim', type=int, default=16)
parser.add_argument('--acd_hidden_dim', type=int, default=288)
parser.add_argument('--acd_global_residual_alpha', type=float, default=0.5)
parser.add_argument('--acd_use_confidence_fusion', action='store_true', default=False)
parser.add_argument('--acd_warmup_steps', type=int, default=5000)
parser.add_argument('--acd_initial_alpha', type=float, default=0.05)
parser.add_argument('--acd_ea_scale', type=float, default=1.0)
parser.add_argument('--acd_pool_ea_multiplier', type=float, default=1.0)
parser.add_argument('--acd_final_ea_multiplier', type=float, default=1.0)

# DHC settings
parser.add_argument('--dhc_consistency_weight', type=float, default=0.2)
parser.add_argument('--dhc_ent_hardneg_weight', type=float, default=0.2)
parser.add_argument('--dhc_attr_hardneg_weight', type=float, default=0.2)
parser.add_argument('--dhc_rel_hardneg_weight', type=float, default=0.2)
```

模型构造入口在 `train_dist_mod.py`：

```python
model = BeaUTyDETR(
    ...
    use_structured_slots=args.use_structured_slots,
    use_late_acd=args.use_late_acd,
    slot_pooling=args.slot_pooling,
    max_rel_anchor_pairs=args.max_rel_anchor_pairs,
    acd_top_m_targets=args.acd_top_m_targets,
    acd_top_k_anchors=args.acd_top_k_anchors,
    acd_geo_dim=args.acd_geo_dim,
    acd_hidden_dim=args.acd_hidden_dim,
    acd_global_residual_alpha=args.acd_global_residual_alpha,
    acd_use_confidence_fusion=args.acd_use_confidence_fusion,
    acd_warmup_steps=args.acd_warmup_steps,
    acd_initial_alpha=args.acd_initial_alpha,
    acd_ea_scale=args.acd_ea_scale,
    acd_pool_ea_multiplier=args.acd_pool_ea_multiplier,
    acd_final_ea_multiplier=args.acd_final_ea_multiplier,
    structured_debug=args.structured_debug
)
```

完整方法的训练脚本示例：

文件：`scripts/block5_s2s_acd_dhc_sr3d.sh`

```bash
CUDA_VISIBLE_DEVICES=0 "${DIST_LAUNCH[@]}" \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --dataset sr3d_spacy --test_dataset sr3d_spacy \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --butd --self_attend --augment_det \
    --use_amp \
    --use_structured_slots \
    --use_late_acd \
    --use_dhc \
    --slot_pooling attention \
    --max_rel_anchor_pairs 3 \
    --acd_top_m_targets 32 \
    --acd_top_k_anchors 16 \
    --acd_use_confidence_fusion \
    --acd_ea_scale 1.0 \
    --dhc_consistency_weight 0.2 \
    --dhc_ent_hardneg_weight 0.2 \
    --dhc_attr_hardneg_weight 0.2 \
    --dhc_rel_hardneg_weight 0.2
```

---

## 7. 方法图说明

这一节的目的不是再解释代码，而是告诉你论文方法图应该怎么画、每个框里写什么、箭头表达什么。

### 7.1 推荐的整图布局

建议把方法图画成三条横向主链，和三创新点严格对应：

```text
Input Layer
├── Point Cloud + Detector Proposals
└── Utterance + Parsed Spans(entity / attr / relation)

Stage A: S2S
Utterance -> RoBERTa -> token features
Parsed spans -> span pooling -> target / attr / relation-anchor slots

Stage B: ACD
decoder queries + proposal boxes
    + target slot / attr slot / relation slots / anchor slots
    -> target-attr coarse scoring
    -> top-M target pool
    -> top-K anchor selection per relation tuple
    -> relation-anchor compositional scoring
    -> final structured grounding score

Stage C: DHC
matched positive queries
    + target / attr / relation slots
    -> entity hard negative
    -> attribute hard negative
    -> relation hard negative
    -> consistency regularization
```

如果画成论文图，推荐分成 4 个 panel：

- `(a) Input Parsing`
- `(b) Span-to-Slot Structured Decomposition`
- `(c) Anchor-Conditioned Compositional Decoder`
- `(d) Decomposition-Guided Structured Supervision`

### 7.2 每个模块框建议写什么

#### Panel (a): Input Parsing

左边画 3D 场景输入：

- `Point Cloud`
- `Detector Proposals`
- `Decoder Queries`

右边画文本输入：

- `Utterance`
- `Entity Spans`
- `Attribute Spans`
- `Relation Spans`
- `Anchor IDs`

这里想表达的是：我们不是从纯文本临时猜结构，而是显式拿到分解结果。

#### Panel (b): S2S

这一块建议画成：

```text
RoBERTa Token Features
    -> Global Pooling -> global slot
    -> Entity Span Pooling + Target Selection -> target slot
    -> Attribute Span Pooling -> attr slot
    -> Relation Span Pooling + Anchor Span Lookup -> relation-anchor tuples
    -> Coverage Statistics -> parse confidence
```

这一块的关键词建议直接写在框里：

- `Structured Slot Memory`
- `Target Slot`
- `Attribute Slot`
- `Relation-Anchor Tuple Slots`

核心强调点：

- 不是 token mask
- 是显式 slot memory
- relation 和 anchor 是绑定的 tuple

#### Panel (c): ACD

ACD 这部分建议画成从左到右的分阶段漏斗：

```text
Query Features
    + target slot + attr slot
    -> Target-Attribute Coarse Score
    -> Candidate Target Pool (Top-M)

Relation Slots + Anchor Slots
    -> Anchor Scoring
    -> Anchor Distribution (Top-K)

Target Pool + Anchor Pool + Geometry
    -> Relation-Pair Scoring
    -> Structured Score

Base Score + EA Score + Structured Score
    -> Residual Fusion
    -> Final Grounding Score
```

建议在图里把这两个超参数单独标出来：

- `pool_ea_multiplier`
- `final_ea_multiplier`

因为它们正好说明：

- EA 分数在候选筛选阶段怎么用
- EA 分数在最终融合阶段怎么用

#### Panel (d): DHC

这一块建议画成一个监督分支，从 `final queries` 和 `structured slots` 分出去：

```text
Last-layer Queries + Hungarian Matches
    + target slot -> Entity Hard Negative
    + attr slot   -> Attribute Hard Negative
    + rel slots   -> Relation Hard Negative

Base Grounding Distribution
    vs
Structured Final Distribution
    -> Consistency Loss
```

图形上最好把 DHC 画成虚线框或下支路，表达它主要属于训练监督而不是推理主干。

### 7.3 推荐图注

中文图注可写成：

> 图 X. S2S-ACD-DHC 总体框架。我们首先利用文本分解标注将输入描述转化为 target、attribute 和 relation-anchor tuple 的结构化 slot memory（S2S）；随后在 decoder query 上执行目标属性粗筛选、anchor-conditioned 关系推理与残差融合（ACD）；最后通过 entity / attribute / relation hard negative 与 consistency regularization 将结构分解进一步转化为监督信号（DHC）。

英文图注可写成：

> Figure X. Overview of S2S-ACD-DHC. The input utterance is first decomposed into structured slot memory including target, attribute, and relation-anchor tuple representations (S2S). The decoder then performs target-attribute coarse filtering, anchor-conditioned relation reasoning, and residual score fusion over decoder queries (ACD). Finally, the structured decomposition is further transformed into supervision through entity, attribute, and relation hard negatives, together with consistency regularization (DHC).

### 7.4 方法图最该强调的三个视觉重点

- `slot memory` 要画成和普通 token 不同的结构块，不然看不出 S2S 的贡献。
- `relation-anchor tuple` 要画成成对结构，不要只画 relation token。
- `DHC` 要明确画成 supervision branch，不然会看起来像只是又加了一个打分头。

---

## 8. 数学公式版

这一节给的是可以直接写进论文方法章节的公式化表述。

### 8.1 符号定义

- 点云输入记为 $\mathcal{P}$。
- 检测器或 proposal 模块输出的候选框记为 $\mathcal{B}=\{b_q\}_{q=1}^{Q}$。
- 文本输入记为 $x=\{w_i\}_{i=1}^{L}$。
- 文本编码器输出 token 特征记为 $H=\{h_i\}_{i=1}^{L},\; h_i\in\mathbb{R}^{d}$。
- entity spans、attribute spans、relation spans 分别记为
  $\mathcal{E}, \mathcal{A}, \mathcal{R}$。
- decoder 最后一层 query 特征记为
  $\mathcal{Q}=\{q_j\}_{j=1}^{Q},\; q_j\in\mathbb{R}^{d}$。

---

### 8.2 S2S：Span-to-Slot Structured Decomposition

#### 8.2.1 全局 slot

给定 token 特征 $H$，全局 slot 记为：

$$
g = \operatorname{Pool}(H)
$$

其中 `Pool` 可以是 mean pooling 或 attention pooling。

#### 8.2.2 entity span 表示与 target slot

对第 $n$ 个 entity span，记其 token 区间为 $[s_n^e,t_n^e)$，则其 span 表示为：

$$
e_n = \operatorname{Pool}\left(\{h_i \mid s_n^e \le i < t_n^e\}\right)
$$

S2S 不直接把所有 entity span 平均，而是通过全局语义 $g$ 选择目标实体：

$$
\alpha_n^e = \frac{\exp(e_n^\top W_t g)}
{\sum_{m}\exp(e_m^\top W_t g)}
$$

$$
t = \sum_n \alpha_n^e e_n
$$

其中 $t\in\mathbb{R}^d$ 为 `target slot`。

#### 8.2.3 attribute slot

对第 $m$ 个 attribute span，先做 span pooling 得到 $a_m$，再聚合为：

$$
a = \frac{1}{\max(1, N_a)} \sum_{m=1}^{N_a} a_m
$$

其中 $a\in\mathbb{R}^d$ 为 `attribute slot`。

#### 8.2.4 relation-anchor tuple slots

对于第 $k$ 个 relation span，先得到 relation 表示 $r_k$。  
若其对应 anchor entity 索引为 $\pi_k$，则 anchor span 表示为：

$$
c_k = e_{\pi_k}
$$

最终得到一组 relation-anchor tuples：

$$
\mathcal{T} = \{(r_k, c_k)\}_{k=1}^{K}
$$

其中 $K$ 为保留的最大 tuple 数。

#### 8.2.5 parse confidence

根据是否存在目标实体、attribute 数量、relation-anchor tuple 数量，构造 coverage 特征：

$$
z = [\mathbb{I}_{\text{target}}, N_a, N_r]
$$

再通过 MLP 得到 parse confidence：

$$
\rho = \sigma(\operatorname{MLP}_{\text{conf}}(z))
$$

因此，S2S 最终输出的结构化记忆可记为：

$$
\mathcal{S} = \{g,\; t,\; a,\; \mathcal{T},\; \rho\}
$$

---

### 8.3 ACD：Anchor-Conditioned Compositional Decoder

#### 8.3.1 基础 grounding score

设文本 token 投影为 $\{\tilde h_i\}_{i=1}^{L}$，query 投影为 $\{\tilde q_j\}_{j=1}^{Q}$。  
基础相似度先定义为：

$$
\gamma_{j,i} = \frac{\tilde q_j^\top \tilde h_i}{\tau}
$$

其中 $\tau$ 为可学习温度参数。

再通过 token attention 聚合得到 baseline grounding score：

$$
\beta_i = \operatorname{softmax}(\operatorname{MLP}_{\text{tok}}(\tilde h_i))
$$

$$
s_j^{\text{base}} = \sum_{i=1}^{L}\beta_i \gamma_{j,i}
$$

#### 8.3.2 target-attribute coarse score

对于每个 query $q_j$，ACD 引入 target slot 和 attribute slot 的粗粒度组合分数：

$$
s_j^{ea} = \operatorname{MLP}_{ea}([q_j;\, t;\, a])
$$

再按比例缩放：

$$
\hat s_j^{ea} = \lambda_{ea}\, s_j^{ea}
$$

其中 $\lambda_{ea}$ 对应实现里的 `acd_ea_scale`。

#### 8.3.3 target candidate pool selection

候选池选择分数定义为：

$$
s_j^{pool} = s_j^{base} + \lambda_{pool}\, \hat s_j^{ea}
$$

选取前 $M$ 个 query 构成 target pool：

$$
\mathcal{M} = \operatorname{TopM}(\{s_j^{pool}\}_{j=1}^{Q})
$$

这里 $\lambda_{pool}$ 对应 `acd_pool_ea_multiplier`。

#### 8.3.4 anchor selection for each relation tuple

对于每个 tuple $(r_k, c_k)$，ACD 对所有 query 计算 anchor score：

$$
s_{k,j}^{anc} = \operatorname{MLP}_{anc}([q_j;\, c_k])
$$

再取前 $K_a$ 个 anchor 候选并归一化：

$$
p_{k,m}^{anc} = \operatorname{softmax}(s_{k,m}^{anc})
$$

#### 8.3.5 relation-aware compositional score

对于 target candidate $q_j$ 与 anchor candidate $q_m$，构造 relation pair score：

$$
s_{k,j,m}^{rel}
=
\operatorname{MLP}_{rel}
\big(
[q_j;\, q_m;\, r_k;\, \phi_{\text{geo}}(b_j,b_m)]
\big)
$$

若不使用几何项，则去掉 $\phi_{\text{geo}}(b_j,b_m)$。

对 anchor 分布加权后得到第 $k$ 个 tuple 对候选 $q_j$ 的结构分数：

$$
\bar s_{k,j}^{rel}
=
\sum_{m} p_{k,m}^{anc} \, s_{k,j,m}^{rel}
$$

再对所有有效 tuple 汇总：

$$
s_j^{struct}
=
\sum_{k=1}^{K} m_k \, \bar s_{k,j}^{rel}
$$

其中 $m_k\in\{0,1\}$ 表示该 tuple 是否有效。

#### 8.3.6 最终融合

最终 grounding score 定义为：

$$
s_j^{final}
=
s_j^{base}
+
\lambda_{final}\,\hat s_j^{ea}
+
\alpha\, s_j^{struct}
$$

其中：

- $\lambda_{final}$ 对应 `acd_final_ea_multiplier`
- $\alpha$ 是固定或可学习的 residual fusion 系数
- 在训练早期，$\alpha$ 还可以采用 warmup 方式逐步增大

最终预测框由最高分 query 给出：

$$
j^* = \arg\max_j s_j^{final}
$$

---

### 8.4 DHC：Decomposition-Guided Hard Negative & Consistency Learning

设 Hungarian matching 得到的正样本 query 集为 $\mathcal{P}^+$，负样本 query 集为 $\mathcal{P}^-$。

#### 8.4.1 ACD ranking loss

ACD 本身先通过排序损失约束 matched query 得分高于 hardest negatives：

$$
\mathcal{L}_{rank}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left[
\log \sum_{j\in \mathcal{P}_b^-}\exp(s_{b,j}^{final})
-
\frac{1}{|\mathcal{P}_b^+|}
\sum_{j\in \mathcal{P}_b^+} s_{b,j}^{final}
+
m_{rank}
\right]_+
$$

其中 $m_{rank}$ 是可学习 margin。

#### 8.4.2 entity hard negative

先计算 query 与 target slot 的相似度：

$$
u_{b,j}^{ent} = \cos(q_{b,j}, t_b)
$$

对应的 entity hard negative loss 为：

$$
\mathcal{L}_{ent}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left[
\log \sum_{j\in \mathcal{P}_b^-}\exp(u_{b,j}^{ent})
-
\frac{1}{|\mathcal{P}_b^+|}
\sum_{j\in \mathcal{P}_b^+} u_{b,j}^{ent}
+
m_{ent}
\right]_+
$$

#### 8.4.3 attribute hard negative

同理，query 与 attribute slot 的相似度为：

$$
u_{b,j}^{attr} = \cos(q_{b,j}, a_b)
$$

其损失为：

$$
\mathcal{L}_{attr}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbb{I}(N_a^b>0)
\left[
\log \sum_{j\in \mathcal{P}_b^-}\exp(u_{b,j}^{attr})
-
\frac{1}{|\mathcal{P}_b^+|}
\sum_{j\in \mathcal{P}_b^+} u_{b,j}^{attr}
+
m_{attr}
\right]_+
$$

#### 8.4.4 relation hard negative

对每个有效 relation slot $r_{b,k}$，定义：

$$
u_{b,k,j}^{rel} = \cos(q_{b,j}, r_{b,k})
$$

relation hard negative loss 写为：

$$
\mathcal{L}_{rel}
=
\frac{1}{B}
\sum_{b=1}^{B}
\sum_{k=1}^{K}
m_{b,k}
\left[
\log \sum_{j\in \mathcal{P}_b^-}\exp(u_{b,k,j}^{rel})
-
\frac{1}{|\mathcal{P}_b^+|}
\sum_{j\in \mathcal{P}_b^+} u_{b,k,j}^{rel}
+
m_{rel}
\right]_+
$$

#### 8.4.5 consistency loss

为了避免结构分支和原始 grounding 分支完全漂离，定义：

$$
p_b^{base} = \operatorname{softmax}(s_b^{base})
$$

$$
p_b^{final} = \operatorname{softmax}(s_b^{final})
$$

一致性损失为：

$$
\mathcal{L}_{cons}
=
\operatorname{KL}(p_b^{final}\,\|\, p_b^{base})
$$

#### 8.4.6 DHC 总损失

于是 DHC 部分的总损失可以写为：

$$
\mathcal{L}_{DHC}
=
\lambda_{cons}\mathcal{L}_{cons}
+
\lambda_{ent}\mathcal{L}_{ent}
+
\lambda_{attr}\mathcal{L}_{attr}
+
\lambda_{rel}\mathcal{L}_{rel}
$$

---

### 8.5 总体训练目标

最终总损失由原始检测/grounding 目标、ACD ranking loss、DHC losses 共同构成：

$$
\mathcal{L}
=
\mathcal{L}_{base}
+
\mathcal{L}_{rank}
+
\mathcal{L}_{DHC}
$$

其中 $\mathcal{L}_{base}$ 在当前实现中对应：

- query point generation loss
- classification loss
- bounding box regression loss
- GIoU loss
- contrastive alignment loss
- span contrastive loss

如果按照代码形式写得更贴近实现，可写成：

$$
\mathcal{L}
=
8\mathcal{L}_{qpg}
+
\frac{1}{N_d+1}
\left(
\mathcal{L}_{ce}
 + 5\mathcal{L}_{bbox}
 + \mathcal{L}_{giou}
 + \mathcal{L}_{align}
 + \lambda_{span}\mathcal{L}_{span}
\right)
+
\mathcal{L}_{rank}
+
\mathcal{L}_{DHC}
$$

其中 $N_d$ 为 decoder 层数。

### 8.6 这一套公式该怎么放进论文

推荐写法顺序：

1. 先给 `Problem Definition`
2. 再给 `S2S` 的公式
3. 然后写 `ACD` 的 query scoring 流程
4. 最后写 `DHC` 的 structured supervision
5. 用一个总损失公式收尾

这样结构最顺，也最符合你当前代码实现。

---

## 9. 最后总结

如果只用一句话概括这套方法：

> 我们不是只给文本“加标签”，而是把文本分解结果真正做成了 **结构化表示、结构化推理、结构化监督** 三位一体的完整框架。

对应到三条创新点就是：

1. **S2S**：把文本分解结果变成 structured slot memory  
2. **ACD**：让 decoder 基于 slot memory 做 anchor-conditioned compositional reasoning  
3. **DHC**：把结构分解结果继续转化成 hard negative 与 consistency supervision  

这三者串起来，才是当前这套方法的完整论文主线。
