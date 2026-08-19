# 3DVG 重构方案（面向论文与 Claude 实施）

## 0. 一句话结论

当前的 `Type Embed + Span-Direct` 还不够成为论文核心，因为它们主要停留在**输入侧标注**和**训练侧约束**，并没有真正改变模型如何做推理。更合适的重构方式不是继续补一个“第三小点”，而是把整套方法重构成：

> **结构化文本分解 → 锚点条件推理 → 组合式监督**

也就是：

1. 把文本分解从“token 打类型标签”升级为“phrase / slot 级结构化记忆”；
2. 把 decoder 从“看所有 token”升级为“显式 target-anchor-relation 推理”；
3. 把 Span-Direct 从“单一 loss”升级为“分解指导的 hard negative / consistency learning”。

---

## 1. 先对原方案再做一次审查

## 1.1 原 Claude 文档里对的地方

原文档抓住了一个核心问题：

- 你已经把文本做了 entity / attr / rel 分解；
- 但 decoder 仍然对所有 token 混合 cross-attention；
- 所以分解信息没有真正进入推理路径。

这个判断是对的。

## 1.2 原 Claude 文档的问题

### 问题 A：创新点还是偏轻

`Type-Decomposed Cross-Attention` 如果只是“三条 attention + 一个 gate”，很容易被 reviewer 认为只是已有思路的重新拼装：

- 以前已经有 text decoupling；
- 以前也有 entity-aware / relation-aware attention；
- 以前也有 language scene graph / structured decomposition。

所以如果只是“按类型分三路”，新意边界不够强。

### 问题 B：relation 路径没有真正落到视觉关系上

关系词最难的不是识别到 `left of / behind / closest to` 这些词本身，
而是：

- 找到 anchor 是谁；
- 判断 candidate target 和 anchor 的几何关系是否满足文本；
- 在多个同类物体中完成 disambiguation。

如果 relation 分支只是对 relation tokens 单独 attention，而没有显式 target-anchor pair reasoning，效果通常有限。

### 问题 C：硬层次化容易伤 NR3D

你现有实验已经说明：

- SR3D 小涨；
- NR3D 没过 baseline。

这说明规则化语言（SR3D）受益于“更强结构先验”，
但自然语言（NR3D）更容易被 parser 噪声、类型误分、长句省略等问题影响。

所以不建议一开始就做：

- hard type mask；
- hard layer specialization；
- 所有 query 都必须走 decomposition-only 路径。

更稳妥的是：

- **soft routing**；
- **global residual**；
- **confidence-aware fallback**。

### 问题 D：三独立 attention 头参数收益比不高

如果每层都加三套 cross-attn，论文上看起来改动很大，但不一定“值”。
真正带来性能上限的，更可能是**关系推理如何建模**，而不是把 QKV 复制三份。

---

## 2. 我建议的最终重构方向

建议把整篇方法重构成一个完整主线：

# **Span-to-Slot Compositional Reasoning for 3D Visual Grounding**

核心思想：

> **不是只把 token 分类型，而是把句子分解成结构化 slot；不是只让 decoder 分路，而是让 target / attribute / relation / anchor 这些 slot 直接驱动候选物体与锚点之间的组合式推理。**

### 最终三条创新点

1. **Span-to-Slot Structured Decomposition (S2S)**  
   把文本分解从 token-level type label 升级为 phrase-level / slot-level structured memory。

2. **Anchor-Conditioned Compositional Decoder (ACD)**  
   把分解后的 slot 真正送进推理路径，显式建模 `target candidate ↔ anchor candidate ↔ relation` 的组合打分。

3. **Decomposition-Guided Hard Negative & Consistency Learning (DHC)**  
   用分解信息构造 type-specific hard negatives，并约束 global path 与 structured path 的一致性，提升 NR3D 鲁棒性。

---

## 3. 创新点 1：Span-to-Slot Structured Decomposition (S2S)

## 3.1 为什么要这样改

你当前的 `Type Embed` 本质上还是：

- 给每个 token 一个类型标签；
- 然后把 token 继续送进同一个语言序列。

这个改动太轻，而且 parser 输出的信息没有真正“结构化”。

更强的做法是把句子转成若干**slot**：

- `global`：整句全局语义
- `target`：目标名词短语
- `attr`：属性短语集合
- `rel-anchor tuples`：关系-锚点对，形如 `(relation_k, anchor_k)`

例如：

> “the red chair to the left of the table near the window”

可以分解为：

- global = 整句
- target = “chair”
- attr = [“red”]
- tuples = [
  - (“to the left of”, “table”),
  - (“near”, “window”)
]

这比简单的 token type mask 强很多，因为它保留了：

- target 自身；
- 属性修饰；
- 多个关系链；
- 每个关系对应哪个 anchor。

## 3.2 具体输出格式

建议解析器最终输出如下结构：

```python
{
    "global_span": [0, T-1],
    "target_span": [l_t, r_t],
    "attr_spans": [[l1, r1], [l2, r2], ...],
    "rel_anchor_pairs": [
        {
            "rel_span": [l_r1, r_r1],
            "anchor_span": [l_a1, r_a1],
            "confidence": c1,
        },
        {
            "rel_span": [l_r2, r_r2],
            "anchor_span": [l_a2, r_a2],
            "confidence": c2,
        }
    ],
    "parse_confidence": c_global,
    "coverage": {
        "has_target": True,
        "num_attrs": 1,
        "num_pairs": 2,
    }
}
```

## 3.3 slot 表征怎么做

不要只保留 span 边界，要把每个 span pooling 成 slot memory：

```python
m_global = pool(lang_feats[global_span])
m_target = pool(lang_feats[target_span])
m_attr   = mean([pool(lang_feats[s]) for s in attr_spans])
m_rel_k  = pool(lang_feats[rel_span_k])
m_anchor_k = pool(lang_feats[anchor_span_k])
```

其中 `pool()` 推荐优先级：

1. attention pooling
2. mean pooling
3. 取首尾 token + MLP

建议先用 **attention pooling**，因为比 mean 更稳。

## 3.4 这一点的关键增强：confidence-aware decomposition

S2S 最关键的不是“分解”，而是要承认分解会错。

因此建议加入两个置信度：

- `pair_confidence c_k`：每个 `(rel, anchor)` 对是否可靠；
- `parse_confidence c_global`：整句结构化分解整体是否可靠。

最终 structured path 的权重不应固定，而应该由置信度控制：

```python
alpha_struct = sigmoid(MLP([m_global, coverage_stats, c_global]))
```

然后用于融合：

```python
score_final = score_global + alpha_struct * score_struct
```

这样能明显降低 NR3D 上 parser 错误带来的负面影响。

## 3.5 这条创新点的论文写法

可以写成：

> We propose a **Span-to-Slot Structured Decomposition** module that converts free-form expressions into a compact slot memory consisting of target, attribute, and relation-anchor tuples, together with confidence-aware reliability estimation. Compared with token-level type embeddings, the proposed slot memory preserves compositional referential structure and provides a more stable interface for downstream reasoning.

---

## 4. 创新点 2：Anchor-Conditioned Compositional Decoder (ACD)

> 这是最核心、最应该当主创新写的部分。

## 4.1 为什么它比“分三路 attention”更强

如果只是：

- entity path 看 entity token；
- attr path 看 attr token；
- rel path 看 rel token；

那么 relation 分支依然没有回答一个关键问题：

> **“和谁发生关系？”**

所以更合理的 decoder 不该只是“类型分路”，
而应显式建模：

- target 候选是谁；
- anchor 候选是谁；
- 它们之间几何关系是否满足 relation slot。

## 4.2 ACD 的整体流程

建议保留原有 BUTD-DETR 主干和全局 decoder 路径，不推翻基线；
在此基础上新增一个 structured reasoning 分支：

### Step 1：保留 baseline 全局路径

```python
score_global(i)
```

它是模型的保底路径，用来保证 parser 失败时性能不至于塌掉。

### Step 2：target-attribute 粗打分

对每个 decoder query / candidate `i`，基于 target slot 和 attr slot 计算一个粗粒度 target score：

```python
s_ea(i) = MLP([
    q_i,
    m_target,
    m_attr,
    q_i * m_target,
    q_i * m_attr,
])
```

这个分支回答的是：

- 它是不是目标类别；
- 它是否满足属性描述。

### Step 3：anchor candidate 选择

对每个关系-锚点对 `k`，给所有 candidate `j` 一个 anchor score：

```python
p_anchor_k(j) = softmax_j(MLP([
    q_j,
    m_anchor_k,
    q_j * m_anchor_k,
]))
```

注意：

- 这里不需要真实 anchor 标注；
- 只是弱选择一个“最像 anchor phrase 的候选”。

### Step 4：target-anchor relation 打分（关键）

对 target candidate `i` 和 anchor candidate `j`，构造 pairwise geometry feature：

```python
g_ij = geo_encode(box_i, box_j)
```

建议 `geo_encode` 至少包含：

- center delta `(dx, dy, dz)`
- Euclidean distance
- size ratio `(dw, dh, dl)`
- IoU / overlap
- direction basis（left/right/front/behind/above/below 的几何编码）
- relative angle / horizontal bearing

然后 relation score：

```python
s_rel_k(i, j) = MLP([
    q_i,
    q_j,
    g_ij,
    m_rel_k,
    m_target,
    m_anchor_k,
])
```

### Step 5：对 anchor 做软聚合

不直接选单一 anchor，而是用软聚合更稳：

```python
s_pair_k(i) = logsumexp_j(
    log p_anchor_k(j) + s_rel_k(i, j)
)
```

### Step 6：组合多个关系对

如果有多个 `(rel, anchor)` 对，则：

```python
s_struct(i) = s_ea(i) + sum_k c_k * s_pair_k(i)
```

其中 `c_k` 是 pair confidence。

### Step 7：与 baseline 融合

```python
s_final(i) = s_global(i) + alpha_struct * s_struct(i)
```

这里的 `alpha_struct` 可以是样本级，也可以 query-specific。

---

## 4.3 这条创新点的本质

ACD 的本质不是“又多加一个 head”，而是：

> **把文本分解后的结构真正映射到候选目标-候选锚点-关系几何三元交互上。**

这点和简单的 type-aware cross-attention 有实质区别。

## 4.4 为什么这条线更有新意

它比原先方案更有新意的原因在于：

1. 不是 token mask，而是 **slot memory**；
2. 不是 relation token 自注意，而是 **target-anchor pair reasoning**；
3. 不是完全替换 baseline，而是 **global path + structured path** 的稳健融合；
4. 不是仅处理单一关系，而是支持 **多个 relation-anchor tuples**。

## 4.5 这条创新点的实现建议

### 最小实现版本（推荐先做）

先不要在 decoder 每一层都插 ACD，先做一个 **late reasoning head**：

- baseline decoder 正常跑完；
- 取最后一层 query features 与 predicted boxes；
- 在输出头之前加 ACD scoring；
- 用 ACD 输出调整 final refer score。

优点：

- 改动小；
- 训练更稳；
- 更容易 ablation。

### 第二阶段增强版

如果 late head 有收益，再进一步把 ACD 轻量接到 decoder 最后两层做 iterative refinement。

不建议一开始就“每层都做 pair reasoning”，复杂度高，调试成本大。

## 4.6 复杂度控制

pairwise relation 的复杂度是 `O(N^2)`，必须降。

建议：

- target 仅保留 top-M（如 16）个候选；
- anchor 仅保留 top-K（如 8）个候选；
- relation scorer 只在最后 1~2 层使用；
- 多个 pair 按 confidence 截断到前 2~3 个。

这样复杂度就可控。

## 4.7 这条创新点的论文写法

> We propose an **Anchor-Conditioned Compositional Decoder** that performs explicit target-anchor-relation reasoning over decoder candidates. Instead of attending to relation words in isolation, the proposed decoder scores candidate target-anchor pairs using pairwise geometry and relation-aware slot memories, thereby grounding decomposed language structure into object-level reasoning.

---

## 5. 创新点 3：Decomposition-Guided Hard Negative & Consistency Learning (DHC)

## 5.1 为什么不能继续只做 Span-Direct

如果第三点仍然只是一个普通 loss，还是会显得偏轻。

更合理的第三点应该和前两点形成闭环：

- S2S 提供结构化分解；
- ACD 用这些结构化信息做推理；
- DHC 用这些分解结果构造更难、更对路的监督。

## 5.2 DHC 包含两个子部分

### (A) type-specific hard negatives

不同类型分支应该打不同负样本：

#### entity / attr 分支的负样本

选与 GT：

- 同类但属性不同；
- 或外观相近但不是 target 的候选。

例如：

- 两个 chair，颜色不同；
- 两个 cabinet，大小不同。

#### relation 分支的负样本

选：

- 同类且属性接近，
- 但相对 anchor 的几何关系错误的候选。

这类负样本才是真正让 relation scorer 学会 disambiguation 的关键。

### (B) global-structured consistency

因为 decomposition 可能会错，所以不能让 structured path 单飞。

建议增加一致性约束：

```python
L_cons = KL(softmax(s_struct), softmax(s_global).detach())
      or KL(softmax(s_global), softmax(s_final).detach())
```

目的不是让两个分支完全一样，
而是防止 structured branch 学出极端噪声分布。

## 5.3 推荐损失形式

总损失可以写成：

```python
L = L_ref
  + λ_ea   * L_entity_attr
  + λ_rel  * L_relation_rank
  + λ_hn   * L_hard_negative
  + λ_cons * L_consistency
```

其中：

- `L_ref`：原 baseline refer loss
- `L_entity_attr`：entity/attr coarse branch 对 GT 的分类约束
- `L_relation_rank`：relation-conditioned final score 对 GT 的约束
- `L_hard_negative`：对难负样本的 margin / ranking loss
- `L_consistency`：global vs structured 一致性

## 5.4 可选增强：counterfactual relation augmentation

如果你愿意进一步加强论文故事，可以加一个可选模块：

- 对可逆关系词做反转：left ↔ right, in front of ↔ behind, above ↔ below
- 生成 counterfactual relation slot
- 要求模型对 counterfactual query 的 target score 更低

即：

```python
L_cf = max(0, margin - s_pos(gt) + s_cf(gt))
```

但这块我建议作为**附加增强**，不要一开始就上。
原因：

- 会增加实现复杂度；
- 要维护关系词映射表；
- 对 parser 精度更敏感。

## 5.5 这条创新点的论文写法

> We further introduce a **Decomposition-Guided Hard Negative and Consistency Learning** strategy, where type-specific hard negatives are mined according to decomposed semantics, and structured predictions are regularized by the global branch for robustness against parsing noise and free-form linguistic variability.

---

## 6. 为什么我认为这版比原 Claude 方案更好

## 6.1 从“类型”升级到了“结构”

原方案核心还是：

- entity / attr / rel 三类型；
- 三路 attention；
- 三段 decoder。

而新方案是：

- target / attr / (rel, anchor) tuple / global；
- slot memory；
- target-anchor-relation 显式推理。

这更符合 3DVG 真正的判别逻辑。

## 6.2 relation 终于落到了几何上

这是最大区别。

原方案虽然关注 relation token，
但没有把 relation phrase 真正变成“候选物体对之间几何约束”。

新方案把 relation 直接变成：

- anchor 选择；
- pairwise geometry scoring；
- 多关系聚合。

这才更有可能真正提升 NR3D。

## 6.3 对 NR3D 更稳

NR3D 是自然语言，容易出现：

- parser 错误；
- anchor 漏检；
- 省略关系；
- 非标准表达。

新方案里：

- 有 global path 兜底；
- 有 parse confidence 控制 structured 权重；
- 有 consistency regularization。

因此比硬 type routing 更稳。

---

## 7. 这套方案可能还有哪些问题

## 7.1 最大风险：parser 不稳

这是你方案天然风险。

### 解决策略

1. structured path 永远不替代 global path；
2. 每个 pair 都有 confidence；
3. 无 target / 无 pair 时直接 fallback；
4. 训练时做 slot dropout，防止模型过度依赖 parser。

---

## 7.2 第二个风险：pairwise 复杂度高

### 解决策略

1. 只在 late stage 做；
2. top-K anchor；
3. top-M target；
4. 多关系 tuple 截断。

---

## 7.3 第三个风险：没有 anchor 标注

### 解决策略

不要一开始做显式 anchor supervision。
先做：

- 弱选择 `p_anchor_k(j)`；
- 通过最终 refer loss 反向学习；
- 观察 attention / top-k 可视化。

如果后面发现 anchor 选择特别不稳，再考虑训练期伪标签。

---

## 8. 我最推荐的实施顺序（很重要）

## Phase 1：先做最小可用版本（MVP）

### 只做以下三件事

1. 把 token type 改成 **slot memory**：`global / target / attr / rel-anchor tuples`
2. 在 decoder 后面接一个 **late ACD head**
3. 用 **global + structured fusion** 输出最终 refer score

### 先不要做

- 每层插入 type-specific cross-attn
- hard hierarchical decoding
- counterfactual relation augmentation
- anchor pseudo label

### 目的

先验证一句话：

> “把分解后的结构转成 slot，并显式做 target-anchor relation scoring，是否能稳定超过 baseline？”

只要这个成立，论文主线就立住了。

---

## Phase 2：再加 DHC

加入：

- type-specific hard negatives
- global-structured consistency

看 NR3D 能否进一步回升。

---

## Phase 3：最后再尝试轻量 decoder 内注入

如果前两阶段有效，再做轻量增强：

- 最后 2 层 decoder 加 slot-guided refinement
- 不是每层都加
- 不是三独立 attention 头

这样既保住了论文故事，又不会一开始把工程复杂度拉爆。

---

## 9. 论文贡献建议写法

可以直接写成下面这样：

### Contribution 1

We propose a **Span-to-Slot Structured Decomposition** module that transforms free-form expressions into a compact compositional memory of target, attributes, and relation-anchor tuples, with confidence-aware reliability estimation for robust reasoning.

### Contribution 2

We propose an **Anchor-Conditioned Compositional Decoder** that explicitly performs target-anchor-relation reasoning over decoder candidates using pairwise geometry, grounding decomposed language structures into object-level visual reasoning.

### Contribution 3

We introduce a **Decomposition-Guided Hard Negative and Consistency Learning** strategy that mines type-specific negatives and regularizes structured reasoning with the global branch, improving robustness on free-form descriptions.

---

## 10. 预期指标（务实版）

### 保守预期

- NR3D：38.7 → 40~42
- SR3D：50.1 → 52~54

### 做得比较顺的情况

- NR3D：41~44
- SR3D：53~55+

### 现实判断

仅靠语言分解和 decoder 小改动，**很难直接追到 TSP3D 那种结构级 scene representation 提升**。

所以目标应该是：

1. 先稳定超过 baseline；
2. 再尽量缩小和 TSP3D 的差距；
3. 如果论文主线清楚，即使没追到 SOTA，也可以投稿。

---

## 11. ablation 建议（一定要做）

建议最少做以下消融：

1. baseline
2. baseline + token type embed
3. baseline + S2S
4. baseline + S2S + ACD
5. baseline + S2S + ACD + DHC
6. 去掉 global fallback
7. 去掉 pairwise geometry
8. 去掉 confidence-aware weighting
9. 去掉 hard negatives

这样你能非常清楚地回答 reviewer：

- 性能提升来自哪里；
- 为什么 slot 比 token type 更好；
- 为什么 relation 必须做 pairwise reasoning；
- 为什么 consistency 对 NR3D 有帮助。

---

## 12. 可直接交给 Claude 的实施提示词

下面我给你 4 组提示词。

---

## Prompt 1：总控版（建议先发这个）

```text
你现在是我的 3D visual grounding 代码合作者。请基于我当前的 BUTD-DETR / BiDecoderLayer 代码，帮我实现一个新的论文级方案，核心必须围绕“文本分解”展开，但不能只停留在 token type embedding 或单纯 loss 上。

我要你实现的目标方法叫：Span-to-Slot Compositional Reasoning for 3DVG。

整体包含三个模块：

1. Span-to-Slot Structured Decomposition (S2S)
   - 不再只输出 token_is_ent / token_is_attr / token_is_rel
   - 而是把文本解析成：
     - global span
     - target span
     - attribute spans
     - relation-anchor pairs: [(rel_span, anchor_span, confidence), ...]
   - 然后把这些 span pooling 成 slot memory：m_global, m_target, m_attr, m_rel_k, m_anchor_k
   - 需要支持没有 attribute / 没有 relation pair 的样本
   - 需要输出 parse_confidence 或至少一个 decomposition reliability score

2. Anchor-Conditioned Compositional Decoder (ACD)
   - 保留原 baseline 的 global decoder / refer score，作为保底路径
   - 在 decoder 最后一层输出后，加一个 late structured reasoning head，而不是一开始就改所有 decoder 层
   - 对每个 candidate query i：
     - 用 m_target 和 m_attr 计算 coarse entity-attribute score s_ea(i)
   - 对每个 relation-anchor pair k：
     - 对所有候选 j 计算 anchor score p_anchor_k(j)
     - 用 target candidate i、anchor candidate j、pairwise geometry g_ij、relation slot m_rel_k、anchor slot m_anchor_k 计算 relation score s_rel_k(i,j)
     - 用 soft aggregation 得到 s_pair_k(i)
   - 最终 structured score: s_struct(i) = s_ea(i) + sum_k c_k * s_pair_k(i)
   - 最终输出: s_final(i) = s_global(i) + alpha_struct * s_struct(i)
   - alpha_struct 应该是可学习、且受 decomposition confidence 控制的

3. Decomposition-Guided Hard Negative & Consistency Learning (DHC)
   - 为 entity/attr 分支构造 hard negatives：同类但属性或外观不同
   - 为 relation 分支构造 hard negatives：同类且属性接近，但与 anchor 的几何关系错误
   - 增加 global score 与 structured score 的 consistency regularization
   - 不要先做太复杂的 counterfactual text rewrite，先做稳定版

工程要求：
- 优先做最小可用版本（MVP）
- 尽量少改动原主干，先加 late head 验证效果
- 给出明确的修改文件列表
- 给出每个新模块的类名、输入输出张量 shape、前向逻辑
- 给出训练时需要增加的 loss
- 给出 ablation 开关和命令行参数
- 如果你认为某一步在当前代码中不容易直接实现，请提供替代实现，不要停留在高层描述

请先输出：
1. 你对当前代码结构的理解
2. 需要新增/修改的文件
3. 每个模块的伪代码
4. 建议的最小实现顺序
5. 可能踩坑点
```

---

## Prompt 2：只实现 S2S 的提示词

```text
请先只实现 Span-to-Slot Structured Decomposition (S2S)，不要动 decoder 主体。

要求：
1. 在现有文本编码输出 lang_feats 的基础上，新增一个 slot builder
2. 输入包括：
   - token embeddings / lang_feats
   - 已有 span 或 type 标注信息
3. 输出包括：
   - m_global
   - m_target
   - m_attr
   - rel_anchor_pairs 的 slot memory：[(m_rel_k, m_anchor_k, c_k), ...]
   - parse_confidence
4. pooling 默认使用 attention pooling
5. 要兼容以下情况：
   - 没有 attr spans
   - 没有 relation-anchor pairs
   - span 长度为 1
6. 给出具体代码：
   - 新增类名
   - forward 输入输出 shape
   - 需要在哪个文件接入
   - 如何把 S2S 输出缓存到后续 decoder / head 中
7. 保留原有 token type 逻辑作为 fallback，不要直接删除

请输出可直接合并到项目里的代码方案，不要只讲思路。
```

---

## Prompt 3：只实现 ACD 的提示词

```text
现在请基于已经存在的 S2S 输出，实现一个 late-stage Anchor-Conditioned Compositional Decoder (ACD) head。

目标：
- 不改动整个 decoder 框架
- 只在最后一层 decoder 输出 query features 和 predicted boxes 后面，增加 structured reasoning head

具体要求：
1. 输入：
   - final query features: q  [B, N, D]
   - predicted boxes: boxes [B, N, 6 or 7]
   - baseline global refer logits: s_global [B, N]
   - S2S 输出的 slot memories
2. 先实现 coarse entity-attribute scoring：
   - s_ea(i)
3. 再实现 anchor selection：
   - p_anchor_k(j)
4. 再实现 pairwise geometry encoding：
   - g_ij 至少包含 center delta、distance、size ratio、direction encoding
5. 再实现 relation scoring：
   - s_rel_k(i,j)
6. 用 soft aggregation 得到 s_pair_k(i)
7. 得到 s_struct(i)
8. 用可学习 alpha_struct 做融合：
   - s_final(i) = s_global(i) + alpha_struct * s_struct(i)
9. 必须控制复杂度：
   - anchor top-K
   - optional target top-M
10. 输出：
   - 需要新增的模块类
   - forward 伪代码
   - 每一步 shape
   - 应该改哪些文件
   - 默认超参数建议

请给我偏工程落地的实现方案，不要泛泛而谈。
```

---

## Prompt 4：只实现 DHC 的提示词

```text
现在请在已有 S2S + ACD 的基础上，实现 Decomposition-Guided Hard Negative & Consistency Learning (DHC)。

要求：
1. 为 entity/attr 分支挖 hard negatives：
   - 候选中与 GT 同类或高相似，但属性不匹配的对象
2. 为 relation 分支挖 hard negatives：
   - 与 GT 同类且属性接近，但相对 anchor 几何关系错误的对象
3. 设计 ranking / margin loss
4. 设计 global branch 与 structured branch 的 consistency loss
5. 所有 loss 都要有开关和权重
6. 给出训练期张量获取方式，不要假设额外标注存在
7. 如果某个 hard negative 在当前 batch 中找不到，要给出 fallback 逻辑
8. 输出：
   - loss 公式
   - 采样逻辑
   - 代码结构建议
   - 默认权重建议
   - 你认为最可能提升 NR3D 的超参数

请用“可以直接编码”的粒度回答。
```

---

## 13. 我的最终建议

如果你现在的目标是：

- **先稳定超过 baseline**；
- **再把方案写得像一篇完整论文**；

那我建议你不要继续沿着“Type Embed 再加强一点、Span-Direct 再复杂一点”去补。

你最应该做的是：

> **把文本分解从“标签”升级成“结构”，再把这个结构真正送进目标-锚点-关系推理。**

这是最符合你当前核心思想、又最可能拉开和现有轻量改动差异的一条线。

