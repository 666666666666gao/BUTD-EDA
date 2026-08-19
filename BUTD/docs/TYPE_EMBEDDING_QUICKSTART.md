# Token-Type Embeddings 快速开始

## 一句话总结

给原始 RoBERTa tokens 添加可学习的类型标注（entity/attr/rel），而不是创建额外的 span tokens。

## 为什么需要这个方案？

之前的 Span-Residual 方案失败了：
- ❌ Gate 值失控（0.17 → 0.78+）
- ❌ 出现 NaN（gate diversity loss）
- ❌ 性能下降（-0.018）

Type Embeddings 方案：
- ✅ 无门控，无 NaN
- ✅ 仅 3K 参数
- ✅ 最坏情况 = baseline

## 快速使用

### 1. 训练（零初始化，最安全）

```bash
cd /home/gb/new\ butd/butd_detr-main
bash scripts/train_sr3d_type_embeddings.sh
```

### 2. 训练（小初始化，类似 BERT）

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 train_dist_mod.py \
    --dataset sr3d_spacy --test_dataset sr3d_spacy \
    --data_root /root/autodl-tmp/DATA_ROOT \
    --use_color --butd --self_attend --augment_det \
    --batch_size 40 --max_epoch 400 \
    --use_soft_token_loss --use_contrastive_align \
    --detect_intermediate --joint_det \
    --use_token_type_embed \
    --token_type_embed_init small \
    --log_dir /root/autodl-tmp/logs/sr3d_type_embeddings_small
```

### 3. 验证 Baseline（不启用 type embeddings）

```bash
# 确保向后兼容性
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 train_dist_mod.py \
    --dataset sr3d_spacy --test_dataset sr3d_spacy \
    --data_root /root/autodl-tmp/DATA_ROOT \
    --use_color --butd --self_attend --augment_det \
    --batch_size 40 --max_epoch 400 \
    --use_soft_token_loss --use_contrastive_align \
    --detect_intermediate --joint_det \
    --log_dir /root/autodl-tmp/logs/sr3d_baseline
```

## 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_token_type_embed` | False | 启用 token type embeddings |
| `--token_type_embed_init` | zeros | 初始化策略：zeros/small/normal |

## 初始化策略对比

| 策略 | 初始化 | 特点 | 推荐场景 |
|------|--------|------|---------|
| `zeros` | 全零 | 开始时完全等同 baseline | **首选**，最安全 |
| `small` | N(0, 0.02) | 类似 BERT segment embeddings | 可能更快收敛 |
| `normal` | N(0, 0.1) | 标准初始化 | 实验性 |

## 预期结果

### 成功标准

- ✅ 训练稳定，无 NaN
- ✅ Acc@0.25 >= 0.357（baseline）
- ✅ 如果有提升（+2-5%），说明类型信息有效

### 对比 Baseline

| 指标 | Baseline | 预期（Type Embeddings） |
|------|---------|------------------------|
| Acc@0.25 | 0.357 | 0.357-0.375 |
| 训练稳定性 | ✓ | ✓ |
| Token 数量 | 58 | 58（相同） |
| 额外参数 | 0 | 3K |

## 检查实现是否生效

### 方法 1：查看配置

```bash
cat /root/autodl-tmp/logs/sr3d_type_embeddings/*/config.json | grep use_token_type_embed
```

应该看到：
```json
"use_token_type_embed": true
```

### 方法 2：查看模型参数

```python
import torch
checkpoint = torch.load('path/to/checkpoint.pth')
print('token_type_embedding.weight' in checkpoint['model'])
# 应该输出 True

# 查看 type embeddings
type_embeds = checkpoint['model']['token_type_embedding.weight']
print(f"Shape: {type_embeds.shape}")  # 应该是 (4, 288)
print(f"Norm: {type_embeds.norm():.4f}")
```

### 方法 3：添加调试日志

在 `models/bdetr.py` 的 `_run_backbones()` 方法中添加：

```python
if self.use_token_type_embed:
    print(f"[Type Embed] Applied to batch_size={token_type_ids.shape[0]}")
    print(f"[Type Embed] Type distribution: "
          f"Other={(token_type_ids==0).sum()}, "
          f"Entity={(token_type_ids==1).sum()}, "
          f"Attr={(token_type_ids==2).sum()}, "
          f"Rel={(token_type_ids==3).sum()}")
```

## 故障排除

### 问题：训练脚本报错 "unknown argument: --use_token_type_embed"

**原因**：参数未添加到 `main_utils.py`

**解决**：检查 `main_utils.py` 第 144-149 行是否包含：
```python
parser.add_argument('--use_token_type_embed', action='store_true', ...)
```

### 问题：模型初始化报错 "unexpected keyword argument 'use_token_type_embed'"

**原因**：参数未传递到模型

**解决**：检查 `train_dist_mod.py` 第 114-116 行是否包含：
```python
use_token_type_embed=args.use_token_type_embed,
token_type_embed_init=args.token_type_embed_init
```

### 问题：性能下降

**解决**：
1. 使用 `--token_type_embed_init zeros`（最安全）
2. 检查 span 数据质量
3. 对比 baseline 确认问题

## 文档索引

- **方案设计**：[TYPE_EMBEDDING_PROPOSAL.md](TYPE_EMBEDDING_PROPOSAL.md)
- **实现细节**：[TYPE_EMBEDDING_IMPLEMENTATION.md](TYPE_EMBEDDING_IMPLEMENTATION.md)
- **快速开始**：本文档

## 下一步

1. ✅ 实现完成
2. ⏳ 运行 baseline 验证
3. ⏳ 运行 type embeddings 实验
4. ⏳ 分析结果
5. ⏳ 如果有效，探索改进方向

## 核心代码位置

| 文件 | 行数 | 内容 |
|------|------|------|
| `models/bdetr.py` | 137 | 添加参数 |
| `models/bdetr.py` | 205-220 | 初始化 type embeddings |
| `models/bdetr.py` | 394-418 | 应用 type embeddings |
| `main_utils.py` | 144-149 | 命令行参数 |
| `train_dist_mod.py` | 114-116 | 传递参数 |
| `scripts/train_sr3d_type_embeddings.sh` | - | 训练脚本 |
