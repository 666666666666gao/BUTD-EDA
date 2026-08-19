# 评估结果日志功能

## 概述

训练代码已修改，现在每次评估（每 `val_freq` 轮）的结果都会自动写入日志文件。

## 修改的文件

### 1. `src/grounding_evaluator.py`

**修改内容**：
- `GroundingEvaluator.print_stats()` 现在返回一个包含所有评估指标的字典
- `GroundingGTEvaluator.print_stats()` 同样返回评估结果字典

**返回的字典格式**：
```python
{
    'last__bbs_acc0.25_top1': 0.366,
    'last__bbs_acc0.25_top5': 0.512,
    'last__bbs_acc0.25_top10': 0.589,
    'last__bbs_acc0.50_top1': 0.213,
    'last__bbs_acc0.50_top5': 0.378,
    'last__bbs_acc0.50_top10': 0.467,
    'last__bbf_acc0.25_top1': 0.368,
    'last__bbf_acc0.25_top5': 0.515,
    'last__bbf_acc0.25_top10': 0.592,
    'last__bbf_acc0.50_top1': 0.215,
    'last__bbf_acc0.50_top5': 0.381,
    'last__bbf_acc0.50_top10': 0.470,
    'easy': 0.412,
    'hard': 0.326,
    'vd': 0.289,
    'vid': 0.398,
    'unique': 0.445,
    'multi': 0.301
}
```

### 2. `train_dist_mod.py`

**修改内容**：
- `evaluate_one_epoch()` 现在返回评估结果字典（而不是 `None`）

### 3. `main_utils.py`

**修改内容**：
- 训练循环中，每次评估后会将结果写入日志文件
- 训练结束后的最终评估也会写入日志文件

## 日志文件位置

评估日志文件会保存在与 checkpoint 相同的目录中：

```
/root/autodl-tmp/logs/<experiment_name>/<dataset>/<timestamp>/
├── ckpt_epoch_5.pth
├── eval_epoch_5.log          # 第 5 轮评估结果
├── ckpt_epoch_10.pth
├── eval_epoch_10.log         # 第 10 轮评估结果
├── ckpt_epoch_15.pth
├── eval_epoch_15.log         # 第 15 轮评估结果
├── ...
├── ckpt_epoch_last.pth
└── eval_epoch_last.log       # 最终评估结果
```

## 日志文件格式

每个评估日志文件的格式如下：

```
==============================================
Evaluation Results - Epoch 10
==============================================

easy: 0.4123
hard: 0.3256
last__bbf_acc0.25_top1: 0.3680
last__bbf_acc0.25_top10: 0.5920
last__bbf_acc0.25_top5: 0.5150
last__bbf_acc0.50_top1: 0.2150
last__bbf_acc0.50_top10: 0.4700
last__bbf_acc0.50_top5: 0.3810
last__bbs_acc0.25_top1: 0.3660
last__bbs_acc0.25_top10: 0.5890
last__bbs_acc0.25_top5: 0.5120
last__bbs_acc0.50_top1: 0.2130
last__bbs_acc0.50_top10: 0.4670
last__bbs_acc0.50_top5: 0.3780
multi: 0.3010
unique: 0.4450
vd: 0.2890
vid: 0.3980

==============================================
```

## 使用方法

### 训练时自动记录

训练时不需要任何额外操作，评估结果会自动记录：

```bash
bash scripts/train_sr3d_type_embed_span_direct.sh
```

### 查看评估结果

```bash
# 查看第 10 轮的评估结果
cat /root/autodl-tmp/logs/<experiment_name>/<dataset>/<timestamp>/eval_epoch_10.log

# 查看最终评估结果
cat /root/autodl-tmp/logs/<experiment_name>/<dataset>/<timestamp>/eval_epoch_last.log
```

### 对比不同轮次的结果

```bash
# 对比第 5 轮和第 10 轮的 hard 指标
grep "hard:" eval_epoch_5.log eval_epoch_10.log
```

### 提取关键指标

```bash
# 提取所有轮次的 last__bbf_acc0.25_top1 (contrastive @0.25 Top-1)
grep "last__bbf_acc0.25_top1:" eval_epoch_*.log

# 提取所有轮次的 hard 指标
grep "^hard:" eval_epoch_*.log
```

## 关键指标说明

### 主要指标

- **last__bbf_acc0.25_top1**: Contrastive 方法在 IoU@0.25 的 Top-1 准确率（最重要）
- **last__bbf_acc0.50_top1**: Contrastive 方法在 IoU@0.50 的 Top-1 准确率（精定位）
- **last__bbs_acc0.25_top1**: Soft-token 方法在 IoU@0.25 的 Top-1 准确率

### 分析指标

- **hard**: 困难样本准确率
- **easy**: 简单样本准确率
- **vd**: View-dependent 样本准确率
- **vid**: View-independent 样本准确率
- **unique**: 唯一目标准确率
- **multi**: 多目标准确率

## 示例：对比 baseline 和新方法

```bash
# Baseline (Type Embedding)
grep "last__bbf_acc0.25_top1:" /root/autodl-tmp/logs/sr3d_type_embeddings/sr3d_spacy/*/eval_epoch_10.log

# 新方法 (Type Embedding + Span-Direct)
grep "last__bbf_acc0.25_top1:" /root/autodl-tmp/logs/sr3d_type_embed_span_direct/sr3d_spacy/*/eval_epoch_10.log
```

## 注意事项

1. **分布式训练**：只有 rank 0 进程会写入日志文件，避免重复写入
2. **文件覆盖**：每次评估会覆盖同名的日志文件
3. **格式一致**：所有指标都保留 4 位小数，便于对比
4. **排序输出**：指标按字母顺序排序，便于查找

## 兼容性

- ✅ 兼容现有的训练脚本
- ✅ 不影响终端输出（仍然会打印到终端）
- ✅ 支持所有评估模式（GroundingEvaluator 和 GroundingGTEvaluator）
- ✅ 支持分布式训练
