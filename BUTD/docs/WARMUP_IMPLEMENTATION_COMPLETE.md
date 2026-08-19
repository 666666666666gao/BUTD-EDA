# Warmup 功能实现完成

## 实现概述

已成功实现 span auxiliary loss 的 warmup 功能，用于方案 A（保守 + Warmup）。

## 修改的文件

### 1. main_utils.py

**添加参数定义**（第 175-176 行）：
```python
parser.add_argument('--span_aux_warmup_steps', type=int, default=0,
                    help='Number of warmup steps for span auxiliary loss (0 = no warmup, default: 0)')
```

**修改 `_compute_loss` 方法**（第 524-558 行）：
- 添加 `epoch`, `batch_idx`, `steps_per_epoch` 参数
- 计算 global_step 和 warmup_factor
- 将 `span_aux_factor` 设置到 `end_points`

**修改 `train_one_epoch` 方法**（第 575-610 行）：
- 计算 `steps_per_epoch`
- 传递 `epoch`, `batch_idx`, `steps_per_epoch` 到 `_compute_loss`

**修改 `_accumulate_stats` 方法**（第 560 行）：
- 添加 `span_aux_factor` 到统计收集

### 2. losses.py

**无需修改**：
- `compute_hungarian_loss` 已经支持从 `end_points` 读取 `span_aux_factor`（第 832-833 行）
- 自动应用到 span loss 权重

## 使用方法

在训练脚本中添加参数：
```bash
--span_aux_warmup_steps 10000  # 约 5 轮 warmup
```

## Warmup 时间表

假设每轮 2000 步，warmup_steps=10000：

| Epoch | Global Step | Warmup Factor | 说明 |
|-------|-------------|---------------|------|
| 0     | 0           | 0.0000        | 开始训练，span loss 权重为 0 |
| 1     | 2000        | 0.2000        | 第 1 轮结束，权重 20% |
| 2     | 4000        | 0.4000        | 第 2 轮结束，权重 40% |
| 3     | 6000        | 0.6000        | 第 3 轮结束，权重 60% |
| 4     | 8000        | 0.8000        | 第 4 轮结束，权重 80% |
| 5     | 10000       | 1.0000        | 第 5 轮开始，权重 100% |
| 6+    | 12000+      | 1.0000        | 保持 100% |

## 监控

训练日志会显示 `span_aux_factor`：
```
Train: [0][500/2000]
loss 2.3456  loss_span_contrastive 0.1234  span_aux_factor 0.0250
```

## 测试

运行测试脚本：
```bash
python test_warmup.py
```

## 下一步

1. 启动方案 A 训练
2. 监控训练日志，确认 warmup 正常工作
3. 对比 baseline 结果

## 相关文档

- [docs/SPAN_WARMUP_IMPLEMENTATION.md](docs/SPAN_WARMUP_IMPLEMENTATION.md) - 详细实现说明
- [docs/SPAN_DIRECT_PLAN_A.md](docs/SPAN_DIRECT_PLAN_A.md) - 方案 A 设计
- [scripts/train_sr3d_type_embed_span_conservative_warmup.sh](scripts/train_sr3d_type_embed_span_conservative_warmup.sh) - 训练脚本
