# Span Auxiliary Loss Warmup 实现说明

## 功能概述

为了避免 span-direct supervision 在训练早期干扰主任务学习，我们实现了 warmup 机制：
- 前 N 步：span loss 权重从 0 逐渐增加到设定值
- N 步之后：span loss 权重保持设定值不变

## 实现方式

### 1. 参数配置

在训练脚本中添加参数：
```bash
--span_aux_warmup_steps 10000  # Warmup 步数（约 5 轮）
```

### 2. Warmup Factor 计算

在 `main_utils.py` 的 `_compute_loss` 方法中：

```python
span_aux_warmup_steps = getattr(args, 'span_aux_warmup_steps', 0)
if span_aux_warmup_steps > 0:
    global_step = epoch * steps_per_epoch + batch_idx
    if global_step < span_aux_warmup_steps:
        span_aux_factor = global_step / span_aux_warmup_steps
    else:
        span_aux_factor = 1.0
    end_points['span_aux_factor'] = span_aux_factor
```

### 3. Loss 计算

在 `losses.py` 的 `compute_hungarian_loss` 中：

```python
span_aux_factor = float(end_points.get('span_aux_factor', 1.0))
effective_span_contrastive_weight = span_contrastive_weight * span_aux_factor

loss = (
    ...
    + effective_span_contrastive_weight * loss_span_contrastive
)
```

## Warmup 时间表

假设每轮约 2000 步，warmup_steps=10000：

| Epoch | Batch | Global Step | Warmup Factor |
|-------|-------|-------------|---------------|
| 0     | 0     | 0           | 0.0000        |
| 0     | 1000  | 1000        | 0.1000        |
| 1     | 0     | 2000        | 0.2000        |
| 2     | 0     | 4000        | 0.4000        |
| 4     | 0     | 8000        | 0.8000        |
| 5     | 0     | 10000       | 1.0000        |
| 6+    | any   | 12000+      | 1.0000        |

## 监控

训练日志中会显示 `span_aux_factor`，可以验证 warmup 是否正常工作：

```
Train: [0][500/2000]
loss 2.3456  loss_span_contrastive 0.1234  span_aux_factor 0.0250
```

## 预期效果

1. **避免早期干扰**：前 5 轮 span loss 权重较小，不会干扰主任务学习
2. **逐渐引入细粒度监督**：随着模型逐渐学会主任务，span loss 权重逐渐增加
3. **最终达到设定值**：5 轮后 span loss 权重达到设定值，提供完整的细粒度监督

## 测试

运行测试脚本验证 warmup 计算：
```bash
python test_warmup.py
```
