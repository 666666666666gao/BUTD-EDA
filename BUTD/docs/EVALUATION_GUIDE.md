# 评估脚本使用指南

## 改进的评估脚本

### 文件位置
`scripts/eval_sr3d_ckpt10_with_log.sh`

### 主要改进

1. ✅ **自动读取所有参数**：从 `config.json` 自动恢复所有模型参数
   - 包括 `use_token_type_embed`（之前缺失）
   - 包括 `token_type_embed_init`
   - 包括所有 span contrastive 参数

2. ✅ **评估结果写入日志**：
   - 输出同时显示在终端和日志文件
   - 日志文件保存在 checkpoint 同目录：`eval_epoch_10.log`
   - 自动提取关键指标摘要

3. ✅ **灵活的命令行参数**：
   - 支持通过 `--checkpoint_path` 指定任意 checkpoint

## 使用方法

### 方法 1：评估默认 checkpoint

脚本默认评估：
```
/root/autodl-tmp/logs/sr3d_type_embed_span_direct_aggressive/sr3d_spacy/1772469891/ckpt_epoch_10.pth
```

直接运行：
```bash
cd /home/gb/new\ butd/butd_detr-main
bash scripts/eval_sr3d_ckpt10_with_log.sh
```

### 方法 2：评估指定 checkpoint

```bash
bash scripts/eval_sr3d_ckpt10_with_log.sh \
  --checkpoint_path /path/to/your/ckpt_epoch_10.pth
```

### 方法 3：评估多个 checkpoint（批量）

```bash
# 评估 Type Embedding only
bash scripts/eval_sr3d_ckpt10_with_log.sh \
  --checkpoint_path /root/autodl-tmp/logs/sr3d_type_embeddings/sr3d_spacy/*/ckpt_epoch_10.pth

# 评估 Type Embedding + Span-Direct (保守版)
bash scripts/eval_sr3d_ckpt10_with_log.sh \
  --checkpoint_path /root/autodl-tmp/logs/sr3d_type_embed_span_direct/sr3d_spacy/*/ckpt_epoch_10.pth

# 评估 Type Embedding + Span-Direct (激进版)
bash scripts/eval_sr3d_ckpt10_with_log.sh \
  --checkpoint_path /root/autodl-tmp/logs/sr3d_type_embed_span_direct_aggressive/sr3d_spacy/*/ckpt_epoch_10.pth
```

## 输出说明

### 1. 终端输出

评估过程会实时显示在终端，包括：
- 配置参数确认
- 评估进度
- 关键指标

### 2. 日志文件

评估完成后，日志保存在：
```
<checkpoint_dir>/eval_epoch_10.log
```

例如：
```
/root/autodl-tmp/logs/sr3d_type_embed_span_direct_aggressive/sr3d_spacy/1772469891/eval_epoch_10.log
```

### 3. 日志内容

日志文件包含：
- 完整的评估输出
- 所有指标的详细结果
- 关键指标摘要（自动提取）

## 查看评估结果

### 实时查看

```bash
# 查看评估进度
tail -f /root/autodl-tmp/logs/sr3d_type_embed_span_direct_aggressive/sr3d_spacy/1772469891/eval_epoch_10.log
```

### 查看关键指标

```bash
# 查看 last@0.25 和 last@0.50
grep "last_.*Acc0.25: Top-1:" eval_epoch_10.log
grep "last_.*Acc0.50: Top-1:" eval_epoch_10.log

# 查看 hard/easy/vid 等分析
grep "Analysis" eval_epoch_10.log
```

### 提取所有指标

```bash
# 提取所有 Acc@0.25 指标
grep "Acc0.25: Top-1:" eval_epoch_10.log

# 提取所有 Acc@0.50 指标
grep "Acc0.50: Top-1:" eval_epoch_10.log
```

## 当前正在运行的评估

### 评估目标
```
Checkpoint: /root/autodl-tmp/logs/sr3d_type_embed_span_direct_aggressive/sr3d_spacy/1772469891/ckpt_epoch_10.pth
```

### 配置
- Type Embedding: ✓ (zeros init)
- Span-Direct Supervision: ✓
- Lambda ent span: 0.5
- Lambda attr span: 1.0
- Lambda rel span: 1.0

### 预期结果

基于之前的 Type Embedding 结果（0.366），加上 Span-Direct 的激进配置，预期：

| 指标 | Type Embed | 预期 (+ Span-Direct) |
|------|-----------|---------------------|
| last@0.25 | 0.366 | **0.372-0.378** |
| Acc@0.50 | 0.213 | **0.220-0.228** |
| hard | 0.326 | **0.332-0.338** |

### 查看进度

```bash
# 方法 1：查看后台任务输出
cat /tmp/claude-0/-home-gb-new-butd-butd-detr-main/tasks/b9p9wszdu.output

# 方法 2：查看日志文件（评估完成后）
cat /root/autodl-tmp/logs/sr3d_type_embed_span_direct_aggressive/sr3d_spacy/1772469891/eval_epoch_10.log
```

## 对比不同配置的结果

### 创建对比脚本

```bash
#!/bin/bash
# compare_results.sh

echo "=============================================="
echo "评估结果对比"
echo "=============================================="

echo ""
echo "1. Baseline (历史数据)"
echo "   last@0.25: 0.357"
echo "   Acc@0.50: 0.222"
echo "   hard: 0.323"

echo ""
echo "2. Type Embedding only"
LOG1="/root/autodl-tmp/logs/sr3d_type_embeddings/sr3d_spacy/*/eval_epoch_10.log"
if [[ -f ${LOG1} ]]; then
  echo "   last@0.25: $(grep 'last_.*Acc0.25: Top-1:' ${LOG1} | tail -1)"
  echo "   Acc@0.50: $(grep 'last_.*Acc0.50: Top-1:' ${LOG1} | tail -1)"
  echo "   hard: $(grep 'hard' ${LOG1} | tail -1)"
else
  echo "   (未评估)"
fi

echo ""
echo "3. Type Embedding + Span-Direct (保守版)"
LOG2="/root/autodl-tmp/logs/sr3d_type_embed_span_direct/sr3d_spacy/*/eval_epoch_10.log"
if [[ -f ${LOG2} ]]; then
  echo "   last@0.25: $(grep 'last_.*Acc0.25: Top-1:' ${LOG2} | tail -1)"
  echo "   Acc@0.50: $(grep 'last_.*Acc0.50: Top-1:' ${LOG2} | tail -1)"
  echo "   hard: $(grep 'hard' ${LOG2} | tail -1)"
else
  echo "   (未评估)"
fi

echo ""
echo "4. Type Embedding + Span-Direct (激进版)"
LOG3="/root/autodl-tmp/logs/sr3d_type_embed_span_direct_aggressive/sr3d_spacy/*/eval_epoch_10.log"
if [[ -f ${LOG3} ]]; then
  echo "   last@0.25: $(grep 'last_.*Acc0.25: Top-1:' ${LOG3} | tail -1)"
  echo "   Acc@0.50: $(grep 'last_.*Acc0.50: Top-1:' ${LOG3} | tail -1)"
  echo "   hard: $(grep 'hard' ${LOG3} | tail -1)"
else
  echo "   (未评估)"
fi

echo ""
echo "=============================================="
```

## 常见问题

### Q1: 评估需要多长时间？

**A**: 约 10-15 分钟（取决于 GPU 和数据集大小）

### Q2: 如何确认评估完成？

**A**: 查看日志文件末尾是否有 "评估完成" 标记：
```bash
tail /root/autodl-tmp/logs/.../eval_epoch_10.log
```

### Q3: 评估失败怎么办？

**A**: 检查日志文件中的错误信息：
```bash
grep -i "error\|exception" eval_epoch_10.log
```

常见问题：
- Checkpoint 路径错误
- Config.json 缺失
- 参数不匹配（模型结构与 checkpoint 不一致）

### Q4: 如何评估其他 epoch 的 checkpoint？

**A**: 修改脚本中的 checkpoint 路径：
```bash
bash scripts/eval_sr3d_ckpt10_with_log.sh \
  --checkpoint_path /path/to/ckpt_epoch_5.pth
```

或创建新的评估脚本（如 `eval_sr3d_ckpt5_with_log.sh`）

## 自动化评估（未来改进）

### 评估所有 checkpoint

```bash
#!/bin/bash
# eval_all_checkpoints.sh

for ckpt in /root/autodl-tmp/logs/*/sr3d_spacy/*/ckpt_epoch_*.pth; do
  echo "评估: ${ckpt}"
  bash scripts/eval_sr3d_ckpt10_with_log.sh --checkpoint_path ${ckpt}
done
```

### 定期评估（cron job）

```bash
# 每天凌晨 2 点评估最新的 checkpoint
0 2 * * * cd /home/gb/new\ butd/butd_detr-main && bash scripts/eval_all_checkpoints.sh
```

## 总结

### 改进点

1. ✅ **自动参数恢复**：从 config.json 读取所有参数
2. ✅ **日志持久化**：评估结果保存到文件
3. ✅ **关键指标提取**：自动提取并显示关键指标
4. ✅ **灵活性**：支持命令行指定 checkpoint

### 使用建议

1. **每次训练后立即评估**：确保结果可追溯
2. **保留评估日志**：方便后续对比和分析
3. **批量评估**：对比不同配置的效果
4. **定期备份**：评估日志和 checkpoint 一起备份

---

**当前评估状态**：正在运行中...

**预计完成时间**：约 10-15 分钟

**查看进度**：
```bash
tail -f /root/autodl-tmp/logs/sr3d_type_embed_span_direct_aggressive/sr3d_spacy/1772469891/eval_epoch_10.log
```
