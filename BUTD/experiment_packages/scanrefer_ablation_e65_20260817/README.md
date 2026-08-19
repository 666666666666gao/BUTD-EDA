# ScanRefer 消融实验复现包（固定 65 轮协议）

本目录集中保存 ScanRefer 全部消融实验的直接启动脚本、串行启动入口和论文表格。外部 BUTD-DETR baseline 只登记原文数据，不重新训练；其余 9 行均从同一官方 detector 初始化独立训练。

> 当前服务器上的正式消融队列已经在运行。不要再次执行 `start_all_in_screen.sh`。本包的启动器带 GPU 占用与重复任务保护，主要用于检查、单行复现或未来整套复现。

## 1. 冻结训练协议

- 数据集：ScanRefer validation，仓库 `scanrefer_spacy` 解析口径。
- 初始化：`/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth`。
- 每个训练行都从官方初始化重新训练，不使用 `--checkpoint_path` 续训。
- 随机种子：0。
- 总轮数：65；每 5 轮验证一次。
- 学习率：MultiStepLR，分别在第 55、60 轮结束后乘 0.1。
- 不启用 early stopping。
- 官方主指标与选权重指标：Overall Acc@0.25，即 `last__bbs_acc0.25_top1`。
- 仅保留该指标严格最优的 `ckpt_best_primary.pth`；最终六项指标必须来自同一最佳权重的 reload evaluation，禁止跨权重拼接。

## 2. 论文主表（当前结果）

数值均为百分比。`‡` 表示外部论文数据；`†` 表示正式训练尚未完成时的当前 strict-best 权重；`–` 表示尚无结果。

| Group | Setting | SACR | RAPF | QAHNL | QAHNL source | Quality | Gate Sup. | Relation | U@0.25 | U@0.50 | M@0.25 | M@0.50 | O@0.25 | O@0.50 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| module | BUTD-DETR (paper)‡ | × | × | × | – | × | × | × | 84.20 | 66.30 | 46.60 | 35.10 | 52.20 | 39.80 |
| module | SACR only | ✓ | × | × | – | × | × | ✓ | – | – | – | – | – | – |
| module | QAHNL only | × | × | ✓ | base | × | × | × | – | – | – | – | – | – |
| module | SACR + QAHNL | ✓ | × | ✓ | structured | × | × | ✓ | – | – | – | – | – | – |
| module | SACR + RAPF | ✓ | ✓ | × | – | ✓ | ✓ | ✓ | – | – | – | – | – | – |
| module | Full model† | ✓ | ✓ | ✓ | fused | ✓ | ✓ | ✓ | 76.32 | 45.24 | 32.61 | 17.64 | 39.14 | 21.76 |
| internal | Full w/o Quality | ✓ | ✓ | ✓ | fused | × | ✓ | ✓ | – | – | – | – | – | – |
| internal | Full w/o Gate supervision | ✓ | ✓ | ✓ | fused | ✓ | × | ✓ | – | – | – | – | – | – |
| internal | Full w/o Relation branch | ✓ | ✓ | ✓ | fused | ✓ | ✓ | × | – | – | – | – | – | – |
| internal | Full with QAHNL base source | ✓ | ✓ | ✓ | base | ✓ | ✓ | ✓ | – | – | – | – | – | – |

表注建议：BUTD-DETR baseline 来自 Jain et al. (ECCV 2022), Table 1 与 Supplementary Table 8；原文使用 ground-truth text labels，因此是非完全同协议的外部参考，不能把与它的差值写成严格同协议因果增益。所有 ours 行使用相同 ScanRefer spaCy 口径、官方初始化、seed 0 和固定 65 轮协议。

RAPF 依赖 SACR 的 structured scores，所以 RAPF-only 以及不含 SACR 的 RAPF+QAHNL 都是结构上无效的组合，不应加入论文表。`Full w/o Relation branch` 仅通过 `--sacr_disable_relation` 关闭 SACR 的关系分支，SACR 结构化属性分支仍保留，RAPF 输入没有被移除；该行不能简称为“w/o SACR”。

可直接粘贴到论文的 LaTeX 版本：

```latex
\begin{table*}[t]
\centering
\caption{Dependency-aware module and internal ablations on ScanRefer. All values are percentages.}
\label{tab:scanrefer_ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc|cc|cc|cc}
\toprule
Setting & SACR & RAPF & QAHNL & Source & Quality & Gate Sup. & Relation & U@.25 & U@.50 & M@.25 & M@.50 & O@.25 & O@.50 \\
\midrule
BUTD-DETR (paper)$^{\ddagger}$ & $\times$ & $\times$ & $\times$ & -- & $\times$ & $\times$ & $\times$ & 84.20 & 66.30 & 46.60 & 35.10 & 52.20 & 39.80 \\
SACR only & $\checkmark$ & $\times$ & $\times$ & -- & $\times$ & $\times$ & $\checkmark$ & -- & -- & -- & -- & -- & -- \\
QAHNL only & $\times$ & $\times$ & $\checkmark$ & base & $\times$ & $\times$ & $\times$ & -- & -- & -- & -- & -- & -- \\
SACR + QAHNL & $\checkmark$ & $\times$ & $\checkmark$ & structured & $\times$ & $\times$ & $\checkmark$ & -- & -- & -- & -- & -- & -- \\
SACR + RAPF & $\checkmark$ & $\checkmark$ & $\times$ & -- & $\checkmark$ & $\checkmark$ & $\checkmark$ & -- & -- & -- & -- & -- & -- \\
Full model$^{\dagger}$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & fused & $\checkmark$ & $\checkmark$ & $\checkmark$ & 76.32 & 45.24 & 32.61 & 17.64 & 39.14 & 21.76 \\
\midrule
Full w/o Quality & $\checkmark$ & $\checkmark$ & $\checkmark$ & fused & $\times$ & $\checkmark$ & $\checkmark$ & -- & -- & -- & -- & -- & -- \\
Full w/o Gate supervision & $\checkmark$ & $\checkmark$ & $\checkmark$ & fused & $\checkmark$ & $\times$ & $\checkmark$ & -- & -- & -- & -- & -- & -- \\
Full w/o Relation branch & $\checkmark$ & $\checkmark$ & $\checkmark$ & fused & $\checkmark$ & $\checkmark$ & $\times$ & -- & -- & -- & -- & -- & -- \\
Full with QAHNL base source & $\checkmark$ & $\checkmark$ & $\checkmark$ & base & $\checkmark$ & $\checkmark$ & $\checkmark$ & -- & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}}
\vspace{2pt}
\parbox{\textwidth}{\footnotesize $^{\ddagger}$External BUTD-DETR paper result using ground-truth text labels; it is not retrained here. $^{\dagger}$Interim strict-best checkpoint; -- denotes a pending valid result. All trained rows use the same ScanRefer spaCy protocol and report one reloaded O@.25-selected checkpoint.}
\end{table*}
```

## 3. 当前正式队列的实际顺序

当前运行中的队列不能在中途改序；它会保证全部模块间消融先于模块内部消融：

| 顺序 | Canonical row | 实验 | 类型 | 当前动作 |
|---:|---|---|---|---|
| 0 | `01_baseline` | BUTD-DETR paper baseline | 外部参考 | 仅登记原文结果，不训练 |
| 1 | `02_full_sacr_rapf_qahnl` | Full model | 模块间 | 已在正式训练 |
| 2 | `03_no_sacr_rapf_qahnl_base` | QAHNL only (base) | 模块间 | 等待 |
| 3 | `04_no_qahnl` | SACR + RAPF | 模块间 | 等待 |
| 4 | `08_sacr_only` | SACR only | 模块间 | 等待 |
| 5 | `09_sacr_qahnl` | SACR + QAHNL (structured) | 模块间 | 等待 |
| 6 | `05_no_quality` | Full w/o Quality | 模块内部 | 等待模块间门完成 |
| 7 | `06_no_gate_supervision` | Full w/o Gate supervision | 模块内部 | 等待 |
| 8 | `07_no_relation` | Full w/o Relation branch | 模块内部 | 等待 |
| 9 | `10_full_qahnl_base_source` | Full with QAHNL base source | 模块内部 | 最后运行 |

## 4. 本复现包的审稿信息优先顺序

`run_all_serial.sh` 用于未来从头复现，顺序按“尽快得到最有解释力的对照”优化，不会修改当前正式队列：

1. Full model；
2. SACR + RAPF（直接检验 Full 中 QAHNL 的边际贡献）；
3. SACR only（给 RAPF 与 QAHNL 两条增量路径提供共同锚点）；
4. SACR + QAHNL (structured)；
5. QAHNL only (base)；
6. Full w/o Quality；
7. Full with QAHNL base source；
8. Full w/o Gate supervision；
9. Full w/o Relation branch。

论文展示顺序不要按时间排列，应使用第 2 节的逻辑顺序：外部 baseline → 单模块 → 合法双模块组合 → Full → 模块内部消融。

## 5. 每一行回答的审稿问题

| 实验 | 具体问题 | 若组件有效，预期现象 |
|---|---|---|
| Full | 三个模块组合是否给出最强完整方法？ | 应优于或接近优于合法子组合 |
| SACR + RAPF | Full 是否需要 QAHNL？ | 若 QAHNL 有效，应低于 Full |
| SACR only | SACR 单独是否形成有效结构化基础？ | 应成为后续 RAPF/QAHNL 增量的稳定锚点 |
| SACR + QAHNL | 不使用 RAPF 时，QAHNL 对 structured scores 是否仍有效？ | 若有效，应高于 SACR only |
| QAHNL only | QAHNL 脱离 SACR/RAPF 是否有独立价值？ | 若其他模块重要，应低于 Full |
| w/o Quality | learned quality signal 是否必要？ | 若必要，应低于 Full |
| QAHNL base source | QAHNL 是否需要 fused evidence？ | 若 fused source 重要，应低于 Full |
| w/o Gate supervision | gate 架构保留时，显式监督是否必要？ | 若必要，应低于 Full |
| w/o Relation branch | SACR 的关系分支是否超出属性分支贡献？ | 若关系建模有效，应低于 Full |

## 6. 脚本与使用方法

- `launch/row02_...sh` 至 `launch/row10_...sh`：9 个可单独直接启动的训练脚本，文件名保留 canonical row id。
- `common.sh`：冻结公共训练协议、GPU 空闲检查和重复任务保护。
- `validate.sh`：只做 Bash 语法与 DRY-RUN 参数审计，不启动训练。
- `run_all_serial.sh`：按第 4 节顺序串行运行 9 个训练行。
- `start_all_in_screen.sh`：在一个 detached screen 中启动整套串行复现。
- `register_external_baseline.sh`：只生成外部 baseline JSON，不训练。

先验证：

```bash
cd '/home/gb/new butd/butd_detr-main/experiment_packages/scanrefer_ablation_e65_20260817'
bash validate.sh
```

查看某一行的完整命令但不启动：

```bash
DRY_RUN=1 bash launch/row04_sacr_rapf_no_qahnl.sh
```

GPU 空闲后，单独直接训练一行：

```bash
CUDA_VISIBLE_DEVICES=0 bash launch/row04_sacr_rapf_no_qahnl.sh
```

GPU 空闲后，未来整套复现：

```bash
CUDA_VISIBLE_DEVICES=0 bash start_all_in_screen.sh
```

默认输出位于：

```text
/home/gb/new butd/butd_detr-main/logs/butd_universal_target/scanrefer_ablation_e65_reproduction/<timestamp>/
```

若 GPU 显存占用不低于 500 MiB，或同一 canonical row 已有活动训练进程，直接启动器会 fail-closed，防止与当前正式队列重复训练。

## 7. 论文解释边界

- 只有同协议训练行之间适合做严格组件归因；外部 baseline 仅作参考。
- 当前是单 seed，不能声称统计稳健性。
- 负结果与持平结果也必须保留，不能按结果删除行。
- 若论文进一步声称 RAPF 优于简单平均/拼接融合，仍需额外的 simple-fusion replacement；当前 9 行不支持该更强主张。
- 若论文进一步声称 QAHNL 的 fused source 优于 structured source，仍需 Full+QAHNL(structured-source) 行；当前只能比较 Full fused 与 Full base。

