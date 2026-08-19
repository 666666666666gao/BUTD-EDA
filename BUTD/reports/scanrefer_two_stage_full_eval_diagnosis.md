# ScanRefer Two-Stage Full Evaluation Diagnosis

Evaluated all checkpoints found under `logs/new_method_v2/scanrefer/two_stage/05_full_sacr_rapf_qahnl/scanrefer_spacy/1777482858` using the same two-stage full configuration with `--eval_report_diagnostic_scores`.

- stdout log directory: `logs/new_method_v2/scanrefer/two_stage/05_full_sacr_rapf_qahnl_eval_all/scanrefer_spacy/1777482858`
- parsed CSV: `reports/scanrefer_two_stage_full_eval_all.csv`
- completed checkpoints: epoch 20, epoch 25, epoch 30, epoch 35, epoch 40, epoch 45, epoch 50, epoch 55
- best primary checkpoint by full-precision fused `last__bbs_acc0.50_top1`: epoch 50 = 0.333403

## Top-1 Summary

| epoch | fused primary@0.25 | fused primary@0.50 | contrastive bbf@0.50 | diag base@0.50 | diag quality@0.50 | diag structured@0.50 | base - fused@0.50 | quality - fused@0.50 | bbs/bbf disagree | bbs IoU | bbf IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.345078 | 0.228229 | 0.326 | 0.318784 | 0.328040 | 0.043858 | 0.090555 | 0.099811 | 0.6231 | 0.2069 | 0.2885 |
| 25 | 0.393458 | 0.257362 | 0.330 | 0.322465 | 0.332036 | 0.183845 | 0.065103 | 0.074674 | 0.5717 | 0.2336 | 0.2918 |
| 30 | 0.436264 | 0.298485 | 0.342 | 0.333719 | 0.342238 | 0.197833 | 0.035233 | 0.043753 | 0.4353 | 0.2611 | 0.2964 |
| 35 | 0.464346 | 0.319836 | 0.360 | 0.351388 | 0.359802 | 0.207299 | 0.031552 | 0.039966 | 0.4643 | 0.2776 | 0.3077 |
| 40 | 0.459403 | 0.322886 | 0.371 | 0.361170 | 0.367585 | 0.026294 | 0.038284 | 0.044699 | 0.4387 | 0.2769 | 0.3150 |
| 45 | 0.460875 | 0.321414 | 0.366 | 0.358225 | 0.365166 | 0.101283 | 0.036811 | 0.043753 | 0.4127 | 0.2803 | 0.3128 |
| 50 | 0.462663 | 0.333403 | 0.375 | 0.363168 | 0.372949 | 0.079512 | 0.029764 | 0.039546 | 0.4263 | 0.2862 | 0.3181 |
| 55 | 0.470551 | 0.333088 | 0.366 | 0.356437 | 0.371371 | 0.033761 | 0.023349 | 0.038284 | 0.3562 | 0.2890 | 0.3136 |

## Diagnosis

The issue is not that the last decoder layer cannot produce usable boxes. Across all evaluated checkpoints, the same last-layer boxes score substantially better when ranked by contrastive/base or quality scores than when ranked by the fused primary score.

The fused primary score never beats the base diagnostic at Acc@0.50 Top-1: `True`. It also never beats the quality diagnostic at Acc@0.50 Top-1: `True`. The best fused result is epoch 50 at 0.333403, while the same checkpoint has base diagnostic 0.363168 and quality diagnostic 0.372949.

The strongest failure signal is the structured diagnostic score: it is consistently below base/quality and becomes extremely poor at later checkpoints, for example epoch 40 structured@0.50 is 0.026294 and epoch 55 structured@0.50 is 0.033761. Because fused ranking includes the RAPF-gated structured residual, this points to harmful structured residual/ranking injection rather than an attention-head or decoder-layer-count issue.

The top-1 disagreement diagnostics support the same conclusion. Last-layer fused-vs-contrastive top-1 disagreement remains high, from 0.6231 at epoch 20 to 0.3562 at epoch 55, and fused top-1 IoU remains below contrastive top-1 IoU at every checkpoint.

## Interpretation

The quality head is not the main failure: quality-only ranking is the best Acc@0.50 Top-1 source in every evaluated checkpoint. The harmful part is the fused composition, especially the structured residual path as it is currently scored and gated into RAPF fusion.

Auxiliary `0head_` through `4head_` rows are not direct fused-score comparisons in the current evaluator; they use base scoring. Therefore `3head_/4head_ bbs > last_ bbs` mostly reflects score-source mismatch, not attention head count or a need to reduce `--num_decoder_layers`.

## Next Proper Experiment

Do not report `bbf` or quality-only diagnostic as the full-model result. The faithful next step is a matched ablation, not post-processing: train/evaluate two-stage variants that isolate fused-score terms, especially structured residual vs quality contribution, under the same schedule and baseline.
