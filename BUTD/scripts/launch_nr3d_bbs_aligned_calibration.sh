#!/bin/bash
set -euo pipefail

cd '/home/gb/new butd/butd_detr-main'
export PATH='/root/miniconda3/envs/bdetr/bin':$PATH
export LD_LIBRARY_PATH='/root/miniconda3/envs/bdetr/lib/python3.7/site-packages/torch/lib:/root/miniconda3/envs/bdetr/lib'
export PYTHONPATH='/home/gb/new butd/butd_detr-main:/home/gb/new butd/butd_detr-main/pointnet2'
export DIAG=1
export NMV2_BATCH_SIZE=96
export NMV2_MAX_EPOCH=2
export NMV2_VAL_FREQ=1
export NMV2_SAVE_FREQ=1000
export NMV2_PRINT_FREQ=25
export NMV2_LOG_ROOT='/home/gb/new butd/butd_detr-main/logs/butd_universal_target/nr3d_bbs_aligned_querygate_calibration_bs96_e2'
export EXTRA_ARGS='--checkpoint_path logs/butd_universal_target/officialdet_finetune_e1/nr3d/03_full_sacr_rapf_qahnl/nr3d_spacy/1786638278/ckpt_epoch_1.pth --universal_modules_train_only --universal_modules_lr 0.0001 --best_checkpoint_only --best_checkpoint_metric last__bbs_acc'
bash scripts/new_method_v2/nr3d/03_full_sacr_rapf_qahnl_nr3d.sh \
  > logs/butd_universal_target/nr3d_bbs_aligned_querygate_calibration_bs96_e2.launch.log 2>&1
