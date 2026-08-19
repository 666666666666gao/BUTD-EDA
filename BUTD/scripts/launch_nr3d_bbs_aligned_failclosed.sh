#!/bin/bash
set -euo pipefail

cd '/home/gb/new butd/butd_detr-main'
export PATH='/root/miniconda3/envs/bdetr/bin':$PATH
export LD_LIBRARY_PATH='/root/miniconda3/envs/bdetr/lib/python3.7/site-packages/torch/lib:/root/miniconda3/envs/bdetr/lib'
export PYTHONPATH='/home/gb/new butd/butd_detr-main:/home/gb/new butd/butd_detr-main/pointnet2'
export DIAG=1
export NMV2_LOG_ROOT='/home/gb/new butd/butd_detr-main/logs/butd_universal_target/nr3d_bbs_aligned_failclosed'
export EXTRA_ARGS='--eval --checkpoint_path logs/butd_universal_target/officialdet_finetune_e1/nr3d/03_full_sacr_rapf_qahnl/nr3d_spacy/1786638278/ckpt_epoch_1.pth --rapf_quality_weight 0 --rapf_struct_residual_clip 0'
bash scripts/new_method_v2/nr3d/03_full_sacr_rapf_qahnl_nr3d.sh \
  > logs/butd_universal_target/nr3d_bbs_aligned_failclosed.launch.log 2>&1
