# S2S-ACD-DHC Experiment Scripts

This directory contains the primary block training scripts for the S2S-ACD-DHC experiment plan.
Scripts are organized by dataset under `scripts/sr3d`, `scripts/nr3d`, and `scripts/scanrefer`.
The shared launcher `scripts/run_grounding_block.sh` keeps the base training configuration aligned with the official BeaUTyDETR script.

## Stage Layout

- `scripts/scanrefer/onestage` contains ScanRefer one-stage grounding scripts.
- `scripts/scanrefer/two-stage` contains ScanRefer detector-based two-stage scripts.
- `scripts/sr3d` and `scripts/nr3d` now use two-stage GT-box settings via `--butd_gt` and do not use detector boxes.

## Experiment Blocks

### Block 0: Baseline Sanity Check
Lock baseline and comparison points before adding new structure.

- `sr3d/block0_baseline_sr3d.sh` - Absolute baseline on SR3D
- `nr3d/block0_baseline_nr3d.sh` - Absolute baseline on NR3D
- `scanrefer/onestage/block0_baseline_scanrefer.sh` - One-stage baseline on ScanRefer
- `scanrefer/two-stage/block0_baseline_scanrefer.sh` - Two-stage detector baseline on ScanRefer

### Block 1: S2S Only
Verify that slot memory alone is clean and non-destructive.

- `sr3d/block1_s2s_only_sr3d.sh` - S2S structured slots only on SR3D
- `nr3d/block1_s2s_only_nr3d.sh` - S2S structured slots only on NR3D
- `scanrefer/onestage/block1_s2s_only_scanrefer.sh` - One-stage S2S structured slots on ScanRefer
- `scanrefer/two-stage/block1_s2s_only_scanrefer.sh` - Two-stage S2S structured slots on ScanRefer

### Block 2: S2S + Late ACD
Add anchor-conditioned reasoning on top of structured slots.

- `sr3d/block2_s2s_acd_sr3d.sh` - S2S + Late ACD on SR3D
- `nr3d/block2_s2s_acd_nr3d.sh` - S2S + Late ACD on NR3D
- `scanrefer/onestage/block2_s2s_acd_scanrefer.sh` - One-stage S2S + Late ACD on ScanRefer
- `scanrefer/two-stage/block2_s2s_acd_scanrefer.sh` - Two-stage S2S + Late ACD on ScanRefer

### Block 3: ACD Geometry Ablation
Remove pairwise geometry while keeping the rest of ACD intact.

- `sr3d/block3_acd_no_geometry_sr3d.sh` - No-geometry ACD on SR3D
- `scanrefer/onestage/block3_acd_no_geometry_scanrefer.sh` - One-stage no-geometry ACD on ScanRefer
- `scanrefer/two-stage/block3_acd_no_geometry_scanrefer.sh` - Two-stage no-geometry ACD on ScanRefer

### Block 4: Confidence Fusion
Add confidence-aware structured fusion.

- `sr3d/block4_s2s_acd_conf_fusion_sr3d.sh` - Confidence fusion on SR3D
- `nr3d/block4_s2s_acd_conf_fusion_nr3d.sh` - Confidence fusion on NR3D
- `scanrefer/onestage/block4_s2s_acd_conf_fusion_scanrefer.sh` - One-stage confidence fusion on ScanRefer
- `scanrefer/two-stage/block4_s2s_acd_conf_fusion_scanrefer.sh` - Two-stage confidence fusion on ScanRefer

### Block 5: DHC Introduction
Add decomposition-guided hard negatives and consistency learning.

- `sr3d/block5_s2s_acd_dhc_sr3d.sh` - S2S + ACD + DHC on SR3D
- `nr3d/block5_s2s_acd_dhc_nr3d.sh` - S2S + ACD + DHC on NR3D
- `scanrefer/onestage/block5_s2s_acd_dhc_scanrefer.sh` - One-stage S2S + ACD + DHC on ScanRefer
- `scanrefer/two-stage/block5_s2s_acd_dhc_scanrefer.sh` - Two-stage S2S + ACD + DHC on ScanRefer

### Block 6: Full Method
Run the main full-method comparison setting.

- `sr3d/block6_full_method_sr3d.sh` - Full method on SR3D
- `nr3d/block6_full_method_nr3d.sh` - Full method on NR3D
- `scanrefer/onestage/block6_full_method_scanrefer.sh` - One-stage full method on ScanRefer
- `scanrefer/two-stage/block6_full_method_scanrefer.sh` - Two-stage full method on ScanRefer

### Block 7: ScanRefer Transfer Slot
Keep the original numbering-compatible ScanRefer transfer entrypoint.

- `scanrefer/onestage/block7_scanrefer_transfer.sh` - One-stage numbering-compatible ScanRefer full-method run
- `scanrefer/two-stage/block7_scanrefer_transfer.sh` - Two-stage numbering-compatible ScanRefer full-method run

## Usage

Run from the project root, for example:

```bash
cd /home/gb/new\ butd/butd_detr-main
./scripts/scanrefer/onestage/block0_baseline_scanrefer.sh
```

## Base Configuration

The primary block scripts now share the official BeaUTyDETR grounding base, with dataset-specific box settings:

- `python -m torch.distributed.launch --nproc_per_node 1`
- `--batch_size 24`
- `--lr_backbone=1e-3 --lr=1e-4`
- `--lr_decay_epochs 30 35`
- `--detect_intermediate --joint_det`
- `--use_soft_token_loss --use_contrastive_align`
- `scanrefer/onestage`: `--detect_intermediate --joint_det --self_attend`
- `scanrefer/two-stage`: `--joint_det --butd --self_attend --augment_det`
- `sr3d` / `nr3d`: `--detect_intermediate --joint_det --butd_gt --self_attend`

You can override `DATA_ROOT`, `PP_CHECKPOINT`, `CUDA_VISIBLE_DEVICES`, and `MASTER_PORT` via environment variables when launching.
