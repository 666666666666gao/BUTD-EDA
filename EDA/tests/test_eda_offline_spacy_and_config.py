import os
import sys
import tempfile
import unittest
import csv
import json
from pathlib import Path
import inspect
from types import SimpleNamespace

import torch
import torch.nn as nn


EDA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDA_ROOT))


class EDAOfflineSpacyAndConfigTest(unittest.TestCase):
    def test_blend_semantic_rerank_outputs_averages_only_residuals(self):
        from models.semantic_rerank_head import (
            blend_semantic_rerank_outputs,
        )

        base = torch.tensor([[1.0, 2.0]])
        primary_residual = torch.tensor([[0.2, -0.4]])
        auxiliary_residual = torch.tensor([[-0.2, 0.8]])
        primary = {
            'semantic_rerank_base_scores': base,
            'semantic_rerank_residual': primary_residual,
            'semantic_rerank_scores': base + primary_residual,
        }
        auxiliary = {
            'semantic_rerank_base_scores': base.clone(),
            'semantic_rerank_residual': auxiliary_residual,
            'semantic_rerank_scores': base + auxiliary_residual,
        }

        output = blend_semantic_rerank_outputs(
            primary, auxiliary, auxiliary_weight=0.25
        )
        expected = 0.75 * primary_residual + 0.25 * auxiliary_residual
        self.assertTrue(torch.equal(
            output['semantic_rerank_residual'], expected
        ))
        self.assertTrue(torch.equal(
            output['semantic_rerank_scores'], base + expected
        ))
        self.assertTrue(torch.equal(
            output['semantic_rerank_primary_residual'], primary_residual
        ))
        self.assertTrue(torch.equal(
            output['semantic_rerank_aux_residual'], auxiliary_residual
        ))

    def test_blend_semantic_rerank_outputs_rejects_different_bases(self):
        from models.semantic_rerank_head import (
            blend_semantic_rerank_outputs,
        )

        primary = {
            'semantic_rerank_base_scores': torch.zeros(1, 2),
            'semantic_rerank_residual': torch.zeros(1, 2),
        }
        auxiliary = {
            'semantic_rerank_base_scores': torch.ones(1, 2),
            'semantic_rerank_residual': torch.zeros(1, 2),
        }
        with self.assertRaisesRegex(ValueError, 'identical base scores'):
            blend_semantic_rerank_outputs(primary, auxiliary)

    def test_load_semantic_rerank_aux_checkpoint_copies_only_rerank_head(self):
        from main_utils import load_semantic_rerank_aux_checkpoint

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.semantic_rerank_head = nn.Linear(3, 1)
                self.semantic_rerank_aux_head = nn.Linear(3, 1)
                self.backbone = nn.Linear(3, 3)

        model = ToyModel()
        primary_before = {
            key: value.detach().clone()
            for key, value in model.semantic_rerank_head.state_dict().items()
        }
        backbone_before = {
            key: value.detach().clone()
            for key, value in model.backbone.state_dict().items()
        }
        source = nn.Linear(3, 1)
        with torch.no_grad():
            source.weight.fill_(2.5)
            source.bias.fill_(-0.75)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'aux.pth')
            torch.save({
                'model': {
                    'module.semantic_rerank_head.' + key: value
                    for key, value in source.state_dict().items()
                },
            }, checkpoint_path)
            args = SimpleNamespace(
                semantic_rerank_aux_checkpoint=checkpoint_path,
                eval=True,
            )
            loaded = load_semantic_rerank_aux_checkpoint(
                args, model
            )

        self.assertEqual(loaded, 2)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(
                model.semantic_rerank_aux_head.state_dict()[key], value
            ))
        for key, value in primary_before.items():
            self.assertTrue(torch.equal(
                model.semantic_rerank_head.state_dict()[key], value
            ))
        for key, value in backbone_before.items():
            self.assertTrue(torch.equal(
                model.backbone.state_dict()[key], value
            ))

    def test_semantic_rerank_head_snapshot_is_lightweight_and_reloadable(self):
        from main_utils import (
            load_semantic_rerank_aux_checkpoint,
            save_semantic_rerank_head_snapshot,
        )

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.semantic_rerank_head = nn.Linear(3, 1)
                self.semantic_rerank_aux_head = nn.Linear(3, 1)
                self.backbone = nn.Linear(3, 3)

        model = ToyModel()
        with torch.no_grad():
            model.semantic_rerank_head.weight.fill_(1.25)
            model.semantic_rerank_head.bias.fill_(-0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(log_dir=tmpdir)
            path = save_semantic_rerank_head_snapshot(args, model, 25)
            checkpoint = torch.load(path, map_location='cpu')
            self.assertEqual(checkpoint['step'], 25)
            self.assertEqual(set(checkpoint['model']), {
                'module.semantic_rerank_head.weight',
                'module.semantic_rerank_head.bias',
            })
            load_args = SimpleNamespace(
                semantic_rerank_aux_checkpoint=path,
                eval=True,
            )
            loaded = load_semantic_rerank_aux_checkpoint(load_args, model)

        self.assertEqual(loaded, 2)
        for key, value in model.semantic_rerank_head.state_dict().items():
            self.assertTrue(torch.equal(
                model.semantic_rerank_aux_head.state_dict()[key], value
            ))

    def test_scanrefer_spacy_uses_offline_annotation_path(self):
        from src.joint_det_dataset import Joint3DDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            scanrefer_dir = Path(tmpdir) / "scanrefer"
            scanrefer_dir.mkdir()
            base = scanrefer_dir / "ScanRefer_filtered_train_spacy.json"
            base.write_text("[]")

            path = Joint3DDataset._scanrefer_annotation_path(
                tmpdir, "train", use_spacy=True
            )

        self.assertEqual(path, str(base))

    def test_scanrefer_spacy_prefers_refined_offline_annotation_path(self):
        from src.joint_det_dataset import Joint3DDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            scanrefer_dir = Path(tmpdir) / "scanrefer"
            scanrefer_dir.mkdir()
            refined = scanrefer_dir / "ScanRefer_filtered_train_spacy_refined.json"
            refined.write_text("[]")
            (scanrefer_dir / "ScanRefer_filtered_train_spacy.json").write_text("[]")

            path = Joint3DDataset._scanrefer_annotation_path(
                tmpdir, "train", use_spacy=True
            )

        self.assertEqual(path, str(refined))

    def test_referit3d_spacy_prefers_refined_csv_path(self):
        from src.joint_det_dataset import Joint3DDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            refer_dir = Path(tmpdir) / "refer_it_3d"
            refer_dir.mkdir()
            refined = refer_dir / "nr3d_spacy_refined.csv"
            refined.write_text("")
            (refer_dir / "nr3d_spacy.csv").write_text("")

            dataset = Joint3DDataset.__new__(Joint3DDataset)
            dataset.data_path = tmpdir
            path = dataset._referit3d_csv_path("nr3d_spacy")

        self.assertEqual(path, str(refined))

    def test_scanrefer_spacy_graph_is_built_without_online_parser(self):
        from src.joint_det_dataset import Joint3DDataset

        anno = {
            "target_slot": {
                "text": "the red chair",
                "start": 10,
                "end": 23,
            },
            "attr_slot": {
                "items": [
                    {"text": "red", "start": 14, "end": 17},
                ],
            },
            "rel_slots": [
                {"text": "next to", "start": 24, "end": 31},
            ],
            "anchor_slots": [
                {"text": "the table", "start": 32, "end": 41},
            ],
        }

        graph_node, graph_edge, auxi_entity = (
            Joint3DDataset._build_graph_from_spacy_slots(anno)
        )

        self.assertEqual(graph_edge, [])
        self.assertEqual(graph_node[0]["target_char_span"], [[10, 23]])
        self.assertEqual(graph_node[0]["mod_char_span"], [[14, 17]])
        self.assertEqual(graph_node[0]["rel_char_span"], [[24, 31]])
        self.assertEqual(auxi_entity["target_char_span"], [[32, 41]])

    def test_spacy_refinement_drops_missing_anchor_relation(self):
        from src.joint_det_dataset import Joint3DDataset

        refined = Joint3DDataset._refine_spacy_decomposition_fields({
            "dataset": "scanrefer_spacy",
            "utterance": "the chair left of the table",
            "rel_slots": [{"text": "left of", "start": 10, "end": 17}],
            "anchor_slots": [],
            "coverage_stats": {},
            "parse_confidence": 1.0,
        })

        self.assertEqual(refined["rel_slots"], [])
        self.assertEqual(refined["anchor_slots"], [])
        self.assertTrue(refined["decomp_weak_generic_mask"])
        self.assertEqual(
            refined["coverage_stats"]["refined_missing_anchor_relation_count"],
            1,
        )
        self.assertLess(refined["parse_confidence"], 1.0)

    def test_spacy_refinement_marks_scene_frame_view_as_weak_not_relation(self):
        from src.joint_det_dataset import Joint3DDataset

        refined = Joint3DDataset._refine_spacy_decomposition_fields({
            "dataset": "nr3d_spacy",
            "utterance": "select the cabinet on the left wall",
            "rel_slots": [{"text": "left wall", "start": 26, "end": 35}],
            "anchor_slots": [{"text": "wall", "start": 31, "end": 35}],
            "coverage_stats": {},
            "parse_confidence": 0.9,
        })

        self.assertEqual(refined["rel_slots"], [])
        self.assertEqual(refined["anchor_slots"], [])
        self.assertTrue(refined["decomp_weak_generic_mask"])
        self.assertEqual(
            refined["coverage_stats"]["refined_scene_frame_relation_count"],
            1,
        )

    def test_spacy_refinement_keeps_anchor_backed_proximity_relation(self):
        from src.joint_det_dataset import Joint3DDataset

        refined = Joint3DDataset._refine_spacy_decomposition_fields({
            "dataset": "sr3d_spacy",
            "utterance": "pick the chair next to the table",
            "rel_slots": [{"text": "next to", "start": 15, "end": 22}],
            "anchor_slots": [{"text": "the table", "start": 23, "end": 32}],
            "coverage_stats": {},
            "parse_confidence": 0.8,
        })

        self.assertEqual(refined["rel_slots"][0]["text"], "next to")
        self.assertEqual(refined["rel_slots"][0]["relation_type"], "proximity")
        self.assertEqual(refined["anchor_slots"][0]["text"], "the table")
        self.assertFalse(refined["decomp_weak_generic_mask"])
        self.assertEqual(
            refined["coverage_stats"]["refined_relation_count"],
            1,
        )

    def test_spacy_refinement_marks_relation_free_parse_noise_train_only(self):
        from src.joint_det_dataset import Joint3DDataset

        refined = Joint3DDataset._refine_spacy_decomposition_fields({
            "dataset": "scanrefer_spacy",
            "description": "the chair second from the right",
            "rel_slots": [],
            "anchor_slots": [],
            "attr_slot": {"items": []},
            "coverage_stats": {"candidate_relation_count": 1},
            "parse_confidence": 1.0,
        })

        self.assertTrue(refined["decomp_train_global_only_mask"])
        self.assertTrue(refined["decomp_train_weak_generic_mask"])
        self.assertFalse(refined.get("decomp_global_only_mask", False))
        self.assertEqual(
            refined["decomp_train_global_only_reason"],
            "relation_free_raw_view_parse_noise",
        )
        self.assertEqual(
            refined["coverage_stats"]["refined_train_global_only_signal_count"],
            1,
        )

    def test_refine_spacy_csv_writes_train_only_mask_columns(self):
        from scripts.refine_spacy_decomposition import _refine_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "nr3d_spacy.csv"
            output_path = root / "nr3d_spacy_refined.csv"
            with input_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "utterance",
                        "entities",
                        "attributes",
                        "target_slot",
                        "attr_slot",
                        "rel_slots",
                        "anchor_slots",
                        "coverage_stats",
                        "slot_mask",
                        "parse_confidence",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "utterance": "the pillow is the top pillow on the couch",
                    "entities": "[]",
                    "attributes": "[]",
                    "target_slot": "{}",
                    "attr_slot": json.dumps({
                        "items": [{"text": "top", "type": "spatial_attribute"}]
                    }),
                    "rel_slots": "[]",
                    "anchor_slots": "[]",
                    "coverage_stats": json.dumps({"spatial_attribute_rows": 1}),
                    "slot_mask": "{}",
                    "parse_confidence": "1.0",
                })

            stats = _refine_csv(input_path, output_path, "nr3d_spacy")
            with output_path.open(newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(stats["train_global_only_rows"], 1)
        self.assertIn("decomp_train_global_only_mask", rows[0])
        self.assertIn("decomp_train_weak_generic_mask", rows[0])
        self.assertEqual(rows[0]["decomp_train_global_only_mask"], "true")
        self.assertEqual(
            rows[0]["decomp_train_global_only_reason"],
            "relation_free_spatial_attribute_parse_noise",
        )

    def test_parser_accepts_sacr_rapf_qahnl_flags(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "train_dist_mod.py",
                "--dataset",
                "scanrefer_spacy",
                "--test_dataset",
                "scanrefer_spacy",
                "--use_structured_slots",
                "--use_sacr",
                "--use_rapf",
                "--use_reliability_gate",
                "--use_quality_head",
                "--rapf_use_quality",
                "--use_qahnl",
                "--eval_use_fused_scores",
                "--eval_use_fused_semantic_scores",
                "--freeze_base_train_heads",
                "--freeze_base_train_align_heads",
                "--freeze_base_train_box_align_heads",
                "--freeze_base_train_component_head",
                "--freeze_base_train_rerank_head",
                "--freeze_base_train_rerank_align_heads",
                "--eval",
                "--use_semantic_rerank_head",
                "--semantic_rerank_aux_checkpoint",
                "/tmp/aux.pth",
                "--semantic_rerank_aux_weight",
                "0.4",
                "--spacy_relation_free_yaw_only_aug",
                "--spacy_relation_free_view_guard_aug",
                "--spacy_relation_free_stable_yaw_aug",
                "--spacy_relation_free_view_small_yaw_aug",
                "--spacy_relation_free_rawview_global_only_train",
                "--spacy_relation_free_none_aug",
                "--spacy_relation_free_compass_guard_aug",
            ]

            args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.dataset, ["scanrefer_spacy"])
        self.assertEqual(args.test_dataset, "scanrefer_spacy")
        self.assertTrue(args.use_structured_slots)
        self.assertTrue(args.use_sacr)
        self.assertTrue(args.use_rapf)
        self.assertTrue(args.use_reliability_gate)
        self.assertTrue(args.use_quality_head)
        self.assertTrue(args.rapf_use_quality)
        self.assertTrue(args.use_qahnl)
        self.assertTrue(args.eval_use_fused_scores)
        self.assertTrue(args.eval_use_fused_semantic_scores)
        self.assertTrue(args.freeze_base_train_heads)
        self.assertTrue(args.freeze_base_train_align_heads)
        self.assertTrue(args.freeze_base_train_box_align_heads)
        self.assertTrue(args.freeze_base_train_component_head)
        self.assertTrue(args.freeze_base_train_rerank_head)
        self.assertTrue(args.freeze_base_train_rerank_align_heads)
        self.assertEqual(args.semantic_rerank_aux_checkpoint, "/tmp/aux.pth")
        self.assertEqual(args.semantic_rerank_aux_weight, 0.4)
        self.assertTrue(args.spacy_relation_free_yaw_only_aug)
        self.assertTrue(args.spacy_relation_free_view_guard_aug)
        self.assertTrue(args.spacy_relation_free_stable_yaw_aug)
        self.assertTrue(args.spacy_relation_free_view_small_yaw_aug)
        self.assertTrue(args.spacy_relation_free_rawview_global_only_train)
        self.assertTrue(args.spacy_relation_free_none_aug)
        self.assertTrue(args.spacy_relation_free_compass_guard_aug)

    def test_parser_defaults_aux_contrastive_base_off_and_accepts_flag(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_dist_mod.py"]
            default_args = parse_option()

            sys.argv = [
                "train_dist_mod.py",
                "--aux_scores_use_contrastive_base",
            ]
            enabled_args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertFalse(default_args.aux_scores_use_contrastive_base)
        self.assertTrue(enabled_args.aux_scores_use_contrastive_base)

    def test_parser_defaults_aux_semantic_eval_base_off_and_accepts_flag(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_dist_mod.py"]
            default_args = parse_option()

            sys.argv = [
                "train_dist_mod.py",
                "--aux_scores_use_semantic_eval_base",
            ]
            enabled_args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertFalse(default_args.aux_scores_use_semantic_eval_base)
        self.assertTrue(enabled_args.aux_scores_use_semantic_eval_base)

    def test_parser_defaults_train_partial_checkpoint_init_off_and_accepts_flag(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_dist_mod.py"]
            default_args = parse_option()

            sys.argv = [
                "train_dist_mod.py",
                "--train_partial_checkpoint_init",
            ]
            enabled_args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertFalse(default_args.train_partial_checkpoint_init)
        self.assertTrue(enabled_args.train_partial_checkpoint_init)

    def test_parser_defaults_sem_iou_rank_off_and_accepts_config(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_dist_mod.py"]
            default_args = parse_option()

            sys.argv = [
                "train_dist_mod.py",
                "--use_sem_iou_rank",
                "--sem_iou_rank_loss_weight",
                "0.07",
                "--sem_iou_rank_pos_iou_thresh",
                "0.5",
                "--sem_iou_rank_neg_iou_thresh",
                "0.2",
                "--sem_iou_rank_margin",
                "0.12",
            ]
            enabled_args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertFalse(default_args.use_sem_iou_rank)
        self.assertTrue(enabled_args.use_sem_iou_rank)
        self.assertEqual(enabled_args.sem_iou_rank_loss_weight, 0.07)
        self.assertEqual(enabled_args.sem_iou_rank_pos_iou_thresh, 0.5)
        self.assertEqual(enabled_args.sem_iou_rank_neg_iou_thresh, 0.2)
        self.assertEqual(enabled_args.sem_iou_rank_margin, 0.12)

    def test_parser_defaults_sem_iou_listwise_off_and_accepts_config(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_dist_mod.py"]
            default_args = parse_option()

            sys.argv = [
                "train_dist_mod.py",
                "--use_sem_iou_listwise",
                "--sem_iou_listwise_loss_weight",
                "0.09",
                "--sem_iou_listwise_topk",
                "24",
                "--sem_iou_listwise_score_temperature",
                "0.25",
                "--sem_iou_listwise_target_iou_power",
                "3.0",
                "--sem_iou_listwise_high_iou_weight",
                "2.5",
            ]
            enabled_args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertFalse(default_args.use_sem_iou_listwise)
        self.assertTrue(enabled_args.use_sem_iou_listwise)
        self.assertEqual(enabled_args.sem_iou_listwise_loss_weight, 0.09)
        self.assertEqual(enabled_args.sem_iou_listwise_topk, 24)
        self.assertEqual(enabled_args.sem_iou_listwise_score_temperature, 0.25)
        self.assertEqual(enabled_args.sem_iou_listwise_target_iou_power, 3.0)
        self.assertEqual(enabled_args.sem_iou_listwise_high_iou_weight, 2.5)

    def test_parser_defaults_sem_iou_top1_off_and_accepts_config(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_dist_mod.py"]
            default_args = parse_option()

            sys.argv = [
                "train_dist_mod.py",
                "--use_sem_iou_top1",
                "--sem_iou_top1_loss_weight",
                "0.03",
                "--sem_iou_top1_pos_iou_thresh",
                "0.45",
                "--sem_iou_top1_iou_gap",
                "0.08",
                "--sem_iou_top1_margin",
                "0.12",
                "--sem_iou_top1_temperature",
                "0.4",
            ]
            enabled_args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertFalse(default_args.use_sem_iou_top1)
        self.assertTrue(enabled_args.use_sem_iou_top1)
        self.assertEqual(enabled_args.sem_iou_top1_loss_weight, 0.03)
        self.assertEqual(enabled_args.sem_iou_top1_pos_iou_thresh, 0.45)
        self.assertEqual(enabled_args.sem_iou_top1_iou_gap, 0.08)
        self.assertEqual(enabled_args.sem_iou_top1_margin, 0.12)
        self.assertEqual(enabled_args.sem_iou_top1_temperature, 0.4)

    def test_parser_defaults_sem_eval_margin_off_and_accepts_config(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_dist_mod.py"]
            default_args = parse_option()

            sys.argv = [
                "train_dist_mod.py",
                "--use_sem_eval_margin",
                "--sem_eval_margin_loss_weight",
                "0.03",
                "--sem_eval_margin_min_pos_iou",
                "0.5",
                "--sem_eval_margin_neg_iou_thresh",
                "0.45",
                "--sem_eval_margin_num_hard_neg",
                "4",
                "--sem_eval_margin_margin",
                "0.08",
                "--sem_eval_margin_temperature",
                "0.4",
            ]
            enabled_args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertFalse(default_args.use_sem_eval_margin)
        self.assertTrue(enabled_args.use_sem_eval_margin)
        self.assertEqual(enabled_args.sem_eval_margin_loss_weight, 0.03)
        self.assertEqual(enabled_args.sem_eval_margin_min_pos_iou, 0.5)
        self.assertEqual(enabled_args.sem_eval_margin_neg_iou_thresh, 0.45)
        self.assertEqual(enabled_args.sem_eval_margin_num_hard_neg, 4)
        self.assertEqual(enabled_args.sem_eval_margin_margin, 0.08)
        self.assertEqual(enabled_args.sem_eval_margin_temperature, 0.4)

    def test_parser_defaults_sem_align_eval_weights_off_and_accepts_flag(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_dist_mod.py"]
            default_args = parse_option()

            sys.argv = [
                "train_dist_mod.py",
                "--sem_align_use_eval_weights",
            ]
            enabled_args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertFalse(default_args.sem_align_use_eval_weights)
        self.assertTrue(enabled_args.sem_align_use_eval_weights)

    def test_parser_defaults_semantic_rerank_off_and_accepts_config(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_dist_mod.py"]
            default_args = parse_option()

            sys.argv = [
                "train_dist_mod.py",
                "--use_semantic_rerank_head",
                "--eval_use_semantic_rerank_scores",
                "--semantic_rerank_loss_weight",
                "0.04",
                "--semantic_rerank_listwise_weight",
                "0.25",
                "--semantic_rerank_threshold_mass_weight",
                "1.5",
                "--semantic_rerank_failure_margin_weight",
                "0.75",
                "--semantic_rerank_failure_margin",
                "0.12",
                "--semantic_rerank_train_use_support_scores",
                "--semantic_rerank_topk",
                "12",
                "--semantic_rerank_residual_scale",
                "0.2",
                "--semantic_rerank_temperature",
                "0.35",
            ]
            enabled_args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertFalse(default_args.use_semantic_rerank_head)
        self.assertFalse(default_args.eval_use_semantic_rerank_scores)
        self.assertEqual(default_args.eval_diagnostic_dump_path, "")
        self.assertEqual(default_args.semantic_rerank_listwise_weight, 1.0)
        self.assertEqual(
            default_args.semantic_rerank_threshold_mass_weight, 0.0
        )
        self.assertEqual(
            default_args.semantic_rerank_failure_margin_weight, 0.0
        )
        self.assertEqual(default_args.semantic_rerank_failure_margin, 0.1)
        self.assertFalse(
            default_args.semantic_rerank_train_use_support_scores
        )
        self.assertTrue(enabled_args.use_semantic_rerank_head)
        self.assertTrue(enabled_args.eval_use_semantic_rerank_scores)
        self.assertEqual(enabled_args.semantic_rerank_loss_weight, 0.04)
        self.assertEqual(enabled_args.semantic_rerank_listwise_weight, 0.25)
        self.assertEqual(
            enabled_args.semantic_rerank_threshold_mass_weight, 1.5
        )
        self.assertEqual(
            enabled_args.semantic_rerank_failure_margin_weight, 0.75
        )
        self.assertEqual(enabled_args.semantic_rerank_failure_margin, 0.12)
        self.assertTrue(
            enabled_args.semantic_rerank_train_use_support_scores
        )
        self.assertEqual(enabled_args.semantic_rerank_topk, 12)
        self.assertEqual(enabled_args.semantic_rerank_residual_scale, 0.2)
        self.assertEqual(enabled_args.semantic_rerank_temperature, 0.35)

    def test_parser_defaults_semantic_component_calibration_off_and_accepts_config(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_dist_mod.py"]
            default_args = parse_option()

            sys.argv = [
                "train_dist_mod.py",
                "--use_semantic_component_calibration",
                "--eval_use_semantic_component_scores",
                "--semantic_component_loss_weight",
                "0.03",
                "--semantic_component_topk",
                "12",
                "--semantic_component_temperature",
                "0.75",
                "--semantic_component_target_iou_power",
                "4.0",
                "--semantic_component_min_target_iou",
                "0.25",
                "--semantic_component_max_delta",
                "0.4",
                "--semantic_component_use_eda_score",
                "--semantic_component_extra_max_weight",
                "0.3",
                "--semantic_component_hard_sample_weight",
                "0.6",
                "--semantic_component_multi_sample_weight",
                "0.2",
            ]
            enabled_args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertFalse(default_args.use_semantic_component_calibration)
        self.assertFalse(default_args.eval_use_semantic_component_scores)
        self.assertTrue(enabled_args.use_semantic_component_calibration)
        self.assertTrue(enabled_args.eval_use_semantic_component_scores)
        self.assertEqual(enabled_args.semantic_component_loss_weight, 0.03)
        self.assertEqual(enabled_args.semantic_component_topk, 12)
        self.assertEqual(enabled_args.semantic_component_temperature, 0.75)
        self.assertEqual(default_args.semantic_component_target_iou_power, 2.0)
        self.assertEqual(enabled_args.semantic_component_target_iou_power, 4.0)
        self.assertEqual(default_args.semantic_component_min_target_iou, 0.01)
        self.assertEqual(enabled_args.semantic_component_min_target_iou, 0.25)
        self.assertEqual(enabled_args.semantic_component_max_delta, 0.4)
        self.assertFalse(default_args.semantic_component_use_eda_score)
        self.assertTrue(enabled_args.semantic_component_use_eda_score)
        self.assertEqual(enabled_args.semantic_component_extra_max_weight, 0.3)
        self.assertEqual(enabled_args.semantic_component_hard_sample_weight, 0.6)
        self.assertEqual(enabled_args.semantic_component_multi_sample_weight, 0.2)

    def test_parser_defaults_semantic_support_off_and_accepts_config(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_dist_mod.py"]
            default_args = parse_option()

            sys.argv = [
                "train_dist_mod.py",
                "--use_semantic_support_adapter",
                "--eval_use_semantic_support_scores",
                "--semantic_support_overlap_weight",
                "0.6",
                "--semantic_support_position_weight",
                "0.1",
                "--semantic_support_overlap_power",
                "0.75",
                "--semantic_support_use_learned_gate",
                "--semantic_support_gate_hidden_dim",
                "12",
                "--semantic_support_gate_max",
                "2.5",
                "--semantic_support_gate_use_query_features",
                "--semantic_support_gate_loss_weight",
                "0.8",
                "--semantic_support_gate_loss_beta",
                "0.2",
                "--use_semantic_threshold_head",
                "--semantic_threshold_hidden_dim",
                "48",
                "--semantic_threshold_residual_scale",
                "0.2",
                "--semantic_threshold_loss_weight",
                "0.7",
                "--semantic_threshold_bce_weight",
                "0.2",
                "--semantic_threshold_pairwise_weight",
                "0.8",
                "--semantic_threshold_pairwise_margin",
                "0.3",
                "--semantic_threshold_pairwise_temperature",
                "0.4",
                "--semantic_threshold_pairwise_high_weight",
                "1.25",
                "--semantic_threshold_focal_gamma",
                "1.5",
                "--freeze_base_train_threshold_head",
                "--freeze_base_train_support_gate",
            ]
            enabled_args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertFalse(default_args.use_semantic_support_adapter)
        self.assertFalse(default_args.eval_use_semantic_support_scores)
        self.assertEqual(default_args.semantic_support_overlap_weight, 0.6075)
        self.assertEqual(default_args.semantic_support_position_weight, 0.1075)
        self.assertEqual(default_args.semantic_support_overlap_power, 0.5)
        self.assertFalse(default_args.semantic_support_use_learned_gate)
        self.assertEqual(default_args.semantic_support_gate_hidden_dim, 16)
        self.assertEqual(default_args.semantic_support_gate_max, 2.0)
        self.assertFalse(default_args.semantic_support_gate_use_query_features)
        self.assertEqual(default_args.semantic_support_gate_loss_weight, 0.0)
        self.assertEqual(default_args.semantic_support_gate_loss_beta, 0.25)
        self.assertFalse(default_args.use_semantic_threshold_head)
        self.assertEqual(default_args.semantic_threshold_hidden_dim, 64)
        self.assertEqual(default_args.semantic_threshold_residual_scale, 0.25)
        self.assertEqual(default_args.semantic_threshold_loss_weight, 1.0)
        self.assertEqual(default_args.semantic_threshold_bce_weight, 1.0)
        self.assertEqual(default_args.semantic_threshold_pairwise_weight, 0.0)
        self.assertEqual(default_args.semantic_threshold_pairwise_margin, 0.25)
        self.assertEqual(
            default_args.semantic_threshold_pairwise_temperature, 0.5
        )
        self.assertEqual(
            default_args.semantic_threshold_pairwise_high_weight, 1.0
        )
        self.assertEqual(default_args.semantic_threshold_focal_gamma, 2.0)
        self.assertFalse(default_args.freeze_base_train_threshold_head)
        self.assertFalse(default_args.freeze_base_train_support_gate)
        self.assertTrue(enabled_args.use_semantic_support_adapter)
        self.assertTrue(enabled_args.eval_use_semantic_support_scores)
        self.assertEqual(enabled_args.semantic_support_overlap_weight, 0.6)
        self.assertEqual(enabled_args.semantic_support_position_weight, 0.1)
        self.assertEqual(enabled_args.semantic_support_overlap_power, 0.75)
        self.assertTrue(enabled_args.semantic_support_use_learned_gate)
        self.assertEqual(enabled_args.semantic_support_gate_hidden_dim, 12)
        self.assertEqual(enabled_args.semantic_support_gate_max, 2.5)
        self.assertTrue(enabled_args.semantic_support_gate_use_query_features)
        self.assertEqual(enabled_args.semantic_support_gate_loss_weight, 0.8)
        self.assertEqual(enabled_args.semantic_support_gate_loss_beta, 0.2)
        self.assertTrue(enabled_args.use_semantic_threshold_head)
        self.assertEqual(enabled_args.semantic_threshold_hidden_dim, 48)
        self.assertEqual(enabled_args.semantic_threshold_residual_scale, 0.2)
        self.assertEqual(enabled_args.semantic_threshold_loss_weight, 0.7)
        self.assertEqual(enabled_args.semantic_threshold_bce_weight, 0.2)
        self.assertEqual(enabled_args.semantic_threshold_pairwise_weight, 0.8)
        self.assertEqual(enabled_args.semantic_threshold_pairwise_margin, 0.3)
        self.assertEqual(
            enabled_args.semantic_threshold_pairwise_temperature, 0.4
        )
        self.assertEqual(
            enabled_args.semantic_threshold_pairwise_high_weight, 1.25
        )
        self.assertEqual(enabled_args.semantic_threshold_focal_gamma, 1.5)
        self.assertTrue(enabled_args.freeze_base_train_threshold_head)
        self.assertTrue(enabled_args.freeze_base_train_support_gate)

    def test_semantic_support_adapter_matches_fixed_formula(self):
        from models.semantic_support_adapter import SemanticSupportAdapter

        adapter = SemanticSupportAdapter(
            overlap_weight=0.6075,
            position_weight=0.1075,
            overlap_power=0.5,
        )
        semantic = torch.tensor([[1.0, 2.0, 4.0]])
        position = torch.tensor([[4.0, 0.0, 1.0]])
        queries = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                                 [3.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                                 [6.0, 0.0, 0.0, 2.0, 2.0, 2.0]]])
        detectors = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                                   [3.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                                   [6.0, 0.0, 0.0, 1.0, 1.0, 1.0]]])
        valid = torch.tensor([[True, False, True]])

        output = adapter(semantic, queries, detectors, valid, position)

        support = torch.tensor([[1.0, 0.0, 0.125]])

        def z(scores):
            centered = scores - scores.mean(dim=1, keepdim=True)
            return centered / centered.std(
                dim=1, keepdim=True, unbiased=False
            ).clamp(min=1e-6)

        expected = (
            z(semantic)
            + 0.6075 * z(support.sqrt())
            + 0.1075 * z(position)
        )
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 0)
        self.assertEqual(adapter.state_dict(), {})
        self.assertTrue(torch.allclose(
            output["semantic_detector_support"], support, atol=1e-6
        ))
        self.assertTrue(torch.allclose(
            output["semantic_support_scores"], expected, atol=1e-6
        ))

    def test_semantic_support_adapter_ignores_masked_detector_boxes(self):
        from models.semantic_support_adapter import SemanticSupportAdapter

        adapter = SemanticSupportAdapter()
        queries = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                                 [5.0, 0.0, 0.0, 2.0, 2.0, 2.0]]])
        detectors = queries.clone()
        semantic = torch.tensor([[0.0, 1.0]])
        position = torch.tensor([[1.0, 0.0]])

        output = adapter(
            semantic, queries, detectors,
            torch.tensor([[True, False]]), position,
        )
        empty_output = adapter(
            semantic, queries, detectors,
            torch.tensor([[False, False]]), position,
        )

        self.assertTrue(torch.allclose(
            output["semantic_detector_support"],
            torch.tensor([[1.0, 0.0]]),
        ))
        self.assertTrue(torch.equal(
            empty_output["semantic_detector_support"],
            torch.zeros(1, 2),
        ))

    def test_learned_semantic_support_gate_preserves_formula_and_backprops(self):
        from models.semantic_support_adapter import SemanticSupportAdapter

        fixed = SemanticSupportAdapter()
        learned = SemanticSupportAdapter(
            use_learned_gate=True, gate_hidden_dim=8, gate_max=2.0
        )
        semantic = torch.tensor([[0.0, 1.0, 0.25]])
        position = torch.tensor([[1.0, 0.0, 0.5]])
        queries = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                                 [3.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                                 [6.0, 0.0, 0.0, 2.0, 2.0, 2.0]]])
        detectors = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                                   [6.0, 0.0, 0.0, 1.0, 1.0, 1.0]]])
        valid = torch.tensor([[True, True]])

        expected = fixed(semantic, queries, detectors, valid, position)
        output = learned(semantic, queries, detectors, valid, position)
        self.assertTrue(torch.allclose(
            output["semantic_support_gate"], torch.ones(1, 1), atol=1e-7
        ))
        self.assertTrue(torch.allclose(
            output["semantic_support_scores"],
            expected["semantic_support_scores"], atol=1e-6,
        ))
        output["semantic_support_scores"][0, 0].backward()
        self.assertTrue(any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in learned.support_gate.parameters()
        ))

    def test_representation_support_gate_preserves_formula_and_backprops(self):
        from models.semantic_support_adapter import SemanticSupportAdapter

        fixed = SemanticSupportAdapter()
        learned = SemanticSupportAdapter(
            use_learned_gate=True,
            gate_hidden_dim=8,
            gate_max=2.0,
            gate_use_query_features=True,
            query_dim=4,
        )
        semantic = torch.tensor([[0.0, 1.0, 0.25]])
        position = torch.tensor([[1.0, 0.0, 0.5]])
        queries = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                                 [3.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                                 [6.0, 0.0, 0.0, 2.0, 2.0, 2.0]]])
        detectors = queries[:, :2].clone()
        valid = torch.tensor([[True, True]])
        query_feats = torch.randn(1, 3, 4)
        target_slot = torch.randn(1, 4)

        expected = fixed(semantic, queries, detectors, valid, position)
        output = learned(
            semantic, queries, detectors, valid, position,
            query_feats=query_feats, target_slot=target_slot,
        )
        self.assertTrue(torch.allclose(
            output["semantic_support_gate"], torch.ones(1, 1), atol=1e-7
        ))
        self.assertTrue(torch.allclose(
            output["semantic_support_scores"],
            expected["semantic_support_scores"], atol=1e-6,
        ))
        output["semantic_support_scores"][0, 0].backward()
        self.assertTrue(any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in learned.support_gate.parameters()
        ))

    def test_semantic_support_gate_loss_supervises_competing_rankings(self):
        from models.losses import _semantic_support_gate_losses

        gate = torch.ones(2, 1, requires_grad=True)
        end_points = {
            "semantic_support_gate": gate,
            "semantic_support_raw_scores": torch.tensor(
                [[1.0, 0.0], [1.0, 0.0]]
            ),
            "semantic_support_fixed_scores": torch.tensor(
                [[0.0, 1.0], [0.0, 1.0]]
            ),
            "center_label": torch.zeros(2, 1, 3),
            "size_gts": torch.ones(2, 1, 3),
            "last_center": torch.tensor([
                [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            ]),
            "last_pred_size": torch.ones(2, 2, 3),
            "language_dataset": ["scanrefer", "scanrefer"],
        }
        losses = _semantic_support_gate_losses(
            end_points,
            {
                "loss_weight": 2.0,
                "beta": 0.25,
                "low_iou_threshold": 0.25,
                "high_iou_threshold": 0.5,
            },
        )

        self.assertAlmostEqual(
            losses["loss_semantic_support_gate_raw"].item(), 0.4375
        )
        self.assertAlmostEqual(
            losses["loss_semantic_support_gate"].item(), 0.875
        )
        self.assertAlmostEqual(
            losses["dbg_semantic_support_gate_valid_batch_ratio"].item(), 1.0
        )
        self.assertAlmostEqual(
            losses["dbg_semantic_support_gate_keep_target_ratio"].item(), 0.5
        )
        losses["loss_semantic_support_gate"].backward()
        self.assertTrue(torch.isfinite(gate.grad).all())
        self.assertGreater(gate.grad[1].item(), 0.0)

    def test_semantic_threshold_loss_supervises_both_iou_cutoffs(self):
        from models.losses import _semantic_threshold_losses

        logits = torch.zeros(1, 2, 2, requires_grad=True)
        end_points = {
            "semantic_threshold_logits": logits,
            "center_label": torch.zeros(1, 1, 3),
            "size_gts": torch.ones(1, 1, 3),
            "last_center": torch.tensor(
                [[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
            ),
            "last_pred_size": torch.ones(1, 2, 3),
            "language_dataset": ["scanrefer"],
        }
        losses = _semantic_threshold_losses(
            end_points,
            {"loss_weight": 0.5, "focal_gamma": 0.0},
        )

        self.assertAlmostEqual(
            losses["loss_semantic_threshold_raw"].item(),
            0.693147,
            places=5,
        )
        self.assertAlmostEqual(
            losses["loss_semantic_threshold"].item(),
            0.3465735,
            places=5,
        )
        self.assertAlmostEqual(
            losses["dbg_semantic_threshold_positive25_ratio"].item(), 0.5
        )
        self.assertAlmostEqual(
            losses["dbg_semantic_threshold_positive50_ratio"].item(), 0.5
        )
        losses["loss_semantic_threshold"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_semantic_threshold_pairwise_loss_prefers_correct_fixed_top2(self):
        from models.losses import _semantic_threshold_losses

        logits = torch.zeros(1, 2, 2, requires_grad=True)
        end_points = {
            "semantic_threshold_logits": logits,
            "semantic_support_scores_without_threshold": torch.tensor(
                [[0.0, 1.0]]
            ),
            "center_label": torch.zeros(1, 1, 3),
            "size_gts": torch.ones(1, 1, 3),
            "last_center": torch.tensor(
                [[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
            ),
            "last_pred_size": torch.ones(1, 2, 3),
            "language_dataset": ["scanrefer"],
        }
        losses = _semantic_threshold_losses(
            end_points,
            {
                "loss_weight": 1.0,
                "bce_weight": 0.0,
                "pairwise_weight": 1.0,
                "pairwise_margin": 0.25,
                "pairwise_temperature": 0.5,
                "pairwise_high_weight": 1.0,
            },
        )

        self.assertGreater(
            losses["dbg_semantic_threshold_pairwise_loss"].item(), 0.0
        )
        self.assertAlmostEqual(
            losses[
                "dbg_semantic_threshold_pairwise25_valid_ratio"
            ].item(),
            1.0,
        )
        self.assertAlmostEqual(
            losses[
                "dbg_semantic_threshold_pairwise50_valid_ratio"
            ].item(),
            1.0,
        )
        losses["loss_semantic_threshold"].backward()
        self.assertLess(logits.grad[0, 0, 0].item(), 0.0)
        self.assertGreater(logits.grad[0, 1, 0].item(), 0.0)
        self.assertLess(logits.grad[0, 0, 1].item(), 0.0)
        self.assertGreater(logits.grad[0, 1, 1].item(), 0.0)

    def test_semantic_rerank_head_initially_preserves_base_scores(self):
        from models.semantic_rerank_head import SemanticRerankHead

        head = SemanticRerankHead(d_model=4, hidden_dim=8, residual_scale=0.2)
        query_feats = torch.randn(2, 3, 4)
        pred_boxes = torch.randn(2, 3, 6)
        base_scores = torch.randn(2, 3)
        quality_scores = torch.rand(2, 3)
        fused_scores = torch.randn(2, 3)

        out = head(
            query_feats,
            pred_boxes,
            base_scores,
            quality_scores=quality_scores,
            fused_scores=fused_scores,
        )

        self.assertTrue(torch.allclose(out["semantic_rerank_scores"], base_scores))
        self.assertTrue(torch.allclose(
            out["semantic_rerank_residual"],
            torch.zeros_like(base_scores),
        ))

    def test_semantic_threshold_head_initially_preserves_scores_and_backprops(self):
        from models.semantic_rerank_head import SemanticRerankHead

        head = SemanticRerankHead(
            d_model=4,
            hidden_dim=8,
            residual_scale=0.2,
            use_threshold_head=True,
            threshold_hidden_dim=6,
            threshold_residual_scale=0.3,
        )
        output = head(
            query_feats=torch.randn(2, 3, 4),
            pred_boxes=torch.randn(2, 3, 6),
            base_scores=torch.randn(2, 3),
            target_slot=torch.randn(2, 4),
            semantic_components=torch.randn(2, 3, 5),
        )

        self.assertEqual(output["semantic_threshold_logits"].shape, (2, 3, 2))
        self.assertTrue(torch.equal(
            output["semantic_threshold_residual"], torch.zeros(2, 3)
        ))
        output["semantic_threshold_logits"].sum().backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in head.threshold_head.parameters()
        ))

    def test_semantic_component_calibrator_initially_preserves_base_scores(self):
        from models.semantic_component_calibrator import SemanticComponentCalibrator

        calibrator = SemanticComponentCalibrator(max_delta=0.25)
        components = torch.tensor(
            [
                [
                    [0.6, 0.1, 0.0, 0.2, 0.05],
                    [0.2, 0.4, 0.1, 0.0, 0.3],
                ]
            ]
        )
        expected = components[..., 0] + components[..., 1]
        expected = expected + components[..., 2] + components[..., 3]
        expected = expected - components[..., 4]

        out = calibrator(components)

        self.assertTrue(torch.allclose(out["semantic_component_scores"], expected))
        self.assertTrue(torch.allclose(
            out["semantic_component_weights"],
            torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0]),
        ))

    def test_semantic_component_calibrator_can_learn_extra_score_residual(self):
        from models.semantic_component_calibrator import SemanticComponentCalibrator

        calibrator = SemanticComponentCalibrator(
            max_delta=0.25,
            extra_score_count=1,
            extra_max_weight=0.5,
        )
        components = torch.tensor(
            [
                [
                    [0.6, 0.1, 0.0, 0.2, 0.05],
                    [0.2, 0.4, 0.1, 0.0, 0.3],
                ]
            ]
        )
        extra_scores = torch.tensor([[[0.0], [10.0]]])
        expected = components[..., 0] + components[..., 1]
        expected = expected + components[..., 2] + components[..., 3]
        expected = expected - components[..., 4]

        out = calibrator(components, extra_scores=extra_scores)

        self.assertTrue(torch.allclose(out["semantic_component_scores"], expected))
        self.assertIn("semantic_component_extra_weights", out)
        self.assertTrue(torch.allclose(
            out["semantic_component_extra_weights"],
            torch.zeros(1),
        ))

        calibrator.extra_logit_weights.data.fill_(10.0)
        shifted = calibrator(components, extra_scores=extra_scores)

        self.assertGreater(
            shifted["semantic_component_scores"][0, 1].item(),
            shifted["semantic_component_scores"][0, 0].item(),
        )

    def test_sem_align_eval_weights_penalize_modifier_mismatch_more(self):
        from models.losses import SetCriterion

        def make_outputs():
            return {
                "proj_queries": torch.tensor([[[2.0, 0.0]]]),
                "proj_tokens": torch.tensor(
                    [[[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]]]
                ),
                "tokenized": {
                    "attention_mask": torch.tensor([[1, 1, 1, 1]])
                },
                "language_dataset": ["scanrefer"],
            }

        targets = [{
            "positive_map": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "modify_positive_map": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            "pron_positive_map": torch.zeros(1, 4),
            "other_entity_map": torch.zeros(1, 4),
            "rel_positive_map": torch.zeros(1, 4),
        }]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        default_criterion = SetCriterion(
            matcher=None,
            losses=["contrastive_align"],
            sem_align_use_eval_weights=False,
        )
        eval_weighted_criterion = SetCriterion(
            matcher=None,
            losses=["contrastive_align"],
            sem_align_use_eval_weights=True,
        )

        default_loss = default_criterion.loss_sem_align(
            make_outputs(), targets, indices, torch.tensor(1.0), auxi_indices=None
        )["loss_sem_align"]
        eval_weighted_loss = eval_weighted_criterion.loss_sem_align(
            make_outputs(), targets, indices, torch.tensor(1.0), auxi_indices=None
        )["loss_sem_align"]

        self.assertGreater(eval_weighted_loss.item(), default_loss.item())

    def test_freeze_base_train_heads_keeps_only_innovation_heads_trainable(self):
        from main_utils import _freeze_base_trainable_heads

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone_net = nn.Linear(2, 2)
                self.proposal_head = nn.Linear(2, 2)
                self.structured_slot_builder = nn.Linear(2, 2)
                self.quality_head = nn.Linear(2, 1)
                self.sacr_head = nn.Linear(2, 1)
                self.reliability_fusion = nn.Linear(2, 1)
                self.semantic_rerank_head = nn.Linear(2, 1)

        model = ToyModel()
        args = SimpleNamespace(freeze_base_train_heads=True)

        trainable_count = _freeze_base_trainable_heads(args, model)
        expected = (
            sum(p.numel() for p in model.structured_slot_builder.parameters())
            + sum(p.numel() for p in model.quality_head.parameters())
            + sum(p.numel() for p in model.sacr_head.parameters())
            + sum(p.numel() for p in model.reliability_fusion.parameters())
            + sum(p.numel() for p in model.semantic_rerank_head.parameters())
        )

        self.assertEqual(trainable_count, expected)
        self.assertFalse(model.backbone_net.weight.requires_grad)
        self.assertFalse(model.proposal_head.weight.requires_grad)
        self.assertTrue(model.structured_slot_builder.weight.requires_grad)
        self.assertTrue(model.quality_head.weight.requires_grad)
        self.assertTrue(model.sacr_head.weight.requires_grad)
        self.assertTrue(model.reliability_fusion.weight.requires_grad)
        self.assertTrue(model.semantic_rerank_head.weight.requires_grad)

    def test_freeze_base_train_component_head_keeps_only_component_calibrator_trainable(self):
        from main_utils import _freeze_base_trainable_heads

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone_net = nn.Linear(2, 2)
                self.quality_head = nn.Linear(2, 1)
                self.sacr_head = nn.Linear(2, 1)
                self.reliability_fusion = nn.Linear(2, 1)
                self.semantic_component_calibrator = nn.Linear(2, 1)

        model = ToyModel()
        args = SimpleNamespace(
            freeze_base_train_heads=False,
            freeze_base_train_align_heads=False,
            freeze_base_train_box_align_heads=False,
            freeze_base_train_component_head=True,
        )

        trainable_count = _freeze_base_trainable_heads(args, model)
        expected = sum(
            p.numel() for p in model.semantic_component_calibrator.parameters()
        )

        self.assertEqual(trainable_count, expected)
        self.assertFalse(model.backbone_net.weight.requires_grad)
        self.assertFalse(model.quality_head.weight.requires_grad)
        self.assertFalse(model.sacr_head.weight.requires_grad)
        self.assertFalse(model.reliability_fusion.weight.requires_grad)
        self.assertTrue(model.semantic_component_calibrator.weight.requires_grad)

    def test_freeze_base_train_rerank_head_keeps_only_semantic_rerank_trainable(self):
        from main_utils import _freeze_base_trainable_heads

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone_net = nn.Linear(2, 2)
                self.quality_head = nn.Linear(2, 1)
                self.sacr_head = nn.Linear(2, 1)
                self.reliability_fusion = nn.Linear(2, 1)
                self.semantic_rerank_head = nn.Linear(2, 1)
                self.semantic_component_calibrator = nn.Linear(2, 1)

        model = ToyModel()
        args = SimpleNamespace(
            freeze_base_train_heads=False,
            freeze_base_train_align_heads=False,
            freeze_base_train_box_align_heads=False,
            freeze_base_train_component_head=False,
            freeze_base_train_rerank_head=True,
        )

        trainable_count = _freeze_base_trainable_heads(args, model)
        expected = sum(
            p.numel() for p in model.semantic_rerank_head.parameters()
        )

        self.assertEqual(trainable_count, expected)
        self.assertFalse(model.backbone_net.weight.requires_grad)
        self.assertFalse(model.quality_head.weight.requires_grad)
        self.assertFalse(model.sacr_head.weight.requires_grad)
        self.assertFalse(model.reliability_fusion.weight.requires_grad)
        self.assertFalse(model.semantic_component_calibrator.weight.requires_grad)
        self.assertTrue(model.semantic_rerank_head.weight.requires_grad)

    def test_freeze_base_train_rerank_align_heads_isolates_rerank_and_alignment(self):
        from main_utils import _freeze_base_trainable_heads

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone_net = nn.Linear(2, 2)
                self.quality_head = nn.Linear(2, 1)
                self.semantic_rerank_head = nn.Linear(2, 1)
                self.contrastive_align_projection_image = nn.Linear(2, 2)
                self.contrastive_align_projection_text = nn.Linear(2, 2)
                self.proposal_head = nn.Linear(2, 6)

        model = ToyModel()
        args = SimpleNamespace(
            freeze_base_train_heads=False,
            freeze_base_train_align_heads=False,
            freeze_base_train_box_align_heads=False,
            freeze_base_train_component_head=False,
            freeze_base_train_rerank_head=False,
            freeze_base_train_rerank_align_heads=True,
        )

        trainable_count = _freeze_base_trainable_heads(args, model)
        expected = sum(
            p.numel() for module in (
                model.semantic_rerank_head,
                model.contrastive_align_projection_image,
                model.contrastive_align_projection_text,
            ) for p in module.parameters()
        )
        self.assertEqual(trainable_count, expected)
        self.assertFalse(model.backbone_net.weight.requires_grad)
        self.assertFalse(model.quality_head.weight.requires_grad)
        self.assertFalse(model.proposal_head.weight.requires_grad)
        self.assertTrue(model.semantic_rerank_head.weight.requires_grad)
        self.assertTrue(model.contrastive_align_projection_image.weight.requires_grad)
        self.assertTrue(model.contrastive_align_projection_text.weight.requires_grad)

    def test_freeze_base_train_align_heads_also_trains_contrastive_projections(self):
        from main_utils import _freeze_base_trainable_heads

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone_net = nn.Linear(2, 2)
                self.proposal_head = nn.Linear(2, 2)
                self.structured_slot_builder = nn.Linear(2, 2)
                self.quality_head = nn.Linear(2, 1)
                self.sacr_head = nn.Linear(2, 1)
                self.reliability_fusion = nn.Linear(2, 1)
                self.semantic_rerank_head = nn.Linear(2, 1)
                self.contrastive_align_projection_image = nn.Linear(2, 2)
                self.contrastive_align_projection_text = nn.Linear(2, 2)

        model = ToyModel()
        args = SimpleNamespace(
            freeze_base_train_heads=False,
            freeze_base_train_align_heads=True,
        )

        trainable_count = _freeze_base_trainable_heads(args, model)
        expected = (
            sum(p.numel() for p in model.structured_slot_builder.parameters())
            + sum(p.numel() for p in model.quality_head.parameters())
            + sum(p.numel() for p in model.sacr_head.parameters())
            + sum(p.numel() for p in model.reliability_fusion.parameters())
            + sum(p.numel() for p in model.semantic_rerank_head.parameters())
            + sum(p.numel() for p in model.contrastive_align_projection_image.parameters())
            + sum(p.numel() for p in model.contrastive_align_projection_text.parameters())
        )

        self.assertEqual(trainable_count, expected)
        self.assertFalse(model.backbone_net.weight.requires_grad)
        self.assertFalse(model.proposal_head.weight.requires_grad)
        self.assertTrue(model.structured_slot_builder.weight.requires_grad)
        self.assertTrue(model.quality_head.weight.requires_grad)
        self.assertTrue(model.sacr_head.weight.requires_grad)
        self.assertTrue(model.reliability_fusion.weight.requires_grad)
        self.assertTrue(model.semantic_rerank_head.weight.requires_grad)
        self.assertTrue(model.contrastive_align_projection_image.weight.requires_grad)
        self.assertTrue(model.contrastive_align_projection_text.weight.requires_grad)

    def test_freeze_base_train_box_align_heads_trains_bbox_and_alignment_heads(self):
        from main_utils import _freeze_base_trainable_heads

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone_net = nn.Linear(2, 2)
                self.cross_encoder = nn.Linear(2, 2)
                self.decoder = nn.Linear(2, 2)
                self.proposal_head = nn.Linear(2, 2)
                self.prediction_heads = nn.ModuleList([
                    nn.Linear(2, 2),
                    nn.Linear(2, 2),
                ])
                self.structured_slot_builder = nn.Linear(2, 2)
                self.quality_head = nn.Linear(2, 1)
                self.sacr_head = nn.Linear(2, 1)
                self.reliability_fusion = nn.Linear(2, 1)
                self.semantic_rerank_head = nn.Linear(2, 1)
                self.contrastive_align_projection_image = nn.Linear(2, 2)
                self.contrastive_align_projection_text = nn.Linear(2, 2)

        model = ToyModel()
        args = SimpleNamespace(
            freeze_base_train_heads=False,
            freeze_base_train_align_heads=False,
            freeze_base_train_box_align_heads=True,
        )

        trainable_count = _freeze_base_trainable_heads(args, model)
        expected = (
            sum(p.numel() for p in model.proposal_head.parameters())
            + sum(p.numel() for p in model.prediction_heads.parameters())
            + sum(p.numel() for p in model.structured_slot_builder.parameters())
            + sum(p.numel() for p in model.quality_head.parameters())
            + sum(p.numel() for p in model.sacr_head.parameters())
            + sum(p.numel() for p in model.reliability_fusion.parameters())
            + sum(p.numel() for p in model.semantic_rerank_head.parameters())
            + sum(p.numel() for p in model.contrastive_align_projection_image.parameters())
            + sum(p.numel() for p in model.contrastive_align_projection_text.parameters())
        )

        self.assertEqual(trainable_count, expected)
        self.assertFalse(model.backbone_net.weight.requires_grad)
        self.assertFalse(model.cross_encoder.weight.requires_grad)
        self.assertFalse(model.decoder.weight.requires_grad)
        self.assertTrue(model.proposal_head.weight.requires_grad)
        self.assertTrue(model.prediction_heads[0].weight.requires_grad)
        self.assertTrue(model.prediction_heads[1].weight.requires_grad)
        self.assertTrue(model.structured_slot_builder.weight.requires_grad)
        self.assertTrue(model.quality_head.weight.requires_grad)
        self.assertTrue(model.sacr_head.weight.requires_grad)
        self.assertTrue(model.reliability_fusion.weight.requires_grad)
        self.assertTrue(model.semantic_rerank_head.weight.requires_grad)
        self.assertTrue(model.contrastive_align_projection_image.weight.requires_grad)
        self.assertTrue(model.contrastive_align_projection_text.weight.requires_grad)

    def test_freeze_base_train_heads_keeps_frozen_modules_eval(self):
        from main_utils import _set_frozen_base_modules_eval

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone_net = nn.Sequential(nn.Dropout(0.5))
                self.quality_head = nn.Sequential(nn.Dropout(0.5))

        model = ToyModel()
        model.train()

        _set_frozen_base_modules_eval(model)

        self.assertFalse(model.backbone_net.training)
        self.assertTrue(model.quality_head.training)

    def test_freeze_base_train_align_heads_keeps_alignment_modules_train(self):
        from main_utils import _set_frozen_base_modules_eval

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone_net = nn.Sequential(nn.Dropout(0.5))
                self.quality_head = nn.Sequential(nn.Dropout(0.5))
                self.contrastive_align_projection_image = nn.Sequential(nn.Dropout(0.5))
                self.contrastive_align_projection_text = nn.Sequential(nn.Dropout(0.5))

        model = ToyModel()
        model.train()
        args = SimpleNamespace(
            freeze_base_train_heads=False,
            freeze_base_train_align_heads=True,
        )

        _set_frozen_base_modules_eval(model, args)

        self.assertFalse(model.backbone_net.training)
        self.assertTrue(model.quality_head.training)
        self.assertTrue(model.contrastive_align_projection_image.training)
        self.assertTrue(model.contrastive_align_projection_text.training)

    def test_freeze_base_train_heads_checkpoint_load_skips_optimizer_state(self):
        from main_utils import load_checkpoint

        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        other_p1 = nn.Parameter(torch.ones(1))
        other_p2 = nn.Parameter(torch.ones(1))
        incompatible_optimizer = torch.optim.AdamW(
            [{"params": [other_p1]}, {"params": [other_p2]}],
            lr=0.01,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "ckpt.pth")
            torch.save(
                {
                    "epoch": 0,
                    "model": model.state_dict(),
                    "optimizer": incompatible_optimizer.state_dict(),
                },
                checkpoint_path,
            )
            args = SimpleNamespace(
                checkpoint_path=checkpoint_path,
                eval=False,
                reduce_lr=False,
                freeze_base_train_heads=True,
                freeze_base_train_align_heads=False,
                start_epoch=1,
            )

            load_checkpoint(args, model, optimizer, scheduler=None)

        self.assertEqual(args.start_epoch, 1)

    def test_freeze_base_train_align_heads_checkpoint_load_skips_optimizer_state(self):
        from main_utils import load_checkpoint

        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        other_p1 = nn.Parameter(torch.ones(1))
        other_p2 = nn.Parameter(torch.ones(1))
        incompatible_optimizer = torch.optim.AdamW(
            [{"params": [other_p1]}, {"params": [other_p2]}],
            lr=0.01,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "ckpt.pth")
            torch.save(
                {
                    "epoch": 0,
                    "model": model.state_dict(),
                    "optimizer": incompatible_optimizer.state_dict(),
                },
                checkpoint_path,
            )
            args = SimpleNamespace(
                checkpoint_path=checkpoint_path,
                eval=False,
                reduce_lr=False,
                freeze_base_train_heads=False,
                freeze_base_train_align_heads=True,
                start_epoch=1,
            )

            load_checkpoint(args, model, optimizer, scheduler=None)

        self.assertEqual(args.start_epoch, 1)

    def test_train_partial_checkpoint_init_allows_missing_model_keys(self):
        from main_utils import load_checkpoint

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(1, 1)
                self.innovation = nn.Linear(1, 1)

        model = ToyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "ckpt.pth")
            partial_state = {
                "base.weight": torch.full_like(model.base.weight, 3.0),
                "base.bias": torch.full_like(model.base.bias, 4.0),
            }
            torch.save(
                {
                    "epoch": 60,
                    "model": partial_state,
                    "optimizer": {"state": {}, "param_groups": []},
                },
                checkpoint_path,
            )
            args = SimpleNamespace(
                checkpoint_path=checkpoint_path,
                eval=False,
                reduce_lr=False,
                freeze_base_train_heads=False,
                freeze_base_train_align_heads=True,
                train_partial_checkpoint_init=True,
                start_epoch=3,
            )

            original_innovation_weight = model.innovation.weight.detach().clone()
            load_checkpoint(args, model, optimizer, scheduler=None)

        self.assertEqual(args.start_epoch, 3)
        torch.testing.assert_close(
            model.base.weight,
            torch.full_like(model.base.weight, 3.0),
        )
        torch.testing.assert_close(
            model.base.bias,
            torch.full_like(model.base.bias, 4.0),
        )
        torch.testing.assert_close(model.innovation.weight, original_innovation_weight)

    def test_model_constructor_accepts_core_innovation_flags(self):
        from models.bdetr import BeaUTyDETR

        signature = inspect.signature(BeaUTyDETR.__init__)

        for name in (
            "use_structured_slots",
            "use_quality_head",
            "use_sacr",
            "use_rapf",
            "rapf_use_quality",
            "use_qahnl",
            "qahnl_score_source",
            "aux_scores_use_contrastive_base",
            "aux_scores_use_semantic_eval_base",
        ):
            self.assertIn(name, signature.parameters)

    def test_aux_base_score_selection_preserves_eda_default(self):
        from models.bdetr import BeaUTyDETR

        model = BeaUTyDETR.__new__(BeaUTyDETR)
        model.aux_scores_use_contrastive_base = False
        end_points = {
            "last_sem_cls_scores": torch.tensor(
                [[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]]
            ),
            "last_proj_queries": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]]
            ),
            "proj_tokens": torch.tensor(
                [[[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]]
            ),
            "text_attention_mask": torch.tensor([[False, False, True]]),
        }
        inputs = {
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0]]]),
        }

        selected = model._select_aux_base_scores(end_points, inputs)
        expected = model._compute_eda_base_scores(end_points, inputs)

        self.assertTrue(torch.allclose(selected, expected))

    def test_aux_base_score_selection_can_prefer_contrastive_scores(self):
        from models.bdetr import BeaUTyDETR

        model = BeaUTyDETR.__new__(BeaUTyDETR)
        model.aux_scores_use_contrastive_base = True
        end_points = {
            "last_sem_cls_scores": torch.tensor(
                [[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]]
            ),
            "last_proj_queries": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]]
            ),
            "proj_tokens": torch.tensor(
                [[[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]]
            ),
            "text_attention_mask": torch.tensor([[False, False, True]]),
        }
        inputs = {
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0]]]),
        }

        selected = model._select_aux_base_scores(end_points, inputs)
        expected = model._compute_contrastive_base_scores(
            end_points["last_proj_queries"],
            end_points["proj_tokens"],
            text_padding_mask=end_points["text_attention_mask"],
        )

        self.assertTrue(torch.allclose(selected, expected))

    def test_aux_base_score_selection_can_prefer_semantic_eval_scores(self):
        from models.bdetr import BeaUTyDETR

        model = BeaUTyDETR.__new__(BeaUTyDETR)
        model.aux_scores_use_contrastive_base = False
        model.aux_scores_use_semantic_eval_base = True
        end_points = {
            "last_proj_queries": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]]
            ),
            "proj_tokens": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0], [0.2, 0.0], [0.0, 0.2]]]
            ),
        }
        inputs = {
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "modify_positive_map": torch.tensor([[[0.0, 0.0, 1.0, 0.0]]]),
            "pron_positive_map": torch.zeros(1, 1, 4),
            "rel_positive_map": torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
            "other_entity_map": torch.tensor([[[0.0, 1.0, 0.0, 0.0]]]),
        }

        selected = model._select_aux_base_scores(end_points, inputs)
        token_scores = torch.matmul(
            end_points["last_proj_queries"],
            end_points["proj_tokens"].transpose(-1, -2),
        ).div(0.07).softmax(-1)
        score_map = (
            inputs["positive_map"]
            + inputs["modify_positive_map"]
            + inputs["pron_positive_map"]
            + inputs["rel_positive_map"]
            - inputs["other_entity_map"]
        )
        expected = (token_scores.unsqueeze(1) * score_map.unsqueeze(2)).sum(-1)
        expected = expected[:, 0]
        contrastive = model._compute_contrastive_base_scores(
            end_points["last_proj_queries"],
            end_points["proj_tokens"],
        )

        self.assertFalse(torch.allclose(contrastive, expected))
        self.assertTrue(torch.allclose(selected, expected))

    def test_loss_accepts_core_innovation_kwargs(self):
        from models.losses import compute_hungarian_loss

        signature = inspect.signature(compute_hungarian_loss)

        for name in (
            "use_quality_head",
            "quality_loss_weight",
            "use_sacr",
            "use_rapf",
            "use_reliability_gate",
            "use_qahnl",
            "qahnl_config",
            "use_sem_iou_rank",
            "sem_iou_rank_config",
            "use_sem_iou_listwise",
            "sem_iou_listwise_config",
            "use_sem_iou_top1",
            "sem_iou_top1_config",
            "use_sem_eval_margin",
            "sem_eval_margin_config",
            "semantic_threshold_config",
            "semantic_support_gate_config",
        ):
            self.assertIn(name, signature.parameters)

    def test_sem_iou_rank_loss_penalizes_semantic_score_iou_mismatch(self):
        from models.losses import _semantic_iou_rank_losses

        def make_end_points(proj_queries):
            return {
                "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
                "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
                "last_center": torch.tensor(
                    [[[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]]
                ),
                "last_pred_size": torch.tensor(
                    [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
                ),
                "last_proj_queries": proj_queries,
                "proj_tokens": torch.tensor(
                    [[[1.0, 0.0], [0.0, 1.0]]]
                ),
                "positive_map": torch.tensor([[[1.0, 0.0]]]),
                "modify_positive_map": torch.zeros(1, 1, 2),
                "pron_positive_map": torch.zeros(1, 1, 2),
                "rel_positive_map": torch.zeros(1, 1, 2),
                "other_entity_map": torch.tensor([[[0.0, 1.0]]]),
                "language_dataset": ["scanrefer"],
            }

        config = {
            "loss_weight": 1.0,
            "pos_iou_thresh": 0.5,
            "neg_iou_thresh": 0.25,
            "topk_iou_pos": 1,
            "num_hard_neg": 1,
            "margin": 0.1,
            "temperature": 1.0,
        }
        mismatched = _semantic_iou_rank_losses(
            make_end_points(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])),
            config,
        )
        aligned = _semantic_iou_rank_losses(
            make_end_points(torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])),
            config,
        )

        self.assertGreater(mismatched["loss_sem_iou_rank"].item(), 1.0)
        self.assertLess(
            aligned["loss_sem_iou_rank"].item(),
            mismatched["loss_sem_iou_rank"].item(),
        )
        self.assertEqual(mismatched["dbg_sem_iou_rank_valid_batch_ratio"], 1.0)

    def test_loss_wires_sem_iou_rank_into_total_loss(self):
        from models.losses import compute_hungarian_loss

        class FakeCriterion:
            def __call__(self, output, target):
                indices = [
                    (
                        torch.tensor([1], dtype=torch.long),
                        torch.tensor([0], dtype=torch.long),
                    )
                ]
                losses = {
                    "loss_ce": torch.tensor(0.0),
                    "loss_bbox": torch.tensor(0.0),
                    "loss_giou": torch.tensor(0.0),
                }
                if "proj_tokens" in output:
                    losses["loss_sem_align"] = torch.tensor(0.0)
                return losses, indices

        end_points = {
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "sem_cls_label": torch.tensor([[0]]),
            "positive_map": torch.tensor([[[1.0, 0.0]]]),
            "modify_positive_map": torch.zeros(1, 1, 2),
            "pron_positive_map": torch.zeros(1, 1, 2),
            "other_entity_map": torch.tensor([[[0.0, 1.0]]]),
            "rel_positive_map": torch.zeros(1, 1, 2),
            "auxi_entity_positive_map": torch.zeros(1, 1, 2),
            "box_label_mask": torch.tensor([[1.0]]),
            "auxi_box": torch.tensor([[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]]),
            "language_dataset": ["scanrefer"],
            "proposal_center": torch.tensor(
                [[[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]]
            ),
            "proposal_pred_size": torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
            ),
            "proposal_sem_cls_scores": torch.zeros(1, 2, 2),
            "proposal_proj_queries": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]]
            ),
            "last_center": torch.tensor(
                [[[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]]
            ),
            "last_pred_size": torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
            ),
            "last_sem_cls_scores": torch.zeros(1, 2, 2),
            "last_proj_queries": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]]
            ),
            "proj_tokens": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            "tokenized": {},
        }

        loss, updated = compute_hungarian_loss(
            end_points,
            num_decoder_layers=1,
            set_criterion=FakeCriterion(),
            use_sem_iou_rank=True,
            sem_iou_rank_config={
                "loss_weight": 1.0,
                "pos_iou_thresh": 0.5,
                "neg_iou_thresh": 0.25,
                "topk_iou_pos": 1,
                "num_hard_neg": 1,
                "margin": 0.1,
                "temperature": 1.0,
            },
        )

        self.assertIn("loss_sem_iou_rank", updated)
        self.assertGreater(updated["loss_sem_iou_rank"].item(), 1.0)
        self.assertGreater(loss.item(), 1.0)

    def test_sem_iou_listwise_loss_prefers_high_iou_semantic_ranking(self):
        from models.losses import _semantic_iou_listwise_losses

        def make_end_points(proj_queries):
            return {
                "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
                "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
                "last_center": torch.tensor(
                    [[[3.0, 3.0, 3.0], [0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]]
                ),
                "last_pred_size": torch.tensor(
                    [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
                ),
                "last_proj_queries": proj_queries,
                "proj_tokens": torch.tensor(
                    [[[1.0, 0.0], [0.0, 1.0]]]
                ),
                "positive_map": torch.tensor([[[1.0, 0.0]]]),
                "modify_positive_map": torch.zeros(1, 1, 2),
                "pron_positive_map": torch.zeros(1, 1, 2),
                "rel_positive_map": torch.zeros(1, 1, 2),
                "other_entity_map": torch.tensor([[[0.0, 1.0]]]),
                "language_dataset": ["scanrefer"],
            }

        config = {
            "loss_weight": 1.0,
            "topk": 3,
            "score_temperature": 0.25,
            "target_iou_power": 2.0,
            "high_iou_threshold": 0.50,
            "high_iou_weight": 2.0,
            "min_target_iou": 0.01,
        }
        mismatched = _semantic_iou_listwise_losses(
            make_end_points(
                torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
            ),
            config,
        )
        aligned = _semantic_iou_listwise_losses(
            make_end_points(
                torch.tensor([[[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]])
            ),
            config,
        )

        self.assertLess(
            aligned["loss_sem_iou_listwise"].item(),
            mismatched["loss_sem_iou_listwise"].item(),
        )
        self.assertEqual(
            mismatched["dbg_sem_iou_listwise_valid_batch_ratio"], 1.0
        )
        self.assertGreater(
            mismatched["dbg_sem_iou_listwise_target_top_iou"].item(),
            0.5,
        )

    def test_sem_iou_listwise_loss_is_finite_when_topk_masks_queries(self):
        from models.losses import _semantic_iou_listwise_losses

        end_points = {
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "last_center": torch.tensor(
                [[[3.0, 3.0, 3.0], [0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]]]
            ),
            "last_pred_size": torch.ones(1, 4, 3),
            "last_proj_queries": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]
            ),
            "proj_tokens": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            "positive_map": torch.tensor([[[1.0, 0.0]]]),
            "modify_positive_map": torch.zeros(1, 1, 2),
            "pron_positive_map": torch.zeros(1, 1, 2),
            "rel_positive_map": torch.zeros(1, 1, 2),
            "other_entity_map": torch.tensor([[[0.0, 1.0]]]),
            "language_dataset": ["scanrefer"],
        }

        result = _semantic_iou_listwise_losses(
            end_points,
            {
                "loss_weight": 1.0,
                "topk": 1,
                "score_temperature": 0.25,
                "target_iou_power": 2.0,
                "high_iou_threshold": 0.50,
                "high_iou_weight": 2.0,
                "min_target_iou": 0.01,
            },
        )

        self.assertTrue(torch.isfinite(result["loss_sem_iou_listwise"]))
        self.assertTrue(torch.isfinite(result["loss_sem_iou_listwise_raw"]))

    def test_loss_wires_sem_iou_listwise_into_total_loss(self):
        from models.losses import compute_hungarian_loss

        class FakeCriterion:
            def __call__(self, output, target):
                indices = [
                    (
                        torch.tensor([1], dtype=torch.long),
                        torch.tensor([0], dtype=torch.long),
                    )
                ]
                losses = {
                    "loss_ce": torch.tensor(0.0),
                    "loss_bbox": torch.tensor(0.0),
                    "loss_giou": torch.tensor(0.0),
                }
                if "proj_tokens" in output:
                    losses["loss_sem_align"] = torch.tensor(0.0)
                return losses, indices

        end_points = {
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "sem_cls_label": torch.tensor([[0]]),
            "positive_map": torch.tensor([[[1.0, 0.0]]]),
            "modify_positive_map": torch.zeros(1, 1, 2),
            "pron_positive_map": torch.zeros(1, 1, 2),
            "other_entity_map": torch.tensor([[[0.0, 1.0]]]),
            "rel_positive_map": torch.zeros(1, 1, 2),
            "auxi_entity_positive_map": torch.zeros(1, 1, 2),
            "box_label_mask": torch.tensor([[1.0]]),
            "auxi_box": torch.tensor([[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]]),
            "language_dataset": ["scanrefer"],
            "proposal_center": torch.tensor(
                [[[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]]
            ),
            "proposal_pred_size": torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
            ),
            "proposal_sem_cls_scores": torch.zeros(1, 2, 2),
            "proposal_proj_queries": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]]
            ),
            "last_center": torch.tensor(
                [[[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]]
            ),
            "last_pred_size": torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
            ),
            "last_sem_cls_scores": torch.zeros(1, 2, 2),
            "last_proj_queries": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]]
            ),
            "proj_tokens": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            "tokenized": {},
        }

        loss, updated = compute_hungarian_loss(
            end_points,
            num_decoder_layers=1,
            set_criterion=FakeCriterion(),
            use_sem_iou_listwise=True,
            sem_iou_listwise_config={
                "loss_weight": 1.0,
                "topk": 2,
                "score_temperature": 0.25,
                "target_iou_power": 2.0,
                "high_iou_threshold": 0.50,
                "high_iou_weight": 2.0,
                "min_target_iou": 0.01,
            },
        )

        self.assertIn("loss_sem_iou_listwise", updated)
        self.assertGreater(updated["loss_sem_iou_listwise"].item(), 1.0)
        self.assertGreater(loss.item(), 1.0)

    def test_sem_iou_top1_loss_penalizes_bad_top1_swaps_only(self):
        from models.losses import _semantic_iou_top1_losses

        def make_end_points(proj_queries):
            return {
                "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
                "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
                "last_center": torch.tensor(
                    [[[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]]
                ),
                "last_pred_size": torch.tensor(
                    [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
                ),
                "last_proj_queries": proj_queries,
                "proj_tokens": torch.tensor(
                    [[[1.0, 0.0], [0.0, 1.0]]]
                ),
                "positive_map": torch.tensor([[[1.0, 0.0]]]),
                "modify_positive_map": torch.zeros(1, 1, 2),
                "pron_positive_map": torch.zeros(1, 1, 2),
                "rel_positive_map": torch.zeros(1, 1, 2),
                "other_entity_map": torch.tensor([[[0.0, 1.0]]]),
                "language_dataset": ["scanrefer"],
            }

        config = {
            "loss_weight": 1.0,
            "pos_iou_thresh": 0.5,
            "iou_gap": 0.05,
            "margin": 0.1,
            "temperature": 0.5,
        }
        mismatched = _semantic_iou_top1_losses(
            make_end_points(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])),
            config,
        )
        aligned = _semantic_iou_top1_losses(
            make_end_points(torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])),
            config,
        )

        self.assertGreater(mismatched["loss_sem_iou_top1"].item(), 1.0)
        self.assertEqual(mismatched["dbg_sem_iou_top1_valid_batch_ratio"], 1.0)
        self.assertLess(
            aligned["loss_sem_iou_top1"].item(),
            mismatched["loss_sem_iou_top1"].item(),
        )
        self.assertEqual(aligned["dbg_sem_iou_top1_valid_batch_ratio"], 0.0)

    def test_loss_wires_sem_iou_top1_into_total_loss(self):
        from models.losses import compute_hungarian_loss

        class FakeCriterion:
            def __call__(self, output, target):
                indices = [
                    (
                        torch.tensor([1], dtype=torch.long),
                        torch.tensor([0], dtype=torch.long),
                    )
                ]
                losses = {
                    "loss_ce": torch.tensor(0.0),
                    "loss_bbox": torch.tensor(0.0),
                    "loss_giou": torch.tensor(0.0),
                }
                if "proj_tokens" in output:
                    losses["loss_sem_align"] = torch.tensor(0.0)
                return losses, indices

        end_points = {
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "sem_cls_label": torch.tensor([[0]]),
            "positive_map": torch.tensor([[[1.0, 0.0]]]),
            "modify_positive_map": torch.zeros(1, 1, 2),
            "pron_positive_map": torch.zeros(1, 1, 2),
            "other_entity_map": torch.tensor([[[0.0, 1.0]]]),
            "rel_positive_map": torch.zeros(1, 1, 2),
            "auxi_entity_positive_map": torch.zeros(1, 1, 2),
            "box_label_mask": torch.tensor([[1.0]]),
            "auxi_box": torch.tensor([[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]]),
            "language_dataset": ["scanrefer"],
            "proposal_center": torch.tensor(
                [[[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]]
            ),
            "proposal_pred_size": torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
            ),
            "proposal_sem_cls_scores": torch.zeros(1, 2, 2),
            "proposal_proj_queries": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]]
            ),
            "last_center": torch.tensor(
                [[[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]]
            ),
            "last_pred_size": torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
            ),
            "last_sem_cls_scores": torch.zeros(1, 2, 2),
            "last_proj_queries": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]]
            ),
            "proj_tokens": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            "tokenized": {},
        }

        loss, updated = compute_hungarian_loss(
            end_points,
            num_decoder_layers=1,
            set_criterion=FakeCriterion(),
            use_sem_iou_top1=True,
            sem_iou_top1_config={
                "loss_weight": 1.0,
                "pos_iou_thresh": 0.5,
                "iou_gap": 0.05,
                "margin": 0.1,
                "temperature": 0.5,
            },
        )

        self.assertIn("loss_sem_iou_top1", updated)
        self.assertGreater(updated["loss_sem_iou_top1"].item(), 1.0)
        self.assertGreater(loss.item(), 1.0)

    def test_sem_eval_margin_loss_uses_matched_high_iou_query(self):
        from models.losses import _semantic_eval_margin_losses

        indices = [
            (
                torch.tensor([1], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
            )
        ]

        def make_end_points(proj_queries):
            return {
                "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
                "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
                "last_center": torch.tensor(
                    [[[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]]
                ),
                "last_pred_size": torch.tensor(
                    [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
                ),
                "last_proj_queries": proj_queries,
                "proj_tokens": torch.tensor(
                    [[[1.0, 0.0], [0.0, 1.0]]]
                ),
                "positive_map": torch.tensor([[[1.0, 0.0]]]),
                "modify_positive_map": torch.zeros(1, 1, 2),
                "pron_positive_map": torch.zeros(1, 1, 2),
                "rel_positive_map": torch.zeros(1, 1, 2),
                "other_entity_map": torch.tensor([[[0.0, 1.0]]]),
                "language_dataset": ["scanrefer"],
            }

        config = {
            "loss_weight": 1.0,
            "min_pos_iou": 0.5,
            "neg_iou_thresh": 0.45,
            "num_hard_neg": 1,
            "margin": 0.1,
            "temperature": 0.5,
        }
        mismatched = _semantic_eval_margin_losses(
            make_end_points(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])),
            indices,
            config,
        )
        aligned = _semantic_eval_margin_losses(
            make_end_points(torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])),
            indices,
            config,
        )

        self.assertGreater(mismatched["loss_sem_eval_margin"].item(), 1.0)
        self.assertEqual(
            mismatched["dbg_sem_eval_margin_valid_batch_ratio"], 1.0
        )
        self.assertLess(
            aligned["loss_sem_eval_margin"].item(),
            mismatched["loss_sem_eval_margin"].item(),
        )

    def test_loss_wires_sem_eval_margin_into_total_loss(self):
        from models.losses import compute_hungarian_loss

        class FakeCriterion:
            def __call__(self, output, target):
                indices = [
                    (
                        torch.tensor([1], dtype=torch.long),
                        torch.tensor([0], dtype=torch.long),
                    )
                ]
                losses = {
                    "loss_ce": torch.tensor(0.0),
                    "loss_bbox": torch.tensor(0.0),
                    "loss_giou": torch.tensor(0.0),
                }
                if "proj_tokens" in output:
                    losses["loss_sem_align"] = torch.tensor(0.0)
                return losses, indices

        end_points = {
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "sem_cls_label": torch.tensor([[0]]),
            "positive_map": torch.tensor([[[1.0, 0.0]]]),
            "modify_positive_map": torch.zeros(1, 1, 2),
            "pron_positive_map": torch.zeros(1, 1, 2),
            "other_entity_map": torch.tensor([[[0.0, 1.0]]]),
            "rel_positive_map": torch.zeros(1, 1, 2),
            "auxi_entity_positive_map": torch.zeros(1, 1, 2),
            "box_label_mask": torch.tensor([[1.0]]),
            "auxi_box": torch.tensor([[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]]),
            "language_dataset": ["scanrefer"],
            "proposal_center": torch.tensor(
                [[[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]]
            ),
            "proposal_pred_size": torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
            ),
            "proposal_sem_cls_scores": torch.zeros(1, 2, 2),
            "proposal_proj_queries": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]]
            ),
            "last_center": torch.tensor(
                [[[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]]
            ),
            "last_pred_size": torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
            ),
            "last_sem_cls_scores": torch.zeros(1, 2, 2),
            "last_proj_queries": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]]
            ),
            "proj_tokens": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            "tokenized": {},
        }

        loss, updated = compute_hungarian_loss(
            end_points,
            num_decoder_layers=1,
            set_criterion=FakeCriterion(),
            use_sem_eval_margin=True,
            sem_eval_margin_config={
                "loss_weight": 1.0,
                "min_pos_iou": 0.5,
                "neg_iou_thresh": 0.45,
                "num_hard_neg": 1,
                "margin": 0.1,
                "temperature": 0.5,
            },
        )

        self.assertIn("loss_sem_eval_margin", updated)
        self.assertGreater(updated["loss_sem_eval_margin"].item(), 1.0)
        self.assertGreater(loss.item(), 1.0)

    def test_loss_wires_enabled_innovation_terms_into_total_loss(self):
        from models.losses import compute_hungarian_loss

        class FakeCriterion:
            def __call__(self, output, target):
                indices = [
                    (
                        torch.tensor([0], dtype=torch.long),
                        torch.tensor([0], dtype=torch.long),
                    )
                ]
                return {
                    "loss_ce": torch.tensor(0.0),
                    "loss_bbox": torch.tensor(0.0),
                    "loss_giou": torch.tensor(0.0),
                }, indices

        end_points = {
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "sem_cls_label": torch.tensor([[0]]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "modify_positive_map": torch.zeros(1, 1, 4),
            "pron_positive_map": torch.zeros(1, 1, 4),
            "other_entity_map": torch.zeros(1, 1, 4),
            "rel_positive_map": torch.zeros(1, 1, 4),
            "auxi_entity_positive_map": torch.zeros(1, 1, 4),
            "box_label_mask": torch.tensor([[1.0]]),
            "auxi_box": torch.tensor([[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]]),
            "language_dataset": ["scanrefer_spacy"],
            "proposal_center": torch.tensor(
                [[[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]]]
            ),
            "proposal_pred_size": torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
            ),
            "proposal_sem_cls_scores": torch.zeros(1, 2, 4),
            "last_center": torch.tensor(
                [[[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]]]
            ),
            "last_pred_size": torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
            ),
            "last_sem_cls_scores": torch.zeros(1, 2, 4),
            "quality_logits": torch.tensor([[0.0, 0.0]]),
            "pred_iou": torch.tensor([[0.4, 0.2]]),
            "base_grounding_scores": torch.tensor([[0.1, 0.9]]),
            "structured_scores": torch.tensor([[0.9, 0.1]]),
            "fused_scores": torch.tensor([[0.2, 0.8]]),
            "rapf_gate": torch.tensor([[0.5, 0.5]]),
            "structured_valid_mask": torch.tensor([True]),
            "global_only_mask": torch.tensor([False]),
        }

        loss, updated = compute_hungarian_loss(
            end_points,
            num_decoder_layers=1,
            set_criterion=FakeCriterion(),
            use_quality_head=True,
            quality_loss_weight=1.0,
            use_rapf=True,
            use_reliability_gate=True,
            rapf_gate_loss_weight=1.0,
            use_qahnl=True,
            qahnl_config={
                "score_source": "fused",
                "loss_weight": 1.0,
                "pos_iou_thresh": 0.25,
                "neg_iou_thresh": 0.1,
                "topk_iou_pos": 1,
                "num_hard_neg": 1,
            },
        )

        self.assertIn("loss_quality", updated)
        self.assertIn("loss_rapf_gate", updated)
        self.assertIn("loss_qahnl", updated)
        self.assertGreater(loss.item(), 0.0)

    def test_qahnl_entity_hardneg_prefers_semantic_entity_confusers(self):
        from models.losses import _qahnl_losses

        end_points = {
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "box_label_mask": torch.tensor([[1.0]]),
            "language_dataset": ["scanrefer_spacy"],
            "last_center": torch.tensor(
                [[[0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [6.0, 6.0, 6.0]]]
            ),
            "last_pred_size": torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
            ),
            "fused_scores": torch.tensor([[0.8, 0.9, 0.2]]),
            "last_sem_cls_scores": torch.tensor(
                [
                    [
                        [4.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 4.0],
                        [4.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "modify_positive_map": torch.zeros(1, 1, 4),
            "pron_positive_map": torch.zeros(1, 1, 4),
            "other_entity_map": torch.zeros(1, 1, 4),
            "auxi_entity_positive_map": torch.zeros(1, 1, 4),
            "rel_positive_map": torch.zeros(1, 1, 4),
        }
        indices = [
            (
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
            )
        ]
        base_config = {
            "score_source": "fused",
            "loss_weight": 1.0,
            "pos_iou_thresh": 0.25,
            "neg_iou_thresh": 0.1,
            "topk_iou_pos": 1,
            "num_hard_neg": 1,
        }

        base = _qahnl_losses(end_points, indices, base_config)
        entity_hard = _qahnl_losses(
            end_points,
            indices,
            {**base_config, "use_entity_hardneg": True},
        )

        self.assertAlmostEqual(base["dbg_qahnl_neg_score"].item(), 0.9)
        self.assertAlmostEqual(entity_hard["dbg_qahnl_neg_score"].item(), 0.2)

    def test_qahnl_semantic_support_source_backpropagates(self):
        from models.losses import _qahnl_losses

        support_scores = torch.tensor(
            [[0.1, 0.9]], requires_grad=True
        )
        end_points = {
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "box_label_mask": torch.tensor([[1.0]]),
            "language_dataset": ["scanrefer_spacy"],
            "last_center": torch.tensor(
                [[[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]]]
            ),
            "last_pred_size": torch.ones(1, 2, 3),
            "semantic_support_scores": support_scores,
        }
        indices = [
            (
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
            )
        ]

        losses = _qahnl_losses(
            end_points,
            indices,
            {
                "score_source": "semantic_support",
                "loss_weight": 1.0,
                "pos_iou_thresh": 0.25,
                "neg_iou_thresh": 0.1,
                "topk_iou_pos": 1,
                "num_hard_neg": 1,
            },
        )
        losses["loss_qahnl"].backward()

        self.assertEqual(
            losses["dbg_qahnl_score_source_semantic_support"], 1.0
        )
        self.assertIsNotNone(support_scores.grad)
        self.assertGreater(support_scores.grad.abs().sum().item(), 0.0)

    def test_position_eval_can_use_fused_scores_as_primary(self):
        from src.grounding_evaluator import GroundingEvaluator

        evaluator = GroundingEvaluator(
            only_root=True,
            thresholds=[0.25, 0.5],
            topks=[1],
            prefixes=["last_"],
        )
        positive_map = torch.zeros(1, 1, 256)
        positive_map[..., 0] = 1.0
        end_points = {
            "positive_map": positive_map,
            "modify_positive_map": torch.zeros(1, 1, 256),
            "pron_positive_map": torch.zeros(1, 1, 256),
            "other_entity_map": torch.zeros(1, 1, 256),
            "auxi_entity_positive_map": torch.zeros(1, 1, 256),
            "rel_positive_map": torch.zeros(1, 1, 256),
            "center_label": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "box_label_mask": torch.tensor([[1.0]]),
            "last_center": torch.tensor([[[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]]]),
            "last_pred_size": torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]),
            "last_sem_cls_scores": torch.tensor(
                [[[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]]]
            ),
            "fused_scores": torch.tensor([[0.0, 10.0]]),
            "eval_use_fused_scores": True,
        }

        evaluator.evaluate_bbox_by_pos_align(end_points, "last_")

        self.assertEqual(evaluator.dets[("last_", 0.25, 1, "bbs")], 1)
        self.assertEqual(evaluator.dets[("last_", 0.5, 1, "bbs")], 1)

    def test_semantic_eval_can_use_learned_rerank_scores(self):
        from src.grounding_evaluator import GroundingEvaluator

        evaluator = GroundingEvaluator(
            only_root=True,
            thresholds=[0.25, 0.5],
            topks=[1],
            prefixes=["last_"],
        )
        positive_map = torch.zeros(1, 1, 256)
        positive_map[..., 0] = 1.0
        end_points = {
            "positive_map": positive_map,
            "modify_positive_map": torch.zeros(1, 1, 256),
            "pron_positive_map": torch.zeros(1, 1, 256),
            "other_entity_map": torch.zeros(1, 1, 256),
            "auxi_entity_positive_map": torch.zeros(1, 1, 256),
            "rel_positive_map": torch.zeros(1, 1, 256),
            "center_label": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "box_label_mask": torch.tensor([[1.0]]),
            "last_center": torch.tensor([[[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]]]),
            "last_pred_size": torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]),
            "proj_tokens": torch.eye(4).unsqueeze(0),
            "last_proj_queries": torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0]]]),
            "semantic_rerank_scores": torch.tensor([[0.0, 10.0]]),
            "eval_use_semantic_rerank_scores": True,
            "is_view_dep": torch.tensor([False]),
            "is_hard": torch.tensor([False]),
            "is_unique": torch.tensor([True]),
        }

        evaluator.evaluate_bbox_by_sem_align(end_points, "last_")

        self.assertEqual(evaluator.dets[("last_", 0.25, 1, "bbf")], 1)
        self.assertEqual(evaluator.dets[("last_", 0.5, 1, "bbf")], 1)

    def test_semantic_support_scores_have_evaluator_priority(self):
        from src.grounding_evaluator import GroundingEvaluator

        end_points = {
            "eval_use_semantic_support_scores": True,
            "eval_use_semantic_rerank_scores": True,
            "semantic_support_scores": torch.tensor([[1.0, 0.0]]),
            "semantic_rerank_scores": torch.tensor([[0.0, 1.0]]),
        }

        self.assertEqual(
            GroundingEvaluator._semantic_score_source(end_points, "last_"),
            "semantic_support_scores",
        )
        self.assertIsNone(
            GroundingEvaluator._semantic_score_source(end_points, "proposal_")
        )

    def test_semantic_eval_can_use_learned_component_scores(self):
        from src.grounding_evaluator import GroundingEvaluator

        evaluator = GroundingEvaluator(
            only_root=True,
            thresholds=[0.25, 0.5],
            topks=[1],
            prefixes=["last_"],
        )
        positive_map = torch.zeros(1, 1, 256)
        positive_map[..., 0] = 1.0
        end_points = {
            "positive_map": positive_map,
            "modify_positive_map": torch.zeros(1, 1, 256),
            "pron_positive_map": torch.zeros(1, 1, 256),
            "other_entity_map": torch.zeros(1, 1, 256),
            "auxi_entity_positive_map": torch.zeros(1, 1, 256),
            "rel_positive_map": torch.zeros(1, 1, 256),
            "center_label": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "box_label_mask": torch.tensor([[1.0]]),
            "last_center": torch.tensor([[[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]]]),
            "last_pred_size": torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]),
            "proj_tokens": torch.eye(4).unsqueeze(0),
            "last_proj_queries": torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0]]]),
            "semantic_component_scores": torch.tensor([[0.0, 10.0]]),
            "eval_use_semantic_component_scores": True,
            "is_view_dep": torch.tensor([False]),
            "is_hard": torch.tensor([False]),
            "is_unique": torch.tensor([True]),
        }

        evaluator.evaluate_bbox_by_sem_align(end_points, "last_")

        self.assertEqual(evaluator.dets[("last_", 0.25, 1, "bbf")], 1)
        self.assertEqual(evaluator.dets[("last_", 0.5, 1, "bbf")], 1)

    def test_semantic_eval_diagnostics_record_learned_score_fixes(self):
        from src.grounding_evaluator import GroundingEvaluator

        evaluator = GroundingEvaluator(
            only_root=True,
            thresholds=[0.25, 0.5],
            topks=[1],
            prefixes=["last_"],
        )
        positive_map = torch.zeros(1, 1, 256)
        positive_map[..., 0] = 1.0
        end_points = {
            "positive_map": positive_map,
            "modify_positive_map": torch.zeros(1, 1, 256),
            "pron_positive_map": torch.zeros(1, 1, 256),
            "other_entity_map": torch.zeros(1, 1, 256),
            "auxi_entity_positive_map": torch.zeros(1, 1, 256),
            "rel_positive_map": torch.zeros(1, 1, 256),
            "center_label": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "box_label_mask": torch.tensor([[1.0]]),
            "last_center": torch.tensor([[[5.0, 5.0, 5.0],
                                          [1.0, 1.0, 1.0]]]),
            "last_pred_size": torch.tensor([[[1.0, 1.0, 1.0],
                                             [1.0, 1.0, 1.0]]]),
            "proj_tokens": torch.eye(4).unsqueeze(0),
            "last_proj_queries": torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0]]]),
            "semantic_component_scores": torch.tensor([[0.0, 10.0]]),
            "semantic_rerank_primary_residual": torch.tensor([[0.1, 0.2]]),
            "semantic_rerank_aux_residual": torch.tensor([[-0.3, 0.4]]),
            "eval_use_semantic_component_scores": True,
            "eval_report_diagnostic_scores": True,
            "eval_diagnostic_dump_path": "unused-test-dump.npz",
            "is_view_dep": torch.tensor([False]),
            "is_hard": torch.tensor([False]),
            "is_unique": torch.tensor([True]),
        }

        evaluator.evaluate_bbox_by_sem_align(end_points, "last_")

        self.assertEqual(evaluator.dets["diag_sem_total"], 1)
        self.assertEqual(evaluator.dets["diag_sem_top1_changed"], 1)
        self.assertEqual(evaluator.dets["diag_sem_fix25"], 1)
        self.assertEqual(evaluator.dets["diag_sem_break25"], 0)
        self.assertEqual(evaluator.dets["diag_sem_fix50"], 1)
        self.assertEqual(evaluator.dets["diag_sem_break50"], 0)
        self.assertAlmostEqual(evaluator.dets["diag_sem_base_top1_iou_sum"], 0.0)
        self.assertAlmostEqual(evaluator.dets["diag_sem_eval_top1_iou_sum"], 1.0)
        self.assertAlmostEqual(evaluator.dets["diag_sem_best_iou_sum"], 1.0)
        dump_row = evaluator.semantic_diagnostic_rows[0]
        self.assertAlmostEqual(
            dump_row["semantic_rerank_primary_residual"][0], 0.1
        )
        self.assertAlmostEqual(
            dump_row["semantic_rerank_primary_residual"][1], 0.2
        )
        self.assertAlmostEqual(
            dump_row["semantic_rerank_aux_residual"][0], -0.3
        )
        self.assertAlmostEqual(
            dump_row["semantic_rerank_aux_residual"][1], 0.4
        )

    def test_semantic_eval_diagnostics_record_rankable_failures(self):
        from src.grounding_evaluator import GroundingEvaluator

        evaluator = GroundingEvaluator(
            only_root=True,
            thresholds=[0.25, 0.5],
            topks=[1],
            prefixes=["last_"],
        )
        positive_map = torch.zeros(1, 1, 256)
        positive_map[..., 0] = 1.0
        end_points = {
            "positive_map": positive_map,
            "modify_positive_map": torch.zeros(1, 1, 256),
            "pron_positive_map": torch.zeros(1, 1, 256),
            "other_entity_map": torch.zeros(1, 1, 256),
            "auxi_entity_positive_map": torch.zeros(1, 1, 256),
            "rel_positive_map": torch.zeros(1, 1, 256),
            "center_label": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "box_label_mask": torch.tensor([[1.0]]),
            "last_center": torch.tensor([[[5.0, 5.0, 5.0],
                                          [1.0, 1.0, 1.0],
                                          [7.0, 7.0, 7.0]]]),
            "last_pred_size": torch.tensor([[[1.0, 1.0, 1.0],
                                             [1.0, 1.0, 1.0],
                                             [1.0, 1.0, 1.0]]]),
            "proj_tokens": torch.eye(4).unsqueeze(0),
            "last_proj_queries": torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0],
                                                 [0.0, 0.0, 1.0, 0.0]]]),
            "semantic_component_scores": torch.tensor([[9.0, 1.0, 0.5]]),
            "eval_use_semantic_component_scores": True,
            "eval_report_diagnostic_scores": True,
            "is_view_dep": torch.tensor([False]),
            "is_hard": torch.tensor([False]),
            "is_unique": torch.tensor([True]),
        }

        evaluator.evaluate_bbox_by_sem_align(end_points, "last_")

        self.assertEqual(evaluator.dets["diag_sem_fail25_total"], 1)
        self.assertEqual(evaluator.dets["diag_sem_fail25_best_rankable"], 1)
        self.assertEqual(evaluator.dets["diag_sem_fail25_top10_rankable"], 1)
        self.assertEqual(evaluator.dets["diag_sem_fail25_unrankable"], 0)
        self.assertEqual(evaluator.dets["diag_sem_fail50_total"], 1)
        self.assertEqual(evaluator.dets["diag_sem_fail50_best_rankable"], 1)
        self.assertEqual(evaluator.dets["diag_sem_fail50_top10_rankable"], 1)
        self.assertEqual(evaluator.dets["diag_sem_fail50_unrankable"], 0)

    def test_semantic_eval_diagnostics_record_class_mismatch_failures(self):
        from src.grounding_evaluator import GroundingEvaluator

        evaluator = GroundingEvaluator(
            only_root=True,
            thresholds=[0.25, 0.5],
            topks=[1],
            prefixes=["last_"],
        )
        positive_map = torch.zeros(1, 1, 256)
        positive_map[..., 0] = 1.0
        end_points = {
            "positive_map": positive_map,
            "modify_positive_map": torch.zeros(1, 1, 256),
            "pron_positive_map": torch.zeros(1, 1, 256),
            "other_entity_map": torch.zeros(1, 1, 256),
            "auxi_entity_positive_map": torch.zeros(1, 1, 256),
            "rel_positive_map": torch.zeros(1, 1, 256),
            "center_label": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "box_label_mask": torch.tensor([[1.0]]),
            "last_center": torch.tensor([[[5.0, 5.0, 5.0],
                                          [1.0, 1.0, 1.0]]]),
            "last_pred_size": torch.tensor([[[1.0, 1.0, 1.0],
                                             [1.0, 1.0, 1.0]]]),
            "last_sem_cls_scores": torch.zeros(1, 2, 19),
            "target_cid": torch.tensor([3]),
            "all_detected_boxes": torch.tensor(
                [[[5.0, 5.0, 5.0, 1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]
            ),
            "all_detected_class_ids": torch.tensor([[4, 3]]),
            "all_detected_bbox_label_mask": torch.tensor([[True, True]]),
            "proj_tokens": torch.eye(4).unsqueeze(0),
            "last_proj_queries": torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0]]]),
            "semantic_component_scores": torch.tensor([[10.0, 0.0]]),
            "eval_use_semantic_component_scores": True,
            "eval_report_diagnostic_scores": True,
            "is_view_dep": torch.tensor([False]),
            "is_hard": torch.tensor([False]),
            "is_unique": torch.tensor([True]),
        }

        evaluator.evaluate_bbox_by_sem_align(end_points, "last_")

        self.assertEqual(evaluator.dets["diag_cls_total"], 1)
        self.assertEqual(evaluator.dets["diag_cls_eval_top_match"], 0)
        self.assertEqual(evaluator.dets["diag_cls_best_iou_match"], 1)
        self.assertEqual(evaluator.dets["diag_cls_fail25_total"], 1)
        self.assertEqual(evaluator.dets["diag_cls_fail25_eval_mismatch"], 1)
        self.assertEqual(evaluator.dets["diag_cls_fail25_eval_match"], 0)

    def test_semantic_eval_records_spacy_augmentation_mode_buckets(self):
        from src.grounding_evaluator import GroundingEvaluator

        evaluator = GroundingEvaluator(
            only_root=True,
            thresholds=[0.25, 0.5],
            topks=[1],
            prefixes=["last_"],
        )
        positive_map = torch.zeros(1, 1, 256)
        positive_map[..., 0] = 1.0
        end_points = {
            "positive_map": positive_map,
            "modify_positive_map": torch.zeros(1, 1, 256),
            "pron_positive_map": torch.zeros(1, 1, 256),
            "other_entity_map": torch.zeros(1, 1, 256),
            "auxi_entity_positive_map": torch.zeros(1, 1, 256),
            "rel_positive_map": torch.zeros(1, 1, 256),
            "center_label": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "box_label_mask": torch.tensor([[1.0]]),
            "last_center": torch.tensor([[[1.0, 1.0, 1.0],
                                          [5.0, 5.0, 5.0]]]),
            "last_pred_size": torch.tensor([[[1.0, 1.0, 1.0],
                                             [1.0, 1.0, 1.0]]]),
            "proj_tokens": torch.eye(4).unsqueeze(0),
            "last_proj_queries": torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0]]]),
            "spacy_rotation_mode_id": torch.tensor([2]),
            "is_view_dep": torch.tensor([False]),
            "is_hard": torch.tensor([False]),
            "is_unique": torch.tensor([True]),
        }

        evaluator.evaluate_bbox_by_sem_align(end_points, "last_")

        self.assertEqual(evaluator.gts["spacy_aug_yaw_only"], 1)
        self.assertEqual(evaluator.dets["spacy_aug_yaw_only"], 1)
        self.assertEqual(evaluator.gts["spacy_aug_yaw_only50"], 1)
        self.assertEqual(evaluator.dets["spacy_aug_yaw_only50"], 1)

    def test_semantic_eval_records_spacy_augmentation_profile_buckets(self):
        from src.grounding_evaluator import GroundingEvaluator

        evaluator = GroundingEvaluator(
            only_root=True,
            thresholds=[0.25, 0.5],
            topks=[1],
            prefixes=["last_"],
        )
        positive_map = torch.zeros(1, 1, 256)
        positive_map[..., 0] = 1.0
        end_points = {
            "positive_map": positive_map,
            "modify_positive_map": torch.zeros(1, 1, 256),
            "pron_positive_map": torch.zeros(1, 1, 256),
            "other_entity_map": torch.zeros(1, 1, 256),
            "auxi_entity_positive_map": torch.zeros(1, 1, 256),
            "rel_positive_map": torch.zeros(1, 1, 256),
            "center_label": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "box_label_mask": torch.tensor([[1.0]]),
            "last_center": torch.tensor([[[1.0, 1.0, 1.0],
                                          [5.0, 5.0, 5.0]]]),
            "last_pred_size": torch.tensor([[[1.0, 1.0, 1.0],
                                             [1.0, 1.0, 1.0]]]),
            "proj_tokens": torch.eye(4).unsqueeze(0),
            "last_proj_queries": torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0]]]),
            "spacy_augmentation_profile_id": torch.tensor([4]),
            "is_view_dep": torch.tensor([False]),
            "is_hard": torch.tensor([False]),
            "is_unique": torch.tensor([True]),
        }

        evaluator.evaluate_bbox_by_sem_align(end_points, "last_")

        self.assertEqual(evaluator.gts["spacy_profile_yaw_relation_free"], 1)
        self.assertEqual(evaluator.dets["spacy_profile_yaw_relation_free"], 1)
        self.assertEqual(evaluator.gts["spacy_profile_yaw_relation_free50"], 1)
        self.assertEqual(evaluator.dets["spacy_profile_yaw_relation_free50"], 1)

    def test_semantic_eval_records_stable_relation_free_profile_bucket(self):
        from src.grounding_evaluator import GroundingEvaluator

        evaluator = GroundingEvaluator(
            only_root=True,
            thresholds=[0.25, 0.5],
            topks=[1],
            prefixes=["last_"],
        )
        positive_map = torch.zeros(1, 1, 256)
        positive_map[..., 0] = 1.0
        end_points = {
            "positive_map": positive_map,
            "modify_positive_map": torch.zeros(1, 1, 256),
            "pron_positive_map": torch.zeros(1, 1, 256),
            "other_entity_map": torch.zeros(1, 1, 256),
            "auxi_entity_positive_map": torch.zeros(1, 1, 256),
            "rel_positive_map": torch.zeros(1, 1, 256),
            "center_label": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "box_label_mask": torch.tensor([[1.0]]),
            "last_center": torch.tensor([[[1.0, 1.0, 1.0],
                                          [5.0, 5.0, 5.0]]]),
            "last_pred_size": torch.tensor([[[1.0, 1.0, 1.0],
                                             [1.0, 1.0, 1.0]]]),
            "proj_tokens": torch.eye(4).unsqueeze(0),
            "last_proj_queries": torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0]]]),
            "spacy_augmentation_profile_id": torch.tensor([6]),
            "is_view_dep": torch.tensor([False]),
            "is_hard": torch.tensor([False]),
            "is_unique": torch.tensor([True]),
        }

        evaluator.evaluate_bbox_by_sem_align(end_points, "last_")

        self.assertEqual(evaluator.gts["spacy_profile_yaw_relation_free_stable"], 1)
        self.assertEqual(evaluator.dets["spacy_profile_yaw_relation_free_stable"], 1)
        self.assertEqual(evaluator.gts["spacy_profile_yaw_relation_free_stable50"], 1)
        self.assertEqual(evaluator.dets["spacy_profile_yaw_relation_free_stable50"], 1)

    def test_semantic_eval_records_small_yaw_relation_free_view_profile_bucket(self):
        from src.grounding_evaluator import GroundingEvaluator

        evaluator = GroundingEvaluator(
            only_root=True,
            thresholds=[0.25, 0.5],
            topks=[1],
            prefixes=["last_"],
        )
        positive_map = torch.zeros(1, 1, 256)
        positive_map[..., 0] = 1.0
        end_points = {
            "positive_map": positive_map,
            "modify_positive_map": torch.zeros(1, 1, 256),
            "pron_positive_map": torch.zeros(1, 1, 256),
            "other_entity_map": torch.zeros(1, 1, 256),
            "auxi_entity_positive_map": torch.zeros(1, 1, 256),
            "rel_positive_map": torch.zeros(1, 1, 256),
            "center_label": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "box_label_mask": torch.tensor([[1.0]]),
            "last_center": torch.tensor([[[1.0, 1.0, 1.0],
                                          [5.0, 5.0, 5.0]]]),
            "last_pred_size": torch.tensor([[[1.0, 1.0, 1.0],
                                             [1.0, 1.0, 1.0]]]),
            "proj_tokens": torch.eye(4).unsqueeze(0),
            "last_proj_queries": torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0]]]),
            "spacy_augmentation_profile_id": torch.tensor([7]),
            "is_view_dep": torch.tensor([False]),
            "is_hard": torch.tensor([False]),
            "is_unique": torch.tensor([True]),
        }

        evaluator.evaluate_bbox_by_sem_align(end_points, "last_")

        bucket = "spacy_profile_small_yaw_relation_free_view"
        self.assertEqual(evaluator.gts[bucket], 1)
        self.assertEqual(evaluator.dets[bucket], 1)
        self.assertEqual(evaluator.gts[f"{bucket}50"], 1)
        self.assertEqual(evaluator.dets[f"{bucket}50"], 1)

    def test_semantic_eval_records_rawview_global_only_profile_bucket(self):
        from src.grounding_evaluator import GroundingEvaluator

        evaluator = GroundingEvaluator(
            only_root=True,
            thresholds=[0.25, 0.5],
            topks=[1],
            prefixes=["last_"],
        )
        positive_map = torch.zeros(1, 1, 256)
        positive_map[..., 0] = 1.0
        end_points = {
            "positive_map": positive_map,
            "modify_positive_map": torch.zeros(1, 1, 256),
            "pron_positive_map": torch.zeros(1, 1, 256),
            "other_entity_map": torch.zeros(1, 1, 256),
            "auxi_entity_positive_map": torch.zeros(1, 1, 256),
            "rel_positive_map": torch.zeros(1, 1, 256),
            "center_label": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "box_label_mask": torch.tensor([[1.0]]),
            "last_center": torch.tensor([[[1.0, 1.0, 1.0],
                                          [5.0, 5.0, 5.0]]]),
            "last_pred_size": torch.tensor([[[1.0, 1.0, 1.0],
                                             [1.0, 1.0, 1.0]]]),
            "proj_tokens": torch.eye(4).unsqueeze(0),
            "last_proj_queries": torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0]]]),
            "spacy_augmentation_profile_id": torch.tensor([8]),
            "is_view_dep": torch.tensor([False]),
            "is_hard": torch.tensor([False]),
            "is_unique": torch.tensor([True]),
        }

        evaluator.evaluate_bbox_by_sem_align(end_points, "last_")

        bucket = "spacy_profile_rawview_relation_free_global_only"
        self.assertEqual(evaluator.gts[bucket], 1)
        self.assertEqual(evaluator.dets[bucket], 1)
        self.assertEqual(evaluator.gts[f"{bucket}50"], 1)
        self.assertEqual(evaluator.dets[f"{bucket}50"], 1)

    def test_semantic_eval_records_compass_guard_profile_bucket(self):
        from src.grounding_evaluator import GroundingEvaluator

        evaluator = GroundingEvaluator(
            only_root=True,
            thresholds=[0.25, 0.5],
            topks=[1],
            prefixes=["last_"],
        )
        positive_map = torch.zeros(1, 1, 256)
        positive_map[..., 0] = 1.0
        end_points = {
            "positive_map": positive_map,
            "modify_positive_map": torch.zeros(1, 1, 256),
            "pron_positive_map": torch.zeros(1, 1, 256),
            "other_entity_map": torch.zeros(1, 1, 256),
            "auxi_entity_positive_map": torch.zeros(1, 1, 256),
            "rel_positive_map": torch.zeros(1, 1, 256),
            "center_label": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "box_label_mask": torch.tensor([[1.0]]),
            "last_center": torch.tensor([[[1.0, 1.0, 1.0],
                                          [5.0, 5.0, 5.0]]]),
            "last_pred_size": torch.tensor([[[1.0, 1.0, 1.0],
                                             [1.0, 1.0, 1.0]]]),
            "proj_tokens": torch.eye(4).unsqueeze(0),
            "last_proj_queries": torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0]]]),
            "spacy_augmentation_profile_id": torch.tensor([10]),
            "is_view_dep": torch.tensor([False]),
            "is_hard": torch.tensor([False]),
            "is_unique": torch.tensor([True]),
        }

        evaluator.evaluate_bbox_by_sem_align(end_points, "last_")

        bucket = "spacy_profile_none_relation_free_compass"
        self.assertEqual(evaluator.gts[bucket], 1)
        self.assertEqual(evaluator.dets[bucket], 1)
        self.assertEqual(evaluator.gts[f"{bucket}50"], 1)
        self.assertEqual(evaluator.dets[f"{bucket}50"], 1)

    def test_loss_wires_semantic_rerank_term_and_backpropagates(self):
        from models.losses import compute_hungarian_loss

        class FakeCriterion:
            def __call__(self, output, target):
                indices = [
                    (
                        torch.tensor([0], dtype=torch.long),
                        torch.tensor([0], dtype=torch.long),
                    )
                ]
                return {
                    "loss_ce": torch.tensor(0.0),
                    "loss_bbox": torch.tensor(0.0),
                    "loss_giou": torch.tensor(0.0),
                }, indices

        rerank_scores = torch.tensor([[3.0, 0.0]], requires_grad=True)
        end_points = {
            "center_label": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "size_gts": torch.tensor([[[1.0, 1.0, 1.0]]]),
            "sem_cls_label": torch.tensor([[0]]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "modify_positive_map": torch.zeros(1, 1, 4),
            "pron_positive_map": torch.zeros(1, 1, 4),
            "other_entity_map": torch.zeros(1, 1, 4),
            "rel_positive_map": torch.zeros(1, 1, 4),
            "auxi_entity_positive_map": torch.zeros(1, 1, 4),
            "box_label_mask": torch.tensor([[1.0]]),
            "auxi_box": torch.tensor([[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]),
            "language_dataset": ["scanrefer"],
            "proposal_center": torch.tensor(
                [[[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]]]
            ),
            "proposal_pred_size": torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
            ),
            "proposal_sem_cls_scores": torch.zeros(1, 2, 4),
            "last_center": torch.tensor(
                [[[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]]]
            ),
            "last_pred_size": torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
            ),
            "last_sem_cls_scores": torch.zeros(1, 2, 4),
            "semantic_rerank_scores": rerank_scores,
            "semantic_support_scores": rerank_scores * 1.0,
        }

        loss, updated = compute_hungarian_loss(
            end_points,
            num_decoder_layers=1,
            set_criterion=FakeCriterion(),
            use_semantic_rerank_head=True,
            semantic_rerank_config={
                "loss_weight": 1.0,
                "listwise_weight": 0.0,
                "threshold_mass_weight": 1.0,
                "train_use_support_scores": True,
                "topk": 2,
                "temperature": 0.5,
            },
        )
        loss.backward()

        self.assertIn("loss_semantic_rerank", updated)
        self.assertIn("dbg_semantic_rerank_threshold_mass_loss", updated)
        self.assertEqual(
            updated["dbg_semantic_rerank_train_uses_support"].item(), 1.0
        )
        self.assertGreater(updated["loss_semantic_rerank"].item(), 0.0)
        self.assertIsNotNone(rerank_scores.grad)
        self.assertGreater(rerank_scores.grad.abs().sum().item(), 0.0)

    def test_semantic_rerank_failure_margin_only_updates_failed_top1(self):
        from models.losses import _semantic_rerank_losses

        rerank_scores = torch.tensor(
            [[3.0, 0.0], [0.0, 3.0]], requires_grad=True
        )
        end_points = {
            "center_label": torch.tensor(
                [[[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]]
            ),
            "size_gts": torch.ones(2, 1, 3),
            "last_center": torch.tensor(
                [
                    [[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]],
                    [[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]],
                ]
            ),
            "last_pred_size": torch.ones(2, 2, 3),
            "semantic_rerank_scores": rerank_scores,
            "language_dataset": ["scanrefer", "scanrefer"],
        }

        losses = _semantic_rerank_losses(
            end_points,
            {
                "loss_weight": 1.0,
                "listwise_weight": 0.0,
                "threshold_mass_weight": 0.0,
                "failure_margin_weight": 1.0,
                "failure_margin": 0.1,
                "topk": 2,
                "temperature": 0.5,
            },
        )
        losses["loss_semantic_rerank"].backward()

        self.assertGreater(
            losses["dbg_semantic_rerank_failure25_valid_ratio"].item(), 0.0
        )
        self.assertGreater(
            losses["dbg_semantic_rerank_failure50_valid_ratio"].item(), 0.0
        )
        self.assertGreater(rerank_scores.grad[0].abs().sum().item(), 0.0)
        self.assertEqual(rerank_scores.grad[1].abs().sum().item(), 0.0)

    def test_loss_wires_semantic_component_calibration_and_backpropagates(self):
        from models.losses import compute_hungarian_loss

        class FakeCriterion:
            def __call__(self, output, target):
                indices = [
                    (
                        torch.tensor([0], dtype=torch.long),
                        torch.tensor([0], dtype=torch.long),
                    )
                ]
                return {
                    "loss_ce": torch.tensor(0.0),
                    "loss_bbox": torch.tensor(0.0),
                    "loss_giou": torch.tensor(0.0),
                }, indices

        def make_end_points(component_scores):
            return {
                "center_label": torch.tensor(
                    [
                        [[1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0]],
                    ]
                ),
                "size_gts": torch.tensor(
                    [
                        [[1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0]],
                    ]
                ),
                "sem_cls_label": torch.tensor([[0], [0]]),
                "positive_map": torch.tensor(
                    [
                        [[1.0, 0.0, 0.0, 0.0]],
                        [[1.0, 0.0, 0.0, 0.0]],
                    ]
                ),
                "modify_positive_map": torch.zeros(2, 1, 4),
                "pron_positive_map": torch.zeros(2, 1, 4),
                "other_entity_map": torch.zeros(2, 1, 4),
                "rel_positive_map": torch.zeros(2, 1, 4),
                "auxi_entity_positive_map": torch.zeros(2, 1, 4),
                "box_label_mask": torch.tensor([[1.0], [1.0]]),
                "auxi_box": torch.tensor(
                    [
                        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                    ]
                ),
                "language_dataset": ["scanrefer", "scanrefer"],
                "proposal_center": torch.tensor(
                    [
                        [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]],
                        [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]],
                    ]
                ),
                "proposal_pred_size": torch.tensor(
                    [
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    ]
                ),
                "proposal_sem_cls_scores": torch.zeros(2, 2, 4),
                "last_center": torch.tensor(
                    [
                        [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]],
                        [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]],
                    ]
                ),
                "last_pred_size": torch.tensor(
                    [
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    ]
                ),
                "last_sem_cls_scores": torch.zeros(2, 2, 4),
                "semantic_component_scores": component_scores,
                "is_hard": torch.tensor([True, False]),
                "is_unique": torch.tensor([False, True]),
            }

        base_scores = torch.tensor(
            [[0.0, 4.0], [4.0, 0.0]], requires_grad=True
        )
        base_loss, base_updated = compute_hungarian_loss(
            make_end_points(base_scores),
            num_decoder_layers=1,
            set_criterion=FakeCriterion(),
            use_semantic_component_calibration=True,
            semantic_component_config={
                "loss_weight": 1.0,
                "topk": 2,
                "temperature": 0.5,
            },
        )

        weighted_scores = torch.tensor(
            [[0.0, 4.0], [4.0, 0.0]], requires_grad=True
        )
        weighted_loss, updated = compute_hungarian_loss(
            make_end_points(weighted_scores),
            num_decoder_layers=1,
            set_criterion=FakeCriterion(),
            use_semantic_component_calibration=True,
            semantic_component_config={
                "loss_weight": 1.0,
                "topk": 2,
                "temperature": 0.5,
                "hard_sample_weight": 1.0,
                "multi_sample_weight": 0.5,
            },
        )
        weighted_loss.backward()

        self.assertIn("loss_semantic_component", updated)
        self.assertGreater(updated["loss_semantic_component"].item(), 0.0)
        self.assertGreater(
            updated["loss_semantic_component"].item(),
            base_updated["loss_semantic_component"].item(),
        )
        self.assertIsNotNone(weighted_scores.grad)
        self.assertGreater(weighted_scores.grad.abs().sum().item(), 0.0)

    def test_spatial_backbone_adapter_parser_defaults_off(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_dist_mod.py"]
            args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertFalse(args.use_spatial_backbone_adapter)

    def test_spatial_attention_matches_reference_equations(self):
        from models.encoder_decoder_layers import (
            CheckpointCompatibleSpatialAttention,
            calc_pairwise_locs,
        )

        torch.manual_seed(5)
        module = CheckpointCompatibleSpatialAttention(
            d_model=8, n_head=2, dropout=0.0
        ).eval()
        query = torch.randn(2, 3, 8)
        text = torch.randn(2, 8)
        locations = calc_pairwise_locs(torch.randn(2, 3, 3))

        output, attention = module(
            query, query, query, locations, txt_embeds=text
        )

        self.assertEqual(tuple(output.shape), (2, 3, 8))
        self.assertEqual(tuple(attention.shape), (2, 2, 3, 3))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(attention).all())
        self.assertTrue(torch.allclose(
            attention.sum(dim=-1), torch.ones_like(attention[..., 0]),
            atol=1e-6,
        ))

    def test_batch_merge_preserves_model_structured_metadata_on_key_overlap(self):
        from main_utils import BaseTrainTester

        end_points = {
            "decomp_global_only_mask": torch.tensor([True]),
        }
        batch_data = {
            "decomp_global_only_mask": torch.tensor([False]),
            "positive_map": torch.ones(1, 1, 4),
        }

        merged = BaseTrainTester._merge_batch_data(end_points, batch_data)

        self.assertTrue(merged["decomp_global_only_mask"].item())
        self.assertTrue(torch.equal(merged["positive_map"], batch_data["positive_map"]))

    def test_accumulate_stats_averages_vector_ratio_tensors(self):
        from main_utils import BaseTrainTester

        stat_dict = BaseTrainTester._accumulate_stats(
            {},
            {
                "metadata_conflict_ratio": torch.tensor([0.0, 0.5]),
                "loss": torch.tensor(2.0),
            },
        )

        self.assertEqual(stat_dict["loss"], 2.0)
        self.assertEqual(stat_dict["metadata_conflict_ratio"], 0.25)

    def test_ddp_unused_parameter_detection_is_enabled_for_structured_slots(self):
        from main_utils import BaseTrainTester

        self.assertTrue(
            BaseTrainTester._ddp_find_unused_parameters(
                SimpleNamespace(use_structured_slots=True)
            )
        )
        self.assertFalse(
            BaseTrainTester._ddp_find_unused_parameters(
                SimpleNamespace(use_structured_slots=False)
            )
        )


if __name__ == "__main__":
    unittest.main()
