import argparse
import os
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn

from main_utils import _load_model_state_with_selector_compat
from main_utils import _freeze_non_selector_parameters
from main_utils import _freeze_non_detector_policy_adapter_parameters
from main_utils import _freeze_non_acd_parameters
from main_utils import _freeze_rapf_parameters
from main_utils import _freeze_quality_head_parameters
from main_utils import _should_restore_checkpoint_train_state
from main_utils import _dataloader_parallelism_kwargs
from main_utils import BaseTrainTester
from main_utils import parse_option
from models.losses import (
    _detector_policy_adapter_losses,
    _source_pool_selector_losses,
)
from models.acd_head import resolve_acd_base_scores
from models.bdetr import BeaUTyDETR
from models.source_pool_selector import (
    SourcePoolSelectorHead,
    compute_contrastive_token_base_scores,
    compute_soft_token_base_scores,
)
from models.detector_policy_sources import (
    DETECTOR_POLICY_SCORE_KEYS,
    DetectorPolicyAdapterHead,
    build_detector_policy_features,
    build_detector_policy_score_sources,
)
import train_dist_mod
from src.grounding_evaluator import GroundingEvaluator


class TestSourcePoolSelector(unittest.TestCase):
    def test_joint_selector_adapter_train_only_keeps_both_small_modules_trainable(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(2, 2)
                self.source_pool_selector = nn.Linear(2, 2)
                self.detector_policy_adapter = nn.Linear(2, 2)

        args = argparse.Namespace(
            source_pool_selector_train_only=True,
            detector_policy_adapter_train_only=True,
            source_pool_selector_lr=0.001,
            detector_policy_adapter_lr=0.002,
            weight_decay=0.0,
        )
        model = TinyModel()

        selector_count = _freeze_non_selector_parameters(args, model)
        adapter_count = _freeze_non_detector_policy_adapter_parameters(args, model)

        self.assertEqual(selector_count, 6)
        self.assertEqual(adapter_count, 6)
        self.assertFalse(model.backbone.weight.requires_grad)
        self.assertTrue(model.source_pool_selector.weight.requires_grad)
        self.assertTrue(model.detector_policy_adapter.weight.requires_grad)

        optimizer = BaseTrainTester.get_optimizer(args, model)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)
        self.assertEqual(optimizer.param_groups[1]["lr"], 0.002)

    def test_freeze_rapf_parameters_only_freezes_reliability_fusion(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(2, 2)
                self.reliability_fusion = nn.Linear(2, 1)
                self.quality_head = nn.Linear(2, 1)

        args = argparse.Namespace(freeze_rapf=True)
        model = TinyModel()

        count = _freeze_rapf_parameters(args, model)

        self.assertEqual(count, 3)
        self.assertFalse(model.reliability_fusion.weight.requires_grad)
        self.assertFalse(model.reliability_fusion.bias.requires_grad)
        self.assertTrue(model.quality_head.weight.requires_grad)
        self.assertTrue(model.backbone.weight.requires_grad)

    def test_freeze_quality_head_parameters_only_freezes_quality_head(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(2, 2)
                self.reliability_fusion = nn.Linear(2, 1)
                self.quality_head = nn.Linear(2, 1)

        args = argparse.Namespace(freeze_quality_head=True)
        model = TinyModel()

        count = _freeze_quality_head_parameters(args, model)

        self.assertEqual(count, 3)
        self.assertFalse(model.quality_head.weight.requires_grad)
        self.assertFalse(model.quality_head.bias.requires_grad)
        self.assertTrue(model.reliability_fusion.weight.requires_grad)
        self.assertTrue(model.backbone.weight.requires_grad)

    def test_freeze_diagnostics_skip_checkpoint_optimizer_state(self):
        args = argparse.Namespace(
            eval=False,
            reduce_lr=False,
            source_pool_selector_train_only=False,
            detector_policy_adapter_train_only=False,
            freeze_rapf=True,
            freeze_quality_head=False,
        )

        self.assertFalse(
            _should_restore_checkpoint_train_state(args, has_fresh_params=False)
        )

    def test_parser_accepts_rapf_quality_freeze_flags(self):
        argv = [
            "prog",
            "--use_structured_slots",
            "--use_sacr",
            "--use_rapf",
            "--use_quality_head",
            "--freeze_rapf",
            "--freeze_quality_head",
            "--checkpoint_path",
            "model.pth",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertTrue(args.freeze_rapf)
        self.assertTrue(args.freeze_quality_head)

    def test_head_scores_each_query_and_tolerates_missing_sources(self):
        head = SourcePoolSelectorHead(d_model=4, hidden_dim=8)
        query_feats = torch.randn(2, 3, 4)
        pred_boxes = torch.randn(2, 3, 6)

        out = head(
            query_feats,
            pred_boxes,
            source_scores={
                "base": torch.randn(2, 3),
                "quality": None,
                "fused": torch.randn(2, 3),
            },
        )

        self.assertEqual(out["selector_scores"].shape, (2, 3))

    def test_soft_token_base_scores_match_bbs_positive_map_scoring(self):
        sem_cls_scores = torch.tensor([[
            [0.0, 5.0],
            [5.0, 0.0],
            [0.0, 4.0],
        ]])
        positive_map = torch.tensor([[[0.0, 1.0]]])

        scores = compute_soft_token_base_scores(
            sem_cls_scores,
            positive_map,
            box_label_mask=torch.tensor([[1]]),
        )

        self.assertGreater(scores[0, 0].item(), scores[0, 1].item())
        self.assertGreater(scores[0, 2].item(), scores[0, 1].item())

    def test_contrastive_token_base_scores_match_bbf_positive_map_scoring(self):
        proj_tokens = torch.tensor([[
            [1.0, 0.0],
            [-1.0, 0.0],
        ]])
        proj_queries = torch.tensor([[
            [-1.0, 0.0],
            [1.0, 0.0],
        ]])
        positive_map = torch.tensor([[[1.0, 0.0]]])

        scores = compute_contrastive_token_base_scores(
            proj_queries,
            proj_tokens,
            positive_map,
            box_label_mask=torch.tensor([[1]]),
        )

        self.assertGreater(scores[0, 1].item(), scores[0, 0].item())

    def test_detector_policy_sources_build_deployable_score_tensors(self):
        pred_boxes = torch.tensor([[
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]])
        quality_scores = torch.tensor([[0.10, 0.40]])
        det_boxes = torch.tensor([[
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]])
        det_mask = torch.tensor([[1]]).bool()
        det_class_ids = torch.tensor([[1]])
        det_logits = torch.tensor([[[0.0, 4.0, 0.0]]])
        target_cid = torch.tensor([1])

        sources = build_detector_policy_score_sources(
            pred_boxes=pred_boxes,
            quality_scores=quality_scores,
            det_boxes=det_boxes,
            det_bbox_label_mask=det_mask,
            det_class_ids=det_class_ids,
            det_logits=det_logits,
            target_cid=target_cid,
        )

        self.assertIn("detector_jointtight", sources)
        self.assertIn("detector_strongcoarse", sources)
        self.assertIn("detector_countsplit", sources)
        self.assertEqual(sources["detector_jointtight"].shape, (1, 2))
        self.assertGreater(
            sources["detector_jointtight"][0, 0].item(),
            quality_scores[0, 0].item(),
        )

    def test_detector_confblend_sources_match_confident_detector_overlap_blends(self):
        pred_boxes = torch.tensor([[
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]])
        quality_scores = torch.tensor([[0.20, 0.60]])
        det_boxes = torch.tensor([[
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]])
        det_mask = torch.tensor([[1]]).bool()
        det_class_ids = torch.tensor([[1]])
        det_logits = torch.tensor([[[0.0, 4.0, 0.0]]])
        target_cid = torch.tensor([1])

        sources = build_detector_policy_score_sources(
            pred_boxes=pred_boxes,
            quality_scores=quality_scores,
            det_boxes=det_boxes,
            det_bbox_label_mask=det_mask,
            det_class_ids=det_class_ids,
            det_logits=det_logits,
            target_cid=target_cid,
        )

        self.assertIn("detector_confblend035", DETECTOR_POLICY_SCORE_KEYS)
        self.assertIn("detector_confblend05", DETECTOR_POLICY_SCORE_KEYS)
        self.assertEqual(
            DETECTOR_POLICY_SCORE_KEYS["detector_confblend035"],
            "detector_confblend035_scores",
        )
        self.assertEqual(
            DETECTOR_POLICY_SCORE_KEYS["detector_confblend05"],
            "detector_confblend05_scores",
        )
        torch.testing.assert_close(
            sources["detector_confblend035"],
            torch.tensor([[0.55, 0.60]]),
        )
        torch.testing.assert_close(
            sources["detector_confblend05"],
            torch.tensor([[0.70, 0.60]]),
        )

    def test_detector_run174boost_matches_diagnostic_formula(self):
        pred_boxes = torch.tensor([
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ],
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ],
        ])
        quality_scores = torch.tensor([
            [0.20, 0.60],
            [0.20, 0.50],
        ])
        det_boxes = torch.tensor([
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [9.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [9.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ],
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ],
        ])
        det_mask = torch.ones(2, 3).bool()
        det_class_ids = torch.tensor([
            [1, 2, 2],
            [1, 1, 1],
        ])
        det_logits = torch.tensor([
            [
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
                [0.0, 0.0, 4.0],
            ],
            [
                [0.0, 4.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 4.0, 0.0],
            ],
        ])
        target_cid = torch.tensor([1, 1])

        sources = build_detector_policy_score_sources(
            pred_boxes=pred_boxes,
            quality_scores=quality_scores,
            det_boxes=det_boxes,
            det_bbox_label_mask=det_mask,
            det_class_ids=det_class_ids,
            det_logits=det_logits,
            target_cid=target_cid,
        )

        self.assertEqual(
            DETECTOR_POLICY_SCORE_KEYS["detector_run174boost"],
            "detector_run174boost_scores",
        )
        torch.testing.assert_close(
            sources["detector_run174boost"],
            torch.tensor([
                [0.75, 0.60],
                [0.60, 0.50],
            ]),
        )

    def test_detector_countsplit_lowonly_matches_run183_policy(self):
        pred_boxes = torch.tensor([
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ],
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ],
        ])
        quality_scores = torch.tensor([
            [0.10, 0.40],
            [0.20, 0.50],
        ])
        det_boxes = torch.tensor([
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [9.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [9.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ],
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ],
        ])
        det_mask = torch.ones(2, 3).bool()
        det_class_ids = torch.tensor([
            [1, 2, 2],
            [1, 1, 1],
        ])
        det_logits = torch.tensor([
            [
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
                [0.0, 0.0, 4.0],
            ],
            [
                [0.0, 4.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 4.0, 0.0],
            ],
        ])
        target_cid = torch.tensor([1, 1])

        sources = build_detector_policy_score_sources(
            pred_boxes=pred_boxes,
            quality_scores=quality_scores,
            det_boxes=det_boxes,
            det_bbox_label_mask=det_mask,
            det_class_ids=det_class_ids,
            det_logits=det_logits,
            target_cid=target_cid,
        )

        lowonly = sources["detector_countsplit_lowonly"]
        self.assertGreater(lowonly[0, 0].item(), quality_scores[0, 0].item())
        torch.testing.assert_close(lowonly[1], quality_scores[1])

    def test_detector_countsplit_guarded_switches_only_with_support_guard(self):
        pred_boxes = torch.tensor([
            [
                [0.05, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ],
            [
                [1.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ],
        ])
        quality_scores = torch.tensor([
            [0.78, 0.72],
            [0.78, 0.72],
        ])
        det_boxes = torch.tensor([
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ],
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ],
        ])
        det_mask = torch.ones(2, 3).bool()
        det_class_ids = torch.ones(2, 3).long()
        det_logits = torch.tensor([
            [
                [0.0, 4.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 4.0, 0.0],
            ],
            [
                [0.0, 4.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 4.0, 0.0],
            ],
        ])
        target_cid = torch.tensor([1, 1])

        sources = build_detector_policy_score_sources(
            pred_boxes=pred_boxes,
            quality_scores=quality_scores,
            det_boxes=det_boxes,
            det_bbox_label_mask=det_mask,
            det_class_ids=det_class_ids,
            det_logits=det_logits,
            target_cid=target_cid,
        )

        guarded_top = sources["detector_countsplit_guarded"].argmax(dim=1)
        lowonly_top = sources["detector_countsplit_lowonly"].argmax(dim=1)
        jointtight_top = sources["detector_jointtight"].argmax(dim=1)

        self.assertEqual(int(guarded_top[0].item()), int(lowonly_top[0].item()))
        self.assertEqual(
            int(guarded_top[1].item()),
            int(jointtight_top[1].item()),
        )

    def test_detector_countsplit_guarded_thresholds_can_be_relaxed(self):
        pred_boxes = torch.tensor([[
            [1.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]])
        quality_scores = torch.tensor([[0.78, 0.72]])
        det_boxes = torch.tensor([[
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]])
        det_mask = torch.ones(1, 3).bool()
        det_class_ids = torch.tensor([[1, 1, 2]])
        det_logits = torch.tensor([[
            [0.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
        ]])
        target_cid = torch.tensor([1])
        previous_support = os.environ.get(
            "NMV2_DETECTOR_COUNTSPLIT_GUARD_SUPPORT_DROP"
        )
        previous_quality = os.environ.get(
            "NMV2_DETECTOR_COUNTSPLIT_GUARD_QUALITY_DROP"
        )
        try:
            os.environ["NMV2_DETECTOR_COUNTSPLIT_GUARD_SUPPORT_DROP"] = "1.0"
            os.environ["NMV2_DETECTOR_COUNTSPLIT_GUARD_QUALITY_DROP"] = "1.0"
            sources = build_detector_policy_score_sources(
                pred_boxes=pred_boxes,
                quality_scores=quality_scores,
                det_boxes=det_boxes,
                det_bbox_label_mask=det_mask,
                det_class_ids=det_class_ids,
                det_logits=det_logits,
                target_cid=target_cid,
            )
        finally:
            if previous_support is None:
                os.environ.pop(
                    "NMV2_DETECTOR_COUNTSPLIT_GUARD_SUPPORT_DROP", None
                )
            else:
                os.environ[
                    "NMV2_DETECTOR_COUNTSPLIT_GUARD_SUPPORT_DROP"
                ] = previous_support
            if previous_quality is None:
                os.environ.pop(
                    "NMV2_DETECTOR_COUNTSPLIT_GUARD_QUALITY_DROP", None
                )
            else:
                os.environ[
                    "NMV2_DETECTOR_COUNTSPLIT_GUARD_QUALITY_DROP"
                ] = previous_quality

        self.assertEqual(
            int(sources["detector_countsplit_guarded"].argmax(dim=1)[0].item()),
            int(sources["detector_countsplit_lowonly"].argmax(dim=1)[0].item()),
        )

    def test_detector_countsplit_guarded_allcount_can_switch_high_count(self):
        pred_boxes = torch.tensor([[
            [0.10, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]])
        quality_scores = torch.tensor([[0.75, 0.72]])
        det_boxes = torch.tensor([[
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]])
        det_mask = torch.ones(1, 3).bool()
        det_class_ids = torch.tensor([[1, 1, 1]])
        det_logits = torch.tensor([[
            [0.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
        ]])
        target_cid = torch.tensor([1])

        sources = build_detector_policy_score_sources(
            pred_boxes=pred_boxes,
            quality_scores=quality_scores,
            det_boxes=det_boxes,
            det_bbox_label_mask=det_mask,
            det_class_ids=det_class_ids,
            det_logits=det_logits,
            target_cid=target_cid,
        )

        self.assertEqual(int(sources["detector_countsplit_lowonly"].argmax(1)[0]), 0)
        self.assertEqual(int(sources["detector_jointtight"].argmax(1)[0]), 1)
        self.assertEqual(
            int(sources["detector_countsplit_guarded_allcount"].argmax(1)[0]),
            int(sources["detector_countsplit_lowonly"].argmax(1)[0]),
        )

    def test_detector_policy_adapter_initializes_to_constrained_prior(self):
        features = {
            "quality_scores": torch.tensor([[0.10, 0.40]]),
            "class_scores": torch.tensor([[1.00, 0.00]]),
            "conf_scores": torch.tensor([[0.50, 0.00]]),
            "det_count": torch.tensor([1]),
        }
        head = DetectorPolicyAdapterHead(
            context_dim=0,
            prior_weights=(1.0, 0.12, 0.25, 0.18),
            delta_scale=0.25,
        )

        out = head(features)

        expected = (
            features["quality_scores"]
            + 0.12 * features["class_scores"]
            + 0.25 * features["conf_scores"]
            + 0.18 * features["class_scores"]
        )
        self.assertTrue(torch.allclose(out["scores"], expected))
        self.assertEqual(out["weights"].shape, (1, 4))

    def test_detector_policy_adapter_context_can_adjust_weights(self):
        features = {
            "quality_scores": torch.tensor([[0.10, 0.40]]),
            "class_scores": torch.tensor([[1.00, 0.00]]),
            "conf_scores": torch.tensor([[0.50, 0.00]]),
            "det_count": torch.tensor([1]),
        }
        head = DetectorPolicyAdapterHead(
            context_dim=2,
            prior_weights=(1.0, 0.12, 0.25, 0.18),
            delta_scale=0.25,
        )
        with torch.no_grad():
            head.context_mlp[0].weight.fill_(0.0)
            head.context_mlp[0].bias.fill_(0.0)
            head.context_mlp[0].weight[0, 0] = 1.0
            head.context_mlp[-1].weight.fill_(0.0)
            head.context_mlp[-1].bias.copy_(
                torch.tensor([0.0, 0.0, 0.0, 0.0])
            )
            head.context_mlp[-1].weight[1, 0] = 1.0

        base = head(features, context=torch.zeros(1, 2))["scores"]
        adjusted = head(features, context=torch.ones(1, 2))["scores"]

        self.assertGreater(adjusted[0, 0].item(), base[0, 0].item())

    def test_detector_policy_adapter_loss_backprops_to_adapter(self):
        features = {
            "quality_scores": torch.tensor([[0.10, 0.40, 0.20]]),
            "class_scores": torch.tensor([[1.00, 0.00, 0.00]]),
            "conf_scores": torch.tensor([[0.30, 0.00, 0.00]]),
            "det_count": torch.tensor([1]),
        }
        head = DetectorPolicyAdapterHead(context_dim=0)
        adapter_out = head(features)
        pred_boxes = torch.tensor([[
            [0.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.10, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]])
        end_points = {
            "detector_policy_adapter_scores": adapter_out["scores"],
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.20]]),
            "pred_iou": features["quality_scores"],
            "last_center": pred_boxes[..., :3],
            "last_pred_size": pred_boxes[..., 3:6],
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _detector_policy_adapter_losses(
            end_points,
            weight=1.0,
            candidate_k=3,
            margin=0.50,
            min_iou_gap=0.05,
        )
        losses["loss_detector_policy_adapter"].backward()

        grads = [p.grad for p in head.parameters() if p.grad is not None]
        self.assertTrue(any(grad.abs().sum().item() > 0 for grad in grads))

    def test_freeze_non_detector_policy_adapter_parameters(self):
        model = nn.Module()
        model.detector_policy_adapter = nn.Linear(2, 1)
        model.other = nn.Linear(2, 1)
        args = argparse.Namespace(detector_policy_adapter_train_only=True)

        from main_utils import _freeze_non_detector_policy_adapter_parameters

        count = _freeze_non_detector_policy_adapter_parameters(args, model)

        self.assertEqual(count, 3)
        self.assertTrue(model.detector_policy_adapter.weight.requires_grad)
        self.assertFalse(model.other.weight.requires_grad)

    def test_get_inputs_forwards_positive_map_for_selector_bbs_base(self):
        batch_data = {
            "point_clouds": torch.zeros(1, 2, 3),
            "utterances": ["chair"],
            "positive_map": torch.ones(1, 1, 2),
            "box_label_mask": torch.ones(1, 1),
            "target_cid": torch.tensor([1]),
        }

        inputs = BaseTrainTester._get_inputs(batch_data)

        self.assertIn("positive_map", inputs)
        self.assertIn("box_label_mask", inputs)
        self.assertIn("target_cid", inputs)

    def test_train_dist_get_inputs_forwards_target_cid_for_diagnostics(self):
        batch_data = {
            "point_clouds": torch.zeros(1, 2, 3),
            "utterances": ["chair"],
            "all_detected_boxes": torch.zeros(1, 4, 6),
            "all_detected_bbox_label_mask": torch.ones(1, 4).bool(),
            "all_detected_class_ids": torch.zeros(1, 4).long(),
            "all_detected_logits": torch.zeros(1, 4, 3),
            "target_cid": torch.tensor([1]),
            "text_target_cid": torch.tensor([1]),
        }

        inputs = train_dist_mod.TrainTester._get_inputs(batch_data)

        self.assertIn("target_cid", inputs)
        self.assertIn("text_target_cid", inputs)
        self.assertIn("det_logits", inputs)

    def test_train_dist_get_inputs_forwards_spacy_augmentation_ids(self):
        batch_data = {
            "point_clouds": torch.zeros(1, 2, 3),
            "utterances": ["chair"],
            "all_detected_boxes": torch.zeros(1, 4, 6),
            "all_detected_bbox_label_mask": torch.ones(1, 4).bool(),
            "all_detected_class_ids": torch.zeros(1, 4).long(),
            "spacy_rotation_mode_id": torch.tensor([2]),
            "spacy_augmentation_profile_id": torch.tensor([11]),
        }

        inputs = train_dist_mod.TrainTester._get_inputs(batch_data)

        self.assertIn("spacy_rotation_mode_id", inputs)
        self.assertIn("spacy_augmentation_profile_id", inputs)
        self.assertEqual(int(inputs["spacy_rotation_mode_id"].item()), 2)
        self.assertEqual(
            int(inputs["spacy_augmentation_profile_id"].item()), 11
        )

    def test_eval_detector_policy_target_cid_uses_text_source_without_gt_fallback(self):
        selected = BeaUTyDETR._detector_policy_target_cid({
            "train": False,
            "eval_target_cid_source": "text",
            "target_cid": torch.tensor([1]),
            "text_target_cid": torch.tensor([2]),
        })

        self.assertEqual(int(selected.item()), 2)
        self.assertIsNone(BeaUTyDETR._detector_policy_target_cid({
            "train": False,
            "eval_target_cid_source": "text",
            "target_cid": torch.tensor([1]),
        }))
        train_selected = BeaUTyDETR._detector_policy_target_cid({
            "train": True,
            "eval_target_cid_source": "text",
            "target_cid": torch.tensor([1]),
            "text_target_cid": torch.tensor([2]),
        })
        self.assertEqual(int(train_selected.item()), 1)

    def test_parser_accepts_detector_policy_source_choice_candidates(self):
        argv = [
            "prog",
            "--use_source_pool_selector",
            "--source_pool_selector_candidate_aware",
            "--source_pool_selector_include_detector_policy_choice",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertTrue(args.source_pool_selector_include_detector_policy_choice)

    def test_parser_accepts_explicit_source_choice_candidates(self):
        argv = [
            "prog",
            "--use_source_pool_selector",
            "--source_pool_selector_direct_choice",
            "--source_pool_selector_choice_sources",
            "base,quality,detector_countboost",
            "--source_pool_selector_choice_target",
            "base_override_focal_bce",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_choice_sources,
            ("base", "quality", "detector_countboost"),
        )

    def test_parser_accepts_detector_policy_adapter_explicit_choice(self):
        argv = [
            "prog",
            "--use_source_pool_selector",
            "--use_detector_policy_adapter",
            "--source_pool_selector_direct_choice",
            "--source_pool_selector_choice_sources",
            "base,quality,detector_policy_adapter",
            "--source_pool_selector_source_min_iou_gaps",
            "quality:0.02,detector_policy_adapter:0.10",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_choice_sources,
            ("base", "quality", "detector_policy_adapter"),
        )
        self.assertEqual(
            args.source_pool_selector_source_min_iou_gaps[
                "detector_policy_adapter"
            ],
            0.10,
        )

    def test_explicit_source_choice_candidates_override_expansion(self):
        model = BeaUTyDETR(
            num_class=4,
            input_feature_dim=0,
            num_queries=2,
            contrastive_align_loss=False,
            use_source_pool_selector=True,
            source_pool_selector_hidden_dim=8,
            source_pool_selector_direct_choice=True,
            source_pool_selector_include_contrastive_choice=True,
            source_pool_selector_include_detector_policy_choice=True,
            source_pool_selector_choice_sources=(
                "base", "quality", "detector_countboost"
            ),
        )

        self.assertEqual(
            model.source_pool_selector.candidate_sources,
            ("base", "quality", "detector_countboost"),
        )
        self.assertEqual(
            model.source_pool_selector.choice_mlp[-1].out_features,
            1,
        )

    def test_compute_loss_does_not_forward_choice_source_model_arg(self):
        criterion = mock.Mock(return_value=(torch.tensor(0.0), {}))
        args = argparse.Namespace(
            num_decoder_layers=1,
            query_points_obj_topk=4,
            source_pool_selector_choice_sources=(
                "base", "quality", "detector_countboost"
            ),
        )

        BaseTrainTester._compute_loss(
            {},
            criterion,
            set_criterion=mock.Mock(),
            args=args,
        )

        self.assertNotIn(
            "source_pool_selector_choice_sources",
            criterion.call_args.kwargs,
        )

    def test_parser_accepts_detector_policy_adapter_training_flags(self):
        argv = [
            "prog",
            "--use_detector_policy_adapter",
            "--detector_policy_adapter_train_only",
            "--detector_policy_adapter_context",
            "--detector_policy_adapter_loss_weight",
            "1.0",
            "--eval_use_detector_policy_adapter_scores",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertTrue(args.use_detector_policy_adapter)
        self.assertTrue(args.detector_policy_adapter_train_only)
        self.assertTrue(args.detector_policy_adapter_context)
        self.assertTrue(args.eval_use_detector_policy_adapter_scores)

    def test_parser_accepts_explicit_primary_detector_policy_source(self):
        argv = [
            "prog",
            "--eval_primary_score_source",
            "detector_countsplit_guarded",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.eval_primary_score_source,
            "detector_countsplit_guarded",
        )

    def test_parser_accepts_run174_detector_policy_primary_source(self):
        argv = [
            "prog",
            "--eval_primary_score_source",
            "detector_run174boost",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(args.eval_primary_score_source, "detector_run174boost")

    def test_parser_accepts_confblend_primary_and_choice_sources(self):
        argv = [
            "prog",
            "--use_source_pool_selector",
            "--source_pool_selector_direct_choice",
            "--source_pool_selector_choice_sources",
            "base,quality,detector_confblend035",
            "--source_pool_selector_source_min_iou_gaps",
            "detector_confblend035:0.10",
            "--eval_primary_score_source",
            "detector_confblend05",
            "--eval_selector_choice_source_bias_detector_confblend035",
            "0.25",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(args.eval_primary_score_source, "detector_confblend05")
        self.assertEqual(
            args.eval_selector_choice_source_bias_detector_confblend035,
            0.25,
        )
        self.assertEqual(
            args.source_pool_selector_choice_sources,
            ("base", "quality", "detector_confblend035"),
        )
        self.assertEqual(
            args.source_pool_selector_source_min_iou_gaps[
                "detector_confblend035"
            ],
            0.10,
        )

    def test_parser_accepts_countsplit_lowonly_choice_candidate(self):
        argv = [
            "prog",
            "--use_source_pool_selector",
            "--source_pool_selector_direct_choice",
            "--source_pool_selector_choice_sources",
            "base,detector_countsplit_guarded",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_choice_sources,
            ("base", "detector_countsplit_guarded"),
        )

    def test_quality_topk_detector_source_enables_train_only_teacher(self):
        argv = [
            "prog",
            "--use_quality_head",
            "--quality_topk_rerank_weight",
            "0.2",
            "--quality_topk_rerank_source",
            "detector_countsplit_lowonly",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.quality_topk_rerank_source,
            "detector_countsplit_lowonly",
        )
        with mock.patch.object(train_dist_mod, "BeaUTyDETR") as ctor:
            train_dist_mod.TrainTester.get_model(args)

        self.assertTrue(
            ctor.call_args[1]["use_detector_policy_teacher"]
        )

    def test_eval_loss_disables_detector_teacher_quality_topk(self):
        criterion = mock.Mock(return_value=(torch.tensor(0.0), {}))
        args = argparse.Namespace(
            num_decoder_layers=6,
            query_points_obj_topk=4,
            quality_topk_rerank_weight=0.2,
            quality_topk_rerank_source="detector_countsplit_lowonly",
        )

        BaseTrainTester._compute_loss(
            {},
            criterion,
            set_criterion=mock.Mock(),
            args=args,
        )

        self.assertEqual(
            criterion.call_args[1]["quality_topk_rerank_weight"],
            0.0,
        )
        self.assertEqual(
            criterion.call_args[1]["quality_topk_rerank_source"],
            "detector_countsplit_lowonly",
        )

    def test_source_choice_loss_accepts_detector_policy_sources(self):
        choice_scores = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 2.0]],
            requires_grad=True,
        )
        pred_boxes = torch.tensor([[
            [0.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]])
        end_points = {
            "selector_choice_scores": choice_scores,
            "selector_choice_source_names": (
                "base",
                "fused",
                "quality",
                "contrastive_base",
                "detector_jointtight",
            ),
            "selector_scores": torch.tensor([[0.0, 0.0]]),
            "base_grounding_scores": torch.tensor([[0.1, 0.2]]),
            "fused_scores": torch.tensor([[0.1, 0.2]]),
            "pred_iou": torch.tensor([[0.1, 0.9]]),
            "bbf_base_grounding_scores": torch.tensor([[0.1, 0.2]]),
            "detector_jointtight_scores": torch.tensor([[0.9, 0.1]]),
            "last_center": pred_boxes[..., :3],
            "last_pred_size": pred_boxes[..., 3:6],
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.05,
            pairwise_weight=0.0,
            choice_target="iou",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(
            losses[
                "dbg_source_pool_selector_target_detector_jointtight_ratio"
            ].item(),
            0.99,
        )
        self.assertIsNotNone(choice_scores.grad)

    def test_selector_choice_eval_uses_dynamic_detector_policy_sources(self):
        base_scores = torch.tensor([[0.1, 0.2]])
        end_points = {
            "selector_choice_scores": torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 5.0]]
            ),
            "selector_choice_source_names": (
                "base",
                "fused",
                "quality",
                "contrastive_base",
                "detector_jointtight",
            ),
            "pred_iou": torch.tensor([[0.1, 0.8]]),
            "detector_jointtight_scores": torch.tensor([[0.9, 0.1]]),
        }

        scores = GroundingEvaluator._selector_choice_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )

        self.assertEqual(int(scores[0].argmax().item()), 0)

    def test_print_stats_reports_dynamic_detector_choice_fields(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "selector_choice_source_names": (
                "base",
                "fused",
                "quality",
                "contrastive_base",
                "detector_jointtight",
            ),
        }
        source_names = GroundingEvaluator._selector_choice_source_names(
            5, end_points=end_points
        )
        self.assertIn("detector_jointtight", source_names)
        evaluator.selector_choice_source_names_for_logging = source_names
        evaluator._record_score_alignment(
            "last_",
            "selector_choice_selected_detector_jointtight_ratio",
            torch.tensor([1.0]),
        )
        evaluator._record_score_alignment(
            "last_",
            "selector_choice_oracle_detector_jointtight_ratio",
            torch.tensor([1.0]),
        )
        evaluator._record_score_alignment(
            "last_",
            "selector_choice_logit_detector_jointtight_mean",
            torch.tensor([5.0]),
        )

        results = evaluator.print_stats()

        self.assertIn(
            "last_selector_choice_selected_detector_jointtight_ratio",
            results,
        )
        self.assertIn(
            "last_selector_choice_oracle_detector_jointtight_ratio",
            results,
        )
        self.assertIn(
            "last_selector_choice_logit_detector_jointtight_mean",
            results,
        )
        self.assertEqual(
            results["last_selector_choice_selected_detector_jointtight_ratio"],
            1.0,
        )

    def test_candidate_aware_head_scores_source_top1_queries(self):
        class FixedCandidateScores(torch.nn.Module):
            def __init__(self, values):
                super().__init__()
                self.register_buffer("values", values)

            def forward(self, features):
                return self.values.to(features.device).unsqueeze(-1)

        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            candidate_aware=True,
        )
        fixed_scores = torch.tensor([[
            [9.0, 1.0, 2.0, 3.0],
            [0.0, 7.0, 2.0, 1.0],
            [0.5, 0.6, 8.0, 0.4],
        ]])
        head.candidate_mlp = FixedCandidateScores(fixed_scores)
        query_feats = torch.randn(1, 4, 4)
        pred_boxes = torch.randn(1, 4, 6)
        source_scores = {
            "base": torch.tensor([[0.20, 0.90, 0.10, 0.00]]),
            "fused": torch.tensor([[0.00, 0.10, 0.20, 0.80]]),
            "quality": torch.tensor([[0.70, 0.10, 0.20, 0.30]]),
        }

        out = head(
            query_feats,
            pred_boxes,
            source_scores=source_scores,
        )

        self.assertEqual(out["selector_scores"].shape, (1, 4))
        self.assertEqual(out["selector_source_scores"].shape, (1, 3, 4))
        self.assertEqual(out["selector_choice_scores"].shape, (1, 3))
        self.assertTrue(torch.allclose(
            out["selector_choice_scores"],
            torch.tensor([[1.0, 1.0, 0.5]]),
        ))

    def test_candidate_aware_source_choice_loss_backprops_to_candidate_head(self):
        torch.manual_seed(0)
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            candidate_aware=True,
        )
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        fused_scores = torch.tensor([[0.00, 0.10, 0.80, 0.90]])
        quality_scores = torch.tensor([[0.10, 0.20, 0.95, 0.00]])
        query_feats = torch.randn(1, 4, 4)
        pred_boxes = torch.tensor([[
            [0.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.10, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]])
        out = head(
            query_feats,
            pred_boxes,
            source_scores={
                "base": base_scores,
                "fused": fused_scores,
                "quality": quality_scores,
            },
        )
        end_points = {
            **out,
            "base_grounding_scores": base_scores,
            "fused_scores": fused_scores,
            "pred_iou": quality_scores,
            "last_center": pred_boxes[..., :3],
            "last_pred_size": pred_boxes[..., 3:6],
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.05,
            pairwise_weight=0.0,
            choice_target="quality_override",
        )
        losses["loss_source_pool_selector"].backward()

        candidate_grads = [
            p.grad for p in head.candidate_mlp.parameters()
            if p.grad is not None
        ]
        self.assertTrue(any(
            grad.abs().sum().item() > 0 for grad in candidate_grads
        ))

    def test_direct_choice_head_scores_source_top1_candidates(self):
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            direct_choice=True,
        )
        query_feats = torch.randn(2, 4, 4)
        pred_boxes = torch.randn(2, 4, 6)

        out = head(
            query_feats,
            pred_boxes,
            source_scores={
                "base": torch.tensor([
                    [0.90, 0.20, 0.10, 0.00],
                    [0.00, 0.70, 0.20, 0.10],
                ]),
                "quality": torch.tensor([
                    [0.10, 0.80, 0.20, 0.00],
                    [0.30, 0.10, 0.60, 0.00],
                ]),
                "fused": torch.tensor([
                    [0.00, 0.10, 0.30, 0.90],
                    [0.20, 0.10, 0.00, 0.80],
                ]),
            },
        )

        self.assertEqual(out["selector_scores"].shape, (2, 4))
        self.assertEqual(out["selector_choice_scores"].shape, (2, 3))

    def test_direct_choice_head_can_emit_separate_override_logit(self):
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            direct_choice=True,
            separate_override_head=True,
        )
        query_feats = torch.randn(2, 4, 4)
        pred_boxes = torch.randn(2, 4, 6)

        out = head(
            query_feats,
            pred_boxes,
            source_scores={
                "base": torch.tensor([
                    [0.90, 0.20, 0.10, 0.00],
                    [0.00, 0.70, 0.20, 0.10],
                ]),
                "quality": torch.tensor([
                    [0.10, 0.80, 0.20, 0.00],
                    [0.30, 0.10, 0.60, 0.00],
                ]),
                "fused": torch.tensor([
                    [0.00, 0.10, 0.30, 0.90],
                    [0.20, 0.10, 0.00, 0.80],
                ]),
            },
        )

        self.assertEqual(out["selector_choice_scores"].shape, (2, 3))
        self.assertEqual(out["selector_choice_override_logit"].shape, (2,))

    def test_candidate_aware_head_can_emit_separate_override_logit(self):
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            candidate_aware=True,
            separate_override_head=True,
            override_initial_bias=-1.25,
        )
        query_feats = torch.randn(2, 4, 4)
        pred_boxes = torch.randn(2, 4, 6)

        out = head(
            query_feats,
            pred_boxes,
            source_scores={
                "base": torch.tensor([
                    [0.90, 0.20, 0.10, 0.00],
                    [0.00, 0.70, 0.20, 0.10],
                ]),
                "quality": torch.tensor([
                    [0.10, 0.80, 0.20, 0.00],
                    [0.30, 0.10, 0.60, 0.00],
                ]),
                "fused": torch.tensor([
                    [0.00, 0.10, 0.30, 0.90],
                    [0.20, 0.10, 0.00, 0.80],
                ]),
            },
        )

        self.assertEqual(out["selector_choice_scores"].shape, (2, 3))
        self.assertEqual(out["selector_choice_override_logit"].shape, (2,))
        self.assertTrue(torch.allclose(
            out["selector_choice_override_logit"],
            torch.full((2,), -1.25),
            atol=1e-6,
        ))

    def test_separate_override_head_starts_from_configured_keep_base_prior(self):
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            direct_choice=True,
            separate_override_head=True,
            override_initial_bias=-1.5,
        )
        query_feats = torch.randn(2, 4, 4)
        pred_boxes = torch.randn(2, 4, 6)

        out = head(
            query_feats,
            pred_boxes,
            source_scores={
                "base": torch.randn(2, 4),
                "quality": torch.randn(2, 4),
                "fused": torch.randn(2, 4),
            },
        )

        expected = torch.full((2,), -1.5)
        self.assertTrue(torch.allclose(
            out["selector_choice_override_logit"],
            expected,
            atol=1e-6,
        ))

    def test_direct_choice_head_includes_all_source_context(self):
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            direct_choice=True,
        )

        self.assertEqual(head.direct_choice_relation_dim, 6)
        self.assertEqual(head.direct_choice_rank_dim, 0)
        self.assertEqual(head.choice_mlp[0].in_features, 140)

        rank_head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            direct_choice=True,
            rank_features=True,
        )

        self.assertEqual(rank_head.direct_choice_relation_dim, 6)
        self.assertEqual(rank_head.direct_choice_rank_dim, 9)
        self.assertEqual(rank_head.choice_mlp[0].in_features, 176)

        candidate_rank_head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            candidate_aware=True,
            rank_features=True,
        )

        self.assertEqual(candidate_rank_head.candidate_rank_dim, 9)
        self.assertEqual(candidate_rank_head.candidate_mlp[0].in_features, 35)

    def test_direct_choice_pairdelta_features_have_stable_shape(self):
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            direct_choice=True,
            pairdelta_features=True,
        )

        self.assertEqual(head.direct_choice_pairdelta_dim, 21)
        self.assertEqual(head.choice_mlp[0].in_features, 224)

    def test_candidate_aware_pairdelta_features_have_stable_shape(self):
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            candidate_aware=True,
            pairdelta_features=True,
        )

        self.assertEqual(head.candidate_pairdelta_dim, 21)
        self.assertEqual(head.candidate_mlp[0].in_features, 47)

    def test_direct_choice_relation_features_expose_source_agreement(self):
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            direct_choice=True,
        )
        top_indices = torch.tensor([[1, 1, 2]])
        top_scores = torch.tensor([[[0.90], [0.70], [0.10]]])
        top_margins = torch.tensor([[[0.30], [0.10], [0.05]]])
        present = torch.ones(1, 3, 1)
        choice_boxes = torch.tensor([[
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 2.0, 1.0, 1.0],
        ]])

        relation_features = head._direct_choice_relation_features(
            top_indices,
            top_scores,
            top_margins,
            present,
            choice_boxes,
        )

        self.assertEqual(relation_features.shape, (1, 3, 6))
        self.assertEqual(relation_features[0, 0, 0].item(), 1.0)
        self.assertEqual(relation_features[0, 1, 0].item(), 1.0)
        self.assertEqual(relation_features[0, 2, 0].item(), 0.0)
        self.assertAlmostEqual(relation_features[0, 0, 1].item(), 0.5)
        self.assertGreater(relation_features[0, 0, 2].item(), 0.0)
        self.assertLess(relation_features[0, 2, 2].item(), 0.0)
        self.assertGreater(relation_features[0, 0, 4].item(), 0.0)

    def test_direct_choice_rank_features_expose_cross_source_calibration(self):
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            direct_choice=True,
            rank_features=True,
        )
        top_indices = torch.tensor([[0, 2]])
        source_score_stack = torch.tensor([[
            [0.90, 0.20, 0.10],
            [0.40, 0.30, 0.80],
        ]])
        present = torch.ones(1, 2, 1)

        rank_features = head._direct_choice_rank_features(
            top_indices,
            source_score_stack,
            present,
        )

        self.assertEqual(rank_features.shape, (1, 2, 6))
        self.assertAlmostEqual(rank_features[0, 0, 0].item(), 1.0)
        self.assertAlmostEqual(rank_features[0, 0, 1].item(), 0.0)
        self.assertAlmostEqual(rank_features[0, 0, 3].item(), 2.0 / 3.0)
        self.assertAlmostEqual(rank_features[0, 0, 4].item(), -0.4)
        self.assertAlmostEqual(rank_features[0, 1, 0].item(), 1.0 / 3.0)
        self.assertAlmostEqual(rank_features[0, 1, 1].item(), -0.8)
        self.assertAlmostEqual(rank_features[0, 1, 3].item(), 1.0)
        self.assertAlmostEqual(rank_features[0, 1, 4].item(), 0.0)

    def test_candidate_rank_features_expose_query_relative_scores(self):
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            candidate_aware=True,
            rank_features=True,
        )
        source_score_stack = torch.tensor([[
            [0.90, 0.20, 0.10],
            [0.40, 0.30, 0.80],
        ]])
        present = torch.ones(1, 2, 1)

        rank_features = head._candidate_rank_features(
            source_score_stack,
            present,
        )

        self.assertEqual(rank_features.shape, (1, 3, 6))
        self.assertAlmostEqual(rank_features[0, 0, 0].item(), 1.0)
        self.assertAlmostEqual(rank_features[0, 0, 1].item(), 0.0)
        self.assertAlmostEqual(rank_features[0, 0, 3].item(), 2.0 / 3.0)
        self.assertAlmostEqual(rank_features[0, 0, 4].item(), -0.4)
        self.assertAlmostEqual(rank_features[0, 2, 0].item(), 1.0 / 3.0)
        self.assertAlmostEqual(rank_features[0, 2, 1].item(), -0.8)
        self.assertAlmostEqual(rank_features[0, 2, 3].item(), 1.0)
        self.assertAlmostEqual(rank_features[0, 2, 4].item(), 0.0)

    def test_candidate_aware_context_head_includes_pool_summary(self):
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            candidate_aware=True,
            candidate_context=True,
        )

        self.assertEqual(head.candidate_pre_context_dim, 26)
        self.assertEqual(head.candidate_context_dim, 52)
        self.assertEqual(head.candidate_mlp[0].in_features, 78)

    def test_candidate_aware_context_features_append_selector_context(self):
        base_head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            candidate_aware=True,
            candidate_context=True,
            selector_context_dim=3,
        )
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            candidate_aware=True,
            candidate_context=True,
            selector_context_dim=3,
            context_features=True,
        )
        query_feats = torch.randn(2, 4, 4)
        pred_boxes = torch.randn(2, 4, 6)

        out = head(
            query_feats,
            pred_boxes,
            selector_context=torch.randn(2, 3),
            source_scores={
                "base": torch.tensor([
                    [0.90, 0.20, 0.10, 0.00],
                    [0.00, 0.70, 0.20, 0.10],
                ]),
                "quality": torch.tensor([
                    [0.10, 0.80, 0.20, 0.00],
                    [0.30, 0.10, 0.60, 0.00],
                ]),
                "fused": torch.tensor([
                    [0.00, 0.10, 0.30, 0.90],
                    [0.20, 0.10, 0.00, 0.80],
                ]),
            },
        )

        self.assertEqual(head.context_feature_dim, 3)
        self.assertEqual(
            head.candidate_mlp[0].in_features,
            base_head.candidate_mlp[0].in_features + 3,
        )
        self.assertEqual(out["selector_source_scores"].shape, (2, 3, 4))

    def test_context_features_replace_additive_context_bias(self):
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            candidate_aware=True,
            selector_context_dim=3,
            context_features=True,
        )

        self.assertIsNone(head.context_choice_bias_mlp)
        self.assertIsNone(head.context_override_bias_mlp)

    def test_candidate_aware_context_head_scores_source_query_pairs(self):
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            candidate_aware=True,
            candidate_context=True,
            candidate_context_k=2,
        )
        query_feats = torch.randn(2, 4, 4)
        pred_boxes = torch.randn(2, 4, 6)

        out = head(
            query_feats,
            pred_boxes,
            source_scores={
                "base": torch.tensor([
                    [0.90, 0.20, 0.10, 0.00],
                    [0.00, 0.70, 0.20, 0.10],
                ]),
                "quality": torch.tensor([
                    [0.10, 0.80, 0.20, 0.00],
                    [0.30, 0.10, 0.60, 0.00],
                ]),
                "fused": torch.tensor([
                    [0.00, 0.10, 0.30, 0.90],
                    [0.20, 0.10, 0.00, 0.80],
                ]),
            },
        )

        self.assertEqual(out["selector_source_scores"].shape, (2, 3, 4))

    def test_direct_choice_context_features_append_selector_context(self):
        base_head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            direct_choice=True,
            selector_context_dim=3,
        )
        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            direct_choice=True,
            selector_context_dim=3,
            context_features=True,
        )
        query_feats = torch.randn(2, 4, 4)
        pred_boxes = torch.randn(2, 4, 6)

        out = head(
            query_feats,
            pred_boxes,
            selector_context=torch.randn(2, 3),
            source_scores={
                "base": torch.tensor([
                    [0.90, 0.20, 0.10, 0.00],
                    [0.00, 0.70, 0.20, 0.10],
                ]),
                "quality": torch.tensor([
                    [0.10, 0.80, 0.20, 0.00],
                    [0.30, 0.10, 0.60, 0.00],
                ]),
                "fused": torch.tensor([
                    [0.00, 0.10, 0.30, 0.90],
                    [0.20, 0.10, 0.00, 0.80],
                ]),
            },
        )

        self.assertEqual(head.context_feature_dim, 3)
        self.assertEqual(
            head.choice_feature_dim,
            base_head.choice_feature_dim + 3,
        )
        self.assertEqual(out["selector_choice_scores"].shape, (2, 3))

    def test_direct_choice_context_bias_applies_to_choice_and_override(self):
        class ZeroChoiceScores(nn.Module):
            def forward(self, features):
                return torch.zeros(
                    features.shape[:-1] + (1,),
                    device=features.device,
                    dtype=features.dtype,
                )

        class FixedSourceBias(nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.register_buffer("bias", bias)

            def forward(self, context):
                return self.bias.to(
                    device=context.device,
                    dtype=context.dtype,
                ).unsqueeze(0).expand(context.shape[0], -1)

        class FixedScalarBias(nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.register_buffer("bias", bias)

            def forward(self, context):
                return self.bias.to(
                    device=context.device,
                    dtype=context.dtype,
                ).expand(context.shape[0])

        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            direct_choice=True,
            separate_override_head=True,
            selector_context_dim=3,
        )
        head.choice_mlp = ZeroChoiceScores()
        head.override_mlp = ZeroChoiceScores()
        head.context_choice_bias_mlp = FixedSourceBias(
            torch.tensor([0.25, -0.50, 0.75])
        )
        head.context_override_bias_mlp = FixedScalarBias(
            torch.tensor(0.40)
        )
        query_feats = torch.randn(2, 4, 4)
        pred_boxes = torch.randn(2, 4, 6)

        out = head(
            query_feats,
            pred_boxes,
            selector_context=torch.randn(2, 3),
            source_scores={
                "base": torch.tensor([
                    [0.90, 0.20, 0.10, 0.00],
                    [0.00, 0.70, 0.20, 0.10],
                ]),
                "quality": torch.tensor([
                    [0.10, 0.80, 0.20, 0.00],
                    [0.30, 0.10, 0.60, 0.00],
                ]),
                "fused": torch.tensor([
                    [0.00, 0.10, 0.30, 0.90],
                    [0.20, 0.10, 0.00, 0.80],
                ]),
            },
        )

        expected_choice = torch.tensor([
            [0.25, -0.50, 0.75],
            [0.25, -0.50, 0.75],
        ])
        expected_override = torch.full((2,), 0.40)

        self.assertEqual(out["selector_choice_scores"].shape, (2, 3))
        self.assertTrue(torch.allclose(
            out["selector_choice_scores"],
            expected_choice,
        ))
        self.assertTrue(torch.allclose(
            out["selector_choice_override_logit"],
            expected_override,
        ))

    def test_candidate_aware_context_bias_applies_to_choice_scores(self):
        class ZeroCandidateScores(nn.Module):
            def forward(self, features):
                return torch.zeros(
                    features.shape[:-1] + (1,),
                    device=features.device,
                    dtype=features.dtype,
                )

        class FixedSourceBias(nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.register_buffer("bias", bias)

            def forward(self, context):
                return self.bias.to(
                    device=context.device,
                    dtype=context.dtype,
                ).unsqueeze(0).expand(context.shape[0], -1)

        head = SourcePoolSelectorHead(
            d_model=4,
            hidden_dim=8,
            candidate_aware=True,
            selector_context_dim=3,
        )
        head.candidate_mlp = ZeroCandidateScores()
        head.context_choice_bias_mlp = FixedSourceBias(
            torch.tensor([-0.10, 0.20, 0.40])
        )
        query_feats = torch.randn(2, 4, 4)
        pred_boxes = torch.randn(2, 4, 6)

        out = head(
            query_feats,
            pred_boxes,
            selector_context=torch.randn(2, 3),
            source_scores={
                "base": torch.tensor([
                    [0.90, 0.20, 0.10, 0.00],
                    [0.00, 0.70, 0.20, 0.10],
                ]),
                "quality": torch.tensor([
                    [0.10, 0.80, 0.20, 0.00],
                    [0.30, 0.10, 0.60, 0.00],
                ]),
                "fused": torch.tensor([
                    [0.00, 0.10, 0.30, 0.90],
                    [0.20, 0.10, 0.00, 0.80],
                ]),
            },
        )

        expected_choice = torch.tensor([
            [-0.10, 0.20, 0.40],
            [-0.10, 0.20, 0.40],
        ])

        self.assertEqual(out["selector_source_scores"].shape, (2, 3, 4))
        self.assertEqual(out["selector_choice_scores"].shape, (2, 3))
        self.assertTrue(torch.allclose(
            out["selector_choice_scores"],
            expected_choice,
        ))

    def test_loss_pushes_best_source_pool_candidate_above_competitor(self):
        selector_scores = torch.tensor(
            [[0.95, 0.10, 0.30, 0.20]], requires_grad=True
        )
        end_points = {
            "selector_scores": selector_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.80, 0.10, 0.00]]),
            "fused_scores": torch.tensor([[0.00, 0.10, 0.80, 0.90]]),
            "pred_iou": torch.tensor([[0.70, 0.20, 0.30, 0.60]]),
            "last_center": torch.tensor([[
                [0.80, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_pool",
            candidate_k=2,
            temperature=1.0,
            min_iou_gap=0.2,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(losses["loss_source_pool_selector"].item(), 0.0)
        self.assertLess(selector_scores.grad[0, 3].item(), 0.0)
        self.assertGreater(selector_scores.grad[0, 0].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_valid_ratio"].item(), 1.0
        )

    def test_candidate_aware_loss_pushes_best_source_candidate(self):
        selector_source_scores = torch.tensor([[
            [0.95, 0.10, 0.30, 0.20],
            [0.20, 0.10, 0.30, 0.10],
            [0.50, 0.10, 0.30, 0.20],
        ]], requires_grad=True)
        end_points = {
            "selector_scores": selector_source_scores.max(dim=1).values,
            "selector_source_scores": selector_source_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.80, 0.10, 0.00]]),
            "fused_scores": torch.tensor([[0.00, 0.10, 0.80, 0.90]]),
            "pred_iou": torch.tensor([[0.70, 0.20, 0.30, 0.60]]),
            "last_center": torch.tensor([[
                [0.80, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_pool",
            candidate_k=2,
            temperature=1.0,
            min_iou_gap=0.2,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(losses["loss_source_pool_selector"].item(), 0.0)
        self.assertLess(selector_source_scores.grad[0, 1, 3].item(), 0.0)
        self.assertGreater(selector_source_scores.grad[0, 0, 0].item(), 0.0)

    def test_candidate_aware_threshold_bucket_bce_trains_all_pool_candidates(self):
        selector_source_scores = torch.tensor([[
            [-2.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, -2.0, 2.0],
            [0.0, 2.0, 0.0, -2.0],
        ]], requires_grad=True)
        end_points = {
            "selector_scores": selector_source_scores.max(dim=1).values,
            "selector_source_scores": selector_source_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.80, 0.10, 0.00]]),
            "fused_scores": torch.tensor([[0.00, 0.10, 0.80, 0.90]]),
            "pred_iou": torch.tensor([[0.10, 0.90, 0.20, 0.80]]),
            "last_center": torch.tensor([[
                [0.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.80, 0.0, 0.0],
                [0.20, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_pool",
            candidate_k=2,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_bucket_bce",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_source_scores.grad[0, 0, 0].item(), 0.0)
        self.assertGreater(selector_source_scores.grad[0, 0, 1].item(), 0.0)
        self.assertLess(selector_source_scores.grad[0, 1, 2].item(), 0.0)
        self.assertLess(selector_source_scores.grad[0, 2, 3].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_valid_ratio"].item(), 1.0
        )
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_threshold_bucket_bce"
            ],
            1.0,
        )

    def test_candidate_aware_source_pool_loss_can_target_contrastive_candidate(self):
        selector_source_scores = torch.zeros(1, 4, 4, requires_grad=True)
        end_points = {
            "selector_scores": selector_source_scores.max(dim=1).values,
            "selector_source_scores": selector_source_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.00, 0.90, 0.10, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.10, 0.90, 0.20]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.10, 0.20, 0.95]
            ]),
            "last_center": torch.tensor([[
                [1.50, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.80, 0.0, 0.0],
                [0.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_pool",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertEqual(
            losses["dbg_source_pool_selector_num_candidate_sources"],
            4.0,
        )
        self.assertLess(selector_source_scores.grad[0, 3, 3].item(), 0.0)
        self.assertGreater(selector_source_scores.grad[0, 2, 2].item(), 0.0)

    def test_candidate_aware_source_pool_lse_trains_shared_query_sources(self):
        selector_source_scores = torch.zeros(1, 3, 4, requires_grad=True)
        end_points = {
            "selector_scores": selector_source_scores.max(dim=1).values,
            "selector_source_scores": selector_source_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.80, 0.10, 0.00]]),
            "fused_scores": torch.tensor([[0.00, 0.10, 0.80, 0.90]]),
            "pred_iou": torch.tensor([[0.10, 0.80, 0.20, 0.90]]),
            "last_center": torch.tensor([[
                [0.80, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_pool",
            candidate_k=2,
            temperature=1.0,
            min_iou_gap=0.02,
            pairwise_weight=0.0,
            choice_target="source_pool_lse",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_source_scores.grad[0, 1, 3].item(), 0.0)
        self.assertLess(selector_source_scores.grad[0, 2, 3].item(), 0.0)
        self.assertGreater(selector_source_scores.grad[0, 0, 0].item(), 0.0)
        self.assertEqual(
            losses[
            "dbg_source_pool_selector_choice_target_source_pool_lse"
            ],
            1.0,
        )

    def test_candidate_aware_source_pool_quality_override_defaults_to_quality(self):
        selector_source_scores = torch.tensor([[
            [0.90, 0.10, 0.00, 0.00],
            [0.00, 0.80, 0.10, 0.00],
            [0.00, 0.00, 0.78, 0.00],
        ]], requires_grad=True)
        end_points = {
            "selector_scores": selector_source_scores.max(dim=1).values,
            "selector_source_scores": selector_source_scores,
            "base_grounding_scores": torch.tensor([
                [0.90, 0.10, 0.00, 0.00]
            ]),
            "fused_scores": torch.tensor([
                [0.00, 0.80, 0.10, 0.00]
            ]),
            "pred_iou": torch.tensor([[0.74, 0.10, 0.78, 0.00]]),
            "last_center": torch.tensor([[
                [0.740, 0.0, 0.0],
                [5.000, 0.0, 0.0],
                [0.780, 0.0, 0.0],
                [5.000, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_pool",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.05,
            pairwise_weight=0.0,
            choice_target="quality_override",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertAlmostEqual(
            losses[
                "dbg_source_pool_selector_choice_target_quality_override"
            ],
            1.0,
        )
        self.assertLess(selector_source_scores.grad[0, 2, 2].item(), 0.0)
        self.assertGreater(selector_source_scores.grad[0, 0, 0].item(), 0.0)

    def test_candidate_aware_source_pool_quality_override_uses_override(self):
        selector_source_scores = torch.tensor([[
            [0.90, 0.10, 0.00, 0.00],
            [0.00, 0.70, 0.10, 0.00],
            [0.00, 0.00, 0.68, 0.00],
        ]], requires_grad=True)
        end_points = {
            "selector_scores": selector_source_scores.max(dim=1).values,
            "selector_source_scores": selector_source_scores,
            "base_grounding_scores": torch.tensor([
                [0.90, 0.10, 0.00, 0.00]
            ]),
            "fused_scores": torch.tensor([
                [0.00, 0.70, 0.10, 0.00]
            ]),
            "pred_iou": torch.tensor([[0.10, 0.10, 0.68, 0.00]]),
            "last_center": torch.tensor([[
                [0.665, 0.0, 0.0],
                [5.000, 0.0, 0.0],
                [0.900, 0.0, 0.0],
                [5.000, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_pool",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.05,
            pairwise_weight=0.0,
            choice_target="quality_override",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertAlmostEqual(
            losses[
                "dbg_source_pool_selector_choice_target_quality_override"
            ],
            1.0,
        )
        self.assertLess(selector_source_scores.grad[0, 0, 0].item(), 0.0)
        self.assertGreater(selector_source_scores.grad[0, 2, 2].item(), 0.0)

    def test_candidate_aware_threshold_utility_softmax_trains_multiple_good_candidates(self):
        selector_source_scores = torch.zeros(1, 3, 4, requires_grad=True)
        end_points = {
            "selector_scores": selector_source_scores.max(dim=1).values,
            "selector_source_scores": selector_source_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.00, 0.90, 0.10, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.10, 0.90, 0.00]]),
            "last_center": torch.tensor([[
                [0.00, 0.0, 0.0],
                [0.20, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_pool",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=2.0,
            pairwise_weight=0.0,
            choice_target="threshold_utility_softmax",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_source_scores.grad[0, 0, 0].item(), 0.0)
        self.assertLess(selector_source_scores.grad[0, 1, 1].item(), 0.0)
        self.assertGreater(selector_source_scores.grad[0, 2, 2].item(), 0.0)
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_threshold_utility_softmax"
            ],
            1.0,
        )
        self.assertTrue(torch.isfinite(
            losses["dbg_source_pool_selector_score_gap"]
        ))
        self.assertTrue(torch.isfinite(
            losses["dbg_source_pool_selector_neg_score"]
        ))
        self.assertLess(
            abs(losses["dbg_source_pool_selector_score_gap"].item()),
            1e6,
        )

    def test_direct_choice_threshold_utility_softmax_trains_soft_oracle_sources(self):
        selector_choice_scores = torch.zeros(1, 3, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 3, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_source_names": (
                "base",
                "quality",
                "detector_jointtight",
            ),
            "base_grounding_scores": torch.tensor([[3.0, 1.0, 0.0]]),
            "pred_iou": torch.tensor([[0.0, 3.0, 1.0]]),
            "detector_jointtight_scores": torch.tensor([[0.0, 1.0, 3.0]]),
            "last_center": torch.tensor([[
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=2.0,
            choice_target="threshold_utility_softmax",
            pairwise_weight=0.0,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[0, 1].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_base_ratio"].item(),
            1.0,
        )
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_threshold_utility_softmax"
            ],
            1.0,
        )

    def test_direct_choice_threshold_utility_regression_learns_source_utilities(self):
        selector_choice_scores = torch.tensor(
            [[0.0, 0.0, 2.0]],
            requires_grad=True,
        )
        end_points = {
            "selector_scores": torch.zeros(1, 3, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_source_names": (
                "base",
                "quality",
                "detector_jointtight",
            ),
            "base_grounding_scores": torch.tensor([[3.0, 1.0, 0.0]]),
            "pred_iou": torch.tensor([[0.0, 3.0, 1.0]]),
            "detector_jointtight_scores": torch.tensor([[0.0, 1.0, 3.0]]),
            "last_center": torch.tensor([[
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.05,
            choice_target="threshold_utility_regression",
            pairwise_weight=0.0,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[0, 1].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_threshold_utility_regression"
            ],
            1.0,
        )

    def test_candidate_aware_threshold_utility_hard_uses_metric_margin(self):
        selector_source_scores = torch.zeros(1, 3, 4, requires_grad=True)
        end_points = {
            "selector_scores": selector_source_scores.max(dim=1).values,
            "selector_source_scores": selector_source_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.00, 0.90, 0.10, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.10, 0.90, 0.00]]),
            "last_center": torch.tensor([[
                [0.60, 0.0, 0.0],
                [0.68, 0.0, 0.0],
                [1.40, 0.0, 0.0],
                [5.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_pool",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            pairwise_weight=1.0,
            choice_target="threshold_utility_hard",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_source_scores.grad[0, 0, 0].item(), 0.0)
        self.assertGreater(selector_source_scores.grad[0, 1, 1].item(), 0.0)
        self.assertGreater(
            losses["dbg_source_pool_selector_pairwise_loss"].item(),
            0.9,
        )
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_threshold_utility_hard"
            ],
            1.0,
        )

    def test_direct_choice_loss_pushes_best_source_top1_candidate(self):
        selector_choice_scores = torch.tensor(
            [[0.10, 0.90, 0.20]], requires_grad=True
        )
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.80, 0.10, 0.00]]),
            "fused_scores": torch.tensor([[0.00, 0.10, 0.80, 0.90]]),
            "pred_iou": torch.tensor([[0.40, 0.20, 0.30, 0.60]]),
            "last_center": torch.tensor([[
                [0.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.40, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.2,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(losses["loss_source_pool_selector"].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 1].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_valid_ratio"].item(), 1.0
        )

    def test_direct_choice_loss_can_target_contrastive_source_candidate(self):
        selector_choice_scores = torch.tensor(
            [[0.80, 0.30, 0.20, 0.10]], requires_grad=True
        )
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.20, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.20, 0.90, 0.00]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.10, 0.20, 0.00, 0.95]
            ]),
            "last_center": torch.tensor([[
                [5.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_choice_scores.grad[0, 3].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 0].item(), 0.0)

    def test_direct_choice_loss_can_compute_contrastive_source_scores(self):
        selector_choice_scores = torch.tensor(
            [[0.80, 0.30, 0.20, 0.10]], requires_grad=True
        )
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.20, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.20, 0.90, 0.00]]),
            "proj_tokens": torch.tensor([[
                [1.0, 0.0],
                [-1.0, 0.0],
            ]]),
            "last_proj_queries": torch.tensor([[
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
            ]]),
            "positive_map": torch.tensor([[[1.0, 0.0]]]),
            "last_center": torch.tensor([[
                [5.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_choice_scores.grad[0, 3].item(), 0.0)
        self.assertEqual(
            losses["dbg_source_pool_selector_num_candidate_sources"],
            4.0,
        )

    def test_direct_choice_loss_uses_soft_token_bbs_base_when_available(self):
        selector_choice_scores = torch.tensor(
            [[0.10, 0.90, 0.20]], requires_grad=True
        )
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            # Contrastive/BBF base prefers query 2, but BBS soft-token base
            # below prefers query 0. Source-choice supervision should match
            # the BBS eval source.
            "base_grounding_scores": torch.tensor([[0.10, 0.20, 0.95, 0.00]]),
            "fused_scores": torch.tensor([[0.00, 0.10, 0.20, 0.90]]),
            "pred_iou": torch.tensor([[0.10, 0.20, 0.30, 0.80]]),
            "last_sem_cls_scores": torch.tensor([[
                [0.0, 5.0],
                [5.0, 0.0],
                [5.0, 0.0],
                [5.0, 0.0],
            ]]),
            "positive_map": torch.tensor([[[0.0, 1.0]]]),
            "last_center": torch.tensor([[
                [0.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.40, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.1,
            choice_balance=True,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 1].item(), 0.0)

    def test_direct_choice_loss_balances_rare_source_targets(self):
        selector_choice_scores = torch.zeros(4, 3, requires_grad=True)
        centers = torch.tensor([
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ])
        sem_cls_scores = torch.tensor([[
            [0.0, 5.0],
            [5.0, 0.0],
            [5.0, 0.0],
            [5.0, 0.0],
        ]]).expand(4, -1, -1).clone()
        end_points = {
            "selector_scores": torch.zeros(4, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.zeros(4, 4),
            "fused_scores": torch.tensor([
                [0.0, 0.1, 0.2, 0.9],
                [0.0, 0.1, 0.2, 0.9],
                [0.0, 0.1, 0.2, 0.9],
                [0.9, 0.1, 0.2, 0.0],
            ]),
            "pred_iou": torch.tensor([
                [0.1, 0.2, 0.3, 0.9],
                [0.1, 0.2, 0.3, 0.9],
                [0.1, 0.2, 0.3, 0.9],
                [0.1, 0.2, 0.3, 0.9],
            ]),
            "last_sem_cls_scores": sem_cls_scores,
            "positive_map": torch.tensor([[[0.0, 1.0]]]).expand(4, -1, -1),
            "last_center": centers,
            "last_pred_size": torch.ones(4, 4, 3) * 2.0,
            "center_label": torch.zeros(4, 1, 3),
            "size_gts": torch.ones(4, 1, 3) * 2.0,
            "box_label_mask": torch.ones(4, 1),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.1,
            choice_balance=True,
        )
        losses["loss_source_pool_selector"].backward()

        rare_quality_grad = abs(selector_choice_scores.grad[3, 2].item())
        common_base_grad = abs(selector_choice_scores.grad[0, 0].item())
        self.assertGreater(rare_quality_grad, common_base_grad)

    def test_direct_choice_balance_power_softens_class_weights(self):
        selector_choice_scores = torch.zeros(4, 3, requires_grad=True)
        centers = torch.tensor([
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ])
        sem_cls_scores = torch.tensor([[
            [0.0, 5.0],
            [5.0, 0.0],
            [5.0, 0.0],
            [5.0, 0.0],
        ]]).expand(4, -1, -1).clone()
        end_points = {
            "selector_scores": torch.zeros(4, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.zeros(4, 4),
            "fused_scores": torch.tensor([
                [0.0, 0.1, 0.2, 0.9],
                [0.0, 0.1, 0.2, 0.9],
                [0.0, 0.1, 0.2, 0.9],
                [0.9, 0.1, 0.2, 0.0],
            ]),
            "pred_iou": torch.tensor([
                [0.1, 0.2, 0.3, 0.9],
                [0.1, 0.2, 0.3, 0.9],
                [0.1, 0.2, 0.3, 0.9],
                [0.1, 0.2, 0.3, 0.9],
            ]),
            "last_sem_cls_scores": sem_cls_scores,
            "positive_map": torch.tensor([[[0.0, 1.0]]]).expand(4, -1, -1),
            "last_center": centers,
            "last_pred_size": torch.ones(4, 4, 3) * 2.0,
            "center_label": torch.zeros(4, 1, 3),
            "size_gts": torch.ones(4, 1, 3) * 2.0,
            "box_label_mask": torch.ones(4, 1),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.1,
            choice_balance=True,
            choice_balance_power=0.5,
        )

        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_class_weight_base"].item(),
            (4.0 / (2.0 * 3.0)) ** 0.5,
            places=4,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_class_weight_quality"].item(),
            (4.0 / (2.0 * 1.0)) ** 0.5,
            places=4,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_choice_balance_power"],
            0.5,
            places=4,
        )

    def test_direct_choice_loss_allows_near_tie_source_targets(self):
        selector_choice_scores = torch.zeros(1, 3, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.zeros(1, 4),
            "fused_scores": torch.tensor([[0.0, 0.1, 0.9, 0.2]]),
            "pred_iou": torch.tensor([[0.1, 0.2, 0.3, 0.9]]),
            "last_sem_cls_scores": torch.tensor([[
                [0.0, 5.0],
                [5.0, 0.0],
                [5.0, 0.0],
                [5.0, 0.0],
            ]]),
            "positive_map": torch.tensor([[[0.0, 1.0]]]),
            "last_center": torch.tensor([[
                [0.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.02, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.05,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 1].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[0, 2].item(), 0.0)

    def test_direct_choice_threshold_target_penalizes_cross_threshold_ties(self):
        selector_choice_scores = torch.zeros(1, 3, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.9, 0.1, 0.0, 0.0]]),
            "fused_scores": torch.tensor([[0.0, 0.1, 0.0, 0.9]]),
            "pred_iou": torch.tensor([[0.0, 0.1, 0.2, 0.0]]),
            "last_center": torch.tensor([[
                [0.685, 0.0, 0.0],  # IoU just below 0.50
                [5.000, 0.0, 0.0],
                [5.000, 0.0, 0.0],
                [0.665, 0.0, 0.0],  # IoU just above 0.50
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.05,
            choice_target="threshold_utility",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[0, 1].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_valid_ratio"].item(), 1.0
        )

    def test_direct_choice_threshold_bucket_target_rewards_all_top_bucket_sources(self):
        selector_choice_scores = torch.tensor(
            [[4.0, 0.0, -4.0, 0.0]], requires_grad=True
        )
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [0.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.20, 0.0, 0.0],
                [1.20, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_bucket",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 3].item(), 0.0)

    def test_direct_choice_threshold_bucket_bce_does_not_compete_positives(self):
        selector_choice_scores = torch.tensor(
            [[6.0, -6.0, 0.0, -6.0]], requires_grad=True
        )
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [0.00, 0.0, 0.0],  # base hit
                [5.00, 0.0, 0.0],  # fused miss
                [0.20, 0.0, 0.0],  # quality hit
                [5.00, 0.0, 0.0],  # contrastive miss
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_bucket_bce",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLessEqual(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 1].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 3].item(), 0.0)
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_threshold_bucket_bce"
            ],
            1.0,
        )

    def test_direct_choice_threshold_bucket_argmax_competes_equal_bucket_sources(self):
        selector_choice_scores = torch.zeros(1, 4, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [0.00, 0.0, 0.0],  # base hit, tie-break winner
                [5.00, 0.0, 0.0],  # fused miss
                [0.20, 0.0, 0.0],  # quality hit in same bucket
                [5.00, 0.0, 0.0],  # contrastive miss
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_bucket_argmax",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_threshold_bucket_argmax"
            ],
            1.0,
        )

    def test_direct_choice_threshold_bucket_argmax_ignores_all_miss_bucket(self):
        selector_choice_scores = torch.zeros(1, 4, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [5.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_bucket_argmax",
        )

        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_valid_ratio"].item(), 0.0
        )
        self.assertAlmostEqual(
            losses["loss_source_pool_selector_raw"].item(), 0.0
        )

    def test_direct_choice_threshold_bucket_unique_skips_equal_bucket_sources(self):
        selector_choice_scores = torch.zeros(1, 4, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [0.00, 0.0, 0.0],  # base hit
                [5.00, 0.0, 0.0],  # fused miss
                [0.20, 0.0, 0.0],  # quality hit in same bucket
                [5.00, 0.0, 0.0],  # contrastive miss
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_bucket_unique",
        )

        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_valid_ratio"].item(), 0.0
        )
        self.assertAlmostEqual(
            losses["loss_source_pool_selector_raw"].item(), 0.0
        )

    def test_direct_choice_threshold_bucket_unique_trains_unique_best_bucket(self):
        selector_choice_scores = torch.zeros(1, 4, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [1.00, 0.0, 0.0],  # base Acc@0.25 hit only
                [5.00, 0.0, 0.0],  # fused miss
                [0.50, 0.0, 0.0],  # quality Acc@0.50 hit
                [5.00, 0.0, 0.0],  # contrastive miss
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_bucket_unique",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_valid_ratio"].item(), 1.0
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_quality_ratio"].item(), 1.0
        )
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_threshold_bucket_unique"
            ],
            1.0,
        )

    def test_direct_choice_threshold_bucket_margin_uses_iou_gap_within_equal_bucket(self):
        selector_choice_scores = torch.zeros(1, 4, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [0.50, 0.0, 0.0],  # base Acc@0.50 hit
                [5.00, 0.0, 0.0],  # fused miss
                [0.00, 0.0, 0.0],  # quality better Acc@0.50 hit
                [5.00, 0.0, 0.0],  # contrastive miss
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_bucket_margin",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_valid_ratio"].item(), 1.0
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_quality_ratio"].item(), 1.0
        )
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_threshold_bucket_margin"
            ],
            1.0,
        )

    def test_direct_choice_threshold_bucket_margin_skips_close_equal_bucket_ties(self):
        selector_choice_scores = torch.zeros(1, 4, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [0.50, 0.0, 0.0],  # base hit
                [5.00, 0.0, 0.0],  # fused miss
                [0.50, 0.0, 0.0],  # quality same-IoU hit
                [5.00, 0.0, 0.0],  # contrastive miss
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_bucket_margin",
        )

        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_valid_ratio"].item(), 0.0
        )
        self.assertAlmostEqual(
            losses["loss_source_pool_selector_raw"].item(), 0.0
        )

    def test_direct_choice_base_threshold_gain_defaults_to_base_for_close_same_bucket(self):
        selector_choice_scores = torch.zeros(1, 4, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [0.50, 0.0, 0.0],  # base Acc@0.50 hit
                [5.00, 0.0, 0.0],  # fused miss
                [0.50, 0.0, 0.0],  # quality same bucket, no IoU gain
                [5.00, 0.0, 0.0],  # contrastive miss
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_threshold_gain",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_valid_ratio"].item(), 1.0
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_base_ratio"].item(), 1.0
        )
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_base_threshold_gain"
            ],
            1.0,
        )

    def test_direct_choice_base_threshold_gain_overrides_base_for_same_bucket_iou_gain(self):
        selector_choice_scores = torch.zeros(1, 4, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [0.50, 0.0, 0.0],  # base Acc@0.50 hit
                [5.00, 0.0, 0.0],  # fused miss
                [0.00, 0.0, 0.0],  # quality same bucket with large IoU gain
                [5.00, 0.0, 0.0],  # contrastive miss
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_threshold_gain",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_valid_ratio"].item(), 1.0
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_quality_ratio"].item(), 1.0
        )

    def test_direct_choice_base_threshold_gain_overrides_base_for_threshold_gain(self):
        selector_choice_scores = torch.zeros(1, 4, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [1.00, 0.0, 0.0],  # base Acc@0.25 hit only
                [5.00, 0.0, 0.0],  # fused miss
                [0.50, 0.0, 0.0],  # quality Acc@0.50 hit
                [5.00, 0.0, 0.0],  # contrastive miss
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_threshold_gain",
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_valid_ratio"].item(), 1.0
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_quality_ratio"].item(), 1.0
        )
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_base_threshold_gain"
            ],
            1.0,
        )

    def test_direct_choice_oracle_prior_matches_batch_source_histogram(self):
        matched_choice_scores = torch.tensor([
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ], requires_grad=True)
        mismatch_choice_scores = torch.tensor([
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 3.0],
        ], requires_grad=True)
        common_end_points = {
            "selector_scores": torch.zeros(2, 3, requires_grad=True),
            "base_grounding_scores": torch.tensor([
                [3.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
            ]),
            "fused_scores": torch.tensor([
                [0.0, 3.0, 1.0],
                [0.0, 3.0, 1.0],
            ]),
            "pred_iou": torch.tensor([
                [0.0, 1.0, 3.0],
                [0.0, 1.0, 3.0],
            ]),
            "last_center": torch.tensor([
                [
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ],
                [
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ]),
            "last_pred_size": torch.ones(2, 3, 3) * 2.0,
            "center_label": torch.tensor([
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
            ]),
            "size_gts": torch.tensor([
                [[2.0, 2.0, 2.0]],
                [[2.0, 2.0, 2.0]],
            ]),
            "box_label_mask": torch.tensor([[1], [1]]),
        }

        matched_losses = _source_pool_selector_losses(
            {
                **common_end_points,
                "selector_choice_scores": matched_choice_scores,
            },
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.0,
            pairwise_weight=0.0,
            choice_target="iou",
            oracle_prior_weight=1.0,
        )
        mismatch_losses = _source_pool_selector_losses(
            {
                **common_end_points,
                "selector_choice_scores": mismatch_choice_scores,
            },
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.0,
            pairwise_weight=0.0,
            choice_target="iou",
            oracle_prior_weight=1.0,
        )

        self.assertAlmostEqual(
            matched_losses["dbg_source_pool_selector_target_base_ratio"].item(),
            0.5,
        )
        self.assertAlmostEqual(
            matched_losses["dbg_source_pool_selector_target_quality_ratio"].item(),
            0.5,
        )
        self.assertLess(
            matched_losses["dbg_source_pool_selector_oracle_prior_loss"].item(),
            mismatch_losses["dbg_source_pool_selector_oracle_prior_loss"].item(),
        )
        self.assertLess(
            matched_losses["loss_source_pool_selector"].item(),
            mismatch_losses["loss_source_pool_selector"].item(),
        )

    def test_direct_choice_oracle_prior_weight_contributes_to_loss(self):
        selector_choice_scores = torch.tensor([
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 3.0],
        ], requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(2, 3, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([
                [3.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
            ]),
            "fused_scores": torch.tensor([
                [0.0, 3.0, 1.0],
                [0.0, 3.0, 1.0],
            ]),
            "pred_iou": torch.tensor([
                [0.0, 1.0, 3.0],
                [0.0, 1.0, 3.0],
            ]),
            "last_center": torch.tensor([
                [
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ],
                [
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ]),
            "last_pred_size": torch.ones(2, 3, 3) * 2.0,
            "center_label": torch.tensor([
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
            ]),
            "size_gts": torch.tensor([
                [[2.0, 2.0, 2.0]],
                [[2.0, 2.0, 2.0]],
            ]),
            "box_label_mask": torch.tensor([[1], [1]]),
        }

        no_prior_losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.0,
            pairwise_weight=0.0,
            choice_target="iou",
            oracle_prior_weight=0.0,
        )
        prior_losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.0,
            pairwise_weight=0.0,
            choice_target="iou",
            oracle_prior_weight=1.0,
        )

        expected_delta = prior_losses[
            "dbg_source_pool_selector_oracle_prior_loss"
        ].item()
        actual_delta = (
            prior_losses["loss_source_pool_selector"].item()
            - no_prior_losses["loss_source_pool_selector"].item()
        )
        self.assertGreater(expected_delta, 0.0)
        self.assertAlmostEqual(actual_delta, expected_delta, places=6)

    def test_quality_override_prior_matches_override_head_batch_rate(self):
        selector_choice_scores = torch.zeros(2, 4, requires_grad=True)
        common_end_points = {
            "selector_scores": torch.zeros(2, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([
                [3.0, 1.0, 0.0, 0.0],
                [3.0, 1.0, 0.0, 0.0],
            ]),
            "fused_scores": torch.tensor([
                [0.0, 3.0, 1.0, 0.0],
                [0.0, 3.0, 1.0, 0.0],
            ]),
            "pred_iou": torch.tensor([
                [0.0, 1.0, 3.0, 0.0],
                [0.0, 1.0, 3.0, 0.0],
            ]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 3.0],
            ]),
            "last_center": torch.tensor([
                [
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ],
                [
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ],
            ]),
            "last_pred_size": torch.ones(2, 4, 3) * 2.0,
            "center_label": torch.tensor([
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
            ]),
            "size_gts": torch.tensor([
                [[2.0, 2.0, 2.0]],
                [[2.0, 2.0, 2.0]],
            ]),
            "box_label_mask": torch.tensor([[1], [1]]),
        }
        matched_losses = _source_pool_selector_losses(
            {
                **common_end_points,
                "selector_choice_override_logit": torch.tensor(
                    [3.0, -3.0], requires_grad=True
                ),
            },
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.0,
            pairwise_weight=0.0,
            choice_target="quality_override",
            override_prior_weight=1.0,
        )
        mismatch_losses = _source_pool_selector_losses(
            {
                **common_end_points,
                "selector_choice_override_logit": torch.tensor(
                    [3.0, 3.0], requires_grad=True
                ),
            },
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.0,
            pairwise_weight=0.0,
            choice_target="quality_override",
            override_prior_weight=1.0,
        )

        self.assertAlmostEqual(
            matched_losses[
                "dbg_source_pool_selector_target_override_ratio"
            ].item(),
            0.5,
        )
        self.assertLess(
            matched_losses[
                "dbg_source_pool_selector_override_prior_loss"
            ].item(),
            mismatch_losses[
                "dbg_source_pool_selector_override_prior_loss"
            ].item(),
        )

    def test_quality_override_prior_weight_contributes_to_loss(self):
        selector_choice_scores = torch.zeros(2, 4, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(2, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_override_logit": torch.tensor(
                [3.0, 3.0], requires_grad=True
            ),
            "base_grounding_scores": torch.tensor([
                [3.0, 1.0, 0.0, 0.0],
                [3.0, 1.0, 0.0, 0.0],
            ]),
            "fused_scores": torch.tensor([
                [0.0, 3.0, 1.0, 0.0],
                [0.0, 3.0, 1.0, 0.0],
            ]),
            "pred_iou": torch.tensor([
                [0.0, 1.0, 3.0, 0.0],
                [0.0, 1.0, 3.0, 0.0],
            ]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 3.0],
            ]),
            "last_center": torch.tensor([
                [
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ],
                [
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ],
            ]),
            "last_pred_size": torch.ones(2, 4, 3) * 2.0,
            "center_label": torch.tensor([
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
            ]),
            "size_gts": torch.tensor([
                [[2.0, 2.0, 2.0]],
                [[2.0, 2.0, 2.0]],
            ]),
            "box_label_mask": torch.tensor([[1], [1]]),
        }
        no_prior_losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.0,
            pairwise_weight=0.0,
            choice_target="quality_override",
            override_prior_weight=0.0,
        )
        prior_losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.0,
            pairwise_weight=0.0,
            choice_target="quality_override",
            override_prior_weight=1.0,
        )

        expected_delta = prior_losses[
            "dbg_source_pool_selector_override_prior_loss"
        ].item()
        actual_delta = (
            prior_losses["loss_source_pool_selector"].item()
            - no_prior_losses["loss_source_pool_selector"].item()
        )
        self.assertGreater(expected_delta, 0.0)
        self.assertAlmostEqual(actual_delta, expected_delta, places=6)

    def test_quality_override_source_prior_matches_override_source_histogram(self):
        matched_choice_scores = torch.tensor([
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
        ], requires_grad=True)
        mismatch_choice_scores = torch.tensor([
            [4.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ], requires_grad=True)
        common_end_points = {
            "selector_scores": torch.zeros(2, 4, requires_grad=True),
            "base_grounding_scores": torch.tensor([
                [3.0, 1.0, 0.0, 0.0],
                [3.0, 1.0, 0.0, 0.0],
            ]),
            "fused_scores": torch.tensor([
                [0.0, 3.0, 1.0, 0.0],
                [0.0, 3.0, 1.0, 0.0],
            ]),
            "pred_iou": torch.tensor([
                [0.0, 1.0, 3.0, 0.0],
                [0.0, 1.0, 3.0, 0.0],
            ]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 3.0],
            ]),
            "last_center": torch.tensor([
                [
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ],
                [
                    [5.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ],
            ]),
            "last_pred_size": torch.ones(2, 4, 3) * 2.0,
            "center_label": torch.tensor([
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
            ]),
            "size_gts": torch.tensor([
                [[2.0, 2.0, 2.0]],
                [[2.0, 2.0, 2.0]],
            ]),
            "box_label_mask": torch.tensor([[1], [1]]),
        }

        matched_losses = _source_pool_selector_losses(
            {
                **common_end_points,
                "selector_choice_scores": matched_choice_scores,
            },
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.0,
            pairwise_weight=0.0,
            choice_target="quality_override",
            override_source_prior_weight=1.0,
        )
        mismatch_losses = _source_pool_selector_losses(
            {
                **common_end_points,
                "selector_choice_scores": mismatch_choice_scores,
            },
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.0,
            pairwise_weight=0.0,
            choice_target="quality_override",
            override_source_prior_weight=1.0,
        )

        self.assertAlmostEqual(
            matched_losses[
                "dbg_source_pool_selector_target_override_ratio"
            ].item(),
            1.0,
        )
        self.assertLess(
            matched_losses[
                "dbg_source_pool_selector_override_source_prior_loss"
            ].item(),
            mismatch_losses[
                "dbg_source_pool_selector_override_source_prior_loss"
            ].item(),
        )

    def test_quality_override_source_prior_weight_contributes_to_loss(self):
        selector_choice_scores = torch.tensor([
            [4.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ], requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(2, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([
                [3.0, 1.0, 0.0, 0.0],
                [3.0, 1.0, 0.0, 0.0],
            ]),
            "fused_scores": torch.tensor([
                [0.0, 3.0, 1.0, 0.0],
                [0.0, 3.0, 1.0, 0.0],
            ]),
            "pred_iou": torch.tensor([
                [0.0, 1.0, 3.0, 0.0],
                [0.0, 1.0, 3.0, 0.0],
            ]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 3.0],
            ]),
            "last_center": torch.tensor([
                [
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ],
                [
                    [5.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ],
            ]),
            "last_pred_size": torch.ones(2, 4, 3) * 2.0,
            "center_label": torch.tensor([
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
            ]),
            "size_gts": torch.tensor([
                [[2.0, 2.0, 2.0]],
                [[2.0, 2.0, 2.0]],
            ]),
            "box_label_mask": torch.tensor([[1], [1]]),
        }

        no_prior_losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.0,
            pairwise_weight=0.0,
            choice_target="quality_override",
            override_source_prior_weight=0.0,
        )
        prior_losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.0,
            pairwise_weight=0.0,
            choice_target="quality_override",
            override_source_prior_weight=1.0,
        )

        expected_delta = prior_losses[
            "dbg_source_pool_selector_override_source_prior_loss"
        ].item()
        actual_delta = (
            prior_losses["loss_source_pool_selector"].item()
            - no_prior_losses["loss_source_pool_selector"].item()
        )
        self.assertGreater(expected_delta, 0.0)
        self.assertAlmostEqual(actual_delta, expected_delta, places=6)

    def test_direct_choice_base_override_bce_keeps_base_without_override_gain(self):
        selector_choice_scores = torch.zeros(1, 4, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [0.50, 0.0, 0.0],  # base Acc@0.50 hit
                [5.00, 0.0, 0.0],  # fused miss
                [0.50, 0.0, 0.0],  # quality same bucket, no IoU gain
                [5.00, 0.0, 0.0],  # contrastive miss
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_bce",
            pairwise_weight=0.0,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_base_ratio"].item(), 1.0
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_override_ratio"].item(),
            0.0,
        )
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_base_override_bce"
            ],
            1.0,
        )

    def test_direct_choice_base_override_bce_trains_override_and_false_base_diagnostic(self):
        selector_choice_scores = torch.zeros(1, 4, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [1.00, 0.0, 0.0],  # base Acc@0.25 hit only
                [5.00, 0.0, 0.0],  # fused miss
                [0.50, 0.0, 0.0],  # quality Acc@0.50 hit
                [5.00, 0.0, 0.0],  # contrastive miss
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_bce",
            pairwise_weight=0.0,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_quality_ratio"].item(), 1.0
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_override_ratio"].item(),
            1.0,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_false_base_ratio"].item(), 1.0
        )

    def test_direct_choice_base_override_bce_uses_separate_override_logit(self):
        selector_choice_scores = torch.zeros(1, 4, requires_grad=True)
        override_logit = torch.tensor([1.0], requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_override_logit": override_logit,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [0.50, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.50, 0.0, 0.0],
                [5.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_bce",
            pairwise_weight=0.0,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(override_logit.grad.item(), 0.0)
        self.assertEqual(selector_choice_scores.grad.abs().sum().item(), 0.0)
        self.assertEqual(
            losses["dbg_source_pool_selector_choice_override_head"],
            1.0,
        )

    def test_direct_choice_base_override_focal_bce_uses_separate_override_logit(self):
        selector_choice_scores = torch.zeros(1, 4, requires_grad=True)
        override_logit = torch.tensor([1.0], requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_override_logit": override_logit,
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [0.50, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.50, 0.0, 0.0],
                [5.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_focal_bce",
            pairwise_weight=0.0,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(override_logit.grad.item(), 0.0)
        self.assertEqual(selector_choice_scores.grad.abs().sum().item(), 0.0)
        self.assertEqual(
            losses["dbg_source_pool_selector_choice_override_head"],
            1.0,
        )
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_base_override_focal_bce"
            ],
            1.0,
        )

    def test_base_override_sourcewise_focal_bce_trains_nondefault_logits(self):
        selector_choice_scores = torch.zeros(1, 3, requires_grad=True)
        override_logit = torch.tensor([1.0], requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 3, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_override_logit": override_logit,
            "selector_choice_source_names": (
                "base",
                "quality",
                "detector_jointtight",
            ),
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.90, 0.10]]),
            "detector_jointtight_scores": torch.tensor([[0.00, 0.10, 0.90]]),
            "last_center": torch.tensor([[
                [1.00, 0.0, 0.0],
                [0.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_sourcewise_focal_bce",
            pairwise_weight=0.0,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_choice_scores.grad[0, 1].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertNotEqual(override_logit.grad.abs().item(), 0.0)
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_base_override_sourcewise_focal_bce"
            ],
            1.0,
        )

    def test_base_override_sourcewise_negative_weight_scales_nondefault_loss(self):
        selector_choice_scores = torch.tensor(
            [[0.0, 0.0, 4.0]], requires_grad=True
        )
        override_logit = torch.tensor([1.0], requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 3, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_override_logit": override_logit,
            "selector_choice_source_names": (
                "base",
                "quality",
                "detector_jointtight",
            ),
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.90, 0.10]]),
            "detector_jointtight_scores": torch.tensor([[0.00, 0.10, 0.90]]),
            "last_center": torch.tensor([[
                [1.00, 0.0, 0.0],
                [0.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        baseline = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_sourcewise_focal_bce",
            pairwise_weight=0.0,
            sourcewise_negative_weight=1.0,
        )
        softened = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_sourcewise_focal_bce",
            pairwise_weight=0.0,
            sourcewise_negative_weight=0.25,
        )

        self.assertLess(
            softened["loss_source_pool_selector"].item(),
            baseline["loss_source_pool_selector"].item(),
        )
        self.assertEqual(
            softened[
                "dbg_source_pool_selector_choice_target_base_override_sourcewise_focal_bce"
            ],
            1.0,
        )

    def test_threshold_gain_default_sourcewise_keeps_default_for_iou_only_gain(self):
        selector_choice_scores = torch.zeros(1, 3, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 3, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_source_names": (
                "base",
                "quality",
                "detector_jointtight",
            ),
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.90, 0.00]]),
            "detector_jointtight_scores": torch.tensor([[0.10, 0.00, 0.90]]),
            "last_center": torch.tensor([[
                [5.00, 0.0, 0.0],  # base miss
                [0.00, 0.0, 0.0],  # quality same bucket, higher IoU
                [0.50, 0.0, 0.0],  # detector Acc@0.50 hit
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_gain_default_sourcewise_focal_bce",
            override_default_source="detector_jointtight",
            pairwise_weight=0.0,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(selector_choice_scores.grad[0, 1].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertAlmostEqual(
            losses[
                "dbg_source_pool_selector_target_detector_jointtight_ratio"
            ].item(),
            1.0,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_override_ratio"].item(),
            0.0,
        )
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_threshold_gain_default_sourcewise_focal_bce"
            ],
            1.0,
        )

    def test_threshold_gain_default_sourcewise_trains_threshold_gain_source(self):
        selector_choice_scores = torch.zeros(1, 3, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 3, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_source_names": (
                "base",
                "quality",
                "detector_jointtight",
            ),
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.90, 0.00]]),
            "detector_jointtight_scores": torch.tensor([[0.10, 0.00, 0.90]]),
            "last_center": torch.tensor([[
                [5.00, 0.0, 0.0],  # base miss
                [0.50, 0.0, 0.0],  # quality Acc@0.50 hit
                [1.00, 0.0, 0.0],  # detector Acc@0.25 hit only
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_gain_default_sourcewise_focal_bce",
            override_default_source="detector_jointtight",
            pairwise_weight=0.0,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_choice_scores.grad[0, 1].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_quality_ratio"].item(),
            1.0,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_override_ratio"].item(),
            1.0,
        )
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_threshold_gain_default_sourcewise_focal_bce"
            ],
            1.0,
        )

    def test_precision_gain_default_sourcewise_keeps_default_for_tiny_threshold_gain(self):
        def center_for_iou(iou):
            return [2.0 * (1.0 - iou) / (1.0 + iou), 0.0, 0.0]

        selector_choice_scores = torch.zeros(1, 2, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 3, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_source_names": (
                "detector_jointtight",
                "quality",
            ),
            "detector_jointtight_scores": torch.tensor([[3.0, 1.0, 0.0]]),
            "pred_iou": torch.tensor([[1.0, 3.0, 0.0]]),
            "last_center": torch.tensor([[
                center_for_iou(0.49),
                center_for_iou(0.51),
                [5.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="precision_gain_default_sourcewise_focal_bce",
            override_default_source="detector_jointtight",
            pairwise_weight=0.0,
            source_min_iou_gaps={"quality": 0.05},
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 1].item(), 0.0)
        self.assertAlmostEqual(
            losses[
                "dbg_source_pool_selector_target_detector_jointtight_ratio"
            ].item(),
            1.0,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_override_ratio"].item(),
            0.0,
        )
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_precision_gain_default_sourcewise_focal_bce"
            ],
            1.0,
        )

    def test_precision_gain_default_sourcewise_trains_high_margin_threshold_gain(self):
        def center_for_iou(iou):
            return [2.0 * (1.0 - iou) / (1.0 + iou), 0.0, 0.0]

        selector_choice_scores = torch.zeros(1, 2, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 3, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_source_names": (
                "detector_jointtight",
                "quality",
            ),
            "detector_jointtight_scores": torch.tensor([[3.0, 1.0, 0.0]]),
            "pred_iou": torch.tensor([[1.0, 3.0, 0.0]]),
            "last_center": torch.tensor([[
                center_for_iou(0.30),
                center_for_iou(0.60),
                [5.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="precision_gain_default_sourcewise_focal_bce",
            override_default_source="detector_jointtight",
            pairwise_weight=0.0,
            source_min_iou_gaps={"quality": 0.05},
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[0, 1].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_quality_ratio"].item(),
            1.0,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_override_ratio"].item(),
            1.0,
        )
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_choice_target_precision_gain_default_sourcewise_focal_bce"
            ],
            1.0,
        )

    def test_threshold_gain_default_diffquery_ignores_same_query_keep_default_rows(self):
        def center_for_iou(iou):
            return [2.0 * (1.0 - iou) / (1.0 + iou), 0.0, 0.0]

        selector_choice_scores = torch.zeros(2, 2, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(2, 3, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_source_names": (
                "detector_jointtight",
                "quality",
            ),
            "detector_jointtight_scores": torch.tensor([
                [3.0, 1.0, 0.0],  # same top query as quality: no choice headroom
                [3.0, 1.0, 0.0],  # detector default is better
            ]),
            "pred_iou": torch.tensor([
                [3.0, 1.0, 0.0],
                [1.0, 3.0, 0.0],
            ]),
            "last_center": torch.tensor([
                [
                    center_for_iou(0.70),
                    center_for_iou(0.30),
                    [5.00, 0.0, 0.0],
                ],
                [
                    center_for_iou(0.70),
                    center_for_iou(0.30),
                    [5.00, 0.0, 0.0],
                ],
            ]),
            "last_pred_size": torch.ones(2, 3, 3) * 2.0,
            "center_label": torch.tensor([
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
            ]),
            "size_gts": torch.tensor([
                [[2.0, 2.0, 2.0]],
                [[2.0, 2.0, 2.0]],
            ]),
            "box_label_mask": torch.tensor([[1], [1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_gain_default_diffquery_sourcewise_focal_bce",
            override_default_source="detector_jointtight",
            pairwise_weight=0.0,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertEqual(
            losses["dbg_source_pool_selector_choice_target_threshold_gain_default_diffquery_sourcewise_focal_bce"],
            1.0,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_valid_ratio"].item(),
            0.5,
        )
        self.assertEqual(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertEqual(selector_choice_scores.grad[0, 1].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[1, 0].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[1, 1].item(), 0.0)

    def test_threshold_gain_iou_aux_ranks_iou_only_nondefault_without_override_target(self):
        def make_end_points(choice_scores):
            return {
                "selector_scores": torch.zeros(1, 3, requires_grad=True),
                "selector_choice_scores": choice_scores,
                "selector_choice_source_names": (
                    "base",
                    "quality",
                    "detector_jointtight",
                ),
                "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00]]),
                "pred_iou": torch.tensor([[0.10, 0.90, 0.00]]),
                "detector_jointtight_scores": torch.tensor([[0.10, 0.00, 0.90]]),
                "last_center": torch.tensor([[
                    [1.00, 0.0, 0.0],  # base: same bucket, best IoU
                    [1.10, 0.0, 0.0],  # quality: same bucket, weaker IoU
                    [1.15, 0.0, 0.0],  # detector default: same bucket
                ]]),
                "last_pred_size": torch.ones(1, 3, 3) * 2.0,
                "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
                "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
                "box_label_mask": torch.tensor([[1]]),
            }

        base_ranked_above_quality = _source_pool_selector_losses(
            make_end_points(
                torch.tensor([[1.0, -1.0, 0.0]], requires_grad=True)
            ),
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_gain_default_sourcewise_focal_bce",
            override_default_source="detector_jointtight",
            pairwise_weight=0.0,
            iou_aux_weight=1.0,
            iou_aux_margin=0.5,
        )
        quality_ranked_above_base = _source_pool_selector_losses(
            make_end_points(
                torch.tensor([[-1.0, 1.0, 0.0]], requires_grad=True)
            ),
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_gain_default_sourcewise_focal_bce",
            override_default_source="detector_jointtight",
            pairwise_weight=0.0,
            iou_aux_weight=1.0,
            iou_aux_margin=0.5,
        )

        self.assertLess(
            base_ranked_above_quality["loss_source_pool_selector"].item(),
            quality_ranked_above_base["loss_source_pool_selector"].item(),
        )
        self.assertAlmostEqual(
            quality_ranked_above_base[
                "dbg_source_pool_selector_target_override_ratio"
            ].item(),
            0.0,
        )
        self.assertAlmostEqual(
            quality_ranked_above_base[
                "dbg_source_pool_selector_target_detector_jointtight_ratio"
            ].item(),
            1.0,
        )
        self.assertEqual(
            quality_ranked_above_base[
                "dbg_source_pool_selector_iou_aux_valid_ratio"
            ],
            1.0,
        )
        self.assertGreater(
            quality_ranked_above_base[
                "dbg_source_pool_selector_iou_aux_loss"
            ].item(),
            0.0,
        )

    def test_quality_base_margin_pushes_quality_logit_when_quality_beats_base(self):
        selector_choice_scores = torch.tensor(
            [[1.00, 0.00, 0.00]], requires_grad=True
        )
        end_points = {
            "selector_scores": torch.zeros(1, 3, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_source_names": (
                "base",
                "quality",
                "detector_jointtight",
            ),
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.90, 0.10]]),
            "detector_jointtight_scores": torch.tensor([[0.00, 0.10, 0.90]]),
            "last_center": torch.tensor([[
                [1.00, 0.0, 0.0],
                [0.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_sourcewise_focal_bce",
            pairwise_weight=0.0,
            quality_base_margin_weight=1.0,
            quality_base_margin=0.75,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(
            losses["dbg_source_pool_selector_quality_base_margin_loss"].item(),
            0.0,
        )
        self.assertEqual(
            losses["dbg_source_pool_selector_quality_base_margin_valid_ratio"],
            1.0,
        )
        self.assertGreater(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[0, 1].item(), 0.0)

    def test_parser_accepts_quality_base_margin_options(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_loss_weight",
            "0.05",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_direct_choice",
            "--source_pool_selector_choice_target",
            "base_override_sourcewise_focal_bce",
            "--source_pool_selector_quality_base_margin_weight",
            "0.4",
            "--source_pool_selector_quality_base_margin",
            "0.8",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(args.source_pool_selector_quality_base_margin_weight, 0.4)
        self.assertEqual(args.source_pool_selector_quality_base_margin, 0.8)

    def test_quality_default_margin_pushes_quality_logit_when_quality_beats_default(self):
        selector_choice_scores = torch.tensor(
            [[0.00, 0.00, 1.00]], requires_grad=True
        )
        end_points = {
            "selector_scores": torch.zeros(1, 3, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_source_names": (
                "base",
                "quality",
                "detector_jointtight",
            ),
            "base_grounding_scores": torch.tensor([[0.10, 0.20, 0.90]]),
            "pred_iou": torch.tensor([[0.90, 0.10, 0.00]]),
            "detector_jointtight_scores": torch.tensor([[0.00, 0.10, 0.90]]),
            "last_center": torch.tensor([[
                [0.00, 0.0, 0.0],
                [2.50, 0.0, 0.0],
                [5.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "source_pool_selector_override_default_source": (
                "detector_jointtight"
            ),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_focal_bce",
            pairwise_weight=0.0,
            quality_default_margin_weight=1.0,
            quality_default_margin=0.75,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(
            losses[
                "dbg_source_pool_selector_quality_default_margin_loss"
            ].item(),
            0.0,
        )
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_quality_default_margin_valid_ratio"
            ],
            1.0,
        )
        self.assertLess(selector_choice_scores.grad[0, 1].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 2].item(), 0.0)

    def test_quality_default_bidirectional_margin_pushes_winning_source_only_on_clear_gap(self):
        def center_for_iou(iou):
            return [2.0 * (1.0 - iou) / (1.0 + iou), 0.0, 0.0]

        def make_end_points():
            selector_choice_scores = torch.tensor([
                [0.00, -1.00, 1.00],  # quality should be pushed up
                [0.00, 1.00, -1.00],  # default should be pushed up
                [0.00, 0.00, 0.00],   # same query: ignored
                [0.00, 0.00, 0.00],   # small gap: ignored
            ], requires_grad=True)
            return selector_choice_scores, {
                "selector_scores": torch.zeros(4, 3, requires_grad=True),
                "selector_choice_scores": selector_choice_scores,
                "selector_choice_source_names": (
                    "base",
                    "quality",
                    "detector_jointtight",
                ),
                "base_grounding_scores": torch.tensor([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]),
                "pred_iou": torch.tensor([
                    [3.0, 1.0, 0.0],
                    [1.0, 3.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                ]),
                "detector_jointtight_scores": torch.tensor([
                    [1.0, 3.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [1.0, 3.0, 0.0],
                ]),
                "last_center": torch.tensor([
                    [
                        center_for_iou(0.70),  # quality top query
                        center_for_iou(0.30),  # detector top query
                        [5.00, 0.0, 0.0],
                    ],
                    [
                        center_for_iou(0.70),  # detector top query
                        center_for_iou(0.30),  # quality top query
                        [5.00, 0.0, 0.0],
                    ],
                    [
                        center_for_iou(0.70),  # both sources choose query 0
                        center_for_iou(0.30),
                        [5.00, 0.0, 0.0],
                    ],
                    [
                        center_for_iou(0.54),  # quality top query, small gain only
                        center_for_iou(0.50),  # detector top query
                        [5.00, 0.0, 0.0],
                    ],
                ]),
                "last_pred_size": torch.ones(4, 3, 3) * 2.0,
                "center_label": torch.tensor([
                    [[0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0]],
                ]),
                "size_gts": torch.tensor([
                    [[2.0, 2.0, 2.0]],
                    [[2.0, 2.0, 2.0]],
                    [[2.0, 2.0, 2.0]],
                    [[2.0, 2.0, 2.0]],
                ]),
                "box_label_mask": torch.tensor([[1], [1], [1], [1]]),
                "source_pool_selector_override_default_source": (
                    "detector_jointtight"
                ),
            }

        def run_with_weight(weight):
            scores, end_points = make_end_points()
            losses = _source_pool_selector_losses(
                end_points,
                weight=1.0,
                source="source_choice",
                candidate_k=1,
                temperature=1.0,
                min_iou_gap=0.05,
                choice_target="threshold_gain_default_sourcewise_focal_bce",
                override_default_source="detector_jointtight",
                pairwise_weight=0.0,
                quality_default_bidirectional_margin_weight=weight,
                quality_default_bidirectional_margin=0.75,
            )
            losses["loss_source_pool_selector"].backward()
            return scores.grad.detach().clone(), losses

        base_grad, _ = run_with_weight(0.0)
        margin_grad, losses = run_with_weight(1.0)
        bidirectional_grad = margin_grad - base_grad

        self.assertGreater(
            losses[
                "dbg_source_pool_selector_quality_default_bidirectional_margin_loss"
            ].item(),
            0.0,
        )
        self.assertEqual(
            losses[
                "dbg_source_pool_selector_quality_default_bidirectional_margin_valid_ratio"
            ],
            0.5,
        )
        self.assertLess(bidirectional_grad[0, 1].item(), 0.0)
        self.assertGreater(bidirectional_grad[0, 2].item(), 0.0)
        self.assertGreater(bidirectional_grad[1, 1].item(), 0.0)
        self.assertLess(bidirectional_grad[1, 2].item(), 0.0)
        self.assertEqual(bidirectional_grad[2, 1].item(), 0.0)
        self.assertEqual(bidirectional_grad[2, 2].item(), 0.0)
        self.assertEqual(bidirectional_grad[3, 1].item(), 0.0)
        self.assertEqual(bidirectional_grad[3, 2].item(), 0.0)

    def test_parser_accepts_quality_default_margin_options(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_loss_weight",
            "0.05",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_direct_choice",
            "--source_pool_selector_choice_target",
            "base_override_focal_bce",
            "--source_pool_selector_quality_default_margin_weight",
            "0.3",
            "--source_pool_selector_quality_default_margin",
            "0.6",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_quality_default_margin_weight,
            0.3,
        )
        self.assertEqual(args.source_pool_selector_quality_default_margin, 0.6)

    def test_parser_accepts_quality_default_bidirectional_margin_options(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_loss_weight",
            "0.05",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_candidate_aware",
            "--source_pool_selector_choice_target",
            "threshold_gain_default_sourcewise_focal_bce",
            "--source_pool_selector_quality_default_bidirectional_margin_weight",
            "0.4",
            "--source_pool_selector_quality_default_bidirectional_margin",
            "0.7",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_quality_default_bidirectional_margin_weight,
            0.4,
        )
        self.assertEqual(
            args.source_pool_selector_quality_default_bidirectional_margin,
            0.7,
        )

    def test_parser_accepts_source_choice_iou_aux_options(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_loss_weight",
            "0.05",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_direct_choice",
            "--source_pool_selector_choice_target",
            "threshold_gain_default_sourcewise_focal_bce",
            "--source_pool_selector_iou_aux_weight",
            "0.2",
            "--source_pool_selector_iou_aux_margin",
            "0.4",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(args.source_pool_selector_iou_aux_weight, 0.2)
        self.assertEqual(args.source_pool_selector_iou_aux_margin, 0.4)

    def test_base_override_asymmetric_weights_emphasize_false_base_risk(self):
        selector_choice_scores = torch.zeros(2, 2, requires_grad=True)
        override_logit = torch.tensor([-2.0, -2.0], requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(2, 2, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_override_logit": override_logit,
            "selector_choice_source_names": ("base", "quality"),
            "base_grounding_scores": torch.tensor([
                [3.0, 1.0],
                [3.0, 1.0],
            ]),
            "pred_iou": torch.tensor([
                [1.0, 3.0],
                [1.0, 3.0],
            ]),
            "last_center": torch.tensor([
                [
                    [5.0, 0.0, 0.0],  # base miss
                    [0.0, 0.0, 0.0],  # quality hit, should override
                ],
                [
                    [0.0, 0.0, 0.0],  # base hit, should stay
                    [5.0, 0.0, 0.0],  # quality miss
                ],
            ]),
            "last_pred_size": torch.ones(2, 2, 3) * 2.0,
            "center_label": torch.tensor([
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
            ]),
            "size_gts": torch.tensor([
                [[2.0, 2.0, 2.0]],
                [[2.0, 2.0, 2.0]],
            ]),
            "box_label_mask": torch.tensor([[1], [1]]),
        }

        baseline = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_bce",
            pairwise_weight=0.0,
            false_base_weight=1.0,
            false_override_weight=1.0,
        )
        asymmetric = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_bce",
            pairwise_weight=0.0,
            false_base_weight=3.0,
            false_override_weight=1.0,
        )

        self.assertGreater(
            asymmetric["loss_source_pool_selector"].item(),
            baseline["loss_source_pool_selector"].item(),
        )
        self.assertEqual(
            asymmetric["dbg_source_pool_selector_false_base_weight"],
            3.0,
        )
        self.assertEqual(
            asymmetric["dbg_source_pool_selector_false_override_weight"],
            1.0,
        )

    def test_base_override_source_prior_weight_contributes_to_loss(self):
        selector_choice_scores = torch.tensor([
            [4.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ], requires_grad=True)
        override_logit = torch.tensor([1.0, 1.0], requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(2, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_override_logit": override_logit,
            "base_grounding_scores": torch.tensor([
                [3.0, 1.0, 0.0, 0.0],
                [3.0, 1.0, 0.0, 0.0],
            ]),
            "fused_scores": torch.tensor([
                [0.0, 3.0, 1.0, 0.0],
                [0.0, 3.0, 1.0, 0.0],
            ]),
            "pred_iou": torch.tensor([
                [0.0, 1.0, 3.0, 0.0],
                [0.0, 1.0, 3.0, 0.0],
            ]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 3.0],
            ]),
            "last_center": torch.tensor([
                [
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ],
                [
                    [5.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ],
            ]),
            "last_pred_size": torch.ones(2, 4, 3) * 2.0,
            "center_label": torch.tensor([
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
            ]),
            "size_gts": torch.tensor([
                [[2.0, 2.0, 2.0]],
                [[2.0, 2.0, 2.0]],
            ]),
            "box_label_mask": torch.tensor([[1], [1]]),
        }

        no_prior_losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.0,
            choice_target="base_override_focal_bce",
            pairwise_weight=0.0,
            override_source_prior_weight=0.0,
        )
        prior_losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.0,
            choice_target="base_override_focal_bce",
            pairwise_weight=0.0,
            override_source_prior_weight=1.0,
        )

        expected_delta = prior_losses[
            "dbg_source_pool_selector_override_source_prior_loss"
        ].item()
        actual_delta = (
            prior_losses["loss_source_pool_selector"].item()
            - no_prior_losses["loss_source_pool_selector"].item()
        )
        self.assertGreater(expected_delta, 0.0)
        self.assertAlmostEqual(actual_delta, expected_delta, places=6)

    def test_base_override_source_specific_gap_keeps_detector_strict(self):
        def center_for_iou(iou):
            return [2.0 * (1.0 - iou) / (1.0 + iou), 0.0, 0.0]

        selector_choice_scores = torch.zeros(2, 3, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(2, 3, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_source_names": (
                "base",
                "quality",
                "detector_jointtight",
            ),
            "base_grounding_scores": torch.tensor([
                [3.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
            ]),
            "pred_iou": torch.tensor([
                [0.0, 3.0, 1.0],
                [0.0, 3.0, 1.0],
            ]),
            "detector_jointtight_scores": torch.tensor([
                [0.0, 1.0, 3.0],
                [0.0, 1.0, 3.0],
            ]),
            "last_center": torch.tensor([
                [
                    center_for_iou(0.30),
                    center_for_iou(0.35),
                    center_for_iou(0.10),
                ],
                [
                    center_for_iou(0.30),
                    center_for_iou(0.10),
                    center_for_iou(0.35),
                ],
            ]),
            "last_pred_size": torch.ones(2, 3, 3) * 2.0,
            "center_label": torch.tensor([
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
            ]),
            "size_gts": torch.tensor([
                [[2.0, 2.0, 2.0]],
                [[2.0, 2.0, 2.0]],
            ]),
            "box_label_mask": torch.tensor([[1], [1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_bce",
            pairwise_weight=0.0,
            source_min_iou_gaps={
                "quality": 0.02,
                "detector_jointtight": 0.10,
            },
        )

        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_quality_ratio"].item(),
            0.5,
        )
        self.assertAlmostEqual(
            losses[
                "dbg_source_pool_selector_target_detector_jointtight_ratio"
            ].item(),
            0.0,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_base_ratio"].item(),
            0.5,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_override_ratio"].item(),
            0.5,
        )
        self.assertEqual(
            losses["dbg_source_pool_selector_source_gap_quality"],
            0.02,
        )
        self.assertEqual(
            losses["dbg_source_pool_selector_source_gap_detector_jointtight"],
            0.10,
        )

    def test_direct_choice_override_default_source_can_be_detector_policy(self):
        selector_choice_scores = torch.zeros(1, 5, requires_grad=True)
        override_logit = torch.tensor([-1.0], requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_override_logit": override_logit,
            "selector_choice_source_names": (
                "base",
                "fused",
                "quality",
                "contrastive_base",
                "detector_countsplit",
            ),
            "source_pool_selector_override_default_source": (
                "detector_countsplit"
            ),
            "base_grounding_scores": torch.tensor([
                [0.90, 0.10, 0.00, 0.00]
            ]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "detector_countsplit_scores": torch.tensor([
                [0.05, 0.05, 0.95, 0.05]
            ]),
            "last_center": torch.tensor([[
                [5.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.00, 0.0, 0.0],
                [0.50, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_bce",
            pairwise_weight=0.0,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertAlmostEqual(
            losses[
                "dbg_source_pool_selector_target_detector_countsplit_ratio"
            ].item(),
            1.0,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_target_override_ratio"].item(),
            0.0,
        )
        self.assertGreater(override_logit.grad.item(), 0.0)

    def test_direct_choice_base_override_focal_bce_downweights_easy_binary_samples(self):
        selector_choice_scores = torch.zeros(2, 4, requires_grad=True)
        override_logit = torch.tensor([-4.0, 0.0], requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(2, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_override_logit": override_logit,
            "base_grounding_scores": torch.tensor([
                [0.90, 0.10, 0.00, 0.00],
                [0.90, 0.10, 0.00, 0.00],
            ]),
            "fused_scores": torch.tensor([
                [0.10, 0.90, 0.00, 0.00],
                [0.10, 0.90, 0.00, 0.00],
            ]),
            "pred_iou": torch.tensor([
                [0.00, 0.00, 0.90, 0.10],
                [0.00, 0.00, 0.90, 0.10],
            ]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90],
                [0.00, 0.00, 0.10, 0.90],
            ]),
            "last_center": torch.tensor([
                [
                    [0.50, 0.0, 0.0],
                    [5.00, 0.0, 0.0],
                    [0.50, 0.0, 0.0],
                    [5.00, 0.0, 0.0],
                ],
                [
                    [0.50, 0.0, 0.0],
                    [5.00, 0.0, 0.0],
                    [0.50, 0.0, 0.0],
                    [5.00, 0.0, 0.0],
                ],
            ]),
            "last_pred_size": torch.ones(2, 4, 3) * 2.0,
            "center_label": torch.tensor([
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
            ]),
            "size_gts": torch.tensor([
                [[2.0, 2.0, 2.0]],
                [[2.0, 2.0, 2.0]],
            ]),
            "box_label_mask": torch.tensor([[1], [1]]),
        }

        bce_losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_bce",
            pairwise_weight=0.0,
        )
        focal_losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_focal_bce",
            pairwise_weight=0.0,
        )

        self.assertLess(
            focal_losses["loss_source_pool_selector"].item(),
            bce_losses["loss_source_pool_selector"].item(),
        )

    def test_direct_choice_override_head_controls_training_selected_diagnostics(self):
        selector_choice_scores = torch.tensor(
            [[0.0, 3.0, 4.0, 2.0]], requires_grad=True
        )
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_override_logit": torch.tensor(
                [-1.0], requires_grad=True
            ),
            "base_grounding_scores": torch.tensor([[0.90, 0.10, 0.00, 0.00]]),
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.00]]),
            "pred_iou": torch.tensor([[0.00, 0.00, 0.90, 0.10]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90]
            ]),
            "last_center": torch.tensor([[
                [0.50, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.50, 0.0, 0.0],
                [5.00, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="base_override_bce",
            pairwise_weight=0.0,
        )

        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_selected_base_ratio"].item(),
            1.0,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_selected_override_ratio"].item(),
            0.0,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_false_override_ratio"].item(),
            0.0,
        )

    def test_direct_choice_threshold_bucket_reports_acceptable_and_selected_sources(self):
        selector_choice_scores = torch.tensor([
            [0.0, -2.0, 2.0, -2.0],
            [-2.0, 0.0, -2.0, 2.0],
        ], requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(2, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([
                [0.90, 0.10, 0.00, 0.00],
                [0.00, 0.10, 0.90, 0.00],
            ]),
            "fused_scores": torch.tensor([
                [0.10, 0.90, 0.00, 0.00],
                [0.00, 0.90, 0.10, 0.00],
            ]),
            "pred_iou": torch.tensor([
                [0.00, 0.00, 0.90, 0.10],
                [0.90, 0.00, 0.10, 0.00],
            ]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90],
                [0.00, 0.00, 0.10, 0.90],
            ]),
            "last_center": torch.tensor([
                [
                    [0.00, 0.0, 0.0],
                    [5.00, 0.0, 0.0],
                    [0.20, 0.0, 0.0],
                    [5.00, 0.0, 0.0],
                ],
                [
                    [5.00, 0.0, 0.0],
                    [0.20, 0.0, 0.0],
                    [5.00, 0.0, 0.0],
                    [0.20, 0.0, 0.0],
                ],
            ]),
            "last_pred_size": torch.ones(2, 4, 3) * 2.0,
            "center_label": torch.zeros(2, 1, 3),
            "size_gts": torch.ones(2, 1, 3) * 2.0,
            "box_label_mask": torch.ones(2, 1),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_bucket",
        )

        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_acceptable_base_ratio"].item(),
            0.5,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_acceptable_fused_ratio"].item(),
            0.5,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_acceptable_quality_ratio"].item(),
            0.5,
        )
        self.assertAlmostEqual(
            losses[
                "dbg_source_pool_selector_acceptable_contrastive_base_ratio"
            ].item(),
            0.5,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_selected_quality_ratio"].item(),
            0.5,
        )
        self.assertAlmostEqual(
            losses[
                "dbg_source_pool_selector_selected_contrastive_base_ratio"
            ].item(),
            0.5,
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_selected_acceptable_ratio"].item(),
            1.0,
        )

    def test_direct_choice_reports_per_source_logit_diagnostics(self):
        selector_choice_scores = torch.tensor([
            [4.0, 0.0, -4.0, 2.0],
            [-2.0, 0.0, 5.0, -1.0],
        ], requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(2, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([
                [0.90, 0.10, 0.00, 0.00],
                [0.00, 0.10, 0.90, 0.00],
            ]),
            "fused_scores": torch.tensor([
                [0.10, 0.90, 0.00, 0.00],
                [0.00, 0.90, 0.10, 0.00],
            ]),
            "pred_iou": torch.tensor([
                [0.00, 0.00, 0.90, 0.10],
                [0.90, 0.00, 0.10, 0.00],
            ]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.00, 0.10, 0.90],
                [0.00, 0.00, 0.10, 0.90],
            ]),
            "last_center": torch.tensor([
                [
                    [0.00, 0.0, 0.0],
                    [5.00, 0.0, 0.0],
                    [0.20, 0.0, 0.0],
                    [5.00, 0.0, 0.0],
                ],
                [
                    [5.00, 0.0, 0.0],
                    [0.20, 0.0, 0.0],
                    [5.00, 0.0, 0.0],
                    [0.20, 0.0, 0.0],
                ],
            ]),
            "last_pred_size": torch.ones(2, 4, 3) * 2.0,
            "center_label": torch.zeros(2, 1, 3),
            "size_gts": torch.ones(2, 1, 3) * 2.0,
            "box_label_mask": torch.ones(2, 1),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.02,
            choice_target="threshold_bucket",
        )

        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_logit_base_mean"].item(), 1.0
        )
        self.assertAlmostEqual(
            losses["dbg_source_pool_selector_logit_quality_mean"].item(), 0.5
        )
        self.assertAlmostEqual(
            losses[
                "dbg_source_pool_selector_logit_margin_base_mean"
            ].item(),
            -2.5,
        )
        self.assertAlmostEqual(
            losses[
                "dbg_source_pool_selector_logit_margin_quality_mean"
            ].item(),
            -1.5,
        )

    def test_direct_choice_quality_override_defaults_to_quality_on_small_gain(self):
        selector_choice_scores = torch.zeros(1, 3, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.9, 0.1, 0.0, 0.0]]),
            "fused_scores": torch.tensor([[0.0, 0.1, 0.0, 0.9]]),
            "pred_iou": torch.tensor([[0.0, 0.1, 0.9, 0.0]]),
            "last_center": torch.tensor([[
                [0.740, 0.0, 0.0],  # slightly better than quality
                [5.000, 0.0, 0.0],
                [0.780, 0.0, 0.0],  # quality top, same threshold bucket
                [5.000, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.05,
            choice_target="quality_override",
            pairwise_weight=0.0,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertLess(selector_choice_scores.grad[0, 2].item(), 0.0)
        self.assertAlmostEqual(
            losses[
                "dbg_source_pool_selector_choice_target_quality_override"
            ],
            1.0,
        )

    def test_direct_choice_quality_override_uses_nonquality_cross_threshold_gain(self):
        selector_choice_scores = torch.zeros(1, 3, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.9, 0.1, 0.0, 0.0]]),
            "fused_scores": torch.tensor([[0.0, 0.1, 0.0, 0.9]]),
            "pred_iou": torch.tensor([[0.0, 0.1, 0.9, 0.0]]),
            "last_center": torch.tensor([[
                [0.665, 0.0, 0.0],  # base crosses IoU@0.50
                [5.000, 0.0, 0.0],
                [0.685, 0.0, 0.0],  # quality stays below IoU@0.50
                [5.000, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.05,
            choice_target="quality_override",
            pairwise_weight=0.0,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertLess(selector_choice_scores.grad[0, 0].item(), 0.0)
        self.assertGreater(selector_choice_scores.grad[0, 2].item(), 0.0)

    def test_direct_choice_quality_override_uses_separate_override_logit(self):
        selector_choice_scores = torch.zeros(1, 3, requires_grad=True)
        override_logit = torch.tensor([1.0], requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_override_logit": override_logit,
            "base_grounding_scores": torch.tensor([[0.9, 0.1, 0.0, 0.0]]),
            "fused_scores": torch.tensor([[0.0, 0.1, 0.0, 0.9]]),
            "pred_iou": torch.tensor([[0.0, 0.1, 0.9, 0.0]]),
            "last_center": torch.tensor([[
                [0.740, 0.0, 0.0],
                [5.000, 0.0, 0.0],
                [0.780, 0.0, 0.0],
                [5.000, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.05,
            choice_target="quality_override",
            pairwise_weight=0.0,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertIsNotNone(override_logit.grad)
        self.assertGreater(override_logit.grad.item(), 0.0)
        self.assertEqual(selector_choice_scores.grad.abs().sum().item(), 0.0)
        self.assertEqual(
            losses["dbg_source_pool_selector_choice_override_head"],
            1.0,
        )

    def test_direct_choice_quality_override_margin_pushes_override_logit(self):
        selector_choice_scores = torch.zeros(1, 3, requires_grad=True)
        override_logit = torch.tensor([-0.25], requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "selector_choice_override_logit": override_logit,
            "base_grounding_scores": torch.tensor([[0.9, 0.1, 0.0, 0.0]]),
            "fused_scores": torch.tensor([[0.0, 0.1, 0.0, 0.9]]),
            "pred_iou": torch.tensor([[0.0, 0.1, 0.9, 0.0]]),
            "last_center": torch.tensor([[
                [0.665, 0.0, 0.0],  # base crosses IoU@0.50
                [5.000, 0.0, 0.0],
                [0.685, 0.0, 0.0],  # quality stays below IoU@0.50
                [5.000, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.05,
            choice_target="quality_override",
            pairwise_weight=0.0,
            override_margin_weight=1.0,
            override_margin=0.75,
        )
        losses["loss_source_pool_selector"].backward()

        self.assertGreater(
            losses["dbg_source_pool_selector_override_margin_loss"].item(),
            0.0,
        )
        self.assertLess(override_logit.grad.item(), 0.0)
        self.assertEqual(
            losses["dbg_source_pool_selector_override_margin_weight"],
            1.0,
        )

    def test_direct_choice_loss_uses_pairwise_weight(self):
        selector_choice_scores = torch.zeros(1, 3, requires_grad=True)
        end_points = {
            "selector_scores": torch.zeros(1, 4, requires_grad=True),
            "selector_choice_scores": selector_choice_scores,
            "base_grounding_scores": torch.tensor([[0.9, 0.1, 0.0, 0.0]]),
            "fused_scores": torch.tensor([[0.0, 0.1, 0.0, 0.9]]),
            "pred_iou": torch.tensor([[0.0, 0.1, 0.2, 0.0]]),
            "last_center": torch.tensor([[
                [0.685, 0.0, 0.0],
                [5.000, 0.0, 0.0],
                [5.000, 0.0, 0.0],
                [0.665, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        no_pairwise = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.05,
            pairwise_weight=0.0,
            choice_target="threshold_utility",
        )
        with_pairwise = _source_pool_selector_losses(
            end_points,
            weight=1.0,
            source="source_choice",
            candidate_k=1,
            temperature=1.0,
            min_iou_gap=0.05,
            pairwise_weight=1.0,
            choice_target="threshold_utility",
        )

        self.assertGreater(
            with_pairwise["dbg_source_pool_selector_pairwise_loss"].item(),
            0.0,
        )
        self.assertGreater(
            with_pairwise["loss_source_pool_selector_raw"].item(),
            no_pairwise["loss_source_pool_selector_raw"].item(),
        )
        self.assertEqual(
            with_pairwise["dbg_source_pool_selector_pairwise_weight"],
            1.0,
        )

    def test_parser_accepts_source_pool_selector_options(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_loss_weight",
            "0.2",
            "--source_pool_selector_source",
            "source_pool",
            "--source_pool_selector_k",
            "5",
            "--source_pool_selector_temperature",
            "0.7",
            "--source_pool_selector_min_iou_gap",
            "0.03",
            "--source_pool_selector_source_min_iou_gaps",
            "quality:0.02,detector_jointtight:0.10",
            "--source_pool_selector_train_only",
            "--source_pool_selector_lr",
            "0.003",
            "--source_pool_selector_candidate_aware",
            "--source_pool_selector_pairdelta_features",
            "--source_pool_selector_pairwise_weight",
            "1.25",
            "--source_pool_selector_choice_balance",
            "--source_pool_selector_choice_balance_power",
            "0.5",
            "--source_pool_selector_choice_target",
            "base_override_bce",
            "--source_pool_selector_direct_choice",
            "--source_pool_selector_include_contrastive_choice",
            "--source_pool_selector_rank_features",
            "--source_pool_selector_separate_override_head",
            "--source_pool_selector_override_initial_bias",
            "-1.5",
            "--eval_selector_choice_use_override_head",
            "--eval_selector_choice_source_bias_base",
            "0.25",
            "--eval_selector_choice_source_bias_quality",
            "-0.15",
            "--eval_selector_choice_source_bias_detector_countboost",
            "0.75",
            "--eval_selector_choice_override_threshold",
            "-0.5",
            "--eval_use_selector_choice_scores",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertTrue(args.use_source_pool_selector)
        self.assertEqual(args.source_pool_selector_loss_weight, 0.2)
        self.assertEqual(args.source_pool_selector_source, "source_pool")
        self.assertEqual(args.source_pool_selector_k, 5)
        self.assertEqual(args.source_pool_selector_temperature, 0.7)
        self.assertEqual(args.source_pool_selector_min_iou_gap, 0.03)
        self.assertEqual(
            args.source_pool_selector_source_min_iou_gaps,
            {
                "quality": 0.02,
                "detector_jointtight": 0.10,
            },
        )
        self.assertTrue(args.source_pool_selector_train_only)
        self.assertEqual(args.source_pool_selector_lr, 0.003)
        self.assertTrue(args.source_pool_selector_candidate_aware)
        self.assertTrue(args.source_pool_selector_pairdelta_features)
        self.assertEqual(args.source_pool_selector_pairwise_weight, 1.25)
        self.assertTrue(args.source_pool_selector_choice_balance)
        self.assertEqual(args.source_pool_selector_choice_balance_power, 0.5)
        self.assertEqual(
            args.source_pool_selector_choice_target,
            "base_override_bce",
        )
        self.assertTrue(args.source_pool_selector_direct_choice)
        self.assertTrue(args.source_pool_selector_include_contrastive_choice)
        self.assertTrue(args.source_pool_selector_rank_features)
        self.assertTrue(args.source_pool_selector_separate_override_head)
        self.assertEqual(args.source_pool_selector_override_initial_bias, -1.5)
        self.assertTrue(args.eval_selector_choice_use_override_head)
        self.assertEqual(args.eval_selector_choice_source_bias_base, 0.25)
        self.assertEqual(args.eval_selector_choice_source_bias_quality, -0.15)
        self.assertEqual(
            args.eval_selector_choice_source_bias_detector_countboost,
            0.75,
        )
        self.assertEqual(args.eval_selector_choice_override_threshold, -0.5)
        self.assertTrue(args.eval_use_selector_choice_scores)

    def test_parser_accepts_source_pool_lse_choice_target(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_choice_target",
            "source_pool_lse",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_choice_target,
            "source_pool_lse",
        )

    def test_parser_accepts_threshold_utility_softmax_choice_target(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_choice_target",
            "threshold_utility_softmax",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_choice_target,
            "threshold_utility_softmax",
        )

    def test_parser_accepts_threshold_utility_regression_choice_target(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_choice_target",
            "threshold_utility_regression",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_choice_target,
            "threshold_utility_regression",
        )

    def test_parser_accepts_threshold_utility_hard_choice_target(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_choice_target",
            "threshold_utility_hard",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_choice_target,
            "threshold_utility_hard",
        )

    def test_parser_accepts_base_override_focal_bce_choice_target(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_choice_target",
            "base_override_focal_bce",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_choice_target,
            "base_override_focal_bce",
        )

    def test_parser_accepts_base_override_sourcewise_focal_bce_choice_target(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_choice_target",
            "base_override_sourcewise_focal_bce",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_choice_target,
            "base_override_sourcewise_focal_bce",
        )

    def test_parser_accepts_threshold_gain_default_sourcewise_focal_bce_choice_target(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_choice_target",
            "threshold_gain_default_sourcewise_focal_bce",
            "--source_pool_selector_sourcewise_negative_weight",
            "0.5",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_choice_target,
            "threshold_gain_default_sourcewise_focal_bce",
        )
        self.assertEqual(
            args.source_pool_selector_sourcewise_negative_weight,
            0.5,
        )

    def test_parser_accepts_threshold_gain_default_diffquery_sourcewise_focal_bce_choice_target(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_choice_target",
            "threshold_gain_default_diffquery_sourcewise_focal_bce",
            "--source_pool_selector_sourcewise_negative_weight",
            "0.5",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_choice_target,
            "threshold_gain_default_diffquery_sourcewise_focal_bce",
        )
        self.assertEqual(
            args.source_pool_selector_sourcewise_negative_weight,
            0.5,
        )

    def test_parser_accepts_precision_gain_default_sourcewise_focal_bce_choice_target(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_choice_target",
            "precision_gain_default_sourcewise_focal_bce",
            "--source_pool_selector_sourcewise_negative_weight",
            "0.5",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_choice_target,
            "precision_gain_default_sourcewise_focal_bce",
        )
        self.assertEqual(
            args.source_pool_selector_sourcewise_negative_weight,
            0.5,
        )

    def test_parser_accepts_source_choice_asymmetric_override_weights(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_choice_target",
            "base_override_focal_bce",
            "--source_pool_selector_false_base_weight",
            "1.75",
            "--source_pool_selector_false_override_weight",
            "1.25",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(args.source_pool_selector_false_base_weight, 1.75)
        self.assertEqual(args.source_pool_selector_false_override_weight, 1.25)

    def test_parser_accepts_source_choice_sourcewise_negative_weight(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_choice_target",
            "base_override_sourcewise_focal_bce",
            "--source_pool_selector_sourcewise_negative_weight",
            "0.25",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_sourcewise_negative_weight,
            0.25,
        )

    def test_parser_accepts_source_choice_override_margin_options(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_loss_weight",
            "1.0",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_candidate_aware",
            "--source_pool_selector_choice_target",
            "quality_override",
            "--source_pool_selector_override_margin_weight",
            "2.5",
            "--source_pool_selector_override_margin",
            "0.75",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(args.source_pool_selector_source, "source_choice")
        self.assertEqual(args.source_pool_selector_choice_target, "quality_override")
        self.assertEqual(args.source_pool_selector_override_margin_weight, 2.5)
        self.assertEqual(args.source_pool_selector_override_margin, 0.75)

    def test_parser_accepts_override_default_source_options(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_loss_weight",
            "1.0",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_direct_choice",
            "--source_pool_selector_include_detector_policy_choice",
            "--source_pool_selector_choice_target",
            "base_override_bce",
            "--source_pool_selector_override_default_source",
            "detector_countsplit",
            "--eval_selector_choice_use_override_head",
            "--eval_selector_choice_override_default_source",
            "detector_countsplit",
            "--eval_use_selector_choice_scores",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_override_default_source,
            "detector_countsplit",
        )
        self.assertEqual(
            args.eval_selector_choice_override_default_source,
            "detector_countsplit",
        )

    def test_parser_accepts_candidate_aware_rank_features(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_candidate_aware",
            "--source_pool_selector_rank_features",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertTrue(args.source_pool_selector_candidate_aware)
        self.assertTrue(args.source_pool_selector_rank_features)

    def test_parser_accepts_candidate_aware_context_features(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_candidate_aware",
            "--source_pool_selector_candidate_context",
            "--source_pool_selector_candidate_context_k",
            "3",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertTrue(args.source_pool_selector_candidate_aware)
        self.assertTrue(args.source_pool_selector_candidate_context)
        self.assertEqual(args.source_pool_selector_candidate_context_k, 3)

    def test_parser_accepts_source_pool_selector_text_context(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_text_context",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertTrue(args.source_pool_selector_text_context)

    def test_parser_accepts_source_pool_selector_context_features(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_text_context",
            "--source_pool_selector_context_features",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertTrue(args.source_pool_selector_text_context)
        self.assertTrue(args.source_pool_selector_context_features)

    def test_train_get_model_forwards_source_pool_selector_context_features(self):
        captured = {}

        class DummyModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        args = argparse.Namespace(
            use_color=False,
            use_height=False,
            use_multiview=False,
            use_soft_token_loss=False,
            num_target=1,
            num_decoder_layers=1,
            self_position_embedding="fourier",
            use_contrastive_align=False,
            butd=False,
            butd_gt=False,
            butd_cls=False,
            pp_checkpoint=None,
            self_attend=False,
            use_structured_slots=False,
            use_late_acd=False,
            slot_pooling="attention",
            max_rel_anchor_pairs=3,
            acd_top_m_targets=32,
            acd_top_k_anchors=16,
            acd_geo_dim=16,
            acd_hidden_dim=288,
            acd_global_residual_alpha=0.5,
            acd_use_confidence_fusion=False,
            acd_warmup_steps=5000,
            acd_initial_alpha=0.05,
            acd_ea_scale=1.0,
            acd_pool_ea_multiplier=1.0,
            acd_final_ea_multiplier=1.0,
            acd_disable_struct_rerank=False,
            acd_base_score_source="contrastive",
            dhc_margin_min=0.0,
            dhc_temperature_max=0.0,
            structured_debug=False,
            use_quality_head=False,
            use_sacr=False,
            sacr_top_m_targets=32,
            sacr_top_k_anchors=16,
            sacr_hidden_dim=288,
            sacr_geo_dim=16,
            sacr_disable_relation=False,
            use_rapf=False,
            rapf_hidden_dim=128,
            rapf_initial_gate_bias=-2.0,
            rapf_use_quality=False,
            rapf_quality_weight=0.25,
            rapf_struct_residual_clip=2.0,
            rapf_generic_gate_cap=0.35,
            rapf_quality_anchor_structured_residual=False,
            use_qahnl=False,
            qahnl_score_source="fused",
            use_source_pool_selector=True,
            source_pool_selector_hidden_dim=16,
            source_pool_selector_candidate_aware=True,
            source_pool_selector_direct_choice=False,
            source_pool_selector_include_contrastive_choice=False,
            source_pool_selector_rank_features=False,
            source_pool_selector_pairdelta_features=False,
            source_pool_selector_candidate_context=False,
            source_pool_selector_candidate_context_k=5,
            source_pool_selector_text_context=True,
            source_pool_selector_metadata_context=False,
            source_pool_selector_context_features=True,
            source_pool_selector_separate_override_head=False,
            source_pool_selector_override_initial_bias=-1.5,
        )

        with mock.patch.object(train_dist_mod, "BeaUTyDETR", DummyModel):
            model = train_dist_mod.TrainTester.get_model(args)

        self.assertIsInstance(model, DummyModel)
        self.assertTrue(captured["source_pool_selector_text_context"])
        self.assertTrue(captured["source_pool_selector_context_features"])

    def test_parser_rejects_context_features_without_selector(self):
        argv = [
            "train_dist_mod.py",
            "--source_pool_selector_context_features",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(
                ValueError,
                "--source_pool_selector_context_features requires "
                "--use_source_pool_selector",
            ):
                parse_option()

    def test_parser_rejects_context_features_without_context_source(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_context_features",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(
                ValueError,
                "--source_pool_selector_context_features requires "
                "--source_pool_selector_text_context or "
                "--source_pool_selector_metadata_context",
            ):
                parse_option()

    def test_parser_accepts_source_pool_selector_metadata_context(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_metadata_context",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertTrue(args.source_pool_selector_metadata_context)

    def test_selector_metadata_context_builds_parse_coverage_features(self):
        inputs = {
            "coverage_stats": [{
                "has_target": 1,
                "num_attrs": 2,
                "num_pairs": 1,
                "num_parse_errors": 3,
                "global_only_due_to_parse_error": 0,
                "target_generic_reference": 1,
            }],
            "parse_confidence": [0.65],
            "decomposition_error_flags_count": [2],
        }
        slot_dict = {
            "coverage_stats": {
                "has_target": torch.tensor([True]),
                "num_attrs": torch.tensor([2]),
                "num_pairs": torch.tensor([1]),
                "num_parse_errors": torch.tensor([3]),
                "global_only_due_to_parse_error": torch.tensor([0.0]),
                "target_generic_reference": torch.tensor([1.0]),
            },
            "parse_confidence": torch.tensor([0.65]),
        }

        context = BeaUTyDETR._build_selector_metadata_context(
            inputs,
            slot_dict,
            batch_size=1,
            device=torch.device("cpu"),
        )

        self.assertEqual(context.shape, (1, 8))
        expected = torch.tensor([[
            0.65, 1.0, 2.0, 1.0, 3.0, 0.0, 1.0, 2.0
        ]])
        self.assertTrue(torch.allclose(context, expected))

    def test_parser_accepts_source_choice_feature_dump_path(self):
        argv = [
            "train_dist_mod.py",
            "--eval",
            "--eval_dump_source_choice_features_path",
            "/tmp/source_choice.pt",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.eval_dump_source_choice_features_path,
            "/tmp/source_choice.pt",
        )

    def test_parser_accepts_source_choice_feature_dump_topk(self):
        argv = [
            "train_dist_mod.py",
            "--eval",
            "--eval_dump_source_choice_features_path",
            "/tmp/source_choice.pt",
            "--eval_dump_source_choice_topk",
            "5",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(args.eval_dump_source_choice_topk, 5)

    def test_parser_accepts_selector_choice_hybrid_eval(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_direct_choice",
            "--eval_use_quality_scores",
            "--eval_use_selector_choice_hybrid_scores",
            "--eval_selector_choice_min_margin",
            "0.25",
            "--eval_selector_choice_hybrid_fallback",
            "quality",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertTrue(args.eval_use_selector_choice_hybrid_scores)
        self.assertFalse(args.eval_use_quality_scores)
        self.assertEqual(args.eval_selector_choice_min_margin, 0.25)
        self.assertEqual(args.eval_selector_choice_hybrid_fallback, "quality")

    def test_parser_accepts_source_choice_oracle_prior_weight(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_loss_weight",
            "1.0",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_direct_choice",
            "--source_pool_selector_oracle_prior_weight",
            "0.25",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(args.source_pool_selector_oracle_prior_weight, 0.25)

    def test_parser_accepts_source_choice_override_prior_weight(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_loss_weight",
            "1.0",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_direct_choice",
            "--source_pool_selector_separate_override_head",
            "--source_pool_selector_choice_target",
            "quality_override",
            "--source_pool_selector_override_prior_weight",
            "0.25",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(args.source_pool_selector_override_prior_weight, 0.25)

    def test_parser_accepts_source_choice_override_source_prior_weight(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_loss_weight",
            "1.0",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_direct_choice",
            "--source_pool_selector_choice_target",
            "quality_override",
            "--source_pool_selector_override_source_prior_weight",
            "0.5",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_override_source_prior_weight,
            0.5,
        )

    def test_parser_accepts_base_override_source_prior_weight(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_loss_weight",
            "1.0",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_direct_choice",
            "--source_pool_selector_separate_override_head",
            "--source_pool_selector_choice_target",
            "base_override_focal_bce",
            "--source_pool_selector_override_source_prior_weight",
            "0.5",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(
            args.source_pool_selector_override_source_prior_weight,
            0.5,
        )

    def test_parser_accepts_selector_choice_quality_override_eval(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_direct_choice",
            "--eval_use_quality_scores",
            "--eval_use_selector_choice_quality_override_scores",
            "--eval_selector_choice_min_margin",
            "0.15",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertTrue(args.eval_use_selector_choice_quality_override_scores)
        self.assertFalse(args.eval_use_quality_scores)
        self.assertEqual(args.eval_selector_choice_min_margin, 0.15)

    def test_eval_branch_forwards_selector_choice_quality_override_flag(self):
        class DummyModel(nn.Module):
            def forward(self, inputs):
                self.seen_train_flag = inputs["train"]
                return {}

        def criterion(end_points, *args, **kwargs):
            return torch.tensor(0.0), end_points

        tester = BaseTrainTester.__new__(BaseTrainTester)
        tester._to_gpu = lambda data: data
        model = DummyModel()
        args = argparse.Namespace(
            num_decoder_layers=6,
            query_points_obj_topk=4,
            eval_use_selector_choice_quality_override_scores=True,
            eval_selector_choice_min_margin=0.15,
            print_freq=99,
        )
        batch_data = {
            "point_clouds": torch.zeros(1, 2, 3),
            "utterances": ["chair"],
        }

        _, end_points = tester._main_eval_branch(
            batch_idx=0,
            batch_data=batch_data,
            test_loader=[batch_data],
            model=model,
            stat_dict={},
            criterion=criterion,
            set_criterion=None,
            args=args,
        )

        self.assertFalse(model.seen_train_flag)
        self.assertTrue(
            end_points["eval_use_selector_choice_quality_override_scores"]
        )
        self.assertEqual(end_points["eval_selector_choice_min_margin"], 0.15)

    def test_eval_branch_forwards_explicit_primary_score_source(self):
        class DummyModel(nn.Module):
            def forward(self, inputs):
                self.seen_train_flag = inputs["train"]
                self.seen_eval_target_cid_source = inputs.get(
                    "eval_target_cid_source"
                )
                return {}

        def criterion(end_points, *args, **kwargs):
            return torch.tensor(0.0), end_points

        tester = BaseTrainTester.__new__(BaseTrainTester)
        tester._to_gpu = lambda data: data
        model = DummyModel()
        args = argparse.Namespace(
            num_decoder_layers=6,
            query_points_obj_topk=4,
            eval_primary_score_source="detector_countboost",
            eval_target_cid_source="text",
            eval_use_quality_scores=True,
            print_freq=99,
        )
        batch_data = {
            "point_clouds": torch.zeros(1, 2, 3),
            "utterances": ["chair"],
        }

        _, end_points = tester._main_eval_branch(
            batch_idx=0,
            batch_data=batch_data,
            test_loader=[batch_data],
            model=model,
            stat_dict={},
            criterion=criterion,
            set_criterion=None,
            args=args,
        )

        self.assertFalse(model.seen_train_flag)
        self.assertEqual(
            end_points["eval_primary_score_source"],
            "detector_countboost",
        )
        self.assertEqual(model.seen_eval_target_cid_source, "text")
        self.assertFalse(end_points["eval_use_quality_scores"])

    def test_core_diagnostics_include_source_choice_oracle_prior_metrics(self):
        self.assertTrue(
            BaseTrainTester._is_core_diagnostic_key(
                "dbg_source_pool_selector_oracle_prior_loss"
            )
        )
        self.assertTrue(
            BaseTrainTester._is_core_diagnostic_key(
                "dbg_source_pool_selector_target_base_ratio"
            )
        )
        self.assertTrue(
            BaseTrainTester._is_core_diagnostic_key(
                "dbg_source_pool_selector_selected_quality_ratio"
            )
        )

    def test_get_loaders_can_skip_train_dataset_in_eval_only_mode(self):
        tester = BaseTrainTester.__new__(BaseTrainTester)
        tester.get_datasets = mock.Mock(
            return_value=(None, mock.sentinel.test_dataset)
        )
        args = argparse.Namespace(
            dataset=["scanrefer_spacy"],
            joint_det=False,
            test_dataset="scanrefer_spacy",
            debug=False,
            use_color=False,
            use_height=False,
            use_multiview=False,
            butd=False,
            butd_gt=False,
            butd_cls=False,
            augment_det=False,
            num_workers=0,
            batch_size=2,
        )

        def _fake_loader(dataset, **kwargs):
            loader = mock.Mock()
            loader.dataset = dataset
            loader.sampler = mock.Mock()
            return loader

        with mock.patch("main_utils.DataLoader", side_effect=_fake_loader):
            with mock.patch("main_utils.DistributedSampler") as sampler_mock:
                train_loader, test_loader = tester.get_loaders(
                    args, include_train=False
                )

        self.assertIsNone(train_loader)
        self.assertIs(test_loader.dataset, mock.sentinel.test_dataset)
        tester.get_datasets.assert_called_once_with(
            args, include_train=False
        )
        sampler_mock.assert_called_once_with(
            mock.sentinel.test_dataset, shuffle=False
        )

    def test_selector_eval_flag_overrides_quality_primary_default(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--eval_use_quality_scores",
            "--eval_use_selector_scores",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertTrue(args.eval_use_selector_scores)
        self.assertFalse(args.eval_use_quality_scores)

    def test_parser_accepts_selector_pool_primary_eval(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--eval_use_quality_scores",
            "--eval_use_selector_pool_scores",
            "--eval_selector_pool_k",
            "1",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertTrue(args.eval_use_selector_pool_scores)
        self.assertFalse(args.eval_use_quality_scores)
        self.assertEqual(args.eval_selector_pool_k, 1)

    def test_parser_allows_contrastive_candidate_aware_selector_pool(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_loss_weight",
            "1.0",
            "--source_pool_selector_train_only",
            "--source_pool_selector_candidate_aware",
            "--source_pool_selector_include_contrastive_choice",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertTrue(args.source_pool_selector_candidate_aware)
        self.assertTrue(args.source_pool_selector_include_contrastive_choice)
        self.assertFalse(args.source_pool_selector_direct_choice)

    def test_parser_allows_candidate_aware_source_choice(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_loss_weight",
            "1.0",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_train_only",
            "--source_pool_selector_candidate_aware",
            "--source_pool_selector_include_contrastive_choice",
            "--eval_use_selector_choice_quality_override_scores",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(args.source_pool_selector_source, "source_choice")
        self.assertTrue(args.source_pool_selector_candidate_aware)
        self.assertTrue(args.source_pool_selector_include_contrastive_choice)
        self.assertFalse(args.source_pool_selector_direct_choice)
        self.assertTrue(args.eval_use_selector_choice_quality_override_scores)

    def test_parser_allows_candidate_aware_override_head_eval(self):
        argv = [
            "train_dist_mod.py",
            "--use_source_pool_selector",
            "--source_pool_selector_loss_weight",
            "1.0",
            "--source_pool_selector_source",
            "source_choice",
            "--source_pool_selector_train_only",
            "--source_pool_selector_candidate_aware",
            "--source_pool_selector_include_contrastive_choice",
            "--source_pool_selector_separate_override_head",
            "--source_pool_selector_choice_target",
            "base_override_bce",
            "--eval_selector_choice_use_override_head",
            "--eval_use_selector_choice_scores",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertTrue(args.source_pool_selector_candidate_aware)
        self.assertTrue(args.source_pool_selector_separate_override_head)
        self.assertTrue(args.eval_selector_choice_use_override_head)
        self.assertFalse(args.source_pool_selector_direct_choice)

    def test_dataloader_parallelism_kwargs_skip_prefetch_when_single_worker(self):
        self.assertEqual(
            _dataloader_parallelism_kwargs(0),
            {"persistent_workers": False},
        )
        self.assertEqual(
            _dataloader_parallelism_kwargs(2),
            {"persistent_workers": True, "prefetch_factor": 2},
        )

    def test_evaluator_can_use_selector_as_primary_score_source(self):
        end_points = {
            "selector_scores": torch.tensor([[0.20, 0.90, 0.10]]),
        }

        scores = GroundingEvaluator._single_source_scores(
            end_points,
            "selector",
            bid=0,
            num_obj=1,
            base_scores=None,
        )

        self.assertTrue(
            torch.allclose(scores, torch.tensor([[0.20, 0.90, 0.10]]))
        )

    def test_selector_pool_scores_mask_to_source_topk_candidates(self):
        source_scores = {
            "base": torch.tensor([[0.90, 0.10, 0.20, 0.00]]),
            "quality": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
        }
        selector_scores = torch.tensor([[0.00, 0.20, 10.00, 0.10]])

        masked_scores = GroundingEvaluator._selector_pool_scores(
            source_scores,
            selector_scores,
            candidate_k=1,
        )

        self.assertEqual(masked_scores[0, 0].item(), 0.0)
        self.assertAlmostEqual(masked_scores[0, 1].item(), 0.2)
        self.assertLess(masked_scores[0, 2].item(), -1e20)

    def test_selector_pool_diagnostics_report_selected_vs_oracle_query(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "eval_selector_pool_k": 1,
            "selector_scores": torch.zeros(1, 4),
            "selector_source_scores": torch.tensor([[
                [0.10, 0.00, 0.00, 0.00],
                [0.00, 0.90, 0.00, 0.00],
                [0.00, 0.00, 0.20, 0.00],
            ]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.20, 0.95, 0.00]]),
        }
        pred_bbox = torch.tensor([
            [0.80, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])

        diagnostics = GroundingEvaluator._selector_pool_diagnostics(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
        )

        self.assertEqual(diagnostics["selected_query_id"].item(), 1.0)
        self.assertEqual(diagnostics["oracle_query_id"].item(), 2.0)
        self.assertEqual(diagnostics["selector_pool_oracle_agree"].item(), 0.0)
        self.assertEqual(diagnostics["selector_pool_selected_hit050"].item(), 0.0)
        self.assertEqual(diagnostics["selector_pool_oracle_hit050"].item(), 1.0)
        self.assertGreater(
            diagnostics["selector_pool_iou_gap_to_oracle"].item(),
            0.9,
        )

    def test_selector_pool_diagnostics_include_contrastive_base_source(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "eval_selector_pool_k": 1,
            "selector_source_scores": torch.tensor([[
                [0.10, 0.00, 0.00, 0.00],  # base source
                [0.00, 0.20, 0.00, 0.00],  # fused source
                [0.00, 0.00, 0.30, 0.00],  # quality source
                [0.00, 0.00, 0.00, 0.95],  # contrastive_base source
            ]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.20, 0.70, 0.00]]),
            "bbf_base_grounding_scores": torch.tensor([[
                0.00, 0.10, 0.20, 0.99
            ]]),
        }
        pred_bbox = torch.tensor([
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.00, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])

        diagnostics = GroundingEvaluator._selector_pool_diagnostics(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
        )

        self.assertEqual(diagnostics["selected_query_id"].item(), 3.0)
        self.assertEqual(diagnostics["selector_pool_selected_hit050"].item(), 1.0)

    def test_selector_pool_primary_uses_training_source_pool_sources(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "eval_selector_pool_k": 1,
            "selector_scores": torch.tensor([[0.00, 0.20, 10.00, 0.10]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.20, 0.00]]),
            "structured_scores": torch.tensor([[0.00, 0.10, 0.99, 0.00]]),
        }

        scores = GroundingEvaluator._selector_pool_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )

        self.assertEqual(scores[0, 0].item(), 0.0)
        self.assertAlmostEqual(scores[0, 1].item(), 0.2)
        self.assertLess(scores[0, 2].item(), -1e20)

    def test_selector_pool_primary_uses_source_aware_candidate_logits(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "eval_selector_pool_k": 1,
            "selector_scores": torch.tensor([[0.00, 10.00, 0.00, 0.00]]),
            "selector_source_scores": torch.tensor([[
                [0.10, 0.00, 0.00, 0.00],  # base source
                [0.00, 0.90, 0.00, 0.00],  # fused source
                [0.00, 0.20, 0.00, 0.00],  # quality source
            ]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.20, 0.00]]),
        }

        scores = GroundingEvaluator._selector_pool_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )

        self.assertAlmostEqual(scores[0, 0].item(), 0.1)
        self.assertAlmostEqual(scores[0, 1].item(), 0.9)
        self.assertLess(scores[0, 2].item(), -1e20)

    def test_selector_pool_primary_uses_contrastive_base_source_logits(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "eval_selector_pool_k": 1,
            "selector_source_scores": torch.tensor([[
                [0.10, 0.00, 0.00, 0.00],  # base source
                [0.00, 0.20, 0.00, 0.00],  # fused source
                [0.00, 0.00, 0.30, 0.00],  # quality source
                [0.00, 0.00, 0.00, 0.95],  # contrastive_base source
            ]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.20, 0.70, 0.00]]),
            "bbf_base_grounding_scores": torch.tensor([[
                0.00, 0.10, 0.20, 0.99
            ]]),
        }

        scores = GroundingEvaluator._selector_pool_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )

        self.assertAlmostEqual(scores[0, 3].item(), 0.95)
        self.assertEqual(int(scores[0].argmax().item()), 3)

    def test_selector_pool_quality_override_defaults_to_quality_when_margin_low(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "eval_selector_pool_k": 1,
            "selector_scores": torch.tensor([[0.00, 0.72, 0.00, 0.00]]),
            "selector_source_scores": torch.tensor([[
                [0.10, 0.00, 0.00, 0.00],  # base source
                [0.00, 0.72, 0.00, 0.00],  # fused source
                [0.00, 0.00, 0.70, 0.00],  # quality source
            ]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
            "eval_selector_pool_min_margin": 0.05,
        }

        scores = (
            GroundingEvaluator
            ._selector_pool_quality_override_primary_scores(
                end_points,
                bid=0,
                num_obj=1,
                base_scores=base_scores,
            )
        )

        self.assertEqual(int(scores[0].argmax().item()), 2)
        self.assertAlmostEqual(scores[0, 2].item(), 0.95)

    def test_selector_pool_quality_override_uses_pool_above_quality_margin(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "eval_selector_pool_k": 1,
            "selector_scores": torch.tensor([[0.00, 0.80, 0.00, 0.00]]),
            "selector_source_scores": torch.tensor([[
                [0.10, 0.00, 0.00, 0.00],  # base source
                [0.00, 0.80, 0.00, 0.00],  # fused source
                [0.00, 0.00, 0.70, 0.00],  # quality source
            ]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
            "eval_selector_pool_min_margin": 0.05,
        }

        scores = (
            GroundingEvaluator
            ._selector_pool_quality_override_primary_scores(
                end_points,
                bid=0,
                num_obj=1,
                base_scores=base_scores,
            )
        )

        self.assertEqual(int(scores[0].argmax().item()), 1)
        self.assertAlmostEqual(scores[0, 1].item(), 0.80)

    def test_selector_pool_quality_override_diagnostic_sweeps_margins(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "eval_selector_pool_k": 1,
            "selector_scores": torch.tensor([[0.00, 0.80, 0.00, 0.00]]),
            "selector_source_scores": torch.tensor([[
                [0.10, 0.00, 0.00, 0.00],  # base source
                [0.00, 0.80, 0.00, 0.00],  # fused source
                [0.00, 0.00, 0.70, 0.00],  # quality source
            ]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
        }

        diagnostics = (
            GroundingEvaluator
            ._selector_pool_quality_override_diagnostic_scores(
                end_points,
                bid=0,
                num_obj=1,
                base_scores=base_scores,
            )
        )

        low_margin_scores = diagnostics[
            "selector_pool_quality_override_m0p05_quality"
        ]
        high_margin_scores = diagnostics[
            "selector_pool_quality_override_m0p25_quality"
        ]
        self.assertEqual(int(low_margin_scores[0].argmax().item()), 1)
        self.assertEqual(int(high_margin_scores[0].argmax().item()), 2)

    def test_selector_pool_source_blend_adds_source_scores_to_candidate_logits(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "eval_selector_pool_k": 1,
            "selector_source_scores": torch.tensor([[
                [0.10, 0.00, 0.00, 0.00],  # base source
                [0.00, 0.70, 0.00, 0.00],  # fused source
                [0.00, 0.00, 0.68, 0.00],  # quality source
            ]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
        }

        unblended = GroundingEvaluator._selector_pool_source_blend_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            alpha=0.0,
        )
        blended = GroundingEvaluator._selector_pool_source_blend_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            alpha=0.5,
        )

        self.assertEqual(int(unblended[0].argmax().item()), 1)
        self.assertEqual(int(blended[0].argmax().item()), 2)
        self.assertAlmostEqual(blended[0, 2].item(), 0.68 + (0.5 * 0.95))

    def test_selector_pool_source_blend_includes_contrastive_base_source(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "eval_selector_pool_k": 1,
            "selector_source_scores": torch.tensor([[
                [0.10, 0.00, 0.00, 0.00],  # base source
                [0.00, 0.20, 0.00, 0.00],  # fused source
                [0.00, 0.00, 0.30, 0.00],  # quality source
                [0.00, 0.00, 0.00, 0.15],  # contrastive_base source
            ]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.20, 0.70, 0.00]]),
            "bbf_base_grounding_scores": torch.tensor([[
                0.00, 0.10, 0.20, 0.99
            ]]),
        }

        blended = GroundingEvaluator._selector_pool_source_blend_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            alpha=1.0,
        )

        self.assertEqual(int(blended[0].argmax().item()), 3)
        self.assertAlmostEqual(blended[0, 3].item(), 0.15 + 0.99)

    def test_selector_pool_source_blend_diagnostic_sweeps_alphas(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "eval_selector_pool_k": 1,
            "selector_source_scores": torch.tensor([[
                [0.10, 0.00, 0.00, 0.00],  # base source
                [0.00, 0.70, 0.00, 0.00],  # fused source
                [0.00, 0.00, 0.68, 0.00],  # quality source
            ]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
        }

        diagnostics = (
            GroundingEvaluator
            ._selector_pool_source_blend_diagnostic_scores(
                end_points,
                bid=0,
                num_obj=1,
                base_scores=base_scores,
            )
        )

        self.assertIn("selector_pool_source_blend_a0", diagnostics)
        self.assertIn("selector_pool_source_blend_a0p5", diagnostics)
        self.assertEqual(
            int(diagnostics["selector_pool_source_blend_a0"][0].argmax().item()),
            1,
        )
        self.assertEqual(
            int(diagnostics["selector_pool_source_blend_a0p5"][0].argmax().item()),
            2,
        )

    def test_selector_choice_primary_uses_chosen_source_top1(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[0.10, 0.90, 0.20]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
        }

        scores = GroundingEvaluator._selector_choice_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )

        self.assertLess(scores[0, 0].item(), -1e20)
        self.assertAlmostEqual(scores[0, 1].item(), 0.9)
        self.assertLess(scores[0, 2].item(), -1e20)

    def test_selector_choice_primary_applies_source_biases(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[0.10, 0.20, 0.40]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
            "eval_selector_choice_source_bias_base": 0.40,
        }

        scores = GroundingEvaluator._selector_choice_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )

        self.assertEqual(int(scores[0].argmax().item()), 0)
        self.assertAlmostEqual(scores[0, 0].item(), 0.1)

    def test_selector_choice_primary_can_use_separate_override_head(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[0.10, 1.20, 0.80]]),
            "selector_choice_override_logit": torch.tensor([-0.25]),
            "eval_selector_choice_use_override_head": True,
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
        }

        keep_base_scores = GroundingEvaluator._selector_choice_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )
        end_points["selector_choice_override_logit"] = torch.tensor([0.25])
        override_scores = GroundingEvaluator._selector_choice_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )

        self.assertEqual(int(keep_base_scores[0].argmax().item()), 0)
        self.assertEqual(int(override_scores[0].argmax().item()), 1)

    def test_selector_choice_override_head_can_keep_detector_policy_default(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([
                [0.10, 1.20, 0.80, 0.50, 0.40]
            ]),
            "selector_choice_source_names": (
                "base",
                "fused",
                "quality",
                "contrastive_base",
                "detector_countsplit",
            ),
            "selector_choice_override_logit": torch.tensor([-0.25]),
            "eval_selector_choice_use_override_head": True,
            "eval_selector_choice_override_default_source": (
                "detector_countsplit"
            ),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.10, 0.20, 0.30, 0.90]
            ]),
            "detector_countsplit_scores": torch.tensor([
                [0.05, 0.05, 0.05, 0.99]
            ]),
        }

        keep_scores = GroundingEvaluator._selector_choice_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )
        end_points["selector_choice_override_logit"] = torch.tensor([0.25])
        override_scores = GroundingEvaluator._selector_choice_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )

        self.assertEqual(int(keep_scores[0].argmax().item()), 3)
        self.assertEqual(int(override_scores[0].argmax().item()), 1)

    def test_selector_choice_override_head_respects_eval_threshold(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[0.10, 1.20, 0.80]]),
            "selector_choice_override_logit": torch.tensor([-0.25]),
            "eval_selector_choice_use_override_head": True,
            "eval_selector_choice_override_threshold": -0.50,
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
        }

        scores = GroundingEvaluator._selector_choice_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )

        self.assertEqual(int(scores[0].argmax().item()), 1)

    def test_selector_choice_override_diagnostics_sweep_thresholds(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[0.10, 1.20, 0.80]]),
            "selector_choice_override_logit": torch.tensor([-0.25]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
        }

        sweep_scores = (
            GroundingEvaluator
            ._selector_choice_override_threshold_diagnostic_scores(
                end_points,
                bid=0,
                num_obj=1,
                base_scores=base_scores,
            )
        )

        self.assertEqual(
            int(sweep_scores["selector_choice_override_tneg0p5"][0].argmax()),
            1,
        )
        self.assertEqual(
            int(sweep_scores["selector_choice_override_t0"][0].argmax()),
            0,
        )

    def test_selector_choice_override_diagnostics_include_positive_thresholds(
        self,
    ):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[0.10, 1.20, 0.80]]),
            "selector_choice_override_logit": torch.tensor([0.25]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
        }

        sweep_scores = (
            GroundingEvaluator
            ._selector_choice_override_threshold_diagnostic_scores(
                end_points,
                bid=0,
                num_obj=1,
                base_scores=base_scores,
            )
        )

        self.assertEqual(
            int(sweep_scores["selector_choice_override_t0"][0].argmax()),
            1,
        )
        self.assertEqual(
            int(sweep_scores["selector_choice_override_t0p5"][0].argmax()),
            0,
        )

    def test_selector_choice_hybrid_falls_back_when_margin_is_low(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[0.80, 0.70, 0.10]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
            "eval_selector_choice_min_margin": 0.25,
            "eval_selector_choice_hybrid_fallback": "quality",
        }

        scores = GroundingEvaluator._selector_choice_hybrid_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )

        self.assertEqual(int(scores[0].argmax().item()), 2)
        self.assertAlmostEqual(scores[0, 2].item(), 0.95)

    def test_selector_choice_hybrid_uses_choice_when_margin_is_high(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[0.80, 0.70, 0.10]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
            "eval_selector_choice_min_margin": 0.05,
            "eval_selector_choice_hybrid_fallback": "quality",
        }

        scores = GroundingEvaluator._selector_choice_hybrid_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )

        self.assertEqual(int(scores[0].argmax().item()), 0)
        self.assertAlmostEqual(scores[0, 0].item(), 0.80)

    def test_selector_choice_quality_override_defaults_to_quality(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[0.80, 0.70, 0.78]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
            "eval_selector_choice_min_margin": 0.05,
        }

        scores = GroundingEvaluator._selector_choice_quality_override_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )

        self.assertEqual(int(scores[0].argmax().item()), 2)
        self.assertAlmostEqual(scores[0, 2].item(), 0.95)

    def test_selector_choice_quality_override_uses_nonquality_above_quality_margin(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[0.85, 0.70, 0.78]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
            "eval_selector_choice_min_margin": 0.05,
        }

        scores = GroundingEvaluator._selector_choice_quality_override_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )

        self.assertEqual(int(scores[0].argmax().item()), 0)
        self.assertAlmostEqual(scores[0, 0].item(), 0.85)

    def test_selector_choice_quality_override_can_use_separate_override_head(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[0.85, 0.70, 0.78]]),
            "selector_choice_override_logit": torch.tensor([-0.25]),
            "eval_selector_choice_use_override_head": True,
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
            "eval_selector_choice_min_margin": 0.05,
        }

        keep_quality_scores = (
            GroundingEvaluator
            ._selector_choice_quality_override_primary_scores(
                end_points,
                bid=0,
                num_obj=1,
                base_scores=base_scores,
            )
        )
        end_points["selector_choice_override_logit"] = torch.tensor([0.25])
        override_scores = (
            GroundingEvaluator
            ._selector_choice_quality_override_primary_scores(
                end_points,
                bid=0,
                num_obj=1,
                base_scores=base_scores,
            )
        )

        self.assertEqual(int(keep_quality_scores[0].argmax().item()), 2)
        self.assertEqual(int(override_scores[0].argmax().item()), 0)

    def test_selector_choice_hybrid_diagnostic_sweeps_confidence_margins(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[0.80, 0.70, 0.10]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
        }

        diagnostics = GroundingEvaluator._selector_choice_hybrid_diagnostic_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
        )

        low_margin_scores = diagnostics[
            "selector_choice_hybrid_m0p05_quality"
        ]
        high_margin_scores = diagnostics[
            "selector_choice_hybrid_m0p25_quality"
        ]
        self.assertEqual(int(low_margin_scores[0].argmax().item()), 0)
        self.assertEqual(int(high_margin_scores[0].argmax().item()), 2)

    def test_selector_choice_quality_override_diagnostic_sweeps_margins(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[0.85, 0.70, 0.78]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
        }

        diagnostics = (
            GroundingEvaluator
            ._selector_choice_quality_override_diagnostic_scores(
                end_points,
                bid=0,
                num_obj=1,
                base_scores=base_scores,
            )
        )

        low_margin_scores = diagnostics[
            "selector_choice_quality_override_m0p05_quality"
        ]
        high_margin_scores = diagnostics[
            "selector_choice_quality_override_m0p1_quality"
        ]
        self.assertEqual(int(low_margin_scores[0].argmax().item()), 0)
        self.assertEqual(int(high_margin_scores[0].argmax().item()), 2)

    def test_selector_choice_quality_override_diagnostic_sweeps_override_thresholds(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[0.85, 0.70, 0.78]]),
            "selector_choice_override_logit": torch.tensor([-0.25]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
        }

        diagnostics = (
            GroundingEvaluator
            ._selector_choice_quality_override_diagnostic_scores(
                end_points,
                bid=0,
                num_obj=1,
                base_scores=base_scores,
            )
        )

        lower_threshold_scores = diagnostics[
            "selector_choice_quality_override_tneg0p5_quality"
        ]
        zero_threshold_scores = diagnostics[
            "selector_choice_quality_override_t0_quality"
        ]
        self.assertEqual(int(lower_threshold_scores[0].argmax().item()), 0)
        self.assertEqual(int(zero_threshold_scores[0].argmax().item()), 2)

    def test_selector_choice_diagnostics_compare_selected_source_to_oracle(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[0.10, 0.90, 0.20]]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
        }
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [0.80, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        diagnostics = GroundingEvaluator._selector_choice_diagnostics(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
        )

        self.assertEqual(diagnostics["selected_source_id"].item(), 1.0)
        self.assertEqual(diagnostics["oracle_source_id"].item(), 2.0)
        self.assertEqual(diagnostics["source_choice_oracle_agree"].item(), 0.0)
        self.assertLess(
            diagnostics["source_choice_selected_iou"].item(),
            diagnostics["source_choice_oracle_iou"].item(),
        )

    def test_selector_choice_diagnostics_report_base_override_error_types(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [0.80, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        false_base = GroundingEvaluator._selector_choice_diagnostics(
            {
                "selector_choice_scores": torch.tensor([[2.00, 0.10, 0.20]]),
                "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
                "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
            },
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
        )

        self.assertEqual(
            false_base["source_choice_target_override_ratio"].item(), 1.0
        )
        self.assertEqual(
            false_base["source_choice_selected_override_ratio"].item(), 0.0
        )
        self.assertEqual(
            false_base["source_choice_false_base_ratio"].item(), 1.0
        )
        self.assertEqual(
            false_base["source_choice_false_override_ratio"].item(), 0.0
        )

        false_override = GroundingEvaluator._selector_choice_diagnostics(
            {
                "selector_choice_scores": torch.tensor([[0.10, 0.20, 2.00]]),
                "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
                "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
            },
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=torch.tensor([
                [0.00, 0.0, 0.0, 2.0, 2.0, 2.0],
                [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
                [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
                [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]),
        )

        self.assertEqual(
            false_override["source_choice_target_override_ratio"].item(), 0.0
        )
        self.assertEqual(
            false_override["source_choice_selected_override_ratio"].item(), 1.0
        )
        self.assertEqual(
            false_override["source_choice_false_base_ratio"].item(), 0.0
        )
        self.assertEqual(
            false_override["source_choice_false_override_ratio"].item(), 1.0
        )

    def test_selector_choice_quality_override_diagnostics_use_quality_default(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([[2.00, 0.10, 0.20]]),
            "selector_choice_override_logit": torch.tensor([-0.25]),
            "eval_selector_choice_use_override_head": True,
            "eval_selector_choice_override_threshold": 0.0,
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.70, 0.95, 0.00]]),
        }
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [0.80, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        diagnostics = GroundingEvaluator._selector_choice_diagnostics(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
            selection_mode="selector_choice_quality_override",
        )

        self.assertEqual(diagnostics["selected_source_id"].item(), 2.0)
        self.assertEqual(
            diagnostics["source_choice_selected_quality_ratio"].item(), 1.0
        )
        self.assertEqual(
            diagnostics["source_choice_selected_override_ratio"].item(), 0.0
        )

    def test_source_choice_feature_row_records_source_top1_labels(self):
        end_points = {
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00]]),
            "pred_iou": torch.tensor([[0.20, 0.10, 0.80]]),
            "last_sem_cls_scores": torch.tensor([[
                [2.0, 0.0],
                [0.0, 3.0],
                [0.5, 0.5],
            ]]),
            "last_objectness_scores": torch.tensor([[0.1, 0.2, 0.3]]),
            "seeds_obj_cls_logits": torch.tensor([[
                [0.10, 0.20, 0.40, 0.70]
            ]]),
            "query_points_sample_inds": torch.tensor([[1, 2, 3]]),
        }
        base_scores = torch.tensor([[0.95, 0.20, 0.10]])
        contrastive_scores = torch.tensor([[0.10, 0.30, 0.85]])
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        row = GroundingEvaluator._source_choice_feature_row(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
            extra_source_scores={"contrastive_base": contrastive_scores},
        )

        self.assertEqual(row["oracle_source_id"], 2.0)
        self.assertEqual(row["base_top_query"], 0.0)
        self.assertEqual(row["quality_top_query"], 2.0)
        self.assertEqual(row["contrastive_base_top_query"], 2.0)
        self.assertAlmostEqual(row["quality_top_iou"], 1.0)
        self.assertAlmostEqual(row["contrastive_base_top_iou"], 1.0)
        self.assertAlmostEqual(row["base_score_at_quality_top"], 0.10)
        self.assertAlmostEqual(row["quality_score_at_base_top"], 0.20)
        self.assertAlmostEqual(row["base_top_center_x"], 5.0)
        self.assertAlmostEqual(row["quality_top_center_x"], 0.0)
        self.assertAlmostEqual(row["quality_top_volume"], 8.0)
        self.assertEqual(row["quality_top_sem_cls_available"], 1.0)
        self.assertAlmostEqual(row["quality_top_sem_cls_max"], 0.5)
        self.assertAlmostEqual(row["quality_top_sem_cls_margin"], 0.0)
        self.assertEqual(row["quality_top_objectness_available"], 1.0)
        self.assertAlmostEqual(row["quality_top_objectness_score"], 0.3)
        self.assertEqual(row["quality_top_seed_objectness_available"], 1.0)
        self.assertAlmostEqual(row["quality_top_seed_objectness_score"], 0.70)
        self.assertEqual(row["quality_top_seed_objectness_rank"], 1.0)

    def test_source_choice_feature_row_uses_configured_detector_choice_sources(self):
        end_points = {
            "selector_choice_source_names": (
                "base",
                "quality",
                "detector_jointtight",
            ),
            "pred_iou": torch.tensor([[0.10, 0.80, 0.20]]),
            "detector_jointtight_scores": torch.tensor([[0.20, 0.10, 0.95]]),
        }
        base_scores = torch.tensor([[0.90, 0.10, 0.20]])
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        row = GroundingEvaluator._source_choice_feature_row(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
        )

        self.assertEqual(row["detector_jointtight_available"], 1.0)
        self.assertEqual(row["detector_jointtight_top_query"], 2.0)
        self.assertEqual(row["detector_jointtight_top_iou"], 1.0)
        self.assertEqual(row["oracle_source_id"], 2.0)

    def test_source_choice_feature_row_records_selector_logits_for_calibration(self):
        end_points = {
            "selector_choice_source_names": (
                "detector_jointtight",
                "base",
                "quality",
            ),
            "selector_choice_scores": torch.tensor([[0.20, 0.50, -0.10]]),
            "eval_selector_choice_source_bias_detector_jointtight": 0.30,
            "eval_selector_choice_source_bias_base": -0.10,
            "eval_selector_choice_source_bias_quality": 0.0,
            "pred_iou": torch.tensor([[0.10, 0.20, 0.80]]),
            "detector_jointtight_scores": torch.tensor([[0.70, 0.20, 0.10]]),
        }
        base_scores = torch.tensor([[0.10, 0.90, 0.20]])
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        row = GroundingEvaluator._source_choice_feature_row(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
        )

        self.assertAlmostEqual(
            row["selector_choice_logit_detector_jointtight"], 0.20
        )
        self.assertAlmostEqual(
            row["selector_choice_logit_bias_detector_jointtight"], 0.30
        )
        self.assertAlmostEqual(
            row["selector_choice_applied_logit_detector_jointtight"], 0.50
        )
        self.assertAlmostEqual(row["selector_choice_logit_base"], 0.50)
        self.assertAlmostEqual(row["selector_choice_logit_bias_base"], -0.10)
        self.assertAlmostEqual(row["selector_choice_applied_logit_base"], 0.40)
        self.assertAlmostEqual(
            row["selector_choice_logit_margin_detector_jointtight"], -0.30
        )
        self.assertAlmostEqual(
            row["selector_choice_applied_logit_margin_detector_jointtight"],
            0.10,
        )
        self.assertEqual(row["selector_choice_selected_source_id"], 0.0)

    def test_source_choice_feature_row_records_override_logit_for_replay(self):
        end_points = {
            "selector_choice_source_names": (
                "detector_jointtight",
                "quality",
            ),
            "selector_choice_scores": torch.tensor([[0.20, 0.10]]),
            "selector_choice_override_logit": torch.tensor([0.75]),
            "eval_selector_choice_use_override_head": True,
            "eval_selector_choice_override_threshold": 0.25,
            "eval_selector_choice_override_default_source": (
                "detector_jointtight"
            ),
            "pred_iou": torch.tensor([[0.10, 0.20, 0.80]]),
            "detector_jointtight_scores": torch.tensor([[0.70, 0.20, 0.10]]),
        }
        base_scores = torch.tensor([[0.10, 0.90, 0.20]])
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        row = GroundingEvaluator._source_choice_feature_row(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
        )

        self.assertAlmostEqual(row["selector_choice_override_logit"], 0.75)
        self.assertAlmostEqual(
            row["selector_choice_override_threshold"], 0.25
        )
        self.assertEqual(
            row["selector_choice_override_default_source_id"], 0.0
        )
        self.assertEqual(row["selector_choice_selected_source_id"], 1.0)

    def test_source_choice_feature_row_records_context_and_source_prefixed_rapf_features(self):
        end_points = {
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00, 0.20]]),
            "pred_iou": torch.tensor([[0.20, 0.10, 0.80, 0.70]]),
            "coverage_stats": [{
                "has_target": 1,
                "num_attrs": 2,
                "num_pairs": 1,
                "num_parse_errors": 3,
            }],
            "decomposition_status": ["repaired_target_recovered"],
            "decomp_global_only_mask": [False],
            "decomp_weak_generic_mask": [False],
            "global_only_due_to_parse_error": [False],
            "target_generic_reference": [True],
            "decomposition_error_flags_count": [3],
            "entity_spans": [[{"text": "chair"}, {"text": "table"}]],
            "attr_spans": [[{"text": "red"}]],
            "rel_spans": [[{"text": "left of"}]],
            "anchor_slots": [[{"text": "table"}]],
            "slot_mask": [[1, 1, 0]],
            "is_view_dep": torch.tensor([True]),
            "metadata_conflict_ratio": torch.tensor([1.0]),
            "positive_map_source_explicit_target_slot_ratio": torch.tensor([0.0]),
            "positive_map_source_lexical_exact_match_ratio": torch.tensor([1.0]),
            "positive_map_source_missing_ratio": torch.tensor([0.0]),
            "rapf_gate": torch.tensor([[0.10, 0.20, 0.30, 0.40]]),
            "rapf_base_norm": torch.tensor([[1.10, 1.20, 1.30, 1.40]]),
            "rapf_structured_norm": torch.tensor([[2.10, 2.20, 2.30, 2.40]]),
            "rapf_quality_norm": torch.tensor([[3.10, 3.20, 3.30, 3.40]]),
            "rapf_safe_anchor": torch.tensor([[4.10, 4.20, 4.30, 4.40]]),
            "rapf_delta": torch.tensor([[5.10, 5.20, 5.30, 5.40]]),
        }
        base_scores = torch.tensor([[0.95, 0.20, 0.10, 0.30]])
        contrastive_scores = torch.tensor([[0.10, 0.30, 0.85, 0.40]])
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        row = GroundingEvaluator._source_choice_feature_row(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
            extra_source_scores={"contrastive_base": contrastive_scores},
        )

        self.assertEqual(row["context_has_target"], 1.0)
        self.assertEqual(row["context_status_repaired"], 1.0)
        self.assertEqual(row["context_entity_span_count"], 2.0)
        self.assertEqual(row["context_slot_mask_count"], 2.0)
        self.assertAlmostEqual(row["base_rapf_gate"], 0.10, places=6)
        self.assertAlmostEqual(row["fused_rapf_gate"], 0.20, places=6)
        self.assertAlmostEqual(row["quality_rapf_gate"], 0.30, places=6)
        self.assertAlmostEqual(
            row["contrastive_base_rapf_gate"], 0.30, places=6
        )

    def test_source_choice_feature_row_records_spacy_augmentation_context(self):
        end_points = {
            "fused_scores": torch.tensor([[0.10, 0.90, 0.00]]),
            "pred_iou": torch.tensor([[0.20, 0.10, 0.80]]),
            "spacy_rotation_mode_id": torch.tensor([2]),
            "spacy_augmentation_profile_id": torch.tensor([11]),
        }
        base_scores = torch.tensor([[0.95, 0.20, 0.10]])
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        row = GroundingEvaluator._source_choice_feature_row(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
        )

        self.assertEqual(row["spacy_rotation_mode_id"], 2.0)
        self.assertEqual(row["spacy_aug_yaw_only"], 1.0)
        self.assertEqual(row["spacy_aug_full_natural"], 0.0)
        self.assertEqual(row["spacy_augmentation_profile_id"], 11.0)
        self.assertEqual(row["spacy_profile_stable_direction_sensitive"], 1.0)
        self.assertEqual(row["spacy_profile_yaw_relation_free"], 0.0)

    def test_source_choice_candidate_rows_record_union_topk_candidates(self):
        end_points = {
            "fused_scores": torch.tensor([[0.30, 0.90, 0.00, 0.20]]),
            "pred_iou": torch.tensor([[0.20, 0.10, 0.80, 0.70]]),
            "last_sem_cls_scores": torch.tensor([[
                [2.0, 0.0],
                [0.0, 3.0],
                [1.0, 4.0],
                [0.5, 0.5],
            ]]),
            "last_objectness_scores": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            "seeds_obj_cls_logits": torch.tensor([[
                [0.10, 0.90, 0.40, 0.20, 0.70]
            ]]),
            "query_points_sample_inds": torch.tensor([[1, 2, 3, 4]]),
        }
        base_scores = torch.tensor([[0.95, 0.20, 0.10, 0.30]])
        contrastive_scores = torch.tensor([[0.10, 0.30, 0.85, 0.40]])
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        rows = GroundingEvaluator._source_choice_candidate_rows(
            end_points,
            bid=0,
            example_id=7,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
            topk=2,
            extra_source_scores={"contrastive_base": contrastive_scores},
        )

        self.assertEqual(
            sorted(int(row["candidate_query"]) for row in rows),
            [0, 1, 2, 3],
        )
        oracle_rows = [
            row for row in rows if row["oracle_candidate"] == 1.0
        ]
        self.assertEqual(len(oracle_rows), 1)
        self.assertEqual(oracle_rows[0]["candidate_query"], 2.0)
        query_three = [
            row for row in rows if row["candidate_query"] == 3.0
        ][0]
        self.assertEqual(query_three["example_id"], 7.0)
        self.assertEqual(query_three["base_in_topk"], 1.0)
        self.assertEqual(query_three["quality_in_topk"], 1.0)
        self.assertEqual(query_three["fused_in_topk"], 0.0)
        self.assertAlmostEqual(query_three["quality_score"], 0.70)
        self.assertAlmostEqual(query_three["candidate_center_x"], 0.0)
        self.assertAlmostEqual(query_three["candidate_size_x"], 2.0)
        self.assertAlmostEqual(query_three["candidate_volume"], 8.0)
        self.assertAlmostEqual(query_three["candidate_sem_cls_max"], 0.5)
        self.assertAlmostEqual(query_three["candidate_sem_cls_margin"], 0.0)
        self.assertEqual(query_three["candidate_objectness_available"], 1.0)
        self.assertAlmostEqual(query_three["candidate_objectness_score"], 0.4)
        self.assertEqual(query_three["candidate_seed_objectness_available"], 1.0)
        self.assertAlmostEqual(query_three["candidate_seed_objectness_score"], 0.70)
        self.assertAlmostEqual(
            query_three["candidate_seed_objectness_prob"],
            torch.tensor(0.70).sigmoid().item(),
        )
        self.assertEqual(query_three["candidate_seed_objectness_rank"], 2.0)
        self.assertAlmostEqual(query_three["base_top_center_x"], 5.0)
        self.assertAlmostEqual(query_three["quality_top_center_x"], 0.0)
        self.assertAlmostEqual(query_three["quality_top_volume"], 8.0)
        self.assertEqual(query_three["quality_top_sem_cls_available"], 1.0)

    def test_source_choice_candidate_rows_record_context_features(self):
        end_points = {
            "fused_scores": torch.tensor([[0.30, 0.90, 0.00, 0.20]]),
            "pred_iou": torch.tensor([[0.20, 0.10, 0.80, 0.70]]),
            "parse_confidence": [0.65],
            "coverage_stats": [{
                "has_target": 1,
                "num_attrs": 2,
                "num_pairs": 1,
                "num_parse_errors": 3,
            }],
            "decomposition_status": ["repaired_target_recovered"],
            "decomp_global_only_mask": [False],
            "decomp_weak_generic_mask": [False],
            "global_only_due_to_parse_error": [False],
            "target_generic_reference": [True],
            "decomposition_error_flags_count": [3],
            "entity_spans": [[{"text": "chair"}, {"text": "table"}]],
            "attr_spans": [[{"text": "red"}]],
            "rel_spans": [[{"text": "left of"}]],
            "anchor_slots": [[{"text": "table"}]],
            "slot_mask": [[1, 1, 0]],
            "is_view_dep": torch.tensor([True]),
            "metadata_conflict_ratio": torch.tensor([1.0]),
            "positive_map_source_explicit_target_slot_ratio": torch.tensor([0.0]),
            "positive_map_source_lexical_exact_match_ratio": torch.tensor([1.0]),
            "positive_map_source_missing_ratio": torch.tensor([0.0]),
            "spacy_rotation_mode_id": torch.tensor([2]),
            "spacy_augmentation_profile_id": torch.tensor([11]),
        }
        base_scores = torch.tensor([[0.95, 0.20, 0.10, 0.30]])
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        rows = GroundingEvaluator._source_choice_candidate_rows(
            end_points,
            bid=0,
            example_id=9,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
            topk=2,
        )

        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertAlmostEqual(row["context_parse_confidence"], 0.65)
            self.assertEqual(row["context_has_target"], 1.0)
            self.assertEqual(row["context_num_attrs"], 2.0)
            self.assertEqual(row["context_num_pairs"], 1.0)
            self.assertEqual(row["context_num_parse_errors"], 3.0)
            self.assertEqual(row["context_status_repaired"], 1.0)
            self.assertEqual(row["context_status_ok"], 0.0)
            self.assertEqual(row["context_target_generic_reference"], 1.0)
            self.assertEqual(row["context_entity_span_count"], 2.0)
            self.assertEqual(row["context_attr_span_count"], 1.0)
            self.assertEqual(row["context_rel_span_count"], 1.0)
            self.assertEqual(row["context_anchor_slot_count"], 1.0)
            self.assertEqual(row["context_slot_mask_count"], 2.0)
            self.assertEqual(row["context_is_view_dep"], 1.0)
            self.assertEqual(row["context_metadata_conflict_ratio"], 1.0)
            self.assertEqual(
                row["context_positive_map_lexical_exact_match_ratio"], 1.0
            )
            self.assertEqual(row["spacy_rotation_mode_id"], 2.0)
            self.assertEqual(row["spacy_aug_yaw_only"], 1.0)
            self.assertEqual(row["spacy_augmentation_profile_id"], 11.0)
            self.assertEqual(
                row["spacy_profile_stable_direction_sensitive"], 1.0
            )

    def test_source_choice_candidate_rows_record_layer_stability_features(self):
        end_points = {
            "fused_scores": torch.tensor([[0.85, 0.95, 0.00]]),
            "pred_iou": torch.tensor([[0.20, 0.90, 0.10]]),
            "rapf_gate": torch.tensor([[0.10, 0.20, 0.30]]),
            "rapf_base_norm": torch.tensor([[1.10, 1.20, 1.30]]),
            "rapf_structured_norm": torch.tensor([[2.10, 2.20, 2.30]]),
            "rapf_quality_norm": torch.tensor([[3.10, 3.20, 3.30]]),
            "rapf_safe_anchor": torch.tensor([[4.10, 4.20, 4.30]]),
            "rapf_delta": torch.tensor([[5.10, 5.20, 5.30]]),
            "0head_sem_cls_scores": torch.tensor([[
                [5.0, -5.0],
                [-5.0, 5.0],
                [-5.0, 5.0],
            ]]),
            "1head_sem_cls_scores": torch.tensor([[
                [-5.0, 5.0],
                [5.0, -5.0],
                [-5.0, 5.0],
            ]]),
            "last_sem_cls_scores": torch.tensor([[
                [-5.0, 5.0],
                [5.0, -5.0],
                [-5.0, 5.0],
            ]]),
            "0head_proj_queries": torch.tensor([[
                [1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
            ]]),
            "1head_proj_queries": torch.tensor([[
                [-1.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
            ]]),
            "last_proj_queries": torch.tensor([[
                [-1.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
            ]]),
            "proj_tokens": torch.tensor([[
                [1.0, 0.0],
                [-1.0, 0.0],
            ]]),
        }
        base_scores = torch.tensor([[0.90, 0.80, 0.10]])
        contrastive_scores = torch.tensor([[0.88, 0.96, 0.05]])
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        rows = GroundingEvaluator._source_choice_candidate_rows(
            end_points,
            bid=0,
            example_id=11,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
            topk=2,
            extra_source_scores={"contrastive_base": contrastive_scores},
            target_map=torch.tensor([1.0, 0.0]),
        )

        query_one = [row for row in rows if row["candidate_query"] == 1.0][0]
        self.assertEqual(query_one["layer_stability_bbs_num_layers"], 3.0)
        self.assertEqual(query_one["layer_stability_bbf_num_layers"], 3.0)
        self.assertEqual(query_one["layer_stability_bbs_top1_count"], 2.0)
        self.assertEqual(query_one["layer_stability_bbf_top1_count"], 2.0)
        self.assertEqual(
            query_one["layer_stability_bbs_bbf_top1_agreement_ratio"], 1.0
        )
        self.assertGreater(
            query_one["layer_stability_bbs_score_last_minus_first"], 0.0
        )
        self.assertGreater(
            query_one["layer_stability_bbf_score_last_minus_first"], 0.0
        )
        self.assertEqual(query_one["layer_stability_bbs_top1_change_count"], 1.0)

    def test_source_choice_candidate_rows_record_rapf_features(self):
        end_points = {
            "fused_scores": torch.tensor([[0.30, 0.90, 0.00, 0.20]]),
            "pred_iou": torch.tensor([[0.20, 0.10, 0.80, 0.70]]),
            "rapf_gate": torch.tensor([[0.10, 0.20, 0.30, 0.40]]),
            "rapf_base_norm": torch.tensor([[1.10, 1.20, 1.30, 1.40]]),
            "rapf_structured_norm": torch.tensor([[2.10, 2.20, 2.30, 2.40]]),
            "rapf_quality_norm": torch.tensor([[3.10, 3.20, 3.30, 3.40]]),
            "rapf_safe_anchor": torch.tensor([[4.10, 4.20, 4.30, 4.40]]),
            "rapf_delta": torch.tensor([[5.10, 5.20, 5.30, 5.40]]),
        }
        base_scores = torch.tensor([[0.95, 0.20, 0.10, 0.30]])
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        rows = GroundingEvaluator._source_choice_candidate_rows(
            end_points,
            bid=0,
            example_id=10,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
            topk=2,
        )

        query_three = [row for row in rows if row["candidate_query"] == 3.0][0]
        self.assertAlmostEqual(query_three["rapf_gate"], 0.40, places=6)
        self.assertAlmostEqual(query_three["rapf_base_norm"], 1.40, places=6)
        self.assertAlmostEqual(
            query_three["rapf_structured_norm"], 2.40, places=6
        )
        self.assertAlmostEqual(query_three["rapf_quality_norm"], 3.40, places=6)
        self.assertAlmostEqual(query_three["rapf_safe_anchor"], 4.40, places=6)
        self.assertAlmostEqual(query_three["rapf_delta"], 5.40, places=6)

    def test_source_choice_example_id_advances_per_sample_not_per_row(self):
        evaluator = GroundingEvaluator(
            source_choice_dump_path="/tmp/source_choice.pt",
            source_choice_dump_topk=5,
        )

        first_id = evaluator._next_source_choice_example_id()
        evaluator.source_choice_feature_rows.extend([{}, {}, {}, {}])
        second_id = evaluator._next_source_choice_example_id()
        evaluator.source_choice_feature_rows.extend([{}, {}])
        third_id = evaluator._next_source_choice_example_id()

        self.assertEqual(first_id, 0)
        self.assertEqual(second_id, 1)
        self.assertEqual(third_id, 2)

        evaluator.reset()
        self.assertEqual(evaluator._next_source_choice_example_id(), 0)

    def test_source_choice_sharded_manifest_records_source_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_path = os.path.join(tmpdir, "source_choice.pt")
            evaluator = GroundingEvaluator(
                prefixes=[],
                source_choice_dump_path=dump_path,
                source_choice_dump_topk=1,
            )
            evaluator.selector_choice_source_names_for_logging = (
                "detector_jointtight",
                "base",
                "quality",
            )
            evaluator.source_choice_feature_rows.append({"row": 1.0})

            with mock.patch("utils.misc.is_main_process", return_value=True):
                evaluator.print_stats()

            manifest = torch.load(dump_path, map_location="cpu")

        self.assertEqual(
            tuple(manifest["source_names"]),
            ("detector_jointtight", "base", "quality"),
        )
        self.assertEqual(
            tuple(manifest["selector_choice_source_names"]),
            ("detector_jointtight", "base", "quality"),
        )

    def test_source_choice_sharded_manifest_preserves_candidate_dump_source_names_after_selector_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_path = os.path.join(tmpdir, "source_choice.pt")
            evaluator = GroundingEvaluator(
                prefixes=[],
                source_choice_dump_path=dump_path,
                source_choice_dump_topk=5,
            )
            evaluator.source_choice_dump_source_names_for_logging = (
                "detector_jointtight",
                "base",
                "quality",
                "contrastive_base",
                "target_detector",
                "target_detector_logit",
            )
            evaluator.selector_choice_source_names_for_logging = (
                "detector_jointtight",
                "base",
                "quality",
            )
            evaluator.source_choice_feature_rows.append({"row": 1.0})

            with mock.patch("utils.misc.is_main_process", return_value=True):
                evaluator.print_stats()

            manifest = torch.load(dump_path, map_location="cpu")

        self.assertEqual(
            tuple(manifest["source_names"]),
            (
                "detector_jointtight",
                "base",
                "quality",
                "contrastive_base",
                "target_detector",
                "target_detector_logit",
            ),
        )
        self.assertEqual(
            tuple(manifest["selector_choice_source_names"]),
            ("detector_jointtight", "base", "quality"),
        )

    def test_selector_choice_diagnostics_report_threshold_hits_by_source(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([
                [0.90, 0.10, 0.20, 0.30]
            ]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.20, 0.95, 0.00]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.10, 0.20, 0.95]
            ]),
        }
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [0.80, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.40, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        diagnostics = GroundingEvaluator._selector_choice_diagnostics(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
        )

        self.assertEqual(
            diagnostics["source_choice_selected_hit025"].item(), 1.0
        )
        self.assertEqual(
            diagnostics["source_choice_selected_hit050"].item(), 0.0
        )
        self.assertEqual(
            diagnostics[
                "source_choice_contrastive_base_hit050_ratio"
            ].item(),
            1.0,
        )
        self.assertEqual(
            diagnostics[
                "source_choice_contrastive_base_unique_hit050_ratio"
            ].item(),
            1.0,
        )
        self.assertEqual(
            diagnostics[
                "source_choice_contrastive_base_unique_hit025_ratio"
            ].item(),
            0.0,
        )

    def test_selector_choice_diagnostics_report_per_source_logits(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([
                [1.50, -0.50, 0.25, 0.75]
            ]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.20, 0.95, 0.00]]),
            "bbf_base_grounding_scores": torch.tensor([
                [0.00, 0.10, 0.20, 0.95]
            ]),
        }
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [0.80, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.40, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        diagnostics = GroundingEvaluator._selector_choice_diagnostics(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
        )

        self.assertAlmostEqual(
            diagnostics["source_choice_logit_base_mean"].item(), 1.5
        )
        self.assertAlmostEqual(
            diagnostics[
                "source_choice_logit_margin_base_mean"
            ].item(),
            0.75,
        )
        self.assertAlmostEqual(
            diagnostics["source_choice_logit_fused_mean"].item(), -0.5
        )
        self.assertAlmostEqual(
            diagnostics[
                "source_choice_logit_margin_fused_mean"
            ].item(),
            -2.0,
        )

    def test_selector_choice_diagnostics_can_use_eval_computed_contrastive_scores(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([
                [0.90, 0.10, 0.20, 0.30]
            ]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.20, 0.95, 0.00]]),
        }
        gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        pred_bbox = torch.tensor([
            [0.80, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.00, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.40, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        diagnostics = GroundingEvaluator._selector_choice_diagnostics(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes,
            pred_bbox=pred_bbox,
            extra_source_scores={
                "contrastive_base": torch.tensor([
                    [0.00, 0.10, 0.20, 0.95]
                ])
            },
        )

        self.assertEqual(
            diagnostics[
                "source_choice_contrastive_base_hit050_ratio"
            ].item(),
            1.0,
        )

    def test_selector_choice_primary_can_use_eval_computed_contrastive_scores(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20, 0.00]])
        end_points = {
            "selector_choice_scores": torch.tensor([
                [0.10, 0.20, 0.30, 0.90]
            ]),
            "fused_scores": torch.tensor([[0.10, 0.80, 0.20, 0.00]]),
            "pred_iou": torch.tensor([[0.10, 0.20, 0.95, 0.00]]),
        }

        scores = GroundingEvaluator._selector_choice_primary_scores(
            end_points,
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            extra_source_scores={
                "contrastive_base": torch.tensor([
                    [0.00, 0.10, 0.20, 0.95]
                ])
            },
        )

        self.assertEqual(int(scores[0].argmax().item()), 3)

    def test_checkpoint_loader_allows_missing_selector_keys_only_when_enabled(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.other = nn.Linear(2, 2)
                self.source_pool_selector = nn.Linear(2, 1)

        model = ToyModel()
        checkpoint_state = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if not key.startswith("source_pool_selector.")
        }
        args = argparse.Namespace(use_source_pool_selector=True)

        has_fresh_params = _load_model_state_with_selector_compat(
            args, model, checkpoint_state
        )

        self.assertTrue(has_fresh_params)

    def test_checkpoint_loader_allows_missing_late_acd_keys_only_when_enabled(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.other = nn.Linear(2, 2)
                self.source_pool_selector = nn.Linear(2, 1)
                self.acd_head = nn.Linear(2, 1)
                self.dhc_loss_module = nn.Linear(2, 1)

        model = ToyModel()
        checkpoint_state = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if not (
                key.startswith("acd_head.")
                or key.startswith("dhc_loss_module.")
            )
        }
        args = argparse.Namespace(
            use_source_pool_selector=True,
            use_late_acd=True,
        )

        has_fresh_params = _load_model_state_with_selector_compat(
            args, model, checkpoint_state
        )

        self.assertTrue(has_fresh_params)

    def test_checkpoint_loader_allows_mismatched_selector_keys(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.other = nn.Linear(2, 2)
                self.source_pool_selector = nn.Linear(3, 1)

        model = ToyModel()
        checkpoint_state = {
            key: value.clone()
            for key, value in model.state_dict().items()
        }
        checkpoint_state["other.weight"] = torch.full_like(
            checkpoint_state["other.weight"], 2.0
        )
        checkpoint_state["source_pool_selector.weight"] = torch.ones(1, 2)
        original_selector_weight = model.source_pool_selector.weight.clone()
        args = argparse.Namespace(use_source_pool_selector=True)

        has_fresh_params = _load_model_state_with_selector_compat(
            args, model, checkpoint_state
        )

        self.assertTrue(has_fresh_params)
        self.assertTrue(torch.allclose(
            model.other.weight,
            torch.full_like(model.other.weight, 2.0),
        ))
        self.assertTrue(torch.allclose(
            model.source_pool_selector.weight,
            original_selector_weight,
        ))

    def test_checkpoint_loader_rejects_unrelated_missing_keys(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.other = nn.Linear(2, 2)

        model = ToyModel()
        checkpoint_state = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if key != "other.bias"
        }
        args = argparse.Namespace(use_source_pool_selector=True)

        with self.assertRaisesRegex(RuntimeError, "Missing checkpoint keys"):
            _load_model_state_with_selector_compat(args, model, checkpoint_state)

    def test_selector_train_only_freezes_other_parameters_and_uses_selector_lr(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.other = nn.Linear(2, 2)
                self.source_pool_selector = nn.Linear(2, 1)

        model = ToyModel()
        args = argparse.Namespace(
            source_pool_selector_train_only=True,
            source_pool_selector_lr=0.003,
            lr=0.0001,
            lr_backbone=0.001,
            text_encoder_lr=0.00001,
            weight_decay=0.0005,
        )

        _freeze_non_selector_parameters(args, model)
        optimizer = BaseTrainTester.get_optimizer(args, model)

        self.assertFalse(model.other.weight.requires_grad)
        self.assertFalse(model.other.bias.requires_grad)
        self.assertTrue(model.source_pool_selector.weight.requires_grad)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.003)

    def test_acd_train_only_freezes_other_parameters_and_uses_acd_lr(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.other = nn.Linear(2, 2)
                self.acd_head = nn.Linear(2, 1)
                self.dhc_loss_module = nn.Linear(2, 1)

        model = ToyModel()
        args = argparse.Namespace(
            acd_train_only=True,
            acd_lr=0.004,
            lr=0.0001,
            lr_backbone=0.001,
            text_encoder_lr=0.00001,
            weight_decay=0.0005,
        )

        trainable_count = _freeze_non_acd_parameters(args, model)
        expected = (
            sum(p.numel() for p in model.acd_head.parameters())
            + sum(p.numel() for p in model.dhc_loss_module.parameters())
        )
        self.assertEqual(trainable_count, expected)
        self.assertFalse(model.other.weight.requires_grad)
        self.assertFalse(model.other.bias.requires_grad)
        self.assertTrue(model.acd_head.weight.requires_grad)
        self.assertTrue(model.dhc_loss_module.weight.requires_grad)

        optimizer = BaseTrainTester.get_optimizer(args, model)

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.004)

    def test_acd_quality_base_source_uses_pred_iou_scores(self):
        class DummyAcdHead:
            def compute_base_scores(self, proj_queries, proj_tokens):
                raise AssertionError(
                    "contrastive ACD base should not be used in quality mode"
                )

        end_points = {
            "pred_iou": torch.tensor([[0.1, 0.9], [0.7, 0.3]])
        }

        base_scores = resolve_acd_base_scores(
            "quality",
            DummyAcdHead(),
            end_points,
            torch.randn(2, 2, 4),
            torch.randn(2, 3, 4),
        )

        self.assertTrue(torch.equal(base_scores, end_points["pred_iou"]))

    def test_acd_quality_base_source_requires_pred_iou(self):
        class DummyAcdHead:
            def compute_base_scores(self, proj_queries, proj_tokens):
                raise AssertionError("should not be called")

        with self.assertRaisesRegex(ValueError, "pred_iou is missing"):
            resolve_acd_base_scores(
                "quality",
                DummyAcdHead(),
                {},
                torch.randn(1, 2, 4),
                torch.randn(1, 3, 4),
            )

    def test_stat_accumulation_ignores_string_eval_metadata(self):
        stat_dict = BaseTrainTester._accumulate_stats(
            {},
            {
                "eval_primary_score_source": "selector_choice_hybrid",
                "eval_selector_choice_hybrid_fallback": "quality",
                "eval_primary_score_source_id": 8.0,
            },
        )

        self.assertNotIn("eval_primary_score_source", stat_dict)
        self.assertNotIn("eval_selector_choice_hybrid_fallback", stat_dict)
        self.assertEqual(stat_dict["eval_primary_score_source_id"], 8.0)


if __name__ == "__main__":
    unittest.main()
