import os
import unittest

import torch

from src import grounding_evaluator as grounding_evaluator_module
from src.grounding_evaluator import GroundingEvaluator


class TestGroundingEvaluatorOracleDiagnostics(unittest.TestCase):
    def test_records_source_pool_oracle_for_topk_candidates(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25, 0.5],
            topks=[1, 5, 10],
            prefixes=["last_"],
        )

        source_ious = {
            "base": torch.tensor([
                [0.10, 0.60, 0.00],
                [0.40, 0.40, 0.40],
            ]),
            "quality": torch.tensor([
                [0.30, 0.20, 0.20],
                [0.20, 0.60, 0.10],
            ]),
            "fused": torch.tensor([
                [0.20, 0.20, 0.20],
                [0.49, 0.10, 0.10],
            ]),
        }

        evaluator._record_diagnostic_source_pool_oracle("last_", source_ious)

        self.assertEqual(
            evaluator.diagnostic_dets[("source_pool_oracle", "last_", 0.25, 1)],
            2,
        )
        self.assertEqual(
            evaluator.diagnostic_dets[("source_pool_oracle", "last_", 0.5, 1)],
            0,
        )
        self.assertEqual(
            evaluator.diagnostic_dets[("source_pool_oracle", "last_", 0.5, 5)],
            2,
        )
        self.assertEqual(
            evaluator.diagnostic_gts[("source_pool_oracle", "last_", 0.5, 5)],
            2,
        )

    def test_reranks_candidate_pool_with_secondary_scores(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25, 0.5],
            topks=[1, 5, 10],
            prefixes=["last_"],
        )
        candidate_scores = torch.tensor([[0.90, 0.80, 0.70, 0.10]])
        rerank_scores = torch.tensor([[0.10, 0.60, 0.40, 0.95]])

        indices = evaluator._rerank_candidate_indices(
            candidate_scores,
            rerank_scores,
            candidate_k=3,
        )

        self.assertEqual(indices.tolist(), [[1, 2, 0]])

    def test_reranks_union_of_source_topk_candidates_with_secondary_scores(self):
        source_scores = {
            "base": torch.tensor([[0.90, 0.80, 0.10, 0.00, 0.00]]),
            "fused": torch.tensor([[0.10, 0.85, 0.95, 0.00, 0.00]]),
        }
        rerank_scores = torch.tensor([[0.10, 0.70, 0.95, 0.90, 0.80]])

        indices = GroundingEvaluator._rerank_source_pool_candidate_indices(
            source_scores,
            rerank_scores,
            candidate_k=2,
        )

        self.assertEqual(indices[:, :3].tolist(), [[2, 1, 0]])
        self.assertNotIn(3, indices[0].tolist())
        self.assertNotIn(4, indices[0].tolist())

    def test_diagnostic_sources_include_contrastive_base_scores(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "positive_map": torch.tensor([[[1.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 2, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0],
                [0.0, 5.0],
            ]]),
            "proj_tokens": torch.tensor([[
                [1.0, 0.0],
                [-1.0, 0.0],
            ]]),
            "last_proj_queries": torch.tensor([[
                [-1.0, 0.0],
                [1.0, 0.0],
            ]]),
        }

        evaluator.evaluate_bbox_by_span(end_points, "last_")

        self.assertEqual(
            evaluator.diagnostic_dets[
                ("contrastive_base", "last_", 0.25, 1)
            ],
            1,
        )

    def test_records_target_semantic_oracle_diagnostic_from_target_cid(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25, 0.5],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
            ]]),
            "pred_iou": torch.tensor([[0.90, 0.10, 0.20]]),
        }

        evaluator.evaluate_bbox_by_span(end_points, "last_")

        self.assertEqual(
            evaluator.diagnostic_dets[
                ("target_semantic", "last_", 0.25, 1)
            ],
            1,
        )
        self.assertEqual(
            evaluator.diagnostic_dets[
                ("quality_target_semantic_blend_a1", "last_", 0.5, 1)
            ],
            1,
        )

    def test_text_target_cid_source_does_not_fallback_to_gt_target_cid(self):
        end_points = {
            "eval_target_cid_source": "text",
            "target_cid": torch.tensor([1]),
        }

        self.assertIsNone(GroundingEvaluator._target_cid(end_points, 0))

    def test_text_target_cid_source_uses_text_target_cid(self):
        end_points = {
            "eval_target_cid_source": "text",
            "target_cid": torch.tensor([1]),
            "text_target_cid": torch.tensor([2]),
        }

        self.assertEqual(GroundingEvaluator._target_cid(end_points, 0), 2)

    def test_records_target_detector_topk_quality_rerank_diagnostic(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.10, 0.90, 0.20]]),
            "all_detected_boxes": torch.tensor([[
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
        }

        evaluator.evaluate_bbox_by_span(end_points, "last_")

        self.assertEqual(
            evaluator.diagnostic_dets[
                ("target_detector_top2_quality_rerank", "last_", 0.25, 1)
            ],
            1,
        )

    def test_records_target_detector_logit_overlap_diagnostic(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 2, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }

        evaluator.evaluate_bbox_by_span(end_points, "last_")

        self.assertEqual(
            evaluator.diagnostic_dets[
                ("target_detector_logit_overlap", "last_", 0.25, 1)
            ],
            1,
        )
        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_logit_overlap_blend_a1",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )

    def test_records_target_scene_class_overlap_diagnostic(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 2, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.55, 0.10]]),
            "all_bboxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_class_ids": torch.tensor([[1, 2]]),
        }

        evaluator.evaluate_bbox_by_span(end_points, "last_")

        self.assertEqual(
            evaluator.diagnostic_dets[
                ("target_scene_class_overlap", "last_", 0.25, 1)
            ],
            1,
        )
        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_scene_class_overlap_blend_a1",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )

    def test_records_target_detector_logit_topk_overlap_diagnostic(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 2, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.90, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[2, 2]]),
            "all_detected_logits": torch.tensor([[
                [8.0, 0.0, 7.0],
                [8.0, 7.0, 0.0],
            ]]),
        }

        evaluator.evaluate_bbox_by_span(end_points, "last_")

        self.assertNotIn(
            ("target_detector_class_overlap", "last_", 0.25, 1),
            evaluator.diagnostic_dets,
        )
        self.assertEqual(
            evaluator.diagnostic_dets[
                ("target_detector_logit_top2_overlap", "last_", 0.25, 1)
            ],
            1,
        )
        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_logit_top2_overlap_blend_a1",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )

    def test_detector_overlap_score_sources_reuse_pairwise_iou(self):
        end_points = {
            "target_cid": torch.tensor([1]),
            "all_detected_boxes": torch.tensor([[
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [9.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[2, 1, 2]]),
            "all_detected_logits": torch.tensor([[
                [8.0, 0.0, 7.0],
                [0.0, 8.0, 1.0],
                [8.0, 7.0, 0.0],
            ]]),
        }
        base_scores = torch.tensor([[0.90, 0.10]])
        pred_bbox = torch.tensor([
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])
        calls = []
        original_iou = grounding_evaluator_module._iou3d_par

        def counting_iou(boxes_a, boxes_b):
            calls.append((tuple(boxes_a.shape), tuple(boxes_b.shape)))
            return original_iou(boxes_a, boxes_b)

        grounding_evaluator_module._iou3d_par = counting_iou
        try:
            scores = GroundingEvaluator._target_detector_overlap_score_sources(
                end_points=end_points,
                bid=0,
                num_obj=1,
                base_scores=base_scores,
                pred_bbox=pred_bbox,
                topks=(2, 3),
            )
        finally:
            grounding_evaluator_module._iou3d_par = original_iou

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            set(scores.keys()),
            {
                "target_detector_class_overlap",
                "target_detector_logit_overlap",
                "target_detector_logit_top2_overlap",
                "target_detector_logit_top3_overlap",
                "target_detector_class_conf_gt0p2_overlap",
                "target_detector_class_conf_gt0p3_overlap",
                "target_detector_class_conf_gt0p4_overlap",
                "target_detector_class_conf_gt0p5_overlap",
            },
        )

    def test_detector_quality_margin_guard_records_low_margin_combo(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.55, 0.50, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = (
            "detector_quality_margin_guard"
        )
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_class_conf_gt0p3_"
                    "qmargin_le0p25_sum_c0p25_f0p25",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )

    def test_detector_conf_logit_combo_records_three_way_sum(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.60, 0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 2]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
                [8.0, 7.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = "detector_conf_logit_combo"
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_class_conf_gt0p3_"
                    "logit_top2_sum_c0p15_f0p25_l0p1",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )

    def test_detector_count_gate_records_same_class_count_combo(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.60, 0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = "detector_count_gate_combo"
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_class_conf_gt0p3_"
                    "detcount_le2_sum_c0p15_f0p25",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )

    def test_detector_count_boost_records_run169_base_plus_count_boost(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.60, 0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = "detector_count_boost_combo"
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_class_conf_gt0p3_"
                    "detcount_le2_boost_c0p15_f0p25_xc0p1",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )

    def test_detector_count_boost_records_quality_fallback_without_detector_prior(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 2, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.10, 0.80]]),
            "all_detected_boxes": torch.tensor([[
                [9.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1]]).bool(),
            "all_detected_class_ids": torch.tensor([[2]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = "detector_count_boost_combo"
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        key = (
            "quality_target_detector_class_conf_gt0p3_"
            "detcount_le2_boost_c0p15_f0p25_xc0p15",
            "last_",
            0.25,
            1,
        )
        self.assertEqual(evaluator.diagnostic_gts[key], 1)
        self.assertEqual(evaluator.diagnostic_dets[key], 1)

    def test_detector_count_boost_le2_records_quality_fallback_without_detector_prior(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 2, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.10, 0.80]]),
            "all_detected_boxes": torch.tensor([[
                [9.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1]]).bool(),
            "all_detected_class_ids": torch.tensor([[2]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = "detector_count_boost_le2_combo"
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        key = (
            "quality_target_detector_class_conf_gt0p3_"
            "detcount_le2_boost_c0p15_f0p25_xc0p25",
            "last_",
            0.25,
            1,
        )
        self.assertEqual(evaluator.diagnostic_gts[key], 1)
        self.assertEqual(evaluator.diagnostic_dets[key], 1)

    def test_detector_count_boost_fine_records_quality_fallback_without_detector_prior(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 2, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.10, 0.80]]),
            "all_detected_boxes": torch.tensor([[
                [9.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1]]).bool(),
            "all_detected_class_ids": torch.tensor([[2]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = "detector_count_boost_fine_combo"
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        key = (
            "quality_target_detector_class_conf_gt0p3_"
            "detcount_le2_boostfine_c0p12_f0p25_xc0p15",
            "last_",
            0.25,
            1,
        )
        self.assertEqual(evaluator.diagnostic_gts[key], 1)
        self.assertEqual(evaluator.diagnostic_dets[key], 1)

    def test_detector_count_boost_fine_records_quality_fallback_without_detector_fields(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 2, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.10, 0.80]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = "detector_count_boost_fine_combo"
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        key = (
            "quality_target_detector_class_conf_gt0p3_"
            "detcount_le2_boostfine_c0p12_f0p25_xc0p15",
            "last_",
            0.25,
            1,
        )
        self.assertEqual(evaluator.diagnostic_gts[key], 1)
        self.assertEqual(evaluator.diagnostic_dets[key], 1)

    def test_detector_count_boost_le2_records_stronger_local_boost(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.60, 0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = "detector_count_boost_le2_combo"
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_class_conf_gt0p3_"
                    "detcount_le2_boost_c0p15_f0p25_xc0p25",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )

    def test_detector_count_topswitch_boost_records_guarded_switch(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.99, 0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = (
            "detector_count_topswitch_boost_combo"
        )
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_class_conf_gt0p3_"
                    "topswitch_le2_w0p05_s0p25_c0p15_f0p25_xc0p15",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )

    def test_detector_count_switchdelta_boost_records_supported_switch(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.99, 0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = (
            "detector_count_switchdelta_boost_combo"
        )
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        key = (
            "quality_target_detector_class_conf_gt0p3_"
            "switchdelta_le2_d0p25_c0p15_f0p25_xc0p15",
            "last_",
            0.25,
            1,
        )
        self.assertIn(key, evaluator.diagnostic_dets)
        self.assertEqual(evaluator.diagnostic_dets[key], 1)

    def test_detector_decomp_status_boost_records_repaired_status_boost(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "decomposition_status": ["repaired_structured"],
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.60, 0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = (
            "detector_decomp_status_boost_combo"
        )
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        key = (
            "quality_target_detector_class_conf_gt0p3_"
            "decomp_repaired_boost_c0p15_f0p25_xc0p25",
            "last_",
            0.25,
            1,
        )
        self.assertIn(key, evaluator.diagnostic_dets)
        self.assertEqual(evaluator.diagnostic_dets[key], 1)

    def test_detector_count_boost_fine_records_local_grid_point(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.60, 0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = (
            "detector_count_boost_fine_combo"
        )
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        key = (
            "quality_target_detector_class_conf_gt0p3_"
            "detcount_le2_boostfine_c0p12_f0p25_xc0p15",
            "last_",
            0.25,
            1,
        )
        self.assertIn(key, evaluator.diagnostic_dets)
        self.assertEqual(evaluator.diagnostic_dets[key], 1)

    def test_detector_jointtight_switch_records_policy_choice(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.99, 0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = (
            "detector_jointtight_switch_combo"
        )
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        key = (
            "quality_target_detector_class_conf_gt0p3_"
            "jointtight_le2_sg0p25_qg0p25",
            "last_",
            0.25,
            1,
        )
        self.assertIn(key, evaluator.diagnostic_dets)

    def test_detector_coarsejoint_switch_records_policy_choice(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.99, 0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = (
            "detector_coarsejoint_switch_combo"
        )
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        key = (
            "quality_target_detector_class_conf_gt0p3_"
            "coarsejoint_le2_sg0p25_qg0p25",
            "last_",
            0.25,
            1,
        )
        self.assertIn(key, evaluator.diagnostic_dets)

    def test_detector_countsplit_records_low_count_policy(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.99, 0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = (
            "detector_countsplit_combo"
        )
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        key = (
            "quality_target_detector_class_conf_gt0p3_"
            "countsplit_le2_lc0p25_lf0p25_hc0p05_hf0p10",
            "last_",
            0.25,
            1,
        )
        self.assertIn(key, evaluator.diagnostic_dets)
        self.assertEqual(evaluator.diagnostic_dets[key], 1)

    def test_detector_strongcoarse_joint_switch_records_policy_choice(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.99, 0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = (
            "detector_strongcoarse_joint_switch_combo"
        )
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        key = (
            "quality_target_detector_class_conf_gt0p3_"
            "strongcoarse_joint_le2_sc0p3_sf0p25_sg0p25_qg1",
            "last_",
            0.25,
            1,
        )
        self.assertIn(key, evaluator.diagnostic_dets)

    def test_detector_triswitch_records_policy_choice(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.99, 0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = (
            "detector_triswitch_combo"
        )
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        key = (
            "quality_target_detector_class_conf_gt0p3_"
            "triswitch_le2_lc0p3_lf0p25_tsg0p25_tqg0p5_csg0p25_cqg0p5",
            "last_",
            0.25,
            1,
        )
        self.assertIn(key, evaluator.diagnostic_dets)

    def test_detector_text_context_split_records_relation_policy(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "coverage_stats": [{"num_pairs": 1}],
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.99, 0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = (
            "detector_text_context_split_combo"
        )
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        key = (
            "quality_target_detector_class_conf_gt0p3_"
            "textctx_rel_c0p1_f0p15",
            "last_",
            0.25,
            1,
        )
        self.assertIn(key, evaluator.diagnostic_dets)

    def test_detector_logit_topk_focus_skips_unrelated_oracle_diagnostics(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 2, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.90, 0.10]]),
            "all_bboxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_class_ids": torch.tensor([[1, 2]]),
            "all_detected_boxes": torch.tensor([[
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[2, 2]]),
            "all_detected_logits": torch.tensor([[
                [8.0, 0.0, 7.0],
                [8.0, 7.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = "detector_logit_topk_overlap"
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        self.assertNotIn(
            ("target_scene_class_overlap", "last_", 0.25, 1),
            evaluator.diagnostic_dets,
        )
        self.assertNotIn(
            ("target_semantic", "last_", 0.25, 1),
            evaluator.diagnostic_dets,
        )
        self.assertNotIn(
            ("source_pool_oracle", "last_", 0.25, 1),
            evaluator.diagnostic_dets,
        )
        self.assertEqual(
            evaluator.diagnostic_dets[
                ("target_detector_logit_top2_overlap", "last_", 0.25, 1)
            ],
            1,
        )
        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_logit_top2_overlap_blend_a1",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )

    def test_detector_class_logit_top2_combo_focus_records_combo(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 2, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.90, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [9.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 2]]),
            "all_detected_logits": torch.tensor([[
                [8.0, 7.0, 0.0],
                [8.0, 7.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = (
            "detector_class_logit_top2_combo"
        )
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        self.assertNotIn(
            ("target_scene_class_overlap", "last_", 0.25, 1),
            evaluator.diagnostic_dets,
        )
        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_class_logit_top2_max_blend_a1",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )
        self.assertIn(
            (
                "quality_target_detector_class_logit_top2_sum_c0p35_l0p15",
                "last_",
                0.25,
                1,
            ),
            evaluator.diagnostic_dets,
        )

    def test_detector_proposal_scene_focus_records_pseudo_scene_prior(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.90, 0.30, 0.85]]),
            "all_detected_boxes": torch.tensor([[
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1]]),
            "all_detected_logits": torch.tensor([[
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = "detector_proposal_scene_overlap"
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        self.assertNotIn(
            ("target_scene_class_overlap", "last_", 0.25, 1),
            evaluator.diagnostic_dets,
        )
        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_class_proposal_scene_top5_overlap_blend_a1",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )
        self.assertIn(
            (
                "target_detector_class_proposal_scene_top5_overlap",
                "last_",
                0.25,
                1,
            ),
            evaluator.diagnostic_dets,
        )

    def test_query_semantic_scene_focus_records_pseudo_scene_prior(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [8.0, -8.0, -8.0, -8.0],
                [-8.0, 8.0, -8.0, -8.0],
                [-8.0, 7.0, -8.0, -8.0],
            ]]),
            "pred_iou": torch.tensor([[0.90, 0.30, 0.85]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = "query_semantic_scene_overlap"
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        self.assertNotIn(
            ("target_scene_class_overlap", "last_", 0.25, 1),
            evaluator.diagnostic_dets,
        )
        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_query_semantic_proposal_scene_top5_overlap_blend_a1",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )
        self.assertIn(
            (
                "target_query_semantic_proposal_scene_top5_overlap",
                "last_",
                0.25,
                1,
            ),
            evaluator.diagnostic_dets,
        )

    def test_records_conditional_target_detector_blend_diagnostic(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.70, 0.45, 0.20]]),
            "all_detected_boxes": torch.tensor([[
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[2, 1]]),
        }

        evaluator.evaluate_bbox_by_span(end_points, "last_")

        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_margin_gt0p25_blend_a0p5",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )

    def test_records_detector_local_agreement_focus_diagnostics(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 3, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.90, 0.86, 0.20]]),
            "all_detected_boxes": torch.tensor([[
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[2, 1]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = "detector_local_agreement"
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        self.assertNotIn(
            ("target_scene_class_overlap", "last_", 0.25, 1),
            evaluator.diagnostic_dets,
        )
        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_quality_top2_mask_blend_a0p5",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )
        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_local_gap_le0p05_margin_gt0p05_blend_a0p5",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )

    def test_target_detector_conf_filter_uses_high_conf_class_boxes(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 2, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.55, 0.10]]),
            "all_detected_boxes": torch.tensor([[
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = "detector_conf_filter"
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        self.assertEqual(
            evaluator.diagnostic_dets[
                ("target_detector_class_conf_gt0p3_overlap", "last_", 0.25, 1)
            ],
            1,
        )
        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_class_conf_gt0p3_overlap_blend_a0p5",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )

    def test_detector_conf_combo_focus_records_weighted_combo(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        end_points = {
            "eval_report_diagnostic_scores": True,
            "target_cid": torch.tensor([1]),
            "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": torch.tensor([[
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 2, 3) * 2.0,
            "last_sem_cls_scores": torch.tensor([[
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
            ]]),
            "pred_iou": torch.tensor([[0.55, 0.20]]),
            "all_detected_boxes": torch.tensor([[
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ]]),
            "all_detected_bbox_label_mask": torch.tensor([[1, 1]]).bool(),
            "all_detected_class_ids": torch.tensor([[1, 1]]),
            "all_detected_logits": torch.tensor([[
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
            ]]),
        }
        previous_focus = os.environ.get("NMV2_EVAL_DIAG_FOCUS")
        os.environ["NMV2_EVAL_DIAG_FOCUS"] = "detector_conf_combo"
        try:
            evaluator.evaluate_bbox_by_span(end_points, "last_")
        finally:
            if previous_focus is None:
                os.environ.pop("NMV2_EVAL_DIAG_FOCUS", None)
            else:
                os.environ["NMV2_EVAL_DIAG_FOCUS"] = previous_focus

        self.assertEqual(
            evaluator.diagnostic_dets[
                (
                    "quality_target_detector_class_conf_gt0p3_sum_c0p25_f0p25",
                    "last_",
                    0.25,
                    1,
                )
            ],
            1,
        )

    def test_source_choice_candidate_rows_include_target_detector_source(self):
        base_scores = torch.tensor([[0.90, 0.10, 0.20]])
        quality_scores = torch.tensor([[0.10, 0.80, 0.20]])
        target_detector_scores = torch.tensor([[0.20, 0.30, 0.95]])
        gt_boxes = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]])
        pred_bbox = torch.tensor([
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [9.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])

        rows = GroundingEvaluator._source_choice_candidate_rows(
            end_points={},
            bid=0,
            example_id=7,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes[0],
            pred_bbox=pred_bbox,
            topk=1,
            extra_source_scores={
                "quality": quality_scores,
                "target_detector": target_detector_scores,
            },
        )
        row_by_query = {
            int(row["candidate_query"]): row for row in rows
        }

        detector_row = row_by_query[2]
        quality_row = row_by_query[1]

        self.assertEqual(detector_row["target_detector_available"], 1.0)
        self.assertEqual(detector_row["target_detector_in_topk"], 1.0)
        self.assertAlmostEqual(detector_row["target_detector_score"], 0.95)
        self.assertAlmostEqual(detector_row["target_detector_rank"], 1.0)
        self.assertEqual(quality_row["target_detector_in_topk"], 0.0)
        self.assertIn(
            "source_pair_quality_minus_target_detector_score",
            detector_row,
        )

    def test_source_choice_row_includes_acd_source(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        base_scores = torch.tensor([[0.90, 0.10]])
        gt_boxes = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]])
        pred_bbox = torch.tensor([
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])
        contrastive_scores = torch.tensor([[0.85, 0.15]])
        acd_scores = torch.tensor([[0.10, 0.95]])

        row = evaluator._source_choice_feature_row(
            end_points={},
            bid=0,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes[0],
            pred_bbox=pred_bbox,
            extra_source_scores={
                "contrastive_base": contrastive_scores,
                "acd": acd_scores,
            },
        )

        self.assertEqual(row["oracle_source_id"], 4.0)
        self.assertEqual(row["acd_available"], 1.0)
        self.assertEqual(row["acd_top_iou"], 1.0)
        self.assertEqual(row["source_pair_base_acd_both_available"], 1.0)
        self.assertAlmostEqual(
            row["source_pair_base_minus_acd_top_score"], -0.05
        )
        self.assertAlmostEqual(
            row["source_pair_base_minus_acd_score_at_base_top"], 0.80
        )
        self.assertAlmostEqual(
            row["source_pair_base_acd_top_center_l1_delta"], 5.0
        )

    def test_source_choice_candidate_rows_include_pairwise_deltas(self):
        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        base_scores = torch.tensor([[0.90, 0.10, 0.20]])
        gt_boxes = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]])
        pred_bbox = torch.tensor([
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [8.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ])
        contrastive_scores = torch.tensor([[0.10, 0.80, 0.20]])

        rows = evaluator._source_choice_candidate_rows(
            end_points={},
            bid=0,
            example_id=7,
            num_obj=1,
            base_scores=base_scores,
            gt_boxes=gt_boxes[0],
            pred_bbox=pred_bbox,
            topk=1,
            extra_source_scores={
                "contrastive_base": contrastive_scores,
            },
        )
        row_by_query = {
            int(row["candidate_query"]): row for row in rows
        }
        row = row_by_query[0]

        self.assertEqual(
            row["source_pair_base_contrastive_base_both_available"], 1.0
        )
        self.assertAlmostEqual(
            row["source_pair_base_minus_contrastive_base_score"], 0.80
        )
        self.assertAlmostEqual(
            row["source_pair_base_minus_contrastive_base_rank"], -2.0
        )
        self.assertAlmostEqual(
            row["source_pair_base_minus_contrastive_base_delta_to_top"],
            -0.70,
        )
        self.assertAlmostEqual(
            row["source_pair_base_minus_contrastive_base_top_score"],
            0.10,
        )
        self.assertAlmostEqual(
            row["source_pair_base_contrastive_base_top_center_l1_delta"],
            5.0,
        )


if __name__ == "__main__":
    unittest.main()
