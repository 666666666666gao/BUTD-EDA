import unittest

import torch

from models.losses import _quality_topk_rerank_losses
from models.losses import compute_hungarian_loss


class TestQualityTopkRerankLoss(unittest.TestCase):
    def test_pushes_best_iou_candidate_above_fused_topk_competitor(self):
        pred_iou = torch.tensor([[0.05, 0.10, 0.90]], requires_grad=True)
        end_points = {
            "pred_iou": pred_iou,
            "fused_scores": torch.tensor([[0.05, 0.80, 0.90]]),
            "last_center": torch.tensor([[
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.tensor([[
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _quality_topk_rerank_losses(
            end_points,
            weight=1.0,
            source="fused",
            candidate_k=2,
            margin=0.1,
            min_iou_gap=0.01,
        )
        losses["loss_quality_topk_rerank"].backward()

        self.assertGreater(losses["loss_quality_topk_rerank"].item(), 0.0)
        self.assertLess(pred_iou.grad[0, 1].item(), 0.0)
        self.assertGreater(pred_iou.grad[0, 2].item(), 0.0)
        self.assertAlmostEqual(losses["dbg_quality_topk_rerank_valid_ratio"].item(), 1.0)

    def test_source_pool_union_supervises_candidates_across_score_sources(self):
        pred_iou = torch.tensor(
            [[0.95, 0.10, 0.20, 0.80]], requires_grad=True
        )
        end_points = {
            "pred_iou": pred_iou,
            "base_grounding_scores": torch.tensor([[0.90, 0.80, 0.10, 0.00]]),
            "fused_scores": torch.tensor([[0.00, 0.10, 0.80, 0.90]]),
            "last_center": torch.tensor([[
                [0.10, 0.0, 0.0],
                [0.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.60, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.tensor([[
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _quality_topk_rerank_losses(
            end_points,
            weight=1.0,
            source="source_pool",
            candidate_k=2,
            margin=0.1,
            min_iou_gap=0.2,
        )
        losses["loss_quality_topk_rerank"].backward()

        self.assertGreater(losses["loss_quality_topk_rerank"].item(), 0.0)
        self.assertLess(pred_iou.grad[0, 1].item(), 0.0)
        self.assertGreater(pred_iou.grad[0, 3].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_quality_topk_rerank_valid_ratio"].item(), 1.0
        )
        self.assertAlmostEqual(
            losses["dbg_quality_topk_rerank_source_pool"], 1.0
        )

    def test_external_pool_uses_base_and_fused_but_not_quality_topk(self):
        pred_iou = torch.tensor(
            [[0.95, 0.10, 0.20, 0.80]], requires_grad=True
        )
        end_points = {
            "pred_iou": pred_iou,
            "base_grounding_scores": torch.tensor([[0.90, 0.80, 0.10, 0.00]]),
            "fused_scores": torch.tensor([[0.00, 0.10, 0.80, 0.90]]),
            "last_center": torch.tensor([[
                [0.10, 0.0, 0.0],
                [0.00, 0.0, 0.0],
                [5.00, 0.0, 0.0],
                [0.60, 0.0, 0.0],
            ]]),
            "last_pred_size": torch.ones(1, 4, 3) * 2.0,
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "box_label_mask": torch.tensor([[1]]),
        }

        losses = _quality_topk_rerank_losses(
            end_points,
            weight=1.0,
            source="external_pool",
            candidate_k=2,
            margin=0.1,
            min_iou_gap=0.2,
        )
        losses["loss_quality_topk_rerank"].backward()

        self.assertGreater(losses["loss_quality_topk_rerank"].item(), 0.0)
        self.assertLess(pred_iou.grad[0, 1].item(), 0.0)
        self.assertGreater(pred_iou.grad[0, 3].item(), 0.0)
        self.assertEqual(pred_iou.grad[0, 0].item(), 0.0)
        self.assertAlmostEqual(
            losses["dbg_quality_topk_rerank_num_candidate_sources"], 2.0
        )
        self.assertAlmostEqual(
            losses["dbg_quality_topk_rerank_source_external_pool"], 1.0
        )

    def test_compute_hungarian_loss_adds_quality_topk_rerank_term(self):
        class ZeroSetCriterion:
            def __call__(self, output, target):
                zero = output["pred_boxes"].sum() * 0.0
                indices = [(torch.tensor([0]), torch.tensor([0]))]
                return {
                    "loss_bbox": zero,
                    "loss_giou": zero,
                    "loss_ce": zero,
                }, indices

        pred_iou = torch.tensor([[0.05, 0.10, 0.90]], requires_grad=True)
        centers = torch.tensor([[
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ]])
        sizes = torch.ones(1, 3, 3) * 2.0
        sem_cls_scores = torch.zeros(1, 3, 2)
        end_points = {
            "pred_iou": pred_iou,
            "fused_scores": torch.tensor([[0.05, 0.80, 0.90]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "sem_cls_label": torch.tensor([[1]]),
            "positive_map": torch.tensor([[[0.0, 1.0]]]),
            "box_label_mask": torch.tensor([[1]]),
            "proposal_center": centers,
            "proposal_pred_size": sizes,
            "proposal_sem_cls_scores": sem_cls_scores,
            "last_center": centers,
            "last_pred_size": sizes,
            "last_sem_cls_scores": sem_cls_scores,
        }

        loss, end_points = compute_hungarian_loss(
            end_points,
            num_decoder_layers=1,
            set_criterion=ZeroSetCriterion(),
            quality_topk_rerank_weight=1.0,
            quality_topk_rerank_source="fused",
            quality_topk_rerank_k=2,
            quality_topk_rerank_margin=0.1,
            quality_topk_rerank_min_iou_gap=0.01,
        )

        self.assertIn("loss_quality_topk_rerank", end_points)
        self.assertGreater(end_points["loss_quality_topk_rerank"].item(), 0.0)
        self.assertAlmostEqual(
            loss.item(),
            end_points["loss_quality_topk_rerank"].item(),
        )


if __name__ == "__main__":
    unittest.main()
