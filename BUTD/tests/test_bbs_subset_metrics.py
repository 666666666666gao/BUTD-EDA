import pytest
import torch

import src.grounding_evaluator as grounding_evaluator
from src.grounding_evaluator import GroundingEvaluator


def _fake_iou3d_par(gt_boxes, pred_boxes):
    # One GT and two predictions: query 0 succeeds at 0.25 only; query 1 at 0.50.
    values = torch.tensor([[0.30, 0.70]], device=gt_boxes.device)
    return values, values.argmax(dim=1)


@pytest.mark.parametrize(
    ("is_unique", "expected_subset"),
    [(True, "unique"), (False, "multiple")],
)
def test_official_bbs_subset_metrics_follow_primary_scores(
    monkeypatch, is_unique, expected_subset
):
    monkeypatch.setattr(grounding_evaluator, "_iou3d_par", _fake_iou3d_par)
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25, 0.5],
        topks=[1],
        prefixes=["last_"],
    )
    end_points = {
        "positive_map": torch.tensor([[[1.0, 0.0]]]),
        "box_label_mask": torch.tensor([[1]]),
        "center_label": torch.zeros(1, 1, 3),
        "size_gts": torch.ones(1, 1, 3),
        "last_center": torch.zeros(1, 2, 3),
        "last_pred_size": torch.ones(1, 2, 3),
        # Base ranks query 1, fused ranks query 0.
        "last_sem_cls_scores": torch.tensor(
            [[[0.0, 0.0], [4.0, 0.0]]]
        ),
        "fused_scores": torch.tensor([[2.0, 1.0]]),
        "eval_use_fused_scores": True,
        "is_unique": torch.tensor([is_unique]),
    }

    evaluator.evaluate_bbox_by_span(end_points, "last_")
    results = evaluator.print_stats()

    assert results["last__bbs_acc0.25_top1"] == pytest.approx(1.0)
    assert results["last__bbs_acc0.50_top1"] == pytest.approx(0.0)
    assert results[
        f"last__bbs_{expected_subset}_acc0.25_top1"
    ] == pytest.approx(1.0)
    assert results[
        f"last__bbs_{expected_subset}_acc0.50_top1"
    ] == pytest.approx(0.0)
    assert results[
        f"last__bbs_{expected_subset}_count_acc0.25"
    ] == 1
    other = "multiple" if expected_subset == "unique" else "unique"
    assert results[f"last__bbs_{other}_count_acc0.25"] == 0


def test_bbs_subset_metrics_merge_across_process_payloads(monkeypatch):
    evaluator = GroundingEvaluator(
        thresholds=[0.25, 0.5], topks=[1], prefixes=["last_"]
    )
    payloads = iter(
        [
            [evaluator.dets],
            [evaluator.gts],
            [evaluator.diagnostic_dets],
            [evaluator.diagnostic_gts],
            [
                {("unique", 0.25): 2, ("multiple", 0.25): 3},
                {("unique", 0.25): 5, ("multiple", 0.25): 7},
            ],
            [
                {("unique", 0.25): 4, ("multiple", 0.25): 6},
                {("unique", 0.25): 8, ("multiple", 0.25): 9},
            ],
            [evaluator.decomposition_dets],
            [evaluator.decomposition_gts],
            [evaluator.decomposition_status_counts],
            [evaluator.spacy_augmentation_dets],
            [evaluator.spacy_augmentation_gts],
            [evaluator.spacy_augmentation_counts],
            [evaluator.spacy_source_dets],
            [evaluator.spacy_source_gts],
            [evaluator.spacy_source_counts],
            [evaluator.per_layer_score_source],
            [evaluator.per_layer_has_fused_scores],
            [evaluator.score_alignment_sums],
            [evaluator.score_alignment_gts],
            [evaluator.source_choice_feature_rows],
        ]
    )
    monkeypatch.setattr(grounding_evaluator.misc, "all_gather", lambda _: next(payloads))
    monkeypatch.setattr(grounding_evaluator.misc, "is_main_process", lambda: True)

    evaluator.synchronize_between_processes()

    assert evaluator.bbs_subset_dets[("unique", 0.25)] == 7
    assert evaluator.bbs_subset_dets[("multiple", 0.25)] == 10
    assert evaluator.bbs_subset_gts[("unique", 0.25)] == 12
    assert evaluator.bbs_subset_gts[("multiple", 0.25)] == 15
