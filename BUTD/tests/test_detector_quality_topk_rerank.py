import torch

from models.detector_policy_sources import (
    _quality_topk_target_detector_rerank_scores,
)


def test_detector_reranks_only_inside_quality_topk():
    quality = torch.tensor([[0.90, 0.80, 0.70, 0.60, 0.50]])
    detector = torch.tensor([[0.10, 0.20, 0.95, 1.00, 0.00]])

    top3 = _quality_topk_target_detector_rerank_scores(
        quality, detector, candidate_k=3
    )
    top4 = _quality_topk_target_detector_rerank_scores(
        quality, detector, candidate_k=4
    )

    assert top3.argmax(dim=1).item() == 2
    assert top4.argmax(dim=1).item() == 3
    assert torch.equal(top3[0, [0, 1, 3, 4]], quality[0, [0, 1, 3, 4]])


def test_detector_ties_preserve_quality_top1():
    quality = torch.tensor([[0.90, 0.80, 0.70]])
    detector = torch.zeros_like(quality)
    scores = _quality_topk_target_detector_rerank_scores(
        quality, detector, candidate_k=3
    )
    assert scores.argmax(dim=1).item() == 0
