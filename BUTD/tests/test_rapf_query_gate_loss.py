from unittest.mock import patch

import torch

from models.losses import _rapf_gate_losses


def test_rapf_gate_loss_supervises_helpful_query_not_sample_mean():
    gate = torch.tensor([[0.2, 0.2, 0.2]], requires_grad=True)
    end_points = {
        "rapf_gate": gate,
        "rapf_base_norm": torch.tensor([[2.0, 0.0, 0.0]]),
        "rapf_structured_norm": torch.tensor([[0.0, 3.0, -1.0]]),
        "fused_scores": torch.tensor([[2.0, 1.0, 0.0]]),
        "structured_valid_mask": torch.tensor([True]),
        "base_grounding_scores": torch.tensor([[2.0, 0.0, 0.0]]),
        "structured_scores": torch.tensor([[0.0, 3.0, -1.0]]),
    }
    target_iou = torch.tensor([[0.20, 0.80, 0.10]])

    def score_source(points, source):
        return points[
            "base_grounding_scores"
            if source == "base"
            else "structured_scores"
        ]

    with patch("models.losses._target_iou_matrix", return_value=target_iou), \
            patch("models.losses._score_source", side_effect=score_source), \
            patch(
                "models.losses._dataset_not_scannet_mask",
                return_value=torch.tensor([True]),
            ):
        result = _rapf_gate_losses(end_points, indices=None, weight=1.0)

    result["loss_rapf_gate"].backward()
    # Query 1 can improve over the official BBS top query and must be opened.
    assert gate.grad[0, 1] < 0
    # The other queries cannot help and their gates must be closed.
    assert gate.grad[0, 0] > 0
    assert gate.grad[0, 2] > 0
    assert torch.allclose(
        result["dbg_rapf_gate_label_mean"], torch.tensor(1.0 / 3.0)
    )
