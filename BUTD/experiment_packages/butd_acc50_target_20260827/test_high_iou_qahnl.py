#!/usr/bin/env python3
"""CPU contract test for the Stage-1/Stage-2 high-IoU QAHNL policy."""

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.losses import _qahnl_losses, _target_iou_matrix  # noqa: E402


def _scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _assert_close(actual, expected, *, atol=2e-5, name="value"):
    actual = _scalar(actual)
    if abs(actual - expected) > atol:
        raise AssertionError(
            f"{name}: expected {expected:.8f}, got {actual:.8f}"
        )


def _make_end_points():
    # Concentric cubes inside a unit target have IoU equal to their volume.
    nested_ious = (0.60, 0.40, 0.20)
    nested_sides = [iou ** (1.0 / 3.0) for iou in nested_ious]
    pred_sizes = torch.tensor(
        [[
            [nested_sides[0]] * 3,
            [nested_sides[1]] * 3,
            [nested_sides[2]] * 3,
            [1.0, 1.0, 1.0],
        ]],
        dtype=torch.float32,
    )
    pred_centers = torch.zeros((1, 4, 3), dtype=torch.float32)
    pred_centers[0, 3, 0] = 3.0
    fused_scores = torch.tensor(
        [[0.0, 5.0, 4.0, 3.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    return {
        "fused_scores": fused_scores,
        "last_center": pred_centers,
        "last_pred_size": pred_sizes,
        "center_label": torch.zeros((1, 1, 3), dtype=torch.float32),
        "size_gts": torch.ones((1, 1, 3), dtype=torch.float32),
        "box_label_mask": torch.ones((1, 1), dtype=torch.bool),
        "dataset": ["scanrefer"],
        "decomposition_error_flags_count": torch.zeros(1),
    }


def _config(*, hungarian_rescue):
    return {
        "score_source": "fused",
        "pos_iou_thresh": 0.50,
        "neg_iou_thresh": 0.25,
        "topk_iou_pos": 1,
        "num_hard_neg": 1,
        "negative_sampling": "hardest",
        "disable_top_iou_pos": True,
        "disable_hungarian_pos_rescue": not hungarian_rescue,
        "margin_base": 0.20,
        "margin_iou_lambda": 0.50,
        "margin_min": 0.05,
        "margin_max": 0.50,
        "temperature": 1.0,
        "temperature_max": 6.0,
        "loss_weight": 0.30,
    }


def _assert_disabled_rescue_contract():
    end_points = _make_end_points()
    # Deliberately match the ambiguous-IoU query to the target. With rescue
    # disabled, this query must remain ignored.
    indices = [(
        torch.tensor([1], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
    )]
    ious = _target_iou_matrix(end_points)
    expected = torch.tensor([[0.60, 0.40, 0.20, 0.0]])
    if not torch.allclose(ious, expected, atol=2e-5, rtol=0.0):
        raise AssertionError(f"unexpected synthetic IoUs: {ious.tolist()}")

    losses = _qahnl_losses(
        end_points,
        indices,
        _config(hungarian_rescue=False),
    )
    _assert_close(
        losses["dbg_qahnl_positive_query_ratio"],
        0.25,
        name="positive_query_ratio",
    )
    _assert_close(
        losses["dbg_qahnl_negative_query_ratio"],
        0.25,
        name="negative_query_ratio",
    )
    _assert_close(
        losses["dbg_qahnl_valid_batch_ratio"],
        1.0,
        name="valid_batch_ratio",
    )
    _assert_close(losses["dbg_qahnl_pos_iou"], 0.60, name="positive_iou")
    _assert_close(losses["dbg_qahnl_neg_iou"], 0.20, name="negative_iou")
    _assert_close(
        losses["dbg_qahnl_ambiguous_ignore_ratio"],
        1.0,
        name="ambiguous_ignore_ratio",
    )
    _assert_close(
        losses["dbg_qahnl_ambiguous_as_negative_ratio"],
        0.0,
        name="ambiguous_as_negative_ratio",
    )
    _assert_close(
        losses["dbg_qahnl_top_iou_pos_rescue_disabled"],
        1.0,
        name="top_iou_rescue_disabled",
    )
    _assert_close(
        losses["dbg_qahnl_hungarian_pos_rescue_disabled"],
        1.0,
        name="hungarian_rescue_disabled",
    )

    losses["loss_qahnl"].backward()
    grad = end_points["fused_scores"].grad[0]
    if not grad[0].item() < 0:
        raise AssertionError(f"positive query gradient must be negative: {grad}")
    if not grad[2].item() > 0:
        raise AssertionError(f"hard-negative gradient must be positive: {grad}")
    _assert_close(grad[1], 0.0, atol=1e-8, name="ambiguous_query_gradient")
    _assert_close(grad[3], 0.0, atol=1e-8, name="easy_negative_gradient")


def _assert_hungarian_rescue_changes_contract():
    end_points = _make_end_points()
    indices = [(
        torch.tensor([1], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
    )]
    losses = _qahnl_losses(
        end_points,
        indices,
        _config(hungarian_rescue=True),
    )
    _assert_close(
        losses["dbg_qahnl_positive_query_ratio"],
        0.50,
        name="rescued_positive_query_ratio",
    )
    _assert_close(
        losses["dbg_qahnl_hungarian_pos_rescue_disabled"],
        0.0,
        name="hungarian_rescue_enabled",
    )
    losses["loss_qahnl"].backward()
    grad = end_points["fused_scores"].grad[0]
    if not grad[1].item() < 0:
        raise AssertionError(f"rescued Hungarian query must be positive: {grad}")
    if not grad[2].item() > 0:
        raise AssertionError(f"hard-negative gradient must be positive: {grad}")
    _assert_close(grad[0], 0.0, atol=1e-8, name="lower_score_positive_gradient")
    _assert_close(grad[3], 0.0, atol=1e-8, name="easy_negative_gradient")


def main():
    torch.set_num_threads(1)
    _assert_disabled_rescue_contract()
    _assert_hungarian_rescue_changes_contract()
    print("HIGH_IOU_QAHNL_CONTRACT_PASS")


if __name__ == "__main__":
    main()
