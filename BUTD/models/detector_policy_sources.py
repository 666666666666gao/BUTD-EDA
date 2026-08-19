"""Deployable detector-policy score sources for source-choice training."""

import os

import torch
import torch.nn as nn


DETECTOR_POLICY_SOURCE_NAMES = (
    "detector_countboost",
    "detector_run174boost",
    "detector_countsplit",
    "detector_countsplit_lowonly",
    "detector_countsplit_guarded",
    "detector_countsplit_guarded_allcount",
    "detector_jointtight",
    "detector_strongcoarse",
    "detector_confblend035",
    "detector_confblend05",
)


DETECTOR_POLICY_SCORE_KEYS = {
    "detector_countboost": "detector_countboost_scores",
    "detector_run174boost": "detector_run174boost_scores",
    "detector_countsplit": "detector_countsplit_scores",
    "detector_countsplit_lowonly": "detector_countsplit_lowonly_scores",
    "detector_countsplit_guarded": "detector_countsplit_guarded_scores",
    "detector_countsplit_guarded_allcount": (
        "detector_countsplit_guarded_allcount_scores"
    ),
    "detector_jointtight": "detector_jointtight_scores",
    "detector_strongcoarse": "detector_strongcoarse_scores",
    "detector_confblend035": "detector_confblend035_scores",
    "detector_confblend05": "detector_confblend05_scores",
}


def box_cxcyczwhd_to_xyzxyz(boxes):
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1e4, neginf=-1e4)
    center = boxes[..., :3]
    size = boxes[..., 3:6].abs().clamp(min=1e-6)
    half_size = 0.5 * size
    return torch.cat([center - half_size, center + half_size], dim=-1)


def _pairwise_iou3d(boxes_a, boxes_b):
    min_a, max_a = boxes_a[:, :3], boxes_a[:, 3:]
    min_b, max_b = boxes_b[:, :3], boxes_b[:, 3:]
    inter_min = torch.maximum(min_a[:, None, :], min_b[None, :, :])
    inter_max = torch.minimum(max_a[:, None, :], max_b[None, :, :])
    inter_size = (inter_max - inter_min).clamp(min=0.0)
    intersection = inter_size.prod(dim=-1)
    vol_a = (max_a - min_a).clamp(min=0.0).prod(dim=-1)
    vol_b = (max_b - min_b).clamp(min=0.0).prod(dim=-1)
    union = (vol_a[:, None] + vol_b[None, :] - intersection).clamp(
        min=1e-6
    )
    return intersection / union


def _gather_top(values, top_idx):
    return torch.gather(values, 1, top_idx.unsqueeze(1)).squeeze(1)


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _detector_overlap_sources(
    pred_boxes, det_boxes, det_bbox_label_mask, det_class_ids,
    det_logits, target_cid
):
    if (
        pred_boxes is None
        or det_boxes is None
        or det_bbox_label_mask is None
        or det_class_ids is None
        or det_logits is None
        or target_cid is None
    ):
        return None
    if (
        pred_boxes.dim() != 3
        or det_boxes.dim() != 3
        or det_bbox_label_mask.dim() != 2
        or det_class_ids.dim() != 2
        or det_logits.dim() != 3
    ):
        return None
    B, Q = pred_boxes.shape[:2]
    if (
        det_boxes.shape[0] != B
        or det_bbox_label_mask.shape[0] != B
        or det_class_ids.shape[:2] != det_bbox_label_mask.shape
        or det_logits.shape[:2] != det_bbox_label_mask.shape
    ):
        return None

    device = pred_boxes.device
    dtype = pred_boxes.dtype
    class_scores = torch.zeros(B, Q, device=device, dtype=dtype)
    conf_scores = torch.zeros(B, Q, device=device, dtype=dtype)
    det_count = torch.zeros(B, device=device, dtype=torch.long)
    target_cid = target_cid.to(device=device).view(-1)

    pred_xyz = box_cxcyczwhd_to_xyzxyz(pred_boxes.float())
    det_xyz = box_cxcyczwhd_to_xyzxyz(det_boxes.float().to(device=device))
    det_mask = det_bbox_label_mask.to(device=device).bool()
    det_class_ids = det_class_ids.to(device=device).long()
    det_logits = det_logits.to(device=device).float()

    for bid in range(B):
        if bid >= target_cid.shape[0]:
            continue
        cid = int(target_cid[bid].detach().cpu().item())
        if cid < 0 or cid >= det_logits.shape[-1]:
            continue
        valid = det_mask[bid]
        if not bool(valid.any().detach().item()):
            continue
        pair_ious = _pairwise_iou3d(pred_xyz[bid], det_xyz[bid, valid])
        det_classes = det_class_ids[bid, valid]
        target_conf = det_logits[bid, valid].softmax(dim=-1)[:, cid]
        class_match = det_classes == cid
        if bool(class_match.any().detach().item()):
            class_scores[bid] = pair_ious[:, class_match].max(dim=1).values
        conf_match = class_match & (target_conf > 0.30)
        det_count[bid] = conf_match.long().sum()
        if bool(conf_match.any().detach().item()):
            conf_scores[bid] = pair_ious[:, conf_match].max(dim=1).values

    return class_scores, conf_scores, det_count


def build_detector_policy_features(
    pred_boxes, quality_scores, det_boxes=None, det_bbox_label_mask=None,
    det_class_ids=None, det_logits=None, target_cid=None
):
    """Build detector-policy feature tensors available at inference time."""
    if quality_scores is None:
        return None
    quality_scores = torch.nan_to_num(
        quality_scores.float(), nan=0.0, posinf=1e4, neginf=-1e4
    )
    overlap_sources = _detector_overlap_sources(
        pred_boxes=pred_boxes,
        det_boxes=det_boxes,
        det_bbox_label_mask=det_bbox_label_mask,
        det_class_ids=det_class_ids,
        det_logits=det_logits,
        target_cid=target_cid,
    )
    if overlap_sources is None:
        return None
    class_scores, conf_scores, det_count = overlap_sources
    class_scores = class_scores.to(
        device=quality_scores.device, dtype=quality_scores.dtype
    )
    conf_scores = conf_scores.to(
        device=quality_scores.device, dtype=quality_scores.dtype
    )
    det_count = det_count.to(device=quality_scores.device)
    detector_support = torch.maximum(class_scores, conf_scores)
    low_count = (det_count <= 2).to(dtype=quality_scores.dtype)
    return {
        "quality_scores": quality_scores,
        "class_scores": class_scores,
        "conf_scores": conf_scores,
        "det_count": det_count,
        "low_count": low_count,
        "detector_support": detector_support,
    }


class DetectorPolicyAdapterHead(nn.Module):
    """Learn small context-conditioned deltas on top of detector-policy priors."""

    def __init__(
        self, context_dim=0, hidden_dim=32,
        prior_weights=(1.0, 0.12, 0.25, 0.18), delta_scale=0.25
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.delta_scale = float(delta_scale)
        prior = torch.tensor(prior_weights, dtype=torch.float32).view(1, 4)
        self.register_buffer("prior_weights", prior)
        self.global_delta = nn.Parameter(torch.zeros(1, 4))
        if self.context_dim > 0:
            self.context_mlp = nn.Sequential(
                nn.Linear(self.context_dim, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), 4),
            )
            nn.init.zeros_(self.context_mlp[-1].weight)
            nn.init.zeros_(self.context_mlp[-1].bias)
        else:
            self.context_mlp = None

    def forward(self, features, context=None):
        quality_scores = torch.nan_to_num(
            features["quality_scores"].float(), nan=0.0, posinf=1e4,
            neginf=-1e4
        )
        class_scores = torch.nan_to_num(
            features["class_scores"].float(), nan=0.0, posinf=1e4,
            neginf=-1e4
        ).to(device=quality_scores.device, dtype=quality_scores.dtype)
        conf_scores = torch.nan_to_num(
            features["conf_scores"].float(), nan=0.0, posinf=1e4,
            neginf=-1e4
        ).to(device=quality_scores.device, dtype=quality_scores.dtype)
        low_count = features.get("low_count", None)
        if low_count is None:
            low_count = (features["det_count"].to(quality_scores.device) <= 2)
        low_count = low_count.to(
            device=quality_scores.device, dtype=quality_scores.dtype
        ).view(-1, 1)

        B = quality_scores.shape[0]
        delta = self.global_delta.to(
            device=quality_scores.device, dtype=quality_scores.dtype
        ).expand(B, -1)
        if self.context_mlp is not None and context is not None:
            if context.dim() != 2 or context.shape != (B, self.context_dim):
                raise ValueError(
                    "context must have shape {}".format((B, self.context_dim))
                )
            context = context.detach().float().to(device=quality_scores.device)
            delta = delta + self.context_mlp(context).to(dtype=quality_scores.dtype)
        weights = self.prior_weights.to(
            device=quality_scores.device, dtype=quality_scores.dtype
        ) + self.delta_scale * torch.tanh(delta)
        scores = (
            weights[:, 0:1] * quality_scores
            + weights[:, 1:2] * class_scores
            + weights[:, 2:3] * conf_scores
            + low_count * weights[:, 3:4] * class_scores
        )
        return {
            "scores": scores,
            "weights": weights,
        }


def build_detector_policy_score_sources(
    pred_boxes, quality_scores, det_boxes=None, det_bbox_label_mask=None,
    det_class_ids=None, det_logits=None, target_cid=None
):
    """Build Run179/180/183/184-style deployable detector policy scores.

    These sources use only model outputs and detector predictions available at
    inference time. GT is not used here; GT is used later only by the selector
    loss to decide which source should have been selected during training.
    """
    features = build_detector_policy_features(
        pred_boxes=pred_boxes,
        quality_scores=quality_scores,
        det_boxes=det_boxes,
        det_bbox_label_mask=det_bbox_label_mask,
        det_class_ids=det_class_ids,
        det_logits=det_logits,
        target_cid=target_cid,
    )
    if features is None:
        return {}

    quality_scores = features["quality_scores"]
    class_scores = features["class_scores"]
    conf_scores = features["conf_scores"]
    low_count = features["low_count"]
    low_count_q = low_count.unsqueeze(1)
    detector_support = features["detector_support"]

    countboost_scores = (
        quality_scores
        + 0.12 * class_scores
        + 0.25 * conf_scores
        + low_count_q * 0.18 * class_scores
    )
    run174boost_scores = (
        quality_scores
        + 0.15 * class_scores
        + 0.25 * conf_scores
        + low_count_q * 0.15 * class_scores
    )

    class_weight = torch.where(
        low_count > 0.5,
        quality_scores.new_full(low_count.shape, 0.30),
        quality_scores.new_full(low_count.shape, 0.10),
    ).unsqueeze(1)
    conf_weight = quality_scores.new_full((quality_scores.shape[0], 1), 0.25)
    countsplit_scores = (
        quality_scores + class_weight * class_scores + conf_weight * conf_scores
    )
    countsplit_lowonly_scores = (
        quality_scores
        + low_count_q * (0.30 * class_scores + 0.25 * conf_scores)
    )
    confblend035_scores = quality_scores + 0.35 * conf_scores
    confblend05_scores = quality_scores + 0.50 * conf_scores

    joint_scores = countboost_scores
    tight_base_scores = (
        quality_scores + 0.15 * class_scores + 0.25 * conf_scores
    )
    tight_base_top = tight_base_scores.argmax(dim=1)
    tight_base_support = _gather_top(detector_support, tight_base_top)
    tight_boosted_scores = tight_base_scores + 0.20 * class_scores
    tight_boosted_top = tight_boosted_scores.argmax(dim=1)
    tight_boosted_support = _gather_top(detector_support, tight_boosted_top)
    tight_support_improvement = tight_boosted_support - tight_base_support
    tight_allow_boost = (
        (tight_boosted_top == tight_base_top)
        | (tight_support_improvement >= 0.25)
    )
    tight_scores = (
        tight_base_scores
        + tight_allow_boost.float().unsqueeze(1) * 0.20 * class_scores
    )
    joint_top = joint_scores.argmax(dim=1)
    tight_top = tight_scores.argmax(dim=1)
    joint_support = _gather_top(detector_support, joint_top)
    tight_support = _gather_top(detector_support, tight_top)
    joint_quality = _gather_top(quality_scores, joint_top)
    tight_quality = _gather_top(quality_scores, tight_top)
    use_tight = (
        (low_count > 0.5)
        & (tight_support >= joint_support + 0.50)
        & ((joint_quality - tight_quality) <= 1.00)
    )
    jointtight_scores = torch.where(
        use_tight.unsqueeze(1), tight_scores, joint_scores
    )
    lowonly_top = countsplit_lowonly_scores.argmax(dim=1)
    lowonly_support = _gather_top(detector_support, lowonly_top)
    lowonly_quality = _gather_top(quality_scores, lowonly_top)
    jointtight_top = jointtight_scores.argmax(dim=1)
    jointtight_support = _gather_top(detector_support, jointtight_top)
    jointtight_quality = _gather_top(quality_scores, jointtight_top)
    guarded_support_drop = _env_float(
        "NMV2_DETECTOR_COUNTSPLIT_GUARD_SUPPORT_DROP", 0.10
    )
    guarded_quality_drop = _env_float(
        "NMV2_DETECTOR_COUNTSPLIT_GUARD_QUALITY_DROP", 0.10
    )
    use_countsplit_lowonly = (
        (low_count > 0.5)
        & (lowonly_top != jointtight_top)
        & (lowonly_support >= jointtight_support - guarded_support_drop)
        & ((jointtight_quality - lowonly_quality) <= guarded_quality_drop)
    )
    countsplit_guarded_scores = torch.where(
        use_countsplit_lowonly.unsqueeze(1),
        countsplit_lowonly_scores,
        jointtight_scores,
    )
    use_countsplit_lowonly_allcount = (
        (lowonly_top != jointtight_top)
        & (lowonly_support >= jointtight_support - guarded_support_drop)
        & ((jointtight_quality - lowonly_quality) <= guarded_quality_drop)
    )
    countsplit_guarded_allcount_scores = torch.where(
        use_countsplit_lowonly_allcount.unsqueeze(1),
        countsplit_lowonly_scores,
        jointtight_scores,
    )

    strongcoarse_scores = (
        quality_scores + 0.25 * class_scores + 0.35 * conf_scores
    )
    strongcoarse_top = strongcoarse_scores.argmax(dim=1)
    strongcoarse_support = _gather_top(detector_support, strongcoarse_top)
    strongcoarse_quality = _gather_top(quality_scores, strongcoarse_top)
    use_strongcoarse = (
        (low_count > 0.5)
        & (strongcoarse_support >= joint_support + 0.50)
        & ((joint_quality - strongcoarse_quality) <= 1.00)
    )
    selected_strongcoarse_scores = torch.where(
        use_strongcoarse.unsqueeze(1), strongcoarse_scores, joint_scores
    )

    return {
        "detector_countboost": countboost_scores,
        "detector_run174boost": run174boost_scores,
        "detector_countsplit": countsplit_scores,
        "detector_countsplit_lowonly": countsplit_lowonly_scores,
        "detector_countsplit_guarded": countsplit_guarded_scores,
        "detector_countsplit_guarded_allcount": (
            countsplit_guarded_allcount_scores
        ),
        "detector_jointtight": jointtight_scores,
        "detector_strongcoarse": selected_strongcoarse_scores,
        "detector_confblend035": confblend035_scores,
        "detector_confblend05": confblend05_scores,
    }
