"""Deployable detector-policy score sources for source-choice training."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    "detector_quality_top2_target_rerank",
    "detector_quality_top3_target_rerank",
    "detector_quality_top4_target_rerank",
    "detector_quality_top5_target_rerank",
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
    "detector_quality_top2_target_rerank": (
        "detector_quality_top2_target_rerank_scores"
    ),
    "detector_quality_top3_target_rerank": (
        "detector_quality_top3_target_rerank_scores"
    ),
    "detector_quality_top4_target_rerank": (
        "detector_quality_top4_target_rerank_scores"
    ),
    "detector_quality_top5_target_rerank": (
        "detector_quality_top5_target_rerank_scores"
    ),
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


def _quality_topk_target_detector_rerank_scores(
    quality_scores, detector_scores, candidate_k
):
    """Rerank QualityHead's top-K with deployable detector overlap.

    Only the selected top-1 score is promoted; all other query scores retain
    their QualityHead ordering.  A tiny quality tie-break reproduces the
    evaluator's stable rerank behavior when detector overlaps are equal.
    """
    quality_scores = torch.nan_to_num(
        quality_scores.float(), nan=0.0, posinf=1e4, neginf=-1e4
    )
    detector_scores = torch.nan_to_num(
        detector_scores.float(), nan=0.0, posinf=1e4, neginf=-1e4
    ).to(device=quality_scores.device, dtype=quality_scores.dtype)
    if quality_scores.shape != detector_scores.shape:
        raise ValueError("quality and detector scores must have identical shape")
    k = max(1, min(int(candidate_k), quality_scores.shape[1]))
    candidate_idx = quality_scores.argsort(dim=1, descending=True)[:, :k]
    candidate_detector = torch.gather(detector_scores, 1, candidate_idx)
    # Match GroundingEvaluator._rerank_candidate_indices exactly: select the
    # quality top-K first, then sort that restricted pool by detector overlap.
    best_pos = candidate_detector.argsort(
        dim=1, descending=True
    )[:, :1]
    selected_idx = torch.gather(candidate_idx, 1, best_pos)
    scores = quality_scores.clone()
    promoted = quality_scores.max(dim=1, keepdim=True).values + 1.0
    scores.scatter_(1, selected_idx, promoted)
    return scores


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
    matched_det_boxes = pred_boxes.detach().clone()
    matched_det_support = torch.zeros(B, Q, device=device, dtype=dtype)
    matched_det_confidence = torch.zeros(B, Q, device=device, dtype=dtype)
    matched_det_valid = torch.zeros(B, Q, device=device, dtype=torch.bool)
    second_matched_det_boxes = pred_boxes.detach().clone()
    second_matched_det_support = torch.zeros(
        B, Q, device=device, dtype=dtype
    )
    second_matched_det_confidence = torch.zeros(
        B, Q, device=device, dtype=dtype
    )
    second_matched_det_valid = torch.zeros(
        B, Q, device=device, dtype=torch.bool
    )
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
        # Deployable geometric match used by the box-calibration head.  The
        # matching rule consumes detector predictions and the text-derived
        # target class only; no GT class or GT box is available here.
        overlap_conf = pair_ious * target_conf.unsqueeze(0)
        best_det = overlap_conf.argmax(dim=1)
        matched_det_boxes[bid] = det_boxes[bid, valid][best_det].to(
            device=device, dtype=dtype
        )
        matched_det_support[bid] = pair_ious.gather(
            1, best_det.unsqueeze(1)
        ).squeeze(1).to(dtype=dtype)
        matched_det_confidence[bid] = target_conf[best_det].to(dtype=dtype)
        matched_det_valid[bid] = True
        if overlap_conf.shape[1] >= 2:
            top2_det = overlap_conf.topk(2, dim=1).indices
            second_det = top2_det[:, 1]
            second_matched_det_boxes[bid] = det_boxes[
                bid, valid
            ][second_det].to(device=device, dtype=dtype)
            second_matched_det_support[bid] = pair_ious.gather(
                1, second_det.unsqueeze(1)
            ).squeeze(1).to(dtype=dtype)
            second_matched_det_confidence[bid] = target_conf[
                second_det
            ].to(dtype=dtype)
            second_matched_det_valid[bid] = True
        class_match = det_classes == cid
        if bool(class_match.any().detach().item()):
            class_scores[bid] = pair_ious[:, class_match].max(dim=1).values
        conf_match = class_match & (target_conf > 0.30)
        det_count[bid] = conf_match.long().sum()
        if bool(conf_match.any().detach().item()):
            conf_scores[bid] = pair_ious[:, conf_match].max(dim=1).values

    return (
        class_scores,
        conf_scores,
        det_count,
        matched_det_boxes,
        matched_det_support,
        matched_det_confidence,
        matched_det_valid,
        second_matched_det_boxes,
        second_matched_det_support,
        second_matched_det_confidence,
        second_matched_det_valid,
    )


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
    (
        class_scores,
        conf_scores,
        det_count,
        matched_det_boxes,
        matched_det_support,
        matched_det_confidence,
        matched_det_valid,
        second_matched_det_boxes,
        second_matched_det_support,
        second_matched_det_confidence,
        second_matched_det_valid,
    ) = overlap_sources
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
        "pred_boxes": pred_boxes,
        "matched_det_boxes": matched_det_boxes,
        "matched_det_support": matched_det_support,
        "matched_det_confidence": matched_det_confidence,
        "matched_det_valid": matched_det_valid,
        "second_matched_det_boxes": second_matched_det_boxes,
        "second_matched_det_support": second_matched_det_support,
        "second_matched_det_confidence": second_matched_det_confidence,
        "second_matched_det_valid": second_matched_det_valid,
    }


class DetectorPolicyAdapterHead(nn.Module):
    """Learn small context-conditioned deltas on top of detector-policy priors."""

    def __init__(
        self, context_dim=0, hidden_dim=32, query_dim=288, candidate_k=5,
        prior_weights=(1.0, 0.12, 0.25, 0.18), delta_scale=0.25,
        extended_geometry_actions=False,
        geometry_extension_head=False,
        rank2_rescue_head=False,
        rank2_override_threshold=0.0,
        boundary_refiner=False,
        boundary_refiner_scale=0.25,
        tier_pair_rescue_head=False,
        tier_pair_candidate_k=2,
        tier_pair_override_threshold=0.0,
        alignment_rescue_head=False,
        alignment_candidate_k=16,
        alignment_override_threshold=0.0,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.query_dim = int(query_dim)
        self.candidate_k = int(candidate_k)
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
        # Candidate-aware residual branch.  It is initialized to exact fused
        # parity, so a freshly attached head cannot degrade the base ranking.
        scalar_dim = 20
        self.query_mlp = nn.Sequential(
            nn.Linear(self.query_dim, int(hidden_dim)), nn.ReLU()
        )
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, int(hidden_dim)), nn.ReLU()
        )
        self.rerank_head = nn.Sequential(
            nn.Linear(10 * int(hidden_dim), 2 * int(hidden_dim)), nn.ReLU(),
            nn.Linear(2 * int(hidden_dim), 4),
        )
        nn.init.zeros_(self.rerank_head[-1].weight)
        nn.init.zeros_(self.rerank_head[-1].bias)
        # Fixed, interpretable box-blend actions.  The new head is kept
        # separate from the already-trained semantic reranker so old E6
        # checkpoints load without changing their score-selection behavior.
        # The extended set keeps the legacy six actions as an exact prefix;
        # checkpoint loading can therefore preserve every learned legacy row
        # and append only detector-heavy actions.
        geometry_actions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        if bool(extended_geometry_actions):
            geometry_actions += [0.75, 1.0]
        self.register_buffer(
            "geometry_actions",
            torch.tensor(geometry_actions),
        )
        geometry_scalar_dim = 9
        self.geometry_scalar_mlp = nn.Sequential(
            nn.Linear(geometry_scalar_dim, int(hidden_dim)), nn.ReLU()
        )
        self.geometry_head = nn.Sequential(
            nn.Linear(3 * int(hidden_dim), int(hidden_dim)), nn.ReLU(),
            nn.Linear(int(hidden_dim), len(geometry_actions)),
        )
        nn.init.zeros_(self.geometry_head[-1].weight)
        nn.init.zeros_(self.geometry_head[-1].bias)
        with torch.no_grad():
            self.geometry_head[-1].bias[3] = 1.0
        if bool(geometry_extension_head):
            self.register_buffer(
                "geometry_extension_actions",
                torch.tensor([0.75, 1.0]),
            )
            self.geometry_extension_head = nn.Sequential(
                nn.Linear(3 * int(hidden_dim), int(hidden_dim)), nn.ReLU(),
                nn.Linear(int(hidden_dim), 2),
            )
            nn.init.zeros_(self.geometry_extension_head[-1].weight)
            nn.init.constant_(self.geometry_extension_head[-1].bias, -1.0)
        else:
            self.geometry_extension_head = None
        self.geometry_default_action = 3
        self.geometry_override_margin = 0.5
        self.rank2_override_threshold = float(rank2_override_threshold)
        if bool(rank2_rescue_head):
            self.register_buffer(
                "rank2_actions",
                torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
            )
            # Support/confidence, second-vs-query geometry, and the relative
            # displacement from the accepted top-1 detector match.
            rank2_scalar_dim = 17
            self.rank2_geometry_mlp = nn.Sequential(
                nn.Linear(rank2_scalar_dim, int(hidden_dim)), nn.ReLU()
            )
            self.rank2_rescue_head = nn.Sequential(
                nn.Linear(3 * int(hidden_dim), int(hidden_dim)), nn.ReLU(),
                nn.Linear(int(hidden_dim), 5),
            )
            nn.init.zeros_(self.rank2_rescue_head[-1].weight)
            nn.init.constant_(self.rank2_rescue_head[-1].bias, -4.0)
        else:
            self.rank2_rescue_head = None
        self.boundary_refiner_scale = float(boundary_refiner_scale)
        if bool(boundary_refiner):
            # A query-conditioned continuous correction on top of the
            # accepted detector calibration.  The zero-initialized output
            # gives exact checkpoint parity before focused training.
            self.boundary_refiner_head = nn.Sequential(
                nn.Linear(3 * int(hidden_dim), int(hidden_dim)), nn.ReLU(),
                nn.Linear(int(hidden_dim), 6),
            )
            nn.init.zeros_(self.boundary_refiner_head[-1].weight)
            nn.init.zeros_(self.boundary_refiner_head[-1].bias)
        else:
            self.boundary_refiner_head = None
        self.tier_pair_candidate_k = int(tier_pair_candidate_k)
        self.tier_pair_override_threshold = float(
            tier_pair_override_threshold
        )
        if bool(tier_pair_rescue_head):
            # Compare the accepted query against only its strongest current
            # alternatives.  The pair representation uses frozen,
            # language-conditioned query features.  Its benefit gate starts
            # closed, so adding this head to an old checkpoint is exactly
            # behavior preserving before focused training.
            self.tier_pair_rescue_head = nn.Sequential(
                nn.Linear(6 * int(hidden_dim), 2 * int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(2 * int(hidden_dim), 4),
            )
            nn.init.zeros_(self.tier_pair_rescue_head[-1].weight)
            nn.init.zeros_(self.tier_pair_rescue_head[-1].bias)
            nn.init.constant_(self.tier_pair_rescue_head[-1].bias[3], -4.0)
        else:
            self.tier_pair_rescue_head = None
        self.alignment_candidate_k = int(alignment_candidate_k)
        self.alignment_override_threshold = float(
            alignment_override_threshold
        )
        if bool(alignment_rescue_head):
            # Directly score raw detector proposals instead of restricting
            # rescue to the semantic adapter's query shortlist.  Each raw
            # proposal is paired with its nearest decoder query and with the
            # current incumbent query.  The final gate starts closed, giving
            # exact legacy checkpoint behavior until explicitly trained.
            self.alignment_det_mlp = nn.Sequential(
                nn.Linear(self.query_dim, int(hidden_dim)), nn.ReLU()
            )
            self.alignment_query_mlp = nn.Sequential(
                nn.Linear(self.query_dim, int(hidden_dim)), nn.ReLU()
            )
            self.alignment_text_mlp = nn.Sequential(
                nn.Linear(self.query_dim, int(hidden_dim)), nn.ReLU()
            )
            self.alignment_scalar_mlp = nn.Sequential(
                nn.Linear(25, int(hidden_dim)), nn.ReLU()
            )
            self.alignment_rescue_head = nn.Sequential(
                nn.Linear(5 * int(hidden_dim), 2 * int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(2 * int(hidden_dim), 3),
            )
            nn.init.zeros_(self.alignment_rescue_head[-1].weight)
            nn.init.zeros_(self.alignment_rescue_head[-1].bias)
            nn.init.constant_(self.alignment_rescue_head[-1].bias[2], -4.0)
        else:
            self.alignment_det_mlp = None
            self.alignment_query_mlp = None
            self.alignment_text_mlp = None
            self.alignment_scalar_mlp = None
            self.alignment_rescue_head = None

    @staticmethod
    def _standardize(scores):
        scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        return (scores - scores.mean(dim=1, keepdim=True)) / (
            scores.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        )

    @staticmethod
    def _rank_feature(scores):
        order = scores.argsort(dim=1, descending=True)
        ranks = torch.empty_like(order)
        base = torch.arange(scores.shape[1], device=scores.device).view(1, -1)
        ranks.scatter_(1, order, base.expand_as(order))
        return 1.0 - ranks.float() / max(scores.shape[1] - 1, 1)

    @staticmethod
    def _alignment_box_delta(candidate_boxes, reference_boxes):
        reference_size = reference_boxes[..., 3:].abs().clamp(min=1e-3)
        center_delta = (
            candidate_boxes[..., :3] - reference_boxes[..., :3]
        ) / reference_size
        size_delta = torch.log(
            (candidate_boxes[..., 3:].abs() + 1e-4)
            / (reference_size + 1e-4)
        ).clamp(min=-4.0, max=4.0)
        return torch.cat([center_delta, size_delta], dim=-1)

    def _forward_alignment_rescue(
        self, features, scores, calibrated_boxes, query_feats
    ):
        det_boxes = features.get("det_boxes", None)
        det_mask = features.get("det_bbox_label_mask", None)
        det_class_ids = features.get("det_class_ids", None)
        det_logits = features.get("det_logits", None)
        target_cid = features.get("target_cid", None)
        detected_feats = features.get("detected_feats", None)
        text_context = features.get("alignment_text_context", None)
        pred_boxes = features.get("pred_boxes", None)
        required = (
            det_boxes, det_mask, det_class_ids, det_logits, target_cid,
            detected_feats, pred_boxes,
        )
        if any(value is None for value in required):
            return None
        if (
            det_boxes.dim() != 3
            or det_mask.dim() != 2
            or det_class_ids.dim() != 2
            or det_logits.dim() != 3
            or detected_feats.dim() != 3
            or pred_boxes.dim() != 3
        ):
            return None

        device = scores.device
        dtype = scores.dtype
        B, D = det_boxes.shape[:2]
        Q = scores.shape[1]
        if (
            det_mask.shape != (B, D)
            or det_class_ids.shape != (B, D)
            or det_logits.shape[:2] != (B, D)
            or detected_feats.shape[:2] != (B, D)
            or detected_feats.shape[-1] != self.query_dim
            or pred_boxes.shape[:2] != (B, Q)
            or query_feats.shape[:2] != (B, Q)
        ):
            return None

        det_boxes = det_boxes.detach().to(device=device, dtype=dtype)
        valid = det_mask.detach().to(device=device).bool()
        det_class_ids = det_class_ids.detach().to(device=device).long()
        det_logits = det_logits.detach().to(device=device).float()
        target_cid = target_cid.detach().to(device=device).long().view(-1)
        if target_cid.shape[0] != B or det_logits.shape[-1] <= 0:
            return None
        class_count = det_logits.shape[-1]
        cid_valid = (target_cid >= 0) & (target_cid < class_count)
        safe_cid = target_cid.clamp(min=0, max=class_count - 1)
        det_prob = det_logits.softmax(dim=-1)
        target_confidence = det_prob.gather(
            2, safe_cid.view(B, 1, 1).expand(-1, D, 1)
        ).squeeze(-1)
        target_confidence = torch.where(
            cid_valid.unsqueeze(1), target_confidence,
            torch.zeros_like(target_confidence),
        )
        predicted_confidence = det_prob.max(dim=-1).values
        class_match = (
            det_class_ids == target_cid.unsqueeze(1)
        ) & cid_valid.unsqueeze(1) & valid

        # Hard detector class matches are retained, while target-confidence
        # top-K proposals recover detector boxes whose argmax class is wrong.
        candidate_mask = class_match.clone()
        k = max(1, min(self.alignment_candidate_k, D))
        masked_confidence = target_confidence.masked_fill(
            ~valid, torch.finfo(target_confidence.dtype).min
        )
        top_confidence = torch.topk(
            masked_confidence, k, dim=1
        ).indices
        confidence_candidates = torch.zeros_like(valid)
        confidence_candidates.scatter_(1, top_confidence, True)
        candidate_mask = (
            candidate_mask
            | (
                confidence_candidates
                & valid
                & cid_valid.unsqueeze(1)
            )
        )

        pred_boxes_detached = pred_boxes.detach().to(
            device=device, dtype=dtype
        )
        det_xyz = box_cxcyczwhd_to_xyzxyz(det_boxes.float())
        pred_xyz = box_cxcyczwhd_to_xyzxyz(pred_boxes_detached.float())
        nearest_query = torch.zeros(B, D, device=device, dtype=torch.long)
        nearest_iou = torch.zeros(B, D, device=device, dtype=dtype)
        with torch.no_grad():
            for bid in range(B):
                if not bool(valid[bid].any().detach().item()):
                    continue
                pair_iou = _pairwise_iou3d(det_xyz[bid], pred_xyz[bid])
                best_iou, best_query = pair_iou.max(dim=1)
                nearest_query[bid] = best_query
                nearest_iou[bid] = best_iou.to(dtype=dtype)

        query_feats = query_feats.detach().float().to(device=device)
        nearest_query_feats = torch.gather(
            query_feats,
            1,
            nearest_query.unsqueeze(-1).expand(-1, -1, self.query_dim),
        )
        incumbent_query = scores.detach().argmax(dim=1)
        incumbent_query_feats = torch.gather(
            query_feats,
            1,
            incumbent_query.view(B, 1, 1).expand(-1, 1, self.query_dim),
        ).expand(-1, D, -1)
        nearest_boxes = torch.gather(
            pred_boxes_detached,
            1,
            nearest_query.unsqueeze(-1).expand(-1, -1, 6),
        )
        incumbent_boxes = calibrated_boxes.detach().to(
            device=device, dtype=dtype
        )
        selected_incumbent_boxes = torch.gather(
            incumbent_boxes,
            1,
            incumbent_query.view(B, 1, 1).expand(-1, 1, 6),
        ).expand(-1, D, -1)

        score_z = self._standardize(scores.detach())
        score_rank = self._rank_feature(scores.detach())
        nearest_score_z = torch.gather(score_z, 1, nearest_query)
        nearest_score_rank = torch.gather(score_rank, 1, nearest_query)
        confidence_rank = self._rank_feature(masked_confidence)
        det_box_z = torch.stack([
            self._standardize(det_boxes[..., index])
            for index in range(6)
        ], dim=-1)
        valid_float = valid.to(dtype=dtype).unsqueeze(-1)
        scalar = torch.cat([
            target_confidence.to(dtype=dtype).unsqueeze(-1),
            predicted_confidence.to(dtype=dtype).unsqueeze(-1),
            class_match.to(dtype=dtype).unsqueeze(-1),
            nearest_iou.unsqueeze(-1),
            nearest_score_z.unsqueeze(-1),
            nearest_score_rank.unsqueeze(-1),
            confidence_rank.to(dtype=dtype).unsqueeze(-1),
            det_box_z * valid_float,
            self._alignment_box_delta(
                det_boxes, selected_incumbent_boxes
            ) * valid_float,
            self._alignment_box_delta(
                det_boxes, nearest_boxes
            ) * valid_float,
        ], dim=-1)

        detected_feats = detected_feats.detach().float().to(device=device)
        if text_context is None:
            text_context = torch.zeros(
                B, self.query_dim, device=device, dtype=torch.float32
            )
        else:
            text_context = text_context.detach().float().to(device=device)
            if text_context.shape != (B, self.query_dim):
                return None
        det_hidden = self.alignment_det_mlp(detected_feats)
        nearest_hidden = self.alignment_query_mlp(nearest_query_feats)
        incumbent_hidden = self.alignment_query_mlp(
            incumbent_query_feats
        )
        text_hidden = self.alignment_text_mlp(text_context).unsqueeze(1)
        text_hidden = text_hidden.expand(-1, D, -1)
        scalar_hidden = self.alignment_scalar_mlp(scalar.float())
        alignment_logits = self.alignment_rescue_head(torch.cat([
            det_hidden,
            nearest_hidden,
            incumbent_hidden,
            text_hidden,
            scalar_hidden,
        ], dim=-1))
        hit25_logits = alignment_logits[..., 0]
        hit50_logits = alignment_logits[..., 1]
        rescue_logits = alignment_logits[..., 2]
        decision_logits = (
            rescue_logits + 0.50 * hit50_logits + 0.10 * hit25_logits
        )
        masked_decision = decision_logits.masked_fill(
            ~candidate_mask, torch.finfo(decision_logits.dtype).min
        )
        best_decision, alignment_candidate = masked_decision.max(dim=1)
        batch_index = torch.arange(B, device=device)
        best_rescue_logit = rescue_logits[
            batch_index, alignment_candidate
        ]
        has_candidate = candidate_mask.any(dim=1)
        alignment_gate = (
            has_candidate
            & (best_rescue_logit >= self.alignment_override_threshold)
        )
        selected_det_box = det_boxes[batch_index, alignment_candidate]
        aligned_boxes = incumbent_boxes.clone()
        old_incumbent_box = aligned_boxes[batch_index, incumbent_query]
        aligned_boxes[batch_index, incumbent_query] = torch.where(
            alignment_gate.unsqueeze(1),
            selected_det_box,
            old_incumbent_box,
        )
        return {
            "scores": scores,
            "calibrated_boxes": aligned_boxes,
            "alignment_logits": alignment_logits,
            "alignment_candidate_mask": candidate_mask,
            "alignment_candidate_boxes": det_boxes,
            "alignment_incumbent_query": incumbent_query,
            "alignment_incumbent_boxes": incumbent_boxes,
            "alignment_nearest_query": nearest_query,
            "alignment_candidate": alignment_candidate,
            "alignment_gate": alignment_gate,
            "alignment_best_logit": best_decision,
            "alignment_best_rescue_logit": best_rescue_logit,
        }

    def _forward_tier_pair_rescue(
        self, scores, calibrated_boxes, local_features
    ):
        """Conservatively promote a higher predicted IoU-tier query.

        This forward path is deployable: it consumes only current scores and
        frozen query/scalar representations.  IoU tier labels are constructed
        later in the training loss and never enter this method.
        """
        B, Q = scores.shape
        k = max(1, min(self.tier_pair_candidate_k, Q))
        candidate_query = torch.topk(
            scores.detach(), k, dim=1
        ).indices
        candidate_local = torch.gather(
            local_features.detach(),
            1,
            candidate_query.unsqueeze(-1).expand(
                -1, -1, local_features.shape[-1]
            ),
        )
        incumbent_query = scores.detach().argmax(dim=1)
        incumbent_local = torch.gather(
            local_features.detach(),
            1,
            incumbent_query.view(B, 1, 1).expand(
                -1, 1, local_features.shape[-1]
            ),
        ).expand_as(candidate_local)
        pair_features = torch.cat([
            candidate_local,
            incumbent_local,
            candidate_local - incumbent_local,
        ], dim=-1)
        tier_pair_logits = self.tier_pair_rescue_head(pair_features)
        tier_probability = F.softmax(tier_pair_logits[..., :3], dim=-1)
        expected_utility = (
            tier_probability[..., 1]
            + 5.0 * tier_probability[..., 2]
        )
        incumbent_utility = expected_utility[:, :1]
        benefit_logits = tier_pair_logits[..., 3]
        decision_logits = (
            benefit_logits
            + 0.50 * (expected_utility - incumbent_utility)
        )
        alternative_mask = candidate_query != incumbent_query.unsqueeze(1)
        masked_decision = decision_logits.masked_fill(
            ~alternative_mask,
            torch.finfo(decision_logits.dtype).min,
        )
        best_decision, best_offset = masked_decision.max(dim=1)
        batch_index = torch.arange(B, device=scores.device)
        tier_pair_query = candidate_query[batch_index, best_offset]
        best_benefit_logit = benefit_logits[batch_index, best_offset]
        has_alternative = alternative_mask.any(dim=1)
        tier_pair_gate = (
            has_alternative
            & (best_benefit_logit >= self.tier_pair_override_threshold)
            & (best_decision >= self.tier_pair_override_threshold)
        )

        tier_pair_scores = scores.clone()
        old_selected_score = tier_pair_scores.gather(
            1, tier_pair_query.unsqueeze(1)
        ).squeeze(1)
        promoted_score = torch.where(
            tier_pair_gate,
            tier_pair_scores.max(dim=1).values + scores.new_tensor(1e-3),
            old_selected_score,
        )
        tier_pair_scores.scatter_(
            1, tier_pair_query.unsqueeze(1), promoted_score.unsqueeze(1)
        )
        return {
            "scores": tier_pair_scores,
            "calibrated_boxes": calibrated_boxes,
            "tier_pair_logits": tier_pair_logits,
            "tier_pair_candidate_query": candidate_query,
            "tier_pair_incumbent_query": incumbent_query,
            "tier_pair_query": tier_pair_query,
            "tier_pair_gate": tier_pair_gate,
            "tier_pair_best_logit": best_decision,
            "tier_pair_best_benefit_logit": best_benefit_logit,
        }

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
        fused = features.get("fused_scores", None)
        query_feats = features.get("query_feats", None)
        if fused is None or query_feats is None:
            return {"scores": scores, "weights": weights}

        fused = torch.nan_to_num(fused.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        sources = [
            fused,
            features.get("contrastive_scores", fused),
            quality_scores,
        ]
        candidate_mask = torch.zeros_like(fused, dtype=torch.bool)
        k = max(1, min(self.candidate_k, fused.shape[1]))
        for source in sources:
            candidate_mask.scatter_(1, torch.topk(source.detach(), k, dim=1).indices, True)

        base = features.get("base_scores", fused)
        structured = features.get("structured_scores", fused)
        contrastive = features.get("contrastive_scores", fused)
        pred_boxes = features.get("pred_boxes", None)
        if pred_boxes is None:
            pred_boxes = fused.new_zeros(fused.shape[0], fused.shape[1], 6)
        box_features = torch.stack([
            self._standardize(pred_boxes[..., index]) for index in range(6)
        ], dim=-1)
        scalar = torch.stack([
            self._standardize(fused), self._standardize(base),
            self._standardize(structured), self._standardize(contrastive),
            self._standardize(quality_scores), self._rank_feature(fused),
            self._rank_feature(contrastive), self._rank_feature(quality_scores),
            class_scores, conf_scores,
            features["detector_support"].to(fused),
            low_count.expand_as(fused),
            (class_scores > 0).to(fused.dtype),
            (conf_scores > 0).to(fused.dtype),
        ], dim=-1)
        scalar = torch.cat([scalar, box_features], dim=-1)
        q_hidden = self.query_mlp(query_feats.detach().float())
        s_hidden = self.scalar_mlp(scalar)
        local = torch.cat([q_hidden, s_hidden], dim=-1)
        local = F.dropout(local, p=0.20, training=self.training)
        mask_f = candidate_mask.to(local.dtype).unsqueeze(-1)
        context_hidden = (local * mask_f).sum(dim=1, keepdim=True) / (
            mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        )
        context_hidden = context_hidden.expand_as(local)
        fallback_query = fused.argmax(dim=1)
        fallback_hidden = local.gather(
            1,
            fallback_query.view(B, 1, 1).expand(-1, 1, local.shape[-1]),
        ).expand_as(local)
        logits = self.rerank_head(torch.cat([
            local,
            fallback_hidden,
            local - fallback_hidden,
            context_hidden,
            local - context_hidden,
        ], dim=-1))
        delta = self.delta_scale * torch.tanh(logits[..., 0])
        hit25_logits = logits[..., 1]
        hit50_logits = logits[..., 2]
        rescue_logits = logits[..., 3]

        # Directly optimize and consume the threshold-aligned hit@.50 head.
        # A tiny Full-score tie-break preserves exact Full behavior when the
        # new head is freshly initialized (all hit50 logits are zero).
        decision_logits = (
            hit50_logits + 1e-3 * self._standardize(fused)
        )
        masked_decision = decision_logits.masked_fill(
            ~candidate_mask, torch.finfo(decision_logits.dtype).min
        )
        rescue_query = masked_decision.argmax(dim=1)
        rescue_gate = rescue_query != fallback_query
        rerank_scores = fused.clone()
        fallback_max = fused.max(dim=1).values
        selected_score = torch.where(
            rescue_gate,
            fallback_max + fused.new_tensor(1e-3),
            fused.gather(1, rescue_query.unsqueeze(1)).squeeze(1),
        )
        rerank_scores.scatter_(
            1, rescue_query.unsqueeze(1), selected_score.unsqueeze(1)
        )
        output = {
            "scores": rerank_scores,
            "weights": weights,
            "candidate_mask": candidate_mask,
            "rerank_delta": rerank_scores - fused,
            "candidate_delta": delta,
            "hit25_logits": hit25_logits,
            "hit50_logits": hit50_logits,
            "rescue_logits": rescue_logits,
            "rescue_gate": rescue_gate,
            "rescue_query": rescue_query,
            "fallback_query": fallback_query,
        }
        matched_boxes = features.get("matched_det_boxes", None)
        matched_support = features.get("matched_det_support", None)
        matched_confidence = features.get("matched_det_confidence", None)
        matched_valid = features.get("matched_det_valid", None)
        if (
            matched_boxes is not None
            and matched_support is not None
            and matched_confidence is not None
            and matched_valid is not None
        ):
            pred_boxes = pred_boxes.to(dtype=fused.dtype)
            matched_boxes = matched_boxes.to(
                device=fused.device, dtype=fused.dtype
            )
            matched_support = matched_support.to(
                device=fused.device, dtype=fused.dtype
            )
            matched_confidence = matched_confidence.to(
                device=fused.device, dtype=fused.dtype
            )
            matched_valid = matched_valid.to(device=fused.device).bool()
            pred_size = pred_boxes[..., 3:].abs().clamp(min=1e-3)
            center_delta = (
                matched_boxes[..., :3] - pred_boxes[..., :3]
            ) / pred_size
            size_ratio = torch.log(
                (matched_boxes[..., 3:].abs() + 1e-4)
                / (pred_size + 1e-4)
            ).clamp(min=-4.0, max=4.0)
            geometry_scalar = torch.cat([
                matched_support.unsqueeze(-1),
                matched_confidence.unsqueeze(-1),
                matched_valid.to(fused.dtype).unsqueeze(-1),
                center_delta,
                size_ratio,
            ], dim=-1)
            geometry_hidden = self.geometry_scalar_mlp(geometry_scalar)
            # Detach the old semantic representation: geometry-only training
            # cannot perturb the already accepted E6 query selector.
            geometry_input = torch.cat([
                local.detach(), geometry_hidden
            ], dim=-1)
            geometry_logits = self.geometry_head(geometry_input)
            all_geometry_actions = self.geometry_actions
            geometry_extension_logits = None
            if self.geometry_extension_head is not None:
                geometry_extension_logits = self.geometry_extension_head(
                    geometry_input.detach()
                )
                geometry_logits = torch.cat([
                    geometry_logits, geometry_extension_logits
                ], dim=-1)
                all_geometry_actions = torch.cat([
                    self.geometry_actions,
                    self.geometry_extension_actions,
                ], dim=0)
            best_geometry_logit, best_geometry_action = (
                geometry_logits.max(dim=-1)
            )
            default_geometry_logit = geometry_logits[
                ..., self.geometry_default_action
            ]
            allow_geometry_override = (
                (best_geometry_action != self.geometry_default_action)
                & (
                    best_geometry_logit - default_geometry_logit
                    >= self.geometry_override_margin
                )
            )
            geometry_action = torch.where(
                allow_geometry_override,
                best_geometry_action,
                torch.full_like(
                    best_geometry_action, self.geometry_default_action
                ),
            )
            geometry_alpha = all_geometry_actions.to(
                device=fused.device, dtype=fused.dtype
            )[geometry_action]
            geometry_enabled = (
                matched_valid & (matched_support >= 0.30)
            )
            geometry_alpha = torch.where(
                geometry_enabled, geometry_alpha,
                torch.zeros_like(geometry_alpha),
            )
            geometry_candidates = (
                pred_boxes.unsqueeze(2)
                + all_geometry_actions.to(
                    device=fused.device, dtype=fused.dtype
                ).view(1, 1, -1, 1)
                * (matched_boxes - pred_boxes).unsqueeze(2)
            )
            calibrated_boxes = (
                pred_boxes
                + geometry_alpha.unsqueeze(-1)
                * (matched_boxes - pred_boxes)
            )
            output.update({
                "geometry_logits": geometry_logits,
                "geometry_action": geometry_action,
                "geometry_alpha": geometry_alpha,
                "geometry_support": matched_support,
                "geometry_enabled": geometry_enabled,
                "geometry_override": allow_geometry_override,
                "geometry_candidate_boxes": geometry_candidates,
                "calibrated_boxes": calibrated_boxes,
            })
            if geometry_extension_logits is not None:
                output["geometry_extension_logits"] = (
                    geometry_extension_logits
                )
            second_boxes = features.get("second_matched_det_boxes", None)
            second_support = features.get(
                "second_matched_det_support", None
            )
            second_confidence = features.get(
                "second_matched_det_confidence", None
            )
            second_valid = features.get("second_matched_det_valid", None)
            if (
                self.rank2_rescue_head is not None
                and second_boxes is not None
                and second_support is not None
                and second_confidence is not None
                and second_valid is not None
            ):
                second_boxes = second_boxes.to(
                    device=fused.device, dtype=fused.dtype
                )
                second_support = second_support.to(
                    device=fused.device, dtype=fused.dtype
                )
                second_confidence = second_confidence.to(
                    device=fused.device, dtype=fused.dtype
                )
                second_valid = second_valid.to(device=fused.device).bool()
                second_center_delta = (
                    second_boxes[..., :3] - pred_boxes[..., :3]
                ) / pred_size
                second_size_ratio = torch.log(
                    (second_boxes[..., 3:].abs() + 1e-4)
                    / (pred_size + 1e-4)
                ).clamp(min=-4.0, max=4.0)
                top1_size = matched_boxes[..., 3:].abs().clamp(min=1e-3)
                second_from_top1_center = (
                    second_boxes[..., :3] - matched_boxes[..., :3]
                ) / pred_size
                second_from_top1_size = torch.log(
                    (second_boxes[..., 3:].abs() + 1e-4)
                    / (top1_size + 1e-4)
                ).clamp(min=-4.0, max=4.0)
                rank2_scalar = torch.cat([
                    second_support.unsqueeze(-1),
                    second_confidence.unsqueeze(-1),
                    second_valid.to(fused.dtype).unsqueeze(-1),
                    second_center_delta,
                    second_size_ratio,
                    second_from_top1_center,
                    second_from_top1_size,
                    (second_support - matched_support).unsqueeze(-1),
                    (second_confidence - matched_confidence).unsqueeze(-1),
                ], dim=-1)
                rank2_hidden = self.rank2_geometry_mlp(rank2_scalar)
                rank2_logits = self.rank2_rescue_head(torch.cat([
                    local.detach(), rank2_hidden
                ], dim=-1))
                rank2_enabled = (
                    second_valid
                    & (second_support >= 0.05)
                    & candidate_mask
                )
                rank2_candidate_mask = rank2_enabled.unsqueeze(-1).expand(
                    -1, -1, self.rank2_actions.numel()
                )
                rank2_candidate_boxes = (
                    pred_boxes.unsqueeze(2)
                    + self.rank2_actions.to(
                        device=fused.device, dtype=fused.dtype
                    ).view(1, 1, -1, 1)
                    * (second_boxes - pred_boxes).unsqueeze(2)
                )
                masked_rank2_logits = rank2_logits.masked_fill(
                    ~rank2_candidate_mask,
                    torch.finfo(rank2_logits.dtype).min,
                )
                flat_logits = masked_rank2_logits.reshape(B, -1)
                best_rank2_logit, best_rank2_flat = flat_logits.max(dim=1)
                num_actions = self.rank2_actions.numel()
                rank2_query = torch.div(
                    best_rank2_flat, num_actions, rounding_mode="floor"
                )
                rank2_action = best_rank2_flat.remainder(num_actions)
                rank2_gate = (
                    best_rank2_logit >= self.rank2_override_threshold
                ) & rank2_candidate_mask.reshape(B, -1).any(dim=1)
                batch_index = torch.arange(B, device=fused.device)
                rank2_box = rank2_candidate_boxes[
                    batch_index, rank2_query, rank2_action
                ]
                rank2_scores = rerank_scores.clone()
                old_rank2_score = rank2_scores.gather(
                    1, rank2_query.unsqueeze(1)
                ).squeeze(1)
                promoted_rank2_score = torch.where(
                    rank2_gate,
                    rank2_scores.max(dim=1).values + fused.new_tensor(1e-3),
                    old_rank2_score,
                )
                rank2_scores.scatter_(
                    1,
                    rank2_query.unsqueeze(1),
                    promoted_rank2_score.unsqueeze(1),
                )
                rank2_calibrated_boxes = calibrated_boxes.clone()
                old_rank2_box = rank2_calibrated_boxes[
                    batch_index, rank2_query
                ]
                rank2_calibrated_boxes[
                    batch_index, rank2_query
                ] = torch.where(
                    rank2_gate.unsqueeze(1), rank2_box, old_rank2_box
                )
                output.update({
                    "scores": rank2_scores,
                    "calibrated_boxes": rank2_calibrated_boxes,
                    "rank2_logits": rank2_logits,
                    "rank2_candidate_mask": rank2_candidate_mask,
                    "rank2_candidate_boxes": rank2_candidate_boxes,
                    "rank2_incumbent_query": rescue_query,
                    "rank2_incumbent_boxes": calibrated_boxes,
                    "rank2_query": rank2_query,
                    "rank2_action": rank2_action,
                    "rank2_gate": rank2_gate,
                    "rank2_best_logit": best_rank2_logit,
                })
            if self.boundary_refiner_head is not None:
                pre_refined_boxes = output["calibrated_boxes"]
                raw_refinement = self.boundary_refiner_head(
                    geometry_input.detach()
                )
                bounded_refinement = self.boundary_refiner_scale * torch.tanh(
                    raw_refinement
                )
                reference_size = pre_refined_boxes[..., 3:].abs().clamp(
                    min=1e-3
                )
                refined_center = (
                    pre_refined_boxes[..., :3]
                    + bounded_refinement[..., :3] * reference_size
                )
                refined_size = (
                    reference_size
                    * torch.exp(bounded_refinement[..., 3:])
                ).clamp(min=1e-3)
                refined_boxes = torch.cat([
                    refined_center, refined_size
                ], dim=-1)
                output.update({
                    "pre_refined_calibrated_boxes": pre_refined_boxes,
                    "boundary_refiner_raw": raw_refinement,
                    "boundary_refiner_delta": (
                        refined_boxes - pre_refined_boxes
                    ),
                    "calibrated_boxes": refined_boxes,
                })
        if self.tier_pair_rescue_head is not None:
            tier_pair_output = self._forward_tier_pair_rescue(
                scores=output["scores"],
                calibrated_boxes=output.get("calibrated_boxes", pred_boxes),
                local_features=local,
            )
            output.update(tier_pair_output)
        if self.alignment_rescue_head is not None:
            alignment_output = self._forward_alignment_rescue(
                features=features,
                scores=output["scores"],
                calibrated_boxes=output.get("calibrated_boxes", pred_boxes),
                query_feats=query_feats,
            )
            if alignment_output is not None:
                output.update(alignment_output)
        return output


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
    quality_topk_target_rerank = {
        candidate_k: _quality_topk_target_detector_rerank_scores(
            quality_scores, class_scores, candidate_k
        )
        for candidate_k in (2, 3, 4, 5)
    }

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
        "detector_quality_top2_target_rerank": (
            quality_topk_target_rerank[2]
        ),
        "detector_quality_top3_target_rerank": (
            quality_topk_target_rerank[3]
        ),
        "detector_quality_top4_target_rerank": (
            quality_topk_target_rerank[4]
        ),
        "detector_quality_top5_target_rerank": (
            quality_topk_target_rerank[5]
        ),
    }
