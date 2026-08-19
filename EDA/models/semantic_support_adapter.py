import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSupportAdapter(nn.Module):
    """Rerank semantic queries with support from valid BUTD boxes."""

    def __init__(self, overlap_weight=0.6075, position_weight=0.1075,
                 overlap_power=0.5, use_learned_gate=False,
                 gate_hidden_dim=16, gate_max=2.0,
                 gate_use_query_features=False, query_dim=288):
        super().__init__()
        if overlap_power <= 0:
            raise ValueError("overlap_power must be positive")
        if gate_hidden_dim <= 0:
            raise ValueError("gate_hidden_dim must be positive")
        if gate_max <= 1.0:
            raise ValueError("gate_max must be greater than one")
        self.overlap_weight = float(overlap_weight)
        self.position_weight = float(position_weight)
        self.overlap_power = float(overlap_power)
        self.use_learned_gate = bool(use_learned_gate)
        self.gate_max = float(gate_max)
        self.gate_use_query_features = bool(gate_use_query_features)
        self.query_dim = int(query_dim)
        if self.query_dim <= 0:
            raise ValueError("query_dim must be positive")
        self.support_gate = None
        if self.use_learned_gate:
            gate_input_dim = 16
            if self.gate_use_query_features:
                # Raw query, fixed-support query, target slot, their delta,
                # and three pairwise interactions.  These remain generic
                # decoder representations rather than dataset metadata.
                gate_input_dim += 7 * self.query_dim
            self.support_gate = nn.Sequential(
                nn.Linear(gate_input_dim, int(gate_hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(gate_hidden_dim), 1),
            )
            final = self.support_gate[-1]
            nn.init.zeros_(final.weight)
            initial_probability = 1.0 / self.gate_max
            initial_logit = torch.log(torch.tensor(
                initial_probability / (1.0 - initial_probability)
            )).item()
            nn.init.constant_(final.bias, initial_logit)

    @staticmethod
    def _normalize(scores):
        scores = torch.nan_to_num(
            scores.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        centered = scores - scores.mean(dim=1, keepdim=True)
        scale = centered.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp(min=1e-6)
        return centered / scale

    @staticmethod
    def _box_ends(boxes):
        centers = boxes[..., :3]
        sizes = boxes[..., 3:].clamp(min=1e-6)
        return centers - 0.5 * sizes, centers + 0.5 * sizes

    @staticmethod
    def _top_margin(scores):
        top = torch.topk(scores, min(2, scores.shape[1]), dim=1).values
        if top.shape[1] == 1:
            return top[:, 0]
        return top[:, 0] - top[:, 1]

    @classmethod
    def _gate_features(cls, semantic, detector, position, fixed_scores):
        """Build scale-invariant, task-agnostic confidence summaries."""
        semantic_index = semantic.argmax(dim=1)
        detector_index = detector.argmax(dim=1)
        position_index = position.argmax(dim=1)
        fixed_index = fixed_scores.argmax(dim=1)

        def selected(scores, index):
            return scores.gather(1, index.unsqueeze(1)).squeeze(1)

        features = torch.stack([
            semantic.max(dim=1).values,
            cls._top_margin(semantic),
            detector.max(dim=1).values,
            cls._top_margin(detector),
            position.max(dim=1).values,
            cls._top_margin(position),
            (semantic * detector).mean(dim=1),
            (semantic * position).mean(dim=1),
            selected(detector, semantic_index),
            selected(position, semantic_index),
            selected(semantic, detector_index),
            selected(semantic, position_index),
            semantic_index.eq(detector_index).float(),
            semantic_index.eq(position_index).float(),
            cls._top_margin(fixed_scores),
            semantic_index.ne(fixed_index).float(),
        ], dim=1)
        return torch.nan_to_num(
            features.detach(), nan=0.0, posinf=0.0, neginf=0.0
        )

    @staticmethod
    def _selected_query_features(query_feats, indices):
        gather_index = indices.view(-1, 1, 1).expand(
            -1, 1, query_feats.shape[-1]
        )
        return query_feats.gather(1, gather_index).squeeze(1)

    def _representation_gate_features(self, query_feats, target_slot,
                                      semantic, fixed_scores):
        if query_feats is None:
            raise ValueError(
                "query_feats are required when gate_use_query_features=True"
            )
        if query_feats.dim() != 3 or query_feats.shape[:2] != semantic.shape:
            raise ValueError("query_feats must match semantic scores [B, Q, D]")
        if query_feats.shape[-1] != self.query_dim:
            raise ValueError("query_feats last dimension must match query_dim")

        raw_index = semantic.argmax(dim=1)
        fixed_index = fixed_scores.argmax(dim=1)
        raw_query = self._selected_query_features(query_feats, raw_index)
        fixed_query = self._selected_query_features(query_feats, fixed_index)
        if target_slot is None:
            target_slot = torch.zeros_like(raw_query)
        if target_slot.shape != raw_query.shape:
            raise ValueError("target_slot must have shape [B, D]")

        def normalize(features):
            return F.layer_norm(features.float(), (features.shape[-1],))

        raw_query = normalize(raw_query)
        fixed_query = normalize(fixed_query)
        target_slot = normalize(target_slot)
        features = torch.cat([
            raw_query,
            fixed_query,
            target_slot,
            fixed_query - raw_query,
            raw_query * fixed_query,
            raw_query * target_slot,
            fixed_query * target_slot,
        ], dim=1)
        return torch.nan_to_num(
            features.detach(), nan=0.0, posinf=0.0, neginf=0.0
        )

    @classmethod
    def _max_detector_iou(cls, query_boxes, detector_boxes,
                          detector_valid_mask):
        if query_boxes.dim() != 3 or query_boxes.shape[-1] != 6:
            raise ValueError("query_boxes must have shape [B, Q, 6]")
        if detector_boxes.dim() != 3 or detector_boxes.shape[-1] != 6:
            raise ValueError("detector_boxes must have shape [B, D, 6]")
        if detector_valid_mask.shape != detector_boxes.shape[:2]:
            raise ValueError(
                "detector_valid_mask must match detector_boxes [B, D]"
            )
        if query_boxes.shape[0] != detector_boxes.shape[0]:
            raise ValueError("query and detector batch sizes must match")

        query_boxes = torch.nan_to_num(
            query_boxes.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        detector_boxes = torch.nan_to_num(
            detector_boxes.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        query_min, query_max = cls._box_ends(query_boxes)
        detector_min, detector_max = cls._box_ends(detector_boxes)

        intersection_size = (
            torch.minimum(query_max.unsqueeze(2), detector_max.unsqueeze(1))
            - torch.maximum(query_min.unsqueeze(2), detector_min.unsqueeze(1))
        ).clamp(min=0.0)
        intersection = intersection_size.prod(dim=-1)
        query_volume = (query_max - query_min).prod(dim=-1).unsqueeze(2)
        detector_volume = (
            detector_max - detector_min
        ).prod(dim=-1).unsqueeze(1)
        union = (query_volume + detector_volume - intersection).clamp(min=1e-6)
        pairwise_iou = torch.nan_to_num(
            intersection / union, nan=0.0, posinf=0.0, neginf=0.0
        )

        valid = detector_valid_mask.to(
            device=pairwise_iou.device, dtype=torch.bool
        )
        pairwise_iou = pairwise_iou.masked_fill(~valid.unsqueeze(1), -1.0)
        return pairwise_iou.max(dim=2).values.clamp(min=0.0)

    def forward(self, semantic_scores, query_boxes, detector_boxes,
                detector_valid_mask, position_scores, query_feats=None,
                target_slot=None):
        if semantic_scores.shape != query_boxes.shape[:2]:
            raise ValueError("semantic_scores must match query_boxes [B, Q]")
        if position_scores.shape != semantic_scores.shape:
            raise ValueError("position_scores must match semantic_scores [B, Q]")

        detector_support = self._max_detector_iou(
            query_boxes, detector_boxes, detector_valid_mask
        )
        powered_support = detector_support.pow(self.overlap_power)
        semantic_normalized = self._normalize(semantic_scores)
        detector_normalized = self._normalize(powered_support)
        position_normalized = self._normalize(position_scores)
        support_residual = (
            self.overlap_weight * detector_normalized
            + self.position_weight * position_normalized
        )
        fixed_scores = semantic_normalized + support_residual
        gate = semantic_scores.new_ones(semantic_scores.shape[0], 1).float()
        if self.support_gate is not None:
            features = self._gate_features(
                semantic_normalized,
                detector_normalized,
                position_normalized,
                fixed_scores,
            )
            if self.gate_use_query_features:
                representation_features = self._representation_gate_features(
                    query_feats,
                    target_slot,
                    semantic_normalized,
                    fixed_scores,
                )
                features = torch.cat([
                    features, representation_features
                ], dim=1)
            gate = self.gate_max * torch.sigmoid(self.support_gate(features))
        scores = semantic_normalized + gate * support_residual
        return {
            'semantic_support_scores': scores,
            'semantic_support_raw_scores': semantic_normalized,
            'semantic_support_fixed_scores': fixed_scores,
            'semantic_support_residual': support_residual,
            'semantic_detector_support': detector_support,
            'semantic_support_gate': gate,
        }
