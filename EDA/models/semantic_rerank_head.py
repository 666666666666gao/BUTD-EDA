import torch
import torch.nn as nn
import torch.nn.functional as F


def blend_semantic_rerank_outputs(primary, auxiliary, auxiliary_weight=0.5):
    """Blend two rerank residuals evaluated on the same query features."""
    weight = float(auxiliary_weight)
    if weight < 0.0 or weight > 1.0:
        raise ValueError("auxiliary_weight must be in [0, 1]")
    primary_base = primary['semantic_rerank_base_scores']
    auxiliary_base = auxiliary['semantic_rerank_base_scores']
    if not torch.equal(primary_base, auxiliary_base):
        raise ValueError("rerank heads must use identical base scores")
    primary_residual = primary['semantic_rerank_residual']
    auxiliary_residual = auxiliary['semantic_rerank_residual']
    residual = (
        (1.0 - weight) * primary_residual
        + weight * auxiliary_residual
    )
    output = dict(primary)
    output.update({
        'semantic_rerank_primary_residual': primary_residual,
        'semantic_rerank_aux_residual': auxiliary_residual,
        'semantic_rerank_residual': residual,
        'semantic_rerank_scores': primary_base + residual,
    })
    return output


class SemanticRerankHead(nn.Module):
    """Small residual scorer for semantic-alignment query reranking."""

    def __init__(self, d_model=288, hidden_dim=128, residual_scale=0.1,
                 use_target_conditioning=False, use_threshold_head=False,
                 threshold_hidden_dim=64, threshold_residual_scale=0.25):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.use_target_conditioning = bool(use_target_conditioning)
        self.use_threshold_head = bool(use_threshold_head)
        self.threshold_residual_scale = float(threshold_residual_scale)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 9, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.conditioning_mlp = None
        if self.use_target_conditioning:
            self.conditioning_mlp = nn.Sequential(
                nn.Linear(2 * d_model + 9, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            conditioning_final = self.conditioning_mlp[-1]
            nn.init.zeros_(conditioning_final.weight)
            nn.init.zeros_(conditioning_final.bias)
        self.threshold_head = None
        if self.use_threshold_head:
            if threshold_hidden_dim <= 0:
                raise ValueError("threshold_hidden_dim must be positive")
            # Two text-conditioned query vectors plus generic geometric and
            # score summaries. The two outputs correspond to the standard
            # IoU 0.25 and 0.50 grounding thresholds.
            threshold_input_dim = 2 * d_model + 19
            self.threshold_head = nn.Sequential(
                nn.Linear(threshold_input_dim, int(threshold_hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(threshold_hidden_dim), 2),
            )
            threshold_final = self.threshold_head[-1]
            nn.init.zeros_(threshold_final.weight)
            nn.init.zeros_(threshold_final.bias)

    @staticmethod
    def _normalize_scores(scores):
        scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
        centered = scores - scores.mean(dim=1, keepdim=True)
        scale = centered.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        return centered / scale

    def forward(self, query_feats, pred_boxes, base_scores,
                quality_scores=None, fused_scores=None, target_slot=None,
                semantic_components=None, structured_scores=None,
                target_attr_scores=None, relation_anchor_scores=None,
                parse_confidence=None):
        base_scores = torch.nan_to_num(
            base_scores.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        if quality_scores is None:
            quality_norm = torch.zeros_like(base_scores)
        else:
            quality_norm = self._normalize_scores(quality_scores)
        if fused_scores is None:
            fused_norm = torch.zeros_like(base_scores)
        else:
            fused_norm = self._normalize_scores(fused_scores)

        base_norm = self._normalize_scores(base_scores)
        box_feats = torch.nan_to_num(
            pred_boxes.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        features = torch.cat(
            [
                query_feats.float(),
                box_feats,
                base_norm.unsqueeze(-1),
                quality_norm.unsqueeze(-1),
                fused_norm.unsqueeze(-1),
            ],
            dim=-1,
        )
        rerank_logits = self.mlp(features).squeeze(-1)
        conditioning_logits = torch.zeros_like(rerank_logits)
        if self.conditioning_mlp is not None:
            B, Q, D = query_feats.shape
            if target_slot is None:
                target_slot = query_feats.new_zeros(B, D)
            target_slot = torch.nan_to_num(
                target_slot.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            query_unit = F.normalize(query_feats.float(), dim=-1)
            target_unit = F.normalize(target_slot, dim=-1)
            target_expanded = target_unit.unsqueeze(1).expand(B, Q, D)

            if semantic_components is None:
                component_norm = query_feats.new_zeros(B, Q, 5).float()
            else:
                component_norm = torch.stack(
                    [
                        self._normalize_scores(semantic_components[..., idx])
                        for idx in range(5)
                    ],
                    dim=-1,
                )

            scalar_features = []
            for scores in (
                structured_scores,
                target_attr_scores,
                relation_anchor_scores,
            ):
                if scores is None:
                    scalar_features.append(base_scores.new_zeros(B, Q))
                else:
                    scalar_features.append(self._normalize_scores(scores))
            if parse_confidence is None:
                parse_feature = base_scores.new_zeros(B, Q)
            else:
                parse_feature = parse_confidence.float().reshape(B, 1).expand(B, Q)

            conditioning_features = torch.cat(
                [
                    target_expanded,
                    query_unit * target_expanded,
                    component_norm,
                    torch.stack(scalar_features, dim=-1),
                    parse_feature.unsqueeze(-1),
                ],
                dim=-1,
            )
            conditioning_logits = self.conditioning_mlp(
                conditioning_features
            ).squeeze(-1)

        residual = self.residual_scale * torch.tanh(
            rerank_logits + conditioning_logits
        )
        scores = base_scores + residual
        output = {
            'semantic_rerank_base_scores': base_scores,
            'semantic_rerank_residual': residual,
            'semantic_rerank_conditioning_logits': conditioning_logits,
            'semantic_rerank_scores': scores,
        }
        if self.threshold_head is not None:
            B, Q, D = query_feats.shape
            if target_slot is None:
                target_slot = query_feats.new_zeros(B, D)
            target_slot = torch.nan_to_num(
                target_slot.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            query_unit = F.normalize(query_feats.float(), dim=-1)
            target_unit = F.normalize(target_slot, dim=-1)
            target_interaction = (
                query_unit * target_unit.unsqueeze(1)
            )

            if semantic_components is None:
                component_norm = query_feats.new_zeros(B, Q, 5).float()
            else:
                component_norm = torch.stack([
                    self._normalize_scores(semantic_components[..., idx])
                    for idx in range(5)
                ], dim=-1)

            scalar_features = []
            for scalar_scores in (
                structured_scores,
                target_attr_scores,
                relation_anchor_scores,
            ):
                if scalar_scores is None:
                    scalar_features.append(base_scores.new_zeros(B, Q))
                else:
                    scalar_features.append(
                        self._normalize_scores(scalar_scores)
                    )
            if parse_confidence is None:
                parse_feature = base_scores.new_zeros(B, Q)
            else:
                parse_feature = parse_confidence.float().reshape(
                    B, 1
                ).expand(B, Q)
            rerank_norm = self._normalize_scores(scores)
            threshold_features = torch.cat([
                query_feats.float(),
                target_interaction,
                box_feats,
                base_norm.unsqueeze(-1),
                quality_norm.unsqueeze(-1),
                fused_norm.unsqueeze(-1),
                torch.stack(scalar_features, dim=-1),
                component_norm,
                parse_feature.unsqueeze(-1),
                rerank_norm.unsqueeze(-1),
            ], dim=-1)
            threshold_logits = self.threshold_head(
                threshold_features
            )
            threshold_residual = self.threshold_residual_scale * (
                torch.sigmoid(threshold_logits).sum(dim=-1) - 1.0
            )
            output.update({
                'semantic_threshold_logits': threshold_logits,
                'semantic_threshold_residual': threshold_residual,
            })
        return output
