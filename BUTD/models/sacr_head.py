"""SACR: Structured Anchor-Compositional Reasoning head."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SACRHead(nn.Module):
    """Compute structured target/attribute/relation scores without fusion."""

    def __init__(self, d_model=288, hidden_dim=288,
                 top_m_targets=32, top_k_anchors=16, geo_dim=16,
                 disable_relation=False, disable_target_attr=False,
                 anchor_aggregation='soft', global_residual_weight=0.1,
                 fixed_residual_alpha=None):
        super().__init__()
        if geo_dim < 0:
            raise ValueError("geo_dim must be non-negative")
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.top_m_targets = top_m_targets
        self.top_k_anchors = top_k_anchors
        self.geo_dim = geo_dim
        self.use_geometry = geo_dim > 0
        self.disable_relation = bool(disable_relation)
        self.disable_target_attr = bool(disable_target_attr)
        if anchor_aggregation not in ('soft', 'hard'):
            raise ValueError("anchor_aggregation must be 'soft' or 'hard'")
        self.anchor_aggregation = anchor_aggregation
        self.global_residual_weight = float(global_residual_weight)
        if fixed_residual_alpha is not None and not (
            0.0 <= float(fixed_residual_alpha) <= 1.0
        ):
            raise ValueError("fixed_residual_alpha must be in [0, 1] or None")
        self.fixed_residual_alpha = (
            None
            if fixed_residual_alpha is None
            else float(fixed_residual_alpha)
        )

        self.target_attr_mlp = nn.Sequential(
            nn.Linear(d_model * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.anchor_mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        rel_input_dim = d_model * 3 + (geo_dim if self.use_geometry else 0)
        self.rel_pair_mlp = nn.Sequential(
            nn.Linear(rel_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        if self.use_geometry:
            self.geo_encoder = nn.Sequential(
                nn.Linear(11, geo_dim),
                nn.ReLU(),
            )
        else:
            self.geo_encoder = None

    def forward(self, query_feats, pred_boxes, base_scores, slot_dict,
                global_only_mask=None, weak_generic_target_mask=None):
        with torch.cuda.amp.autocast(enabled=False):
            query_feats = query_feats.float()
            pred_boxes = pred_boxes.float()
            base_scores = base_scores.float()
            B, Q, D = query_feats.shape
            device = query_feats.device

            global_slot = slot_dict['global_slot'].float()
            target_slot = slot_dict['target_slot'].float()
            attr_slot = slot_dict['attr_slot'].float()
            rel_slots = slot_dict['rel_slots'].float()
            anchor_slots = slot_dict['anchor_slots'].float()
            slot_mask = slot_dict['slot_mask'].to(device=device).bool()
            coverage = slot_dict.get('coverage_stats', {})
            has_target = coverage.get(
                'has_target',
                torch.ones(B, device=device, dtype=torch.bool)
            ).to(device=device).bool()

            if global_only_mask is None:
                global_only_mask = torch.zeros(B, device=device, dtype=torch.bool)
            else:
                global_only_mask = global_only_mask.to(device=device).bool()
            if weak_generic_target_mask is None:
                weak_generic_target_mask = torch.zeros(B, device=device, dtype=torch.bool)
            else:
                weak_generic_target_mask = weak_generic_target_mask.to(device=device).bool()

            structured_valid = has_target & (~global_only_mask)
            valid_f = structured_valid.float().unsqueeze(1)
            weak_f = weak_generic_target_mask.float().unsqueeze(1).unsqueeze(2)

            global_expanded = global_slot.unsqueeze(1).expand(B, Q, D)
            if self.disable_target_attr:
                target_attr_scores = torch.zeros_like(base_scores)
            else:
                target_for_entity = target_slot.unsqueeze(1).expand(B, Q, D)
                target_for_entity = target_for_entity * (1.0 - weak_f)
                attr_expanded = attr_slot.unsqueeze(1).expand(B, Q, D)
                target_attr_scores = self.target_attr_mlp(
                    torch.cat(
                        [query_feats, target_for_entity, attr_expanded, global_expanded],
                        dim=-1,
                    )
                ).squeeze(-1)
                if self.global_residual_weight != 0.0:
                    global_scores = self.global_mlp(
                        torch.cat([query_feats, global_expanded], dim=-1)
                    ).squeeze(-1)
                    target_attr_scores = (
                        target_attr_scores
                        + self.global_residual_weight * global_scores
                    )
                target_attr_scores = target_attr_scores * valid_f

            M = min(self.top_m_targets, Q)
            top_m_indices = torch.topk(base_scores + target_attr_scores, M, dim=1).indices
            if self.disable_relation:
                relation_anchor_scores = torch.zeros_like(target_attr_scores)
                p_anchor = self._uniform_anchor_probs(
                    B, rel_slots.shape[1], min(self.top_k_anchors, Q), device
                )
                relation_active_ratio = query_feats.new_tensor(0.0)
            else:
                relation_anchor_scores, p_anchor = self._relation_anchor_scores(
                    query_feats, pred_boxes, top_m_indices,
                    rel_slots, anchor_slots, slot_mask
                )
                relation_active_ratio = (
                    slot_mask.float().mean()
                    if slot_mask.numel() > 0
                    else query_feats.new_tensor(0.0)
                )
            relation_anchor_scores = relation_anchor_scores * valid_f
            structured_scores = target_attr_scores + relation_anchor_scores
            sacr_residual_scores = None
            if self.fixed_residual_alpha is not None:
                sacr_residual_scores = self._fixed_residual_scores(
                    base_scores,
                    structured_scores,
                    self.fixed_residual_alpha,
                )

            eps = 1e-8
            anchor_entropy = -(p_anchor * (p_anchor + eps).log()).sum(dim=-1)
            active_count = slot_mask.float().sum(dim=1).clamp(min=1.0)
            anchor_entropy_mean = (
                anchor_entropy * slot_mask.float()
            ).sum(dim=1) / active_count
            anchor_top1_mass = (
                p_anchor.max(dim=-1).values * slot_mask.float()
            ).sum(dim=1) / active_count

        return {
            'structured_scores': structured_scores,
            'sacr_residual_scores': sacr_residual_scores,
            'target_attr_scores': target_attr_scores,
            'relation_anchor_scores': relation_anchor_scores,
            'anchor_entropy': anchor_entropy_mean,
            'anchor_top1_mass': anchor_top1_mass,
            'relation_active_ratio': relation_active_ratio,
            'structured_valid_mask': structured_valid,
            'weak_generic_target_mask': weak_generic_target_mask,
            'global_only_mask': global_only_mask,
        }

    @staticmethod
    def _normalize_scores(scores):
        scores = torch.nan_to_num(
            scores.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        centered = scores - scores.mean(dim=1, keepdim=True)
        std = centered.pow(2).mean(dim=1, keepdim=True).clamp(
            min=1e-6
        ).sqrt()
        return torch.nan_to_num(
            centered / std, nan=0.0, posinf=0.0, neginf=0.0
        )

    @classmethod
    def _fixed_residual_scores(cls, base_scores, structured_scores, alpha):
        """Non-learned SACR residual injection; no RAPF parameters involved."""
        base_norm = cls._normalize_scores(base_scores)
        struct_norm = cls._normalize_scores(structured_scores)
        scores = base_norm + float(alpha) * (struct_norm - base_norm)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        # BBS masks unrelated objects by multiplication with zero. Preserve
        # ranking while keeping every valid source score strictly positive.
        floor = scores.min(dim=1, keepdim=True).values
        return scores - floor + 1e-6

    def _relation_anchor_scores(self, query_feats, pred_boxes, top_m_indices,
                                rel_slots, anchor_slots, slot_mask):
        B, Q, D = query_feats.shape
        K = rel_slots.shape[1]
        M = top_m_indices.shape[1]
        K_anc = min(self.top_k_anchors, Q)

        anc_expanded = anchor_slots.unsqueeze(2).expand(B, K, Q, D)
        q_expanded = query_feats.unsqueeze(1).expand(B, K, Q, D)
        anchor_scores_all = self.anchor_mlp(
            torch.cat([q_expanded, anc_expanded], dim=-1).reshape(B * K * Q, 2 * D)
        ).reshape(B, K, Q)
        top_anc_indices = torch.topk(anchor_scores_all, K_anc, dim=2).indices
        top_anc_scores = torch.gather(anchor_scores_all, 2, top_anc_indices)
        soft_anchor = F.softmax(top_anc_scores, dim=2)
        if self.anchor_aggregation == 'hard':
            hard_index = soft_anchor.argmax(dim=2, keepdim=True)
            p_anchor = torch.zeros_like(soft_anchor).scatter_(2, hard_index, 1.0)
        else:
            p_anchor = soft_anchor

        tgt_feats = torch.gather(
            query_feats, 1, top_m_indices.unsqueeze(-1).expand(B, M, D)
        )
        tgt_boxes = torch.gather(
            pred_boxes, 1, top_m_indices.unsqueeze(-1).expand(B, M, 6)
        )
        q_for_anc = query_feats.unsqueeze(1).expand(B, K, Q, D)
        anc_feats = torch.gather(
            q_for_anc, 2, top_anc_indices.unsqueeze(-1).expand(B, K, K_anc, D)
        )
        b_for_anc = pred_boxes.unsqueeze(1).expand(B, K, Q, 6)
        anc_boxes = torch.gather(
            b_for_anc, 2, top_anc_indices.unsqueeze(-1).expand(B, K, K_anc, 6)
        )

        tgt_f = tgt_feats.unsqueeze(1).unsqueeze(3).expand(B, K, M, K_anc, D)
        anc_f = anc_feats.unsqueeze(2).expand(B, K, M, K_anc, D)
        rel_f = rel_slots.unsqueeze(2).unsqueeze(3).expand(B, K, M, K_anc, D)
        rel_inputs = [tgt_f, anc_f, rel_f]
        if self.use_geometry:
            tgt_b = tgt_boxes.unsqueeze(1).unsqueeze(3)
            anc_b = anc_boxes.unsqueeze(2)
            geo_feat = self.geo_encoder(self._encode_pair_geometry(tgt_b, anc_b))
            rel_inputs.append(geo_feat)

        rel_input = torch.cat(rel_inputs, dim=-1)
        rel_scores = self.rel_pair_mlp(
            rel_input.reshape(-1, rel_input.shape[-1])
        ).reshape(B, K, M, K_anc)
        weighted = (rel_scores * p_anchor.unsqueeze(2)).sum(dim=3)
        s_struct_m = (weighted * slot_mask.float().unsqueeze(2)).sum(dim=1)
        one_hot = torch.zeros(B, M, Q, device=query_feats.device)
        one_hot.scatter_(2, top_m_indices.unsqueeze(2), 1.0)
        relation_anchor_scores = torch.einsum('bm,bmq->bq', s_struct_m, one_hot)
        return relation_anchor_scores, p_anchor

    @staticmethod
    def _uniform_anchor_probs(batch_size, num_relations, num_anchors, device):
        if num_anchors <= 0:
            return torch.zeros(batch_size, num_relations, 1, device=device)
        return torch.full(
            (batch_size, num_relations, num_anchors),
            1.0 / float(num_anchors),
            device=device,
        )

    @staticmethod
    def _encode_pair_geometry(box_i, box_j):
        box_i = box_i.float()
        box_j = box_j.float()
        eps = 1e-6
        centers_i = box_i[..., :3]
        centers_j = box_j[..., :3]
        size_i = box_i[..., 3:].abs().clamp_min(eps)
        size_j = box_j[..., 3:].abs().clamp_min(eps)
        dx, dy, dz = (centers_i - centers_j).unbind(-1)
        dist = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + eps)
        inv_dist = 1.0 / (dist + eps)
        w_i, h_i, l_i = size_i.unbind(-1)
        w_j, h_j, l_j = size_j.unbind(-1)
        return torch.stack([
            dx, dy, dz, dist,
            dx * inv_dist, dy * inv_dist, dz * inv_dist,
            (w_i * h_i * l_i) / (w_j * h_j * l_j + eps),
            w_i / (w_j + eps),
            h_i / (h_j + eps),
            l_i / (l_j + eps),
        ], dim=-1)
