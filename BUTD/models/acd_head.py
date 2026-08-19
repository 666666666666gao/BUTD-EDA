"""ACD: Anchor-Conditioned Compositional Decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_acd_base_scores(
    base_score_source,
    acd_head,
    end_points,
    proj_queries,
    proj_tokens,
):
    """Resolve the score tensor used as the late-ACD base."""
    if base_score_source == 'contrastive':
        return acd_head.compute_base_scores(proj_queries, proj_tokens)
    if base_score_source == 'quality':
        pred_iou = end_points.get('pred_iou', None)
        if pred_iou is None:
            raise ValueError(
                "ACD quality base requested but pred_iou is missing; "
                "enable --use_quality_head or choose contrastive base."
            )
        return pred_iou
    raise ValueError(
        "Unsupported ACD base score source: {}".format(base_score_source)
    )


class LateACDHead(nn.Module):
    """Late anchor-conditioned reasoning head."""

    def __init__(self, d_model=288, geo_dim=16, hidden_dim=288,
                 top_m_targets=32, top_k_anchors=16,
                 use_confidence_fusion=False,
                 global_residual_alpha=0.5,
                 warmup_steps=5000,
                 initial_alpha=0.05,
                 ea_scale=1.0,
                 pool_ea_multiplier=1.0,
                 final_ea_multiplier=1.0,
                 disable_struct_rerank=False,
                 proj_dim=64):
        super().__init__()
        if geo_dim < 0:
            raise ValueError("geo_dim must be non-negative")
        self.d_model = d_model
        self.geo_dim = geo_dim
        self.use_geometry = geo_dim > 0
        self.hidden_dim = hidden_dim
        self.top_m_targets = top_m_targets
        self.top_k_anchors = top_k_anchors
        self.use_confidence_fusion = use_confidence_fusion
        self.global_residual_alpha = global_residual_alpha
        self.warmup_steps = warmup_steps
        self.initial_alpha = initial_alpha
        self.ea_scale = ea_scale
        self.pool_ea_multiplier = pool_ea_multiplier
        self.final_ea_multiplier = final_ea_multiplier
        self.disable_struct_rerank = disable_struct_rerank

        # Learned reduction from (B, Q, L) contrastive similarities to (B, Q)
        self.base_score_attn = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 1)
        )
        # Learned temperature for contrastive similarities
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

        # Target-attribute coarse scorer
        self.target_attr_mlp = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Anchor scorer
        self.anchor_mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Relation pair scorer
        rel_input_dim = d_model * 3 + (geo_dim if self.use_geometry else 0)
        self.rel_pair_mlp = nn.Sequential(
            nn.Linear(rel_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Geometry encoder
        if self.use_geometry:
            self.geo_encoder = nn.Sequential(
                nn.Linear(11, geo_dim),
                nn.ReLU()
            )
        else:
            self.geo_encoder = None

        # Confidence-aware fusion
        if use_confidence_fusion:
            self.alpha_mlp = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

    def compute_base_scores(self, proj_queries, proj_tokens):
        """Compute base grounding scores via learned attention pooling.

        Args:
            proj_queries: (B, Q, D_proj) L2-normalized query projections
            proj_tokens: (B, L, D_proj) L2-normalized text token projections

        Returns:
            base_scores: (B, Q) per-query grounding scores
        """
        # Contrastive similarities are numerically sensitive under AMP.
        # Compute the ACD scoring path in float32 so temperature scaling and
        # token attention stay well-defined even when the main forward uses AMP.
        with torch.cuda.amp.autocast(enabled=False):
            # Detach: ACD loss should train ACD parameters, not contrastive projections
            proj_queries = proj_queries.detach().float()
            proj_tokens = proj_tokens.detach().float()

            temperature = self.log_temperature.float().exp() + 0.01
            sim = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2)) / temperature
            attn_logits = self.base_score_attn(proj_tokens).squeeze(-1)  # (B, L)
            attn_weights = F.softmax(attn_logits, dim=-1)  # (B, L)
            base_scores = torch.bmm(sim, attn_weights.unsqueeze(-1)).squeeze(-1)
        return base_scores

    def _get_warmup_alpha(self, global_step, target_alpha):
        """Compute warmup-scaled alpha.

        Args:
            global_step: current training step (None at eval → full alpha)
            target_alpha: target alpha value (scalar or (B, 1) tensor)

        Returns:
            Warmup-scaled alpha (same type as target_alpha)
        """
        if global_step is None or self.warmup_steps <= 0:
            return target_alpha
        if global_step >= self.warmup_steps:
            return target_alpha
        # Linear ramp: initial_alpha → target_alpha over warmup_steps
        t = global_step / self.warmup_steps
        if isinstance(target_alpha, torch.Tensor):
            return self.initial_alpha + (target_alpha - self.initial_alpha) * t
        else:
            return self.initial_alpha + (target_alpha - self.initial_alpha) * t

    def forward(self, query_feats, pred_boxes, base_scores, slot_dict,
                end_points=None, global_step=None):
        """
        Args:
            query_feats: (B, Q, D) final decoder query features
            pred_boxes: (B, Q, 6) predicted boxes [cx, cy, cz, w, h, l]
            base_scores: (B, Q) baseline grounding scores (from compute_base_scores)
            slot_dict: structured slot memory
            end_points: optional debug dict
            global_step: current training step for warmup (None = eval, use full alpha)

        Returns:
            dict with structured_scores, final_scores, acd_debug
        """
        with torch.cuda.amp.autocast(enabled=False):
            query_feats = query_feats.float()
            pred_boxes = pred_boxes.float()
            base_scores = base_scores.float()
            B, Q, D = query_feats.shape

            # Extract slots
            target_slot = slot_dict['target_slot'].float()  # (B, D)
            attr_slot = slot_dict['attr_slot'].float()  # (B, D)
            rel_slots = slot_dict['rel_slots'].float()  # (B, K, D)
            anchor_slots = slot_dict['anchor_slots'].float()  # (B, K, D)
            slot_mask = slot_dict['slot_mask']  # (B, K)
            parse_confidence = slot_dict['parse_confidence'].float()  # (B,)
            coverage = slot_dict.get('coverage_stats', {})
            has_target = coverage.get(
                'has_target',
                torch.ones(B, device=query_feats.device, dtype=torch.bool)
            ).to(device=query_feats.device).bool()
            structured_valid = has_target.float().unsqueeze(1)

            # Step A: Target-attribute coarse scoring
            s_ea = self._compute_target_attr_score(query_feats, target_slot, attr_slot)  # (B, Q)
            scaled_s_ea_raw = self.ea_scale * s_ea
            scaled_s_ea = scaled_s_ea_raw * structured_valid

            # Step B: Select candidate pools
            combined_scores = base_scores + self.pool_ea_multiplier * scaled_s_ea
            M = min(self.top_m_targets, Q)
            top_m_indices = torch.topk(combined_scores, M, dim=1).indices  # (B, M)

            # Step C & D: Relation-anchor reasoning (fully vectorized)
            _want_debug = end_points is not None
            _struct_out = self._compute_structured_score_batched(
                query_feats, pred_boxes, top_m_indices,
                rel_slots, anchor_slots, slot_mask,
                return_anchor_stats=_want_debug
            )
            if _want_debug:
                s_struct_raw, p_anchor, _ = _struct_out
            else:
                s_struct_raw = _struct_out  # (B, Q)
            s_struct = s_struct_raw * structured_valid

            # Step E: Safe residual fusion with warmup
            if self.use_confidence_fusion:
                learned_alpha = self._compute_fusion_alpha(slot_dict)  # (B,)
                learned_alpha = learned_alpha.unsqueeze(1)  # (B, 1)
                alpha = self._get_warmup_alpha(global_step, learned_alpha)
            else:
                alpha = self._get_warmup_alpha(global_step, self.global_residual_alpha)

            # The effective target-attribute contribution is explicitly scaled.
            struct_contrib = alpha * s_struct
            if self.disable_struct_rerank:
                final_scores = base_scores + self.final_ea_multiplier * scaled_s_ea
            else:
                final_scores = base_scores + self.final_ea_multiplier * scaled_s_ea + struct_contrib

            result = {
                'structured_scores': s_struct,
                'final_scores': final_scores,
            }

        # Debug stats — emitted as dbg_ keys for automatic logging
        if end_points is not None:
            _alpha_val = alpha.mean().item() if isinstance(alpha, torch.Tensor) else alpha
            valid_mask = structured_valid.squeeze(1).bool()

            # Anchor distribution diagnostics
            # p_anchor: (B, K, K_anc) — softmax over top-K anchors per pair
            eps = 1e-8
            anchor_entropy = -(p_anchor * (p_anchor + eps).log()).sum(dim=-1)  # (B, K)
            # Mask out inactive pairs before averaging
            mask_f = slot_mask.float()  # (B, K)
            active_count = mask_f.sum().clamp(min=1.0)
            avg_anchor_entropy = (anchor_entropy * mask_f).sum() / active_count
            avg_top1_mass = (p_anchor.max(dim=-1).values * mask_f).sum() / active_count

            result['acd_debug'] = {
                's_ea_mean': scaled_s_ea.mean().item(),
                's_struct_mean': s_struct.mean().item(),
                'alpha_mean': _alpha_val,
                'parse_conf_mean': parse_confidence.mean().item(),
            }

            # dbg_ prefixed keys flow into end_points → _accumulate_stats → logger
            end_points['dbg_acd_num_valid_tuples'] = slot_mask.float().sum(dim=1).mean().item()
            end_points['dbg_acd_s_base_mean'] = base_scores.mean().item()
            end_points['dbg_acd_s_base_std'] = base_scores.std(unbiased=False).item()
            end_points['dbg_acd_s_ea_mean'] = scaled_s_ea.mean().item()
            end_points['dbg_acd_s_ea_std'] = scaled_s_ea.std(unbiased=False).item()
            end_points['dbg_acd_s_struct_mean'] = s_struct.mean().item()
            end_points['dbg_acd_s_struct_std'] = s_struct.std(unbiased=False).item()
            end_points['dbg_acd_alpha_mean'] = _alpha_val
            end_points['dbg_acd_parse_conf_mean'] = parse_confidence.mean().item()
            end_points['dbg_acd_parse_conf_std'] = parse_confidence.std(unbiased=False).item()
            end_points['dbg_acd_slot_active_ratio'] = slot_mask.float().mean().item()
            end_points['dbg_acd_anchor_entropy'] = avg_anchor_entropy.item()
            end_points['dbg_acd_anchor_top1_mass'] = avg_top1_mass.item()
            end_points['dbg_acd_pool_ea_multiplier'] = self.pool_ea_multiplier
            end_points['dbg_acd_final_ea_multiplier'] = self.final_ea_multiplier
            end_points['dbg_acd_struct_disabled_in_final'] = float(self.disable_struct_rerank)
            end_points['dbg_acd_struct_contrib_mean'] = struct_contrib.mean().item()
            end_points['dbg_acd_struct_contrib_abs_mean'] = struct_contrib.abs().mean().item()
            end_points['dbg_acd_struct_contrib_std'] = struct_contrib.std(unbiased=False).item()
            end_points['dbg_acd_combined_minus_base_std'] = (combined_scores - base_scores).std(unbiased=False).item()
            final_delta = final_scores - base_scores
            base_top1 = base_scores.argmax(dim=1)
            final_top1 = final_scores.argmax(dim=1)
            top1_changed = (base_top1 != final_top1).float()
            delta_abs_per_sample = final_delta.abs().mean(dim=1)
            valid_count = valid_mask.float().sum().clamp(min=1.0)
            invalid_mask = ~valid_mask
            invalid_count = invalid_mask.float().sum().clamp(min=1.0)
            base_flat = base_scores.reshape(B, -1)
            final_flat = final_scores.reshape(B, -1)
            base_centered = base_flat - base_flat.mean(dim=1, keepdim=True)
            final_centered = final_flat - final_flat.mean(dim=1, keepdim=True)
            corr_num = (base_centered * final_centered).sum(dim=1)
            corr_den = (
                base_centered.pow(2).sum(dim=1).sqrt()
                * final_centered.pow(2).sum(dim=1).sqrt()
            ).clamp(min=1e-8)
            score_corr = corr_num / corr_den
            end_points['dbg_acd_final_score_mean'] = final_scores.mean().item()
            end_points['dbg_acd_final_score_std'] = final_scores.std(unbiased=False).item()
            end_points['dbg_acd_final_minus_base_mean'] = final_delta.mean().item()
            end_points['dbg_acd_final_minus_base_abs_mean'] = final_delta.abs().mean().item()
            end_points['dbg_acd_top1_changed_ratio'] = top1_changed.mean().item()
            end_points['dbg_acd_structured_valid_ratio'] = valid_mask.float().mean().item()
            end_points['dbg_acd_top1_changed_structured_ratio'] = (
                top1_changed * valid_mask.float()
            ).sum().div(valid_count).item()
            end_points['dbg_acd_top1_changed_unstructured_ratio'] = (
                top1_changed * invalid_mask.float()
            ).sum().div(invalid_count).item()
            end_points['dbg_acd_final_minus_base_abs_structured_mean'] = (
                delta_abs_per_sample * valid_mask.float()
            ).sum().div(valid_count).item()
            end_points['dbg_acd_final_minus_base_abs_unstructured_mean'] = (
                delta_abs_per_sample * invalid_mask.float()
            ).sum().div(invalid_count).item()
            end_points['dbg_acd_base_final_score_corr'] = score_corr.mean().item()
            end_points['dbg_warn_acd_nan_scores'] = float(
                torch.isnan(combined_scores).any()
                or torch.isnan(s_struct).any()
                or torch.isnan(final_scores).any()
            )
            end_points['dbg_warn_acd_no_structured_valid'] = float(valid_mask.float().sum().item() == 0)
            end_points['dbg_warn_acd_unstructured_score_shift'] = float(
                end_points['dbg_acd_final_minus_base_abs_unstructured_mean'] > 1e-4
            )
            end_points['dbg_warn_acd_low_top1_mass'] = float(avg_top1_mass.item() < 0.20)
            end_points['dbg_warn_acd_high_entropy'] = float(avg_anchor_entropy.item() > 2.5)
            end_points['dbg_warn_acd_no_rerank_effect'] = float(final_delta.abs().mean().item() < 1e-4)
            end_points['dbg_warn_acd_large_score_shift'] = float(final_delta.abs().mean().item() > 10.0)
            if self.disable_struct_rerank:
                end_points['dbg_warn_acd_struct_branch_large_unfused_shift'] = float(
                    struct_contrib.abs().mean().item() > 0.10
                )

        return result

    def _compute_target_attr_score(self, query_feats, target_slot, attr_slot):
        """Compute target-attribute coarse score. Fully batched."""
        B, Q, D = query_feats.shape
        target_expanded = target_slot.unsqueeze(1).expand(B, Q, D)
        attr_expanded = attr_slot.unsqueeze(1).expand(B, Q, D)
        combined = torch.cat([query_feats, target_expanded, attr_expanded], dim=-1)
        return self.target_attr_mlp(combined).squeeze(-1)  # (B, Q)

    def _compute_structured_score_batched(self, query_feats, pred_boxes, top_m_indices,
                                           rel_slots, anchor_slots, slot_mask,
                                           return_anchor_stats=False):
        """Compute structured relation-anchor score. Fully vectorized."""
        B, Q, D = query_feats.shape
        K = rel_slots.shape[1]
        M = top_m_indices.shape[1]
        K_anc = min(self.top_k_anchors, Q)

        # --- Anchor scoring: score all queries against each anchor slot ---
        # (slot_mask masking handles the no-active-pairs case without a host sync)
        # (B, K, Q, D) anchor_slots expanded
        anc_expanded = anchor_slots.unsqueeze(2).expand(B, K, Q, D)
        # (B, K, Q, D) queries expanded
        q_expanded = query_feats.unsqueeze(1).expand(B, K, Q, D)
        # (B, K, Q, 2D) → anchor_mlp → (B, K, Q)
        anchor_input = torch.cat([q_expanded, anc_expanded], dim=-1)
        anchor_scores_all = self.anchor_mlp(
            anchor_input.reshape(B * K * Q, 2 * D)
        ).reshape(B, K, Q)

        # Select top-K_anc anchors per pair
        top_anc_indices = torch.topk(anchor_scores_all, K_anc, dim=2).indices  # (B, K, K_anc)

        # Anchor distribution (softmax over selected anchors)
        top_anc_scores = torch.gather(anchor_scores_all, 2, top_anc_indices)  # (B, K, K_anc)
        p_anchor = F.softmax(top_anc_scores, dim=2)  # (B, K, K_anc)

        # --- Gather features for target pool and anchor pool ---
        # Target pool features: (B, M, D) and boxes: (B, M, 6)
        tgt_idx_exp = top_m_indices.unsqueeze(-1).expand(B, M, D)
        tgt_feats = torch.gather(query_feats, 1, tgt_idx_exp)  # (B, M, D)
        tgt_idx_box = top_m_indices.unsqueeze(-1).expand(B, M, 6)
        tgt_boxes = torch.gather(pred_boxes, 1, tgt_idx_box)  # (B, M, 6)

        # Anchor pool features per pair: (B, K, K_anc, D) and boxes: (B, K, K_anc, 6)
        anc_idx_exp = top_anc_indices.unsqueeze(-1).expand(B, K, K_anc, D)
        q_for_anc = query_feats.unsqueeze(1).expand(B, K, Q, D)
        anc_feats = torch.gather(q_for_anc, 2, anc_idx_exp)  # (B, K, K_anc, D)

        anc_idx_box = top_anc_indices.unsqueeze(-1).expand(B, K, K_anc, 6)
        b_for_anc = pred_boxes.unsqueeze(1).expand(B, K, Q, 6)
        anc_boxes = torch.gather(b_for_anc, 2, anc_idx_box)  # (B, K, K_anc, 6)

        # --- Relation pair scoring: (B, K, M, K_anc) ---
        # Expand features for concatenation
        tgt_f = tgt_feats.unsqueeze(1).unsqueeze(3).expand(B, K, M, K_anc, D)
        anc_f = anc_feats.unsqueeze(2).expand(B, K, M, K_anc, D)
        rel_f = rel_slots.unsqueeze(2).unsqueeze(3).expand(B, K, M, K_anc, D)

        rel_inputs = [tgt_f, anc_f, rel_f]
        if self.use_geometry:
            # Expand: tgt (B, 1, M, 1, 6) vs anc (B, K, 1, K_anc, 6)
            tgt_b = tgt_boxes.unsqueeze(1).unsqueeze(3)  # (B, 1, M, 1, 6)
            anc_b = anc_boxes.unsqueeze(2)                # (B, K, 1, K_anc, 6)
            geo_raw = self._encode_pair_geometry_batched(tgt_b, anc_b)  # (B, K, M, K_anc, 11)
            geo_feat = self.geo_encoder(geo_raw)  # (B, K, M, K_anc, geo_dim)
            rel_inputs.append(geo_feat)

        rel_input = torch.cat(rel_inputs, dim=-1)
        flat_input = rel_input.reshape(-1, rel_input.shape[-1])
        rel_scores = self.rel_pair_mlp(flat_input).reshape(B, K, M, K_anc)

        # Weight by anchor distribution and sum over anchors
        # (B, K, M, K_anc) * (B, K, 1, K_anc) → sum over K_anc → (B, K, M)
        weighted = (rel_scores * p_anchor.unsqueeze(2)).sum(dim=3)

        # Apply slot_mask: (B, K, 1) * (B, K, M) → sum over K → (B, M)
        mask_f = slot_mask.float().unsqueeze(2)  # (B, K, 1)
        s_struct_m = (weighted * mask_f).sum(dim=1)  # (B, M)

        # Scatter back to (B, Q) using dense matmul for clean gradient flow
        # One-hot encode indices: (B, M) → (B, M, Q) then contract with scores
        one_hot = torch.zeros(B, M, Q, device=query_feats.device)
        one_hot.scatter_(2, top_m_indices.unsqueeze(2), 1.0)
        s_struct = torch.einsum('bm,bmq->bq', s_struct_m, one_hot)

        if return_anchor_stats:
            return s_struct, p_anchor, slot_mask
        return s_struct

    @staticmethod
    def _encode_pair_geometry_batched(box_i, box_j):
        """Encode pairwise geometry between boxes. Fully vectorized.

        Args:
            box_i: (..., 6) [cx, cy, cz, w, h, l]
            box_j: (..., 6) [cx, cy, cz, w, h, l]

        Returns:
            geo_raw: (..., 11)
        """
        box_i = box_i.float()
        box_j = box_j.float()
        eps = 1e-6

        centers_i = box_i[..., :3]
        centers_j = box_j[..., :3]
        size_i = box_i[..., 3:].abs().clamp_min(eps)
        size_j = box_j[..., 3:].abs().clamp_min(eps)

        cx_i, cy_i, cz_i = centers_i.unbind(-1)
        cx_j, cy_j, cz_j = centers_j.unbind(-1)
        w_i, h_i, l_i = size_i.unbind(-1)
        w_j, h_j, l_j = size_j.unbind(-1)

        dx = cx_i - cx_j
        dy = cy_i - cy_j
        dz = cz_i - cz_j
        dist = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + eps)
        inv_dist = 1.0 / (dist + eps)

        return torch.stack([
            dx, dy, dz, dist,
            dx * inv_dist, dy * inv_dist, dz * inv_dist,
            (w_i * h_i * l_i) / (w_j * h_j * l_j + eps),
            w_i / (w_j + eps),
            h_i / (h_j + eps),
            l_i / (l_j + eps),
        ], dim=-1)

    def _compute_fusion_alpha(self, slot_dict):
        """Compute confidence-aware fusion weight.

        Raw counts fed directly — the MLP learns its own normalization.
        """
        parse_confidence = slot_dict['parse_confidence'].float()  # (B,)
        num_pairs = slot_dict['coverage_stats']['num_pairs'].float()  # (B,)
        has_target = slot_dict['coverage_stats']['has_target'].float()  # (B,)

        fusion_input = torch.stack([parse_confidence, num_pairs, has_target], dim=1)
        alpha = self.alpha_mlp(fusion_input).squeeze(-1)
        return alpha
