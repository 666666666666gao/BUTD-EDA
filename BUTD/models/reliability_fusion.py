"""RAPF: reliability-aware probabilistic fusion."""

import torch
import torch.nn as nn


class ReliabilityFusion(nn.Module):
    """Fuse baseline and structured scores with a learned reliability gate."""

    def __init__(self, hidden_dim=128, initial_gate_bias=-2.0,
                 use_quality=False, quality_weight=0.25,
                 generic_gate_cap=0.35, residual_clip=2.0,
                 quality_anchor_structured_residual=False):
        super().__init__()
        self.use_quality = use_quality
        self.quality_weight = float(quality_weight)
        self.generic_gate_cap = float(generic_gate_cap)
        self.residual_clip = float(residual_clip)
        self.quality_anchor_structured_residual = bool(
            quality_anchor_structured_residual
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(15, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        final = self.gate_mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, float(initial_gate_bias))

    @staticmethod
    def _normalize_scores(scores, valid_query_mask=None):
        scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if valid_query_mask is None:
            valid_query_mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            valid_query_mask = valid_query_mask.to(device=scores.device).bool()
        valid_f = valid_query_mask.float()
        count = valid_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (scores * valid_f).sum(dim=1, keepdim=True) / count
        centered = (scores - mean) * valid_f
        var = centered.pow(2).sum(dim=1, keepdim=True) / count
        std = var.clamp(min=1e-6).sqrt()
        normalized = centered / std
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized.masked_fill(~valid_query_mask, 0.0)

    def forward(self, base_scores, structured_scores, quality_scores=None,
                structured_valid_mask=None, global_only_mask=None,
                weak_generic_target_mask=None, parse_confidence=None,
                decomposition_error_flags_count=None,
                anchor_entropy=None, anchor_top1_mass=None,
                valid_query_mask=None):
        with torch.cuda.amp.autocast(enabled=False):
            base_scores = base_scores.float()
            structured_scores = torch.nan_to_num(
                structured_scores.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            B, Q = base_scores.shape
            device = base_scores.device

            if structured_valid_mask is None:
                structured_valid_mask = torch.ones(B, device=device, dtype=torch.bool)
            else:
                structured_valid_mask = structured_valid_mask.to(device=device).bool()
            if global_only_mask is None:
                global_only_mask = torch.zeros(B, device=device, dtype=torch.bool)
            else:
                global_only_mask = global_only_mask.to(device=device).bool()
            if weak_generic_target_mask is None:
                weak_generic_target_mask = torch.zeros(B, device=device, dtype=torch.bool)
            else:
                weak_generic_target_mask = weak_generic_target_mask.to(device=device).bool()
            if parse_confidence is None:
                parse_confidence = torch.ones(B, device=device)
            else:
                parse_confidence = parse_confidence.to(device=device).float()
            if decomposition_error_flags_count is None:
                decomposition_error_flags_count = torch.zeros(B, device=device)
            else:
                decomposition_error_flags_count = decomposition_error_flags_count.to(device=device).float()
            if anchor_entropy is None:
                anchor_entropy = torch.zeros(B, device=device)
            else:
                anchor_entropy = anchor_entropy.to(device=device).float().view(-1)[:B]
            if anchor_top1_mass is None:
                anchor_top1_mass = torch.zeros(B, device=device)
            else:
                anchor_top1_mass = anchor_top1_mass.to(device=device).float().view(-1)[:B]

            if valid_query_mask is None:
                valid_query_mask = torch.ones_like(base_scores, dtype=torch.bool)
            else:
                valid_query_mask = valid_query_mask.to(device=device).bool()

            base_norm = self._normalize_scores(base_scores, valid_query_mask)
            struct_norm = self._normalize_scores(
                structured_scores * structured_valid_mask.float().unsqueeze(1),
                valid_query_mask,
            )
            quality_available = quality_scores is not None
            if quality_scores is None:
                quality_norm = torch.zeros_like(base_norm)
            else:
                quality_norm = self._normalize_scores(quality_scores.float(), valid_query_mask)
            if self.use_quality and quality_available:
                safe_anchor = base_norm + self.quality_weight * quality_norm
            else:
                safe_anchor = base_norm
            residual_anchor = (
                safe_anchor
                if self.quality_anchor_structured_residual
                else base_norm
            )
            raw_residual = struct_norm - residual_anchor
            delta = raw_residual.clamp(
                min=-self.residual_clip,
                max=self.residual_clip,
            )

            base_probs = torch.softmax(base_norm.masked_fill(~valid_query_mask, -1e4), dim=1)
            struct_probs = torch.softmax(struct_norm.masked_fill(~valid_query_mask, -1e4), dim=1)
            eps = 1e-8
            log_q = max(float(Q), 2.0)
            base_entropy = (
                -(base_probs * (base_probs + eps).log()).sum(dim=1)
                / base_scores.new_tensor(log_q).log()
            )
            top2 = torch.topk(base_norm, k=min(2, Q), dim=1).values
            if top2.shape[1] == 1:
                base_top1_margin = torch.zeros(B, device=device)
            else:
                base_top1_margin = top2[:, 0] - top2[:, 1]
            top1_disagreement = (
                base_norm.argmax(dim=1) != struct_norm.argmax(dim=1)
            ).float()
            mix = 0.5 * (base_probs + struct_probs)
            js_divergence = 0.5 * (
                (base_probs * ((base_probs + eps).log() - (mix + eps).log())).sum(dim=1)
                + (struct_probs * ((struct_probs + eps).log() - (mix + eps).log())).sum(dim=1)
            )
            quality_max = quality_norm.max(dim=1).values

            features = torch.stack([
                base_norm,
                struct_norm,
                quality_norm,
                delta,
                parse_confidence.unsqueeze(1).expand(B, Q),
                base_entropy.unsqueeze(1).expand(B, Q),
                base_top1_margin.unsqueeze(1).expand(B, Q),
                top1_disagreement.unsqueeze(1).expand(B, Q),
                js_divergence.unsqueeze(1).expand(B, Q),
                quality_max.unsqueeze(1).expand(B, Q),
                global_only_mask.float().unsqueeze(1).expand(B, Q),
                weak_generic_target_mask.float().unsqueeze(1).expand(B, Q),
                decomposition_error_flags_count.unsqueeze(1).expand(B, Q),
                anchor_entropy.unsqueeze(1).expand(B, Q),
                anchor_top1_mass.unsqueeze(1).expand(B, Q),
            ], dim=-1)
            gate = torch.sigmoid(self.gate_mlp(features).squeeze(-1))
            gate = gate * structured_valid_mask.float().unsqueeze(1)
            gate = gate.masked_fill(global_only_mask.unsqueeze(1), 0.0)
            if self.generic_gate_cap > 0:
                capped_gate = gate.clamp(max=self.generic_gate_cap)
                gate = torch.where(
                    weak_generic_target_mask.unsqueeze(1),
                    capped_gate,
                    gate,
                )

            if self.quality_anchor_structured_residual:
                fused_scores = safe_anchor + gate * delta
            else:
                fused_scores = base_norm + gate * delta
            if self.use_quality and not self.quality_anchor_structured_residual:
                fused_scores = fused_scores + self.quality_weight * quality_norm
            fused_scores = torch.nan_to_num(fused_scores, nan=0.0, posinf=0.0, neginf=0.0)
            # Grounding evaluation masks non-matching proposals by multiplying
            # their scores by zero.  A z-normalized (negative) score would no
            # longer be ranking-equivalent under that operation.  A per-sample
            # additive shift leaves every margin and argmax unchanged while
            # making all valid fused scores strictly positive.
            fused_score_floor = fused_scores.min(dim=1, keepdim=True).values
            fused_scores = fused_scores - fused_score_floor + 1e-6

            error_count = decomposition_error_flags_count.view(B)[:B]
            repaired_mask = (
                structured_valid_mask
                & (~global_only_mask)
                & (~weak_generic_target_mask)
                & (error_count > 0)
            )
            ok_mask = (
                structured_valid_mask
                & (~global_only_mask)
                & (~weak_generic_target_mask)
                & (error_count <= 0)
            )
            top1_changed = (
                base_norm.argmax(dim=1) != fused_scores.argmax(dim=1)
            ).float()
            fused_delta = fused_scores - base_norm
            if self.residual_clip == 0:
                residual_clip_ratio = (raw_residual.abs() > 0).float().mean()
            elif self.residual_clip > 0:
                residual_clip_ratio = (
                    raw_residual.abs() > float(self.residual_clip)
                ).float().mean()
            else:
                residual_clip_ratio = base_scores.new_tensor(0.0)

            def _masked_query_mean(values, sample_mask):
                sample_f = sample_mask.float().unsqueeze(1)
                denom = sample_f.sum().clamp(min=1.0) * values.shape[1]
                return (values * sample_f).sum() / denom

        return {
            'fused_scores': fused_scores,
            'rapf_gate': gate,
            'rapf_base_norm': base_norm,
            'rapf_structured_norm': struct_norm,
            'rapf_quality_norm': quality_norm,
            'rapf_safe_anchor': safe_anchor,
            'rapf_delta': delta,
            'rapf_fused_score_floor': fused_score_floor,
            'dbg_rapf_gate_mean': gate.mean(),
            'dbg_rapf_gate_std': gate.std(unbiased=False),
            'dbg_rapf_gate_min': gate.min(),
            'dbg_rapf_gate_max': gate.max(),
            'dbg_rapf_gate_ok_mean': _masked_query_mean(gate, ok_mask),
            'dbg_rapf_gate_repaired_mean': _masked_query_mean(gate, repaired_mask),
            'dbg_rapf_gate_weak_generic_mean': _masked_query_mean(
                gate, weak_generic_target_mask
            ),
            'dbg_rapf_gate_global_only_mean': _masked_query_mean(
                gate, global_only_mask
            ),
            'dbg_rapf_gate_structured_mean': (
                gate * structured_valid_mask.float().unsqueeze(1)
            ).sum() / (
                structured_valid_mask.float().sum().clamp(min=1.0) * Q
            ),
            'dbg_rapf_global_only_gate_mean': (
                gate * global_only_mask.float().unsqueeze(1)
            ).sum() / (global_only_mask.float().sum().clamp(min=1.0) * Q),
            'dbg_rapf_generic_gate_mean': (
                gate * weak_generic_target_mask.float().unsqueeze(1)
            ).sum() / (weak_generic_target_mask.float().sum().clamp(min=1.0) * Q),
            'dbg_rapf_structured_valid_ratio': structured_valid_mask.float().mean(),
            'dbg_rapf_global_only_ratio': global_only_mask.float().mean(),
            'dbg_rapf_generic_target_ratio': weak_generic_target_mask.float().mean(),
            'dbg_rapf_quality_enabled': float(self.use_quality),
            'dbg_rapf_quality_anchor_enabled': float(
                self.quality_anchor_structured_residual
            ),
            'dbg_rapf_safe_score_used': float(
                self.quality_anchor_structured_residual
            ),
            'dbg_rapf_residual_clip': float(self.residual_clip),
            'dbg_rapf_residual_abs_mean': raw_residual.abs().mean(),
            'dbg_rapf_residual_clip_ratio': residual_clip_ratio,
            'dbg_rapf_delta_mean': delta.mean(),
            'dbg_rapf_delta_abs_mean': delta.abs().mean(),
            'dbg_rapf_fused_minus_base_abs_mean': fused_delta.abs().mean(),
            'dbg_rapf_top1_changed_ratio': top1_changed.mean(),
            'dbg_rapf_base_entropy_mean': base_entropy.mean(),
            'dbg_rapf_entropy_mean': base_entropy.mean(),
            'dbg_rapf_base_top1_margin_mean': base_top1_margin.mean(),
            'dbg_rapf_margin_mean': base_top1_margin.mean(),
            'dbg_rapf_top1_disagreement_ratio': top1_disagreement.mean(),
            'dbg_rapf_disagree_ratio': top1_disagreement.mean(),
            'dbg_rapf_js_divergence_mean': js_divergence.mean(),
            'dbg_rapf_js_mean': js_divergence.mean(),
            'dbg_rapf_quality_max_mean': quality_max.mean(),
            'dbg_warn_rapf_nan_scores': float(
                (not torch.isfinite(fused_scores).all().detach().item())
                or (not torch.isfinite(gate).all().detach().item())
            ),
            'dbg_warn_rapf_nan_scores_ratio': float(
                (not torch.isfinite(fused_scores).all().detach().item())
                or (not torch.isfinite(gate).all().detach().item())
            ),
        }
