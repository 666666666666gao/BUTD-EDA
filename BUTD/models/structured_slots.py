"""S2S: Span-to-Slot Structured Decomposition."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredSlotBuilder(nn.Module):
    """Build structured slot memory from parsed spans."""

    def __init__(self, d_model=288, pooling='attention', max_pairs=3):
        super().__init__()
        self.d_model = d_model
        self.pooling = pooling
        self.max_pairs = max_pairs

        if pooling == 'attention':
            self.global_attn = nn.Linear(d_model, 1)
            self.target_attn = nn.Linear(d_model, 1)
            self.attr_attn = nn.Linear(d_model, 1)
            self.rel_attn = nn.Linear(d_model, 1)
            self.anchor_attn = nn.Linear(d_model, 1)

        # Learned target selection: scores each entity span's relevance as target
        self.target_select = nn.Linear(d_model, d_model)

        # Learned parse confidence from coverage statistics
        # Input: [has_target (1), num_attrs_raw (1), num_pairs_raw (1)]
        # Raw counts fed directly — the MLP learns its own normalization
        self.confidence_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, token_feats, tokenized, entity_spans=None,
                attr_spans=None, rel_spans=None, anchor_types=None,
                anchor_ids=None, utterances=None):
        """
        Args:
            token_feats: (B, L, D) text token features
            tokenized: dict with 'input_ids', 'attention_mask'
            entity_spans: (B, max_ent, 2) entity span indices
            attr_spans: (B, max_attr, 2) attribute span indices
            rel_spans: (B, max_rel, 2) relation span indices
            anchor_types: (B, max_rel) anchor type per relation
            anchor_ids: (B, max_rel) anchor entity id per relation

        Returns:
            slot_dict with structured memory
        """
        B, L, D = token_feats.shape
        device = token_feats.device

        attn_mask = tokenized['attention_mask']  # (B, L)

        # Global slot: pool over all valid tokens (already batched)
        global_slot = self._pool_tokens(token_feats, attn_mask,
                                        self.global_attn if self.pooling == 'attention' else None)

        # Target slot: learned selection over entity spans
        target_slot, has_target = self._build_target_slot(
            token_feats, entity_spans, global_slot)

        # Attribute slot: aggregate all attribute spans
        attr_slot, num_attrs = self._build_attr_slot(token_feats, attr_spans)

        # Relation-anchor pairs
        rel_slots, anchor_slots, slot_mask, num_pairs = self._build_rel_anchor_pairs(
            token_feats, rel_spans, entity_spans, anchor_ids)

        # Parse confidence (learned MLP, raw counts as input)
        parse_confidence = self._compute_parse_confidence(has_target, num_attrs, num_pairs)

        return {
            'global_slot': global_slot,       # (B, D)
            'target_slot': target_slot,       # (B, D)
            'attr_slot': attr_slot,           # (B, D)
            'rel_slots': rel_slots,           # (B, K, D)
            'anchor_slots': anchor_slots,     # (B, K, D)
            'parse_confidence': parse_confidence,  # (B,)
            'slot_mask': slot_mask,            # (B, K)
            'coverage_stats': {
                'has_target': has_target,      # (B,)
                'num_attrs': num_attrs,        # (B,)
                'num_pairs': num_pairs,        # (B,)
            }
        }

    def _pool_tokens(self, feats, mask, attn_layer=None):
        """Pool token features with mask. Fully batched."""
        if self.pooling == 'mean':
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (feats * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        elif self.pooling == 'attention' and attn_layer is not None:
            scores = attn_layer(feats).squeeze(-1)
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (feats * weights).sum(dim=1)
        else:
            pooled = feats.mean(dim=1)
        return pooled

    def _span_to_mask(self, spans, L, device):
        """Convert (start, end) span pairs to position masks without .item().

        Args:
            spans: (..., 2) integer tensor of (start, end) pairs
            L: sequence length
            device: torch device

        Returns:
            masks: (..., L) boolean masks
            valid: (...) boolean — True where start >= 0 and end > start and end <= L
        """
        start = spans[..., 0]  # (...)
        end = spans[..., 1]    # (...)
        valid = (start >= 0) & (end > start) & (end <= L)

        positions = torch.arange(L, device=device)  # (L,)
        # Broadcast: (..., 1) vs (L,) → (..., L)
        masks = (positions >= start.unsqueeze(-1)) & (positions < end.unsqueeze(-1))
        masks = masks & valid.unsqueeze(-1)  # zero out invalid spans
        return masks, valid

    def _masked_attn_pool(self, feats, masks, attn_layer):
        """Attention-pool features using span masks. No .item() calls.

        Args:
            feats: (B, L, D)
            masks: (B, N, L) boolean masks for N spans
            attn_layer: nn.Linear(D, 1)

        Returns:
            pooled: (B, N, D)
        """
        B, L, D = feats.shape
        N = masks.shape[1]

        scores = attn_layer(feats).squeeze(-1)  # (B, L)
        # Expand for N spans: (B, 1, L) → masked per span
        scores_exp = scores.unsqueeze(1).expand(B, N, L)  # (B, N, L)
        scores_exp = scores_exp.masked_fill(~masks, torch.finfo(scores_exp.dtype).min)
        weights = F.softmax(scores_exp, dim=2)  # (B, N, L)

        # Weighted sum: (B, N, L, 1) * (B, 1, L, D) → sum over L → (B, N, D)
        feats_exp = feats.unsqueeze(1)  # (B, 1, L, D)
        pooled = (weights.unsqueeze(-1) * feats_exp).sum(dim=2)  # (B, N, D)
        return pooled

    def _masked_mean_pool(self, feats, masks):
        """Mean-pool features using span masks. No .item() calls.

        Args:
            feats: (B, L, D)
            masks: (B, N, L) boolean masks

        Returns:
            pooled: (B, N, D)
        """
        mask_f = masks.float().unsqueeze(-1)  # (B, N, L, 1)
        feats_exp = feats.unsqueeze(1)  # (B, 1, L, D)
        pooled = (feats_exp * mask_f).sum(dim=2) / (mask_f.sum(dim=2) + 1e-8)
        return pooled

    def _pool_spans(self, feats, masks, attn_layer):
        """Pool spans using attention or mean, controlled by self.pooling."""
        if self.pooling == 'attention':
            return self._masked_attn_pool(feats, masks, attn_layer)
        else:
            return self._masked_mean_pool(feats, masks)

    def _build_target_slot(self, token_feats, entity_spans, global_slot):
        """Build target slot by selecting among entity spans via learned attention.

        Fully vectorized — no per-batch loops or .item() calls.
        """
        B, L, D = token_feats.shape
        device = token_feats.device

        if entity_spans is None:
            return (torch.zeros(B, D, device=device),
                    torch.zeros(B, device=device, dtype=torch.bool))

        N_ent = entity_spans.shape[1]

        # Build span masks: (B, N_ent, L) and validity: (B, N_ent)
        masks, valid = self._span_to_mask(entity_spans, L, device)

        # has_target: any valid entity span in the batch element
        has_target = valid.any(dim=1)  # (B,)

        # Pool each entity span: (B, N_ent, D)
        span_reps = self._pool_spans(token_feats, masks, self.target_attn)

        # Zero out invalid spans
        span_reps = span_reps * valid.unsqueeze(-1).float()

        # Learned selection: use global_slot as query
        query = self.target_select(global_slot)  # (B, D)
        # Score each span: (B, N_ent)
        select_scores = torch.bmm(span_reps, query.unsqueeze(-1)).squeeze(-1)
        # Mask invalid spans
        select_scores = select_scores.masked_fill(~valid, torch.finfo(select_scores.dtype).min)
        select_weights = F.softmax(select_scores, dim=1).unsqueeze(-1)  # (B, N_ent, 1)

        # Weighted combination
        target_slot = (span_reps * select_weights).sum(dim=1)  # (B, D)

        # Zero out batch elements with no valid spans
        target_slot = target_slot * has_target.unsqueeze(-1).float()

        return target_slot, has_target

    def _build_attr_slot(self, token_feats, attr_spans):
        """Build aggregated attribute slot. Vectorized."""
        B, L, D = token_feats.shape
        device = token_feats.device

        if attr_spans is None:
            return (torch.zeros(B, D, device=device),
                    torch.zeros(B, device=device, dtype=torch.long))

        N_attr = attr_spans.shape[1]

        masks, valid = self._span_to_mask(attr_spans, L, device)  # (B, N_attr, L), (B, N_attr)
        num_attrs = valid.long().sum(dim=1)  # (B,)

        span_reps = self._pool_spans(token_feats, masks, self.attr_attn)  # (B, N_attr, D)
        span_reps = span_reps * valid.unsqueeze(-1).float()

        # Mean over valid attribute spans
        valid_count = num_attrs.float().clamp(min=1.0).unsqueeze(-1)  # (B, 1)
        attr_slot = span_reps.sum(dim=1) / valid_count  # (B, D)

        # Zero out if no valid attrs
        attr_slot = attr_slot * (num_attrs > 0).unsqueeze(-1).float()

        return attr_slot, num_attrs

    def _build_rel_anchor_pairs(self, token_feats, rel_spans, entity_spans, anchor_ids):
        """Build relation-anchor tuple slots. Vectorized."""
        B, L, D = token_feats.shape
        device = token_feats.device
        K = self.max_pairs

        rel_slots = torch.zeros(B, K, D, device=device)
        anchor_slots = torch.zeros(B, K, D, device=device)
        slot_mask = torch.zeros(B, K, device=device, dtype=torch.bool)
        num_pairs = torch.zeros(B, device=device, dtype=torch.long)

        if rel_spans is None:
            return rel_slots, anchor_slots, slot_mask, num_pairs

        N_rel = min(rel_spans.shape[1], K)

        # Pool relation spans
        rel_masks, rel_valid = self._span_to_mask(rel_spans[:, :N_rel], L, device)
        rel_pooled = self._pool_spans(token_feats, rel_masks, self.rel_attn)  # (B, N_rel, D)

        # Build anchor spans from anchor_ids → entity_spans lookup
        if anchor_ids is not None and entity_spans is not None:
            # Clamp anchor_ids to valid range
            aid = anchor_ids[:, :N_rel].clamp(0, entity_spans.shape[1] - 1)  # (B, N_rel)
            # Gather entity spans for each anchor: (B, N_rel, 2)
            aid_exp = aid.unsqueeze(-1).expand(B, N_rel, 2)
            anc_spans = torch.gather(entity_spans, 1, aid_exp)  # (B, N_rel, 2)

            anc_masks, anc_valid = self._span_to_mask(anc_spans, L, device)
            anc_pooled = self._pool_spans(token_feats, anc_masks, self.anchor_attn)

            # A pair is valid only if both relation and anchor spans are valid
            # and anchor_id was in valid range
            aid_in_range = (anchor_ids[:, :N_rel] >= 0) & (anchor_ids[:, :N_rel] < entity_spans.shape[1])
            pair_valid = rel_valid & anc_valid & aid_in_range  # (B, N_rel)
        else:
            anc_pooled = torch.zeros(B, N_rel, D, device=device)
            pair_valid = torch.zeros(B, N_rel, device=device, dtype=torch.bool)

        # Fill in up to K slots (take first K valid pairs per batch)
        # Since N_rel <= K, we can directly assign
        rel_slots[:, :N_rel] = rel_pooled * pair_valid.unsqueeze(-1).float()
        anchor_slots[:, :N_rel] = anc_pooled * pair_valid.unsqueeze(-1).float()
        slot_mask[:, :N_rel] = pair_valid
        num_pairs = pair_valid.long().sum(dim=1)

        return rel_slots, anchor_slots, slot_mask, num_pairs

    def _compute_parse_confidence(self, has_target, num_attrs, num_pairs):
        """Compute parse confidence from span coverage via learned MLP.

        Raw counts are fed directly — the MLP learns its own normalization.
        """
        feats = torch.stack([
            has_target.float(),
            num_attrs.float(),
            num_pairs.float(),
        ], dim=1)  # (B, 3)
        confidence = self.confidence_mlp(feats).squeeze(-1)  # (B,)
        return confidence
