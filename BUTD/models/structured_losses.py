"""DHC: Decomposition-Guided Hard Negative & Consistency Learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DHCLossModule(nn.Module):
    """Learned parameters for DHC losses.

    Holds learned margin and temperature parameters, replacing fixed constants.
    """

    def __init__(self, margin_min=0.0, temperature_max=0.0):
        super().__init__()
        self.log_margin_entity = nn.Parameter(torch.tensor(-1.6))    # ~softplus -> 0.2
        self.log_margin_attr = nn.Parameter(torch.tensor(-1.9))      # ~softplus -> 0.15
        self.log_margin_rel = nn.Parameter(torch.tensor(-1.9))       # ~softplus -> 0.15
        self.log_margin_acd_rank = nn.Parameter(torch.tensor(-0.7))  # ~softplus -> 0.5
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self.margin_min = float(margin_min)
        self.temperature_max = float(temperature_max)

    def _apply_margin_floor(self, value):
        if self.margin_min > 0:
            floor = value.new_tensor(self.margin_min)
            value = torch.maximum(value, floor)
        return value

    def _apply_temperature_cap(self, value):
        if self.temperature_max > 0:
            cap = value.new_tensor(self.temperature_max)
            value = torch.minimum(value, cap)
        return value

    def raw_margin_entity(self):
        return F.softplus(self.log_margin_entity)

    def raw_margin_attr(self):
        return F.softplus(self.log_margin_attr)

    def raw_margin_rel(self):
        return F.softplus(self.log_margin_rel)

    def raw_margin_acd_rank(self):
        return F.softplus(self.log_margin_acd_rank)

    def raw_temperature(self):
        return self.log_temperature.exp() + 0.01

    def margin_entity(self):
        return self._apply_margin_floor(self.raw_margin_entity())

    def margin_attr(self):
        return self._apply_margin_floor(self.raw_margin_attr())

    def margin_rel(self):
        return self._apply_margin_floor(self.raw_margin_rel())

    def margin_acd_rank(self):
        return self._apply_margin_floor(self.raw_margin_acd_rank())

    def temperature(self):
        return self._apply_temperature_cap(self.raw_temperature())


def _build_pos_mask(indices, B, Q, device):
    """Build (B, Q) boolean positive mask from Hungarian matching indices.

    Also returns (B,) valid mask for batch elements with at least one match.
    Fully vectorized — no Python per-element branching on device tensors.
    """
    # Collect all (batch_idx, query_idx) pairs into flat tensors
    batch_ids = []
    query_ids = []
    for b in range(B):
        idx = indices[b][0]
        if len(idx) > 0:
            n = len(idx)
            batch_ids.append(torch.full((n,), b, dtype=torch.long))
            query_ids.append(idx if isinstance(idx, torch.Tensor) else torch.tensor(idx, dtype=torch.long))
    pos_mask = torch.zeros(B, Q, device=device, dtype=torch.bool)
    valid = torch.zeros(B, device=device, dtype=torch.bool)
    if batch_ids:
        b_flat = torch.cat(batch_ids).to(device)
        q_flat = torch.cat(query_ids).to(device)
        pos_mask[b_flat, q_flat] = True
        valid = pos_mask.any(dim=1)
    return pos_mask, valid


def _build_target_pos_mask(indices, B, Q, device, target_gt_idx=0):
    """Build (B, Q) boolean mask for queries matched to the target GT only."""
    batch_ids = []
    query_ids = []
    pos_mask = torch.zeros(B, Q, device=device, dtype=torch.bool)
    valid = torch.zeros(B, device=device, dtype=torch.bool)

    for b in range(B):
        src_idx, tgt_idx = indices[b]
        if len(src_idx) == 0:
            continue
        src_idx = src_idx if isinstance(src_idx, torch.Tensor) else torch.tensor(src_idx, dtype=torch.long)
        tgt_idx = tgt_idx if isinstance(tgt_idx, torch.Tensor) else torch.tensor(tgt_idx, dtype=torch.long)
        keep = (tgt_idx == target_gt_idx)
        if keep.any():
            chosen = src_idx[keep]
            batch_ids.append(torch.full((len(chosen),), b, dtype=torch.long))
            query_ids.append(chosen)

    if batch_ids:
        b_flat = torch.cat(batch_ids).to(device)
        q_flat = torch.cat(query_ids).to(device)
        pos_mask[b_flat, q_flat] = True
        valid = pos_mask.any(dim=1)
    return pos_mask, valid


def _dataset_not_scannet_mask(end_points, B, device):
    """Return samples that are not synthetic ScanNet detection-only rows."""
    dataset = end_points.get('dataset', None)
    if dataset is None:
        return torch.ones(B, device=device, dtype=torch.bool)

    if hasattr(dataset, 'tolist'):
        dataset = dataset.tolist()
    if isinstance(dataset, str):
        is_valid = [dataset.strip().lower() != 'scannet'] * B
    elif isinstance(dataset, (list, tuple)):
        is_valid = []
        for item in list(dataset)[:B]:
            if isinstance(item, bytes):
                name = item.decode('utf-8', errors='ignore')
            else:
                name = str(item)
            is_valid.append(name.strip().lower() != 'scannet')
        if len(is_valid) < B:
            is_valid.extend([True] * (B - len(is_valid)))
    else:
        return torch.ones(B, device=device, dtype=torch.bool)

    return torch.tensor(is_valid, device=device, dtype=torch.bool)


def _endpoint_bool_mask(end_points, key, B, device):
    value = end_points.get(key, None)
    if value is None:
        return torch.zeros(B, device=device, dtype=torch.bool)
    if torch.is_tensor(value):
        return value.to(device=device).bool().view(-1)[:B]
    if isinstance(value, (list, tuple)):
        out = []
        for item in list(value)[:B]:
            if isinstance(item, (list, tuple)) and len(item) == 1:
                item = item[0]
            if isinstance(item, bytes):
                item = item.decode('utf-8', errors='ignore')
            if isinstance(item, str):
                out.append(item.strip().lower() in {'1', 'true', 'yes'})
            else:
                out.append(bool(item))
        if len(out) < B:
            out.extend([False] * (B - len(out)))
        return torch.tensor(out, device=device, dtype=torch.bool)
    return torch.zeros(B, device=device, dtype=torch.bool)


def _status_mask(end_points, status_name, B, device):
    value = end_points.get('decomposition_status', None)
    out = [False] * B
    if isinstance(value, str):
        out = [value == status_name] * B
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(list(value)[:B]):
            if isinstance(item, bytes):
                item = item.decode('utf-8', errors='ignore')
            out[i] = str(item) == status_name
    return torch.tensor(out, device=device, dtype=torch.bool)


def _global_only_decomposition_mask(end_points, B, device):
    if 'global_only_mask' in end_points:
        return _endpoint_bool_mask(end_points, 'global_only_mask', B, device)
    if 'decomp_global_only_mask' in end_points:
        return _endpoint_bool_mask(end_points, 'decomp_global_only_mask', B, device)
    slot_dict = end_points.get('slot_dict', None)
    if slot_dict and 'coverage_stats' in slot_dict:
        coverage = slot_dict['coverage_stats']
        has_target = coverage.get(
            'has_target',
            torch.ones(B, device=device, dtype=torch.bool),
        ).to(device=device).bool()
        global_parse = coverage.get(
            'global_only_due_to_parse_error',
            torch.zeros(B, device=device),
        ).to(device=device).float() > 0
        missing = coverage.get(
            'missing_target',
            torch.zeros(B, device=device),
        ).to(device=device).float() > 0
        return (~has_target) | global_parse | missing
    return _endpoint_bool_mask(end_points, 'global_only_due_to_parse_error', B, device)


def _weak_generic_decomposition_mask(end_points, B, device):
    if 'weak_generic_target_mask' in end_points:
        return _endpoint_bool_mask(end_points, 'weak_generic_target_mask', B, device)
    if 'decomp_weak_generic_mask' in end_points:
        return _endpoint_bool_mask(end_points, 'decomp_weak_generic_mask', B, device)
    slot_dict = end_points.get('slot_dict', None)
    if slot_dict and 'coverage_stats' in slot_dict:
        coverage = slot_dict['coverage_stats']
        return (
            (coverage.get(
                'overgeneric_target_remaining',
                torch.zeros(B, device=device),
            ).to(device=device).float() > 0)
            | (coverage.get(
                'target_overgeneric_canonical',
                torch.zeros(B, device=device),
            ).to(device=device).float() > 0)
            | (coverage.get(
                'generic_target',
                torch.zeros(B, device=device),
            ).to(device=device).float() > 0)
        )
    return _endpoint_bool_mask(end_points, 'target_generic_reference', B, device)


def _structured_valid_mask(end_points, B, device):
    """Return samples with parsed target spans for structured supervision."""
    dataset_valid = _dataset_not_scannet_mask(end_points, B, device)
    slot_dict = end_points.get('slot_dict', None)
    if not slot_dict or 'coverage_stats' not in slot_dict:
        return dataset_valid & (~_global_only_decomposition_mask(end_points, B, device))
    has_target = slot_dict['coverage_stats'].get('has_target', None)
    if has_target is None:
        return dataset_valid & (~_global_only_decomposition_mask(end_points, B, device))
    return (
        has_target.to(device=device).bool()
        & dataset_valid
        & (~_global_only_decomposition_mask(end_points, B, device))
    )


def _write_structured_debug(losses, end_points, B, device):
    """Write common structured-supervision coverage diagnostics."""
    structured_valid = _structured_valid_mask(end_points, B, device)
    dataset_valid = _dataset_not_scannet_mask(end_points, B, device)
    losses['dbg_struct_valid_batch_ratio'] = structured_valid.float().mean()
    losses['dbg_struct_non_scannet_batch_ratio'] = dataset_valid.float().mean()
    losses['dbg_struct_scannet_batch_ratio'] = (~dataset_valid).float().mean()
    losses['dbg_warn_struct_no_valid_samples'] = float(
        structured_valid.float().sum().detach().item() == 0
    )

    slot_dict = end_points.get('slot_dict', None)
    if slot_dict and 'coverage_stats' in slot_dict:
        coverage = slot_dict['coverage_stats']
        has_target = coverage.get('has_target', None)
        if has_target is not None:
            has_target = has_target.to(device=device).bool()
            losses['dbg_struct_has_target_ratio'] = has_target.float().mean()
            losses['dbg_warn_struct_scannet_has_target'] = float(
                ((~dataset_valid) & has_target).any().detach().item()
            )
        if 'num_attrs' in coverage:
            losses['dbg_struct_num_attrs_mean'] = coverage['num_attrs'].float().mean()
        if 'num_pairs' in coverage:
            losses['dbg_struct_num_pairs_mean'] = coverage['num_pairs'].float().mean()

    return structured_valid, dataset_valid


def _masked_hard_negative(similarities, neg_mask, dim):
    """Return the hardest negative score along dim."""
    neg_fill = torch.finfo(similarities.dtype).min
    return similarities.masked_fill(~neg_mask, neg_fill).max(dim=dim).values


def _batched_hardneg_contrastive(similarities, pos_mask, valid, margin):
    """Compute hard-negative contrastive loss over a batch. Vectorized.

    Args:
        similarities: (B, Q) per-query similarity scores
        pos_mask: (B, Q) boolean — True for positive (matched) queries
        valid: (B,) boolean — True for batch elements with matches
        margin: scalar tensor (learned)

    Returns:
        loss: scalar tensor (averaged over B, matching original loop semantics)
    """
    B = similarities.shape[0]
    neg_mask = ~pos_mask  # (B, Q)

    # Positive scores: mean over positives per batch element
    pos_scores = similarities.masked_fill(~pos_mask, 0.0).sum(dim=1)  # (B,)
    pos_count = pos_mask.float().sum(dim=1).clamp(min=1.0)  # (B,)
    pos_mean = pos_scores / pos_count  # (B,)

    # Negative scores: hardest negative per batch element.
    # Do not use raw logsumexp here: with hundreds of queries it adds a
    # log(num_negatives) offset and makes the margin almost always violated.
    neg_hard = _masked_hard_negative(similarities, neg_mask, dim=1)

    # Margin loss per batch element, masked by valid
    per_batch = F.relu(neg_hard - pos_mean + margin) * valid.float()  # (B,)

    valid_count = valid.float().sum().clamp(min=1.0)
    return per_batch.sum() / valid_count


def _hardneg_debug_stats(similarities, pos_mask, valid, margin):
    """Summarize positive-vs-hard-negative separation for debug logging."""
    neg_mask = ~pos_mask
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)

    pos_scores = similarities.masked_fill(~pos_mask, 0.0).sum(dim=1)
    pos_count = pos_mask.float().sum(dim=1).clamp(min=1.0)
    pos_mean = pos_scores / pos_count

    neg_for_lse = similarities.masked_fill(~neg_mask, torch.finfo(similarities.dtype).min)
    neg_lse_raw = neg_for_lse.logsumexp(dim=1)
    neg_count = neg_mask.float().sum(dim=1).clamp(min=1.0)
    neg_logmeanexp = neg_lse_raw - neg_count.log()
    neg_hard = _masked_hard_negative(similarities, neg_mask, dim=1)
    gap = pos_mean - neg_hard
    violation = (neg_hard - pos_mean + margin > 0) & valid

    return {
        'pos_mean': (pos_mean * valid_f).sum() / valid_count,
        'neg_hard': (neg_hard * valid_f).sum() / valid_count,
        'neg_lse_raw': (neg_lse_raw * valid_f).sum() / valid_count,
        'neg_logmeanexp': (neg_logmeanexp * valid_f).sum() / valid_count,
        'gap': (gap * valid_f).sum() / valid_count,
        'violation_ratio': violation.float().sum() / valid_count,
        'valid_ratio': valid.float().mean(),
    }


def loss_dhc_consistency(end_points, weight=0.2):
    """Global-structured consistency loss. Detached contrastive base."""
    if 'acd_final_scores' not in end_points or 'dhc_temperature' not in end_points:
        return torch.tensor(0.0, device=end_points['seed_xyz'].device)

    proj_tokens = end_points.get('proj_tokens', None)
    proj_queries = end_points.get('last_proj_queries', None)
    if proj_tokens is None or proj_queries is None:
        return torch.tensor(0.0, device=end_points['seed_xyz'].device)

    proj_queries_d = proj_queries.detach()
    proj_tokens_d = proj_tokens.detach()

    temperature = end_points['dhc_temperature']
    sim = torch.matmul(proj_queries_d, proj_tokens_d.transpose(-1, -2)) / temperature
    base_scores = sim.logsumexp(dim=-1)  # (B, Q)
    struct_scores = end_points['acd_final_scores']  # (B, Q)

    valid = _structured_valid_mask(end_points, struct_scores.shape[0], struct_scores.device)
    per_sample = F.kl_div(
        F.log_softmax(struct_scores, dim=-1),
        F.softmax(base_scores, dim=-1),
        reduction='none'
    ).sum(dim=1)
    valid_f = valid.float()
    kl_loss = (per_sample * valid_f).sum() / valid_f.sum().clamp(min=1.0)
    return weight * kl_loss


def loss_dhc_entity_hardneg(end_points, indices, weight=0.2):
    """Entity hard negative contrastive loss. Vectorized."""
    if (
        'slot_dict' not in end_points
        or 'last_queries' not in end_points
        or 'dhc_margin_entity' not in end_points
    ):
        return torch.tensor(0.0, device=end_points['seed_xyz'].device)

    slot_dict = end_points['slot_dict']
    target_slot = slot_dict['target_slot']  # (B, D)
    query_feats = end_points['last_queries']  # (B, Q, D)
    B, Q, D = query_feats.shape

    target_expanded = target_slot.unsqueeze(1).expand(B, Q, D)
    similarities = F.cosine_similarity(query_feats, target_expanded, dim=-1)  # (B, Q)

    pos_mask, valid = _build_target_pos_mask(indices, B, Q, query_feats.device)
    valid = (
        valid
        & _structured_valid_mask(end_points, B, query_feats.device)
        & (~_weak_generic_decomposition_mask(end_points, B, query_feats.device))
    )
    margin = end_points['dhc_margin_entity']
    loss = _batched_hardneg_contrastive(similarities, pos_mask, valid, margin)
    return weight * loss


def loss_dhc_attribute_hardneg(end_points, indices, weight=0.2):
    """Attribute hard negative contrastive loss. Vectorized."""
    if (
        'slot_dict' not in end_points
        or 'last_queries' not in end_points
        or 'dhc_margin_attr' not in end_points
    ):
        return torch.tensor(0.0, device=end_points['seed_xyz'].device)

    slot_dict = end_points['slot_dict']
    attr_slot = slot_dict['attr_slot']  # (B, D)
    num_attrs = slot_dict['coverage_stats']['num_attrs']  # (B,)
    query_feats = end_points['last_queries']  # (B, Q, D)
    B, Q, D = query_feats.shape

    attr_expanded = attr_slot.unsqueeze(1).expand(B, Q, D)
    similarities = F.cosine_similarity(query_feats, attr_expanded, dim=-1)  # (B, Q)

    pos_mask, valid = _build_target_pos_mask(indices, B, Q, query_feats.device)
    # Also require num_attrs > 0
    valid = (
        valid
        & _structured_valid_mask(end_points, B, query_feats.device)
        & (num_attrs > 0)
    )

    margin = end_points['dhc_margin_attr']
    loss = _batched_hardneg_contrastive(similarities, pos_mask, valid, margin)
    return weight * loss


def loss_dhc_relation_hardneg(end_points, indices, weight=0.2):
    """Relation hard negative contrastive loss. Vectorized over batch and pairs."""
    if (
        'slot_dict' not in end_points
        or 'last_queries' not in end_points
        or 'dhc_margin_rel' not in end_points
    ):
        return torch.tensor(0.0, device=end_points['seed_xyz'].device)

    slot_dict = end_points['slot_dict']
    rel_slots = slot_dict['rel_slots']  # (B, K, D)
    slot_mask = slot_dict['slot_mask']  # (B, K)
    query_feats = end_points['last_queries']  # (B, Q, D)
    B, Q, D = query_feats.shape
    K = rel_slots.shape[1]

    pos_mask, valid = _build_target_pos_mask(indices, B, Q, query_feats.device)
    valid = valid & _structured_valid_mask(end_points, B, query_feats.device)
    margin = end_points['dhc_margin_rel']

    # Compute cosine similarity for all pairs at once: (B, K, Q)
    # query_feats: (B, 1, Q, D),  rel_slots: (B, K, 1, D) → (B, K, Q)
    sims = F.cosine_similarity(
        query_feats.unsqueeze(1).expand(B, K, Q, D),
        rel_slots.unsqueeze(2).expand(B, K, Q, D),
        dim=-1
    )  # (B, K, Q)

    # Per-pair validity: batch has matches AND pair slot is active
    pair_valid = valid.unsqueeze(1) & slot_mask  # (B, K)

    # Compute loss for each pair, then sum
    # Expand pos_mask to (B, K, Q)
    pos_mask_k = pos_mask.unsqueeze(1).expand(B, K, Q)
    neg_mask_k = ~pos_mask_k

    pos_scores = sims.masked_fill(~pos_mask_k, 0.0).sum(dim=2)  # (B, K)
    pos_count = pos_mask_k.float().sum(dim=2).clamp(min=1.0)     # (B, K)
    pos_mean = pos_scores / pos_count                              # (B, K)

    neg_hard = _masked_hard_negative(sims, neg_mask_k, dim=2)     # (B, K)

    per_pair = F.relu(neg_hard - pos_mean + margin) * pair_valid.float()  # (B, K)
    loss = per_pair.sum() / pair_valid.float().sum().clamp(min=1.0)

    return weight * loss


def compute_dhc_losses(end_points, indices, config):
    """Compute all DHC losses."""
    losses = {}

    losses['loss_dhc_consistency'] = loss_dhc_consistency(
        end_points,
        weight=config.get('dhc_consistency_weight', 0.2)
    )
    losses['loss_dhc_ent_hardneg'] = loss_dhc_entity_hardneg(
        end_points, indices,
        weight=config.get('dhc_ent_hardneg_weight', 0.2)
    )
    losses['loss_dhc_attr_hardneg'] = loss_dhc_attribute_hardneg(
        end_points, indices,
        weight=config.get('dhc_attr_hardneg_weight', 0.2)
    )
    losses['loss_dhc_rel_hardneg'] = loss_dhc_relation_hardneg(
        end_points, indices,
        weight=config.get('dhc_rel_hardneg_weight', 0.2)
    )

    device = end_points['seed_xyz'].device
    slot_dict = end_points.get('slot_dict', {})
    debug_B = end_points['last_queries'].shape[0] if 'last_queries' in end_points else None
    if debug_B is not None:
        _write_structured_debug(losses, end_points, debug_B, device)
    if 'dhc_margin_entity' in end_points:
        losses['dbg_dhc_margin_entity'] = end_points['dhc_margin_entity'].detach()
    if 'dhc_margin_attr' in end_points:
        losses['dbg_dhc_margin_attr'] = end_points['dhc_margin_attr'].detach()
    if 'dhc_margin_rel' in end_points:
        losses['dbg_dhc_margin_rel'] = end_points['dhc_margin_rel'].detach()
    if 'dhc_margin_acd_rank' in end_points:
        losses['dbg_dhc_margin_acd_rank'] = end_points['dhc_margin_acd_rank'].detach()
    if 'dhc_temperature' in end_points:
        losses['dbg_dhc_temperature'] = end_points['dhc_temperature'].detach()

    if indices is not None and 'last_queries' in end_points:
        B, Q = end_points['last_queries'].shape[:2]
        pos_mask, valid = _build_target_pos_mask(indices, B, Q, device)
        structured_valid = _structured_valid_mask(end_points, B, device)
        valid = valid & structured_valid
        losses['dbg_dhc_structured_valid_batch_ratio'] = structured_valid.float().mean()
        losses['dbg_dhc_valid_batch_ratio'] = valid.float().mean()
        losses['dbg_dhc_positive_query_ratio'] = pos_mask.float().mean()
        if 'coverage_stats' in slot_dict:
            coverage = slot_dict['coverage_stats']
            if 'num_attrs' in coverage:
                attr_valid = valid & (coverage['num_attrs'] > 0)
                losses['dbg_dhc_attr_valid_batch_ratio'] = attr_valid.float().mean()
            if 'num_pairs' in coverage:
                rel_valid = valid & (coverage['num_pairs'] > 0)
                losses['dbg_dhc_rel_valid_batch_ratio'] = rel_valid.float().mean()

        query_feats = end_points['last_queries']
        if 'slot_dict' in end_points:
            target_slot = end_points['slot_dict']['target_slot']
            target_sims = F.cosine_similarity(
                query_feats,
                target_slot.unsqueeze(1).expand_as(query_feats),
                dim=-1
            )
            if 'dhc_margin_entity' in end_points:
                entity_stats = _hardneg_debug_stats(
                    target_sims, pos_mask, valid, end_points['dhc_margin_entity']
                )
                losses['dbg_dhc_ent_pos_mean'] = entity_stats['pos_mean']
                losses['dbg_dhc_ent_neg_hard'] = entity_stats['neg_hard']
                losses['dbg_dhc_ent_neg_lse_raw'] = entity_stats['neg_lse_raw']
                losses['dbg_dhc_ent_neg_logmeanexp'] = entity_stats['neg_logmeanexp']
                losses['dbg_dhc_ent_gap'] = entity_stats['gap']
                losses['dbg_dhc_ent_violation_ratio'] = entity_stats['violation_ratio']

            attr_slot = end_points['slot_dict']['attr_slot']
            attr_sims = F.cosine_similarity(
                query_feats,
                attr_slot.unsqueeze(1).expand_as(query_feats),
                dim=-1
            )
            attr_valid = valid
            if 'coverage_stats' in end_points['slot_dict']:
                attr_valid = valid & (end_points['slot_dict']['coverage_stats']['num_attrs'] > 0)
            if 'dhc_margin_attr' in end_points:
                attr_stats = _hardneg_debug_stats(
                    attr_sims, pos_mask, attr_valid, end_points['dhc_margin_attr']
                )
                losses['dbg_dhc_attr_pos_mean'] = attr_stats['pos_mean']
                losses['dbg_dhc_attr_neg_hard'] = attr_stats['neg_hard']
                losses['dbg_dhc_attr_neg_lse_raw'] = attr_stats['neg_lse_raw']
                losses['dbg_dhc_attr_neg_logmeanexp'] = attr_stats['neg_logmeanexp']
                losses['dbg_dhc_attr_gap'] = attr_stats['gap']
                losses['dbg_dhc_attr_violation_ratio'] = attr_stats['violation_ratio']

            rel_slots = end_points['slot_dict']['rel_slots']
            slot_mask = end_points['slot_dict']['slot_mask']
            batch_size, num_pairs, num_queries = rel_slots.shape[0], rel_slots.shape[1], query_feats.shape[1]
            rel_sims = F.cosine_similarity(
                query_feats.unsqueeze(1).expand(batch_size, num_pairs, num_queries, query_feats.shape[-1]),
                rel_slots.unsqueeze(2).expand(batch_size, num_pairs, num_queries, rel_slots.shape[-1]),
                dim=-1
            )
            pos_mask_k = pos_mask.unsqueeze(1).expand(batch_size, num_pairs, num_queries)
            pair_valid = valid.unsqueeze(1) & slot_mask
            pair_valid_f = pair_valid.float()
            pair_valid_count = pair_valid_f.sum().clamp(min=1.0)
            neg_mask_k = ~pos_mask_k
            pos_scores = rel_sims.masked_fill(~pos_mask_k, 0.0).sum(dim=2)
            pos_count = pos_mask_k.float().sum(dim=2).clamp(min=1.0)
            pos_mean = pos_scores / pos_count
            neg_for_lse = rel_sims.masked_fill(~neg_mask_k, torch.finfo(rel_sims.dtype).min)
            neg_lse_raw = neg_for_lse.logsumexp(dim=2)
            neg_count = neg_mask_k.float().sum(dim=2).clamp(min=1.0)
            neg_logmeanexp = neg_lse_raw - neg_count.log()
            neg_hard = _masked_hard_negative(rel_sims, neg_mask_k, dim=2)
            rel_gap = pos_mean - neg_hard
            rel_violation = (
                neg_hard - pos_mean + end_points['dhc_margin_rel'] > 0
            ) & pair_valid
            losses['dbg_dhc_rel_pos_mean'] = (pos_mean * pair_valid_f).sum() / pair_valid_count
            losses['dbg_dhc_rel_neg_hard'] = (neg_hard * pair_valid_f).sum() / pair_valid_count
            losses['dbg_dhc_rel_neg_lse_raw'] = (neg_lse_raw * pair_valid_f).sum() / pair_valid_count
            losses['dbg_dhc_rel_neg_logmeanexp'] = (neg_logmeanexp * pair_valid_f).sum() / pair_valid_count
            losses['dbg_dhc_rel_gap'] = (rel_gap * pair_valid_f).sum() / pair_valid_count
            losses['dbg_dhc_rel_violation_ratio'] = rel_violation.float().sum() / pair_valid_count

    if 'acd_final_scores' in end_points and 'proj_tokens' in end_points and 'last_proj_queries' in end_points and 'dhc_temperature' in end_points:
        proj_queries = end_points['last_proj_queries'].detach()
        proj_tokens = end_points['proj_tokens'].detach()
        temperature = end_points['dhc_temperature']
        base_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2)) / temperature
        base_scores = base_scores.logsumexp(dim=-1)
        struct_scores = end_points['acd_final_scores']
        base_probs = F.softmax(base_scores, dim=-1)
        struct_probs = F.softmax(struct_scores, dim=-1)
        valid = _structured_valid_mask(end_points, struct_scores.shape[0], struct_scores.device)
        valid_f = valid.float()
        js_forward = F.kl_div(
            torch.log(struct_probs + 1e-8), base_probs, reduction='none'
        ).sum(dim=1)
        js_backward = F.kl_div(
            torch.log(base_probs + 1e-8), struct_probs, reduction='none'
        ).sum(dim=1)
        losses['dbg_dhc_consistency_js_proxy'] = (
            0.5 * ((js_forward + js_backward) * valid_f).sum()
            / valid_f.sum().clamp(min=1.0)
        )
        losses['dbg_dhc_consistency_top1_agreement'] = (
            ((base_scores.argmax(dim=1) == struct_scores.argmax(dim=1)).float() * valid_f).sum()
            / valid_f.sum().clamp(min=1.0)
        )

    total_dhc = (
        losses['loss_dhc_consistency']
        + losses['loss_dhc_ent_hardneg']
        + losses['loss_dhc_attr_hardneg']
        + losses['loss_dhc_rel_hardneg']
    )
    losses['dbg_dhc_total_loss'] = total_dhc.detach()
    losses['dbg_warn_dhc_zero_total'] = float(total_dhc.detach().item() <= 1e-8)
    losses['dbg_warn_dhc_nan_total'] = float(torch.isnan(total_dhc.detach()).item())
    if 'dbg_dhc_ent_gap' in losses:
        losses['dbg_warn_dhc_ent_negative_gap'] = float(losses['dbg_dhc_ent_gap'].detach().item() < 0.0)
    if 'dbg_dhc_attr_gap' in losses:
        losses['dbg_warn_dhc_attr_negative_gap'] = float(losses['dbg_dhc_attr_gap'].detach().item() < 0.0)
    if 'dbg_dhc_rel_gap' in losses:
        losses['dbg_warn_dhc_rel_negative_gap'] = float(losses['dbg_dhc_rel_gap'].detach().item() < 0.0)
    if 'dbg_dhc_consistency_top1_agreement' in losses:
        losses['dbg_warn_dhc_low_top1_agreement'] = float(
            losses['dbg_dhc_consistency_top1_agreement'].detach().item() < 0.30
        )

    return losses


def _char_spans_to_token_spans(tokenized, span_batches, device, min_slots=1):
    """Convert batched char-level spans into padded token-span tensors."""
    batch_size = tokenized['input_ids'].shape[0]
    max_spans = max(
        (len(spans) for spans in span_batches if isinstance(spans, list)),
        default=0
    )
    max_spans = max(max_spans, min_slots)
    span_tensor = torch.full((batch_size, max_spans, 2), -1, dtype=torch.long, device=device)

    for b in range(batch_size):
        spans = span_batches[b] if b < len(span_batches) and isinstance(span_batches[b], list) else []
        for i, span in enumerate(spans[:max_spans]):
            if not isinstance(span, dict):
                continue
            start = span.get('start', None)
            end = span.get('end', None)
            if start is None or end is None:
                continue
            try:
                start = int(start)
                end = int(end)
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue

            beg_pos = tokenized.char_to_token(b, start)
            end_pos = tokenized.char_to_token(b, end - 1)

            if beg_pos is None:
                for offset in (1, 2):
                    probe = start + offset
                    if probe >= end:
                        break
                    beg_pos = tokenized.char_to_token(b, probe)
                    if beg_pos is not None:
                        break

            if end_pos is None:
                for offset in (1, 2):
                    probe = end - 1 - offset
                    if probe < start:
                        break
                    end_pos = tokenized.char_to_token(b, probe)
                    if end_pos is not None:
                        break

            if beg_pos is None or end_pos is None or end_pos < beg_pos:
                continue

            span_tensor[b, i, 0] = beg_pos
            span_tensor[b, i, 1] = end_pos + 1

    return span_tensor


def _pool_token_spans(token_feats, span_tensor):
    """Mean-pool token features over padded token spans."""
    B, L, D = token_feats.shape
    device = token_feats.device
    positions = torch.arange(L, device=device)
    start = span_tensor[..., 0]
    end = span_tensor[..., 1]
    valid = (start >= 0) & (end > start) & (end <= L)
    masks = (positions >= start.unsqueeze(-1)) & (positions < end.unsqueeze(-1))
    masks = masks & valid.unsqueeze(-1)
    mask_f = masks.float().unsqueeze(-1)
    pooled = (token_feats.unsqueeze(1) * mask_f).sum(dim=2) / (mask_f.sum(dim=2) + 1e-8)
    pooled = pooled * valid.unsqueeze(-1).float()
    return pooled, valid


def compute_s2s_aux_losses(end_points, weight=1.0):
    """Supervise slot_dict directly from the dataset's decomposed spans."""
    if 'slot_dict' not in end_points or 'text_feats' not in end_points or 'tokenized' not in end_points:
        zero = torch.tensor(0.0, device=end_points['seed_xyz'].device)
        return {'loss_s2s_aux': zero}

    slot_dict = end_points['slot_dict']
    token_feats = end_points['text_feats']
    tokenized = end_points['tokenized']
    device = token_feats.device
    B, L, D = token_feats.shape

    losses = {}
    structured_valid, _ = _write_structured_debug(losses, end_points, B, device)
    structured_valid_f = structured_valid.float()
    structured_count = structured_valid_f.sum().clamp(min=1.0)

    # Global slot: match deterministic masked mean over valid tokens.
    valid_tokens = tokenized['attention_mask'].bool().unsqueeze(-1).float()
    oracle_global = (token_feats.detach() * valid_tokens).sum(dim=1) / (valid_tokens.sum(dim=1) + 1e-8)
    global_per_sample = 1 - F.cosine_similarity(
        slot_dict['global_slot'], oracle_global, dim=-1
    )
    losses['loss_s2s_global'] = (
        global_per_sample * structured_valid_f
    ).sum() / structured_count

    # Target slot: prefer the dataset's explicit target_slot span; fall back to positive_map[:, 0].
    oracle_target = None
    target_valid = None
    target_source = None
    target_slot_batch = end_points.get('target_slot', None)
    if target_slot_batch is not None:
        target_span_batches = []
        for span in target_slot_batch:
            if isinstance(span, dict):
                target_span_batches.append([span])
            else:
                target_span_batches.append([])
        target_tensor = _char_spans_to_token_spans(tokenized, target_span_batches, device, min_slots=1)
        oracle_target_spans, target_valid_spans = _pool_token_spans(token_feats.detach(), target_tensor)
        oracle_target = oracle_target_spans[:, 0]
        target_valid = target_valid_spans[:, 0]
        target_source = 'target_slot'

    if oracle_target is None or target_valid is None or not target_valid.any():
        positive_map = end_points.get('positive_map', None)
        if positive_map is not None:
            target_weights = positive_map[:, 0, :L].float().to(device)
            target_valid = target_weights.sum(dim=1) > 0
            target_weights = target_weights / (target_weights.sum(dim=1, keepdim=True) + 1e-8)
            oracle_target = torch.bmm(target_weights.unsqueeze(1), token_feats.detach()).squeeze(1)
            target_source = 'positive_map_fallback'

    if oracle_target is not None and target_valid is not None:
        target_valid = target_valid & structured_valid
        losses['dbg_s2s_target_valid_ratio'] = target_valid.float().mean()
        losses['dbg_s2s_target_source_target_slot'] = float(target_source == 'target_slot')
        losses['dbg_s2s_target_source_fallback'] = float(target_source == 'positive_map_fallback')
        if target_valid.any():
            target_cosine = F.cosine_similarity(
                slot_dict['target_slot'][target_valid],
                oracle_target[target_valid],
                dim=-1
            )
            losses['dbg_s2s_target_cosine'] = target_cosine.mean()
            losses['dbg_s2s_target_cosine_std'] = target_cosine.std(unbiased=False)
            losses['loss_s2s_target'] = (1 - target_cosine).mean()
            losses['dbg_warn_s2s_target_low_cosine'] = float(target_cosine.mean().item() < 0.70)
        else:
            losses['dbg_s2s_target_cosine'] = torch.tensor(0.0, device=device)
            losses['dbg_s2s_target_cosine_std'] = torch.tensor(0.0, device=device)
            losses['dbg_warn_s2s_target_no_valid'] = 1.0

    # Attribute slots: average the exact attribute spans from the dataset.
    attr_batches = end_points.get('attr_spans', None)
    if attr_batches is not None:
        attr_tensor = _char_spans_to_token_spans(tokenized, attr_batches, device)
        oracle_attr_spans, attr_valid = _pool_token_spans(token_feats.detach(), attr_tensor)
        attr_count = attr_valid.long().sum(dim=1)
        attr_mask = (attr_count > 0) & structured_valid
        if attr_mask.any():
            oracle_attr = oracle_attr_spans.sum(dim=1) / attr_count.float().clamp(min=1.0).unsqueeze(-1)
            losses['loss_s2s_attr'] = (
                1 - F.cosine_similarity(slot_dict['attr_slot'][attr_mask], oracle_attr[attr_mask], dim=-1)
            ).mean()

    # Relation / anchor slots: supervise the exact paired span tuples.
    rel_batches = end_points.get('rel_spans', None)
    ent_batches = end_points.get('entity_spans', None)
    anchor_ids = end_points.get('anchor_ids', None)
    if rel_batches is not None and ent_batches is not None and anchor_ids is not None:
        rel_tensor = _char_spans_to_token_spans(tokenized, rel_batches, device)
        ent_tensor = _char_spans_to_token_spans(tokenized, ent_batches, device)
        oracle_rel, rel_valid = _pool_token_spans(token_feats.detach(), rel_tensor)
        oracle_ent, _ = _pool_token_spans(token_feats.detach(), ent_tensor)
        K = slot_dict['rel_slots'].shape[1]
        N_rel = min(rel_tensor.shape[1], K)
        aid = anchor_ids[:, :N_rel].to(device)
        aid_clamped = aid.clamp(min=0, max=max(ent_tensor.shape[1] - 1, 0))
        gather_idx = aid_clamped.unsqueeze(-1).expand(-1, -1, D)
        oracle_anchor = torch.gather(oracle_ent[:, :ent_tensor.shape[1]], 1, gather_idx) if ent_tensor.shape[1] > 0 else torch.zeros(B, N_rel, D, device=device)
        anchor_valid = (aid >= 0) & (aid < ent_tensor.shape[1]) if ent_tensor.shape[1] > 0 else torch.zeros(B, N_rel, dtype=torch.bool, device=device)
        pair_valid = rel_valid[:, :N_rel] & anchor_valid & structured_valid.unsqueeze(1)
        if pair_valid.any():
            pred_rel = slot_dict['rel_slots'][:, :N_rel][pair_valid]
            pred_anchor = slot_dict['anchor_slots'][:, :N_rel][pair_valid]
            gold_rel = oracle_rel[:, :N_rel][pair_valid]
            gold_anchor = oracle_anchor[pair_valid]
            losses['loss_s2s_rel'] = (1 - F.cosine_similarity(pred_rel, gold_rel, dim=-1)).mean()
            losses['loss_s2s_anchor'] = (1 - F.cosine_similarity(pred_anchor, gold_anchor, dim=-1)).mean()

    # Parse confidence: supervise against exact structural coverage ratio.
    has_target = slot_dict['coverage_stats']['has_target'].float()
    has_attr = (slot_dict['coverage_stats']['num_attrs'] > 0).float()
    has_pair = (slot_dict['coverage_stats']['num_pairs'] > 0).float()
    confidence_target = torch.stack([has_target, has_attr, has_pair], dim=1).mean(dim=1)
    confidence_per_sample = F.mse_loss(
        slot_dict['parse_confidence'],
        confidence_target,
        reduction='none'
    )
    losses['loss_s2s_conf'] = (
        confidence_per_sample * structured_valid_f
    ).sum() / structured_count

    component_losses = [
        v for k, v in losses.items()
        if k.startswith('loss_') and k != 'loss_s2s_aux' and isinstance(v, torch.Tensor)
    ]
    if component_losses:
        losses['loss_s2s_aux'] = weight * torch.stack(component_losses).mean()
    else:
        losses['loss_s2s_aux'] = torch.tensor(0.0, device=device)
    return losses
