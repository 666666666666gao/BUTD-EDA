"""Source-pool selector head for grounding candidate reranking."""

import torch
import torch.nn as nn


def compute_soft_token_base_scores(
    sem_cls_scores, positive_map, box_label_mask=None
):
    """Compute BBS soft-token grounding scores for the primary target row."""
    sem_scores = sem_cls_scores.float().softmax(-1)
    positive_map = positive_map.float()
    if positive_map.dim() != 3 or positive_map.shape[1] < 1:
        raise ValueError("positive_map must have shape (B, G, T) with G >= 1")
    target_map = positive_map[:, 0]
    if sem_scores.shape[-1] != target_map.shape[-1]:
        aligned = sem_scores.new_zeros(
            sem_scores.shape[0], sem_scores.shape[1], target_map.shape[-1]
        )
        copy_dim = min(sem_scores.shape[-1], target_map.shape[-1])
        aligned[:, :, :copy_dim] = sem_scores[:, :, :copy_dim]
        sem_scores = aligned
    scores = (sem_scores * target_map.unsqueeze(1)).sum(-1)
    if box_label_mask is not None:
        valid = box_label_mask[:, 0].to(scores.device).bool()
        scores = scores * valid.float().unsqueeze(1)
    return scores


def compute_contrastive_token_base_scores(
    proj_queries, proj_tokens, positive_map, box_label_mask=None
):
    """Compute BBF contrastive grounding scores for the primary target row."""
    proj_queries = proj_queries.float()
    proj_tokens = proj_tokens.float()
    token_scores = torch.matmul(
        proj_queries, proj_tokens.transpose(-1, -2)
    )
    token_scores = (token_scores / 0.07).softmax(-1)
    positive_map = positive_map.float()
    if positive_map.dim() != 3 or positive_map.shape[1] < 1:
        raise ValueError("positive_map must have shape (B, G, T) with G >= 1")
    target_map = positive_map[:, 0]
    if token_scores.shape[-1] != target_map.shape[-1]:
        aligned = token_scores.new_zeros(
            token_scores.shape[0],
            token_scores.shape[1],
            target_map.shape[-1],
        )
        copy_dim = min(token_scores.shape[-1], target_map.shape[-1])
        aligned[:, :, :copy_dim] = token_scores[:, :, :copy_dim]
        token_scores = aligned
    scores = (token_scores * target_map.unsqueeze(1)).sum(-1)
    if box_label_mask is not None:
        valid = box_label_mask[:, 0].to(scores.device).bool()
        scores = scores * valid.float().unsqueeze(1)
    return scores


class SourcePoolSelectorHead(nn.Module):
    """Predict an independent rerank score for each query.

    The selector is intentionally separate from the IoU quality head so local
    top-k supervision does not distort the existing quality score.
    """

    DEFAULT_SOURCES = ("base", "structured", "quality", "fused")
    DEFAULT_CANDIDATE_SOURCES = ("base", "fused", "quality")

    def __init__(self, d_model=288, hidden_dim=288, score_sources=None,
                 candidate_aware=False, candidate_sources=None,
                 direct_choice=False, source_embed_dim=8,
                 rank_features=False, separate_override_head=False,
                 override_initial_bias=-1.5,
                 candidate_context=False, candidate_context_k=5,
                 pairdelta_features=False, selector_context_dim=0,
                 context_features=False):
        super().__init__()
        self.score_sources = tuple(score_sources or self.DEFAULT_SOURCES)
        self.candidate_aware = bool(candidate_aware)
        self.direct_choice = bool(direct_choice)
        self.rank_features = bool(rank_features)
        self.pairdelta_features = bool(pairdelta_features)
        self.separate_override_head = bool(separate_override_head)
        self.candidate_context = bool(candidate_context)
        self.candidate_context_k = int(candidate_context_k)
        self.selector_context_dim = int(selector_context_dim)
        self.context_features = bool(context_features)
        self.context_feature_dim = (
            self.selector_context_dim if self.context_features else 0
        )
        self.candidate_sources = tuple(
            candidate_sources or self.DEFAULT_CANDIDATE_SOURCES
        )
        score_dim = len(self.score_sources) * 2
        feature_dim = d_model + 6 + score_dim
        self.candidate_rank_dim = (
            len(self.candidate_sources) * 3
            if self.rank_features else 0
        )
        self.candidate_pairdelta_dim = (
            len(self.candidate_sources) * 7
            if self.pairdelta_features else 0
        )
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        if self.candidate_aware or self.direct_choice:
            self.source_embedding = nn.Embedding(
                len(self.candidate_sources), source_embed_dim
            )
        else:
            self.source_embedding = None
        if self.candidate_aware:
            self.candidate_pre_context_dim = (
                feature_dim + source_embed_dim
                + self.candidate_rank_dim
                + self.candidate_pairdelta_dim
            )
            self.candidate_context_dim = (
                self.candidate_pre_context_dim * 2
                if self.candidate_context else 0
            )
            self.candidate_choice_feature_dim = (
                self.candidate_pre_context_dim
                + self.candidate_context_dim
                + self.context_feature_dim
            )
            self.candidate_choice_context_dim = (
                self.candidate_choice_feature_dim
                * len(self.candidate_sources)
            )
            self.candidate_mlp = nn.Sequential(
                nn.Linear(
                    self.candidate_pre_context_dim
                    + self.candidate_context_dim
                    + self.context_feature_dim,
                    hidden_dim,
                ),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.candidate_pre_context_dim = 0
            self.candidate_context_dim = 0
            self.candidate_choice_feature_dim = 0
            self.candidate_choice_context_dim = 0
            self.candidate_mlp = None
        if self.candidate_aware and self.separate_override_head:
            self.override_mlp = nn.Sequential(
                nn.Linear(self.candidate_choice_context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            nn.init.zeros_(self.override_mlp[-1].weight)
            nn.init.constant_(
                self.override_mlp[-1].bias,
                float(override_initial_bias),
            )
        if self.direct_choice:
            self.direct_choice_relation_dim = 6
            self.direct_choice_rank_dim = (
                len(self.candidate_sources) * 3
                if self.rank_features else 0
            )
            self.direct_choice_pairdelta_dim = (
                len(self.candidate_sources) * 7
                if self.pairdelta_features else 0
            )
            self.choice_feature_dim = (
                feature_dim + source_embed_dim + 3
                + self.direct_choice_relation_dim
                + self.direct_choice_rank_dim
                + self.direct_choice_pairdelta_dim
                + self.context_feature_dim
            )
            self.choice_context_dim = (
                self.choice_feature_dim * len(self.candidate_sources)
            )
            self.choice_mlp = nn.Sequential(
                nn.Linear(
                    self.choice_feature_dim + self.choice_context_dim,
                    hidden_dim,
                ),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            if self.separate_override_head:
                self.override_mlp = nn.Sequential(
                    nn.Linear(self.choice_context_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                nn.init.zeros_(self.override_mlp[-1].weight)
                nn.init.constant_(
                    self.override_mlp[-1].bias,
                    float(override_initial_bias),
                )
            else:
                self.override_mlp = None
        else:
            self.direct_choice_relation_dim = 0
            self.direct_choice_rank_dim = 0
            self.direct_choice_pairdelta_dim = 0
            self.choice_mlp = None
            if not (self.candidate_aware and self.separate_override_head):
                self.override_mlp = None
        if self.selector_context_dim > 0 and not self.context_features:
            self.context_choice_bias_mlp = nn.Linear(
                self.selector_context_dim, len(self.candidate_sources)
            )
            nn.init.zeros_(self.context_choice_bias_mlp.weight)
            nn.init.zeros_(self.context_choice_bias_mlp.bias)
            if self.separate_override_head:
                self.context_override_bias_mlp = nn.Linear(
                    self.selector_context_dim, 1
                )
                nn.init.zeros_(self.context_override_bias_mlp.weight)
                nn.init.zeros_(self.context_override_bias_mlp.bias)
            else:
                self.context_override_bias_mlp = None
        else:
            self.context_choice_bias_mlp = None
            self.context_override_bias_mlp = None

    @staticmethod
    def _direct_choice_relation_features(
        top_indices, top_scores, top_margins, present, choice_boxes
    ):
        B, S = top_indices.shape
        device = top_indices.device
        dtype = top_scores.dtype
        present = present.float()
        present_flat = present.squeeze(-1)
        other_mask = (~torch.eye(S, device=device, dtype=torch.bool)).float()
        pair_valid = (
            present_flat.unsqueeze(2)
            * present_flat.unsqueeze(1)
            * other_mask.view(1, S, S)
        )
        denom = pair_valid.sum(dim=2).clamp(min=1.0)

        same_query = (
            top_indices.unsqueeze(2) == top_indices.unsqueeze(1)
        ).float() * pair_valid
        same_count = same_query.sum(dim=2)
        same_any = (same_count > 0).float()
        same_fraction = same_count / denom

        top_scores = top_scores.squeeze(-1).float()
        top_margins = top_margins.squeeze(-1).float()
        other_score_mean = (
            top_scores.unsqueeze(1) * pair_valid
        ).sum(dim=2) / denom
        other_margin_mean = (
            top_margins.unsqueeze(1) * pair_valid
        ).sum(dim=2) / denom
        score_delta_to_others = top_scores - other_score_mean
        margin_delta_to_others = top_margins - other_margin_mean

        centers = choice_boxes[..., :3].float()
        sizes = choice_boxes[..., 3:6].float()
        center_delta = (
            centers.unsqueeze(2) - centers.unsqueeze(1)
        ).abs().mean(dim=-1)
        size_delta = (
            sizes.unsqueeze(2) - sizes.unsqueeze(1)
        ).abs().mean(dim=-1)
        mean_center_delta = (center_delta * pair_valid).sum(dim=2) / denom
        mean_size_delta = (size_delta * pair_valid).sum(dim=2) / denom

        relation_features = torch.stack([
            same_any,
            same_fraction,
            score_delta_to_others,
            margin_delta_to_others,
            mean_center_delta,
            mean_size_delta,
        ], dim=-1).to(dtype=dtype)
        return relation_features * present.to(dtype=dtype)

    @staticmethod
    def _direct_choice_rank_features(
        top_indices, source_score_stack, present
    ):
        B, S = top_indices.shape
        if source_score_stack.dim() != 3:
            raise ValueError(
                "source_score_stack must have shape (B, S, Q)"
            )
        if source_score_stack.shape[:2] != (B, S):
            raise ValueError(
                "source_score_stack must align with top_indices"
            )
        source_scores = source_score_stack.float()
        device = source_scores.device
        dtype = source_scores.dtype
        Q = source_scores.shape[-1]
        safe_top_indices = top_indices.long().clamp(min=0, max=max(Q - 1, 0))
        gather_idx = safe_top_indices.unsqueeze(1).expand(B, S, S)
        scores_at_candidates = torch.gather(
            source_scores, 2, gather_idx
        ).transpose(1, 2)

        source_scores_by_candidate = source_scores.unsqueeze(2)
        candidate_values = (
            scores_at_candidates.transpose(1, 2).unsqueeze(-1)
        )
        rank_percentile = (
            source_scores_by_candidate <= candidate_values
        ).float().mean(dim=-1).transpose(1, 2)

        source_top = source_scores.max(dim=-1).values
        delta_to_top = scores_at_candidates - source_top.unsqueeze(1)

        source_mean = source_scores.mean(dim=-1)
        source_std = source_scores.std(
            dim=-1, unbiased=False
        ).clamp(min=1e-6)
        z_score = (
            scores_at_candidates - source_mean.unsqueeze(1)
        ) / source_std.unsqueeze(1)

        present_flat = present.squeeze(-1).float()
        valid = (
            present_flat.unsqueeze(2) * present_flat.unsqueeze(1)
        ).to(dtype=dtype)
        rank_features = torch.stack([
            rank_percentile,
            delta_to_top,
            z_score,
        ], dim=-1)
        rank_features = rank_features * valid.unsqueeze(-1)
        return rank_features.reshape(B, S, S * 3).to(device=device)

    @staticmethod
    def _direct_choice_pairdelta_features(
        top_indices, top_scores, top_margins, present, choice_boxes,
        source_score_stack
    ):
        B, S = top_indices.shape
        if source_score_stack.dim() != 3:
            raise ValueError(
                "source_score_stack must have shape (B, S, Q)"
            )
        if source_score_stack.shape[:2] != (B, S):
            raise ValueError(
                "source_score_stack must align with top_indices"
            )
        source_scores = source_score_stack.float()
        device = source_scores.device
        dtype = top_scores.dtype
        Q = source_scores.shape[-1]

        present_flat = present.squeeze(-1).float().to(device=device)
        pair_valid = (
            present_flat.unsqueeze(2) * present_flat.unsqueeze(1)
        )
        top_indices = top_indices.to(device=device)
        safe_top_indices = top_indices.long().clamp(min=0, max=max(Q - 1, 0))
        gather_idx = safe_top_indices.unsqueeze(1).expand(B, S, S)
        scores_at_current_top = torch.gather(
            source_scores, 2, gather_idx
        ).transpose(1, 2)

        top_scores = top_scores.squeeze(-1).float().to(device=device)
        top_margins = top_margins.squeeze(-1).float().to(device=device)
        same_query = (
            top_indices.unsqueeze(2) == top_indices.unsqueeze(1)
        ).float() * pair_valid
        top_score_delta = (
            top_scores.unsqueeze(2) - top_scores.unsqueeze(1)
        ) * pair_valid
        top_margin_delta = (
            top_margins.unsqueeze(2) - top_margins.unsqueeze(1)
        ) * pair_valid
        score_at_top_delta = (
            top_scores.unsqueeze(2) - scores_at_current_top
        ) * pair_valid

        centers = choice_boxes[..., :3].float().to(device=device)
        sizes = choice_boxes[..., 3:6].float().to(device=device)
        center_l1 = (
            centers.unsqueeze(2) - centers.unsqueeze(1)
        ).abs().sum(dim=-1) * pair_valid
        size_l1 = (
            sizes.unsqueeze(2) - sizes.unsqueeze(1)
        ).abs().sum(dim=-1) * pair_valid
        volumes = sizes.clamp(min=0.0).prod(dim=-1)
        volume_delta = (
            volumes.unsqueeze(2) - volumes.unsqueeze(1)
        ) * pair_valid

        pairdelta = torch.stack([
            same_query,
            top_score_delta,
            top_margin_delta,
            score_at_top_delta,
            center_l1,
            size_l1,
            volume_delta,
        ], dim=-1)
        return pairdelta.reshape(B, S, S * 7).to(dtype=dtype)

    @staticmethod
    def _candidate_rank_features(source_score_stack, present):
        if source_score_stack.dim() != 3:
            raise ValueError(
                "source_score_stack must have shape (B, S, Q)"
            )
        source_scores = source_score_stack.float()
        B, S, Q = source_scores.shape
        device = source_scores.device
        dtype = source_scores.dtype

        rank_percentile = (
            source_scores.unsqueeze(2) <= source_scores.unsqueeze(-1)
        ).float().mean(dim=-1).transpose(1, 2)

        source_top = source_scores.max(dim=-1).values
        delta_to_top = (
            source_scores - source_top.unsqueeze(-1)
        ).transpose(1, 2)

        source_mean = source_scores.mean(dim=-1)
        source_std = source_scores.std(
            dim=-1, unbiased=False
        ).clamp(min=1e-6)
        z_score = (
            (source_scores - source_mean.unsqueeze(-1))
            / source_std.unsqueeze(-1)
        ).transpose(1, 2)

        present_flat = present.squeeze(-1).float().to(device=device)
        valid = present_flat.unsqueeze(1).expand(B, Q, S).to(dtype=dtype)
        rank_features = torch.stack([
            rank_percentile,
            delta_to_top,
            z_score,
        ], dim=-1)
        rank_features = rank_features * valid.unsqueeze(-1)
        return rank_features.reshape(B, Q, S * 3).to(device=device)

    @staticmethod
    def _candidate_pairdelta_features(
        source_score_stack, present, pred_boxes
    ):
        if source_score_stack.dim() != 3:
            raise ValueError(
                "source_score_stack must have shape (B, S, Q)"
            )
        source_scores = source_score_stack.float()
        B, S, Q = source_scores.shape
        device = source_scores.device
        dtype = source_scores.dtype
        pred_boxes = pred_boxes.float().to(device=device)

        present_flat = present.squeeze(-1).float().to(device=device)
        pair_valid = (
            present_flat.unsqueeze(2) * present_flat.unsqueeze(1)
        )
        pair_valid_q = pair_valid.unsqueeze(-1)

        score_delta = (
            source_scores.unsqueeze(2) - source_scores.unsqueeze(1)
        ) * pair_valid_q

        rank_percentile = (
            source_scores.unsqueeze(2) <= source_scores.unsqueeze(-1)
        ).float().mean(dim=-1)
        rank_delta = (
            rank_percentile.unsqueeze(2)
            - rank_percentile.unsqueeze(1)
        ) * pair_valid_q

        source_top = source_scores.max(dim=-1).values
        delta_to_top = source_top.unsqueeze(-1) - source_scores
        delta_to_top_delta = (
            delta_to_top.unsqueeze(2) - delta_to_top.unsqueeze(1)
        ) * pair_valid_q

        top_score_delta = (
            source_top.unsqueeze(2) - source_top.unsqueeze(1)
        ).unsqueeze(-1).expand(B, S, S, Q) * pair_valid_q

        top_idx = source_scores.argmax(dim=-1)
        batch_idx = torch.arange(B, device=device).view(B, 1)
        top_boxes = pred_boxes[batch_idx, top_idx]
        centers = top_boxes[..., :3]
        sizes = top_boxes[..., 3:6]
        center_l1 = (
            centers.unsqueeze(2) - centers.unsqueeze(1)
        ).abs().sum(dim=-1).unsqueeze(-1).expand(B, S, S, Q)
        center_l1 = center_l1 * pair_valid_q
        size_l1 = (
            sizes.unsqueeze(2) - sizes.unsqueeze(1)
        ).abs().sum(dim=-1).unsqueeze(-1).expand(B, S, S, Q)
        size_l1 = size_l1 * pair_valid_q
        volumes = sizes.clamp(min=0.0).prod(dim=-1)
        volume_delta = (
            volumes.unsqueeze(2) - volumes.unsqueeze(1)
        ).unsqueeze(-1).expand(B, S, S, Q) * pair_valid_q

        pairdelta = torch.stack([
            score_delta,
            rank_delta,
            delta_to_top_delta,
            top_score_delta,
            center_l1,
            size_l1,
            volume_delta,
        ], dim=-1)
        pairdelta = pairdelta.permute(0, 1, 3, 2, 4)
        return pairdelta.reshape(B, S, Q, S * 7).to(dtype=dtype)

    @staticmethod
    def _candidate_pool_context(
        candidate_features, source_scores, candidate_sources, k
    ):
        if candidate_features.dim() != 4:
            raise ValueError(
                "candidate_features must have shape (B, S, Q, C)"
            )
        B, S, Q, C = candidate_features.shape
        device = candidate_features.device
        dtype = candidate_features.dtype
        mask = torch.zeros(B, S, Q, device=device, dtype=torch.bool)
        top_k = max(1, min(int(k), Q))
        for source_idx, source in enumerate(candidate_sources[:S]):
            scores = source_scores.get(source, None)
            if scores is None or scores.shape != (B, Q):
                continue
            scores = torch.nan_to_num(
                scores.detach().float(),
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            )
            top_idx = torch.topk(scores, top_k, dim=1).indices
            mask[:, source_idx].scatter_(1, top_idx, True)

        mask_f = mask.unsqueeze(-1).to(dtype=dtype)
        denom = mask_f.sum(dim=(1, 2)).clamp(min=1.0)
        mean_context = (candidate_features * mask_f).sum(
            dim=(1, 2)
        ) / denom

        min_value = torch.finfo(dtype).min
        masked_features = candidate_features.masked_fill(
            ~mask.unsqueeze(-1), min_value
        )
        max_context = masked_features.amax(dim=(1, 2))
        has_candidates = mask.flatten(1).any(dim=1)
        max_context = torch.where(
            has_candidates.unsqueeze(-1),
            max_context,
            torch.zeros_like(max_context),
        )

        context = torch.cat([mean_context, max_context], dim=-1)
        return context.view(B, 1, 1, 2 * C).expand(B, S, Q, 2 * C)

    @staticmethod
    def _candidate_choice_scores(
        selector_source_scores, source_scores=None, candidate_sources=None
    ):
        if selector_source_scores.dim() != 3:
            raise ValueError(
                "selector_source_scores must have shape (B, S, Q)"
            )
        if source_scores is None or candidate_sources is None:
            return selector_source_scores.max(dim=-1).values

        B, S, Q = selector_source_scores.shape
        choice_rows = []
        for source_idx, source in enumerate(tuple(candidate_sources)[:S]):
            scores = source_scores.get(source, None)
            if scores is None:
                choice_rows.append(
                    selector_source_scores[:, source_idx].max(dim=-1).values
                )
                continue
            if scores.shape != (B, Q):
                raise ValueError(
                    f"{source} scores must have shape {(B, Q)} for "
                    "candidate-aware source choice"
                )
            scores = torch.nan_to_num(
                scores.detach().float().to(selector_source_scores.device),
                nan=0.0, posinf=1e4, neginf=-1e4,
            )
            top_idx = scores.argmax(dim=-1)
            choice_rows.append(
                selector_source_scores[:, source_idx]
                .gather(1, top_idx.view(B, 1))
                .squeeze(1)
            )
        return torch.stack(choice_rows, dim=1)

    @staticmethod
    def _candidate_choice_context(
        candidate_features, selector_source_scores, source_scores,
        candidate_sources
    ):
        if candidate_features.dim() != 4:
            raise ValueError(
                "candidate_features must have shape (B, S, Q, C)"
            )
        if selector_source_scores.dim() != 3:
            raise ValueError(
                "selector_source_scores must have shape (B, S, Q)"
            )
        B, S, Q, _ = candidate_features.shape
        if selector_source_scores.shape[:2] != (B, S):
            raise ValueError(
                "selector_source_scores must align with candidate_features"
            )
        batch_idx = torch.arange(B, device=candidate_features.device)
        choice_rows = []
        for source_idx, source in enumerate(tuple(candidate_sources)[:S]):
            scores = source_scores.get(source, None)
            if scores is None or scores.shape != (B, Q):
                top_idx = selector_source_scores[:, source_idx].argmax(dim=1)
            else:
                scores = torch.nan_to_num(
                    scores.detach().float().to(candidate_features.device),
                    nan=0.0, posinf=1e4, neginf=-1e4,
                )
                top_idx = scores.argmax(dim=-1)
            choice_rows.append(
                candidate_features[batch_idx, source_idx, top_idx]
            )
        return torch.stack(choice_rows, dim=1).reshape(B, -1)

    def _selector_context(self, selector_context, batch_size, device):
        if selector_context is None or self.selector_context_dim <= 0:
            return None
        if selector_context.dim() != 2:
            raise ValueError("selector_context must have shape (B, C)")
        if selector_context.shape != (batch_size, self.selector_context_dim):
            raise ValueError(
                "selector_context must have shape {}".format(
                    (batch_size, self.selector_context_dim)
                )
            )
        return selector_context.detach().float().to(device=device)

    def _context_choice_bias(self, selector_context, batch_size, device):
        context = self._selector_context(selector_context, batch_size, device)
        if context is None or self.context_choice_bias_mlp is None:
            return None
        return self.context_choice_bias_mlp(context)

    def _context_override_bias(self, selector_context, batch_size, device):
        context = self._selector_context(selector_context, batch_size, device)
        if context is None or self.context_override_bias_mlp is None:
            return None
        return self.context_override_bias_mlp(context).squeeze(-1)

    def _context_feature_block(self, selector_context, batch_size, device):
        context = self._selector_context(selector_context, batch_size, device)
        if (
            context is None
            or not self.context_features
            or self.context_feature_dim <= 0
        ):
            return None
        return context

    def forward(
        self, query_feats, pred_boxes, source_scores=None,
        selector_context=None
    ):
        with torch.cuda.amp.autocast(enabled=False):
            query_feats = query_feats.detach().float()
            pred_boxes = pred_boxes.detach().float()
            B, Q = query_feats.shape[:2]
            device = query_feats.device
            context_choice_bias = self._context_choice_bias(
                selector_context, B, device
            )
            context_override_bias = self._context_override_bias(
                selector_context, B, device
            )
            context_feature_block = self._context_feature_block(
                selector_context, B, device
            )
            score_features = []
            source_scores = source_scores or {}
            for source in self.score_sources:
                scores = source_scores.get(source, None)
                if scores is None:
                    values = torch.zeros(B, Q, 1, device=device)
                    present = torch.zeros(B, Q, 1, device=device)
                else:
                    scores = torch.nan_to_num(
                        scores.detach().float(),
                        nan=0.0,
                        posinf=1e4,
                        neginf=-1e4,
                    )
                    values = scores.unsqueeze(-1)
                    present = torch.ones_like(values)
                score_features.extend([values, present])
            features = torch.cat(
                [query_feats, pred_boxes] + score_features, dim=-1
            )
            selector_scores = self.mlp(features).squeeze(-1)
            out = {"selector_scores": selector_scores}
            if self.candidate_aware or self.direct_choice:
                out["selector_choice_source_names"] = self.candidate_sources
            if self.candidate_aware:
                source_ids = torch.arange(
                    len(self.candidate_sources), device=device
                )
                source_emb = self.source_embedding(source_ids)
                source_emb = source_emb.view(1, -1, 1, source_emb.shape[-1])
                source_emb = source_emb.expand(B, -1, Q, -1)
                candidate_features = features.unsqueeze(1).expand(
                    B, len(self.candidate_sources), Q, features.shape[-1]
                )
                candidate_feature_parts = [candidate_features, source_emb]
                if (
                    self.candidate_rank_dim > 0
                    or self.candidate_pairdelta_dim > 0
                ):
                    source_score_rows = []
                    present_rows = []
                    for source in self.candidate_sources:
                        scores = source_scores.get(source, None)
                        if scores is None:
                            scores = torch.zeros(B, Q, device=device)
                            present = torch.zeros(B, 1, device=device)
                        else:
                            scores = torch.nan_to_num(
                                scores.detach().float(),
                                nan=0.0,
                                posinf=1e4,
                                neginf=-1e4,
                            )
                            present = torch.ones(B, 1, device=device)
                        source_score_rows.append(scores)
                        present_rows.append(present)
                    source_score_stack = torch.stack(source_score_rows, dim=1)
                    present_stack = torch.stack(present_rows, dim=1)
                else:
                    source_score_stack = None
                    present_stack = None
                if self.candidate_rank_dim > 0:
                    rank_features = self._candidate_rank_features(
                        source_score_stack,
                        present_stack,
                    )
                    rank_features = rank_features.unsqueeze(1).expand(
                        B, len(self.candidate_sources), Q,
                        rank_features.shape[-1],
                    )
                    candidate_feature_parts.append(rank_features)
                if self.candidate_pairdelta_dim > 0:
                    candidate_feature_parts.append(
                        self._candidate_pairdelta_features(
                            source_score_stack,
                            present_stack,
                            pred_boxes,
                        )
                    )
                candidate_features = torch.cat(candidate_feature_parts, dim=-1)
                if self.candidate_context:
                    context_features = self._candidate_pool_context(
                        candidate_features,
                        source_scores,
                        self.candidate_sources,
                        self.candidate_context_k,
                    )
                    candidate_features = torch.cat(
                        [candidate_features, context_features], dim=-1
                    )
                if context_feature_block is not None:
                    context_features = context_feature_block.view(
                        B, 1, 1, -1
                    ).expand(
                        B,
                        len(self.candidate_sources),
                        Q,
                        context_feature_block.shape[-1],
                    )
                    candidate_features = torch.cat(
                        [candidate_features, context_features], dim=-1
                    )
                selector_source_scores = self.candidate_mlp(
                    candidate_features
                ).squeeze(-1)
                out["selector_source_scores"] = selector_source_scores
                out["selector_choice_scores"] = self._candidate_choice_scores(
                    selector_source_scores,
                    source_scores,
                    self.candidate_sources,
                )
                if context_choice_bias is not None:
                    out["selector_choice_scores"] = (
                        out["selector_choice_scores"] + context_choice_bias
                    )
                if self.separate_override_head:
                    choice_context = self._candidate_choice_context(
                        candidate_features,
                        selector_source_scores,
                        source_scores,
                        self.candidate_sources,
                    )
                    out["selector_choice_override_logit"] = (
                        self.override_mlp(choice_context).squeeze(-1)
                    )
                    if context_override_bias is not None:
                        out["selector_choice_override_logit"] = (
                            out["selector_choice_override_logit"]
                            + context_override_bias
                        )
            if self.direct_choice:
                source_ids = torch.arange(
                    len(self.candidate_sources), device=device
                )
                source_emb = self.source_embedding(source_ids)
                batch_idx = torch.arange(B, device=device)
                choice_rows = []
                top_indices_rows = []
                top_score_rows = []
                top_margin_rows = []
                present_rows = []
                choice_box_rows = []
                source_score_rows = []
                for source_idx, source in enumerate(self.candidate_sources):
                    scores = source_scores.get(source, None)
                    if scores is None:
                        scores = torch.zeros(B, Q, device=device)
                        top_idx = torch.zeros(
                            B, device=device, dtype=torch.long
                        )
                        top_score = torch.zeros(B, 1, device=device)
                        top_margin = torch.zeros(B, 1, device=device)
                        present = torch.zeros(B, 1, device=device)
                    else:
                        scores = torch.nan_to_num(
                            scores.detach().float(),
                            nan=0.0,
                            posinf=1e4,
                            neginf=-1e4,
                        )
                        top_count = min(2, Q)
                        top_values, top_indices = torch.topk(
                            scores, top_count, dim=1
                        )
                        top_idx = top_indices[:, 0]
                        top_score = top_values[:, 0:1]
                        if top_count > 1:
                            top_margin = (
                                top_values[:, 0] - top_values[:, 1]
                            ).unsqueeze(-1)
                        else:
                            top_margin = torch.zeros(B, 1, device=device)
                        present = torch.ones(B, 1, device=device)
                    source_score_rows.append(scores)
                    top_indices_rows.append(top_idx)
                    top_score_rows.append(top_score)
                    top_margin_rows.append(top_margin)
                    present_rows.append(present)
                    choice_box_rows.append(pred_boxes[batch_idx, top_idx])
                    source_feature = source_emb[source_idx].view(1, -1).expand(
                        B, -1
                    )
                    choice_rows.append(torch.cat([
                        features[batch_idx, top_idx],
                        source_feature,
                        top_score,
                        top_margin,
                        present,
                    ], dim=-1))
                choice_features = torch.stack(choice_rows, dim=1)
                relation_features = self._direct_choice_relation_features(
                    torch.stack(top_indices_rows, dim=1),
                    torch.stack(top_score_rows, dim=1),
                    torch.stack(top_margin_rows, dim=1),
                    torch.stack(present_rows, dim=1),
                    torch.stack(choice_box_rows, dim=1),
                )
                choice_feature_parts = [choice_features, relation_features]
                if self.rank_features:
                    rank_features = self._direct_choice_rank_features(
                        torch.stack(top_indices_rows, dim=1),
                        torch.stack(source_score_rows, dim=1),
                        torch.stack(present_rows, dim=1),
                    )
                    choice_feature_parts.append(rank_features)
                if self.direct_choice_pairdelta_dim > 0:
                    choice_feature_parts.append(
                        self._direct_choice_pairdelta_features(
                            torch.stack(top_indices_rows, dim=1),
                            torch.stack(top_score_rows, dim=1),
                            torch.stack(top_margin_rows, dim=1),
                            torch.stack(present_rows, dim=1),
                            torch.stack(choice_box_rows, dim=1),
                            torch.stack(source_score_rows, dim=1),
                        )
                    )
                if context_feature_block is not None:
                    choice_feature_parts.append(
                        context_feature_block.view(B, 1, -1).expand(
                            B,
                            len(self.candidate_sources),
                            context_feature_block.shape[-1],
                        )
                    )
                choice_features = torch.cat(choice_feature_parts, dim=-1)
                choice_context = choice_features.reshape(B, -1)
                if self.override_mlp is not None:
                    out["selector_choice_override_logit"] = (
                        self.override_mlp(choice_context).squeeze(-1)
                    )
                    if context_override_bias is not None:
                        out["selector_choice_override_logit"] = (
                            out["selector_choice_override_logit"]
                            + context_override_bias
                        )
                choice_context = choice_context.unsqueeze(1).expand(
                    B, len(self.candidate_sources), -1
                )
                choice_features = torch.cat(
                    [choice_features, choice_context], dim=-1
                )
                out["selector_choice_scores"] = self.choice_mlp(
                    choice_features
                ).squeeze(-1)
                if context_choice_bias is not None:
                    out["selector_choice_scores"] = (
                        out["selector_choice_scores"] + context_choice_bias
                    )
        return out
