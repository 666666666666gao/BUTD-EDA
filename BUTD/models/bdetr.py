# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast

from .backbone_module import Pointnet2Backbone
from .modules import (
    PointsObjClsModule, GeneralSamplingModule,
    ClsAgnosticPredictHead, PositionEmbeddingLearned
)
from .encoder_decoder_layers import (
    BiEncoder, BiEncoderLayer, BiDecoderLayer
)
from .structured_slots import StructuredSlotBuilder
from .structured_losses import DHCLossModule
from .acd_head import LateACDHead, resolve_acd_base_scores
from .quality_head import QualityHead
from .source_pool_selector import (
    SourcePoolSelectorHead,
    compute_contrastive_token_base_scores,
    compute_soft_token_base_scores,
)
from .detector_policy_sources import (
    DetectorPolicyAdapterHead,
    build_detector_policy_features,
    build_detector_policy_score_sources,
)
from .sacr_head import SACRHead
from .reliability_fusion import ReliabilityFusion


def _build_seeded_optional_module(seed, factory):
    """Initialize an optional head without perturbing shared-model RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return factory()


def build_token_span_tensors(tokenized, span_batches, device, min_slots=1):
    """Convert character-level span dicts to padded token-span tensors."""
    batch_size = tokenized['input_ids'].shape[0]
    max_spans = max(
        (len(spans) for spans in span_batches if isinstance(spans, list)),
        default=0
    )
    max_spans = max(max_spans, min_slots)
    span_tensor = torch.full(
        (batch_size, max_spans, 2),
        -1,
        dtype=torch.long,
        device=device
    )

    for b in range(batch_size):
        spans = span_batches[b] if b < len(span_batches) and isinstance(span_batches[b], list) else []
        for i, span in enumerate(spans[:max_spans]):
            if not isinstance(span, dict):
                continue
            start_char = int(span.get('start', 0))
            end_char = int(span.get('end', 0))
            if end_char <= start_char:
                continue

            beg_pos = tokenized.char_to_token(b, start_char)
            end_pos = tokenized.char_to_token(b, end_char - 1)

            if beg_pos is None:
                for offset in (1, 2):
                    probe = start_char + offset
                    if probe >= end_char:
                        break
                    beg_pos = tokenized.char_to_token(b, probe)
                    if beg_pos is not None:
                        break

            if end_pos is None:
                for offset in (1, 2):
                    probe = end_char - 1 - offset
                    if probe < start_char:
                        break
                    end_pos = tokenized.char_to_token(b, probe)
                    if end_pos is not None:
                        break

            if beg_pos is None or end_pos is None or end_pos < beg_pos:
                continue

            span_tensor[b, i, 0] = beg_pos
            span_tensor[b, i, 1] = end_pos + 1

    return span_tensor


class BeaUTyDETR(nn.Module):
    """
    3D language grounder.

    Args:
        num_class (int): number of semantics classes to predict
        num_obj_class (int): number of object classes
        input_feature_dim (int): feat_dim of pointcloud (without xyz)
        num_queries (int): Number of queries generated
        num_decoder_layers (int): number of decoder layers
        self_position_embedding (str or None): how to compute pos embeddings
        contrastive_align_loss (bool): contrast queries and token features
        d_model (int): dimension of features
        butd (bool): use detected box stream
        pointnet_ckpt (str or None): path to pre-trained pp++ checkpoint
        self_attend (bool): add self-attention in encoder
    """

    @staticmethod
    def _detector_policy_target_cid(inputs):
        if bool(inputs.get('train', False)):
            return inputs.get('target_cid', None)
        source = str(inputs.get('eval_target_cid_source', 'gt')).strip().lower()
        if source == 'text':
            return inputs.get('text_target_cid', None)
        return inputs.get('target_cid', None)

    def __init__(self, num_class=256, num_obj_class=485,
                 input_feature_dim=3,
                 num_queries=256,
                 num_decoder_layers=6, self_position_embedding='loc_learned',
                 contrastive_align_loss=True,
                 d_model=288, butd=True, pointnet_ckpt=None,
                 self_attend=True,
                 # --- S2S-ACD-DHC structured reasoning ---
                 use_structured_slots=False,
                 use_late_acd=False,
                 slot_pooling='attention',
                 max_rel_anchor_pairs=3,
                 acd_top_m_targets=32,
                 acd_top_k_anchors=16,
                 acd_geo_dim=16,
                 acd_hidden_dim=288,
                 acd_global_residual_alpha=0.5,
                 acd_use_confidence_fusion=False,
                 acd_warmup_steps=5000,
                 acd_initial_alpha=0.05,
                 acd_ea_scale=1.0,
                 acd_pool_ea_multiplier=1.0,
                 acd_final_ea_multiplier=1.0,
                 acd_disable_struct_rerank=False,
                 acd_base_score_source='contrastive',
                 dhc_margin_min=0.0,
                 dhc_temperature_max=0.0,
                 structured_debug=False,
                 use_quality_head=False,
                 use_sacr=False,
                 sacr_top_m_targets=32,
                 sacr_top_k_anchors=16,
                 sacr_hidden_dim=288,
                 sacr_geo_dim=16,
                 sacr_disable_relation=False,
                 use_rapf=False,
                 rapf_hidden_dim=128,
                 rapf_initial_gate_bias=-2.0,
                 rapf_use_quality=False,
                 rapf_quality_weight=0.25,
                 rapf_struct_residual_clip=2.0,
                 rapf_generic_gate_cap=0.35,
                 rapf_quality_anchor_structured_residual=False,
                 use_qahnl=False,
                 qahnl_score_source='fused',
                 use_source_pool_selector=False,
                 source_pool_selector_hidden_dim=288,
                 source_pool_selector_candidate_aware=False,
                 source_pool_selector_direct_choice=False,
                 source_pool_selector_include_contrastive_choice=False,
                 source_pool_selector_rank_features=False,
                 source_pool_selector_pairdelta_features=False,
                 source_pool_selector_candidate_context=False,
                 source_pool_selector_candidate_context_k=5,
                 source_pool_selector_include_detector_policy_choice=False,
                 source_pool_selector_choice_sources=None,
                 source_pool_selector_text_context=False,
                 source_pool_selector_metadata_context=False,
                 source_pool_selector_context_features=False,
                 source_pool_selector_separate_override_head=False,
                 source_pool_selector_override_initial_bias=-1.5,
                 use_detector_policy_adapter=False,
                 use_detector_policy_teacher=False,
                 detector_policy_adapter_context=False,
                 detector_policy_adapter_hidden_dim=32,
                 detector_policy_adapter_delta_scale=0.25):
        """Initialize layers."""
        super().__init__()

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.self_position_embedding = self_position_embedding
        self.contrastive_align_loss = contrastive_align_loss
        self.butd = butd

        # Visual encoder
        self.backbone_net = Pointnet2Backbone(
            input_feature_dim=input_feature_dim,
            width=1
        )
        if input_feature_dim == 3 and pointnet_ckpt is not None:
            self.backbone_net.load_state_dict(
                torch.load(pointnet_ckpt, map_location='cpu'), strict=False
            )

        # Text Encoder (single-path RoBERTa)
        t_type = "/root/autodl-tmp/DATA_ROOT/roberta-base"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type)
        self.text_encoder = RobertaModel.from_pretrained(t_type)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1)
        )

        # Box encoder
        if self.butd:
            self.butd_class_embeddings = nn.Embedding(num_obj_class, 768)
            saved_embeddings = torch.from_numpy(np.load(
                'data/class_embeddings3d.npy', allow_pickle=True
            ))
            self.butd_class_embeddings.weight.data.copy_(saved_embeddings)
            self.butd_class_embeddings.requires_grad = False
            self.class_embeddings = nn.Linear(768, d_model - 128)
            self.box_embeddings = PositionEmbeddingLearned(6, 128)

        # Cross-encoder
        self.pos_embed = PositionEmbeddingLearned(3, d_model)
        bi_layer = BiEncoderLayer(
            d_model, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=256,
            self_attend_lang=self_attend, self_attend_vis=self_attend,
            use_butd_enc_attn=butd
        )
        self.cross_encoder = BiEncoder(bi_layer, 3)

        # Query initialization
        self.points_obj_cls = PointsObjClsModule(d_model)
        self.gsample_module = GeneralSamplingModule()
        self.decoder_query_proj = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Proposal (layer for size and center)
        self.proposal_head = ClsAgnosticPredictHead(
            num_class, 1, num_queries, d_model,
            objectness=False, heading=False,
            compute_sem_scores=True
        )

        # Transformer decoder layers
        self.use_structured_slots = use_structured_slots
        self.use_late_acd = use_late_acd
        self.acd_base_score_source = acd_base_score_source
        self.use_quality_head = use_quality_head
        self.use_sacr = use_sacr
        self.use_rapf = use_rapf
        self.use_qahnl = use_qahnl
        self.qahnl_score_source = qahnl_score_source
        self.use_source_pool_selector = use_source_pool_selector
        self.use_detector_policy_adapter = bool(use_detector_policy_adapter)
        self.use_detector_policy_teacher = bool(use_detector_policy_teacher)
        self.detector_policy_adapter_context = bool(
            detector_policy_adapter_context
        )
        self.source_pool_selector_text_context = bool(
            source_pool_selector_text_context
        )
        self.source_pool_selector_metadata_context = bool(
            source_pool_selector_metadata_context
        )
        self.structured_debug = structured_debug
        self.needs_span_token_alignment = self.use_structured_slots
        self.needs_base_grounding_scores = (
            self.use_sacr
            or self.use_rapf
            or self.use_source_pool_selector
            or self.use_detector_policy_adapter
            or (self.use_quality_head and self.contrastive_align_loss)
            or (self.use_qahnl and self.qahnl_score_source == 'base')
        )

        if use_structured_slots:
            self.structured_slot_builder = _build_seeded_optional_module(
                101,
                lambda: StructuredSlotBuilder(
                    d_model=d_model,
                    pooling=slot_pooling,
                    max_pairs=max_rel_anchor_pairs,
                ),
            )
        else:
            self.structured_slot_builder = None

        if use_late_acd:
            self.acd_head = LateACDHead(
                d_model=d_model,
                geo_dim=acd_geo_dim,
                hidden_dim=acd_hidden_dim,
                top_m_targets=acd_top_m_targets,
                top_k_anchors=acd_top_k_anchors,
                use_confidence_fusion=acd_use_confidence_fusion,
                global_residual_alpha=acd_global_residual_alpha,
                warmup_steps=acd_warmup_steps,
                initial_alpha=acd_initial_alpha,
                ea_scale=acd_ea_scale,
                pool_ea_multiplier=acd_pool_ea_multiplier,
                final_ea_multiplier=acd_final_ea_multiplier,
                disable_struct_rerank=acd_disable_struct_rerank,
                proj_dim=64  # matches contrastive_align_projection output dim
            )
            # DHC loss module holds learned margins/temperatures
            self.dhc_loss_module = DHCLossModule(
                margin_min=dhc_margin_min,
                temperature_max=dhc_temperature_max
            )
        else:
            self.acd_head = None
            self.dhc_loss_module = None

        self.quality_head = (
            _build_seeded_optional_module(
                103, lambda: QualityHead(d_model=d_model)
            )
            if use_quality_head else None
        )
        source_pool_candidate_sources = None
        if source_pool_selector_choice_sources is not None:
            source_pool_candidate_sources = tuple(
                str(source).strip()
                for source in source_pool_selector_choice_sources
                if str(source).strip()
            )
        elif (
            source_pool_selector_include_contrastive_choice
            or source_pool_selector_include_detector_policy_choice
        ):
            candidate_sources = ['base', 'fused', 'quality']
            if source_pool_selector_include_contrastive_choice:
                candidate_sources.append('contrastive_base')
            if source_pool_selector_include_detector_policy_choice:
                candidate_sources.extend([
                    'detector_countboost',
                    'detector_run174boost',
                    'detector_countsplit',
                    'detector_countsplit_lowonly',
                    'detector_countsplit_guarded',
                    'detector_countsplit_guarded_allcount',
                    'detector_jointtight',
                    'detector_strongcoarse',
                    'detector_confblend035',
                    'detector_confblend05',
                ])
            source_pool_candidate_sources = tuple(candidate_sources)
        self.source_pool_selector = (
            SourcePoolSelectorHead(
                d_model=d_model,
                hidden_dim=source_pool_selector_hidden_dim,
                candidate_aware=source_pool_selector_candidate_aware,
                direct_choice=source_pool_selector_direct_choice,
                candidate_sources=source_pool_candidate_sources,
                rank_features=source_pool_selector_rank_features,
                pairdelta_features=source_pool_selector_pairdelta_features,
                candidate_context=source_pool_selector_candidate_context,
                candidate_context_k=source_pool_selector_candidate_context_k,
                selector_context_dim=(
                    (d_model if source_pool_selector_text_context else 0)
                    + (8 if source_pool_selector_metadata_context else 0)
                ),
                context_features=source_pool_selector_context_features,
                separate_override_head=(
                    source_pool_selector_separate_override_head
                ),
                override_initial_bias=(
                    source_pool_selector_override_initial_bias
                ),
            )
            if use_source_pool_selector else None
        )
        detector_context_dim = (
            d_model + 8 if self.detector_policy_adapter_context else 0
        )
        self.detector_policy_adapter = (
            DetectorPolicyAdapterHead(
                context_dim=detector_context_dim,
                hidden_dim=detector_policy_adapter_hidden_dim,
                delta_scale=detector_policy_adapter_delta_scale,
            )
            if self.use_detector_policy_adapter else None
        )
        self.sacr_head = (
            _build_seeded_optional_module(
                107,
                lambda: SACRHead(
                    d_model=d_model,
                    hidden_dim=sacr_hidden_dim,
                    top_m_targets=sacr_top_m_targets,
                    top_k_anchors=sacr_top_k_anchors,
                    geo_dim=sacr_geo_dim,
                    disable_relation=sacr_disable_relation,
                ),
            )
            if use_sacr else None
        )
        self.reliability_fusion = (
            _build_seeded_optional_module(
                109,
                lambda: ReliabilityFusion(
                    hidden_dim=rapf_hidden_dim,
                    initial_gate_bias=rapf_initial_gate_bias,
                    use_quality=rapf_use_quality,
                    quality_weight=rapf_quality_weight,
                    generic_gate_cap=rapf_generic_gate_cap,
                    residual_clip=rapf_struct_residual_clip,
                    quality_anchor_structured_residual=(
                        rapf_quality_anchor_structured_residual
                    ),
                ),
            )
            if use_rapf else None
        )

        self.decoder = nn.ModuleList()
        for layer_idx in range(self.num_decoder_layers):
            self.decoder.append(BiDecoderLayer(
                d_model, n_heads=8, dim_feedforward=256,
                dropout=0.1, activation="relu",
                self_position_embedding=self_position_embedding, butd=self.butd
            ))

        # Prediction heads
        self.prediction_heads = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.prediction_heads.append(ClsAgnosticPredictHead(
                num_class, 1, num_queries, d_model,
                objectness=False, heading=False,
                compute_sem_scores=True
            ))

        # Extra layers for contrastive losses
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )
            self.contrastive_align_projection_text = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )

        # Init
        self.init_bn_momentum()

    @staticmethod
    def _metadata_bool_mask(inputs, key, batch_size, device):
        values = inputs.get(key, None)
        if values is None:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        if torch.is_tensor(values):
            return values.to(device=device).bool().view(-1)[:batch_size]
        if isinstance(values, (list, tuple)):
            out = []
            for item in list(values)[:batch_size]:
                if isinstance(item, (list, tuple)) and len(item) == 1:
                    item = item[0]
                if isinstance(item, bytes):
                    item = item.decode('utf-8', errors='ignore')
                if isinstance(item, str):
                    out.append(item.strip().lower() in {'1', 'true', 'yes'})
                else:
                    out.append(bool(item))
            if len(out) < batch_size:
                out.extend([False] * (batch_size - len(out)))
            return torch.tensor(out, device=device, dtype=torch.bool)
        return torch.zeros(batch_size, device=device, dtype=torch.bool)

    @staticmethod
    def _metadata_status_mask(inputs, status_name, batch_size, device):
        statuses = inputs.get('decomposition_status', None)
        out = [False] * batch_size
        if isinstance(statuses, str):
            out = [statuses == status_name] * batch_size
        elif isinstance(statuses, (list, tuple)):
            for i, status in enumerate(list(statuses)[:batch_size]):
                if isinstance(status, bytes):
                    status = status.decode('utf-8', errors='ignore')
                out[i] = str(status) == status_name
        return torch.tensor(out, device=device, dtype=torch.bool)

    @staticmethod
    def _coverage_tensor(inputs, key, batch_size, device, default=0.0):
        coverage = inputs.get('coverage_stats', None)
        values = []
        if isinstance(coverage, dict):
            value = coverage.get(key, default)
            if torch.is_tensor(value):
                return value.to(device=device).float().view(-1)[:batch_size]
            values = [value] * batch_size
        elif isinstance(coverage, (list, tuple)):
            for item in list(coverage)[:batch_size]:
                if isinstance(item, dict):
                    values.append(item.get(key, default))
                else:
                    values.append(default)
        if len(values) < batch_size:
            values.extend([default] * (batch_size - len(values)))
        parsed = []
        for value in values[:batch_size]:
            try:
                parsed.append(float(value))
            except (TypeError, ValueError):
                parsed.append(float(default))
        return torch.tensor(parsed, device=device).float()

    @staticmethod
    def _metadata_numeric_tensor(inputs, key, batch_size, device, default=0.0):
        values = inputs.get(key, None)
        if values is None:
            return torch.full((batch_size,), float(default), device=device)
        if torch.is_tensor(values):
            return values.to(device=device).float().view(-1)[:batch_size]
        parsed = []
        if isinstance(values, (list, tuple)):
            for item in list(values)[:batch_size]:
                if isinstance(item, (list, tuple)) and len(item) == 1:
                    item = item[0]
                try:
                    parsed.append(float(item))
                except (TypeError, ValueError):
                    parsed.append(float(default))
        if len(parsed) < batch_size:
            parsed.extend([float(default)] * (batch_size - len(parsed)))
        return torch.tensor(parsed[:batch_size], device=device).float()

    @staticmethod
    def _coverage_has_key(inputs, key):
        coverage = inputs.get('coverage_stats', None)
        if isinstance(coverage, dict):
            return key in coverage
        if isinstance(coverage, (list, tuple)):
            return any(isinstance(item, dict) and key in item for item in coverage)
        return False

    def _apply_authoritative_coverage(self, inputs, slot_dict, batch_size, device):
        """Override slot metadata from dataset coverage_stats when present."""
        if not slot_dict:
            return slot_dict
        coverage = dict(slot_dict.get('coverage_stats', {}))
        defaults = {
            'has_target': coverage.get(
                'has_target',
                torch.ones(batch_size, device=device, dtype=torch.bool),
            ).to(device=device).float(),
            'num_attrs': coverage.get(
                'num_attrs',
                torch.zeros(batch_size, device=device),
            ).to(device=device).float(),
            'num_pairs': coverage.get(
                'num_pairs',
                torch.zeros(batch_size, device=device),
            ).to(device=device).float(),
        }
        for key, default in defaults.items():
            if self._coverage_has_key(inputs, key):
                value = self._coverage_tensor(
                    inputs, key, batch_size, device,
                    default=float(default.float().mean().detach().item())
                )
                coverage[key] = value.bool() if key == 'has_target' else value.long()
            else:
                coverage[key] = default.bool() if key == 'has_target' else default.long()

        for key in (
            'overgeneric_target_remaining',
            'target_overgeneric_canonical',
            'target_generic_reference',
            'global_only_due_to_parse_error',
            'missing_target',
            'generic_target',
            'num_parse_errors',
        ):
            coverage[key] = self._coverage_tensor(inputs, key, batch_size, device)

        slot_dict['coverage_stats'] = coverage
        if self._coverage_has_key(inputs, 'parse_confidence'):
            slot_dict['parse_confidence'] = self._coverage_tensor(
                inputs, 'parse_confidence', batch_size, device, default=1.0
            ).clamp(0.0, 1.0)
        elif 'parse_confidence' in inputs:
            slot_dict['parse_confidence'] = self._metadata_numeric_tensor(
                inputs, 'parse_confidence', batch_size, device, default=1.0
            ).clamp(0.0, 1.0)
        return slot_dict

    def _build_decomposition_masks(self, inputs, slot_dict, batch_size, device):
        coverage = slot_dict.get('coverage_stats', {}) if slot_dict else {}
        has_target = coverage.get(
            'has_target',
            torch.ones(batch_size, device=device, dtype=torch.bool)
        ).to(device=device).bool()

        global_only_mask = (
            (~has_target)
            | (coverage.get(
                'global_only_due_to_parse_error',
                torch.zeros(batch_size, device=device),
            ).to(device=device).float() > 0)
            | (coverage.get(
                'missing_target',
                torch.zeros(batch_size, device=device),
            ).to(device=device).float() > 0)
        )

        weak_generic_mask = (
            (coverage.get(
                'target_generic_reference',
                torch.zeros(batch_size, device=device),
            ).to(device=device).float() > 0)
            |
            (coverage.get(
                'overgeneric_target_remaining',
                torch.zeros(batch_size, device=device),
            ).to(device=device).float() > 0)
            | (coverage.get(
                'target_overgeneric_canonical',
                torch.zeros(batch_size, device=device),
            ).to(device=device).float() > 0)
            | (coverage.get(
                'generic_target',
                torch.zeros(batch_size, device=device),
            ).to(device=device).float() > 0)
        )
        legacy_global = (
            self._metadata_bool_mask(
                inputs, 'global_only_due_to_parse_error', batch_size, device
            )
            | self._metadata_status_mask(
                inputs, 'global_only_target_unresolved', batch_size, device
            )
        )
        legacy_generic = (
            self._metadata_bool_mask(
                inputs, 'target_generic_reference', batch_size, device
            )
            | self._metadata_status_mask(
                inputs, 'weak_generic_target_recovered', batch_size, device
            )
        )
        conflict_ratio = (
            (legacy_global != global_only_mask)
            | (legacy_generic != weak_generic_mask)
        ).float().mean()
        return global_only_mask, weak_generic_mask, conflict_ratio

    @staticmethod
    def _compute_contrastive_base_scores(proj_queries, proj_tokens, text_padding_mask=None):
        with torch.cuda.amp.autocast(enabled=False):
            proj_queries = proj_queries.float()
            proj_tokens = proj_tokens.float()
            sim = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2)) / 0.07
            if text_padding_mask is not None:
                valid_tokens = ~text_padding_mask.to(device=sim.device).bool()
                sim = sim.masked_fill(
                    ~valid_tokens.unsqueeze(1),
                    torch.finfo(sim.dtype).min,
                )
                denom = valid_tokens.float().sum(dim=1, keepdim=True).clamp(min=1.0).log()
            else:
                denom = sim.new_tensor(float(sim.shape[-1])).log()
            base_scores = sim.logsumexp(dim=-1) - denom
        return base_scores

    @staticmethod
    def _pool_selector_text_context(text_feats, text_padding_mask):
        text_feats = text_feats.float()
        valid_tokens = ~text_padding_mask.to(text_feats.device).bool()
        weights = valid_tokens.unsqueeze(-1).to(dtype=text_feats.dtype)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (text_feats * weights).sum(dim=1) / denom

    @staticmethod
    def _build_selector_metadata_context(inputs, slot_dict, batch_size, device):
        coverage = slot_dict.get('coverage_stats', {}) if slot_dict else {}

        def coverage_value(key, default=0.0):
            if key in coverage:
                value = coverage[key]
                if torch.is_tensor(value):
                    return value.to(device=device).float().view(-1)[:batch_size]
            return BeaUTyDETR._coverage_tensor(
                inputs, key, batch_size, device, default=default
            )

        if slot_dict and 'parse_confidence' in slot_dict:
            parse_confidence = slot_dict['parse_confidence'].to(
                device=device
            ).float().view(-1)[:batch_size]
        elif BeaUTyDETR._coverage_has_key(inputs, 'parse_confidence'):
            parse_confidence = BeaUTyDETR._coverage_tensor(
                inputs, 'parse_confidence', batch_size, device, default=1.0
            )
        else:
            parse_confidence = BeaUTyDETR._metadata_numeric_tensor(
                inputs, 'parse_confidence', batch_size, device, default=1.0
            )
        parse_confidence = parse_confidence.clamp(0.0, 1.0)

        has_target = coverage_value('has_target', 1.0).clamp(0.0, 1.0)
        num_attrs = coverage_value('num_attrs', 0.0).clamp(min=0.0)
        num_pairs = coverage_value('num_pairs', 0.0).clamp(min=0.0)
        num_parse_errors = coverage_value(
            'num_parse_errors', 0.0
        ).clamp(min=0.0)
        global_only = coverage_value(
            'global_only_due_to_parse_error', 0.0
        ).clamp(0.0, 1.0)
        target_generic = coverage_value(
            'target_generic_reference', 0.0
        ).clamp(0.0, 1.0)
        error_count = BeaUTyDETR._metadata_numeric_tensor(
            inputs,
            'decomposition_error_flags_count',
            batch_size,
            device,
            default=0.0,
        ).clamp(min=0.0)

        return torch.stack([
            parse_confidence,
            has_target,
            num_attrs,
            num_pairs,
            num_parse_errors,
            global_only,
            target_generic,
            error_count,
        ], dim=1).float()

    def _run_backbones(self, inputs):
        """Run visual and text backbones."""
        # Visual encoder
        end_points = self.backbone_net(inputs['point_clouds'], end_points={})
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = end_points['fp2_xyz']
        end_points['seed_features'] = end_points['fp2_features']

        # Text encoder (single-path RoBERTa)
        tokenized = self.tokenizer.batch_encode_plus(
            inputs['text'], padding="longest", return_tensors="pt",
            return_offsets_mapping=self.needs_span_token_alignment,
            return_special_tokens_mask=self.needs_span_token_alignment
        ).to(inputs['point_clouds'].device)
        encoded_text = self.text_encoder(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask']
        )
        text_feats = self.text_projector(encoded_text.last_hidden_state)
        # Invert attention mask that we get from huggingface
        # because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        end_points['text_feats'] = text_feats
        end_points['text_attention_mask'] = text_attention_mask
        end_points['tokenized'] = tokenized

        # Build structured span tensors aligned with full RoBERTa tokens.
        entity_spans = inputs.get('entity_spans', None)
        attr_spans = inputs.get('attr_spans', None)
        rel_spans = inputs.get('rel_spans', None)
        # Build structured slots if enabled
        if self.structured_slot_builder is not None:
            device = inputs['point_clouds'].device
            entity_spans_tensor = build_token_span_tensors(
                tokenized, entity_spans or [[] for _ in range(tokenized['input_ids'].shape[0])], device
            )
            attr_spans_tensor = build_token_span_tensors(
                tokenized, attr_spans or [[] for _ in range(tokenized['input_ids'].shape[0])], device
            )
            rel_spans_tensor = build_token_span_tensors(
                tokenized, rel_spans or [[] for _ in range(tokenized['input_ids'].shape[0])], device
            )
            anchor_ids = inputs.get('anchor_ids', None)

            slot_dict = self.structured_slot_builder(
                token_feats=end_points['text_feats'],
                tokenized=tokenized,
                entity_spans=entity_spans_tensor,
                attr_spans=attr_spans_tensor,
                rel_spans=rel_spans_tensor,
                anchor_types=inputs.get('anchors', None),
                anchor_ids=anchor_ids,
                utterances=inputs.get('text', None)
            )
            slot_dict = self._apply_authoritative_coverage(
                inputs,
                slot_dict,
                tokenized['input_ids'].shape[0],
                device,
            )
            end_points['slot_dict'] = slot_dict

            # Debug logging
            if self.structured_debug:
                end_points['slot_debug'] = {
                    'parse_conf_mean': slot_dict['parse_confidence'].mean().item(),
                    'num_pairs_mean': slot_dict['coverage_stats']['num_pairs'].float().mean().item(),
                    'has_target_ratio': slot_dict['coverage_stats']['has_target'].float().mean().item(),
                }

        return end_points

    def _generate_queries(self, xyz, features, end_points):
        # kps sampling
        points_obj_cls_logits = self.points_obj_cls(features)
        end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
        sample_inds = torch.topk(
            torch.sigmoid(points_obj_cls_logits).squeeze(1),
            self.num_queries
        )[1].int()
        xyz, features, sample_inds = self.gsample_module(
            xyz, features, sample_inds
        )
        end_points['query_points_xyz'] = xyz  # (B, V, 3)
        end_points['query_points_feature'] = features  # (B, F, V)
        end_points['query_points_sample_inds'] = sample_inds  # (B, V)
        return end_points

    def forward(self, inputs):
        """
        Forward pass.
        Args:
            inputs: dict
                {point_clouds, text}
                point_clouds (tensor): (B, Npoint, 3 + input_channels)
                text (list): ['text0', 'text1', ...], len(text) = B

                more keys if butd is enabled:
                    det_bbox_label_mask
                    det_boxes
                    det_class_ids
        Returns:
            end_points: dict
        """
        # Within-modality encoding
        end_points = self._run_backbones(inputs)
        points_xyz = end_points['fp2_xyz']  # (B, points, 3)
        points_features = end_points['fp2_features']  # (B, F, points)
        text_feats = end_points['text_feats']  # (B, L, F)
        text_padding_mask = end_points['text_attention_mask']  # (B, L)

        # Box encoding
        if self.butd:
            # attend on those features
            detected_mask = ~inputs['det_bbox_label_mask']
            detected_feats = torch.cat([
                self.box_embeddings(inputs['det_boxes']),
                self.class_embeddings(self.butd_class_embeddings(
                    inputs['det_class_ids']
                )).transpose(1, 2)  # 92.5, 84.9
            ], 1).transpose(1, 2).contiguous()
        else:
            detected_mask = None
            detected_feats = None

        # Cross-modality encoding
        points_features, text_feats = self.cross_encoder(
            vis_feats=points_features.transpose(1, 2).contiguous(),
            pos_feats=self.pos_embed(points_xyz).transpose(1, 2).contiguous(),
            padding_mask=torch.zeros(
                len(points_xyz), points_xyz.size(1)
            ).to(points_xyz.device).bool(),
            text_feats=text_feats,
            text_padding_mask=text_padding_mask,
            end_points=end_points,
            detected_feats=detected_feats,
            detected_mask=detected_mask
        )
        points_features = points_features.transpose(1, 2)
        points_features = points_features.contiguous()  # (B, F, points)
        end_points["text_memory"] = text_feats
        end_points['seed_features'] = points_features

        if self.contrastive_align_loss:
            proj_tokens = F.normalize(
                self.contrastive_align_projection_text(text_feats), p=2, dim=-1
            )
            end_points['proj_tokens'] = proj_tokens

        # Query Points Generation
        end_points = self._generate_queries(
            points_xyz, points_features, end_points
        )
        cluster_feature = end_points['query_points_feature']  # (B, F, V)
        cluster_xyz = end_points['query_points_xyz']  # (B, V, 3)
        query = self.decoder_query_proj(cluster_feature)
        query = query.transpose(1, 2).contiguous()  # (B, V, F)
        if self.contrastive_align_loss:
            end_points['proposal_proj_queries'] = F.normalize(
                self.contrastive_align_projection_image(query), p=2, dim=-1
            )

        # Proposals (one for each query)
        proposal_center, proposal_size = self.proposal_head(
            cluster_feature,
            base_xyz=cluster_xyz,
            end_points=end_points,
            prefix='proposal_'
        )
        base_xyz = proposal_center.detach().clone()  # (B, V, 3)
        base_size = proposal_size.detach().clone()  # (B, V, 3)
        query_mask = None

        # Decoder

        for i in range(self.num_decoder_layers):
            prefix = 'last_' if i == self.num_decoder_layers-1 else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_position_embedding == 'loc_learned':
                query_pos = torch.cat([base_xyz, base_size], -1)
            else:
                raise NotImplementedError

            # Transformer Decoder Layer
            query = self.decoder[i](
                query, points_features.transpose(1, 2).contiguous(),
                text_feats, query_pos,
                query_mask,
                text_padding_mask,
                detected_feats=(
                    detected_feats if self.butd
                    else None
                ),
                detected_mask=detected_mask if self.butd else None,
                layer_idx=i,
                num_layers=self.num_decoder_layers,
            )  # (B, V, F)
            if self.contrastive_align_loss:
                end_points[f'{prefix}proj_queries'] = F.normalize(
                    self.contrastive_align_projection_image(query), p=2, dim=-1
                )

            # Prediction
            base_xyz, base_size = self.prediction_heads[i](
                query.transpose(1, 2).contiguous(),  # (B, F, V)
                base_xyz=cluster_xyz,
                end_points=end_points,
                prefix=prefix
            )
            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()

        last_query = query
        last_boxes = torch.cat([base_xyz, base_size], dim=-1)
        if (
            self.quality_head is not None
            or self.sacr_head is not None
            or self.reliability_fusion is not None
            or self.acd_head is not None
            or self.dhc_loss_module is not None
        ):
            end_points['last_queries'] = last_query

        base_grounding_scores = None
        if self.needs_base_grounding_scores:
            proj_tokens = end_points.get('proj_tokens', None)
            proj_queries = end_points.get('last_proj_queries', None)
            assert proj_tokens is not None and proj_queries is not None, \
                "SACR/RAPF/QA-HNL base scores require contrastive_align_loss=True"
            contrastive_base_scores = self._compute_contrastive_base_scores(
                proj_queries,
                proj_tokens,
                text_padding_mask=end_points.get('text_attention_mask', None)
            )
            # The official BBS metric ranks queries with the target span's
            # soft-token scores.  Keep BBF's contrastive scores available as a
            # diagnostic source, but anchor SACR/RAPF/QA-HNL to the same score
            # family that the primary metric evaluates.
            base_grounding_scores = contrastive_base_scores
            if 'last_sem_cls_scores' in end_points:
                if 'positive_map' not in inputs:
                    raise RuntimeError(
                        "SACR/RAPF official-BBS alignment requires "
                        "positive_map in model inputs"
                    )
                base_grounding_scores = compute_soft_token_base_scores(
                    end_points['last_sem_cls_scores'],
                    inputs['positive_map'],
                    box_label_mask=inputs.get('box_label_mask', None),
                )
            end_points['base_grounding_scores'] = base_grounding_scores
            end_points['bbs_base_grounding_scores'] = base_grounding_scores
            end_points['bbf_base_grounding_scores'] = contrastive_base_scores

        if self.quality_head is not None:
            quality_out = self.quality_head(last_query, last_boxes)
            end_points.update(quality_out)

        if self.sacr_head is not None and 'slot_dict' in end_points:
            global_only_mask, weak_generic_mask, metadata_conflict_ratio = self._build_decomposition_masks(
                inputs, end_points['slot_dict'], last_query.shape[0], last_query.device
            )
            error_count = self._metadata_numeric_tensor(
                inputs,
                'decomposition_error_flags_count',
                last_query.shape[0],
                last_query.device,
                default=0.0,
            )
            sacr_out = self.sacr_head(
                query_feats=last_query,
                pred_boxes=last_boxes,
                base_scores=base_grounding_scores,
                slot_dict=end_points['slot_dict'],
                global_only_mask=global_only_mask,
                weak_generic_target_mask=weak_generic_mask,
            )
            end_points['structured_scores'] = sacr_out['structured_scores']
            end_points['target_attr_scores'] = sacr_out['target_attr_scores']
            end_points['relation_anchor_scores'] = sacr_out['relation_anchor_scores']
            end_points['anchor_entropy'] = sacr_out['anchor_entropy']
            end_points['anchor_top1_mass'] = sacr_out['anchor_top1_mass']
            end_points['structured_valid_mask'] = sacr_out['structured_valid_mask']
            end_points['weak_generic_target_mask'] = sacr_out['weak_generic_target_mask']
            end_points['global_only_mask'] = sacr_out['global_only_mask']
            end_points['decomp_global_only_mask'] = sacr_out['global_only_mask']
            end_points['decomp_weak_generic_mask'] = sacr_out['weak_generic_target_mask']
            end_points['decomposition_error_flags_count'] = error_count
            end_points['dbg_metadata_conflict_ratio'] = metadata_conflict_ratio.detach()
            end_points['dbg_sacr_anchor_entropy'] = sacr_out['anchor_entropy'].mean().detach()
            end_points['dbg_sacr_anchor_top1_mass'] = sacr_out['anchor_top1_mass'].mean().detach()
            end_points['dbg_sacr_relation_active_ratio'] = (
                sacr_out['relation_active_ratio'].detach()
            )
            end_points['dbg_sacr_structured_valid_ratio'] = (
                sacr_out['structured_valid_mask'].float().mean().detach()
            )
            end_points['dbg_sacr_global_only_ratio'] = (
                sacr_out['global_only_mask'].float().mean().detach()
            )
            end_points['dbg_sacr_weak_generic_ratio'] = (
                sacr_out['weak_generic_target_mask'].float().mean().detach()
            )

        if self.reliability_fusion is not None:
            if 'structured_scores' not in end_points:
                raise RuntimeError("RAPF requires SACR structured_scores")
            parse_confidence = None
            if 'slot_dict' in end_points:
                parse_confidence = end_points['slot_dict'].get('parse_confidence', None)
            rapf_out = self.reliability_fusion(
                base_scores=base_grounding_scores,
                structured_scores=end_points['structured_scores'],
                quality_scores=end_points.get('pred_iou', None),
                structured_valid_mask=end_points.get('structured_valid_mask', None),
                global_only_mask=end_points.get('global_only_mask', None),
                weak_generic_target_mask=end_points.get('weak_generic_target_mask', None),
                parse_confidence=parse_confidence,
                decomposition_error_flags_count=end_points.get(
                    'decomposition_error_flags_count', None
                ),
                anchor_entropy=end_points.get('anchor_entropy', None),
                anchor_top1_mass=end_points.get('anchor_top1_mass', None),
            )
            end_points.update(rapf_out)

        detector_policy_features = None
        selector_detector_scores = {}
        train_detector_policy_teacher = (
            self.use_detector_policy_teacher
            and bool(inputs.get('train', self.training))
        )
        if (
            self.source_pool_selector is not None
            or self.detector_policy_adapter is not None
            or train_detector_policy_teacher
            or str(inputs.get('eval_primary_score_source', 'base')).startswith(
                'detector_'
            )
        ):
            detector_policy_target_cid = self._detector_policy_target_cid(
                inputs
            )
            detector_policy_features = build_detector_policy_features(
                pred_boxes=last_boxes,
                quality_scores=end_points.get('pred_iou', None),
                det_boxes=inputs.get('det_boxes', None),
                det_bbox_label_mask=inputs.get('det_bbox_label_mask', None),
                det_class_ids=inputs.get('det_class_ids', None),
                det_logits=inputs.get('det_logits', None),
                target_cid=detector_policy_target_cid,
            )
            selector_detector_scores = build_detector_policy_score_sources(
                pred_boxes=last_boxes,
                quality_scores=end_points.get('pred_iou', None),
                det_boxes=inputs.get('det_boxes', None),
                det_bbox_label_mask=inputs.get('det_bbox_label_mask', None),
                det_class_ids=inputs.get('det_class_ids', None),
                det_logits=inputs.get('det_logits', None),
                target_cid=detector_policy_target_cid,
            )
            for source_name, source_scores in selector_detector_scores.items():
                end_points[f'{source_name}_scores'] = source_scores

        selector_context = None
        if (
            self.source_pool_selector_text_context
            or self.source_pool_selector_metadata_context
            or self.detector_policy_adapter_context
        ):
            selector_context = torch.cat([part for part in (
                self._pool_selector_text_context(
                    end_points['text_feats'],
                    end_points['text_attention_mask'],
                ) if (
                    self.source_pool_selector_text_context
                    or self.detector_policy_adapter_context
                ) else None,
                self._build_selector_metadata_context(
                    inputs,
                    end_points.get('slot_dict', None),
                    last_query.shape[0],
                    last_query.device,
                ) if (
                    self.source_pool_selector_metadata_context
                    or self.detector_policy_adapter_context
                ) else None,
            ) if part is not None], dim=-1)

        if (
            self.detector_policy_adapter is not None
            and detector_policy_features is not None
        ):
            adapter_context = (
                selector_context if self.detector_policy_adapter_context else None
            )
            adapter_out = self.detector_policy_adapter(
                detector_policy_features,
                context=adapter_context,
            )
            end_points['detector_policy_adapter_scores'] = adapter_out['scores']
            end_points['detector_policy_adapter_weights'] = adapter_out['weights']
            end_points['detector_policy_adapter_prior_weights'] = (
                self.detector_policy_adapter.prior_weights
            )

        if self.source_pool_selector is not None:
            selector_base_scores = base_grounding_scores
            if (
                'positive_map' in inputs
                and 'last_sem_cls_scores' in end_points
            ):
                selector_base_scores = compute_soft_token_base_scores(
                    end_points['last_sem_cls_scores'],
                    inputs['positive_map'],
                    box_label_mask=inputs.get('box_label_mask', None),
                )
                end_points['bbs_base_grounding_scores'] = selector_base_scores
            selector_contrastive_scores = None
            if (
                'positive_map' in inputs
                and 'proj_tokens' in end_points
                and 'last_proj_queries' in end_points
            ):
                selector_contrastive_scores = (
                    compute_contrastive_token_base_scores(
                        end_points['last_proj_queries'],
                        end_points['proj_tokens'],
                        inputs['positive_map'],
                        box_label_mask=inputs.get('box_label_mask', None),
                    )
                )
                end_points['bbf_base_grounding_scores'] = (
                    selector_contrastive_scores
                )
            selector_source_scores = {
                'base': selector_base_scores,
                'structured': end_points.get('structured_scores', None),
                'quality': end_points.get('pred_iou', None),
                'fused': end_points.get('fused_scores', None),
                'contrastive_base': selector_contrastive_scores,
                'detector_policy_adapter': end_points.get(
                    'detector_policy_adapter_scores', None
                ),
            }
            selector_source_scores.update(selector_detector_scores)
            selector_out = self.source_pool_selector(
                query_feats=last_query,
                pred_boxes=last_boxes,
                source_scores=selector_source_scores,
                selector_context=selector_context,
            )
            end_points.update(selector_out)

        # Apply late ACD head if enabled
        if self.acd_head is not None and 'slot_dict' in end_points:
            # Get final predictions
            last_query = query  # (B, V, F)
            last_boxes = torch.cat([base_xyz, base_size], dim=-1)  # (B, V, 6)
            # Compute base_scores via the configured late-ACD base source.
            proj_tokens = end_points.get('proj_tokens', None)
            proj_queries = end_points.get('last_proj_queries', None)
            assert proj_tokens is not None and proj_queries is not None, \
                "ACD requires contrastive_align_loss=True (proj_tokens/proj_queries)"
            last_base_scores = resolve_acd_base_scores(
                self.acd_base_score_source,
                self.acd_head,
                end_points,
                proj_queries,
                proj_tokens,
            )

            # Run ACD head
            acd_out = self.acd_head(
                query_feats=last_query,
                pred_boxes=last_boxes,
                base_scores=last_base_scores,
                slot_dict=end_points['slot_dict'],
                end_points=end_points if (self.structured_debug or self.use_late_acd or self.dhc_loss_module is not None) else None,
                global_step=getattr(self, '_acd_global_step', None)
            )

            # Store ACD outputs
            end_points['acd_structured_scores'] = acd_out['structured_scores']
            end_points['acd_final_scores'] = acd_out['final_scores']
            if self.structured_debug and 'acd_debug' in acd_out:
                end_points['acd_debug'] = acd_out['acd_debug']

        # Materialize DHC scalar parameters inside forward so DDP sees them as
        # part of the model graph. Using module parameters directly in the loss
        # code outside forward can trigger "mark a variable ready only once".
        if self.dhc_loss_module is not None:
            raw_margin_entity = self.dhc_loss_module.raw_margin_entity()
            raw_margin_attr = self.dhc_loss_module.raw_margin_attr()
            raw_margin_rel = self.dhc_loss_module.raw_margin_rel()
            raw_margin_acd_rank = self.dhc_loss_module.raw_margin_acd_rank()
            raw_temperature = self.dhc_loss_module.raw_temperature()
            end_points['dhc_margin_entity'] = self.dhc_loss_module.margin_entity()
            end_points['dhc_margin_attr'] = self.dhc_loss_module.margin_attr()
            end_points['dhc_margin_rel'] = self.dhc_loss_module.margin_rel()
            end_points['dhc_margin_acd_rank'] = self.dhc_loss_module.margin_acd_rank()
            end_points['dhc_temperature'] = self.dhc_loss_module.temperature()
            end_points['dbg_dhc_margin_entity_raw'] = raw_margin_entity.detach()
            end_points['dbg_dhc_margin_attr_raw'] = raw_margin_attr.detach()
            end_points['dbg_dhc_margin_rel_raw'] = raw_margin_rel.detach()
            end_points['dbg_dhc_margin_acd_rank_raw'] = raw_margin_acd_rank.detach()
            end_points['dbg_dhc_temperature_raw'] = raw_temperature.detach()
            end_points['dbg_dhc_margin_min'] = float(self.dhc_loss_module.margin_min)
            end_points['dbg_dhc_temperature_max'] = float(self.dhc_loss_module.temperature_max)
            if self.dhc_loss_module.margin_min > 0:
                margin_floor = float(self.dhc_loss_module.margin_min)
                end_points['dbg_dhc_margin_entity_at_floor'] = float(raw_margin_entity.detach().item() <= margin_floor + 1e-6)
                end_points['dbg_dhc_margin_attr_at_floor'] = float(raw_margin_attr.detach().item() <= margin_floor + 1e-6)
                end_points['dbg_dhc_margin_rel_at_floor'] = float(raw_margin_rel.detach().item() <= margin_floor + 1e-6)
                end_points['dbg_dhc_margin_acd_rank_at_floor'] = float(raw_margin_acd_rank.detach().item() <= margin_floor + 1e-6)
            if self.dhc_loss_module.temperature_max > 0:
                temp_cap = float(self.dhc_loss_module.temperature_max)
                end_points['dbg_dhc_temperature_at_cap'] = float(raw_temperature.detach().item() >= temp_cap - 1e-6)

        return end_points

    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1
