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
    BiEncoder, BiEncoderLayer, BiDecoderLayer, calc_pairwise_locs
)
from .structured_slots import StructuredSlotBuilder
from .quality_head import QualityHead
from .sacr_head import SACRHead
from .reliability_fusion import ReliabilityFusion
from .semantic_rerank_head import (
    SemanticRerankHead, blend_semantic_rerank_outputs
)
from .semantic_component_calibrator import SemanticComponentCalibrator
from .semantic_support_adapter import SemanticSupportAdapter


def build_token_span_tensors(tokenized, span_batches, device, min_slots=1):
    """Convert character-level span dicts to padded RoBERTa token spans."""
    batch_size = tokenized['input_ids'].shape[0]
    max_spans = max(
        (len(spans) for spans in span_batches if isinstance(spans, list)),
        default=0,
    )
    max_spans = max(max_spans, min_slots)
    span_tensor = torch.full(
        (batch_size, max_spans, 2),
        -1,
        dtype=torch.long,
        device=device,
    )

    for b in range(batch_size):
        spans = (
            span_batches[b]
            if b < len(span_batches) and isinstance(span_batches[b], list)
            else []
        )
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

    def __init__(self, num_class=256, num_obj_class=485,
                 input_feature_dim=3,
                 num_queries=256,
                 num_decoder_layers=6, self_position_embedding='loc_learned',
                 contrastive_align_loss=True,
                 d_model=288, butd=True, pointnet_ckpt=None, data_path=None,
                 self_attend=True,
                 use_structured_slots=False,
                 slot_pooling='attention',
                 max_rel_anchor_pairs=3,
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
                 aux_scores_use_contrastive_base=False,
                 aux_scores_use_semantic_eval_base=False,
                 use_semantic_rerank_head=False,
                 semantic_rerank_hidden_dim=128,
                 semantic_rerank_residual_scale=0.1,
                 semantic_rerank_use_target_conditioning=False,
                 use_semantic_rerank_aux_head=False,
                 semantic_rerank_aux_weight=0.5,
                 use_semantic_threshold_head=False,
                 semantic_threshold_hidden_dim=64,
                 semantic_threshold_residual_scale=0.25,
                 use_semantic_component_calibration=False,
                 semantic_component_max_delta=0.25,
                 semantic_component_use_eda_score=False,
                 semantic_component_extra_max_weight=0.25,
                 use_semantic_support_adapter=False,
                 semantic_support_overlap_weight=0.6075,
                 semantic_support_position_weight=0.1075,
                 semantic_support_overlap_power=0.5,
                 semantic_support_use_learned_gate=False,
                 semantic_support_gate_hidden_dim=16,
                 semantic_support_gate_max=2.0,
                 semantic_support_gate_use_query_features=False,
                 use_spatial_backbone_adapter=False):
        """Initialize layers."""
        super().__init__()

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.self_position_embedding = self_position_embedding
        self.contrastive_align_loss = contrastive_align_loss
        self.butd = butd
        self.use_spatial_backbone_adapter = bool(
            use_spatial_backbone_adapter
        )
        self.use_structured_slots = use_structured_slots
        self.structured_debug = structured_debug
        self.use_quality_head = use_quality_head
        self.use_sacr = use_sacr
        self.use_rapf = use_rapf
        self.use_qahnl = use_qahnl
        self.qahnl_score_source = qahnl_score_source
        self.aux_scores_use_contrastive_base = aux_scores_use_contrastive_base
        self.aux_scores_use_semantic_eval_base = (
            aux_scores_use_semantic_eval_base
        )
        self.use_semantic_rerank_head = use_semantic_rerank_head
        self.semantic_rerank_aux_weight = float(semantic_rerank_aux_weight)
        self.use_semantic_component_calibration = (
            use_semantic_component_calibration
        )
        self.use_semantic_support_adapter = bool(
            use_semantic_support_adapter
        )
        self.semantic_component_use_eda_score = bool(
            semantic_component_use_eda_score
        )
        self.needs_span_token_alignment = bool(use_structured_slots)
        self.needs_base_grounding_scores = bool(
            use_sacr or use_rapf or use_qahnl or use_semantic_support_adapter
        )

        # Visual encoder
        self.backbone_net = Pointnet2Backbone(
            input_feature_dim=input_feature_dim,
            width=1
        )
        if input_feature_dim == 3 and pointnet_ckpt is not None:
            self.backbone_net.load_state_dict(torch.load(
                pointnet_ckpt
            ), strict=False)

        # Text Encoder
        # # (1) online
        # t_type = "roberta-base"
        # NOTE (2) offline: load from the local folder.
        t_type = f'{data_path}roberta-base/'
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type, local_files_only=True)
        self.text_encoder = RobertaModel.from_pretrained(t_type, local_files_only=True)
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
            use_butd_enc_attn=butd,
            use_spatial_backbone_adapter=self.use_spatial_backbone_adapter,
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
        self.decoder = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.decoder.append(BiDecoderLayer(
                d_model, n_heads=8, dim_feedforward=256,
                dropout=0.1, activation="relu",
                self_position_embedding=self_position_embedding, butd=self.butd,
                use_spatial_backbone_adapter=(
                    self.use_spatial_backbone_adapter
                ),
            ))

        # Prediction heads
        self.prediction_heads = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.prediction_heads.append(ClsAgnosticPredictHead(
                num_class, 1, num_queries, d_model,
                objectness=False, heading=False,
                compute_sem_scores=True
            ))

        self.structured_slot_builder = (
            StructuredSlotBuilder(
                d_model=d_model,
                pooling=slot_pooling,
                max_pairs=max_rel_anchor_pairs,
            )
            if use_structured_slots else None
        )
        self.quality_head = (
            QualityHead(d_model=d_model, hidden_dim=d_model)
            if use_quality_head else None
        )
        self.sacr_head = (
            SACRHead(
                d_model=d_model,
                hidden_dim=sacr_hidden_dim,
                top_m_targets=sacr_top_m_targets,
                top_k_anchors=sacr_top_k_anchors,
                geo_dim=sacr_geo_dim,
                disable_relation=sacr_disable_relation,
            )
            if use_sacr else None
        )
        self.reliability_fusion = (
            ReliabilityFusion(
                hidden_dim=rapf_hidden_dim,
                initial_gate_bias=rapf_initial_gate_bias,
                use_quality=rapf_use_quality,
                quality_weight=rapf_quality_weight,
                generic_gate_cap=rapf_generic_gate_cap,
                residual_clip=rapf_struct_residual_clip,
                quality_anchor_structured_residual=(
                    rapf_quality_anchor_structured_residual
                ),
            )
            if use_rapf else None
        )
        self.semantic_rerank_head = (
            SemanticRerankHead(
                d_model=d_model,
                hidden_dim=semantic_rerank_hidden_dim,
                residual_scale=semantic_rerank_residual_scale,
                use_target_conditioning=(
                    semantic_rerank_use_target_conditioning
                ),
                use_threshold_head=use_semantic_threshold_head,
                threshold_hidden_dim=semantic_threshold_hidden_dim,
                threshold_residual_scale=(
                    semantic_threshold_residual_scale
                ),
            )
            if use_semantic_rerank_head else None
        )
        self.semantic_rerank_aux_head = (
            SemanticRerankHead(
                d_model=d_model,
                hidden_dim=semantic_rerank_hidden_dim,
                residual_scale=semantic_rerank_residual_scale,
                use_target_conditioning=False,
                use_threshold_head=False,
            )
            if use_semantic_rerank_aux_head else None
        )
        self.semantic_component_calibrator = (
            SemanticComponentCalibrator(
                max_delta=semantic_component_max_delta,
                extra_score_count=(
                    1 if self.semantic_component_use_eda_score else 0
                ),
                extra_max_weight=semantic_component_extra_max_weight,
            )
            if use_semantic_component_calibration else None
        )
        self.semantic_support_adapter = (
            SemanticSupportAdapter(
                overlap_weight=semantic_support_overlap_weight,
                position_weight=semantic_support_position_weight,
                overlap_power=semantic_support_overlap_power,
                use_learned_gate=semantic_support_use_learned_gate,
                gate_hidden_dim=semantic_support_gate_hidden_dim,
                gate_max=semantic_support_gate_max,
                gate_use_query_features=(
                    semantic_support_gate_use_query_features
                ),
                query_dim=d_model,
            )
            if use_semantic_support_adapter else None
        )

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
    def _metadata_numeric_tensor(inputs, key, batch_size, device, default=0.0):
        value = inputs.get(key, None)
        if value is None:
            return torch.full((batch_size,), float(default), device=device)
        if torch.is_tensor(value):
            tensor = value.to(device=device).float().view(-1)
        elif isinstance(value, (list, tuple)):
            vals = []
            for item in value[:batch_size]:
                try:
                    vals.append(float(item))
                except (TypeError, ValueError):
                    vals.append(float(default))
            tensor = torch.tensor(vals, device=device, dtype=torch.float32)
        else:
            tensor = torch.full((batch_size,), float(value), device=device)
        if tensor.numel() < batch_size:
            pad = torch.full(
                (batch_size - tensor.numel(),),
                float(default),
                device=device,
            )
            tensor = torch.cat([tensor, pad], dim=0)
        return tensor[:batch_size]

    @staticmethod
    def _metadata_bool_tensor(inputs, key, batch_size, device, default=False):
        value = inputs.get(key, None)
        if value is None:
            return torch.full(
                (batch_size,), bool(default), device=device, dtype=torch.bool
            )
        if torch.is_tensor(value):
            tensor = value.to(device=device).bool().view(-1)
        elif isinstance(value, (list, tuple)):
            tensor = torch.tensor(
                [bool(v) for v in value[:batch_size]],
                device=device,
                dtype=torch.bool,
            )
        else:
            tensor = torch.full(
                (batch_size,), bool(value), device=device, dtype=torch.bool
            )
        if tensor.numel() < batch_size:
            pad = torch.full(
                (batch_size - tensor.numel(),),
                bool(default),
                device=device,
                dtype=torch.bool,
            )
            tensor = torch.cat([tensor, pad], dim=0)
        return tensor[:batch_size]

    def _build_decomposition_masks(self, inputs, slot_dict, batch_size, device):
        global_only = self._metadata_bool_tensor(
            inputs, 'decomp_global_only_mask', batch_size, device, default=False
        )
        weak_generic = self._metadata_bool_tensor(
            inputs, 'decomp_weak_generic_mask', batch_size, device, default=False
        )
        metadata_conflict_ratio = self._metadata_numeric_tensor(
            inputs, 'metadata_conflict_ratio', batch_size, device, default=0.0
        ).mean()
        coverage = slot_dict.get('coverage_stats', {})
        has_target = coverage.get('has_target', None)
        if has_target is not None:
            global_only = global_only | (~has_target.to(device=device).bool())
        return global_only, weak_generic, metadata_conflict_ratio

    @staticmethod
    def _crop_or_pad_map(pmap, target_dim):
        if pmap.shape[-1] == target_dim:
            return pmap
        if pmap.shape[-1] > target_dim:
            return pmap[..., :target_dim]
        pad_shape = list(pmap.shape)
        pad_shape[-1] = target_dim - pmap.shape[-1]
        return torch.cat([pmap, pmap.new_zeros(pad_shape)], dim=-1)

    def _compute_eda_base_scores(self, end_points, inputs):
        sem_logits = end_points.get('last_sem_cls_scores', None)
        if sem_logits is None or 'positive_map' not in inputs:
            return None
        sem_scores = sem_logits.softmax(-1)
        score = sem_logits.new_zeros(sem_logits.shape[:2])
        terms = (
            ('positive_map', 1.0),
            ('modify_positive_map', 1.0),
            ('pron_positive_map', 1.0),
            ('rel_positive_map', 1.0),
            ('other_entity_map', -1.0),
        )
        for key, weight in terms:
            if key not in inputs:
                continue
            pmap = inputs[key]
            if not torch.is_tensor(pmap):
                continue
            pmap = pmap.to(device=sem_logits.device, dtype=sem_scores.dtype)
            if pmap.dim() == 3:
                pmap = pmap[:, :1]
            elif pmap.dim() == 2:
                pmap = pmap.unsqueeze(1)
            pmap = self._crop_or_pad_map(pmap, sem_scores.shape[-1])
            term = (sem_scores.unsqueeze(1) * pmap.unsqueeze(2)).sum(-1)
            score = score + float(weight) * term[:, 0]
        return score

    def _compute_contrastive_base_scores(self, proj_queries, proj_tokens,
                                         text_padding_mask=None):
        scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
        if text_padding_mask is not None:
            scores = scores.masked_fill(text_padding_mask.unsqueeze(1), -1e4)
        return scores.max(dim=-1).values

    def _compute_contrastive_base_scores_from_end_points(self, end_points):
        proj_tokens = end_points.get('proj_tokens', None)
        proj_queries = end_points.get('last_proj_queries', None)
        if proj_tokens is None or proj_queries is None:
            return None
        return self._compute_contrastive_base_scores(
            proj_queries,
            proj_tokens,
            text_padding_mask=end_points.get('text_attention_mask', None),
        )

    def _compute_semantic_eval_components(self, end_points, inputs):
        proj_tokens = end_points.get('proj_tokens', None)
        proj_queries = end_points.get('last_proj_queries', None)
        if proj_tokens is None or proj_queries is None:
            return None

        raw_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
        sem_scores_ = (raw_scores / 0.07).softmax(-1)
        target_dim = max(256, sem_scores_.shape[-1])
        sem_scores = sem_scores_.new_zeros(
            sem_scores_.shape[0],
            sem_scores_.shape[1],
            target_dim,
        )
        sem_scores[:, :, :sem_scores_.shape[-1]] = sem_scores_

        components = []
        terms = (
            'positive_map',
            'modify_positive_map',
            'pron_positive_map',
            'rel_positive_map',
            'other_entity_map',
        )
        has_map = False
        for key in terms:
            if key not in inputs:
                components.append(sem_scores.new_zeros(sem_scores.shape[:2]))
                continue
            pmap = inputs[key]
            if not torch.is_tensor(pmap):
                components.append(sem_scores.new_zeros(sem_scores.shape[:2]))
                continue
            has_map = True
            pmap = pmap.to(device=sem_scores.device, dtype=sem_scores.dtype)
            if pmap.dim() == 3:
                pmap = pmap[:, :1]
            elif pmap.dim() == 2:
                pmap = pmap.unsqueeze(1)
            pmap = self._crop_or_pad_map(pmap, sem_scores.shape[-1])
            term = (sem_scores.unsqueeze(1) * pmap.unsqueeze(2)).sum(-1)
            components.append(term[:, 0])

        if not has_map:
            return None
        return torch.stack(components, dim=-1)

    def _compute_semantic_eval_base_scores(self, end_points, inputs):
        components = self._compute_semantic_eval_components(end_points, inputs)
        if components is None:
            return None
        weights = components.new_tensor([1.0, 1.0, 1.0, 1.0, -1.0])
        score = (components * weights.view(1, 1, -1)).sum(dim=-1)
        return score

    def _select_aux_base_scores(self, end_points, inputs):
        if getattr(self, 'aux_scores_use_semantic_eval_base', False):
            semantic_eval_scores = self._compute_semantic_eval_base_scores(
                end_points, inputs
            )
            if semantic_eval_scores is not None:
                return semantic_eval_scores

        if self.aux_scores_use_contrastive_base:
            contrastive_scores = (
                self._compute_contrastive_base_scores_from_end_points(end_points)
            )
            if contrastive_scores is not None:
                return contrastive_scores

        eda_scores = self._compute_eda_base_scores(end_points, inputs)
        if eda_scores is not None:
            return eda_scores

        contrastive_scores = (
            self._compute_contrastive_base_scores_from_end_points(end_points)
        )
        if contrastive_scores is not None:
            return contrastive_scores

        raise AssertionError(
            "SACR/RAPF/QA-HNL base scores require either EDA positive maps "
            "or contrastive alignment outputs"
        )
    
    
    # BRIEF visual and text backbones.
    def _run_backbones(self, inputs):
        """Run visual and text backbones."""
        # step 1. Visual encoder
        end_points = self.backbone_net(inputs['point_clouds'], end_points={})
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = end_points['fp2_xyz']
        end_points['seed_features'] = end_points['fp2_features']
        
        # step 2. Text encoder
        tokenized = self.tokenizer.batch_encode_plus(
            inputs['text'], padding="longest", return_tensors="pt",
            return_offsets_mapping=self.needs_span_token_alignment,
            return_special_tokens_mask=self.needs_span_token_alignment
        ).to(inputs['point_clouds'].device)
        
        encoded_text = self.text_encoder(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
        )
        text_feats = self.text_projector(encoded_text.last_hidden_state)

        # Invert attention mask that we get from huggingface
        # because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()

        end_points['text_feats'] = text_feats
        end_points['text_attention_mask'] = text_attention_mask
        end_points['tokenized'] = tokenized
        if self.structured_slot_builder is not None:
            device = inputs['point_clouds'].device
            batch_size = tokenized['input_ids'].shape[0]
            entity_spans_tensor = build_token_span_tensors(
                tokenized,
                inputs.get('entity_spans', [[] for _ in range(batch_size)]),
                device,
            )
            attr_spans_tensor = build_token_span_tensors(
                tokenized,
                inputs.get('attr_spans', [[] for _ in range(batch_size)]),
                device,
            )
            rel_spans_tensor = build_token_span_tensors(
                tokenized,
                inputs.get('rel_spans', [[] for _ in range(batch_size)]),
                device,
            )
            anchor_ids = inputs.get(
                'anchor_span_ids',
                inputs.get('anchor_ids', None),
            )
            slot_dict = self.structured_slot_builder(
                token_feats=end_points['text_feats'],
                tokenized=tokenized,
                entity_spans=entity_spans_tensor,
                attr_spans=attr_spans_tensor,
                rel_spans=rel_spans_tensor,
                anchor_ids=anchor_ids,
                utterances=inputs.get('text', None),
            )
            slot_dict['parse_confidence'] = self._metadata_numeric_tensor(
                inputs, 'parse_confidence', batch_size, device, default=1.0
            )
            end_points['slot_dict'] = slot_dict
            if self.structured_debug:
                end_points['slot_debug'] = {
                    'parse_conf_mean': slot_dict['parse_confidence'].mean().item(),
                    'has_target_ratio': slot_dict['coverage_stats'][
                        'has_target'
                    ].float().mean().item(),
                }
        return end_points

    # BRIEF generate query.
    def _generate_queries(self, xyz, features, end_points):
        # kps sampling
        points_obj_cls_logits = self.points_obj_cls(features)
        end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
        
        # top-k
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
    
    # BRIEF forward.
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
        # STEP 1. vision and text encoding
        end_points = self._run_backbones(inputs)
        points_xyz = end_points['fp2_xyz']
        points_features = end_points['fp2_features']
        text_feats = end_points['text_feats']
        text_padding_mask = end_points['text_attention_mask']
        
        # STEP 2. Box encoding
        if self.butd:
            # attend on those features
            detected_mask = ~inputs['det_bbox_label_mask']

            # step box position.    det_boxes ([B, 132, 6]) -->  ([B, 128, 132])
            box_embeddings = self.box_embeddings(inputs['det_boxes'])
            # step box class        det_class_ids ([B, 132])  -->  ([B, 132, 160])
            class_embeddings = self.class_embeddings(self.butd_class_embeddings(inputs['det_class_ids']))
            # step box feature     ([B, 132, 288])
            detected_feats = torch.cat([box_embeddings, class_embeddings.transpose(1, 2)]
                                        , 1).transpose(1, 2).contiguous()
        else:
            detected_mask = None
            detected_feats = None

        # STEP 3. Cross-modality encoding
        spatial_point_xyz = (
            calc_pairwise_locs(points_xyz)
            if self.use_spatial_backbone_adapter else None
        )
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
            detected_mask=detected_mask,
            spatial_point_xyz=spatial_point_xyz,
        )
        points_features = points_features.transpose(1, 2)
        points_features = points_features.contiguous()
        end_points["text_memory"] = text_feats
        end_points['seed_features'] = points_features
        
        # STEP 4. text projection --> 64
        if self.contrastive_align_loss:
            proj_tokens = F.normalize(
                self.contrastive_align_projection_text(text_feats), p=2, dim=-1
            )
            end_points['proj_tokens'] = proj_tokens     # ([B, L, 64])

        # STEP 5. Query Points Generation
        end_points = self._generate_queries(
            points_xyz, points_features, end_points
        )
        cluster_feature = end_points['query_points_feature']    # (B, F=288, V=256)
        cluster_xyz = end_points['query_points_xyz']            # (B, V=256, 3)
        query = self.decoder_query_proj(cluster_feature)        
        query = query.transpose(1, 2).contiguous()              # (B, V=256, F=288)
        # projection 288 --> 64
        if self.contrastive_align_loss: 
            end_points['proposal_proj_queries'] = F.normalize(
                self.contrastive_align_projection_image(query), p=2, dim=-1
            )

        # STEP 6.Proposals
        proposal_center, proposal_size = self.proposal_head(
            cluster_feature,
            base_xyz=cluster_xyz,
            end_points=end_points,
            prefix='proposal_'
        )
        base_xyz = proposal_center.detach().clone()
        base_size = proposal_size.detach().clone()
        query_mask = None

        # STEP 7. Decoder
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

            # step Transformer Decoder Layer
            query = self.decoder[i](
                query, points_features.transpose(1, 2).contiguous(),
                text_feats, query_pos,
                query_mask,
                text_padding_mask,
                detected_feats=(
                    detected_feats if self.butd
                    else None
                ),
                detected_mask=detected_mask if self.butd else None
            )  # (B, V, F)
            # step project
            if self.contrastive_align_loss:
                end_points[f'{prefix}proj_queries'] = F.normalize(
                    self.contrastive_align_projection_image(query), p=2, dim=-1
                )

            # step box Prediction head
            base_xyz, base_size = self.prediction_heads[i](
                query.transpose(1, 2).contiguous(),     # ([B, F=288, V=256])
                base_xyz=cluster_xyz,                   # ([B, 256, 3])
                end_points=end_points,  # 
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
            or self.semantic_rerank_head is not None
            or self.semantic_component_calibrator is not None
        ):
            end_points['last_queries'] = last_query

        base_grounding_scores = None
        if self.needs_base_grounding_scores:
            base_grounding_scores = self._select_aux_base_scores(
                end_points, inputs
            )
            end_points['base_grounding_scores'] = base_grounding_scores

        if self.quality_head is not None:
            end_points.update(self.quality_head(last_query, last_boxes))

        if self.sacr_head is not None and 'slot_dict' in end_points:
            global_only_mask, weak_generic_mask, metadata_conflict_ratio = (
                self._build_decomposition_masks(
                    inputs, end_points['slot_dict'], last_query.shape[0],
                    last_query.device,
                )
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
            end_points.update({
                'structured_scores': sacr_out['structured_scores'],
                'target_attr_scores': sacr_out['target_attr_scores'],
                'relation_anchor_scores': sacr_out['relation_anchor_scores'],
                'anchor_entropy': sacr_out['anchor_entropy'],
                'anchor_top1_mass': sacr_out['anchor_top1_mass'],
                'structured_valid_mask': sacr_out['structured_valid_mask'],
                'weak_generic_target_mask': sacr_out['weak_generic_target_mask'],
                'global_only_mask': sacr_out['global_only_mask'],
                'decomp_global_only_mask': sacr_out['global_only_mask'],
                'decomp_weak_generic_mask': sacr_out['weak_generic_target_mask'],
                'decomposition_error_flags_count': error_count,
                'dbg_metadata_conflict_ratio': metadata_conflict_ratio.detach(),
                'dbg_sacr_anchor_entropy': sacr_out['anchor_entropy'].mean().detach(),
                'dbg_sacr_anchor_top1_mass': sacr_out['anchor_top1_mass'].mean().detach(),
                'dbg_sacr_relation_active_ratio': (
                    sacr_out['relation_active_ratio'].detach()
                ),
                'dbg_sacr_structured_valid_ratio': (
                    sacr_out['structured_valid_mask'].float().mean().detach()
                ),
                'dbg_sacr_global_only_ratio': (
                    sacr_out['global_only_mask'].float().mean().detach()
                ),
                'dbg_sacr_weak_generic_ratio': (
                    sacr_out['weak_generic_target_mask'].float().mean().detach()
                ),
            })

        if self.reliability_fusion is not None:
            if 'structured_scores' not in end_points:
                raise RuntimeError("RAPF requires SACR structured_scores")
            parse_confidence = None
            if 'slot_dict' in end_points:
                parse_confidence = end_points['slot_dict'].get(
                    'parse_confidence', None
                )
            rapf_out = self.reliability_fusion(
                base_scores=base_grounding_scores,
                structured_scores=end_points['structured_scores'],
                quality_scores=end_points.get('pred_iou', None),
                structured_valid_mask=end_points.get('structured_valid_mask', None),
                global_only_mask=end_points.get('global_only_mask', None),
                weak_generic_target_mask=end_points.get(
                    'weak_generic_target_mask', None
                ),
                parse_confidence=parse_confidence,
                decomposition_error_flags_count=end_points.get(
                    'decomposition_error_flags_count', None
                ),
                anchor_entropy=end_points.get('anchor_entropy', None),
                anchor_top1_mass=end_points.get('anchor_top1_mass', None),
            )
            end_points.update(rapf_out)

        if self.semantic_component_calibrator is not None:
            semantic_components = self._compute_semantic_eval_components(
                end_points, inputs
            )
            if semantic_components is None:
                raise RuntimeError(
                    "Semantic component calibration requires contrastive "
                    "outputs and language positive maps"
                )
            base_weights = semantic_components.new_tensor(
                [1.0, 1.0, 1.0, 1.0, -1.0]
            )
            extra_scores = None
            if self.semantic_component_use_eda_score:
                extra_scores = self._compute_eda_base_scores(end_points, inputs)
                if extra_scores is None:
                    raise RuntimeError(
                        "Semantic component EDA score calibration requires "
                        "language positive maps"
                    )
            component_out = self.semantic_component_calibrator(
                semantic_components,
                extra_scores=extra_scores,
            )
            end_points.update({
                'semantic_component_raw_scores': semantic_components,
                'semantic_eval_base_scores': (
                    semantic_components * base_weights.view(1, 1, -1)
                ).sum(dim=-1),
                **component_out,
            })
            if extra_scores is not None:
                end_points['semantic_component_extra_scores'] = extra_scores

        if self.semantic_rerank_head is not None:
            semantic_components = self._compute_semantic_eval_components(
                end_points, inputs
            )
            if semantic_components is None:
                raise RuntimeError(
                    "Semantic rerank head requires contrastive outputs and "
                    "language positive maps"
                )
            base_weights = semantic_components.new_tensor(
                [1.0, 1.0, 1.0, 1.0, -1.0]
            )
            semantic_eval_scores = (
                semantic_components * base_weights.view(1, 1, -1)
            ).sum(dim=-1)
            end_points['semantic_component_raw_scores'] = semantic_components
            end_points['semantic_eval_base_scores'] = semantic_eval_scores
            slot_dict = end_points.get('slot_dict', {})
            rerank_kwargs = dict(
                query_feats=last_query,
                pred_boxes=last_boxes,
                base_scores=semantic_eval_scores,
                quality_scores=end_points.get('pred_iou', None),
                fused_scores=end_points.get('fused_scores', None),
                target_slot=slot_dict.get('target_slot', None),
                semantic_components=semantic_components,
                structured_scores=end_points.get('structured_scores', None),
                target_attr_scores=end_points.get('target_attr_scores', None),
                relation_anchor_scores=end_points.get(
                    'relation_anchor_scores', None
                ),
                parse_confidence=slot_dict.get('parse_confidence', None),
            )
            rerank_out = self.semantic_rerank_head(**rerank_kwargs)
            if self.semantic_rerank_aux_head is not None:
                auxiliary_out = self.semantic_rerank_aux_head(**rerank_kwargs)
                rerank_out = blend_semantic_rerank_outputs(
                    rerank_out,
                    auxiliary_out,
                    auxiliary_weight=self.semantic_rerank_aux_weight,
                )
            end_points.update(rerank_out)

        if self.semantic_support_adapter is not None:
            if 'semantic_rerank_scores' not in end_points:
                raise RuntimeError(
                    "Semantic support adaptation requires semantic rerank scores"
                )
            if not self.butd:
                raise RuntimeError(
                    "Semantic support adaptation requires the BUTD box stream"
                )
            support_out = self.semantic_support_adapter(
                semantic_scores=end_points['semantic_rerank_scores'],
                query_boxes=last_boxes,
                detector_boxes=inputs['det_boxes'],
                detector_valid_mask=inputs['det_bbox_label_mask'],
                position_scores=base_grounding_scores,
                query_feats=last_query,
                target_slot=end_points.get(
                    'slot_dict', {}
                ).get('target_slot', None),
            )
            if 'semantic_threshold_residual' in end_points:
                support_out['semantic_support_scores_without_threshold'] = (
                    support_out['semantic_support_scores']
                )
                support_out['semantic_support_scores'] = (
                    support_out['semantic_support_scores']
                    + end_points['semantic_threshold_residual']
                )
            end_points.update(support_out)

        return end_points

    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1
