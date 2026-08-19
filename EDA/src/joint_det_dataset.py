# ------------------------------------------------------------------------
# Modification: EDA
# Created: 05/21/2022
# Author: Yanmin Wu
# E-mail: wuyanminmax@gmail.com
# https://github.com/yanmin-wu/EDA 
# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""Dataset and data loader."""

import csv
from collections import defaultdict
import h5py
import json
import multiprocessing as mp
import os
import random
import re
from six.moves import cPickle

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
import wandb

import copy

from data.model_util_scannet import ScannetDatasetConfig
from data.scannet_utils import read_label_mapping
from src.visual_data_handlers import Scan
from .scannet_classes import REL_ALIASES, VIEW_DEP_RELS

NUM_CLASSES = 485
DC = ScannetDatasetConfig(NUM_CLASSES)
DC18 = ScannetDatasetConfig(18)
MAX_NUM_OBJ = 132


class Joint3DDataset(Dataset):
    """Dataset utilities for ReferIt3D."""

    SPACY_ROTATION_MODE_NAMES = {
        -1: "not_spacy",
        1: "none",
        2: "yaw_only",
        3: "full_natural",
    }
    SPACY_AUGMENTATION_PROFILE_NAMES = {
        -1: "not_spacy",
        1: "none",
        2: "yaw_relation",
        3: "full_natural_relation_free",
        4: "yaw_relation_free",
        5: "none_relation_free_view",
        6: "yaw_relation_free_stable",
        7: "small_yaw_relation_free_view",
        8: "rawview_relation_free_global_only",
        9: "none_relation_free",
        10: "none_relation_free_compass",
    }

    def __init__(self, dataset_dict={'sr3d': 1, 'scannet': 10},
                 test_dataset='sr3d',
                 split='train', overfit=False,
                 data_path='./',
                 use_color=False, use_height=False, use_multiview=False,
                 detect_intermediate=False,
                 butd=False, butd_gt=False, butd_cls=False, augment_det=False,
                 disable_box_jitter=False,
                 spacy_relation_free_yaw_only_aug=False,
                 spacy_relation_free_view_guard_aug=False,
                 spacy_relation_free_stable_yaw_aug=False,
                 spacy_relation_free_view_small_yaw_aug=False,
                 spacy_relation_free_rawview_global_only_train=False,
                 spacy_relation_free_none_aug=False,
                 spacy_relation_free_compass_guard_aug=False,
                 wo_obj_name="None"):
        """Initialize dataset (here for ReferIt3D utterances)."""
        self.dataset_dict = dataset_dict
        self.test_dataset = test_dataset
        self.split = split
        self.use_color = use_color
        self.use_height = use_height
        self.overfit = overfit
        self.detect_intermediate = detect_intermediate
        self.augment = self.split == 'train'
        # self.augment = False
        self.use_multiview = use_multiview
        self.data_path = data_path if data_path.endswith('/') else data_path + '/'
        self.visualize = False  # manually set this to True to debug
        self.butd = butd
        self.butd_gt = butd_gt
        self.butd_cls = butd_cls
        self.joint_det = (  # joint usage of detection/grounding phrases
            'scannet' in dataset_dict
            and len(dataset_dict.keys()) > 1
            and self.split == 'train'
        )
        self.augment_det = augment_det
        self.disable_box_jitter = disable_box_jitter
        self.spacy_relation_free_yaw_only_aug = (
            spacy_relation_free_yaw_only_aug
        )
        self.spacy_relation_free_view_guard_aug = (
            spacy_relation_free_view_guard_aug
        )
        self.spacy_relation_free_stable_yaw_aug = (
            spacy_relation_free_stable_yaw_aug
        )
        self.spacy_relation_free_view_small_yaw_aug = (
            spacy_relation_free_view_small_yaw_aug
        )
        self.spacy_relation_free_rawview_global_only_train = (
            spacy_relation_free_rawview_global_only_train
        )
        self.spacy_relation_free_none_aug = spacy_relation_free_none_aug
        self.spacy_relation_free_compass_guard_aug = (
            spacy_relation_free_compass_guard_aug
        )
        self.wo_obj_name = wo_obj_name

        self.mean_rgb = np.array([109.8, 97.2, 83.8]) / 256
        
        # step 1. semantic label
        self.label_map = read_label_mapping(
            'data/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='id'
        )
        self.label_map18 = read_label_mapping(
            'data/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='nyu40id'
        )
        self.label_mapclass = read_label_mapping(
            'data/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='nyu40class'
        )   

        self.multiview_path = os.path.join(
            f'{self.data_path}/scanrefer_2d_feats',
            "enet_feats_maxpool.hdf5"
        )
        self.multiview_data = {}

        # step 2. transformer tokenizer
        # # 1) online
        # self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        # 2) offline
        self.tokenizer = RobertaTokenizerFast.from_pretrained(f'{self.data_path}roberta-base/', local_files_only=True)
        
        if os.path.exists('data/cls_results.json'):
            with open('data/cls_results.json') as fid:
                self.cls_results = json.load(fid)

        print('Loading %s files, take a breath!' % split)
        
        # step 3. generate or load train/val_v3scans.pkl
        if not os.path.exists(f'{self.data_path}/{split}_v3scans.pkl'):
            save_data(f'{data_path}/{split}_v3scans.pkl', split, data_path)
        self.scans = unpickle_data(f'{self.data_path}/{split}_v3scans.pkl')
        self.scans = list(self.scans)[0]
        
        # step 4. load text dataset
        if self.split != 'train':
            self.annos = self.load_annos(test_dataset)
        else:
            self.annos = []
            for dset, cnt in dataset_dict.items():  
                if cnt > 0:
                    _annos = self.load_annos(dset)
                    self.annos += (_annos * cnt)

        # if self.visualize:
        #     wandb.init(project="vis", name="debug")

    # BRIEF load text data
    def load_annos(self, dset):
        """Load annotations of given dataset."""
        loaders = {
            'nr3d': self.load_nr3d_annos,
            'nr3d_spacy': lambda: self.load_nr3d_annos(dset='nr3d_spacy'),
            'sr3d': self.load_sr3d_annos,
            'sr3d_spacy': lambda: self.load_sr3d_annos(dset='sr3d_spacy'),
            'sr3d+': self.load_sr3dplus_annos,
            'scanrefer': self.load_scanrefer_annos,
            'scanrefer_spacy': lambda: self.load_scanrefer_annos(dset='scanrefer_spacy'),
            'scannet': self.load_scannet_annos      # scannet detection augmentation
        }
        annos = loaders[dset]()
        if self.overfit:
            annos = annos[:128]
        return annos

    def load_sr3dplus_annos(self):
        """Load annotations of sr3d/sr3d+."""
        return self.load_sr3d_annos(dset='sr3d+')

    @staticmethod
    def _json_field(value, default):
        if value in (None, ''):
            return default
        if isinstance(value, str):
            return json.loads(value)
        return value

    @staticmethod
    def _bool_field(value, default=False):
        if value in (None, ''):
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ('1', 'true', 'yes', 'y')

    @staticmethod
    def _span_bounds(span):
        if not isinstance(span, dict):
            return None
        try:
            start = int(span.get('start'))
            end = int(span.get('end'))
        except (TypeError, ValueError):
            return None
        if start < 0 or end <= start:
            return None
        return [start, end]

    @classmethod
    def _build_graph_from_spacy_slots(cls, anno):
        target_slot = anno.get('target_slot', {})
        attr_slot = anno.get('attr_slot', {})
        rel_slots = anno.get('rel_slots', [])
        anchor_slots = anno.get('anchor_slots', [])

        target_span = cls._span_bounds(target_slot)
        target_node = {
            'node_id': 0,
            'node_type': 'Object',
            'target_char_span': [target_span] if target_span else [],
            'mod_char_span': [],
            'pron_char_span': [],
            'rel_char_span': [],
            'lemma_head': target_slot.get('text', '') if isinstance(target_slot, dict) else '',
        }
        for item in attr_slot.get('items', []) if isinstance(attr_slot, dict) else []:
            span = cls._span_bounds(item)
            if span:
                target_node['mod_char_span'].append(span)
        for item in rel_slots if isinstance(rel_slots, list) else []:
            span = cls._span_bounds(item)
            if span:
                target_node['rel_char_span'].append(span)

        graph_node = [target_node]
        for idx, anchor in enumerate(anchor_slots if isinstance(anchor_slots, list) else [], start=1):
            span = cls._span_bounds(anchor)
            if not span:
                continue
            graph_node.append({
                'node_id': idx,
                'node_type': 'Object',
                'target_char_span': [span],
                'mod_char_span': [],
                'pron_char_span': [],
                'rel_char_span': [],
                'lemma_head': anchor.get('head_lemma', anchor.get('text', '')),
            })
        auxi_entity = graph_node[1] if len(graph_node) > 1 else None
        return graph_node, [], auxi_entity

    @classmethod
    def _spacy_span_lists(cls, anno):
        entity_spans = []
        seen = set()

        def add_entity(span):
            bounds = cls._span_bounds(span)
            if not bounds:
                return -1
            text = span.get('text', '') if isinstance(span, dict) else ''
            key = (bounds[0], bounds[1], text)
            if key in seen:
                for i, item in enumerate(entity_spans):
                    if (item['start'], item['end'], item.get('text', '')) == key:
                        return i
            seen.add(key)
            entity_spans.append({'start': bounds[0], 'end': bounds[1], 'text': text})
            return len(entity_spans) - 1

        add_entity(anno.get('target_slot', {}))
        for item in anno.get('entities', []) if isinstance(anno.get('entities', []), list) else []:
            add_entity(item)

        attr_items = []
        attr_slot = anno.get('attr_slot', {})
        if isinstance(attr_slot, dict):
            attr_items = attr_slot.get('items', [])
        attr_spans = []
        for item in attr_items:
            bounds = cls._span_bounds(item)
            if bounds:
                attr_spans.append({
                    'start': bounds[0],
                    'end': bounds[1],
                    'text': item.get('text', ''),
                })
        rel_spans = []
        for item in anno.get('rel_slots', []):
            bounds = cls._span_bounds(item)
            if bounds:
                rel_spans.append({
                    'start': bounds[0],
                    'end': bounds[1],
                    'text': item.get('text', ''),
                    'head': item.get('head', ''),
                    'tail': item.get('tail', ''),
                })
        anchor_span_ids = [
            add_entity(item)
            for item in anno.get('anchor_slots', []) if isinstance(item, dict)
        ]
        if len(anchor_span_ids) < len(rel_spans):
            anchor_span_ids.extend([-1] * (len(rel_spans) - len(anchor_span_ids)))
        return entity_spans, attr_spans, rel_spans, anchor_span_ids[:len(rel_spans)]

    @staticmethod
    def _spacy_slot_text(slot):
        if not isinstance(slot, dict):
            return str(slot)
        values = []
        for key in (
            'text', 'surface_text', 'pattern', 'canonical_text',
            'head', 'tail', 'relation', 'rel', 'lemma_head', 'head_lemma',
        ):
            value = slot.get(key, '')
            if isinstance(value, (list, tuple)):
                values.extend(str(item) for item in value)
            else:
                values.append(str(value))
        return ' '.join(values).replace('_', ' ')

    @staticmethod
    def _spacy_has_term(text, terms):
        pattern = r'\b(?:' + '|'.join(re.escape(term) for term in terms) + r')\b'
        return re.search(pattern, str(text).lower().replace('_', ' ')) is not None

    @classmethod
    def _spacy_relation_rule_type(cls, relation_text, anchor_text=''):
        text = f'{relation_text} {anchor_text}'.lower().replace('_', ' ')
        scene_terms = (
            'wall', 'corner', 'side', 'edge', 'room', 'area', 'section',
            'part', 'end', 'entrance', 'doorway',
        )
        proximity_terms = (
            'near', 'next to', 'close to', 'beside', 'by', 'around',
            'adjacent', 'closest to', 'nearest to',
        )
        vertical_terms = (
            'on', 'on top of', 'above', 'below', 'under', 'beneath',
            'over', 'top of',
        )
        ordinal_terms = (
            'leftmost', 'rightmost', 'frontmost', 'backmost',
            'topmost', 'bottommost',
        )
        if (
            cls._spacy_has_compass_direction_word(text)
            or (
                cls._spacy_has_raw_view_word(text)
                and cls._spacy_has_term(text, scene_terms)
            )
        ):
            return 'scene_frame'
        if cls._spacy_has_term(text, ordinal_terms):
            return 'ordinal_scene_extreme'
        if cls._spacy_has_term(relation_text, ('between',)):
            return 'between'
        if cls._spacy_has_raw_view_word(relation_text):
            return 'object_relative_view'
        if any(term in text for term in proximity_terms):
            return 'proximity'
        if any(term in text for term in vertical_terms):
            return 'vertical_support'
        return 'generic_spatial'

    @staticmethod
    def _spacy_utterance_text(anno):
        if anno.get('utterance'):
            return str(anno.get('utterance'))
        if anno.get('description'):
            return str(anno.get('description'))
        tokens = anno.get('tokens', anno.get('token', []))
        if isinstance(tokens, list):
            return ' '.join(str(token) for token in tokens)
        return str(tokens or '')

    @classmethod
    def _spacy_has_weak_relation_free_signal(cls, anno):
        coverage_stats = anno.get('coverage_stats', {})
        if isinstance(coverage_stats, dict):
            noisy_keys = (
                'candidate_relation_count',
                'invalid_relation_count',
                'spatial_attribute_rows',
                'spatial_info_routed_to_attr',
                'refined_dropped_relation_count',
                'refined_weak_relation_count',
            )
            if any(
                cls._spacy_positive_count(coverage_stats.get(key, 0))
                for key in noisy_keys
            ):
                return True

        reason = str(anno.get('manual_review_fix_reason', '')).lower()
        noisy_reason_terms = (
            'background',
            'invalid_relation',
            'relation_dropped',
            'relation_tuples_dropped',
        )
        return any(term in reason for term in noisy_reason_terms)

    @classmethod
    def _spacy_train_global_only_reason(cls, anno):
        if not str(anno.get('dataset', '')).endswith('_spacy'):
            return ''

        rel_slots = anno.get('rel_slots', [])
        if not isinstance(rel_slots, list):
            rel_slots = [rel_slots]
        if any(cls._spacy_relation_text(slot).strip() for slot in rel_slots):
            return ''
        if not cls._spacy_has_weak_relation_free_signal(anno):
            return ''

        utterance = cls._spacy_utterance_text(anno)
        if cls._spacy_has_raw_view_word(utterance):
            return 'relation_free_raw_view_parse_noise'
        if cls._spacy_has_compass_direction_word(utterance):
            return 'relation_free_compass_parse_noise'
        if cls._spacy_spatial_attribute_mode(anno) is not None:
            return 'relation_free_spatial_attribute_parse_noise'
        return ''

    @classmethod
    def _refine_spacy_decomposition_fields(cls, anno):
        if not str(anno.get('dataset', '')).endswith('_spacy'):
            return anno

        refined = dict(anno)
        rel_slots = refined.get('rel_slots', [])
        anchor_slots = refined.get('anchor_slots', [])
        coverage_stats = dict(refined.get('coverage_stats', {}) or {})
        if not isinstance(rel_slots, list):
            rel_slots = [rel_slots]
        if not isinstance(anchor_slots, list):
            anchor_slots = [anchor_slots]

        kept_rel_slots = []
        kept_anchor_slots = []
        dropped = 0
        weak = 0
        missing_anchor = 0
        scene_frame = 0
        between_missing = 0
        type_counts = defaultdict(int)

        for idx, rel in enumerate(rel_slots):
            if not isinstance(rel, dict):
                dropped += 1
                weak += 1
                continue
            rel_text = cls._spacy_slot_text(rel)
            paired_anchor = (
                anchor_slots[idx]
                if idx < len(anchor_slots) and isinstance(anchor_slots[idx], dict)
                else None
            )
            anchor_text = cls._spacy_slot_text(paired_anchor) if paired_anchor else ''
            relation_type = cls._spacy_relation_rule_type(rel_text, anchor_text)
            type_counts[relation_type] += 1

            if relation_type in ('scene_frame', 'ordinal_scene_extreme'):
                dropped += 1
                weak += 1
                if relation_type == 'scene_frame':
                    scene_frame += 1
                continue
            if relation_type == 'between' and len(anchor_slots) < 2:
                dropped += 1
                weak += 1
                between_missing += 1
                continue
            if paired_anchor is None:
                dropped += 1
                weak += 1
                missing_anchor += 1
                continue

            rel_copy = dict(rel)
            rel_copy['relation_type'] = relation_type
            kept_rel_slots.append(rel_copy)
            kept_anchor_slots.append(dict(paired_anchor))

        coverage_stats['refined_relation_count'] = len(kept_rel_slots)
        coverage_stats['refined_dropped_relation_count'] = dropped
        coverage_stats['refined_weak_relation_count'] = weak
        coverage_stats['refined_missing_anchor_relation_count'] = missing_anchor
        coverage_stats['refined_scene_frame_relation_count'] = scene_frame
        coverage_stats['refined_between_missing_anchor_count'] = between_missing
        for relation_type, count in type_counts.items():
            coverage_stats[f'refined_{relation_type}_count'] = count

        refined['rel_slots'] = kept_rel_slots
        refined['anchor_slots'] = kept_anchor_slots
        refined['coverage_stats'] = coverage_stats
        refined['decomp_weak_generic_mask'] = bool(
            refined.get('decomp_weak_generic_mask', False) or weak > 0
        )
        train_global_only_reason = cls._spacy_train_global_only_reason(refined)
        if train_global_only_reason:
            refined['decomp_train_global_only_mask'] = True
            refined['decomp_train_weak_generic_mask'] = True
            refined['decomp_train_global_only_reason'] = train_global_only_reason
            coverage_stats['refined_train_global_only_signal_count'] = 1
            if train_global_only_reason == 'relation_free_raw_view_parse_noise':
                coverage_stats[
                    'refined_train_global_only_raw_view_count'
                ] = 1
            elif train_global_only_reason == 'relation_free_compass_parse_noise':
                coverage_stats[
                    'refined_train_global_only_compass_count'
                ] = 1
            elif (
                train_global_only_reason
                == 'relation_free_spatial_attribute_parse_noise'
            ):
                coverage_stats[
                    'refined_train_global_only_spatial_attribute_count'
                ] = 1
            refined['coverage_stats'] = coverage_stats
        refined['decomposition_error_flags_count'] = int(
            refined.get('decomposition_error_flags_count', 0) or 0
        ) + dropped
        if dropped:
            parse_confidence = float(refined.get('parse_confidence', 1.0) or 1.0)
            refined['parse_confidence'] = max(
                0.05, parse_confidence - 0.10 * dropped
            )
        return refined

    @staticmethod
    def _scanrefer_annotation_path(data_path, split, use_spacy=False):
        data_path = data_path if data_path.endswith('/') else data_path + '/'
        split = 'val' if split in ('val', 'test') else split
        suffixes = (
            (f'_{split}_spacy_refined.json', f'_{split}_spacy.json')
            if use_spacy else (f'_{split}.json',)
        )
        candidates = []
        for suffix in suffixes:
            candidates.extend([
                os.path.join(data_path, 'scanrefer', 'ScanRefer_filtered' + suffix),
                os.path.join(data_path, 'ScanRefer', 'ScanRefer_filtered' + suffix),
            ])
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    @staticmethod
    def _scanrefer_scan_ids_path(data_path, split):
        data_path = data_path if data_path.endswith('/') else data_path + '/'
        split = 'val' if split in ('val', 'test') else split
        candidates = [
            os.path.join(data_path, 'scanrefer', f'ScanRefer_filtered_{split}.txt'),
            os.path.join(data_path, 'ScanRefer', f'ScanRefer_filtered_{split}.txt'),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def _referit3d_csv_path(self, dset):
        names = (
            (f'{dset}_refined.csv', f'{dset}.csv')
            if str(dset).endswith('_spacy') else (f'{dset}.csv',)
        )
        candidates = []
        for name in names:
            candidates.extend([
                os.path.join(self.data_path, 'refer_it_3d', name),
                os.path.join(self.data_path, 'ReferIt3D', name),
            ])
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def _parse_spacy_csv_fields(self, line, headers, dset=''):
        entities = self._json_field(line[headers['entities']], [])
        attributes = self._json_field(line[headers['attributes']], [])
        rel_slots = self._json_field(line[headers['rel_slots']], [])
        target_slot = self._json_field(line[headers['target_slot']], {})
        attr_slot = self._json_field(line[headers['attr_slot']], {'items': []})
        anchor_slots = self._json_field(line[headers['anchor_slots']], [])
        coverage_stats = self._json_field(line[headers['coverage_stats']], {})
        anno = {
            'entities': entities,
            'attributes': attributes,
            'target_slot': target_slot,
            'attr_slot': attr_slot,
            'rel_slots': rel_slots,
            'anchor_slots': anchor_slots,
            'coverage_stats': coverage_stats,
            'utterance': line[headers['utterance']]
            if 'utterance' in headers else '',
            'parse_confidence': float(line[headers['parse_confidence']])
            if 'parse_confidence' in headers and line[headers['parse_confidence']] else 1.0,
            'decomp_global_only_mask': self._bool_field(
                line[headers['decomp_global_only_mask']]
            ) if 'decomp_global_only_mask' in headers else False,
            'decomp_weak_generic_mask': self._bool_field(
                line[headers['decomp_weak_generic_mask']]
            ) if 'decomp_weak_generic_mask' in headers else False,
            'decomp_train_global_only_mask': self._bool_field(
                line[headers['decomp_train_global_only_mask']]
            ) if 'decomp_train_global_only_mask' in headers else False,
            'decomp_train_weak_generic_mask': self._bool_field(
                line[headers['decomp_train_weak_generic_mask']]
            ) if 'decomp_train_weak_generic_mask' in headers else False,
            'decomp_train_global_only_reason': (
                line[headers['decomp_train_global_only_reason']]
                if 'decomp_train_global_only_reason' in headers else ''
            ),
        }
        anno['dataset'] = dset
        anno = self._refine_spacy_decomposition_fields(anno)
        graph_node, graph_edge, auxi_entity = self._build_graph_from_spacy_slots(anno)
        entity_spans, attr_spans, rel_spans, anchor_span_ids = self._spacy_span_lists(anno)
        anno.update({
            'graph_node': graph_node,
            'graph_edge': graph_edge,
            'auxi_entity': auxi_entity,
            'entity_spans': entity_spans,
            'attr_spans': attr_spans,
            'rel_spans': rel_spans,
            'anchor_span_ids': anchor_span_ids,
        })
        anno.pop('dataset', None)
        return anno

    def load_sr3d_annos(self, dset='sr3d'):
        """Load annotations of sr3d/sr3d+."""
        split = self.split
        if split == 'val':
            split = 'test'
        with open('data/meta_data/sr3d_%s_scans.txt' % split) as f:
            scan_ids = set(eval(f.read()))
        with open(self._referit3d_csv_path(dset)) as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            headers = {header: h for h, header in enumerate(headers)}
            annos = []
            is_spacy = dset.endswith('_spacy')
            for line in csv_reader:
                if not (
                    line[headers['scan_id']] in scan_ids
                    and (
                        is_spacy
                        or str(line[headers['mentions_target_class']]).lower() == 'true'
                    )
                ):
                    continue
                sample = {
                    'scan_id': line[headers['scan_id']],
                    'target_id': int(line[headers['target_id']]),
                    'distractor_ids': eval(line[headers['distractor_ids']]),
                    'utterance': line[headers['utterance']],
                    'target': line[headers['instance_type']],
                    'anchors': eval(line[headers['anchors_types']]),
                    'anchor_ids': eval(line[headers['anchor_ids']]),
                    'dataset': dset
                }
                if is_spacy:
                    sample.update(self._parse_spacy_csv_fields(line, headers, dset))
                    sample['anchor_obj_ids'] = eval(line[headers['anchor_ids']])
                annos.append(sample)
            if not is_spacy:
                Scene_graph_parse(annos)

        return annos

    def load_nr3d_annos(self, dset='nr3d'):
        """Load annotations of nr3d."""
        split = self.split
        if split == 'val':
            split = 'test'
        with open('data/meta_data/nr3d_%s_scans.txt' % split) as f:
            scan_ids = set(eval(f.read()))
        with open(self._referit3d_csv_path(dset)) as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            headers = {header: h for h, header in enumerate(headers)}
            annos = []
            is_spacy = dset.endswith('_spacy')
            for line in csv_reader:
                if not (
                    line[headers['scan_id']] in scan_ids
                    and (
                        str(line[headers['correct_guess']]).lower() == 'true'
                        or split != 'test'
                    )
                ):
                    continue
                sample = {
                    'scan_id': line[headers['scan_id']],
                    'target_id': int(line[headers['target_id']]),
                    'target': line[headers['instance_type']],
                    'utterance': line[headers['utterance']],
                    'anchor_ids': [],
                    'anchors': [],
                    'dataset': dset
                }
                if is_spacy:
                    sample.update(self._parse_spacy_csv_fields(line, headers, dset))
                    sample['anchor_obj_ids'] = eval(line[headers['anchor_ids']])
                annos.append(sample)

        if not is_spacy:
            Scene_graph_parse(annos)

        # Add distractor info
        for anno in annos:
            anno['distractor_ids'] = [
                ind
                for ind in
                range(len(self.scans[anno['scan_id']].three_d_objects))
                if self.scans[anno['scan_id']].get_object_instance_label(ind)
                == anno['target']
                and ind != anno['target_id']
            ]

        # NOTE [BUTD-DETR] Filter out sentences that do not explicitly mention the target class
        # annos = [anno for anno in annos if anno['target'] in anno['utterance']]

        return annos

    
    # BRIEF load ScanRefer
    def load_scanrefer_annos(self, dset='scanrefer'):
        """Load annotations of ScanRefer."""
        use_spacy = dset == 'scanrefer_spacy'
        split = self.split
        if split in ('val', 'test'):
            split = 'val'
        with open(self._scanrefer_scan_ids_path(self.data_path, split)) as f:
            scan_ids = [line.rstrip().strip('\n') for line in f.readlines()]
        with open(self._scanrefer_annotation_path(self.data_path, split, use_spacy)) as f:
            reader = json.load(f)
        if self.wo_obj_name != "None":
            with open(self.wo_obj_name) as f:
                reader = json.load(f)
        
        # STEP 1. load utterance
        annos = []
        for anno in reader:
            if anno['scene_id'] not in scan_ids:
                continue
            if use_spacy:
                anno = self._refine_spacy_decomposition_fields(dict(anno))
                graph_node, graph_edge, auxi_entity = (
                    self._build_graph_from_spacy_slots(anno)
                )
                entity_spans, attr_spans, rel_spans, anchor_span_ids = (
                    self._spacy_span_lists(anno)
                )
                utterance = anno.get('description', ' '.join(anno.get('tokens', [])))
            else:
                graph_node = graph_edge = auxi_entity = None
                entity_spans = attr_spans = rel_spans = anchor_span_ids = []
                utterance = ' '.join(anno['token'])
            sample = {
                'scan_id': anno['scene_id'],
                'target_id': int(anno['object_id']),
                'distractor_ids': [],
                'utterance': utterance,
                'target': ' '.join(str(anno['object_name']).split('_')),
                'anchors': [],      
                'anchor_ids': [],   
                'anchor_span_ids': anchor_span_ids,
                'dataset': dset,
                'entity_spans': entity_spans,
                'attr_spans': attr_spans,
                'rel_spans': rel_spans,
                'description': anno.get('description', utterance),
                'scene_id': anno.get('scene_id', anno['scene_id']),
                'object_id': anno.get('object_id', anno['object_id']),
                'object_name': anno.get('object_name', ''),
                'ann_id': anno.get('ann_id', ''),
                'target_slot': anno.get('target_slot', {}),
                'attr_slot': anno.get('attr_slot', {}),
                'rel_slots': anno.get('rel_slots', []),
                'anchor_slots': anno.get('anchor_slots', []),
                'slot_mask': anno.get('slot_mask', {}),
                'coverage_stats': anno.get('coverage_stats', {}),
                'parse_confidence': anno.get('parse_confidence', 1.0),
                'decomp_global_only_mask': anno.get('decomp_global_only_mask', False),
                'decomp_weak_generic_mask': anno.get('decomp_weak_generic_mask', False),
                'decomp_train_global_only_mask': anno.get(
                    'decomp_train_global_only_mask', False
                ),
                'decomp_train_weak_generic_mask': anno.get(
                    'decomp_train_weak_generic_mask', False
                ),
                'decomp_train_global_only_reason': anno.get(
                    'decomp_train_global_only_reason', ''
                ),
                'decomposition_error_flags_count': anno.get(
                    'decomposition_error_flags_count', 0
                ),
            }
            if use_spacy:
                sample['graph_node'] = graph_node
                sample['graph_edge'] = graph_edge
                sample['auxi_entity'] = auxi_entity
            annos.append(sample)

        if not use_spacy:
            Scene_graph_parse(annos)

        # # NOTE BUTD-DETR unreasonable approach, add GT object name
        # num = 0
        # for anno in annos:
        #     if anno['target'] not in anno['utterance']:
        #         num+=1
        #         anno['utterance'] = (
        #             ' '.join(anno['utterance'].split(' . ')[0].split()[:-1])
        #             + ' ' + anno['target'] + ' . '
        #             + ' . '.join(anno['utterance'].split(' . ')[1:])
        #         )
        
        # STEP 3. Add distractor info
        scene2obj = defaultdict(list)
        sceneobj2used = defaultdict(list)
        for anno in annos:
            nyu_labels = [
                self.label_mapclass[
                    self.scans[anno['scan_id']].get_object_instance_label(ind)
                ]
                for ind in
                range(len(self.scans[anno['scan_id']].three_d_objects))
            ]
            labels = [DC18.type2class.get(lbl, 17) for lbl in nyu_labels]
            anno['distractor_ids'] = [
                ind
                for ind in
                range(len(self.scans[anno['scan_id']].three_d_objects))
                if labels[ind] == labels[anno['target_id']]
                and ind != anno['target_id']
            ][:32]
            if anno['target_id'] not in sceneobj2used[anno['scan_id']]:
                sceneobj2used[anno['scan_id']].append(anno['target_id'])
                scene2obj[anno['scan_id']].append(labels[anno['target_id']])
        
        # STEP 4. Add unique-multi
        for anno in annos:
            if anno['scan_id'] not in list(self.scans.keys()):
                continue

            nyu_labels = [
                self.label_mapclass[
                    self.scans[anno['scan_id']].get_object_instance_label(ind)
                ]
                for ind in
                range(len(self.scans[anno['scan_id']].three_d_objects))
            ]
            labels = [DC18.type2class.get(lbl, 17) for lbl in nyu_labels]
            anno['unique'] = (
                np.array(scene2obj[anno['scan_id']])
                == labels[anno['target_id']]
            ).sum() == 1
        return annos


    # BRIEF scannet detection prompt.
    def load_scannet_annos(self):
        """Load annotations of scannet."""
        split = 'train' if self.split == 'train' else 'val'
        with open('data/meta_data/scannetv2_%s.txt' % split) as f:
            scan_ids = [line.rstrip() for line in f]

        annos = []
        for scan_id in scan_ids:
            if scan_id not in list(self.scans.keys()):
                continue

            scan = self.scans[scan_id]
            # Ignore scans that have no object in our vocabulary
            keep = np.array([
                self.label_map[
                    scan.get_object_instance_label(ind)
                ] in DC.nyu40id2class
                for ind in range(len(scan.three_d_objects))
            ])
            if keep.any():
                # this will get populated randomly each time
                annos.append({
                    'scan_id': scan_id,
                    'target_id': [],
                    'distractor_ids': [],
                    'utterance': '',
                    'target': [],
                    'anchors': [],
                    'anchor_ids': [],
                    'dataset': 'scannet'
                })
        if self.split == 'train':
            annos = [
                anno for a, anno in enumerate(annos)
                if a not in {965, 977}
            ]
        return annos
    
    # BRIEF smaple classes for detection prompt
    def _sample_classes(self, scan_id):
        """Sample classes for the scannet detection sentences."""
        scan = self.scans[scan_id]
        sampled_classes = set([
            self.label_map[scan.get_object_instance_label(ind)]
            for ind in range(len(scan.three_d_objects))
        ])
        sampled_classes = list(sampled_classes & set(DC.nyu40id2class))
        # sample 10 classes
        if self.split == 'train' and self.random_utt:  # random utterance
            if len(sampled_classes) > 10:
                sampled_classes = random.sample(sampled_classes, 10)
            ret = [DC.class2type[DC.nyu40id2class[i]] for i in sampled_classes]
            random.shuffle(ret)
        else:
            ret = [
                'cabinet', 'bed', 'chair', 'couch', 'table', 'door',
                'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
                'other furniture'
            ]
        return ret
    
    # BRIEF constract utterance for scannet
    def _create_scannet_utterance(self, sampled_classes):
        if self.split == 'train' and self.random_utt:
            neg_names = []
            while len(neg_names) < 10:
                _ind = np.random.randint(0, len(DC.class2type))
                if DC.class2type[_ind] not in neg_names + sampled_classes:
                    neg_names.append(DC.class2type[_ind])
            mixed_names = sorted(list(set(sampled_classes + neg_names)))
            random.shuffle(mixed_names)
        else:
            mixed_names = sampled_classes
        utterance = ' . '.join(mixed_names)
        return utterance

    def _load_multiview(self, scan_id):
        """Load multi-view data of given scan-id."""
        pid = mp.current_process().pid
        if pid not in self.multiview_data:
            self.multiview_data[pid] = h5py.File(
                self.multiview_path, "r", libver="latest"
            )
        return self.multiview_data[pid][scan_id]
    
    # BRIEF point cloud augmentation
    def _augment(self, pc, color, rotate):
        augmentations = {}
        no_direction_rotation = rotate == "none"
        yaw_only_rotation = rotate == "yaw_only"
        yaw_stable_rotation = rotate == "yaw_stable"
        small_yaw_rotation = rotate == "small_yaw"
        stable_jitter = yaw_stable_rotation

        # Rotate/flip only if we don't have a view_dep sentence
        if rotate is True or yaw_only_rotation or yaw_stable_rotation:
            theta_z = 90*np.random.randint(0, 4) + (2*np.random.rand() - 1) * 5
            # Flipping along the YZ plane
            augmentations['yz_flip'] = np.random.random() > 0.5
            if augmentations['yz_flip']:
                pc[:, 0] = -pc[:, 0]
            # Flipping along the XZ plane
            augmentations['xz_flip'] = np.random.random() > 0.5
            if augmentations['xz_flip']:
                pc[:, 1] = -pc[:, 1]
        elif small_yaw_rotation:
            theta_z = (2*np.random.rand() - 1) * 3
            augmentations['yz_flip'] = False
            augmentations['xz_flip'] = False
        elif no_direction_rotation:
            theta_z = 0.0
        else:
            theta_z = (2*np.random.rand() - 1) * 5
        augmentations['theta_z'] = theta_z
        pc[:, :3] = rot_z(pc[:, :3], theta_z)
        # Rotate around x
        theta_x = (
            0.0
            if (
                no_direction_rotation
                or yaw_only_rotation
                or yaw_stable_rotation
                or small_yaw_rotation
            )
            else (2*np.random.rand() - 1) * 2.5
        )
        augmentations['theta_x'] = theta_x
        pc[:, :3] = rot_x(pc[:, :3], theta_x)
        # Rotate around y
        theta_y = (
            0.0
            if (
                no_direction_rotation
                or yaw_only_rotation
                or yaw_stable_rotation
                or small_yaw_rotation
            )
            else (2*np.random.rand() - 1) * 2.5
        )
        augmentations['theta_y'] = theta_y
        pc[:, :3] = rot_y(pc[:, :3], theta_y)

        # Add noise
        if stable_jitter:
            noise = np.zeros((len(pc), 3), dtype=pc.dtype)
        else:
            noise = np.random.rand(len(pc), 3) * 5e-3
        augmentations['noise'] = noise
        pc[:, :3] = pc[:, :3] + noise

        # Translate/shift
        if stable_jitter:
            augmentations['shift'] = np.zeros((1, 3), dtype=pc.dtype)
        else:
            augmentations['shift'] = np.random.random((3,))[None, :] - 0.5
        pc[:, :3] += augmentations['shift']

        # Scale
        if stable_jitter:
            augmentations['scale'] = 1.0
        else:
            augmentations['scale'] = 0.98 + 0.04*np.random.random()
        pc[:, :3] *= augmentations['scale']

        # Color
        if color is not None and not stable_jitter:
            color += self.mean_rgb
            color *= 0.98 + 0.04*np.random.random((len(color), 3))
            color -= self.mean_rgb
        return pc, color, augmentations
    
    # BRIEF get point clouds
    def _get_pc(self, anno, scan):
        """Return a point cloud representation of current scene."""
        scan_id = anno['scan_id']
        rel_name = "none"
        if anno['dataset'].startswith('sr3d'):
            rel_name = self._find_rel(anno['utterance'])

        # a. Color
        color = None
        if self.use_color:
            color = scan.color - self.mean_rgb

        # b. Multi-view 2d features
        multiview_data = None
        if self.use_multiview:
            multiview_data = self._load_multiview(scan_id)

        # c. Augmentations
        augmentations = {}
        if self.split == 'train' and self.augment:
            spacy_rotation_mode = self._spacy_rotation_mode_for_sample(anno)
            rotate_natural = (
                anno['dataset'] in ('nr3d', 'nr3d_spacy', 'scanrefer', 'scanrefer_spacy')
                and self._augment_nr3d(anno['utterance'])
                and spacy_rotation_mode is None
            )
            rotate_sr3d = (
                anno['dataset'].startswith('sr3d')
                and rel_name not in VIEW_DEP_RELS
            )
            rotate_else = anno['dataset'] == 'scannet'
            rotate = rotate_sr3d or rotate_natural or rotate_else
            if spacy_rotation_mode is not None:
                rotate = spacy_rotation_mode
            pc, color, augmentations = self._augment(scan.pc, color, rotate)
            scan.pc = pc

        # d. Height
        height = None
        if self.use_height:
            floor_height = np.percentile(scan.pc[:, 2], 0.99)
            height = np.expand_dims(scan.pc[:, 2] - floor_height, 1)

        # e. Concatenate representations
        point_cloud = scan.pc
        if color is not None:
            point_cloud = np.concatenate((point_cloud, color), 1)
        if height is not None:
            point_cloud = np.concatenate([point_cloud, height], 1)
        if multiview_data is not None:
            point_cloud = np.concatenate([point_cloud, multiview_data], 1)

        return point_cloud, augmentations, scan.color
    
    # BRIEF get position label [scannet]
    def _get_token_positive_map(self, anno):
        """Return correspondence of boxes to tokens."""
        # Token start-end span in characters
        caption = ' '.join(anno['utterance'].replace(',', ' ,').split())
        caption = ' ' + caption + ' '
        
        tokens_positive = np.zeros((MAX_NUM_OBJ, 2))
        if isinstance(anno['target'], list):
            cat_names = anno['target']
        else:
            cat_names = [anno['target']]
        if self.detect_intermediate:
            cat_names += anno['anchors']
        for c, cat_name in enumerate(cat_names):
            start_span = caption.find(' ' + cat_name + ' ')
            len_ = len(cat_name)
            if start_span < 0:
                start_span = caption.find(' ' + cat_name)
                len_ = len(caption[start_span+1:].split()[0])
            if start_span < 0:
                start_span = caption.find(cat_name)
                orig_start_span = start_span
                while caption[start_span - 1] != ' ':
                    start_span -= 1
                len_ = len(cat_name) + orig_start_span - start_span
                while caption[len_ + start_span] != ' ':
                    len_ += 1
            
            end_span = start_span + len_
            assert start_span > -1, caption
            assert end_span > 0, caption
            tokens_positive[c][0] = start_span
            tokens_positive[c][1] = end_span

        # Positive map (for soft token prediction)
        tokenized = self.tokenizer.batch_encode_plus(
            [' '.join(anno['utterance'].replace(',', ' ,').split())],
            padding="longest", return_tensors="pt"
        )

        positive_map = np.zeros((MAX_NUM_OBJ, 256))

        # note: empty in scannet prompt
        modify_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        pron_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        other_entity_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        rel_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        auxi_entity_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        
        # main object component
        gt_map = get_positive_map(tokenized, tokens_positive[:len(cat_names)])
        positive_map[:len(cat_names)] = gt_map
        return tokens_positive, positive_map, modify_positive_map, pron_positive_map, \
            other_entity_positive_map, auxi_entity_positive_map, rel_positive_map


    #################################################
    # BRIEF Get text position label by text parsing #
    #################################################
    def _get_token_positive_map_by_parse(self, anno, auxi_box):
        caption = ' '.join(anno['utterance'].replace(',', ' ,').split())

        # node and edge
        graph_node = anno["graph_node"]
        graph_edge = anno["graph_edge"]

        # step main/modify(attri)/pron/other(auxi)/rel
        target_char_span = np.zeros((MAX_NUM_OBJ, 2))   # target(main)
        modify_char_span = np.zeros((MAX_NUM_OBJ, 2))   # modify(attri)
        pron_char_span = np.zeros((MAX_NUM_OBJ, 2))     # pron
        rel_char_span = np.zeros((MAX_NUM_OBJ, 2))      # rel
        assert graph_node[0]['node_id'] == 0
        main_entity_target = graph_node[0]['target_char_span']
        main_entity_modify = graph_node[0]['mod_char_span']
        main_entity_pron   = graph_node[0]['pron_char_span']
        main_entity_rel   = graph_node[0]['rel_char_span']

        # other(auxi) object token
        other_target_char_span = np.zeros((MAX_NUM_OBJ, 2))
        other_entity_target = []
        if len(graph_node) > 1:
            for node in graph_node:
                if node["node_id"] != 0 and node["node_type"] == "Object":
                    for span in node['target_char_span']:
                        other_entity_target.append(span)

        num_t = 0
        num_m = 0
        num_p = 0
        num_o = 0
        num_r = 0
        # target(main obj.) token
        for t, target in enumerate(main_entity_target):
            target_char_span[t] = target
            num_t = t+1
        # modify(attribute) token
        for m, modify in enumerate(main_entity_modify):
            modify_char_span[m] = modify
            num_m = m+1
        # pron token
        for p, pron in enumerate(main_entity_pron):
            pron_char_span[p] = pron
            num_p = p+1
        for o, other in enumerate(other_entity_target):
            other_target_char_span[o] = other
            num_o = o+1
        # rel token add 0727
        for r, rel in enumerate(main_entity_rel):
            rel_char_span[r] = rel
            num_r = r+1

        tokenized = self.tokenizer.batch_encode_plus(
            [' '.join(anno['utterance'].replace(',', ' ,').split())],
            padding="longest", return_tensors="pt"
        )

        target_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        modify_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        pron_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        other_entity_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        rel_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        gt_map_t = get_positive_map(tokenized, target_char_span[:num_t])
        gt_map_m = get_positive_map(tokenized, modify_char_span[:num_m])
        gt_map_p = get_positive_map(tokenized, pron_char_span[:num_p])
        gt_map_o = get_positive_map(tokenized, other_target_char_span[:num_o])
        gt_map_r = get_positive_map(tokenized, rel_char_span[:num_r])
        
        gt_map_t = gt_map_t.sum(axis=0)
        gt_map_m = gt_map_m.sum(axis=0)
        gt_map_p = gt_map_p.sum(axis=0)
        gt_map_o = gt_map_o.sum(axis=0)
        gt_map_r = gt_map_r.sum(axis=0)

        # NOTE text position label
        target_positive_map[:1] = gt_map_t          # main object
        modify_positive_map[:1] = gt_map_m          # attribute
        pron_positive_map[:1]   = gt_map_p          # pron
        other_entity_positive_map[:1] = gt_map_o    # auxi obj
        rel_positive_map[:1]   = gt_map_r           # relation

        # auxi
        auxi_entity_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        if auxi_box is not None:
            auxi_entity = anno["auxi_entity"]['target_char_span']
            num_a = 0
            # char span
            auxi_entity_char_span = np.zeros((MAX_NUM_OBJ, 2))
            for a, auxi in enumerate(auxi_entity):
                auxi_entity_char_span[a] = auxi
                num_a = a+1
            # position label
            gt_map_a = get_positive_map(tokenized, auxi_entity_char_span[:num_a])
            gt_map_a = gt_map_a.sum(axis=0)
            auxi_entity_positive_map[:1] = gt_map_a

            # note SR3D 
            if anno['dataset'] == 'sr3d':
                target_positive_map[1] = gt_map_a

        return target_char_span, target_positive_map, modify_positive_map, pron_positive_map, \
            other_entity_positive_map, auxi_entity_positive_map, rel_positive_map


    # BRIEF get GT Box.
    def _get_target_boxes(self, anno, scan):
        """Return gt boxes to detect."""
        bboxes = np.zeros((MAX_NUM_OBJ, 6))
        if isinstance(anno['target_id'], list):
            tids = anno['target_id']
        else:  # referit dataset
            tids = [anno['target_id']]
            # TODO SR3D: anchor object
            if self.detect_intermediate:
                # tids += anno.get('anchor_ids', [])    # BUTD-DETR
                # EDA
                if anno['auxi_entity'] is not None and len(anno['anchor_ids']):
                    tids.append(anno['anchor_ids'][0])
        point_instance_label = -np.ones(len(scan.pc))
        for t, tid in enumerate(tids):
            point_instance_label[scan.three_d_objects[tid]['points']] = t
        
        bboxes[:len(tids)] = np.stack([
            scan.get_object_bbox(tid).reshape(-1) for tid in tids
        ])
        bboxes = np.concatenate((
            (bboxes[:, :3] + bboxes[:, 3:]) * 0.5,
            bboxes[:, 3:] - bboxes[:, :3]
        ), 1)
        if (
            self.split == 'train'
            and self.augment
            and not getattr(self, 'disable_box_jitter', False)
        ):  # jitter boxes
            bboxes[:len(tids)] *= (0.95 + 0.1*np.random.random((len(tids), 6)))
        bboxes[len(tids):, :3] = 1000
        
        box_label_mask = np.zeros(MAX_NUM_OBJ)
        box_label_mask[:len(tids)] = 1
        
        return bboxes, box_label_mask, point_instance_label

    def _get_scene_objects(self, scan):
        # Objects to keep
        keep_ = np.array([
            self.label_map[
                scan.get_object_instance_label(ind)
            ] in DC.nyu40id2class
            for ind in range(len(scan.three_d_objects))
        ])[:MAX_NUM_OBJ]    # keep_ (object_num)
        keep = np.array([False] * MAX_NUM_OBJ)
        keep[:len(keep_)] = keep_

        # Class ids 
        cid = np.array([
            DC.nyu40id2class[self.label_map[scan.get_object_instance_label(k)]]
            for k, kept in enumerate(keep) if kept
        ])
        class_ids = np.zeros((MAX_NUM_OBJ,))
        class_ids[keep] = cid

        # constract object boxes
        all_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        all_bboxes_ = np.stack([
            scan.get_object_bbox(k).reshape(-1)
            for k, kept in enumerate(keep) if kept
        ])
        # cx, cy, cz, w, h, d
        all_bboxes_ = np.concatenate((
            (all_bboxes_[:, :3] + all_bboxes_[:, 3:]) * 0.5,
            all_bboxes_[:, 3:] - all_bboxes_[:, :3]
        ), 1)
        all_bboxes[keep] = all_bboxes_
        if (
            self.split == 'train'
            and self.augment
            and not getattr(self, 'disable_box_jitter', False)
        ):
            all_bboxes *= (0.95 + 0.1*np.random.random((len(all_bboxes), 6)))

        # Which boxes we're interested for
        all_bbox_label_mask = keep
        return class_ids, all_bboxes, all_bbox_label_mask
    

    # BRIEF Search for pseudo-labels of auxiliary objects [not used!]
    def _get_auxi_boxes(self, anno, class_ids, all_bboxes, all_bbox_label_mask, gt_bboxes):
        auxi_box = None

        if anno["dataset"] == "scannet":
            return auxi_box

        if anno["auxi_entity"] is not None:
            auxi_label_lemma = anno["auxi_entity"]["lemma_head"]
            if auxi_label_lemma in self.label_map:
                if self.label_map[auxi_label_lemma] not in list((DC.nyu40id2class).keys()):
                    return auxi_box
                
                cls_id = DC.nyu40id2class[self.label_map[auxi_label_lemma]]
                dis_min = 100
                target_box = gt_bboxes[0]
                for idx, mask in enumerate(all_bbox_label_mask):
                    if anno['target_id'] == idx or mask == False:
                        continue
                    if class_ids[idx] == cls_id:
                        dis = target_box[:3] - all_bboxes[idx][:3]
                        dis = np.sum(dis**2)
                        
                        if dis < dis_min:
                            dis_min = dis
                            auxi_box = all_bboxes[idx]
        return auxi_box

    # BRIEF GroupFree detection boxes
    def _get_detected_objects(self, split, scan_id, augmentations):
        # Initialize
        all_detected_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        all_detected_bbox_label_mask = np.array([False] * MAX_NUM_OBJ)
        detected_class_ids = np.zeros((MAX_NUM_OBJ,))
        detected_logits = np.zeros((MAX_NUM_OBJ, NUM_CLASSES))

        # note single stage method
        if not self.butd and not self.butd_cls:
            return (
                all_detected_bboxes, all_detected_bbox_label_mask,
                detected_class_ids, detected_logits
            )

        # Load: class, box, pc, logits
        detected_dict = np.load(
            f'{self.data_path}/group_free_pred_bboxes/group_free_pred_bboxes_{split}/{scan_id}.npy',
            allow_pickle=True
        ).item()

        all_bboxes_ = np.array(detected_dict['box'])
        classes = detected_dict['class']
        cid = np.array([DC.nyu40id2class[
            self.label_map[c]] for c in detected_dict['class']
        ])
        all_bboxes_ = np.concatenate((
            (all_bboxes_[:, :3] + all_bboxes_[:, 3:]) * 0.5,
            all_bboxes_[:, 3:] - all_bboxes_[:, :3]
        ), 1)

        assert len(classes) < MAX_NUM_OBJ
        assert len(classes) == all_bboxes_.shape[0]

        num_objs = len(classes)
        all_detected_bboxes[:num_objs] = all_bboxes_
        all_detected_bbox_label_mask[:num_objs] = np.array([True] * num_objs)
        detected_class_ids[:num_objs] = cid
        detected_logits[:num_objs] = detected_dict['logits']    # logits
        # Match current augmentations
        valid_detected = all_detected_bbox_label_mask.astype(bool)
        if self.augment and self.split == 'train' and valid_detected.any():
            all_det_pts = box2points(all_detected_bboxes[valid_detected]).reshape(-1, 3)
            if augmentations.get('yz_flip', False):
                all_det_pts[:, 0] = -all_det_pts[:, 0]
            if augmentations.get('xz_flip', False):
                all_det_pts[:, 1] = -all_det_pts[:, 1]
            all_det_pts = rot_z(all_det_pts, augmentations['theta_z'])
            all_det_pts = rot_x(all_det_pts, augmentations['theta_x'])
            all_det_pts = rot_y(all_det_pts, augmentations['theta_y'])
            all_det_pts += augmentations['shift']
            all_det_pts *= augmentations['scale']
            all_detected_bboxes[valid_detected] = points2box(all_det_pts.reshape(-1, 8, 3))
        
        if self.augment_det and self.split == 'train' and valid_detected.any():
            valid_bboxes = all_detected_bboxes[valid_detected]
            min_ = valid_bboxes.min(0)
            max_ = valid_bboxes.max(0)
            rand_box = (
                (max_ - min_)[None]
                * np.random.random(valid_bboxes.shape)
                + min_
            )
            corrupt = np.random.random(len(valid_bboxes)) > 0.7
            valid_indices = np.where(valid_detected)[0]
            corrupt_indices = valid_indices[corrupt]
            all_detected_bboxes[corrupt_indices] = rand_box[corrupt]
            detected_class_ids[corrupt_indices] = np.random.randint(
                0, len(DC.nyu40ids), len(corrupt_indices)
            )
        return (
            all_detected_bboxes, all_detected_bbox_label_mask,
            detected_class_ids, detected_logits
        )

    # BRIEF data
    def __getitem__(self, index):
        """Get current batch for input index."""
        split = self.split

        # step Read annotation and point clouds
        anno = self.annos[index]
        language_dataset = anno['dataset']
        scan = self.scans[anno['scan_id']]
        scan.pc = np.copy(scan.orig_pc)

        # step constract anno (used only for [scannet])
        self.random_utt = False
        if anno['dataset'] == 'scannet':
            self.random_utt = self.joint_det and np.random.random() > 0.5
            sampled_classes = self._sample_classes(anno['scan_id'])
            utterance = self._create_scannet_utterance(sampled_classes)
            
            if not self.random_utt:  # detection18 phrase
                anno['target_id'] = np.where(np.array([
                    self.label_map18[
                        scan.get_object_instance_label(ind)
                    ] in DC18.nyu40id2class
                    for ind in range(len(scan.three_d_objects))
                ])[:MAX_NUM_OBJ])[0].tolist()
            else:
                anno['target_id'] = np.where(np.array([
                    self.label_map[
                        scan.get_object_instance_label(ind)
                    ] in DC.nyu40id2class
                    and
                    DC.class2type[DC.nyu40id2class[self.label_map[
                        scan.get_object_instance_label(ind)
                    ]]] in sampled_classes
                    for ind in range(len(scan.three_d_objects))
                ])[:MAX_NUM_OBJ])[0].tolist()
            
            # Target names
            if not self.random_utt:
                anno['target'] = [
                    DC18.class2type[DC18.nyu40id2class[self.label_map18[
                        scan.get_object_instance_label(ind)
                    ]]]
                    if self.label_map18[
                        scan.get_object_instance_label(ind)
                    ] != 39
                    else 'other furniture'
                    for ind in anno['target_id']
                ]
            else:
                anno['target'] = [
                    DC.class2type[DC.nyu40id2class[self.label_map[
                        scan.get_object_instance_label(ind)
                    ]]]
                    for ind in anno['target_id']
                ]
            anno['utterance'] = utterance

        # step Point cloud representation
        point_cloud, augmentations, og_color = self._get_pc(anno, scan)

        # step "Target" boxes: append anchors if they're to be detected
        gt_bboxes, box_label_mask, point_instance_label = \
            self._get_target_boxes(anno, scan)

        # step Scene gt boxes
        (
            class_ids, all_bboxes, all_bbox_label_mask
        ) = self._get_scene_objects(scan)

        # not used
        auxi_box = self._get_auxi_boxes(anno, class_ids, all_bboxes, all_bbox_label_mask, gt_bboxes)

        ##########################
        # STEP Get text position #
        ##########################
        if anno['dataset'] == 'scannet':
            tokens_positive, positive_map, modify_positive_map, pron_positive_map, \
                other_entity_map, auxi_entity_positive_map, rel_positive_map = self._get_token_positive_map(anno)
        else:
            # note text parsing
            tokens_positive, positive_map, modify_positive_map, pron_positive_map, \
                other_entity_map, auxi_entity_positive_map, rel_positive_map = self._get_token_positive_map_by_parse(anno, auxi_box)
        if auxi_box is None:
            auxi_box = np.zeros((1, 6))
        else:
            auxi_box = np.expand_dims(auxi_box, axis=0)
        
        # step groupfree Detected boxes
        (
            all_detected_bboxes, all_detected_bbox_label_mask,
            detected_class_ids, detected_logits
        ) = self._get_detected_objects(split, anno['scan_id'], augmentations)

        # Assume a perfect object detector
        if self.butd_gt:
            all_detected_bboxes = all_bboxes
            all_detected_bbox_label_mask = all_bbox_label_mask
            detected_class_ids = class_ids

        # Assume a perfect object proposal stage
        if self.butd_cls:
            all_detected_bboxes = all_bboxes
            all_detected_bbox_label_mask = all_bbox_label_mask
            detected_class_ids = np.zeros((len(all_bboxes,)))
            classes = np.array(self.cls_results[anno['scan_id']])
            detected_class_ids[all_bbox_label_mask] = classes[classes > -1]

        # Visualize for debugging
        if self.visualize and anno['dataset'].startswith('sr3d'):
            self._visualize_scene(anno, point_cloud, og_color, all_bboxes)

        # Return
        _labels = np.zeros(MAX_NUM_OBJ)
        if not isinstance(anno['target_id'], int) and not self.random_utt:
            _labels[:len(anno['target_id'])] = np.array([
                DC18.nyu40id2class[self.label_map18[
                    scan.get_object_instance_label(ind)
                ]]
                for ind in anno['target_id']
            ])
        ret_dict = {
            'box_label_mask': box_label_mask.astype(np.float32),
            'center_label': gt_bboxes[:, :3].astype(np.float32),
            'sem_cls_label': _labels.astype(np.int64),
            'size_gts': gt_bboxes[:, 3:].astype(np.float32),
        }
        ret_dict.update({
            "scan_ids": anno['scan_id'],
            "point_clouds": point_cloud.astype(np.float32),
            "og_color": og_color.astype(np.float32),
            "utterances": (
                ' '.join(anno['utterance'].replace(',', ' ,').split())
                + ' . not mentioned'
            ),
            "language_dataset": language_dataset,
            "tokens_positive": tokens_positive.astype(np.int64),
            # NOTE text component position label
            "positive_map": positive_map.astype(np.float32),                # main object
            "modify_positive_map": modify_positive_map.astype(np.float32),  # modift(attribute)
            "pron_positive_map": pron_positive_map.astype(np.float32),      # pron
            "other_entity_map": other_entity_map.astype(np.float32),        # other(auxi) object
            "rel_positive_map": rel_positive_map.astype(np.float32),        # relation
            "auxi_entity_positive_map": auxi_entity_positive_map.astype(np.float32),
            "auxi_box":auxi_box.astype(np.float32),
            "relation": (
                self._find_rel(anno['utterance'])
                if anno['dataset'].startswith('sr3d')
                else "none"
            ),
            "target_name": scan.get_object_instance_label(
                anno['target_id'] if isinstance(anno['target_id'], int)
                else anno['target_id'][0]
            ),
            "target_id": (
                anno['target_id'] if isinstance(anno['target_id'], int)
                else anno['target_id'][0]
            ),
            "point_instance_label": point_instance_label.astype(np.int64),
            "all_bboxes": all_bboxes.astype(np.float32),
            "all_bbox_label_mask": all_bbox_label_mask.astype(np.bool8),
            "all_class_ids": class_ids.astype(np.int64),
            "distractor_ids": np.array(
                anno['distractor_ids']
                + [-1] * (32 - len(anno['distractor_ids']))
            ).astype(int),
            "anchor_ids": np.array(
                anno['anchor_ids']
                + [-1] * (32 - len(anno['anchor_ids']))
            ).astype(int),
            "anchor_span_ids": np.array(
                anno.get('anchor_span_ids', [])
                + [-1] * (32 - len(anno.get('anchor_span_ids', [])))
            ).astype(int),
            "all_detected_boxes": all_detected_bboxes.astype(np.float32),
            "all_detected_bbox_label_mask": all_detected_bbox_label_mask.astype(np.bool8),
            "all_detected_class_ids": detected_class_ids.astype(np.int64),
            "all_detected_logits": detected_logits.astype(np.float32),
            "is_view_dep": self._is_view_dep(anno['utterance']),
            "is_hard": len(anno['distractor_ids']) > 1,
            "is_unique": len(anno['distractor_ids']) == 0,
            "target_cid": (
                class_ids[anno['target_id']]
                if isinstance(anno['target_id'], int)
                else class_ids[anno['target_id'][0]]
            ),
            "target_slot": anno.get('target_slot', {}),
            "entity_spans": anno.get('entity_spans', []),
            "attr_spans": anno.get('attr_spans', []),
            "rel_spans": anno.get('rel_spans', []),
            "description": anno.get('description', anno.get('utterance', '')),
            "scene_id": anno.get('scene_id', anno.get('scan_id', '')),
            "object_id": anno.get('object_id', anno.get('target_id', -1)),
            "object_name": anno.get('object_name', anno.get('target', '')),
            "ann_id": anno.get('ann_id', ''),
            "attr_slot": anno.get('attr_slot', {}),
            "rel_slots": anno.get('rel_slots', []),
            "anchor_slots": anno.get('anchor_slots', []),
            "slot_mask": anno.get('slot_mask', {}),
            "parse_confidence": anno.get('parse_confidence', 1.0),
            "coverage_stats": anno.get('coverage_stats', {}),
            "spacy_rotation_mode_id": np.int64(
                self._spacy_rotation_mode_id_for_sample(anno)
            ),
            "spacy_augmentation_profile_id": np.int64(
                self._spacy_augmentation_profile_id(anno)
            ),
            "decomp_global_only_mask": self._decomp_global_only_mask_for_sample(
                anno
            ),
            "decomp_weak_generic_mask": self._decomp_weak_generic_mask_for_sample(
                anno
            ),
            "decomp_train_global_only_mask": anno.get(
                'decomp_train_global_only_mask', False
            ),
            "decomp_train_weak_generic_mask": anno.get(
                'decomp_train_weak_generic_mask', False
            ),
            "decomp_train_global_only_reason": anno.get(
                'decomp_train_global_only_reason', ''
            ),
            "decomposition_error_flags_count": anno.get(
                'decomposition_error_flags_count', 0
            ),
            "metadata_conflict_ratio": float(
                anno.get('metadata_conflict_count', 0) > 0
            ),
        })

        return ret_dict

    @staticmethod
    def _is_view_dep(utterance):
        """Check whether to augment based on nr3d utterance."""
        rels = [
            'front', 'behind', 'back', 'left', 'right', 'facing',
            'leftmost', 'rightmost', 'looking', 'across'
        ]
        utterance = ' ' + str(utterance).lower().replace('_', ' ') + ' '
        return any(' ' + rel + ' ' in utterance for rel in rels)
    
    @staticmethod
    def _find_rel(utterance):
        utterance = ' ' + utterance.replace(',', ' ,') + ' '
        relation = "none"
        sorted_rel_list = sorted(REL_ALIASES, key=len, reverse=True)
        for rel in sorted_rel_list:
            if ' ' + rel + ' ' in utterance:
                relation = REL_ALIASES[rel]
                break
        return relation
    
    @staticmethod
    def _augment_nr3d(utterance):
        """Check whether to augment based on nr3d utterance."""
        rels = [
            'front', 'behind', 'back', 'left', 'right', 'facing',
            'leftmost', 'rightmost', 'looking', 'across'
        ]
        utterance = ' ' + str(utterance).lower().replace('_', ' ') + ' '
        return not any(' ' + rel + ' ' in utterance for rel in rels)

    @staticmethod
    def _spacy_has_raw_view_word(utterance):
        rels = (
            'front', 'behind', 'back', 'left', 'right', 'facing',
            'leftmost', 'rightmost', 'looking', 'across'
        )
        pattern = r'\b(?:' + '|'.join(re.escape(rel) for rel in rels) + r')\b'
        return re.search(pattern, str(utterance).lower().replace('_', ' ')) is not None

    @staticmethod
    def _spacy_has_compass_direction_word(utterance):
        text = str(utterance).lower().replace('_', ' ')
        compass = (
            r'north(?:ern)?',
            r'south(?:ern)?',
            r'east(?:ern)?',
            r'west(?:ern)?',
            r'north[-\s]?east(?:ern)?',
            r'north[-\s]?west(?:ern)?',
            r'south[-\s]?east(?:ern)?',
            r'south[-\s]?west(?:ern)?',
        )
        compass_pat = r'(?:' + '|'.join(compass) + r')(?:[-\s]?most)?'
        spatial_nouns = (
            'side', 'corner', 'edge', 'wall', 'area', 'section', 'part',
            'end', 'entrance', 'room'
        )
        direction_context = (
            r'\b(?:to|toward|towards|from|closest\s+to|nearest\s+to|'
            r'farthest|furthest|directly|approximately|in|on|at|along|'
            r'against|below|above|under|over)\s+(?:the\s+)?'
        )
        patterns = (
            direction_context + compass_pat + r'\b',
            r'\b' + compass_pat + r'\s+(?:' + '|'.join(spatial_nouns) + r')\b',
        )
        return any(re.search(pattern, text) is not None for pattern in patterns)

    @staticmethod
    def _spacy_relation_text(slot):
        if isinstance(slot, dict):
            values = []
            for key in (
                'text', 'surface_text', 'pattern', 'frame_cue_text',
                'head', 'tail', 'relation', 'rel', 'lemma_head'
            ):
                value = slot.get(key, '')
                if isinstance(value, (list, tuple)):
                    values.extend(str(item) for item in value)
                else:
                    values.append(str(value))
            return ' '.join(values).replace('_', ' ')
        return str(slot).replace('_', ' ')

    @staticmethod
    def _spacy_positive_count(value):
        try:
            return int(value) > 0
        except (TypeError, ValueError):
            return False

    @classmethod
    def _spacy_relation_free_rawview_parse_noise(cls, anno):
        if not str(anno.get('dataset', '')).endswith('_spacy'):
            return False
        if cls._spacy_rotation_mode(anno) is not None:
            return False
        if not cls._spacy_has_raw_view_word(anno.get('utterance', '')):
            return False

        coverage_stats = anno.get('coverage_stats', {})
        if isinstance(coverage_stats, dict):
            noisy_keys = (
                'candidate_relation_count',
                'invalid_relation_count',
                'spatial_attribute_rows',
                'spatial_info_routed_to_attr',
            )
            if any(
                cls._spacy_positive_count(coverage_stats.get(key, 0))
                for key in noisy_keys
            ):
                return True

        reason = str(anno.get('manual_review_fix_reason', '')).lower()
        noisy_reason_terms = (
            'background',
            'invalid_relation',
            'relation_dropped',
            'relation_tuples_dropped',
        )
        return any(term in reason for term in noisy_reason_terms)

    def _decomp_global_only_mask_for_sample(self, anno):
        global_only = bool(anno.get('decomp_global_only_mask', False))
        if (
            self.split == 'train'
            and bool(anno.get('decomp_train_global_only_mask', False))
        ):
            return True
        if (
            getattr(
                self, 'spacy_relation_free_rawview_global_only_train', False
            )
            and self.split == 'train'
            and self._spacy_relation_free_rawview_parse_noise(anno)
        ):
            return True
        return global_only

    def _decomp_weak_generic_mask_for_sample(self, anno):
        weak_generic = bool(anno.get('decomp_weak_generic_mask', False))
        if (
            self.split == 'train'
            and bool(anno.get('decomp_train_weak_generic_mask', False))
        ):
            return True
        return weak_generic

    @classmethod
    def _spacy_has_frame_cue(cls, anno):
        coverage_stats = anno.get('coverage_stats', {})
        if isinstance(coverage_stats, dict):
            if cls._spacy_positive_count(coverage_stats.get('frame_cue_count', 0)):
                return True
        rel_slots = anno.get('rel_slots', [])
        if not isinstance(rel_slots, list):
            rel_slots = [rel_slots]
        for slot in rel_slots:
            if not isinstance(slot, dict):
                continue
            if slot.get('frame_cue_flag') or slot.get('frame_cue_text'):
                return True
        return False

    @staticmethod
    def _spacy_spatial_attr_text(item):
        if isinstance(item, dict):
            values = []
            for key in ('text', 'canonical_text', 'head', 'type'):
                values.append(str(item.get(key, '')))
            return ' '.join(values).replace('_', ' ')
        return str(item).replace('_', ' ')

    @classmethod
    def _spacy_spatial_attribute_mode(cls, anno):
        attr_slot = anno.get('attr_slot', {})
        items = attr_slot.get('items', []) if isinstance(attr_slot, dict) else []
        spatial_terms = (
            'top', 'bottom', 'upper', 'lower', 'center', 'centre', 'middle',
            'corner', 'side', 'edge', 'above', 'below', 'under', 'over',
            'beneath'
        )

        has_spatial_attr = False
        spatial_text = []
        for item in items if isinstance(items, list) else []:
            text = cls._spacy_spatial_attr_text(item)
            padded = ' ' + text.lower().strip() + ' '
            if (
                isinstance(item, dict)
                and item.get('type') == 'spatial_attribute'
            ) or any(f' {term} ' in padded for term in spatial_terms):
                has_spatial_attr = True
                spatial_text.append(text)

        coverage_stats = anno.get('coverage_stats', {})
        if isinstance(coverage_stats, dict):
            has_spatial_attr = (
                has_spatial_attr
                or cls._spacy_positive_count(
                    coverage_stats.get('spatial_attribute_rows', 0)
                )
            )

        if not has_spatial_attr:
            return None
        if spatial_text and not cls._augment_nr3d(' '.join(spatial_text)):
            return "none"
        return "yaw_only"

    @classmethod
    def _spacy_rotation_mode(cls, anno):
        """Choose geometry augmentation that preserves decomposed relation meaning."""
        if not str(anno.get('dataset', '')).endswith('_spacy'):
            return None
        if not cls._augment_nr3d(anno.get('utterance', '')):
            return "none"

        rel_slots = anno.get('rel_slots', [])
        if not isinstance(rel_slots, list):
            rel_slots = [rel_slots]
        has_relation = False
        for slot in rel_slots:
            relation_text = cls._spacy_relation_text(slot)
            if not relation_text.strip():
                continue
            has_relation = True
            if (
                not cls._augment_nr3d(relation_text)
                or cls._spacy_has_raw_view_word(relation_text)
                or cls._spacy_has_compass_direction_word(relation_text)
            ):
                return "none"
        return "yaw_only" if has_relation else None

    @classmethod
    def _spacy_rotation_mode_id(cls, anno):
        if not str(anno.get('dataset', '')).endswith('_spacy'):
            return -1
        mode = cls._spacy_rotation_mode(anno)
        if mode == "none":
            return 1
        if mode == "yaw_only":
            return 2
        return 3

    def _spacy_rotation_mode_for_sample(self, anno):
        mode = self._spacy_rotation_mode(anno)
        if (
            mode is None
            and str(anno.get('dataset', '')).endswith('_spacy')
            and getattr(self, 'spacy_relation_free_yaw_only_aug', False)
        ):
            if (
                getattr(self, 'spacy_relation_free_view_guard_aug', False)
                and self._spacy_has_raw_view_word(anno.get('utterance', ''))
            ):
                return "none"
            if (
                getattr(self, 'spacy_relation_free_view_small_yaw_aug', False)
                and self._spacy_has_raw_view_word(anno.get('utterance', ''))
            ):
                return "small_yaw"
            if (
                getattr(
                    self, 'spacy_relation_free_compass_guard_aug', False
                )
                and self._spacy_has_compass_direction_word(
                    anno.get('utterance', '')
                )
            ):
                return "none"
            if getattr(self, 'spacy_relation_free_none_aug', False):
                return "none"
            if getattr(self, 'spacy_relation_free_stable_yaw_aug', False):
                return "yaw_stable"
            return "yaw_only"
        return mode

    def _spacy_rotation_mode_id_for_sample(self, anno):
        if not str(anno.get('dataset', '')).endswith('_spacy'):
            return -1
        mode = self._spacy_rotation_mode_for_sample(anno)
        if mode == "none":
            return 1
        if mode == "yaw_only":
            return 2
        if mode == "yaw_stable":
            return 2
        if mode == "small_yaw":
            return 2
        return 3

    def _spacy_augmentation_profile_id(self, anno):
        if not str(anno.get('dataset', '')).endswith('_spacy'):
            return -1
        base_mode = self._spacy_rotation_mode(anno)
        if base_mode == "none":
            return 1
        if base_mode == "yaw_only":
            return 2
        if getattr(self, 'spacy_relation_free_yaw_only_aug', False):
            if (
                getattr(
                    self,
                    'spacy_relation_free_rawview_global_only_train',
                    False,
                )
                and self._spacy_relation_free_rawview_parse_noise(anno)
            ):
                return 8
            if (
                getattr(
                    self, 'spacy_relation_free_compass_guard_aug', False
                )
                and self._spacy_has_compass_direction_word(
                    anno.get('utterance', '')
                )
            ):
                return 10
            if getattr(self, 'spacy_relation_free_none_aug', False):
                return 9
            if (
                getattr(self, 'spacy_relation_free_view_guard_aug', False)
                and self._spacy_has_raw_view_word(anno.get('utterance', ''))
            ):
                return 5
            if (
                getattr(self, 'spacy_relation_free_view_small_yaw_aug', False)
                and self._spacy_has_raw_view_word(anno.get('utterance', ''))
            ):
                return 7
            if getattr(self, 'spacy_relation_free_stable_yaw_aug', False):
                return 6
            return 4
        return 3

    @classmethod
    def _spacy_relations_allow_rotation(cls, anno):
        """Keep decomposed view-dependent relation slots geometrically valid."""
        return cls._spacy_rotation_mode(anno) != "none"

    def _visualize_scene(self, anno, point_cloud, og_color, all_bboxes):
        target_id = anno['target_id']
        distractor_ids = np.array(
            anno['distractor_ids']
            + [-1] * (10 - len(anno['distractor_ids']))
        ).astype(int)
        anchor_ids = np.array(
            anno['anchor_ids']
            + [-1] * (10 - len(anno['anchor_ids']))
        ).astype(int)
        point_cloud[:, 3:] = (og_color + self.mean_rgb) * 256

        all_boxes_points = box2points(all_bboxes[..., :6])

        target_box = all_boxes_points[target_id]
        anchors_boxes = all_boxes_points[[
            i.item() for i in anchor_ids if i != -1
        ]]
        distractors_boxes = all_boxes_points[[
            i.item() for i in distractor_ids if i != -1
        ]]

        wandb.log({
            "ground_truth_point_scene": wandb.Object3D({
                "type": "lidar/beta",
                "points": point_cloud,
                "boxes": np.array(
                    [  # target
                        {
                            "corners": target_box.tolist(),
                            "label": "target",
                            "color": [0, 255, 0]
                        }
                    ]
                    + [  # anchors
                        {
                            "corners": c.tolist(),
                            "label": "anchor",
                            "color": [0, 0, 255]
                        }
                        for c in anchors_boxes
                    ]
                    + [  # distractors
                        {
                            "corners": c.tolist(),
                            "label": "distractor",
                            "color": [0, 255, 255]
                        }
                        for c in distractors_boxes
                    ]
                    + [  # other
                        {
                            "corners": c.tolist(),
                            "label": "other",
                            "color": [255, 0, 0]
                        }
                        for i, c in enumerate(all_boxes_points)
                        if i not in (
                            [target_id]
                            + anchor_ids.tolist()
                            + distractor_ids.tolist()
                        )
                    ]
                )
            }),
            "utterance": wandb.Html(anno['utterance']),
        })
    
    def __len__(self):
        """Return number of utterances."""
        return len(self.annos)

# BRIEF Construct position label(map)
def get_positive_map(tokenized, tokens_positive):
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)  # ([positive], 256])
    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list
        beg = int(beg)
        end = int(end)
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        positive_map[j, beg_pos:end_pos + 1].fill_(1)

    positive_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-12)
    return positive_map.numpy()


def rot_x(pc, theta):
    """Rotate along x-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [1.0, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ]),
        pc.T
    ).T


def rot_y(pc, theta):
    """Rotate along y-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1.0, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ]),
        pc.T
    ).T


def rot_z(pc, theta):
    """Rotate along z-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1.0]
        ]),
        pc.T
    ).T

def box2points(box):
    """Convert box center/hwd coordinates to vertices (8x3)."""
    x_min, y_min, z_min = (box[:, :3] - (box[:, 3:] / 2)).transpose(1, 0)
    x_max, y_max, z_max = (box[:, :3] + (box[:, 3:] / 2)).transpose(1, 0)
    return np.stack((
        np.concatenate((x_min[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_max[:, None]), 1)
    ), axis=1)


def points2box(box):
    """Convert vertices (Nx8x3) to box center/hwd coordinates (Nx6)."""
    return np.concatenate((
        (box.min(1) + box.max(1)) / 2,
        box.max(1) - box.min(1)
    ), axis=1)

# BRIEF load scannet
def scannet_loader(iter_obj):
    """Load the scans in memory, helper function."""
    scan_id, scan_path = iter_obj
    print(scan_id)
    return Scan(scan_id, scan_path, True)

# BRIEF Save all scans to pickle.
def save_data(filename, split, data_path):
    """Save all scans to pickle."""
    import multiprocessing as mp

    # Read all scan files
    scan_path = data_path + 'scans/'
    with open('data/meta_data/scannetv2_%s.txt' % split) as f:
        scan_ids = [line.rstrip() for line in f]    # train/val scene id list.
    print('{} scans found.'.format(len(scan_ids)))

    # Load data
    n_items = len(scan_ids)
    n_processes = 4  # min(mp.cpu_count(), n_items)
    pool = mp.Pool(n_processes)
    chunks = int(n_items / n_processes)
    all_scans = dict()
    iter_obj = [
        (scan_id, scan_path)
        for scan_id in scan_ids
    ]

    for i, data in enumerate(
        pool.imap(scannet_loader, iter_obj, chunksize=chunks)
    ):
        all_scans[scan_ids[i]] = data
    pool.close()
    pool.join()

    # Save data
    print('pickle time')
    pickle_data(filename, all_scans)


def pickle_data(file_name, *args):
    """Use (c)Pickle to save multiple objects in a single file."""
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()

# BRIEF read from pkl
def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data()."""
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()


#########################
# BRIEF Text decoupling #
#########################
def Scene_graph_parse(annos):
    import sys
    sys.path.append(os.getcwd())
    import sng_parser

    print('Begin text decoupling......')
    for anno in annos:
        caption = ' '.join(anno['utterance'].replace(',', ' , ').split())

        # some error or typo in ScanRefer.
        caption = ' '.join(caption.replace("'m", "am").split())
        caption = ' '.join(caption.replace("'s", "is").split())
        caption = ' '.join(caption.replace("2-tiered", "2 - tiered").split())
        caption = ' '.join(caption.replace("4-drawers", "4 - drawers").split())
        caption = ' '.join(caption.replace("5-drawer", "5 - drawer").split())
        caption = ' '.join(caption.replace("8-hole", "8 - hole").split())
        caption = ' '.join(caption.replace("7-shaped", "7 - shaped").split())
        caption = ' '.join(caption.replace("2-door", "2 - door").split())
        caption = ' '.join(caption.replace("3-compartment", "3 - compartment").split())
        caption = ' '.join(caption.replace("computer/", "computer /").split())
        caption = ' '.join(caption.replace("3-tier", "3 - tier").split())
        caption = ' '.join(caption.replace("3-seater", "3 - seater").split())
        caption = ' '.join(caption.replace("4-seat", "4 - seat").split())
        caption = ' '.join(caption.replace("theses", "these").split())
        
        # some error or typo in NR3D.
        if anno['dataset'] == 'nr3d':
            caption = ' '.join(caption.replace('.', ' .').split())
            caption = ' '.join(caption.replace(';', ' ; ').split())
            caption = ' '.join(caption.replace('-', ' ').split())
            caption = ' '.join(caption.replace('"', ' ').split())
            caption = ' '.join(caption.replace('?', ' ').split())
            caption = ' '.join(caption.replace("*", " ").split())
            caption = ' '.join(caption.replace(':', ' ').split())
            caption = ' '.join(caption.replace('$', ' ').split())
            caption = ' '.join(caption.replace("#", " ").split())
            caption = ' '.join(caption.replace("/", " / ").split())
            caption = ' '.join(caption.replace("you're", "you are").split())
            caption = ' '.join(caption.replace("isn't", "is not").split())
            caption = ' '.join(caption.replace("thats", "that is").split())
            caption = ' '.join(caption.replace("doesn't", "does not").split())
            caption = ' '.join(caption.replace("doesnt", "does not").split())
            caption = ' '.join(caption.replace("itis", "it is").split())
            caption = ' '.join(caption.replace("left-hand", "left - hand").split())
            caption = ' '.join(caption.replace("[", " [ ").split())
            caption = ' '.join(caption.replace("]", " ] ").split())
            caption = ' '.join(caption.replace("(", " ( ").split())
            caption = ' '.join(caption.replace(")", " ) ").split())
            caption = ' '.join(caption.replace("wheel-chair", "wheel - chair").split())
            caption = ' '.join(caption.replace(";s", "is").split())
            caption = ' '.join(caption.replace("tha=e", "the").split())
            caption = ' '.join(caption.replace("it’s", "it is").split())
            caption = ' '.join(caption.replace("’s", " is").split())
            caption = ' '.join(caption.replace("isnt", "is not").split())
            caption = ' '.join(caption.replace("Don't", "Do not").split())
            caption = ' '.join(caption.replace("arent", "are not").split())
            caption = ' '.join(caption.replace("cant", "can not").split())
            caption = ' '.join(caption.replace("you’re", "you are").split())
            caption = ' '.join(caption.replace('!', ' !').split())
            caption = ' '.join(caption.replace('id the', ' , the').split())
            caption = ' '.join(caption.replace('youre', 'you are').split())

            caption = ' '.join(caption.replace("'", ' ').split())

            if caption[0] == "'":
                caption = caption[1:]
            if caption[-1] == "'":
                caption = caption[:-1]
        
        anno['utterance'] = caption

        # text parsing
        graph_node, graph_edge = sng_parser.parse(caption)

        # NOTE If no node is parsed, add "this is an object ." at the beginning of the sentence
        if (len(graph_node) < 1) or \
            (len(graph_node) > 0 and graph_node[0]["node_id"] != 0):
            caption = "This is an object . " + caption
            anno['utterance'] = caption

            # parse again
            graph_node, graph_edge = sng_parser.parse(caption)

        # node and edge
        anno["graph_node"] = graph_node
        anno["graph_edge"] = graph_edge

        # auxi object
        auxi_entity = None
        for node in graph_node:
            if (node["node_id"] != 0) and (node["node_type"] == "Object"):
                auxi_entity = node
                break
        anno["auxi_entity"] = auxi_entity
    
    print('End text decoupling!')
