# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""Dataset and data loader for ReferIt3D."""

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

    _TEXT_TARGET_CLASS_ALIASES = None

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
        11: "stable_direction_sensitive",
    }

    def __init__(self, dataset_dict={'sr3d': 1, 'scannet': 10},
                 test_dataset='sr3d',
                 split='train', overfit=False,
                 data_path='./',
                 use_color=False, use_height=False, use_multiview=False,
                 detect_intermediate=False,
                 butd=False, butd_gt=False, butd_cls=False, augment_det=False,
                 disable_box_jitter=False,
                 disable_train_augmentation=False,
                 spacy_relation_free_yaw_only_aug=False,
                 spacy_relation_free_view_guard_aug=False,
                 spacy_relation_free_stable_yaw_aug=False,
                 spacy_relation_free_view_small_yaw_aug=False,
                 spacy_relation_free_rawview_global_only_train=False,
                 spacy_relation_free_none_aug=False,
                 spacy_relation_free_compass_guard_aug=False,
                 spacy_direction_sensitive_no_jitter_aug=False,
                 scanrefer_inject_spacy_decomp=False,
                 text_target_alias_policy='strict'):
        """Initialize dataset (here for ReferIt3D utterances)."""
        self.dataset_dict = dataset_dict
        self.test_dataset = test_dataset
        self.split = split
        self.use_color = use_color
        self.use_height = use_height
        self.overfit = overfit
        self.detect_intermediate = detect_intermediate
        self.augment = self.split == 'train'
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
        self.disable_train_augmentation = disable_train_augmentation
        self.spacy_relation_free_yaw_only_aug = spacy_relation_free_yaw_only_aug
        self.spacy_relation_free_view_guard_aug = spacy_relation_free_view_guard_aug
        self.spacy_relation_free_stable_yaw_aug = spacy_relation_free_stable_yaw_aug
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
        self.spacy_direction_sensitive_no_jitter_aug = (
            spacy_direction_sensitive_no_jitter_aug
        )
        self.scanrefer_inject_spacy_decomp = scanrefer_inject_spacy_decomp
        self.text_target_alias_policy = text_target_alias_policy

        self.mean_rgb = np.array([109.8, 97.2, 83.8]) / 256
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
        self.tokenizer = RobertaTokenizerFast.from_pretrained("/root/autodl-tmp/DATA_ROOT/roberta-base")
        if os.path.exists('data/cls_results.json'):
            with open('data/cls_results.json') as fid:
                self.cls_results = json.load(fid)  # {scan_id: [class0, ...]}

        # load
        print('Loading %s files, take a breath!' % split)
        if not os.path.exists(f'{self.data_path}/{split}_v3scans.pkl'):
            save_data(f'{data_path}/{split}_v3scans.pkl', split, data_path)
        self.scans = unpickle_data(f'{self.data_path}/{split}_v3scans.pkl')
        self.scans = list(self.scans)[0]
        if self.split != 'train':
            self.annos = self.load_annos(test_dataset)
        else:
            self.annos = []
            for dset, cnt in dataset_dict.items():
                if cnt > 0:
                    _annos = self.load_annos(dset)
                    self.annos += (_annos * cnt)

        if self.visualize:
            wandb.init(project="vis", name="debug")

    def _train_augmentation_enabled(self):
        return (
            self.split == 'train'
            and getattr(self, 'augment', self.split == 'train')
            and not getattr(self, 'disable_train_augmentation', False)
        )

    def load_annos(self, dset):
        """Load annotations of given dataset."""
        loaders = {
            'nr3d': self.load_nr3d_annos,
            # nr3d_spacy must read nr3d_spacy.csv (contains entity/attr/rel columns)
            'nr3d_spacy': lambda: self.load_nr3d_annos(dset='nr3d_spacy'),
            'sr3d': self.load_sr3d_annos,
            # sr3d_spacy must read sr3d_spacy.csv (contains entity/attr/rel columns)
            'sr3d_spacy': lambda: self.load_sr3d_annos(dset='sr3d_spacy'),
            'sr3d+': self.load_sr3dplus_annos,
            'scanrefer': self.load_scanrefer_annos,
            'scanrefer_spacy': lambda: self.load_scanrefer_annos(dset='scanrefer_spacy'),
            'scannet': self.load_scannet_annos
        }
        annos = loaders[dset]()
        if self.overfit:
            annos = annos[:128]
        return annos

    @staticmethod
    def _normalize_anchor_text(text):
        return ' '.join(str(text).lower().replace('_', ' ').split())

    @classmethod
    def _infer_anchor_span_ids(cls, anchors, entity_spans):
        anchor_span_ids = []
        used = set()
        for anchor in anchors:
            anchor_norm = cls._normalize_anchor_text(anchor)
            match_idx = -1
            for idx, span in enumerate(entity_spans):
                if idx in used:
                    continue
                span_norm = cls._normalize_anchor_text(span.get('text', ''))
                if not span_norm:
                    continue
                if (span_norm == anchor_norm or anchor_norm in span_norm or span_norm in anchor_norm):
                    match_idx = idx
                    used.add(idx)
                    break
            anchor_span_ids.append(match_idx)
        return anchor_span_ids

    @staticmethod
    def _as_bool(value, default=False):
        if value is None:
            return bool(default)
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y'}
        return bool(value)

    @staticmethod
    def _normalize_text_target_phrase(text):
        text = str(text or '').lower().replace('_', ' ')
        text = re.sub(r'[^a-z0-9 ]+', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    @classmethod
    def _text_target_class_aliases(cls):
        """Return deployable text aliases for ScanNet 485 class ids."""
        if cls._TEXT_TARGET_CLASS_ALIASES is not None:
            return cls._TEXT_TARGET_CLASS_ALIASES

        aliases = {}
        for name, cid in DC.type2class.items():
            norm = cls._normalize_text_target_phrase(name)
            if norm:
                aliases[norm] = int(cid)
            if norm and not norm.endswith('s'):
                aliases.setdefault(norm + 's', int(cid))

        metadata_path = 'data/meta_data/scannetv2-labels.combined.tsv'
        if os.path.exists(metadata_path):
            with open(metadata_path) as csvfile:
                reader = csv.DictReader(csvfile, delimiter='\t')
                for row in reader:
                    category = cls._normalize_text_target_phrase(
                        row.get('category', '')
                    )
                    cid = DC.type2class.get(category, None)
                    if cid is None:
                        continue
                    for key in (
                        'raw_category', 'category', 'nyuClass',
                        'nyu40class', 'ModelNet40', 'ModelNet10',
                        'ShapeNetCore55', 'mpcat40',
                    ):
                        alias = cls._normalize_text_target_phrase(
                            row.get(key, '')
                        )
                        if alias:
                            aliases.setdefault(alias, int(cid))
                        if alias and not alias.endswith('s'):
                            aliases.setdefault(alias + 's', int(cid))

        cls._TEXT_TARGET_CLASS_ALIASES = tuple(
            sorted(
                aliases.items(),
                key=lambda item: (
                    -len(item[0].split()),
                    -len(item[0]),
                    item[0],
                ),
            )
        )
        return cls._TEXT_TARGET_CLASS_ALIASES

    @staticmethod
    def _text_target_alias_policy_remap(alias_policy):
        policy = str(alias_policy or 'strict').strip().lower()
        if policy in ('strict', 'none', ''):
            return {}
        high_precision = {
            'can': DC.type2class['trash can'],
            'painting': DC.type2class['picture'],
            'sofa': DC.type2class['couch'],
        }
        if policy == 'run325_high_precision':
            return high_precision
        if policy == 'run325_no_computer':
            remap = dict(high_precision)
            remap.update({
                'board': DC.type2class['whiteboard'],
                'round table': DC.type2class['table'],
            })
            return remap
        raise ValueError(
            'Unsupported text_target_alias_policy: {}'.format(alias_policy)
        )

    @classmethod
    def _target_slot_text_candidates(cls, anno):
        target_slot = anno.get('target_slot', {})
        candidates = []
        if isinstance(target_slot, dict):
            for key in ('text', 'lemma', 'head', 'name', 'span'):
                value = target_slot.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.append(value)
        elif isinstance(target_slot, str) and target_slot.strip():
            candidates.append(target_slot)
        elif isinstance(target_slot, (list, tuple)):
            for item in target_slot:
                if isinstance(item, str) and item.strip():
                    candidates.append(item)
                elif isinstance(item, dict):
                    value = item.get('text')
                    if isinstance(value, str) and value.strip():
                        candidates.append(value)
        return candidates

    @classmethod
    def _text_target_cid_from_annotation(cls, anno, alias_policy='strict'):
        """Infer target class from deployable target-slot text only.

        This intentionally avoids object_name/object_id/target_id and does not
        scan the full description, where anchors such as "door" can be
        mistaken for the target in generic references.
        """
        alias_remap = cls._text_target_alias_policy_remap(alias_policy)
        for text in cls._target_slot_text_candidates(anno):
            norm_text = ' ' + cls._normalize_text_target_phrase(text) + ' '
            if not norm_text.strip():
                continue
            for alias, cid in cls._text_target_class_aliases():
                if ' ' + alias + ' ' in norm_text:
                    return int(alias_remap.get(alias, cid))
        return -1

    @classmethod
    def _metadata_from_coverage(cls, coverage_stats, legacy=None):
        """Build authoritative decomposition metadata from coverage_stats."""
        coverage = coverage_stats if isinstance(coverage_stats, dict) else {}
        legacy = legacy or {}
        coverage_decomp_keys = {
            'decomposition_status',
            'has_target',
            'global_only_due_to_parse_error',
            'missing_target',
            'target_generic_reference',
            'overgeneric_target_remaining',
            'target_overgeneric_canonical',
            'generic_target',
            'decomposition_error_flags',
        }
        coverage_has_decomp = any(key in coverage for key in coverage_decomp_keys)

        def _coverage_bool(name, default=False):
            if name in coverage:
                return cls._as_bool(coverage.get(name), default)
            if (
                not coverage_has_decomp
                and name in legacy
                and legacy.get(name) is not None
            ):
                return cls._as_bool(legacy.get(name), default)
            return bool(default)

        legacy_status = legacy.get('decomposition_status', None)
        legacy_global_status = legacy_status == 'global_only_target_unresolved'
        legacy_generic_status = legacy_status == 'weak_generic_target_recovered'
        has_target = _coverage_bool('has_target', not legacy_global_status)
        global_only = (
            (not has_target)
            or _coverage_bool('global_only_due_to_parse_error', False)
            or _coverage_bool('missing_target', False)
            or (legacy_global_status and not coverage_has_decomp)
        )
        target_generic = (
            _coverage_bool('target_generic_reference', False)
            or _coverage_bool('overgeneric_target_remaining', False)
            or _coverage_bool('target_overgeneric_canonical', False)
            or _coverage_bool('generic_target', False)
            or (legacy_generic_status and not coverage_has_decomp)
        )

        flags = coverage.get(
            'decomposition_error_flags',
            legacy.get('decomposition_error_flags', {}) if not coverage_has_decomp else {},
        )
        if not isinstance(flags, dict):
            flags = {}
        flags = dict(flags)
        flags.setdefault('missing_target', int(global_only))
        flags.setdefault('generic_target', int(target_generic))
        error_count = sum(
            int(cls._as_bool(v, False))
            for v in flags.values()
            if isinstance(v, (bool, int, float, str))
        )

        if 'decomposition_status' in coverage:
            status = str(coverage.get('decomposition_status') or 'ok')
            global_only = global_only or status == 'global_only_target_unresolved'
            target_generic = target_generic or status == 'weak_generic_target_recovered'
        elif global_only:
            status = 'global_only_target_unresolved'
        elif target_generic:
            status = 'weak_generic_target_recovered'
        elif error_count > 0:
            status = 'repaired_target_recovered'
        else:
            status = 'ok'
        flags['missing_target'] = int(
            cls._as_bool(flags.get('missing_target', 0), False) or global_only
        )
        flags['generic_target'] = int(
            cls._as_bool(flags.get('generic_target', 0), False) or target_generic
        )
        error_count = sum(
            int(cls._as_bool(v, False))
            for v in flags.values()
            if isinstance(v, (bool, int, float, str))
        )

        conflicts = []
        conflict_flags = {
            'decomposition_status': 0,
            'global_only_due_to_parse_error': 0,
            'target_generic_reference': 0,
        }
        if (
            legacy.get('decomposition_status', None) is not None
            and str(legacy['decomposition_status']) != status
        ):
            conflicts.append('decomposition_status')
            conflict_flags['decomposition_status'] = 1
        if (
            legacy.get('global_only_due_to_parse_error', None) is not None
            and cls._as_bool(
            legacy['global_only_due_to_parse_error'], False
            ) != global_only
        ):
            conflicts.append('global_only_due_to_parse_error')
            conflict_flags['global_only_due_to_parse_error'] = 1
        if (
            legacy.get('target_generic_reference', None) is not None
            and cls._as_bool(
            legacy['target_generic_reference'], False
            ) != target_generic
        ):
            conflicts.append('target_generic_reference')
            conflict_flags['target_generic_reference'] = 1

        parse_confidence = coverage.get(
            'parse_confidence',
            legacy.get('parse_confidence', 1.0),
        )
        try:
            parse_confidence = float(parse_confidence)
        except (TypeError, ValueError):
            parse_confidence = 1.0

        return {
            'coverage_stats': coverage,
            'decomposition_status': status,
            'global_only_due_to_parse_error': global_only,
            'target_generic_reference': target_generic,
            'decomposition_error_flags': flags,
            'decomposition_error_flags_count': error_count,
            'decomp_global_only_mask': global_only,
            'decomp_weak_generic_mask': target_generic,
            'parse_confidence': max(0.0, min(1.0, parse_confidence)),
            'metadata_conflict_count': len(conflicts),
            'metadata_conflict_examples': conflicts[:3],
            'metadata_conflict_flags': conflict_flags,
        }

    @staticmethod
    def _parse_spans_from_csv(line, headers):
        """
        Parse structured spans and metadata from CSV columns.
        Missing columns are allowed for legacy datasets; malformed structured
        JSON in present columns raises an explicit error.
        """
        def _json_column(name, default):
            if name not in headers:
                return default
            try:
                raw = line[headers[name]]
            except IndexError as exc:
                raise ValueError(f"CSV row is missing structured column {name}") from exc
            if raw == '':
                return default
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in structured column {name}: {raw[:120]}") from exc

        entities = _json_column('entities', [])
        attributes = _json_column('attributes', [])
        relations = _json_column('relations', [])
        target_slot = _json_column('target_slot', {})
        coverage_stats = _json_column('coverage_stats', {})
        legacy = {}
        for name in (
            'decomposition_status',
            'global_only_due_to_parse_error',
            'target_generic_reference',
            'parse_confidence',
        ):
            if name in headers:
                try:
                    raw = line[headers[name]]
                except IndexError as exc:
                    raise ValueError(f"CSV row is missing structured column {name}") from exc
                if raw != '':
                    legacy[name] = raw
        if 'decomposition_error_flags' in headers:
            legacy['decomposition_error_flags'] = _json_column(
                'decomposition_error_flags', {}
            )

        entity_spans = [
            {'start': e.get('start', 0), 'end': e.get('end', 0), 'text': e.get('text', '')}
            for e in entities if isinstance(e, dict)
        ]
        attr_spans = [
            {'start': a.get('start', 0), 'end': a.get('end', 0), 'text': a.get('text', '')}
            for a in attributes if isinstance(a, dict)
        ]
        rel_spans = [
            {'start': r.get('start', 0), 'end': r.get('end', 0),
             'text': r.get('text', ''), 'head': r.get('head', ''), 'tail': r.get('tail', '')}
            for r in relations if isinstance(r, dict)
        ]

        meta = Joint3DDataset._metadata_from_coverage(
            coverage_stats,
            legacy=legacy,
        )
        meta.update({
            'target_slot': target_slot,
        })
        return entity_spans, attr_spans, rel_spans, meta

    def load_sr3dplus_annos(self):
        """Load annotations of sr3d/sr3d+."""
        return self.load_sr3d_annos(dset='sr3d+')

    def load_sr3d_annos(self, dset='sr3d'):
        """Load annotations of sr3d/sr3d+."""
        split = self.split
        if split == 'val':
            split = 'test'
        with open('data/meta_data/sr3d_%s_scans.txt' % split) as f:
            scan_ids = set(eval(f.read()))
        with open(self.data_path + 'refer_it_3d/%s.csv' % dset) as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            headers = {header: h for h, header in enumerate(headers)}
            annos = []
            is_spacy = dset.endswith('_spacy')
            for i, line in enumerate(csv_reader):
                mentions_target = (
                    True if is_spacy else
                    str(line[headers['mentions_target_class']]).lower() == 'true'
                )
                if (line[headers['scan_id']] in scan_ids and
                    (is_spacy or mentions_target)):
                    # Parse spans from CSV
                    entity_spans, attr_spans, rel_spans, structured_meta = \
                        self._parse_spans_from_csv(line, headers)
                    annos.append({
                        'scan_id': line[headers['scan_id']],
                        'target_id': int(line[headers['target_id']]),
                        'distractor_ids': eval(line[headers['distractor_ids']]),
                        'utterance': line[headers['utterance']],
                        'target': line[headers['instance_type']],
                        'anchors': eval(line[headers['anchors_types']]),
                        'anchor_obj_ids': eval(line[headers['anchor_ids']]),
                        'dataset': dset,
                        'entity_spans': entity_spans,
                        'attr_spans': attr_spans,
                        'rel_spans': rel_spans,
                        'anchor_ids': self._infer_anchor_span_ids(eval(line[headers['anchors_types']]), entity_spans),
                        **structured_meta,
                    })
        return annos

    def load_nr3d_annos(self, dset='nr3d'):
        """Load annotations of nr3d."""
        split = self.split
        if split == 'val':
            split = 'test'
        with open('data/meta_data/nr3d_%s_scans.txt' % split) as f:
            scan_ids = set(eval(f.read()))
        with open(self.data_path + 'refer_it_3d/%s.csv' % dset) as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            headers = {header: h for h, header in enumerate(headers)}
            annos = []
            is_spacy = dset.endswith('_spacy')
            for i, line in enumerate(csv_reader):
                mentions_target = (
                    True if is_spacy else
                    str(line[headers['mentions_target_class']]).lower() == 'true'
                )
                if (line[headers['scan_id']] in scan_ids and
                    (is_spacy or mentions_target) and
                    (str(line[headers['correct_guess']]).lower() == 'true' or split != 'test')):
                    # Parse spans from CSV
                    entity_spans, attr_spans, rel_spans, structured_meta = \
                        self._parse_spans_from_csv(line, headers)
                    anno = {
                        'scan_id': line[headers['scan_id']],
                        'target_id': int(line[headers['target_id']]),
                        'target': line[headers['instance_type']],
                        'utterance': line[headers['utterance']],
                        'anchor_ids': [],
                        'anchor_obj_ids': [],
                        'anchors': [],
                        'dataset': dset,
                        'entity_spans': entity_spans,
                        'attr_spans': attr_spans,
                        'rel_spans': rel_spans,
                        **structured_meta,
                    }
                    annos.append(anno)
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
        return annos

    @staticmethod
    def _normalize_scanrefer_target(anno):
        target = anno.get('object_name', '')
        return ' '.join(str(target).split('_'))

    @staticmethod
    def _get_scanrefer_utterance(anno, use_spacy=False):
        if use_spacy:
            return anno.get('description', ' '.join(anno.get('tokens', [])))
        return ' '.join(anno['token'])

    @staticmethod
    def _scanrefer_split_name(split):
        return 'val' if split in ('val', 'test') else split

    def _scanrefer_annotation_path(self, split, use_spacy=False, refined=False):
        split = self._scanrefer_split_name(split)
        suffixes = []
        if use_spacy:
            if refined:
                suffixes.append('_%s_spacy_refined.json' % split)
            suffixes.append('_%s_spacy.json' % split)
        else:
            suffixes.append('_%s.json' % split)
        candidates = []
        for suffix in suffixes:
            candidates.extend([
                os.path.join(
                    self.data_path, 'scanrefer', 'ScanRefer_filtered' + suffix
                ),
                os.path.join(
                    self.data_path, 'ScanRefer', 'ScanRefer_filtered' + suffix
                ),
            ])
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def _scanrefer_scan_ids_path(self, split):
        split = self._scanrefer_split_name(split)
        candidates = [
            os.path.join(
                self.data_path, 'scanrefer', 'ScanRefer_filtered_%s.txt' % split
            ),
            os.path.join(
                self.data_path, 'ScanRefer', 'ScanRefer_filtered_%s.txt' % split
            ),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    @staticmethod
    def _scanrefer_key(anno, fallback_ann_id=''):
        return (
            str(anno.get('scene_id', '')),
            str(anno.get('object_id', '')),
            str(anno.get('ann_id', anno.get('ann_id_key', fallback_ann_id))),
        )

    @staticmethod
    def _build_scanrefer_spacy_spans(anno):
        entity_spans = []
        seen_spans = {}

        def _validated_bounds(span):
            if not isinstance(span, dict):
                return None
            start = span.get('start', None)
            end = span.get('end', None)
            if start is None or end is None:
                return None
            try:
                start = int(start)
                end = int(end)
            except (TypeError, ValueError):
                return None
            if start < 0 or end <= start:
                return None
            return start, end

        def _add_entity(span):
            bounds = _validated_bounds(span)
            if bounds is None:
                return None
            start, end = bounds
            key = (start, end, span.get('text', ''))
            if key in seen_spans:
                return seen_spans[key]
            item = {
                'start': start,
                'end': end,
                'text': span.get('text', '')
            }
            seen_spans[key] = len(entity_spans)
            entity_spans.append(item)
            return seen_spans[key]

        def _collect_spans(spans, include_links=False):
            collected = []
            for span in spans:
                bounds = _validated_bounds(span)
                if bounds is None:
                    continue
                start, end = bounds
                item = {
                    'start': start,
                    'end': end,
                    'text': span.get('text', '')
                }
                if include_links:
                    item['head'] = span.get('head', '')
                    item['tail'] = span.get('tail', '')
                collected.append(item)
            return collected

        _add_entity(anno.get('target_slot'))
        for span in anno.get('entities', []):
            _add_entity(span)

        attr_slot = anno.get('attr_slot', {})
        attr_spans = _collect_spans(attr_slot.get('items', []))

        rel_spans = _collect_spans(anno.get('rel_slots', []), include_links=True)

        anchor_ids = []
        for anchor in anno.get('anchor_slots', []):
            anchor_idx = _add_entity(anchor)
            anchor_ids.append(anchor_idx if anchor_idx is not None else -1)

        if len(anchor_ids) < len(rel_spans):
            anchor_ids.extend([-1] * (len(rel_spans) - len(anchor_ids)))
        elif len(anchor_ids) > len(rel_spans):
            anchor_ids = anchor_ids[:len(rel_spans)]

        return entity_spans, attr_spans, rel_spans, anchor_ids

    def _load_scanrefer_spacy_lookup(self, split):
        path = self._scanrefer_annotation_path(
            split, use_spacy=True, refined=True
        )
        with open(path) as f:
            reader = json.load(f)
        lookup = {}
        for i, anno in enumerate(reader):
            lookup[self._scanrefer_key(anno, fallback_ann_id=str(i))] = anno
        return lookup

    @staticmethod
    def _scanrefer_spacy_inject_fields(raw_anno, spacy_anno):
        if spacy_anno is None:
            return raw_anno
        merged = dict(raw_anno)
        for field in (
            'target_slot',
            'entities',
            'attributes',
            'attr_slot',
            'rel_slots',
            'anchor_slots',
            'slot_mask',
            'coverage_stats',
            'parse_confidence',
            'decomp_global_only_mask',
            'decomp_weak_generic_mask',
            'decomp_train_global_only_mask',
            'decomp_train_weak_generic_mask',
            'decomp_train_global_only_reason',
            'decomposition_status',
            'global_only_due_to_parse_error',
            'target_generic_reference',
            'decomposition_error_flags',
            'decomposition_error_flags_count',
            'manual_review_fix_reason',
            'manual_review_fix_applied',
        ):
            if field in spacy_anno:
                merged[field] = spacy_anno[field]
        if 'description' not in merged and 'description' in spacy_anno:
            merged['description'] = spacy_anno['description']
        if 'tokens' in spacy_anno:
            merged['tokens'] = spacy_anno['tokens']
        return merged

    def load_scanrefer_annos(self, dset='scanrefer'):
        """Load annotations of ScanRefer."""
        use_spacy = dset == 'scanrefer_spacy'
        inject_spacy = (
            dset == 'scanrefer'
            and getattr(self, 'scanrefer_inject_spacy_decomp', False)
        )
        split = self._scanrefer_split_name(self.split)
        with open(self._scanrefer_scan_ids_path(split)) as f:
            scan_ids = [line.rstrip().strip('\n') for line in f.readlines()]
        anno_path = self._scanrefer_annotation_path(
            split, use_spacy=use_spacy, refined=use_spacy
        )
        with open(anno_path) as f:
            reader = json.load(f)
        spacy_lookup = self._load_scanrefer_spacy_lookup(split) if inject_spacy else {}
        annos = []
        for i, anno in enumerate(reader):
            if anno['scene_id'] not in scan_ids:
                continue
            spacy_hit = False
            if inject_spacy:
                spacy_anno = spacy_lookup.get(
                    self._scanrefer_key(anno, fallback_ann_id=str(i))
                )
                spacy_hit = spacy_anno is not None
                anno = self._scanrefer_spacy_inject_fields(anno, spacy_anno)
            entity_spans = []
            attr_spans = []
            rel_spans = []
            anchor_ids = []
            target_slot = {}
            if use_spacy or inject_spacy:
                raw_target_slot = anno.get('target_slot', {})
                if isinstance(raw_target_slot, dict):
                    target_slot = raw_target_slot
                entity_spans, attr_spans, rel_spans, anchor_ids = \
                    self._build_scanrefer_spacy_spans(anno)
            coverage_stats = anno.get('coverage_stats', {})
            if not isinstance(coverage_stats, dict):
                coverage_stats = {}
            structured_meta = self._metadata_from_coverage(
                coverage_stats,
                legacy={
                    'decomposition_status': anno.get('decomposition_status', None),
                    'global_only_due_to_parse_error': anno.get(
                        'global_only_due_to_parse_error', None
                    ),
                    'target_generic_reference': anno.get(
                        'target_generic_reference', None
                    ),
                    'decomposition_error_flags': anno.get(
                        'decomposition_error_flags', {}
                    ),
                    'parse_confidence': anno.get('parse_confidence', 1.0),
                },
            )
            sample = {
                'scan_id': anno['scene_id'],
                'target_id': int(anno['object_id']),
                'distractor_ids': [],
                'utterance': self._get_scanrefer_utterance(anno, use_spacy=use_spacy),
                'target': self._normalize_scanrefer_target(anno),
                'anchors': [],
                'anchor_ids': anchor_ids,
                'anchor_obj_ids': [],
                'dataset': dset,
                'target_slot': target_slot,
                'entity_spans': entity_spans,
                'attr_spans': attr_spans,
                'rel_spans': rel_spans,
                'description': anno.get('description', ''),
                'scene_id': anno.get('scene_id', anno['scene_id']),
                'object_id': anno.get('object_id', anno['object_id']),
                'object_name': anno.get('object_name', ''),
                'ann_id': anno.get('ann_id', anno.get('ann_id_key', str(i))),
                'attr_slot': anno.get('attr_slot', {}),
                'rel_slots': anno.get('rel_slots', []),
                'anchor_slots': anno.get('anchor_slots', []),
                'slot_mask': anno.get('slot_mask', []),
                'decomp_train_global_only_mask': anno.get(
                    'decomp_train_global_only_mask', False
                ),
                'decomp_train_weak_generic_mask': anno.get(
                    'decomp_train_weak_generic_mask', False
                ),
                'decomp_train_global_only_reason': anno.get(
                    'decomp_train_global_only_reason', ''
                ),
                'dbg_data_scanrefer_spacy_injected_ratio': float(inject_spacy),
                'dbg_data_scanrefer_spacy_inject_hit_ratio': float(
                    inject_spacy and spacy_hit
                ),
                'dbg_data_scanrefer_spacy_inject_miss_ratio': float(
                    inject_spacy and not spacy_hit
                ),
                **structured_meta,
            }
            annos.append(sample)

        # Add distractor info
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
        # Add unique-multi
        for anno in annos:
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

    def load_scannet_annos(self):
        """Load annotations of scannet."""
        split = 'train' if self.split == 'train' else 'val'
        with open('data/meta_data/scannetv2_%s.txt' % split) as f:
            scan_ids = [line.rstrip() for line in f]
        annos = []
        for scan_id in scan_ids:
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
        else:  # fixed detection sentence
            ret = [
                'cabinet', 'bed', 'chair', 'couch', 'table', 'door',
                'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
                'other furniture'
            ]
        return ret

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

    def _augment(self, pc, color, rotate):
        augmentations = {}
        no_direction_rotation = rotate == "none"
        no_direction_stable_rotation = rotate == "none_stable"
        yaw_only_rotation = rotate == "yaw_only"
        yaw_stable_rotation = rotate == "yaw_stable"
        small_yaw_rotation = rotate == "small_yaw"
        stable_jitter = yaw_stable_rotation or no_direction_stable_rotation

        # Rotate/flip only if we don't have a view_dep sentence
        if rotate is True or yaw_only_rotation or yaw_stable_rotation:
            theta_z = 90 * np.random.randint(0, 4) + 10 * np.random.rand() - 5
            # Flipping along the YZ plane
            augmentations['yz_flip'] = np.random.random() > 0.5
            if augmentations['yz_flip']:
                pc[:, 0] = -pc[:, 0]
            # Flipping along the XZ plane
            augmentations['xz_flip'] = np.random.random() > 0.5
            if augmentations['xz_flip']:
                pc[:, 1] = -pc[:, 1]
        elif small_yaw_rotation:
            theta_z = (2 * np.random.rand() - 1) * 3
            augmentations['yz_flip'] = False
            augmentations['xz_flip'] = False
        elif no_direction_rotation or no_direction_stable_rotation:
            theta_z = 0.0
            augmentations['yz_flip'] = False
            augmentations['xz_flip'] = False
        else:
            theta_z = (2 * np.random.rand() - 1) * 5
            augmentations['yz_flip'] = False
            augmentations['xz_flip'] = False
        augmentations['theta_z'] = theta_z
        pc[:, :3] = rot_z(pc[:, :3], theta_z)
        # Rotate around x
        theta_x = (
            0.0
            if (
                no_direction_rotation
                or no_direction_stable_rotation
                or yaw_only_rotation
                or yaw_stable_rotation
                or small_yaw_rotation
            )
            else (2 * np.random.rand() - 1) * 2.5
        )
        augmentations['theta_x'] = theta_x
        pc[:, :3] = rot_x(pc[:, :3], theta_x)
        # Rotate around y
        theta_y = (
            0.0
            if (
                no_direction_rotation
                or no_direction_stable_rotation
                or yaw_only_rotation
                or yaw_stable_rotation
                or small_yaw_rotation
            )
            else (2 * np.random.rand() - 1) * 2.5
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
            augmentations['scale'] = 0.98 + 0.04 * np.random.random()
        pc[:, :3] *= augmentations['scale']

        # Color
        if color is not None and not stable_jitter:
            color += self.mean_rgb
            color_scale = 0.98 + 0.04 * np.random.random((len(color), 3))
            augmentations['color_scale_delta_abs_mean'] = float(
                np.abs(color_scale - 1.0).mean()
            )
            color *= color_scale
            color -= self.mean_rgb
        else:
            augmentations['color_scale_delta_abs_mean'] = 0.0
        return pc, color, augmentations

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
        if self._train_augmentation_enabled():
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

    def _get_token_positive_map(self, anno):
        """Return correspondence of boxes to tokens.

        Uses explicit spans and deterministic string matching only.
        Missing target alignment stays empty; no pseudo fallback is fabricated.
        """
        # Token start-end span in characters
        caption = ' '.join(anno['utterance'].replace(',', ' ,').split())
        caption_with_spaces = ' ' + caption + ' '
        tokens_positive = np.zeros((MAX_NUM_OBJ, 2))

        if isinstance(anno['target'], list):
            cat_names = anno['target']
        else:
            cat_names = [anno['target']]
        if self.detect_intermediate:
            cat_names += anno['anchors']

        entity_spans = anno.get('entity_spans', [])
        target_slot = anno.get('target_slot', {})
        global_only_target = bool(
            anno.get('global_only_due_to_parse_error', False)
            or anno.get('decomp_global_only_mask', False)
            or anno.get('decomposition_status', '') == 'global_only_target_unresolved'
        )
        target_source = 'missing'

        for c, cat_name in enumerate(cat_names):
            found = False
            source = 'missing'
            if c == 0 and global_only_target:
                continue

            candidate_spans = []
            if c == 0 and isinstance(target_slot, dict):
                candidate_spans.append(('explicit_target_slot', target_slot))
            candidate_spans.extend([
                ('entity_exact_match', span) for span in entity_spans
            ])
            cat_name_normalized = str(cat_name).lower().strip()
            cat_name_singular = cat_name_normalized.rstrip('s')
            for span_source, span in candidate_spans:
                if not isinstance(span, dict):
                    continue
                span_text = span.get('text', '').lower().strip()
                span_text_singular = span_text.rstrip('s')
                exact_match = (
                    span_text == cat_name_normalized
                    or span_text_singular == cat_name_singular
                )
                if span_source == 'explicit_target_slot' or exact_match:
                    start_span = span.get('start', -1)
                    end_span = span.get('end', -1)
                    if start_span >= 0 and end_span > start_span:
                        tokens_positive[c][0] = start_span
                        tokens_positive[c][1] = end_span
                        found = True
                        source = span_source
                        break

            if not found:
                start_span = caption_with_spaces.find(' ' + cat_name + ' ')
                len_ = len(cat_name)
                if start_span < 0:
                    start_span = caption_with_spaces.find(' ' + cat_name)
                    len_ = len(caption_with_spaces[start_span + 1:].split()[0]) if start_span >= 0 else 0
                if start_span < 0:
                    start_span = caption_with_spaces.find(cat_name)
                    if start_span > 0:
                        orig_start_span = start_span
                        while start_span > 0 and caption_with_spaces[start_span - 1] != ' ':
                            start_span -= 1
                        len_ = len(cat_name) + orig_start_span - start_span
                        while len_ + start_span < len(caption_with_spaces) and caption_with_spaces[len_ + start_span] != ' ':
                            len_ += 1

                if start_span >= 0:
                    end_span = start_span + len_
                    tokens_positive[c][0] = start_span
                    tokens_positive[c][1] = end_span
                    source = 'lexical_exact_match'
                    found = True
            if c == 0:
                target_source = source if found else 'missing'

        # Positive map (for soft token prediction)
        tokenized = self.tokenizer.batch_encode_plus(
            [' '.join(anno['utterance'].replace(',', ' ,').split())],
            padding="longest", return_tensors="pt"
        )
        positive_map = np.zeros((MAX_NUM_OBJ, 256))
        gt_map = get_positive_map(tokenized, tokens_positive[:len(cat_names)])
        positive_map[:len(cat_names)] = gt_map
        if positive_map[0].sum() <= 0:
            target_source = 'missing'
        source_flags = {
            'explicit_target_slot': float(target_source == 'explicit_target_slot'),
            'entity_exact_match': float(target_source == 'entity_exact_match'),
            'lexical_exact_match': float(target_source == 'lexical_exact_match'),
            'missing': float(target_source == 'missing'),
        }
        diagnostics = {
            'positive_map_target_missing': float(positive_map[0].sum() <= 0),
            'positive_map_fallback_used': 0.0,
            'positive_map_global_only_target_empty': float(
                global_only_target and positive_map[0].sum() <= 0
            ),
            'dbg_warn_global_only_target_positive_map': float(
                global_only_target and positive_map[0].sum() > 0
            ),
            'positive_map_source_explicit_target_slot': source_flags['explicit_target_slot'],
            'positive_map_source_entity_exact_match': source_flags['entity_exact_match'],
            'positive_map_source_lexical_exact_match': source_flags['lexical_exact_match'],
            'positive_map_source_missing': source_flags['missing'],
        }
        return tokens_positive, positive_map, diagnostics

    def _get_target_boxes(self, anno, scan):
        """Return gt boxes to detect."""
        bboxes = np.zeros((MAX_NUM_OBJ, 6))
        if isinstance(anno['target_id'], list):  # scannet
            tids = anno['target_id']
        else:  # referit dataset
            tids = [anno['target_id']]
            if self.detect_intermediate:
                tids += anno.get('anchor_obj_ids', [])
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
            self._train_augmentation_enabled()
            and not getattr(self, 'disable_box_jitter', False)
        ):  # jitter boxes
            bboxes[:len(tids)] *= 0.95 + 0.1 * np.random.random((len(tids), 6))
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
        ])[:MAX_NUM_OBJ]
        keep = np.array([False] * MAX_NUM_OBJ)
        keep[:len(keep_)] = True  # keep_

        # Class ids
        cid = np.array([
            DC.nyu40id2class[self.label_map[scan.get_object_instance_label(k)]]
            if keep_[k] else 325  # this is the 'object' class
            for k, kept in enumerate(keep) if kept
        ])
        class_ids = np.zeros((MAX_NUM_OBJ,))
        class_ids[keep] = cid

        # Object boxes
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
            self._train_augmentation_enabled()
            and not getattr(self, 'disable_box_jitter', False)
        ):
            all_bboxes *= 0.95 + 0.1 * np.random.random((len(all_bboxes), 6))

        # Which boxes we're interested for
        all_bbox_label_mask = keep
        return class_ids, all_bboxes, all_bbox_label_mask

    def _get_detected_objects(self, split, scan_id, augmentations):
        # Initialize
        all_detected_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        all_detected_bbox_label_mask = np.array([False] * MAX_NUM_OBJ)
        detected_class_ids = np.zeros((MAX_NUM_OBJ,))
        detected_logits = np.zeros((MAX_NUM_OBJ, NUM_CLASSES))

        # Load
        detected_dict = np.load(
            f'{self.data_path}group_free_pred_bboxes/group_free_pred_bboxes_{split}/{scan_id}.npy',
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
        detected_logits[:num_objs] = detected_dict['logits']
        # Match current augmentations
        valid_detected = all_detected_bbox_label_mask.astype(bool)
        valid_count = int(valid_detected.sum())
        augment_det_diag = {
            'active': float(
                getattr(self, 'augment_det', False)
                and self._train_augmentation_enabled()
            ),
            'applied': 0.0,
            'valid_obj_count': valid_count,
            'corrupt_obj_count': 0,
            'corrupt_obj_ratio': 0.0,
        }
        if self._train_augmentation_enabled() and valid_detected.any():
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
        if (
            getattr(self, 'augment_det', False)
            and self._train_augmentation_enabled()
            and valid_detected.any()
        ):
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
            corrupt_count = int(corrupt.sum())
            augment_det_diag.update({
                'applied': 1.0,
                'corrupt_obj_count': corrupt_count,
                'corrupt_obj_ratio': corrupt_count / max(valid_count, 1),
            })
            all_detected_bboxes[corrupt_indices] = rand_box[corrupt]
            detected_class_ids[corrupt_indices] = np.random.randint(
                0, len(DC.nyu40ids), len(corrupt_indices)
            )
        return (
            all_detected_bboxes, all_detected_bbox_label_mask,
            detected_class_ids, detected_logits, augment_det_diag
        )

    def __getitem__(self, index):
        """Get current batch for input index."""
        split = self.split

        # Read annotation
        anno = self.annos[index]
        scan = self.scans[anno['scan_id']]
        scan.pc = np.copy(scan.orig_pc)

        # Populate anno (used only for scannet)
        self.random_utt = False
        if anno['dataset'] == 'scannet':
            self.random_utt = self.joint_det and np.random.random() > 0.5
            sampled_classes = self._sample_classes(anno['scan_id'])
            utterance = self._create_scannet_utterance(sampled_classes)
            # Target ids
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

        # Point cloud representation
        point_cloud, augmentations, og_color = self._get_pc(anno, scan)

        # "Target" boxes: append anchors if they're to be detected
        gt_bboxes, box_label_mask, point_instance_label = \
            self._get_target_boxes(anno, scan)

        # Positive map for soft-token and contrastive losses
        _, positive_map, positive_map_diag = self._get_token_positive_map(anno)

        # Scene gt boxes
        (
            class_ids, all_bboxes, all_bbox_label_mask
        ) = self._get_scene_objects(scan)

        # Detected boxes
        detected_result = self._get_detected_objects(
            split, anno['scan_id'], augmentations
        )
        if len(detected_result) == 5:
            (
                all_detected_bboxes, all_detected_bbox_label_mask,
                detected_class_ids, detected_logits, augment_det_diag
            ) = detected_result
        else:
            (
                all_detected_bboxes, all_detected_bbox_label_mask,
                detected_class_ids, detected_logits
            ) = detected_result
            augment_det_diag = {
                'active': float(
                    getattr(self, 'augment_det', False)
                    and self._train_augmentation_enabled()
                ),
                'applied': 0.0,
                'valid_obj_count': int(
                    np.asarray(all_detected_bbox_label_mask).astype(bool).sum()
                ),
                'corrupt_obj_count': 0,
                'corrupt_obj_ratio': 0.0,
            }

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
            # detected_class_ids[all_bbox_label_mask] = classes[classes > -1]
            classes[classes == -1] = 325  # 'object' class
            _k = all_bbox_label_mask.sum()
            detected_class_ids[:_k] = classes[:_k]

        augment_det_effective = bool(
            float(augment_det_diag.get('applied', 0.0)) > 0.0
            and not self.butd_gt
            and not self.butd_cls
        )
        augment_det_corrupt_count = int(
            augment_det_diag.get('corrupt_obj_count', 0)
        )
        augment_det_corrupt_ratio = (
            float(augment_det_diag.get('corrupt_obj_ratio', 0.0))
            if augment_det_effective else 0.0
        )
        augmentation_diag = self._data_augmentation_diagnostics(
            anno, augmentations, augment_det_effective,
            augment_det_corrupt_count
        )

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
        decomp_global_only_mask = self._decomp_global_only_mask_for_sample(anno)
        decomp_weak_generic_mask = self._decomp_weak_generic_mask_for_sample(anno)
        target_cid = (
            class_ids[anno['target_id']]
            if isinstance(anno['target_id'], int)
            else class_ids[anno['target_id'][0]]
        )
        text_target_cid = self._text_target_cid_from_annotation(
            anno,
            alias_policy=getattr(self, 'text_target_alias_policy', 'strict'),
        )
        text_target_valid = int(text_target_cid) >= 0

        ret_dict = {
            'box_label_mask': box_label_mask.astype(np.float32),
            'center_label': gt_bboxes[:, :3].astype(np.float32),
            'sem_cls_label': _labels.astype(np.int64),
            'size_gts': gt_bboxes[:, 3:].astype(np.float32),
        }
        ret_dict.update({
            "scan_ids": anno['scan_id'],
            "dataset": anno.get('dataset', ''),
            "point_clouds": point_cloud.astype(np.float32),
            "utterances": (
                ' '.join(anno['utterance'].replace(',', ' ,').split())
                + ' . not mentioned'
            ),
            "positive_map": positive_map.astype(np.float32),
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
                anno.get('anchor_ids', [])
                + [-1] * (32 - len(anno['anchor_ids']))
            ).astype(int),
            "all_detected_boxes": all_detected_bboxes.astype(np.float32),
            "all_detected_bbox_label_mask": all_detected_bbox_label_mask.astype(np.bool8),
            "all_detected_class_ids": detected_class_ids.astype(np.int64),
            "all_detected_logits": detected_logits.astype(np.float32),
            "is_view_dep": self._is_view_dep(anno['utterance']),
            "is_hard": len(anno['distractor_ids']) > 1,
            "is_unique": len(anno['distractor_ids']) == 0,
            "target_cid": target_cid,
            "text_target_cid": np.int64(text_target_cid),
            # Add spans for structured slot supervision.
            "target_slot": anno.get('target_slot', []),
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
            "slot_mask": anno.get('slot_mask', []),
            "parse_confidence": anno.get('parse_confidence', 1.0),
            "coverage_stats": anno.get('coverage_stats', {}),
            "spacy_rotation_mode_id": np.int64(
                self._spacy_rotation_mode_id_for_sample(anno)
            ),
            "spacy_augmentation_profile_id": np.int64(
                self._spacy_augmentation_profile_id(anno)
            ),
            "decomposition_status": anno.get('decomposition_status', 'ok'),
            "decomp_global_only_mask": decomp_global_only_mask,
            "decomp_weak_generic_mask": decomp_weak_generic_mask,
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
            "global_only_due_to_parse_error": anno.get(
                'global_only_due_to_parse_error', False
            ),
            "target_generic_reference": anno.get('target_generic_reference', False),
            "decomposition_error_flags": anno.get('decomposition_error_flags', {}),
            "metadata_conflict_examples": anno.get('metadata_conflict_examples', []),
            "metadata_conflict_ratio": float(
                anno.get('metadata_conflict_count', 0) > 0
            ),
            "metadata_conflict_decomposition_status_ratio": float(
                anno.get('metadata_conflict_flags', {}).get(
                    'decomposition_status', 0
                )
            ),
            "metadata_conflict_global_only_due_to_parse_error_ratio": float(
                anno.get('metadata_conflict_flags', {}).get(
                    'global_only_due_to_parse_error', 0
                )
            ),
            "metadata_conflict_target_generic_reference_ratio": float(
                anno.get('metadata_conflict_flags', {}).get(
                    'target_generic_reference', 0
                )
            ),
            "positive_map_target_missing_ratio": positive_map_diag[
                'positive_map_target_missing'
            ],
            "positive_map_fallback_used_ratio": positive_map_diag[
                'positive_map_fallback_used'
            ],
            "positive_map_global_only_target_empty_ratio": positive_map_diag[
                'positive_map_global_only_target_empty'
            ],
            "positive_map_source_explicit_target_slot_ratio": positive_map_diag[
                'positive_map_source_explicit_target_slot'
            ],
            "positive_map_source_entity_exact_match_ratio": positive_map_diag[
                'positive_map_source_entity_exact_match'
            ],
            "positive_map_source_lexical_exact_match_ratio": positive_map_diag[
                'positive_map_source_lexical_exact_match'
            ],
            "positive_map_source_missing_ratio": positive_map_diag[
                'positive_map_source_missing'
            ],
            "dbg_data_decomp_ok_ratio": float(
                anno.get('decomposition_status', 'ok') == 'ok'
            ),
            "dbg_data_decomp_repaired_ratio": float(
                str(anno.get('decomposition_status', '')).startswith('repaired')
            ),
            "dbg_data_decomp_weak_generic_ratio": float(
                anno.get('decomposition_status', '') == 'weak_generic_target_recovered'
            ),
            "dbg_data_decomp_global_only_ratio": float(
                anno.get('decomposition_status', '') == 'global_only_target_unresolved'
            ),
            "dbg_data_decomp_global_only_effective_ratio": float(
                decomp_global_only_mask
            ),
            "dbg_data_decomp_weak_generic_effective_ratio": float(
                decomp_weak_generic_mask
            ),
            "dbg_data_augment_det_active_ratio": float(
                augment_det_diag.get('active', 0.0)
            ),
            "dbg_data_augment_det_effective_ratio": float(
                augment_det_effective
            ),
            "dbg_data_augment_det_corrupt_any_ratio": float(
                augment_det_effective and augment_det_corrupt_count > 0
            ),
            "dbg_data_augment_det_corrupt_obj_ratio": (
                augment_det_corrupt_ratio
            ),
            "dbg_data_scanrefer_spacy_injected_ratio": float(
                anno.get('dbg_data_scanrefer_spacy_injected_ratio', 0.0)
            ),
            "dbg_data_scanrefer_spacy_inject_hit_ratio": float(
                anno.get('dbg_data_scanrefer_spacy_inject_hit_ratio', 0.0)
            ),
            "dbg_data_scanrefer_spacy_inject_miss_ratio": float(
                anno.get('dbg_data_scanrefer_spacy_inject_miss_ratio', 0.0)
            ),
            "dbg_data_text_target_cid_valid_ratio": float(text_target_valid),
            "dbg_data_text_target_cid_matches_gt_ratio": float(
                text_target_valid and int(text_target_cid) == int(target_cid)
            ),
            "dbg_positive_map_explicit_target_slot_ratio": positive_map_diag[
                'positive_map_source_explicit_target_slot'
            ],
            "dbg_positive_map_entity_exact_match_ratio": positive_map_diag[
                'positive_map_source_entity_exact_match'
            ],
            "dbg_positive_map_lexical_exact_match_ratio": positive_map_diag[
                'positive_map_source_lexical_exact_match'
            ],
            "dbg_positive_map_fallback_used_ratio": positive_map_diag[
                'positive_map_fallback_used'
            ],
            "dbg_positive_map_missing_ratio": positive_map_diag[
                'positive_map_source_missing'
            ],
            "dbg_positive_map_global_only_target_empty_ratio": positive_map_diag[
                'positive_map_global_only_target_empty'
            ],
            "dbg_warn_global_only_target_positive_map_ratio": positive_map_diag[
                'dbg_warn_global_only_target_positive_map'
            ],
            "dbg_metadata_conflict_decomposition_status_ratio": float(
                anno.get('metadata_conflict_flags', {}).get(
                    'decomposition_status', 0
                )
            ),
            "dbg_metadata_conflict_global_only_due_to_parse_error_ratio": float(
                anno.get('metadata_conflict_flags', {}).get(
                    'global_only_due_to_parse_error', 0
                )
            ),
            "dbg_metadata_conflict_target_generic_reference_ratio": float(
                anno.get('metadata_conflict_flags', {}).get(
                    'target_generic_reference', 0
                )
            ),
            "dbg_loader_sample_kept_ratio": 1.0,
        })
        ret_dict.update(augmentation_diag)
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
            'leftmost', 'rightmost', 'looking', 'across',
        )
        pattern = r'\b(?:' + '|'.join(re.escape(rel) for rel in rels) + r')\b'
        return re.search(
            pattern,
            str(utterance).lower().replace('_', ' '),
        ) is not None

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
            'end', 'entrance', 'room',
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
                'head', 'tail', 'relation', 'rel', 'lemma_head',
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

    @staticmethod
    def _augmentation_shift_l2(augmentations):
        shift = augmentations.get('shift', 0.0)
        try:
            return float(np.linalg.norm(np.asarray(shift, dtype=np.float32)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _augmentation_noise_abs_mean(augmentations):
        noise = augmentations.get('noise', 0.0)
        try:
            return float(np.abs(np.asarray(noise, dtype=np.float32)).mean())
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _has_spacy_relation_slots(cls, anno):
        rel_slots = anno.get('rel_slots', [])
        if not isinstance(rel_slots, list):
            rel_slots = [rel_slots]
        return any(cls._spacy_relation_text(slot).strip() for slot in rel_slots)

    def _data_augmentation_diagnostics(
        self, anno, augmentations, augment_det_effective,
        augment_det_corrupt_count
    ):
        dataset_name = str(anno.get('dataset', ''))
        is_spacy = dataset_name.endswith('_spacy')
        has_relation = self._has_spacy_relation_slots(anno)
        is_relation_free = is_spacy and not has_relation
        has_view_word = (
            is_spacy and self._spacy_has_raw_view_word(anno.get('utterance', ''))
        )
        has_compass_word = (
            is_spacy
            and self._spacy_has_compass_direction_word(anno.get('utterance', ''))
        )
        direction_sensitive = bool(
            is_spacy and (
                has_relation
                or has_view_word
                or has_compass_word
                or self._spacy_rotation_mode(anno) == 'none'
            )
        )

        theta_z = float(augmentations.get('theta_z', 0.0))
        theta_x = float(augmentations.get('theta_x', 0.0))
        theta_y = float(augmentations.get('theta_y', 0.0))
        has_flip = bool(
            augmentations.get('yz_flip', False)
            or augmentations.get('xz_flip', False)
        )
        scale_delta = abs(float(augmentations.get('scale', 1.0)) - 1.0)
        shift_l2 = self._augmentation_shift_l2(augmentations)
        noise_abs_mean = self._augmentation_noise_abs_mean(augmentations)
        color_delta_abs_mean = abs(
            float(augmentations.get('color_scale_delta_abs_mean', 0.0))
        )
        pc_aug_active = bool(augmentations)
        yaw_aug = bool(pc_aug_active and abs(theta_z) > 1e-6)
        pitch_roll_aug = bool(
            pc_aug_active and (abs(theta_x) > 1e-6 or abs(theta_y) > 1e-6)
        )
        shift_aug = bool(pc_aug_active and shift_l2 > 1e-6)
        scale_aug = bool(pc_aug_active and scale_delta > 1e-6)
        noise_aug = bool(pc_aug_active and noise_abs_mean > 1e-9)
        color_aug = bool(pc_aug_active and color_delta_abs_mean > 1e-9)
        rigid_aug = bool(
            yaw_aug
            or pitch_roll_aug
            or has_flip
        )
        jitter_aug = bool(
            shift_aug
            or scale_aug
            or noise_aug
        )
        box_jitter_active = bool(
            self._train_augmentation_enabled()
            and not getattr(self, 'disable_box_jitter', False)
        )

        return {
            "dbg_data_aug_pc_active_ratio": float(pc_aug_active),
            "dbg_data_aug_yaw_abs_deg": abs(theta_z),
            "dbg_data_aug_pitch_abs_deg": abs(theta_x),
            "dbg_data_aug_roll_abs_deg": abs(theta_y),
            "dbg_data_aug_flip_any_ratio": float(has_flip),
            "dbg_data_aug_shift_l2": shift_l2,
            "dbg_data_aug_scale_delta_abs": scale_delta,
            "dbg_data_aug_noise_abs_mean": noise_abs_mean,
            "dbg_data_aug_color_delta_abs_mean": color_delta_abs_mean,
            "dbg_data_aug_yaw_active_ratio": float(yaw_aug),
            "dbg_data_aug_pitch_roll_active_ratio": float(pitch_roll_aug),
            "dbg_data_aug_shift_active_ratio": float(shift_aug),
            "dbg_data_aug_scale_active_ratio": float(scale_aug),
            "dbg_data_aug_noise_active_ratio": float(noise_aug),
            "dbg_data_aug_color_active_ratio": float(color_aug),
            "dbg_data_aug_rigid_any_ratio": float(rigid_aug),
            "dbg_data_aug_jitter_any_ratio": float(jitter_aug),
            "dbg_data_aug_box_jitter_active_ratio": float(box_jitter_active),
            "dbg_data_spacy_relation_slot_ratio": float(is_spacy and has_relation),
            "dbg_data_spacy_relation_free_ratio": float(is_relation_free),
            "dbg_data_spacy_view_word_ratio": float(has_view_word),
            "dbg_data_spacy_compass_word_ratio": float(has_compass_word),
            "dbg_data_spacy_direction_sensitive_ratio": float(
                direction_sensitive
            ),
            "dbg_data_spacy_direction_sensitive_rigid_aug_ratio": float(
                direction_sensitive and rigid_aug
            ),
            "dbg_data_spacy_direction_sensitive_yaw_aug_ratio": float(
                direction_sensitive and yaw_aug
            ),
            "dbg_data_spacy_direction_sensitive_pitch_roll_aug_ratio": float(
                direction_sensitive and pitch_roll_aug
            ),
            "dbg_data_spacy_direction_sensitive_jitter_aug_ratio": float(
                direction_sensitive and jitter_aug
            ),
            "dbg_data_spacy_direction_sensitive_shift_aug_ratio": float(
                direction_sensitive and shift_aug
            ),
            "dbg_data_spacy_direction_sensitive_scale_aug_ratio": float(
                direction_sensitive and scale_aug
            ),
            "dbg_data_spacy_direction_sensitive_noise_aug_ratio": float(
                direction_sensitive and noise_aug
            ),
            "dbg_data_spacy_direction_sensitive_color_aug_ratio": float(
                direction_sensitive and color_aug
            ),
            "dbg_data_spacy_direction_sensitive_box_jitter_ratio": float(
                direction_sensitive and box_jitter_active
            ),
            "dbg_data_spacy_direction_sensitive_det_corrupt_ratio": float(
                direction_sensitive
                and augment_det_effective
                and augment_det_corrupt_count > 0
            ),
        }

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
            getattr(self, 'spacy_relation_free_rawview_global_only_train', False)
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
        is_spacy = str(anno.get('dataset', '')).endswith('_spacy')
        if (
            is_spacy
            and getattr(self, 'spacy_direction_sensitive_no_jitter_aug', False)
        ):
            has_direction_word = (
                self._spacy_has_raw_view_word(anno.get('utterance', ''))
                or self._spacy_has_compass_direction_word(anno.get('utterance', ''))
            )
            if mode == "none":
                return "none_stable"
            if mode == "yaw_only" and (
                self._has_spacy_relation_slots(anno) or has_direction_word
            ):
                return "yaw_stable"
        if (
            mode is None
            and is_spacy
            and getattr(self, 'spacy_relation_free_yaw_only_aug', False)
        ):
            if (
                getattr(self, 'spacy_relation_free_view_guard_aug', False)
                and self._spacy_has_raw_view_word(anno.get('utterance', ''))
            ):
                return (
                    "none_stable"
                    if getattr(self, 'spacy_direction_sensitive_no_jitter_aug', False)
                    else "none"
                )
            if (
                getattr(self, 'spacy_relation_free_view_small_yaw_aug', False)
                and self._spacy_has_raw_view_word(anno.get('utterance', ''))
            ):
                return "small_yaw"
            if (
                getattr(self, 'spacy_relation_free_compass_guard_aug', False)
                and self._spacy_has_compass_direction_word(
                    anno.get('utterance', '')
                )
            ):
                return (
                    "none_stable"
                    if getattr(self, 'spacy_direction_sensitive_no_jitter_aug', False)
                    else "none"
                )
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
        if mode in ("none", "none_stable"):
            return 1
        if mode in ("yaw_only", "yaw_stable", "small_yaw"):
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
            if getattr(self, 'spacy_direction_sensitive_no_jitter_aug', False):
                sample_mode = self._spacy_rotation_mode_for_sample(anno)
                if sample_mode in ("none_stable", "yaw_stable"):
                    return 11
            if (
                getattr(self, 'spacy_relation_free_rawview_global_only_train', False)
                and self._spacy_relation_free_rawview_parse_noise(anno)
            ):
                return 8
            if (
                getattr(self, 'spacy_relation_free_compass_guard_aug', False)
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


def get_positive_map(tokenized, tokens_positive):
    """Construct a map of box-token associations."""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list
        beg = int(beg)
        end = int(end)
        if end <= beg:
            continue
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


def scannet_loader(iter_obj):
    """Load the scans in memory, helper function."""
    scan_id, scan_path = iter_obj
    print(scan_id)
    return Scan(scan_id, scan_path, True)


def save_data(filename, split, data_path):
    """Save all scans to pickle."""
    import multiprocessing as mp

    # Read all scan files
    scan_path = data_path + 'scans/'
    with open('data/meta_data/scannetv2_%s.txt' % split) as f:
        scan_ids = [line.rstrip() for line in f]
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
