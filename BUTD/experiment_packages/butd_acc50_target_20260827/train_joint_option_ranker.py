#!/usr/bin/env python
"""Train a scene-split candidate/detector/action ranker from ScanRefer train.

The validation split is never used for model or gate selection.  Each option
combines one deployable semantic candidate, one detector-match rule and one
box interpolation action.  IoU is used only as a training/evaluation label.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import torch


SCORE_KEYS = (
    'adapter_hit50_logit_at_candidate',
    'adapter_hit25_logit_at_candidate',
    'adapter_fused_at_candidate',
    'adapter_rescue_logit_at_candidate',
    'adapter_score_at_candidate',
    'adapter_delta_at_candidate',
)
MATCH_POWERS = (0.0, 0.5, 1.0)
ACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DECOMP_NAMES = ('ok', 'repaired', 'global_only', 'other')
SPACY_NAMES = (
    'spacy_aug_full_natural', 'spacy_aug_none', 'spacy_aug_yaw_only'
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def xyzxyz(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    center = boxes[..., :3]
    size = np.maximum(np.abs(boxes[..., 3:]), 1e-6)
    return np.concatenate([center - size / 2, center + size / 2], axis=-1)


def aligned_iou(boxes_a, boxes_b):
    a = xyzxyz(boxes_a)
    b = xyzxyz(boxes_b)
    intersection = np.maximum(
        np.minimum(a[..., 3:], b[..., 3:])
        - np.maximum(a[..., :3], b[..., :3]),
        0.0,
    ).prod(axis=-1)
    va = (a[..., 3:] - a[..., :3]).prod(axis=-1)
    vb = (b[..., 3:] - b[..., :3]).prod(axis=-1)
    return intersection / np.maximum(va + vb - intersection, 1e-9)


def pair_iou(one_box, boxes):
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    repeated = np.repeat(np.asarray(one_box, np.float32)[None], len(boxes), 0)
    return aligned_iou(repeated, np.asarray(boxes, np.float32))


def standardize(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float32))
    return (values - values.mean()) / (values.std() + 1e-6)


def normalized_rank(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float32))
    order = np.argsort(-values, kind='stable')
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks / max(1, len(values) - 1)


def candidate_positions(row, max_candidates):
    n = len(row['adapter_candidate_query'])
    arrays = {
        key: np.nan_to_num(np.asarray(row[key], dtype=np.float32))
        for key in SCORE_KEYS
    }
    priority = []

    def add(position):
        position = int(position)
        if position not in priority:
            priority.append(position)

    add(int(np.argmax(arrays['adapter_score_at_candidate'])))
    for key in SCORE_KEYS:
        for position in np.argsort(-arrays[key], kind='stable')[:2]:
            add(position)
    for position in np.argsort(
        -arrays['adapter_hit50_logit_at_candidate'], kind='stable'
    ):
        add(position)
    assert len(priority) == n
    return priority[:max_candidates]


def feature_names():
    names = []
    short = ('hit50', 'hit25', 'fused', 'rescue', 'adapter', 'delta')
    names += ['z_' + key for key in short]
    names += ['rank_' + key for key in short]
    names += ['z_minus_baseline_' + key for key in short]
    names += ['is_baseline', 'is_top_hit50']
    names += ['pred_log_size_x', 'pred_log_size_y', 'pred_log_size_z',
              'pred_log_volume']
    names += ['candidate_count_norm', 'detector_count_norm', 'text_cid_valid']
    names += ['decomp_' + key for key in DECOMP_NAMES]
    names += ['spacy_' + key for key in SPACY_NAMES]
    names += ['match_support', 'match_confidence', 'match_support_confidence']
    names += ['center_delta_norm_x', 'center_delta_norm_y',
              'center_delta_norm_z']
    names += ['log_size_ratio_x', 'log_size_ratio_y', 'log_size_ratio_z']
    names += ['match_conf_power']
    names += ['alpha', 'alpha_squared']
    names += ['alpha_{:.1f}'.format(value) for value in ACTIONS]
    names += ['option_log_size_x', 'option_log_size_y',
              'option_log_size_z', 'option_log_volume']
    return names


FEATURE_NAMES = feature_names()


@dataclass
class GroupMeta:
    scene_id: str
    example_id: int
    size: int
    baseline_index: int


def row_options(row, max_candidates=8):
    queries = list(row['adapter_candidate_query'])
    positions = candidate_positions(row, max_candidates)
    score_arrays = [
        np.nan_to_num(np.asarray(row[key], dtype=np.float32))
        for key in SCORE_KEYS
    ]
    z_arrays = [standardize(values) for values in score_arrays]
    rank_arrays = [normalized_rank(values) for values in score_arrays]
    baseline_position = int(np.argmax(score_arrays[4]))
    baseline_z = np.asarray(
        [values[baseline_position] for values in z_arrays], dtype=np.float32
    )
    top_hit50 = int(np.argmax(score_arrays[0]))
    pred_boxes = np.asarray(row['adapter_box_at_candidate'], dtype=np.float32)
    gt_box = np.asarray(row['gt_box'], dtype=np.float32)
    det_boxes = np.asarray(row.get('detected_box', []), dtype=np.float32)
    det_conf = np.asarray(
        row.get('detected_target_confidence', []), dtype=np.float32
    )
    if det_boxes.ndim != 2 or det_boxes.shape[-1] != 6:
        det_boxes = np.zeros((0, 6), dtype=np.float32)
    if len(det_conf) != len(det_boxes):
        det_conf = np.zeros(len(det_boxes), dtype=np.float32)
    det_conf = np.clip(np.nan_to_num(det_conf), 0.0, 1.0)
    decomp = str(row.get('decomposition_status', 'other'))
    if decomp not in DECOMP_NAMES:
        decomp = 'other'
    spacy = str(row.get('spacy_augmentation_bucket', ''))
    sample_features = [
        len(queries) / 15.0,
        len(det_boxes) / 80.0,
        float(row.get('text_target_cid') is not None),
    ]
    sample_features += [float(decomp == key) for key in DECOMP_NAMES]
    sample_features += [float(spacy == key) for key in SPACY_NAMES]
    features = []
    ious = []
    baseline_index = None
    for position in positions:
        pred = pred_boxes[position]
        pred_size = np.maximum(np.abs(pred[3:]), 1e-5)
        candidate_features = []
        candidate_features += [float(values[position]) for values in z_arrays]
        candidate_features += [float(values[position]) for values in rank_arrays]
        candidate_features += [
            float(z_arrays[idx][position] - baseline_z[idx])
            for idx in range(len(SCORE_KEYS))
        ]
        candidate_features += [
            float(position == baseline_position), float(position == top_hit50)
        ]
        candidate_features += list(np.log(pred_size))
        candidate_features += [float(np.log(np.prod(pred_size) + 1e-8))]
        support = pair_iou(pred, det_boxes)
        for match_power in MATCH_POWERS:
            if len(det_boxes):
                match_score = support * np.power(det_conf + 1e-6, match_power)
                match_index = int(np.argmax(match_score))
                matched = det_boxes[match_index]
                matched_support = float(support[match_index])
                matched_conf = float(det_conf[match_index])
            else:
                matched = pred.copy()
                matched_support = 0.0
                matched_conf = 0.0
            matched_size = np.maximum(np.abs(matched[3:]), 1e-5)
            center_delta = (matched[:3] - pred[:3]) / (
                0.5 * (matched_size + pred_size) + 1e-5
            )
            size_ratio = np.log(matched_size / pred_size)
            pair_features = [
                matched_support, matched_conf,
                matched_support * matched_conf,
            ]
            pair_features += list(center_delta)
            pair_features += list(size_ratio)
            pair_features += [float(match_power)]
            for action_index, alpha in enumerate(ACTIONS):
                option = pred + alpha * (matched - pred)
                option[3:] = np.maximum(np.abs(option[3:]), 1e-5)
                option_size = option[3:]
                action_features = [alpha, alpha * alpha]
                action_features += [
                    float(index == action_index)
                    for index in range(len(ACTIONS))
                ]
                action_features += list(np.log(option_size))
                action_features += [float(np.log(np.prod(option_size) + 1e-8))]
                vector = (
                    candidate_features + sample_features + pair_features
                    + action_features
                )
                assert len(vector) == len(FEATURE_NAMES), (
                    len(vector), len(FEATURE_NAMES)
                )
                if (
                    position == baseline_position
                    and match_power == MATCH_POWERS[0]
                    and action_index == 0
                ):
                    baseline_index = len(features)
                features.append(vector)
                ious.append(float(aligned_iou(option[None], gt_box[None])[0]))
    assert baseline_index is not None
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(ious, dtype=np.float32),
        baseline_index,
    )


def relevance(ious, mode='ordinal'):
    if mode == 'binary50':
        return (np.asarray(ious) >= 0.50).astype(np.int32)
    assert mode == 'ordinal', mode
    labels = np.zeros(len(ious), dtype=np.int32)
    labels[ious >= 0.10] = 1
    labels[ious >= 0.25] = 2
    labels[ious >= 0.50] = 3
    labels[ious >= 0.75] = 4
    return labels


def load_rows(path):
    payload = torch.load(path, map_location='cpu')
    rows = payload['rows']
    assert rows, path
    return rows


def build_dataset(rows, max_candidates, require_scene):
    group_features = []
    group_ious = []
    metas = []
    for index, row in enumerate(rows):
        scene_id = str(row.get('scene_id', ''))
        if require_scene:
            assert scene_id, ('missing scene_id', index)
        x, ious, baseline = row_options(row, max_candidates=max_candidates)
        group_features.append(x)
        group_ious.append(ious)
        metas.append(GroupMeta(
            scene_id=scene_id,
            example_id=int(row.get('example_id', index)),
            size=len(x),
            baseline_index=baseline,
        ))
        if (index + 1) % 1000 == 0:
            print('built_groups={}/{}'.format(index + 1, len(rows)), flush=True)
    return group_features, group_ious, metas


def scene_bucket(scene_id):
    value = int(hashlib.sha1(scene_id.encode('utf-8')).hexdigest()[:8], 16)
    return value % 100


def split_indices(metas):
    result = {'train': [], 'dev': [], 'test': []}
    for index, meta in enumerate(metas):
        bucket = scene_bucket(meta.scene_id)
        split = 'train' if bucket < 70 else ('dev' if bucket < 85 else 'test')
        result[split].append(index)
    scenes = {
        split: {metas[index].scene_id for index in indices}
        for split, indices in result.items()
    }
    assert scenes['train'].isdisjoint(scenes['dev'])
    assert scenes['train'].isdisjoint(scenes['test'])
    assert scenes['dev'].isdisjoint(scenes['test'])
    return result


def materialize(group_features, group_ious, metas, indices,
                label_mode='ordinal'):
    x = np.concatenate([group_features[index] for index in indices], axis=0)
    ious = np.concatenate([group_ious[index] for index in indices], axis=0)
    groups = np.asarray([metas[index].size for index in indices], dtype=np.int32)
    baselines = np.asarray(
        [metas[index].baseline_index for index in indices], dtype=np.int32
    )
    return x, relevance(ious, mode=label_mode), ious, groups, baselines


def group_decisions(scores, ious, groups, baselines, threshold):
    selected = []
    baseline_ious = []
    gaps = []
    cursor = 0
    for size, baseline in zip(groups, baselines):
        group_scores = scores[cursor:cursor + size]
        group_ious = ious[cursor:cursor + size]
        best = int(np.argmax(group_scores))
        gap = float(group_scores[best] - group_scores[int(baseline)])
        chosen = best if gap >= threshold else int(baseline)
        selected.append(float(group_ious[chosen]))
        baseline_ious.append(float(group_ious[int(baseline)]))
        gaps.append(gap)
        cursor += size
    assert cursor == len(scores)
    return (
        np.asarray(selected, dtype=np.float32),
        np.asarray(baseline_ious, dtype=np.float32),
        np.asarray(gaps, dtype=np.float32),
    )


def normalize_group_scores(scores, groups):
    scores = np.asarray(scores, dtype=np.float32)
    normalized = np.empty_like(scores)
    cursor = 0
    for size in groups:
        size = int(size)
        values = scores[cursor:cursor + size]
        normalized[cursor:cursor + size] = (
            values - values.mean()
        ) / (values.std() + 1e-6)
        cursor += size
    assert cursor == len(scores)
    return normalized


def metrics(ious):
    return {
        'acc025': float(np.mean(ious >= 0.25)),
        'acc050': float(np.mean(ious >= 0.50)),
        'mean_iou': float(np.mean(ious)),
        'count': int(len(ious)),
    }


def choose_gate(scores, ious, groups, baselines):
    _, baseline_ious, gaps = group_decisions(
        scores, ious, groups, baselines, np.inf
    )
    baseline_metrics = metrics(baseline_ious)
    finite = gaps[np.isfinite(gaps)]
    thresholds = list(np.unique(np.quantile(finite, np.linspace(0, 1, 201))))
    thresholds += [float('-inf'), float('inf')]
    rows = []
    for threshold in thresholds:
        selected, _, _ = group_decisions(
            scores, ious, groups, baselines, float(threshold)
        )
        result = metrics(selected)
        result['threshold'] = float(threshold)
        result['changed_ratio'] = float(np.mean(gaps >= threshold))
        result['preserves_acc025'] = bool(
            result['acc025'] >= baseline_metrics['acc025'] - 0.001
        )
        rows.append(result)
    feasible = [row for row in rows if row['preserves_acc025']]
    best = max(feasible, key=lambda row: (
        row['acc050'], row['acc025'], row['mean_iou'], -row['changed_ratio']
    ))
    return best, baseline_metrics


def evaluate_split(booster, x, ious, groups, baselines, threshold, iteration):
    scores = booster.predict(x, num_iteration=iteration)
    selected, baseline_ious, gaps = group_decisions(
        scores, ious, groups, baselines, threshold
    )
    return {
        'selected': metrics(selected),
        'baseline': metrics(baseline_ious),
        'changed_ratio': float(np.mean(gaps >= threshold)),
        'score_gap_mean': float(np.mean(gaps)),
    }


def train(args):
    os.makedirs(args.output_dir, exist_ok=False)
    rows = load_rows(args.train_dump)
    group_features, group_ious, metas = build_dataset(
        rows, args.max_candidates, require_scene=True
    )
    splits = split_indices(metas)
    arrays = {
        split: materialize(
            group_features, group_ious, metas, splits[split]
        )
        for split in ('train', 'dev', 'test')
    }
    x_train, y_train, _, groups_train, _ = arrays['train']
    x_dev, y_dev, iou_dev, groups_dev, baseline_dev = arrays['dev']
    print('split_groups', {key: len(value) for key, value in splits.items()})
    print('split_options', {
        key: int(arrays[key][0].shape[0]) for key in arrays
    })
    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        label_gain=[0, 1, 4, 16, 24],
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=0,
        n_jobs=args.num_threads,
        verbosity=-1,
    )
    ranker.fit(
        x_train, y_train,
        group=groups_train.tolist(),
        eval_set=[(x_dev, y_dev)],
        eval_group=[groups_dev.tolist()],
        eval_at=[1],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(20)],
        feature_name=FEATURE_NAMES,
    )
    iteration = int(ranker.best_iteration_ or ranker.n_estimators)
    dev_scores = ranker.booster_.predict(x_dev, num_iteration=iteration)
    gate, dev_baseline = choose_gate(
        dev_scores, iou_dev, groups_dev, baseline_dev
    )
    internal = {
        'dev': evaluate_split(
            ranker.booster_, x_dev, iou_dev, groups_dev, baseline_dev,
            gate['threshold'], iteration
        ),
    }
    x_test, _, iou_test, groups_test, baseline_test = arrays['test']
    internal['test'] = evaluate_split(
        ranker.booster_, x_test, iou_test, groups_test, baseline_test,
        gate['threshold'], iteration
    )
    model_path = os.path.join(args.output_dir, 'joint_option_ranker.txt')
    ranker.booster_.save_model(model_path, num_iteration=iteration)
    split_scenes = {
        split: sorted({metas[index].scene_id for index in indices})
        for split, indices in splits.items()
    }
    receipt = {
        'protocol': 'scene_hash_70_15_15_train_only',
        'train_dump': os.path.abspath(args.train_dump),
        'train_dump_sha256': sha256(args.train_dump),
        'script_sha256': sha256(os.path.abspath(__file__)),
        'model_path': os.path.abspath(model_path),
        'max_candidates': args.max_candidates,
        'match_powers': list(MATCH_POWERS),
        'actions': list(ACTIONS),
        'feature_names': FEATURE_NAMES,
        'best_iteration': iteration,
        'gate': gate,
        'dev_baseline_for_gate': dev_baseline,
        'internal': internal,
        'split_group_counts': {key: len(value) for key, value in splits.items()},
        'split_scene_counts': {
            key: len(value) for key, value in split_scenes.items()
        },
        'split_scenes_sha256': {
            key: hashlib.sha256(
                '\n'.join(value).encode('utf-8')
            ).hexdigest()
            for key, value in split_scenes.items()
        },
    }
    receipt['model_sha256'] = sha256(model_path)
    lock_path = os.path.join(args.output_dir, 'locked_policy.json')
    with open(lock_path, 'w', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def evaluate(args):
    lock = json.load(open(args.lock_json, 'r', encoding='utf-8'))
    assert sha256(args.model) == lock['model_sha256']
    assert lock['match_powers'] == list(MATCH_POWERS)
    assert lock['actions'] == list(ACTIONS)
    assert lock['feature_names'] == FEATURE_NAMES
    rows = load_rows(args.dump)
    group_features, group_ious, metas = build_dataset(
        rows, int(lock['max_candidates']), require_scene=False
    )
    x = np.concatenate(group_features, axis=0)
    ious = np.concatenate(group_ious, axis=0)
    groups = np.asarray([meta.size for meta in metas], dtype=np.int32)
    baselines = np.asarray(
        [meta.baseline_index for meta in metas], dtype=np.int32
    )
    booster = lgb.Booster(model_file=args.model)
    result = evaluate_split(
        booster, x, ious, groups, baselines,
        float(lock['gate']['threshold']), int(lock['best_iteration'])
    )
    result.update({
        'dump': os.path.abspath(args.dump),
        'dump_sha256': sha256(args.dump),
        'model_sha256': lock['model_sha256'],
        'lock_sha256': sha256(args.lock_json),
        'threshold': lock['gate']['threshold'],
        'goal_achieved_offline': bool(
            result['selected']['acc025'] > 0.5391
            and result['selected']['acc050'] > 0.4241
        ),
        'diagnostic_only': True,
    })
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


def make_fixed_ranker(n_estimators, num_threads, random_state):
    return lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        label_gain=[0, 1, 4, 16, 24],
        n_estimators=int(n_estimators),
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=int(random_state),
        n_jobs=int(num_threads),
        verbosity=-1,
    )


def crossfit_train(args):
    os.makedirs(args.output_dir, exist_ok=False)
    source_lock = json.load(open(
        args.source_lock_json, 'r', encoding='utf-8'
    ))
    assert source_lock['feature_names'] == FEATURE_NAMES
    assert source_lock['match_powers'] == list(MATCH_POWERS)
    assert source_lock['actions'] == list(ACTIONS)
    max_candidates = int(source_lock['max_candidates'])
    fixed_iterations = int(source_lock['best_iteration'])
    rows = load_rows(args.train_dump)
    group_features, group_ious, metas = build_dataset(
        rows, max_candidates=max_candidates, require_scene=True
    )
    fold_indices = {fold: [] for fold in range(args.num_folds)}
    for index, meta in enumerate(metas):
        fold_indices[scene_bucket(meta.scene_id) % args.num_folds].append(index)
    assert all(fold_indices.values())
    oof_scores_by_group = [None] * len(metas)
    models = []
    fold_receipts = []
    all_indices = set(range(len(metas)))
    for fold in range(args.num_folds):
        heldout = fold_indices[fold]
        training = sorted(all_indices.difference(heldout))
        x_train, y_train, _, groups_train, _ = materialize(
            group_features, group_ious, metas, training
        )
        x_heldout, _, iou_heldout, groups_heldout, baseline_heldout = (
            materialize(group_features, group_ious, metas, heldout)
        )
        print('fold={} train_groups={} train_options={} heldout_groups={} '
              'heldout_options={}'.format(
                  fold, len(training), len(x_train),
                  len(heldout), len(x_heldout)), flush=True)
        ranker = make_fixed_ranker(
            fixed_iterations, args.num_threads, random_state=fold
        )
        ranker.fit(
            x_train, y_train,
            group=groups_train.tolist(),
            feature_name=FEATURE_NAMES,
            callbacks=[lgb.log_evaluation(0)],
        )
        heldout_scores = ranker.booster_.predict(
            x_heldout, num_iteration=fixed_iterations
        )
        cursor = 0
        for group_index, group_size in zip(heldout, groups_heldout):
            oof_scores_by_group[group_index] = heldout_scores[
                cursor:cursor + int(group_size)
            ].astype(np.float32, copy=True)
            cursor += int(group_size)
        assert cursor == len(heldout_scores)
        model_name = 'joint_option_ranker_fold{}.txt'.format(fold)
        model_path = os.path.join(args.output_dir, model_name)
        ranker.booster_.save_model(
            model_path, num_iteration=fixed_iterations
        )
        fold_receipts.append({
            'fold': fold,
            'train_groups': len(training),
            'heldout_groups': len(heldout),
            'heldout_baseline': metrics(group_decisions(
                heldout_scores, iou_heldout, groups_heldout,
                baseline_heldout, np.inf
            )[1]),
        })
        models.append({
            'path': os.path.abspath(model_path),
            'sha256': sha256(model_path),
            'fold': fold,
        })
        del x_train, y_train, x_heldout, heldout_scores, ranker
        gc.collect()
    assert all(value is not None for value in oof_scores_by_group)
    all_order = list(range(len(metas)))
    oof_scores = np.concatenate(oof_scores_by_group, axis=0)
    all_ious = np.concatenate(group_ious, axis=0)
    all_groups = np.asarray([meta.size for meta in metas], dtype=np.int32)
    all_baselines = np.asarray(
        [meta.baseline_index for meta in metas], dtype=np.int32
    )
    gate, baseline = choose_gate(
        oof_scores, all_ious, all_groups, all_baselines
    )
    selected, baseline_ious, gaps = group_decisions(
        oof_scores, all_ious, all_groups, all_baselines,
        float(gate['threshold'])
    )
    receipt = {
        'protocol': 'five_fold_scene_crossfit_oof_gate',
        'train_dump': os.path.abspath(args.train_dump),
        'train_dump_sha256': sha256(args.train_dump),
        'source_lock_json': os.path.abspath(args.source_lock_json),
        'source_lock_sha256': sha256(args.source_lock_json),
        'script_sha256': sha256(os.path.abspath(__file__)),
        'max_candidates': max_candidates,
        'match_powers': list(MATCH_POWERS),
        'actions': list(ACTIONS),
        'feature_names': FEATURE_NAMES,
        'fixed_iterations': fixed_iterations,
        'num_folds': int(args.num_folds),
        'models': models,
        'gate': gate,
        'oof': {
            'baseline': metrics(baseline_ious),
            'selected': metrics(selected),
            'changed_ratio': float(np.mean(gaps >= gate['threshold'])),
            'score_gap_mean': float(np.mean(gaps)),
        },
        'folds': fold_receipts,
    }
    lock_path = os.path.join(args.output_dir, 'locked_ensemble_policy.json')
    with open(lock_path, 'w', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def ensemble_evaluate(args):
    lock = json.load(open(args.lock_json, 'r', encoding='utf-8'))
    assert lock['protocol'] == 'five_fold_scene_crossfit_oof_gate'
    assert lock['feature_names'] == FEATURE_NAMES
    assert lock['match_powers'] == list(MATCH_POWERS)
    assert lock['actions'] == list(ACTIONS)
    for model in lock['models']:
        assert sha256(model['path']) == model['sha256']
    rows = load_rows(args.dump)
    group_features, group_ious, metas = build_dataset(
        rows, int(lock['max_candidates']), require_scene=False
    )
    x = np.concatenate(group_features, axis=0)
    ious = np.concatenate(group_ious, axis=0)
    groups = np.asarray([meta.size for meta in metas], dtype=np.int32)
    baselines = np.asarray(
        [meta.baseline_index for meta in metas], dtype=np.int32
    )
    score_sum = np.zeros(len(x), dtype=np.float64)
    for model in lock['models']:
        booster = lgb.Booster(model_file=model['path'])
        score_sum += booster.predict(
            x, num_iteration=int(lock['fixed_iterations'])
        )
    scores = score_sum / len(lock['models'])
    threshold = float(lock['gate']['threshold'])
    selected, baseline_ious, gaps = group_decisions(
        scores, ious, groups, baselines, threshold
    )
    result = {
        'selected': metrics(selected),
        'baseline': metrics(baseline_ious),
        'changed_ratio': float(np.mean(gaps >= threshold)),
        'score_gap_mean': float(np.mean(gaps)),
        'dump': os.path.abspath(args.dump),
        'dump_sha256': sha256(args.dump),
        'lock_sha256': sha256(args.lock_json),
        'threshold': threshold,
        'goal_achieved_offline': bool(
            metrics(selected)['acc025'] > 0.5391
            and metrics(selected)['acc050'] > 0.4241
        ),
        'diagnostic_only': True,
    }
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


def binary50_train(args):
    os.makedirs(args.output_dir, exist_ok=False)
    rows = load_rows(args.train_dump)
    group_features, group_ious, metas = build_dataset(
        rows, args.max_candidates, require_scene=True
    )
    splits = split_indices(metas)
    arrays = {
        split: materialize(
            group_features, group_ious, metas, splits[split],
            label_mode='binary50'
        )
        for split in ('train', 'dev', 'test')
    }
    x_train, y_train, _, groups_train, _ = arrays['train']
    x_dev, y_dev, iou_dev, groups_dev, baseline_dev = arrays['dev']
    configs = (
        {'name': 'conservative', 'num_leaves': 15, 'max_depth': 6,
         'min_child_samples': 300},
        {'name': 'balanced', 'num_leaves': 31, 'max_depth': 8,
         'min_child_samples': 200},
        {'name': 'wide', 'num_leaves': 63, 'max_depth': 10,
         'min_child_samples': 100},
    )
    candidates = []
    for config_index, config in enumerate(configs):
        ranker = lgb.LGBMRanker(
            objective='lambdarank', metric='ndcg', label_gain=[0, 1],
            n_estimators=800, learning_rate=0.05,
            num_leaves=config['num_leaves'],
            max_depth=config['max_depth'],
            min_child_samples=config['min_child_samples'],
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=config_index,
            n_jobs=args.num_threads, verbosity=-1,
        )
        ranker.fit(
            x_train, y_train,
            group=groups_train.tolist(),
            eval_set=[(x_dev, y_dev)],
            eval_group=[groups_dev.tolist()],
            eval_at=[1],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(20)],
            feature_name=FEATURE_NAMES,
        )
        iteration = int(ranker.best_iteration_ or ranker.n_estimators)
        dev_scores = ranker.booster_.predict(x_dev, num_iteration=iteration)
        gate, dev_baseline = choose_gate(
            dev_scores, iou_dev, groups_dev, baseline_dev
        )
        dev_result = evaluate_split(
            ranker.booster_, x_dev, iou_dev, groups_dev, baseline_dev,
            gate['threshold'], iteration
        )
        candidates.append({
            'config': config,
            'iteration': iteration,
            'gate': gate,
            'dev_baseline_for_gate': dev_baseline,
            'dev': dev_result,
            'booster': ranker.booster_,
        })
        print('binary50_candidate', json.dumps({
            key: value for key, value in candidates[-1].items()
            if key != 'booster'
        }, sort_keys=True), flush=True)
    selected_index = max(range(len(candidates)), key=lambda index: (
        candidates[index]['dev']['selected']['acc050'],
        candidates[index]['dev']['selected']['acc025'],
        candidates[index]['dev']['selected']['mean_iou'],
        -candidates[index]['dev']['changed_ratio'],
    ))
    selected = candidates[selected_index]
    x_test, _, iou_test, groups_test, baseline_test = arrays['test']
    test_result = evaluate_split(
        selected['booster'], x_test, iou_test, groups_test, baseline_test,
        selected['gate']['threshold'], selected['iteration']
    )
    model_path = os.path.join(args.output_dir, 'binary50_option_ranker.txt')
    selected['booster'].save_model(
        model_path, num_iteration=selected['iteration']
    )
    receipt_candidates = []
    for candidate in candidates:
        receipt_candidates.append({
            key: value for key, value in candidate.items() if key != 'booster'
        })
    receipt = {
        'protocol': 'scene_hash_train_only_binary50_config_selection',
        'train_dump': os.path.abspath(args.train_dump),
        'train_dump_sha256': sha256(args.train_dump),
        'script_sha256': sha256(os.path.abspath(__file__)),
        'model_path': os.path.abspath(model_path),
        'max_candidates': args.max_candidates,
        'match_powers': list(MATCH_POWERS),
        'actions': list(ACTIONS),
        'feature_names': FEATURE_NAMES,
        'best_iteration': selected['iteration'],
        'gate': selected['gate'],
        'selected_config_index': selected_index,
        'selected_config': selected['config'],
        'candidates': receipt_candidates,
        'internal': {'dev': selected['dev'], 'test': test_result},
        'split_group_counts': {key: len(value) for key, value in splits.items()},
    }
    receipt['model_sha256'] = sha256(model_path)
    lock_path = os.path.join(args.output_dir, 'locked_binary50_policy.json')
    with open(lock_path, 'w', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def blend_train(args):
    os.makedirs(args.output_dir, exist_ok=False)
    ordinal_lock = json.load(open(
        args.ordinal_lock_json, 'r', encoding='utf-8'
    ))
    binary_lock = json.load(open(
        args.binary_lock_json, 'r', encoding='utf-8'
    ))
    assert sha256(args.ordinal_model) == ordinal_lock['model_sha256']
    assert sha256(args.binary_model) == binary_lock['model_sha256']
    assert ordinal_lock['max_candidates'] == binary_lock['max_candidates']
    assert ordinal_lock['feature_names'] == binary_lock['feature_names']
    max_candidates = int(ordinal_lock['max_candidates'])
    rows = load_rows(args.train_dump)
    group_features, group_ious, metas = build_dataset(
        rows, max_candidates, require_scene=True
    )
    splits = split_indices(metas)
    arrays = {
        split: materialize(
            group_features, group_ious, metas, splits[split]
        )
        for split in ('dev', 'test')
    }
    ordinal = lgb.Booster(model_file=args.ordinal_model)
    binary = lgb.Booster(model_file=args.binary_model)
    x_dev, _, iou_dev, groups_dev, baseline_dev = arrays['dev']
    ordinal_dev = normalize_group_scores(
        ordinal.predict(
            x_dev, num_iteration=int(ordinal_lock['best_iteration'])
        ), groups_dev
    )
    binary_dev = normalize_group_scores(
        binary.predict(
            x_dev, num_iteration=int(binary_lock['best_iteration'])
        ), groups_dev
    )
    candidates = []
    for binary_weight in np.linspace(0.0, 1.0, 21):
        scores = (
            (1.0 - binary_weight) * ordinal_dev
            + binary_weight * binary_dev
        )
        gate, dev_baseline = choose_gate(
            scores, iou_dev, groups_dev, baseline_dev
        )
        selected, baseline_ious, gaps = group_decisions(
            scores, iou_dev, groups_dev, baseline_dev,
            float(gate['threshold'])
        )
        candidates.append({
            'binary_weight': float(binary_weight),
            'ordinal_weight': float(1.0 - binary_weight),
            'gate': gate,
            'dev': {
                'selected': metrics(selected),
                'baseline': metrics(baseline_ious),
                'changed_ratio': float(np.mean(gaps >= gate['threshold'])),
            },
        })
    selected_index = max(range(len(candidates)), key=lambda index: (
        candidates[index]['dev']['selected']['acc050'],
        candidates[index]['dev']['selected']['acc025'],
        candidates[index]['dev']['selected']['mean_iou'],
        -candidates[index]['dev']['changed_ratio'],
    ))
    chosen = candidates[selected_index]
    x_test, _, iou_test, groups_test, baseline_test = arrays['test']
    ordinal_test = normalize_group_scores(
        ordinal.predict(
            x_test, num_iteration=int(ordinal_lock['best_iteration'])
        ), groups_test
    )
    binary_test = normalize_group_scores(
        binary.predict(
            x_test, num_iteration=int(binary_lock['best_iteration'])
        ), groups_test
    )
    test_scores = (
        chosen['ordinal_weight'] * ordinal_test
        + chosen['binary_weight'] * binary_test
    )
    test_selected, test_baseline, test_gaps = group_decisions(
        test_scores, iou_test, groups_test, baseline_test,
        float(chosen['gate']['threshold'])
    )
    receipt = {
        'protocol': 'train_dev_locked_group_standardized_ordinal_binary_blend',
        'train_dump': os.path.abspath(args.train_dump),
        'train_dump_sha256': sha256(args.train_dump),
        'script_sha256': sha256(os.path.abspath(__file__)),
        'max_candidates': max_candidates,
        'feature_names': FEATURE_NAMES,
        'match_powers': list(MATCH_POWERS),
        'actions': list(ACTIONS),
        'ordinal_model': os.path.abspath(args.ordinal_model),
        'ordinal_model_sha256': ordinal_lock['model_sha256'],
        'ordinal_iteration': int(ordinal_lock['best_iteration']),
        'binary_model': os.path.abspath(args.binary_model),
        'binary_model_sha256': binary_lock['model_sha256'],
        'binary_iteration': int(binary_lock['best_iteration']),
        'selected_index': selected_index,
        'selected': chosen,
        'candidates': candidates,
        'test': {
            'selected': metrics(test_selected),
            'baseline': metrics(test_baseline),
            'changed_ratio': float(
                np.mean(test_gaps >= chosen['gate']['threshold'])
            ),
        },
    }
    lock_path = os.path.join(args.output_dir, 'locked_blend_policy.json')
    with open(lock_path, 'w', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def blend_evaluate(args):
    lock = json.load(open(args.lock_json, 'r', encoding='utf-8'))
    assert lock['protocol'].startswith('train_dev_locked_group_standardized')
    assert sha256(lock['ordinal_model']) == lock['ordinal_model_sha256']
    assert sha256(lock['binary_model']) == lock['binary_model_sha256']
    rows = load_rows(args.dump)
    group_features, group_ious, metas = build_dataset(
        rows, int(lock['max_candidates']), require_scene=False
    )
    x = np.concatenate(group_features, axis=0)
    ious = np.concatenate(group_ious, axis=0)
    groups = np.asarray([meta.size for meta in metas], dtype=np.int32)
    baselines = np.asarray(
        [meta.baseline_index for meta in metas], dtype=np.int32
    )
    ordinal = lgb.Booster(model_file=lock['ordinal_model'])
    binary = lgb.Booster(model_file=lock['binary_model'])
    ordinal_scores = normalize_group_scores(
        ordinal.predict(x, num_iteration=int(lock['ordinal_iteration'])),
        groups
    )
    binary_scores = normalize_group_scores(
        binary.predict(x, num_iteration=int(lock['binary_iteration'])),
        groups
    )
    selected_config = lock['selected']
    scores = (
        float(selected_config['ordinal_weight']) * ordinal_scores
        + float(selected_config['binary_weight']) * binary_scores
    )
    threshold = float(selected_config['gate']['threshold'])
    selected, baseline_ious, gaps = group_decisions(
        scores, ious, groups, baselines, threshold
    )
    result = {
        'selected': metrics(selected),
        'baseline': metrics(baseline_ious),
        'changed_ratio': float(np.mean(gaps >= threshold)),
        'score_gap_mean': float(np.mean(gaps)),
        'binary_weight': float(selected_config['binary_weight']),
        'ordinal_weight': float(selected_config['ordinal_weight']),
        'threshold': threshold,
        'dump': os.path.abspath(args.dump),
        'dump_sha256': sha256(args.dump),
        'lock_sha256': sha256(args.lock_json),
        'goal_achieved_offline': bool(
            metrics(selected)['acc025'] > 0.5391
            and metrics(selected)['acc050'] > 0.4241
        ),
        'diagnostic_only': True,
    }
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


def option_weights(groups):
    weights = []
    for size in groups:
        size = int(size)
        weights.extend([1.0 / size] * size)
    result = np.asarray(weights, dtype=np.float32)
    result *= len(result) / result.sum()
    return result


def pointwise_train(args):
    os.makedirs(args.output_dir, exist_ok=False)
    rows = load_rows(args.train_dump)
    group_features, group_ious, metas = build_dataset(
        rows, args.max_candidates, require_scene=True
    )
    splits = split_indices(metas)
    arrays = {
        split: materialize(
            group_features, group_ious, metas, splits[split],
            label_mode='binary50'
        )
        for split in ('train', 'dev', 'test')
    }
    x_train, y_train_binary, iou_train, groups_train, _ = arrays['train']
    x_dev, y_dev_binary, iou_dev, groups_dev, baseline_dev = arrays['dev']
    train_weights = option_weights(groups_train)
    dev_weights = option_weights(groups_dev)
    specs = (
        {'name': 'classifier_balanced', 'kind': 'classifier',
         'num_leaves': 31, 'max_depth': 8, 'min_child_samples': 200},
        {'name': 'classifier_wide', 'kind': 'classifier',
         'num_leaves': 63, 'max_depth': 10, 'min_child_samples': 100},
        {'name': 'regression_l1', 'kind': 'regressor',
         'num_leaves': 31, 'max_depth': 8, 'min_child_samples': 200},
        {'name': 'regression_huber', 'kind': 'regressor_huber',
         'num_leaves': 31, 'max_depth': 8, 'min_child_samples': 200},
    )
    candidates = []
    for spec_index, spec in enumerate(specs):
        common = dict(
            n_estimators=800, learning_rate=0.05,
            num_leaves=spec['num_leaves'], max_depth=spec['max_depth'],
            min_child_samples=spec['min_child_samples'],
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=spec_index,
            n_jobs=args.num_threads, verbosity=-1,
        )
        if spec['kind'] == 'classifier':
            learner = lgb.LGBMClassifier(objective='binary', **common)
            train_target = y_train_binary
            dev_target = y_dev_binary
            metric = 'binary_logloss'
        else:
            objective = 'regression_l1' if spec['kind'] == 'regressor' else 'huber'
            learner = lgb.LGBMRegressor(objective=objective, **common)
            train_target = iou_train
            dev_target = iou_dev
            metric = 'l1'
        learner.fit(
            x_train, train_target,
            sample_weight=train_weights,
            eval_set=[(x_dev, dev_target)],
            eval_sample_weight=[dev_weights],
            eval_metric=metric,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(20)],
            feature_name=FEATURE_NAMES,
        )
        iteration = int(learner.best_iteration_ or learner.n_estimators)
        if spec['kind'] == 'classifier':
            dev_scores = learner.predict_proba(
                x_dev, num_iteration=iteration
            )[:, 1]
        else:
            dev_scores = learner.predict(x_dev, num_iteration=iteration)
        gate, dev_baseline = choose_gate(
            dev_scores, iou_dev, groups_dev, baseline_dev
        )
        selected_dev, baseline_ious, gaps = group_decisions(
            dev_scores, iou_dev, groups_dev, baseline_dev,
            float(gate['threshold'])
        )
        candidates.append({
            'spec': spec,
            'iteration': iteration,
            'gate': gate,
            'dev': {
                'selected': metrics(selected_dev),
                'baseline': metrics(baseline_ious),
                'changed_ratio': float(np.mean(gaps >= gate['threshold'])),
            },
            'booster': learner.booster_,
        })
        print('pointwise_candidate', json.dumps({
            key: value for key, value in candidates[-1].items()
            if key != 'booster'
        }, sort_keys=True), flush=True)
    selected_index = max(range(len(candidates)), key=lambda index: (
        candidates[index]['dev']['selected']['acc050'],
        candidates[index]['dev']['selected']['acc025'],
        candidates[index]['dev']['selected']['mean_iou'],
        -candidates[index]['dev']['changed_ratio'],
    ))
    chosen = candidates[selected_index]
    x_test, _, iou_test, groups_test, baseline_test = arrays['test']
    test_scores = chosen['booster'].predict(
        x_test, num_iteration=chosen['iteration']
    )
    test_selected, test_baseline, test_gaps = group_decisions(
        test_scores, iou_test, groups_test, baseline_test,
        float(chosen['gate']['threshold'])
    )
    model_path = os.path.join(args.output_dir, 'pointwise_option_model.txt')
    chosen['booster'].save_model(
        model_path, num_iteration=chosen['iteration']
    )
    receipt_candidates = [
        {key: value for key, value in candidate.items() if key != 'booster'}
        for candidate in candidates
    ]
    receipt = {
        'protocol': 'scene_hash_train_only_equal_group_weight_pointwise',
        'train_dump': os.path.abspath(args.train_dump),
        'train_dump_sha256': sha256(args.train_dump),
        'script_sha256': sha256(os.path.abspath(__file__)),
        'model_path': os.path.abspath(model_path),
        'model_kind': chosen['spec']['kind'],
        'model_sha256': sha256(model_path),
        'best_iteration': chosen['iteration'],
        'max_candidates': args.max_candidates,
        'match_powers': list(MATCH_POWERS),
        'actions': list(ACTIONS),
        'feature_names': FEATURE_NAMES,
        'gate': chosen['gate'],
        'selected_index': selected_index,
        'selected_spec': chosen['spec'],
        'candidates': receipt_candidates,
        'internal': {
            'dev': chosen['dev'],
            'test': {
                'selected': metrics(test_selected),
                'baseline': metrics(test_baseline),
                'changed_ratio': float(
                    np.mean(test_gaps >= chosen['gate']['threshold'])
                ),
            },
        },
    }
    lock_path = os.path.join(args.output_dir, 'locked_pointwise_policy.json')
    with open(lock_path, 'w', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def mixed_binary50_train(args):
    os.makedirs(args.output_dir, exist_ok=False)
    clean_rows = load_rows(args.clean_dump)
    augmented_rows = load_rows(args.augmented_dump)
    assert len(clean_rows) == len(augmented_rows)
    for index, (clean, augmented) in enumerate(zip(clean_rows, augmented_rows)):
        clean_key = (
            str(clean.get('scene_id', '')), str(clean.get('ann_id', '')),
            str(clean.get('object_id', '')),
        )
        augmented_key = (
            str(augmented.get('scene_id', '')),
            str(augmented.get('ann_id', '')),
            str(augmented.get('object_id', '')),
        )
        assert clean_key == augmented_key, (index, clean_key, augmented_key)
    clean_features, clean_ious, clean_metas = build_dataset(
        clean_rows, args.max_candidates, require_scene=True
    )
    augmented_features, augmented_ious, augmented_metas = build_dataset(
        augmented_rows, args.max_candidates, require_scene=True
    )
    splits = split_indices(clean_metas)
    for clean_meta, augmented_meta in zip(clean_metas, augmented_metas):
        assert clean_meta.scene_id == augmented_meta.scene_id
    clean_train = materialize(
        clean_features, clean_ious, clean_metas, splits['train'],
        label_mode='binary50'
    )
    augmented_train = materialize(
        augmented_features, augmented_ious, augmented_metas, splits['train'],
        label_mode='binary50'
    )
    dev = materialize(
        clean_features, clean_ious, clean_metas, splits['dev'],
        label_mode='binary50'
    )
    test = materialize(
        clean_features, clean_ious, clean_metas, splits['test'],
        label_mode='binary50'
    )
    x_dev, y_dev, iou_dev, groups_dev, baseline_dev = dev
    training_variants = (
        ('augmented_only', augmented_train[0], augmented_train[1],
         augmented_train[3]),
        ('clean_plus_augmented',
         np.concatenate([clean_train[0], augmented_train[0]], axis=0),
         np.concatenate([clean_train[1], augmented_train[1]], axis=0),
         np.concatenate([clean_train[3], augmented_train[3]], axis=0)),
    )
    candidates = []
    for variant_index, (name, x_train, y_train, groups_train) in enumerate(
        training_variants
    ):
        ranker = lgb.LGBMRanker(
            objective='lambdarank', metric='ndcg', label_gain=[0, 1],
            n_estimators=800, learning_rate=0.05,
            num_leaves=31, max_depth=8, min_child_samples=200,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=variant_index,
            n_jobs=args.num_threads, verbosity=-1,
        )
        ranker.fit(
            x_train, y_train, group=groups_train.tolist(),
            eval_set=[(x_dev, y_dev)], eval_group=[groups_dev.tolist()],
            eval_at=[1],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(20)],
            feature_name=FEATURE_NAMES,
        )
        iteration = int(ranker.best_iteration_ or ranker.n_estimators)
        dev_scores = ranker.booster_.predict(x_dev, num_iteration=iteration)
        gate, _ = choose_gate(dev_scores, iou_dev, groups_dev, baseline_dev)
        dev_selected, dev_baseline, dev_gaps = group_decisions(
            dev_scores, iou_dev, groups_dev, baseline_dev,
            float(gate['threshold'])
        )
        candidates.append({
            'name': name,
            'iteration': iteration,
            'gate': gate,
            'dev': {
                'selected': metrics(dev_selected),
                'baseline': metrics(dev_baseline),
                'changed_ratio': float(np.mean(dev_gaps >= gate['threshold'])),
            },
            'booster': ranker.booster_,
        })
        print('mixed_binary50_candidate', json.dumps({
            key: value for key, value in candidates[-1].items()
            if key != 'booster'
        }, sort_keys=True), flush=True)
    selected_index = max(range(len(candidates)), key=lambda index: (
        candidates[index]['dev']['selected']['acc050'],
        candidates[index]['dev']['selected']['acc025'],
        candidates[index]['dev']['selected']['mean_iou'],
    ))
    chosen = candidates[selected_index]
    x_test, _, iou_test, groups_test, baseline_test = test
    test_scores = chosen['booster'].predict(
        x_test, num_iteration=chosen['iteration']
    )
    test_selected, test_baseline, test_gaps = group_decisions(
        test_scores, iou_test, groups_test, baseline_test,
        float(chosen['gate']['threshold'])
    )
    model_path = os.path.join(args.output_dir, 'mixed_binary50_ranker.txt')
    chosen['booster'].save_model(model_path, num_iteration=chosen['iteration'])
    receipt = {
        'protocol': 'clean_dev_augmented_or_mixed_train_binary50',
        'clean_dump': os.path.abspath(args.clean_dump),
        'clean_dump_sha256': sha256(args.clean_dump),
        'augmented_dump': os.path.abspath(args.augmented_dump),
        'augmented_dump_sha256': sha256(args.augmented_dump),
        'script_sha256': sha256(os.path.abspath(__file__)),
        'model_path': os.path.abspath(model_path),
        'model_sha256': sha256(model_path),
        'best_iteration': chosen['iteration'],
        'max_candidates': args.max_candidates,
        'match_powers': list(MATCH_POWERS),
        'actions': list(ACTIONS),
        'feature_names': FEATURE_NAMES,
        'gate': chosen['gate'],
        'selected_index': selected_index,
        'selected_name': chosen['name'],
        'candidates': [
            {key: value for key, value in candidate.items() if key != 'booster'}
            for candidate in candidates
        ],
        'internal': {
            'dev': chosen['dev'],
            'test': {
                'selected': metrics(test_selected),
                'baseline': metrics(test_baseline),
                'changed_ratio': float(
                    np.mean(test_gaps >= chosen['gate']['threshold'])
                ),
            },
        },
    }
    lock_path = os.path.join(args.output_dir, 'locked_mixed_policy.json')
    with open(lock_path, 'w', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def oracle(args):
    rows = load_rows(args.dump)
    baseline_ious = []
    oracle_ious = []
    option_counts = []
    for index, row in enumerate(rows):
        _, ious, baseline = row_options(
            row, max_candidates=args.max_candidates
        )
        baseline_ious.append(float(ious[baseline]))
        oracle_ious.append(float(ious.max()))
        option_counts.append(len(ious))
        if (index + 1) % 1000 == 0:
            print('oracle_groups={}/{}'.format(index + 1, len(rows)), flush=True)
    result = {
        'baseline': metrics(np.asarray(baseline_ious, dtype=np.float32)),
        'oracle': metrics(np.asarray(oracle_ious, dtype=np.float32)),
        'option_count_mean': float(np.mean(option_counts)),
        'option_count_max': int(max(option_counts)),
        'max_candidates': int(args.max_candidates),
        'dump': os.path.abspath(args.dump),
        'dump_sha256': sha256(args.dump),
        'diagnostic_only': True,
    }
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


def self_test():
    row = {
        'example_id': 0,
        'scene_id': 'scene0000_00',
        'adapter_candidate_query': [1, 2, 3],
        'adapter_hit50_logit_at_candidate': [3.0, 2.0, 1.0],
        'adapter_hit25_logit_at_candidate': [1.0, 3.0, 2.0],
        'adapter_fused_at_candidate': [2.0, 3.0, 1.0],
        'adapter_rescue_logit_at_candidate': [3.0, 1.0, 2.0],
        'adapter_score_at_candidate': [1.0, 3.0, 2.0],
        'adapter_delta_at_candidate': [0.0, 0.2, 0.1],
        'adapter_box_at_candidate': [
            [0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 1],
            [2, 0, 0, 1, 1, 1],
        ],
        'gt_box': [1, 0, 0, 1, 1, 1],
        'detected_box': [[1.1, 0, 0, 1, 1, 1]],
        'detected_target_confidence': [0.9],
        'text_target_cid': 1,
        'decomposition_status': 'ok',
        'spacy_augmentation_bucket': 'spacy_aug_none',
    }
    x, ious, baseline = row_options(row, max_candidates=3)
    assert x.shape == (3 * len(MATCH_POWERS) * len(ACTIONS), len(FEATURE_NAMES))
    assert baseline >= 0 and abs(float(ious[baseline]) - 1.0) < 1e-6
    changed_gt = dict(row)
    changed_gt['gt_box'] = [20, 20, 20, 1, 1, 1]
    x_changed_gt, ious_changed_gt, baseline_changed_gt = row_options(
        changed_gt, max_candidates=3
    )
    assert baseline_changed_gt == baseline
    assert np.array_equal(x_changed_gt, x)
    assert not np.array_equal(ious_changed_gt, ious)
    metas = [GroupMeta('scene{:04d}'.format(i), i, 1, 0) for i in range(300)]
    splits = split_indices(metas)
    assert all(splits.values())
    assert sum(map(len, splits.values())) == 300
    scores = np.asarray([0.0, 2.0, 0.0, 1.0], dtype=np.float32)
    test_ious = np.asarray([0.3, 0.6, 0.6, 0.2], dtype=np.float32)
    selected, baseline_ious, _ = group_decisions(
        scores, test_ious, np.asarray([2, 2]), np.asarray([0, 0]), 0.5
    )
    assert np.allclose(selected, [0.6, 0.2])
    assert np.allclose(baseline_ious, [0.3, 0.6])
    rng = np.random.RandomState(0)
    fit_x = rng.normal(size=(60, len(FEATURE_NAMES))).astype(np.float32)
    fit_y = np.tile(np.asarray([0, 1, 2, 3, 4, 0], dtype=np.int32), 10)
    fit_groups = [6] * 10
    smoke_ranker = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg',
        label_gain=[0, 1, 4, 16, 24], n_estimators=2,
        num_leaves=7, min_child_samples=2, random_state=0,
        n_jobs=1, verbosity=-1,
    )
    smoke_ranker.fit(
        fit_x, fit_y, group=fit_groups,
        feature_name=FEATURE_NAMES,
    )
    assert smoke_ranker.predict(fit_x).shape == (60,)
    print('JOINT_OPTION_RANKER_SELFTEST_PASS features={}'.format(
        len(FEATURE_NAMES)
    ))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('self-test')
    train_parser = sub.add_parser('train')
    train_parser.add_argument('train_dump')
    train_parser.add_argument('output_dir')
    train_parser.add_argument('--max-candidates', type=int, default=8)
    train_parser.add_argument('--num-threads', type=int, default=16)
    eval_parser = sub.add_parser('evaluate')
    eval_parser.add_argument('dump')
    eval_parser.add_argument('model')
    eval_parser.add_argument('lock_json')
    eval_parser.add_argument('output_json')
    oracle_parser = sub.add_parser('oracle')
    oracle_parser.add_argument('dump')
    oracle_parser.add_argument('output_json')
    oracle_parser.add_argument('--max-candidates', type=int, default=8)
    crossfit_parser = sub.add_parser('crossfit-train')
    crossfit_parser.add_argument('train_dump')
    crossfit_parser.add_argument('source_lock_json')
    crossfit_parser.add_argument('output_dir')
    crossfit_parser.add_argument('--num-folds', type=int, default=5)
    crossfit_parser.add_argument('--num-threads', type=int, default=16)
    ensemble_parser = sub.add_parser('ensemble-evaluate')
    ensemble_parser.add_argument('dump')
    ensemble_parser.add_argument('lock_json')
    ensemble_parser.add_argument('output_json')
    binary_parser = sub.add_parser('binary50-train')
    binary_parser.add_argument('train_dump')
    binary_parser.add_argument('output_dir')
    binary_parser.add_argument('--max-candidates', type=int, default=8)
    binary_parser.add_argument('--num-threads', type=int, default=16)
    blend_parser = sub.add_parser('blend-train')
    blend_parser.add_argument('train_dump')
    blend_parser.add_argument('ordinal_model')
    blend_parser.add_argument('ordinal_lock_json')
    blend_parser.add_argument('binary_model')
    blend_parser.add_argument('binary_lock_json')
    blend_parser.add_argument('output_dir')
    blend_eval_parser = sub.add_parser('blend-evaluate')
    blend_eval_parser.add_argument('dump')
    blend_eval_parser.add_argument('lock_json')
    blend_eval_parser.add_argument('output_json')
    pointwise_parser = sub.add_parser('pointwise-train')
    pointwise_parser.add_argument('train_dump')
    pointwise_parser.add_argument('output_dir')
    pointwise_parser.add_argument('--max-candidates', type=int, default=8)
    pointwise_parser.add_argument('--num-threads', type=int, default=16)
    mixed_parser = sub.add_parser('mixed-binary50-train')
    mixed_parser.add_argument('clean_dump')
    mixed_parser.add_argument('augmented_dump')
    mixed_parser.add_argument('output_dir')
    mixed_parser.add_argument('--max-candidates', type=int, default=8)
    mixed_parser.add_argument('--num-threads', type=int, default=16)
    args = parser.parse_args()
    if args.command == 'self-test':
        self_test()
    elif args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'oracle':
        oracle(args)
    elif args.command == 'crossfit-train':
        crossfit_train(args)
    elif args.command == 'ensemble-evaluate':
        ensemble_evaluate(args)
    elif args.command == 'binary50-train':
        binary50_train(args)
    elif args.command == 'blend-train':
        blend_train(args)
    elif args.command == 'blend-evaluate':
        blend_evaluate(args)
    elif args.command == 'pointwise-train':
        pointwise_train(args)
    else:
        mixed_binary50_train(args)


if __name__ == '__main__':
    main()
