#!/usr/bin/env python
"""Train a conservative Stage29-to-semantic override gate.

Both option rankers are frozen.  Gate fitting uses only scene buckets 85--89,
gate/config selection uses 90--94, and 95--99 is an untouched internal test.
All three ranges are outside the base rankers' bucket-0--84 fit/selection set.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metrics(ious):
    values = np.asarray(ious, dtype=np.float32)
    return {
        'count': int(len(values)),
        'mean_iou': float(values.mean()) if len(values) else 0.0,
        'acc025': float((values > 0.25).mean()) if len(values) else 0.0,
        'acc050': float((values > 0.50).mean()) if len(values) else 0.0,
    }


def load_stack(old_script, semantic_script, old_model, old_lock,
               semantic_model, semantic_lock):
    old_module = load_module(old_script, 'stage29_option_ranker_module')
    semantic_module = load_module(
        semantic_script, 'stage74_semantic_ranker_module'
    )
    old_policy = json.load(open(old_lock, 'r', encoding='utf-8'))
    semantic_policy = json.load(open(semantic_lock, 'r', encoding='utf-8'))
    assert old_policy['feature_names'] == old_module.FEATURE_NAMES
    assert semantic_policy['feature_names'] == semantic_module.FEATURE_NAMES
    assert int(old_policy['max_candidates']) == int(
        semantic_policy['max_candidates']
    )
    semantic_feature_position = {
        name: index for index, name in enumerate(semantic_module.FEATURE_NAMES)
    }
    old_feature_indices = np.asarray([
        semantic_feature_position[name] for name in old_module.FEATURE_NAMES
    ], dtype=np.int64)
    return {
        'old_module': old_module,
        'semantic_module': semantic_module,
        'old_policy': old_policy,
        'semantic_policy': semantic_policy,
        'old_booster': lgb.Booster(model_file=old_model),
        'semantic_booster': lgb.Booster(model_file=semantic_model),
        'old_feature_indices': old_feature_indices,
        'max_candidates': int(old_policy['max_candidates']),
    }


def strategy_choice_indices(scores, group_sizes, baselines, threshold):
    choices = []
    score_gaps = []
    cursor = 0
    for size, baseline in zip(group_sizes, baselines):
        size = int(size)
        baseline = int(baseline)
        group_scores = scores[cursor:cursor + size]
        model_choice = int(np.argmax(group_scores))
        gap = float(group_scores[model_choice] - group_scores[baseline])
        local_choice = model_choice if gap >= float(threshold) else baseline
        choices.append(cursor + local_choice)
        score_gaps.append(gap)
        cursor += size
    assert cursor == len(scores)
    return np.asarray(choices, dtype=np.int64), np.asarray(
        score_gaps, dtype=np.float32
    )


def build_pair_features(rows, stack, require_scene):
    semantic_group_features = []
    group_ious = []
    baselines = []
    group_sizes = []
    scene_ids = []
    example_ids = []
    semantic_module = stack['semantic_module']
    for index, row in enumerate(rows):
        scene_id = str(row.get('scene_id', ''))
        if require_scene:
            assert scene_id, ('missing scene_id', index)
        x, ious, baseline = semantic_module.row_options(
            row, max_candidates=stack['max_candidates']
        )
        semantic_group_features.append(x)
        group_ious.append(ious)
        baselines.append(int(baseline))
        group_sizes.append(len(x))
        scene_ids.append(scene_id)
        example_ids.append(int(row.get('example_id', index)))
        if (index + 1) % 1000 == 0:
            print('pair_groups={}/{}'.format(index + 1, len(rows)), flush=True)

    semantic_x = np.concatenate(semantic_group_features, axis=0)
    ious = np.concatenate(group_ious, axis=0)
    old_x = semantic_x[:, stack['old_feature_indices']]
    old_scores = stack['old_booster'].predict(
        old_x, num_iteration=int(stack['old_policy']['best_iteration'])
    ).astype(np.float32)
    semantic_scores = stack['semantic_booster'].predict(
        semantic_x,
        num_iteration=int(stack['semantic_policy']['best_iteration']),
    ).astype(np.float32)
    old_choices, old_gaps = strategy_choice_indices(
        old_scores, group_sizes, baselines,
        float(stack['old_policy']['gate']['threshold']),
    )
    semantic_choices, semantic_gaps = strategy_choice_indices(
        semantic_scores, group_sizes, baselines,
        float(stack['semantic_policy']['gate']['threshold']),
    )
    old_ious = ious[old_choices]
    semantic_ious = ious[semantic_choices]

    old_at_old = old_scores[old_choices]
    old_at_semantic = old_scores[semantic_choices]
    semantic_at_old = semantic_scores[old_choices]
    semantic_at_semantic = semantic_scores[semantic_choices]
    semantic_difference = (
        semantic_x[semantic_choices] - semantic_x[old_choices]
    )
    old_absolute = old_x[old_choices]
    semantic_absolute_old_schema = old_x[semantic_choices]
    scalar_features = np.stack([
        old_at_old,
        old_at_semantic,
        old_at_semantic - old_at_old,
        semantic_at_old,
        semantic_at_semantic,
        semantic_at_semantic - semantic_at_old,
        old_gaps,
        semantic_gaps,
        old_gaps - float(stack['old_policy']['gate']['threshold']),
        semantic_gaps - float(stack['semantic_policy']['gate']['threshold']),
        (old_choices == semantic_choices).astype(np.float32),
        np.linalg.norm(semantic_difference, axis=1).astype(np.float32),
    ], axis=1).astype(np.float32)
    pair_x = np.concatenate([
        semantic_difference,
        old_absolute,
        semantic_absolute_old_schema,
        scalar_features,
    ], axis=1).astype(np.float32)
    pair_feature_names = (
        ['semantic_diff_' + name for name in stack['semantic_module'].FEATURE_NAMES]
        + ['stage29_choice_' + name for name in stack['old_module'].FEATURE_NAMES]
        + ['semantic_choice_' + name for name in stack['old_module'].FEATURE_NAMES]
        + [
            'stage29_score_at_stage29_choice',
            'stage29_score_at_semantic_choice',
            'stage29_score_semantic_minus_stage29',
            'semantic_score_at_stage29_choice',
            'semantic_score_at_semantic_choice',
            'semantic_score_semantic_minus_stage29',
            'stage29_model_gap',
            'semantic_model_gap',
            'stage29_gate_margin',
            'semantic_gate_margin',
            'strategies_same_option',
            'semantic_option_feature_l2_delta',
        ]
    )
    assert pair_x.shape[1] == len(pair_feature_names), (
        pair_x.shape, len(pair_feature_names)
    )
    return {
        'x': pair_x,
        'feature_names': pair_feature_names,
        'old_ious': old_ious.astype(np.float32),
        'semantic_ious': semantic_ious.astype(np.float32),
        'old_choices': old_choices,
        'semantic_choices': semantic_choices,
        'scene_ids': scene_ids,
        'example_ids': example_ids,
    }


def pair_summary(data, selected=None, override_mask=None):
    old_ious = data['old_ious']
    semantic_ious = data['semantic_ious']
    if selected is None:
        selected = old_ious
    if override_mask is None:
        override_mask = np.zeros(len(old_ious), dtype=bool)
    old_hit50 = old_ious > 0.50
    semantic_hit50 = semantic_ious > 0.50
    old_hit25 = old_ious > 0.25
    semantic_hit25 = semantic_ious > 0.25
    different_choice = data['old_choices'] != data['semantic_choices']
    return {
        'stage29': metrics(old_ious),
        'semantic': metrics(semantic_ious),
        'oracle_pair': metrics(np.maximum(old_ious, semantic_ious)),
        'selected': metrics(selected),
        'different_choice_ratio': float(different_choice.mean()),
        'override_ratio': float((override_mask & different_choice).mean()),
        'potential_fix050_count': int((~old_hit50 & semantic_hit50).sum()),
        'potential_break050_count': int((old_hit50 & ~semantic_hit50).sum()),
        'potential_fix025_count': int((~old_hit25 & semantic_hit25).sum()),
        'potential_break025_count': int((old_hit25 & ~semantic_hit25).sum()),
        'applied_fix050_count': int(
            (override_mask & ~old_hit50 & semantic_hit50).sum()
        ),
        'applied_break050_count': int(
            (override_mask & old_hit50 & ~semantic_hit50).sum()
        ),
        'applied_fix025_count': int(
            (override_mask & ~old_hit25 & semantic_hit25).sum()
        ),
        'applied_break025_count': int(
            (override_mask & old_hit25 & ~semantic_hit25).sum()
        ),
    }


def scene_bucket(scene_id):
    value = int(hashlib.sha1(scene_id.encode('utf-8')).hexdigest()[:8], 16)
    return value % 100


def choose_threshold(scores, data, mask, preserve_acc025_drop=0.0):
    scores = np.asarray(scores, dtype=np.float32)
    indices = np.nonzero(mask)[0]
    old_ious = data['old_ious'][indices]
    semantic_ious = data['semantic_ious'][indices]
    candidate_scores = scores[indices][
        data['old_choices'][indices] != data['semantic_choices'][indices]
    ]
    if len(candidate_scores):
        quantiles = np.unique(np.quantile(
            candidate_scores, np.linspace(0.0, 1.0, 401)
        ))
        thresholds = list(map(float, quantiles))
        thresholds.append(float(candidate_scores.max()) + 1.0)
    else:
        thresholds = [1.0]
    baseline = metrics(old_ious)
    best = None
    for threshold in thresholds:
        override = scores[indices] >= threshold
        selected = np.where(override, semantic_ious, old_ious)
        result = metrics(selected)
        if result['acc025'] + 1e-12 < (
            baseline['acc025'] - float(preserve_acc025_drop)
        ):
            continue
        actual_override = override & (
            data['old_choices'][indices] != data['semantic_choices'][indices]
        )
        key = (
            result['acc050'], result['acc025'], result['mean_iou'],
            -float(actual_override.mean()),
        )
        item = {
            'threshold': float(threshold),
            'selected': result,
            'stage29': baseline,
            'override_ratio': float(actual_override.mean()),
        }
        if best is None or key > best[0]:
            best = (key, item)
    assert best is not None
    return best[1]


def mask_for_buckets(data, start, end):
    buckets = np.asarray([scene_bucket(scene) for scene in data['scene_ids']])
    return (buckets >= int(start)) & (buckets <= int(end))


def train_gate(args):
    os.makedirs(args.output_dir, exist_ok=False)
    payload = torch.load(args.train_dump, map_location='cpu')
    assert args.meta_train_start <= args.meta_train_end
    assert args.meta_train_end < args.meta_dev_start <= args.meta_dev_end
    assert args.meta_dev_end < args.meta_test_start <= args.meta_test_end
    rows = [
        row for row in payload['rows']
        if args.meta_train_start
        <= scene_bucket(str(row['scene_id']))
        <= args.meta_test_end
    ]
    stack = load_stack(
        args.old_script, args.semantic_script,
        args.old_model, args.old_lock,
        args.semantic_model, args.semantic_lock,
    )
    data = build_pair_features(rows, stack, require_scene=True)
    split_masks = {
        'train': mask_for_buckets(
            data, args.meta_train_start, args.meta_train_end
        ),
        'dev': mask_for_buckets(
            data, args.meta_dev_start, args.meta_dev_end
        ),
        'test': mask_for_buckets(
            data, args.meta_test_start, args.meta_test_end
        ),
    }
    assert all(mask.any() for mask in split_masks.values())
    assert not np.any(split_masks['train'] & split_masks['dev'])
    assert not np.any(split_masks['train'] & split_masks['test'])
    assert not np.any(split_masks['dev'] & split_masks['test'])
    old_hit50 = data['old_ious'] > 0.50
    semantic_hit50 = data['semantic_ious'] > 0.50
    conflict = old_hit50 != semantic_hit50
    labels = semantic_hit50.astype(np.int32)
    configs = (
        {'name': 'conservative', 'num_leaves': 7, 'max_depth': 4,
         'min_child_samples': 50},
        {'name': 'balanced', 'num_leaves': 15, 'max_depth': 6,
         'min_child_samples': 30},
        {'name': 'wide', 'num_leaves': 31, 'max_depth': 8,
         'min_child_samples': 20},
    )
    train_indices = np.nonzero(split_masks['train'] & conflict)[0]
    dev_indices = np.nonzero(split_masks['dev'] & conflict)[0]
    assert len(np.unique(labels[train_indices])) == 2
    assert len(np.unique(labels[dev_indices])) == 2
    positive = max(1, int(labels[train_indices].sum()))
    negative = max(1, int(len(train_indices) - positive))
    train_weights = np.where(
        labels[train_indices] > 0,
        len(train_indices) / (2.0 * positive),
        len(train_indices) / (2.0 * negative),
    ).astype(np.float32)
    candidates = []
    for config_index, config in enumerate(configs):
        model = lgb.LGBMClassifier(
            objective='binary', metric='binary_logloss',
            n_estimators=800, learning_rate=0.03,
            num_leaves=config['num_leaves'],
            max_depth=config['max_depth'],
            min_child_samples=config['min_child_samples'],
            subsample=0.8, subsample_freq=1, colsample_bytree=0.75,
            reg_lambda=2.0, random_state=config_index,
            n_jobs=args.num_threads, verbosity=-1,
        )
        model.fit(
            data['x'][train_indices], labels[train_indices],
            sample_weight=train_weights,
            eval_set=[(data['x'][dev_indices], labels[dev_indices])],
            callbacks=[lgb.early_stopping(60), lgb.log_evaluation(20)],
            feature_name=data['feature_names'],
        )
        iteration = int(model.best_iteration_ or model.n_estimators)
        dev_scores = model.booster_.predict(
            data['x'], num_iteration=iteration
        )
        gate = choose_threshold(
            dev_scores, data, split_masks['dev'],
            preserve_acc025_drop=args.preserve_acc025_drop,
        )
        candidates.append({
            'config': config,
            'iteration': iteration,
            'gate': gate,
            'booster': model.booster_,
        })
        print('gate_candidate', json.dumps({
            key: value for key, value in candidates[-1].items()
            if key != 'booster'
        }, sort_keys=True), flush=True)
    selected_index = max(range(len(candidates)), key=lambda index: (
        candidates[index]['gate']['selected']['acc050'],
        candidates[index]['gate']['selected']['acc025'],
        candidates[index]['gate']['selected']['mean_iou'],
        -candidates[index]['gate']['override_ratio'],
    ))
    selected = candidates[selected_index]
    all_scores = selected['booster'].predict(
        data['x'], num_iteration=selected['iteration']
    )
    threshold = float(selected['gate']['threshold'])
    internal = {}
    for split, mask in split_masks.items():
        override = all_scores >= threshold
        chosen = np.where(
            override[mask], data['semantic_ious'][mask], data['old_ious'][mask]
        )
        subset = {
            key: (value[mask] if isinstance(value, np.ndarray) else value)
            for key, value in data.items()
        }
        internal[split] = pair_summary(
            subset, selected=chosen, override_mask=override[mask]
        )

    model_path = os.path.join(args.output_dir, 'semantic_override_gate.txt')
    selected['booster'].save_model(
        model_path, num_iteration=selected['iteration']
    )
    lock = {
        'protocol': 'scene_bucket_disjoint_semantic_override_gate',
        'train_dump': os.path.abspath(args.train_dump),
        'train_dump_sha256': sha256(args.train_dump),
        'script_sha256': sha256(os.path.abspath(__file__)),
        'old_script': os.path.abspath(args.old_script),
        'old_script_sha256': sha256(args.old_script),
        'semantic_script': os.path.abspath(args.semantic_script),
        'semantic_script_sha256': sha256(args.semantic_script),
        'old_model': os.path.abspath(args.old_model),
        'old_model_sha256': sha256(args.old_model),
        'old_lock': os.path.abspath(args.old_lock),
        'old_lock_sha256': sha256(args.old_lock),
        'semantic_model': os.path.abspath(args.semantic_model),
        'semantic_model_sha256': sha256(args.semantic_model),
        'semantic_lock': os.path.abspath(args.semantic_lock),
        'semantic_lock_sha256': sha256(args.semantic_lock),
        'model_path': os.path.abspath(model_path),
        'model_sha256': sha256(model_path),
        'feature_names': data['feature_names'],
        'best_iteration': selected['iteration'],
        'threshold': threshold,
        'selected_config_index': selected_index,
        'selected_config': selected['config'],
        'preserve_acc025_drop': float(args.preserve_acc025_drop),
        'split_bucket_ranges': {
            'train': [args.meta_train_start, args.meta_train_end],
            'dev': [args.meta_dev_start, args.meta_dev_end],
            'test': [args.meta_test_start, args.meta_test_end],
        },
        'split_group_counts': {
            key: int(mask.sum()) for key, mask in split_masks.items()
        },
        'conflict_counts': {
            key: int((mask & conflict).sum())
            for key, mask in split_masks.items()
        },
        'internal': internal,
        'candidates': [
            {key: value for key, value in item.items() if key != 'booster'}
            for item in candidates
        ],
    }
    lock_path = os.path.join(args.output_dir, 'locked_semantic_override_gate.json')
    with open(lock_path, 'w', encoding='utf-8') as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
    print(json.dumps(lock, indent=2, sort_keys=True))


def load_locked_stack(lock):
    hashes = {
        'old_script': 'old_script_sha256',
        'semantic_script': 'semantic_script_sha256',
        'old_model': 'old_model_sha256',
        'old_lock': 'old_lock_sha256',
        'semantic_model': 'semantic_model_sha256',
        'semantic_lock': 'semantic_lock_sha256',
        'model_path': 'model_sha256',
    }
    for key, hash_key in hashes.items():
        assert sha256(lock[key]) == lock[hash_key], key
    stack = load_stack(
        lock['old_script'], lock['semantic_script'],
        lock['old_model'], lock['old_lock'],
        lock['semantic_model'], lock['semantic_lock'],
    )
    return stack


def evaluate_gate(args):
    lock = json.load(open(args.lock_json, 'r', encoding='utf-8'))
    assert sha256(args.model) == lock['model_sha256']
    assert os.path.abspath(args.model) == lock['model_path']
    stack = load_locked_stack(lock)
    rows = torch.load(args.dump, map_location='cpu')['rows']
    data = build_pair_features(rows, stack, require_scene=False)
    assert data['feature_names'] == lock['feature_names']
    booster = lgb.Booster(model_file=args.model)
    scores = booster.predict(
        data['x'], num_iteration=int(lock['best_iteration'])
    )
    override = scores >= float(lock['threshold'])
    selected = np.where(override, data['semantic_ious'], data['old_ious'])
    result = pair_summary(data, selected=selected, override_mask=override)
    result.update({
        'dump': os.path.abspath(args.dump),
        'dump_sha256': sha256(args.dump),
        'model_sha256': sha256(args.model),
        'lock_sha256': sha256(args.lock_json),
        'threshold': float(lock['threshold']),
        'goal_achieved_offline': bool(
            result['selected']['acc025'] > 0.5391
            and result['selected']['acc050'] > 0.4241
        ),
        'diagnostic_only': True,
    })
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


def diagnose(args):
    stack = load_stack(
        args.old_script, args.semantic_script,
        args.old_model, args.old_lock,
        args.semantic_model, args.semantic_lock,
    )
    rows = torch.load(args.dump, map_location='cpu')['rows']
    data = build_pair_features(rows, stack, require_scene=False)
    result = pair_summary(data)
    result.update({
        'dump': os.path.abspath(args.dump),
        'dump_sha256': sha256(args.dump),
        'diagnostic_only': True,
    })
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


def add_stack_args(parser):
    parser.add_argument('old_script')
    parser.add_argument('semantic_script')
    parser.add_argument('old_model')
    parser.add_argument('old_lock')
    parser.add_argument('semantic_model')
    parser.add_argument('semantic_lock')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    diagnose_parser = sub.add_parser('diagnose')
    diagnose_parser.add_argument('dump')
    add_stack_args(diagnose_parser)
    diagnose_parser.add_argument('output_json')
    train_parser = sub.add_parser('train')
    train_parser.add_argument('train_dump')
    add_stack_args(train_parser)
    train_parser.add_argument('output_dir')
    train_parser.add_argument('--num-threads', type=int, default=32)
    train_parser.add_argument('--preserve-acc025-drop', type=float, default=0.0)
    train_parser.add_argument('--meta-train-start', type=int, default=85)
    train_parser.add_argument('--meta-train-end', type=int, default=89)
    train_parser.add_argument('--meta-dev-start', type=int, default=90)
    train_parser.add_argument('--meta-dev-end', type=int, default=94)
    train_parser.add_argument('--meta-test-start', type=int, default=95)
    train_parser.add_argument('--meta-test-end', type=int, default=99)
    evaluate_parser = sub.add_parser('evaluate')
    evaluate_parser.add_argument('dump')
    evaluate_parser.add_argument('model')
    evaluate_parser.add_argument('lock_json')
    evaluate_parser.add_argument('output_json')
    args = parser.parse_args()
    if args.command == 'diagnose':
        diagnose(args)
    elif args.command == 'train':
        train_gate(args)
    elif args.command == 'evaluate':
        evaluate_gate(args)
    else:
        raise AssertionError(args.command)


if __name__ == '__main__':
    main()
