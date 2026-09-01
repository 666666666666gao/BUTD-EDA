#!/usr/bin/env python
"""Train a Stage29-to-projected override gate from scene-level OOF predictions."""

import argparse
import gc
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


def scene_bucket(scene_id):
    value = int(hashlib.sha1(scene_id.encode('utf-8')).hexdigest()[:8], 16)
    return value % 100


def build_groups(rows, projected_module, max_candidates):
    projected_groups = []
    iou_groups = []
    baselines = []
    scenes = []
    example_ids = []
    for index, row in enumerate(rows):
        scene = str(row.get('scene_id', ''))
        assert scene, index
        x, ious, baseline = projected_module.row_options(
            row, max_candidates=max_candidates
        )
        projected_groups.append(x)
        iou_groups.append(ious)
        baselines.append(int(baseline))
        scenes.append(scene)
        example_ids.append(int(row.get('example_id', index)))
        if (index + 1) % 1000 == 0:
            print('oof_built_groups={}/{}'.format(index + 1, len(rows)), flush=True)
    return projected_groups, iou_groups, baselines, scenes, example_ids


def materialize(projected_groups, iou_groups, baselines, indices,
                old_feature_indices):
    projected_x = np.concatenate(
        [projected_groups[index] for index in indices], axis=0
    )
    ious = np.concatenate([iou_groups[index] for index in indices], axis=0)
    sizes = np.asarray(
        [len(projected_groups[index]) for index in indices], dtype=np.int32
    )
    baseline_array = np.asarray(
        [baselines[index] for index in indices], dtype=np.int32
    )
    old_x = projected_x[:, old_feature_indices]
    return projected_x, old_x, ious.astype(np.float32), sizes, baseline_array


def make_ranker(config, iterations, random_state, num_threads):
    return lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', label_gain=[0, 1],
        n_estimators=int(iterations), learning_rate=0.05,
        num_leaves=int(config['num_leaves']),
        max_depth=int(config['max_depth']),
        min_child_samples=int(config['min_child_samples']),
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=int(random_state),
        n_jobs=int(num_threads), verbosity=-1,
        deterministic=True, force_col_wise=True,
    )


def choice_indices(scores, sizes, baselines, threshold):
    choices = []
    raw_gaps = []
    cursor = 0
    for size, baseline in zip(sizes, baselines):
        size = int(size)
        baseline = int(baseline)
        group = scores[cursor:cursor + size]
        top = int(np.argmax(group))
        gap = float(group[top] - group[baseline])
        choices.append(cursor + (top if gap >= float(threshold) else baseline))
        raw_gaps.append(gap)
        cursor += size
    assert cursor == len(scores)
    return np.asarray(choices, dtype=np.int64), np.asarray(
        raw_gaps, dtype=np.float32
    )


def normalize_scores(scores, sizes):
    output = np.empty_like(scores, dtype=np.float32)
    cursor = 0
    for size in sizes:
        size = int(size)
        values = scores[cursor:cursor + size].astype(np.float32)
        output[cursor:cursor + size] = (
            values - values.mean()
        ) / (values.std() + 1e-6)
        cursor += size
    return output


def meta_feature_names(projected_names, old_names):
    return (
        ['projected_diff_' + name for name in projected_names]
        + ['stage29_choice_' + name for name in old_names]
        + ['projected_choice_' + name for name in old_names]
        + [
            'stage29_norm_at_stage29_choice',
            'stage29_norm_at_projected_choice',
            'stage29_norm_projected_minus_stage29',
            'projected_norm_at_stage29_choice',
            'projected_norm_at_projected_choice',
            'projected_norm_projected_minus_stage29',
            'stage29_norm_top_gap',
            'projected_norm_top_gap',
            'stage29_raw_gate_margin',
            'projected_raw_gate_margin',
            'strategies_same_option',
            'projected_option_feature_l2_delta',
        ]
    )


def build_meta_features(projected_x, old_x, ious, sizes, baselines,
                        old_scores, projected_scores,
                        old_threshold, projected_threshold,
                        feature_names):
    old_choices, old_raw_gaps = choice_indices(
        old_scores, sizes, baselines, old_threshold
    )
    projected_choices, projected_raw_gaps = choice_indices(
        projected_scores, sizes, baselines, projected_threshold
    )
    old_norm = normalize_scores(old_scores, sizes)
    projected_norm = normalize_scores(projected_scores, sizes)
    old_norm_gaps = []
    projected_norm_gaps = []
    cursor = 0
    for group_index, size in enumerate(sizes):
        size = int(size)
        old_local = int(old_choices[group_index] - cursor)
        projected_local = int(projected_choices[group_index] - cursor)
        old_group = old_norm[cursor:cursor + size]
        projected_group = projected_norm[cursor:cursor + size]
        old_norm_gaps.append(float(old_group.max() - old_group[old_local]))
        projected_norm_gaps.append(
            float(projected_group.max() - projected_group[projected_local])
        )
        cursor += size
    projected_difference = (
        projected_x[projected_choices] - projected_x[old_choices]
    )
    scalars = np.stack([
        old_norm[old_choices],
        old_norm[projected_choices],
        old_norm[projected_choices] - old_norm[old_choices],
        projected_norm[old_choices],
        projected_norm[projected_choices],
        projected_norm[projected_choices] - projected_norm[old_choices],
        np.asarray(old_norm_gaps, dtype=np.float32),
        np.asarray(projected_norm_gaps, dtype=np.float32),
        old_raw_gaps - float(old_threshold),
        projected_raw_gaps - float(projected_threshold),
        (old_choices == projected_choices).astype(np.float32),
        np.linalg.norm(projected_difference, axis=1).astype(np.float32),
    ], axis=1).astype(np.float32)
    meta_x = np.concatenate([
        projected_difference,
        old_x[old_choices],
        old_x[projected_choices],
        scalars,
    ], axis=1).astype(np.float32)
    assert meta_x.shape[1] == len(feature_names), (
        meta_x.shape, len(feature_names)
    )
    return {
        'x': meta_x,
        'old_iou': ious[old_choices].astype(np.float32),
        'projected_iou': ious[projected_choices].astype(np.float32),
        'same_choice': old_choices == projected_choices,
    }


def pair_summary(old_iou, projected_iou, scores=None, threshold=None):
    old_iou = np.asarray(old_iou, dtype=np.float32)
    projected_iou = np.asarray(projected_iou, dtype=np.float32)
    if scores is None:
        override = np.zeros(len(old_iou), dtype=bool)
    else:
        override = np.asarray(scores) >= float(threshold)
    selected = np.where(override, projected_iou, old_iou)
    old_hit25 = old_iou > 0.25
    projected_hit25 = projected_iou > 0.25
    old_hit50 = old_iou > 0.50
    projected_hit50 = projected_iou > 0.50
    return {
        'stage29': metrics(old_iou),
        'projected': metrics(projected_iou),
        'oracle_pair': metrics(np.maximum(old_iou, projected_iou)),
        'selected': metrics(selected),
        'override_ratio': float(override.mean()),
        'fix025_count': int((override & ~old_hit25 & projected_hit25).sum()),
        'break025_count': int((override & old_hit25 & ~projected_hit25).sum()),
        'fix050_count': int((override & ~old_hit50 & projected_hit50).sum()),
        'break050_count': int((override & old_hit50 & ~projected_hit50).sum()),
        'potential_fix050_count': int((~old_hit50 & projected_hit50).sum()),
        'potential_break050_count': int((old_hit50 & ~projected_hit50).sum()),
    }


def choose_threshold(scores, old_iou, projected_iou, preserve_acc025_drop):
    scores = np.asarray(scores, dtype=np.float32)
    quantiles = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 501)))
    thresholds = list(map(float, quantiles))
    thresholds.append(float(scores.max()) + 1.0)
    baseline = metrics(old_iou)
    best = None
    for threshold in thresholds:
        summary = pair_summary(old_iou, projected_iou, scores, threshold)
        result = summary['selected']
        if result['acc025'] + 1e-12 < (
            baseline['acc025'] - float(preserve_acc025_drop)
        ):
            continue
        key = (
            result['acc050'], result['acc025'], result['mean_iou'],
            -summary['override_ratio'],
        )
        item = {'threshold': float(threshold), 'summary': summary}
        if best is None or key > best[0]:
            best = (key, item)
    assert best is not None
    return best[1]


def classifier_configs():
    return (
        {'name': 'conservative', 'num_leaves': 7, 'max_depth': 4,
         'min_child_samples': 80},
        {'name': 'balanced', 'num_leaves': 15, 'max_depth': 6,
         'min_child_samples': 50},
        {'name': 'wide', 'num_leaves': 31, 'max_depth': 8,
         'min_child_samples': 30},
    )


def train(args):
    os.makedirs(args.output_dir, exist_ok=False)
    old_module = load_module(args.old_script, 'crossfit_stage29_module')
    projected_module = load_module(
        args.projected_script, 'crossfit_projected_module'
    )
    old_lock = json.load(open(args.old_lock, 'r', encoding='utf-8'))
    projected_lock = json.load(open(
        args.projected_lock, 'r', encoding='utf-8'
    ))
    assert sha256(args.old_model) == old_lock['model_sha256']
    assert sha256(args.projected_model) == projected_lock['model_sha256']
    assert old_lock['feature_names'] == old_module.FEATURE_NAMES
    assert projected_lock['feature_names'] == projected_module.FEATURE_NAMES
    assert int(old_lock['max_candidates']) == int(
        projected_lock['max_candidates']
    )
    projected_positions = {
        name: index for index, name in enumerate(projected_module.FEATURE_NAMES)
    }
    old_indices = np.asarray([
        projected_positions[name] for name in old_module.FEATURE_NAMES
    ], dtype=np.int64)
    names = meta_feature_names(
        projected_module.FEATURE_NAMES, old_module.FEATURE_NAMES
    )
    rows = torch.load(args.train_dump, map_location='cpu')['rows']
    groups = build_groups(
        rows, projected_module, int(old_lock['max_candidates'])
    )
    projected_groups, iou_groups, baselines, scenes, example_ids = groups
    folds = np.asarray(
        [scene_bucket(scene) % int(args.num_folds) for scene in scenes],
        dtype=np.int32,
    )
    meta_rows = [None] * len(rows)
    fold_receipts = []
    all_indices = np.arange(len(rows), dtype=np.int64)
    for fold in range(int(args.num_folds)):
        held_indices = all_indices[folds == fold].tolist()
        train_indices = all_indices[folds != fold].tolist()
        train_proj, train_old, train_iou, train_sizes, _ = materialize(
            projected_groups, iou_groups, baselines, train_indices, old_indices
        )
        held_proj, held_old, held_iou, held_sizes, held_baselines = materialize(
            projected_groups, iou_groups, baselines, held_indices, old_indices
        )
        train_labels = (train_iou >= 0.50).astype(np.int32)
        old_ranker = make_ranker(
            old_lock['selected_config'], old_lock['best_iteration'],
            old_lock['selected_config_index'], args.num_threads,
        )
        old_ranker.fit(
            train_old, train_labels, group=train_sizes.tolist(),
            feature_name=old_module.FEATURE_NAMES,
            callbacks=[lgb.log_evaluation(0)],
        )
        old_scores = old_ranker.booster_.predict(
            held_old, num_iteration=int(old_lock['best_iteration'])
        ).astype(np.float32)
        del old_ranker, train_old
        gc.collect()
        projected_ranker = make_ranker(
            projected_lock['selected_config'],
            projected_lock['best_iteration'],
            projected_lock['selected_config_index'], args.num_threads,
        )
        projected_ranker.fit(
            train_proj, train_labels, group=train_sizes.tolist(),
            feature_name=projected_module.FEATURE_NAMES,
            callbacks=[lgb.log_evaluation(0)],
        )
        projected_scores = projected_ranker.booster_.predict(
            held_proj, num_iteration=int(projected_lock['best_iteration'])
        ).astype(np.float32)
        del projected_ranker, train_proj, train_iou, train_labels
        gc.collect()
        meta = build_meta_features(
            held_proj, held_old, held_iou, held_sizes, held_baselines,
            old_scores, projected_scores,
            float(old_lock['gate']['threshold']),
            float(projected_lock['gate']['threshold']), names,
        )
        for local_index, global_index in enumerate(held_indices):
            meta_rows[global_index] = (
                meta['x'][local_index],
                float(meta['old_iou'][local_index]),
                float(meta['projected_iou'][local_index]),
            )
        fold_receipts.append({
            'fold': fold,
            'train_groups': len(train_indices),
            'heldout_groups': len(held_indices),
            'heldout_pair': pair_summary(
                meta['old_iou'], meta['projected_iou']
            ),
        })
        print('OOF_FOLD_COMPLETE', json.dumps(fold_receipts[-1], sort_keys=True), flush=True)
        del held_proj, held_old, held_iou, held_sizes, held_baselines
        del old_scores, projected_scores, meta
        gc.collect()
    assert all(item is not None for item in meta_rows)
    meta_x = np.stack([item[0] for item in meta_rows]).astype(np.float32)
    old_iou = np.asarray([item[1] for item in meta_rows], dtype=np.float32)
    projected_iou = np.asarray(
        [item[2] for item in meta_rows], dtype=np.float32
    )
    buckets = np.asarray([scene_bucket(scene) for scene in scenes], dtype=np.int32)
    split_masks = {
        'train': buckets < 70,
        'dev': (buckets >= 70) & (buckets < 85),
        'test': buckets >= 85,
    }
    conflict = (old_iou > 0.50) != (projected_iou > 0.50)
    labels = (projected_iou > 0.50).astype(np.int32)
    train_conflict = np.nonzero(split_masks['train'] & conflict)[0]
    dev_conflict = np.nonzero(split_masks['dev'] & conflict)[0]
    assert len(np.unique(labels[train_conflict])) == 2
    assert len(np.unique(labels[dev_conflict])) == 2
    positive = max(1, int(labels[train_conflict].sum()))
    negative = max(1, len(train_conflict) - positive)
    weights = np.where(
        labels[train_conflict] > 0,
        len(train_conflict) / (2.0 * positive),
        len(train_conflict) / (2.0 * negative),
    ).astype(np.float32)
    candidates = []
    for config_index, config in enumerate(classifier_configs()):
        classifier = lgb.LGBMClassifier(
            objective='binary', n_estimators=1000, learning_rate=0.03,
            num_leaves=config['num_leaves'], max_depth=config['max_depth'],
            min_child_samples=config['min_child_samples'],
            subsample=0.8, subsample_freq=1, colsample_bytree=0.75,
            reg_lambda=2.0, random_state=config_index,
            n_jobs=args.num_threads, verbosity=-1,
            deterministic=True, force_col_wise=True,
        )
        classifier.fit(
            meta_x[train_conflict], labels[train_conflict],
            sample_weight=weights,
            eval_set=[(meta_x[dev_conflict], labels[dev_conflict])],
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(20)],
            feature_name=names,
        )
        iteration = int(classifier.best_iteration_ or classifier.n_estimators)
        all_scores = classifier.booster_.predict(
            meta_x, num_iteration=iteration
        ).astype(np.float32)
        gate = choose_threshold(
            all_scores[split_masks['dev']],
            old_iou[split_masks['dev']],
            projected_iou[split_masks['dev']],
            args.preserve_acc025_drop,
        )
        candidates.append({
            'config': config,
            'iteration': iteration,
            'threshold': gate['threshold'],
            'dev': gate['summary'],
            'test': pair_summary(
                old_iou[split_masks['test']],
                projected_iou[split_masks['test']],
                all_scores[split_masks['test']], gate['threshold'],
            ),
            'booster': classifier.booster_,
        })
        print('OOF_GATE_CANDIDATE', json.dumps({
            key: value for key, value in candidates[-1].items()
            if key != 'booster'
        }, sort_keys=True), flush=True)
    selected_index = max(range(len(candidates)), key=lambda index: (
        candidates[index]['dev']['selected']['acc050'],
        candidates[index]['dev']['selected']['acc025'],
        candidates[index]['dev']['selected']['mean_iou'],
        -candidates[index]['dev']['override_ratio'],
    ))
    selected = candidates[selected_index]
    model_path = os.path.join(args.output_dir, 'oof_override_gate.txt')
    selected['booster'].save_model(
        model_path, num_iteration=selected['iteration']
    )
    oof_path = os.path.join(args.output_dir, 'oof_meta_features.npz')
    with open(oof_path + '.tmp', 'wb') as handle:
        np.savez(
            handle, x=meta_x, old_iou=old_iou,
            projected_iou=projected_iou, buckets=buckets,
            example_ids=np.asarray(example_ids, dtype=np.int64),
        )
    os.replace(oof_path + '.tmp', oof_path)
    lock = {
        'protocol': 'five_fold_scene_oof_base_predictions_meta_70_15_15',
        'train_dump': os.path.abspath(args.train_dump),
        'train_dump_sha256': sha256(args.train_dump),
        'script_sha256': sha256(os.path.abspath(__file__)),
        'num_folds': int(args.num_folds),
        'fold_receipts': fold_receipts,
        'feature_names': names,
        'model_path': os.path.abspath(model_path),
        'model_sha256': sha256(model_path),
        'oof_path': os.path.abspath(oof_path),
        'oof_sha256': sha256(oof_path),
        'selected_config_index': selected_index,
        'selected_config': selected['config'],
        'best_iteration': selected['iteration'],
        'threshold': selected['threshold'],
        'preserve_acc025_drop': float(args.preserve_acc025_drop),
        'split_group_counts': {
            key: int(mask.sum()) for key, mask in split_masks.items()
        },
        'conflict_counts': {
            key: int((mask & conflict).sum())
            for key, mask in split_masks.items()
        },
        'internal': {
            'train': pair_summary(
                old_iou[split_masks['train']],
                projected_iou[split_masks['train']],
            ),
            'dev': selected['dev'],
            'test': selected['test'],
        },
        'candidates': [
            {key: value for key, value in item.items() if key != 'booster'}
            for item in candidates
        ],
    }
    for key in (
        'old_script', 'projected_script', 'old_model', 'projected_model',
        'old_lock', 'projected_lock',
    ):
        path = os.path.abspath(getattr(args, key))
        lock[key] = path
        lock[key + '_sha256'] = sha256(path)
    lock_path = os.path.join(args.output_dir, 'locked_oof_override_gate.json')
    with open(lock_path, 'w', encoding='utf-8') as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
    print(json.dumps(lock, indent=2, sort_keys=True))


def evaluate(args):
    lock = json.load(open(args.lock_json, 'r', encoding='utf-8'))
    assert sha256(args.model) == lock['model_sha256']
    for key in (
        'old_script', 'projected_script', 'old_model', 'projected_model',
        'old_lock', 'projected_lock',
    ):
        assert sha256(lock[key]) == lock[key + '_sha256'], key
    old_module = load_module(lock['old_script'], 'eval_stage29_module')
    projected_module = load_module(
        lock['projected_script'], 'eval_projected_module'
    )
    old_policy = json.load(open(lock['old_lock'], 'r', encoding='utf-8'))
    projected_policy = json.load(open(
        lock['projected_lock'], 'r', encoding='utf-8'
    ))
    positions = {
        name: index for index, name in enumerate(projected_module.FEATURE_NAMES)
    }
    old_indices = np.asarray(
        [positions[name] for name in old_module.FEATURE_NAMES], dtype=np.int64
    )
    rows = torch.load(args.dump, map_location='cpu')['rows']
    groups = build_groups(
        rows, projected_module, int(old_policy['max_candidates'])
    )
    projected_groups, iou_groups, baselines, _, _ = groups
    indices = list(range(len(rows)))
    projected_x, old_x, ious, sizes, baseline_array = materialize(
        projected_groups, iou_groups, baselines, indices, old_indices
    )
    old_booster = lgb.Booster(model_file=lock['old_model'])
    projected_booster = lgb.Booster(model_file=lock['projected_model'])
    old_scores = old_booster.predict(
        old_x, num_iteration=int(old_policy['best_iteration'])
    ).astype(np.float32)
    projected_scores = projected_booster.predict(
        projected_x, num_iteration=int(projected_policy['best_iteration'])
    ).astype(np.float32)
    names = meta_feature_names(
        projected_module.FEATURE_NAMES, old_module.FEATURE_NAMES
    )
    assert names == lock['feature_names']
    meta = build_meta_features(
        projected_x, old_x, ious, sizes, baseline_array,
        old_scores, projected_scores,
        float(old_policy['gate']['threshold']),
        float(projected_policy['gate']['threshold']), names,
    )
    classifier = lgb.Booster(model_file=args.model)
    scores = classifier.predict(
        meta['x'], num_iteration=int(lock['best_iteration'])
    )
    result = pair_summary(
        meta['old_iou'], meta['projected_iou'],
        scores, float(lock['threshold']),
    )
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


def add_stack_args(parser):
    parser.add_argument('old_script')
    parser.add_argument('projected_script')
    parser.add_argument('old_model')
    parser.add_argument('projected_model')
    parser.add_argument('old_lock')
    parser.add_argument('projected_lock')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    train_parser = sub.add_parser('train')
    train_parser.add_argument('train_dump')
    add_stack_args(train_parser)
    train_parser.add_argument('output_dir')
    train_parser.add_argument('--num-folds', type=int, default=5)
    train_parser.add_argument('--num-threads', type=int, default=32)
    train_parser.add_argument('--preserve-acc025-drop', type=float, default=0.0)
    eval_parser = sub.add_parser('evaluate')
    eval_parser.add_argument('dump')
    eval_parser.add_argument('model')
    eval_parser.add_argument('lock_json')
    eval_parser.add_argument('output_json')
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        raise AssertionError(args.command)


if __name__ == '__main__':
    main()
