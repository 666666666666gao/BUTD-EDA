#!/usr/bin/env python
"""Lock a conservative blend of Stage29, projected, and raw-PCA rankers."""

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


def scene_bucket(scene_id):
    value = int(hashlib.sha1(scene_id.encode('utf-8')).hexdigest()[:8], 16)
    return value % 100


def load_policy(path):
    return json.load(open(path, 'r', encoding='utf-8'))


def load_stack(args_or_lock):
    paths = args_or_lock
    old_module = load_module(paths.old_script, 'blend_stage29_module')
    projected_module = load_module(
        paths.projected_script, 'blend_projected_module'
    )
    raw_module = load_module(paths.raw_script, 'blend_rawpca_module')
    old_lock = load_policy(paths.old_lock)
    projected_lock = load_policy(paths.projected_lock)
    raw_lock = load_policy(paths.raw_lock)
    assert old_lock['feature_names'] == old_module.FEATURE_NAMES
    assert projected_lock['feature_names'] == projected_module.FEATURE_NAMES
    assert raw_lock['feature_names'] == raw_module.FEATURE_NAMES
    assert len({
        int(old_lock['max_candidates']),
        int(projected_lock['max_candidates']),
        int(raw_lock['max_candidates']),
    }) == 1
    raw_positions = {
        name: index for index, name in enumerate(raw_module.FEATURE_NAMES)
    }
    old_indices = np.asarray(
        [raw_positions[name] for name in old_module.FEATURE_NAMES],
        dtype=np.int64,
    )
    projected_indices = np.asarray(
        [raw_positions[name] for name in projected_module.FEATURE_NAMES],
        dtype=np.int64,
    )
    return {
        'old_module': old_module,
        'projected_module': projected_module,
        'raw_module': raw_module,
        'old_lock': old_lock,
        'projected_lock': projected_lock,
        'raw_lock': raw_lock,
        'old_booster': lgb.Booster(model_file=paths.old_model),
        'projected_booster': lgb.Booster(model_file=paths.projected_model),
        'raw_booster': lgb.Booster(model_file=paths.raw_model),
        'old_indices': old_indices,
        'projected_indices': projected_indices,
        'max_candidates': int(old_lock['max_candidates']),
    }


def strategy_choices(scores, group_sizes, baselines, threshold):
    choices = []
    cursor = 0
    for size, baseline in zip(group_sizes, baselines):
        size = int(size)
        baseline = int(baseline)
        group = scores[cursor:cursor + size]
        top = int(np.argmax(group))
        gap = float(group[top] - group[baseline])
        choices.append(cursor + (top if gap >= float(threshold) else baseline))
        cursor += size
    assert cursor == len(scores)
    return np.asarray(choices, dtype=np.int64)


def normalize_group_scores(scores, group_sizes):
    normalized = np.empty_like(scores, dtype=np.float32)
    cursor = 0
    for size in group_sizes:
        size = int(size)
        values = scores[cursor:cursor + size].astype(np.float32)
        normalized[cursor:cursor + size] = (
            values - values.mean()
        ) / (values.std() + 1e-6)
        cursor += size
    assert cursor == len(scores)
    return normalized


def build_data(rows, stack, require_scene):
    feature_groups = []
    iou_groups = []
    baselines = []
    group_sizes = []
    scenes = []
    for index, row in enumerate(rows):
        scene = str(row.get('scene_id', ''))
        if require_scene:
            assert scene, index
        x, ious, baseline = stack['raw_module'].row_options(
            row, max_candidates=stack['max_candidates']
        )
        feature_groups.append(x)
        iou_groups.append(ious)
        baselines.append(int(baseline))
        group_sizes.append(len(x))
        scenes.append(scene)
        if (index + 1) % 1000 == 0:
            print('blend_groups={}/{}'.format(index + 1, len(rows)), flush=True)
    raw_x = np.concatenate(feature_groups, axis=0)
    ious = np.concatenate(iou_groups, axis=0).astype(np.float32)
    scores = {
        'stage29': stack['old_booster'].predict(
            raw_x[:, stack['old_indices']],
            num_iteration=int(stack['old_lock']['best_iteration']),
        ).astype(np.float32),
        'projected': stack['projected_booster'].predict(
            raw_x[:, stack['projected_indices']],
            num_iteration=int(stack['projected_lock']['best_iteration']),
        ).astype(np.float32),
        'rawpca': stack['raw_booster'].predict(
            raw_x, num_iteration=int(stack['raw_lock']['best_iteration']),
        ).astype(np.float32),
    }
    choices = {
        'stage29': strategy_choices(
            scores['stage29'], group_sizes, baselines,
            float(stack['old_lock']['gate']['threshold']),
        ),
        'projected': strategy_choices(
            scores['projected'], group_sizes, baselines,
            float(stack['projected_lock']['gate']['threshold']),
        ),
        'rawpca': strategy_choices(
            scores['rawpca'], group_sizes, baselines,
            float(stack['raw_lock']['gate']['threshold']),
        ),
    }
    normalized = {
        name: normalize_group_scores(values, group_sizes)
        for name, values in scores.items()
    }
    return {
        'ious': ious,
        'group_sizes': np.asarray(group_sizes, dtype=np.int32),
        'baselines': np.asarray(baselines, dtype=np.int32),
        'scenes': scenes,
        'scores': scores,
        'normalized_scores': normalized,
        'choices': choices,
    }


def blend_top_and_gaps(data, weights):
    blended = sum(
        float(weights[name]) * data['normalized_scores'][name]
        for name in ('stage29', 'projected', 'rawpca')
    )
    top_choices = []
    gaps = []
    cursor = 0
    for group_index, size in enumerate(data['group_sizes']):
        size = int(size)
        group = blended[cursor:cursor + size]
        top_local = int(np.argmax(group))
        top = cursor + top_local
        default = int(data['choices']['stage29'][group_index])
        top_choices.append(top)
        gaps.append(float(blended[top] - blended[default]))
        cursor += size
    return np.asarray(top_choices, dtype=np.int64), np.asarray(
        gaps, dtype=np.float32
    )


def decision_summary(data, selected_choices, override_mask):
    old_choices = data['choices']['stage29']
    old_ious = data['ious'][old_choices]
    selected_ious = data['ious'][selected_choices]
    oracle_three = np.maximum.reduce([
        data['ious'][data['choices']['stage29']],
        data['ious'][data['choices']['projected']],
        data['ious'][data['choices']['rawpca']],
    ])
    old_hit25 = old_ious > 0.25
    selected_hit25 = selected_ious > 0.25
    old_hit50 = old_ious > 0.50
    selected_hit50 = selected_ious > 0.50
    return {
        'stage29': metrics(old_ious),
        'projected': metrics(data['ious'][data['choices']['projected']]),
        'rawpca': metrics(data['ious'][data['choices']['rawpca']]),
        'oracle_three': metrics(oracle_three),
        'selected': metrics(selected_ious),
        'override_ratio': float(override_mask.mean()),
        'fix025_count': int((override_mask & ~old_hit25 & selected_hit25).sum()),
        'break025_count': int((override_mask & old_hit25 & ~selected_hit25).sum()),
        'fix050_count': int((override_mask & ~old_hit50 & selected_hit50).sum()),
        'break050_count': int((override_mask & old_hit50 & ~selected_hit50).sum()),
    }


def subset_data(data, mask):
    indices = np.nonzero(mask)[0]
    # Rebuild group-local score arrays so absolute choice indices stay valid.
    rows = []
    for group_index in indices:
        start = int(data['choices']['stage29'][group_index])
        del start  # Choices are reconstructed below; silence accidental use.
    group_offsets = np.cumsum(np.r_[0, data['group_sizes']])
    compact_scores = {name: [] for name in data['scores']}
    compact_normalized = {name: [] for name in data['normalized_scores']}
    compact_ious = []
    compact_sizes = []
    compact_baselines = []
    compact_scenes = []
    for group_index in indices:
        left = int(group_offsets[group_index])
        right = int(group_offsets[group_index + 1])
        compact_ious.append(data['ious'][left:right])
        compact_sizes.append(right - left)
        compact_baselines.append(int(data['baselines'][group_index]))
        compact_scenes.append(data['scenes'][group_index])
        for name in compact_scores:
            compact_scores[name].append(data['scores'][name][left:right])
            compact_normalized[name].append(
                data['normalized_scores'][name][left:right]
            )
    compact = {
        'ious': np.concatenate(compact_ious),
        'group_sizes': np.asarray(compact_sizes, dtype=np.int32),
        'baselines': np.asarray(compact_baselines, dtype=np.int32),
        'scenes': compact_scenes,
        'scores': {
            name: np.concatenate(parts) for name, parts in compact_scores.items()
        },
        'normalized_scores': {
            name: np.concatenate(parts)
            for name, parts in compact_normalized.items()
        },
    }
    compact['choices'] = {
        'stage29': strategy_choices(
            compact['scores']['stage29'], compact_sizes, compact_baselines,
            float(data['thresholds']['stage29']),
        ),
        'projected': strategy_choices(
            compact['scores']['projected'], compact_sizes, compact_baselines,
            float(data['thresholds']['projected']),
        ),
        'rawpca': strategy_choices(
            compact['scores']['rawpca'], compact_sizes, compact_baselines,
            float(data['thresholds']['rawpca']),
        ),
    }
    compact['thresholds'] = dict(data['thresholds'])
    return compact


def choose_threshold(dev, weights, preserve_acc025_drop):
    top_choices, gaps = blend_top_and_gaps(dev, weights)
    default_choices = dev['choices']['stage29']
    changed = top_choices != default_choices
    if changed.any():
        quantiles = np.unique(np.quantile(
            gaps[changed], np.linspace(0.0, 1.0, 401)
        ))
        thresholds = list(map(float, quantiles))
        thresholds.append(float(gaps[changed].max()) + 1.0)
    else:
        thresholds = [1.0]
    baseline = metrics(dev['ious'][default_choices])
    best = None
    for threshold in thresholds:
        override = changed & (gaps >= threshold)
        choices = np.where(override, top_choices, default_choices)
        result = metrics(dev['ious'][choices])
        if result['acc025'] + 1e-12 < (
            baseline['acc025'] - float(preserve_acc025_drop)
        ):
            continue
        key = (
            result['acc050'], result['acc025'], result['mean_iou'],
            -float(override.mean()),
        )
        item = {
            'threshold': float(threshold),
            'summary': decision_summary(dev, choices, override),
        }
        if best is None or key > best[0]:
            best = (key, item)
    assert best is not None
    return best[1]


def apply_locked(data, weights, threshold):
    top_choices, gaps = blend_top_and_gaps(data, weights)
    default = data['choices']['stage29']
    override = (top_choices != default) & (gaps >= float(threshold))
    selected = np.where(override, top_choices, default)
    return decision_summary(data, selected, override)


def weight_grid():
    result = []
    for old_units in range(5):
        for projected_units in range(5 - old_units):
            raw_units = 4 - old_units - projected_units
            if old_units + projected_units + raw_units != 4:
                continue
            result.append({
                'stage29': old_units / 4.0,
                'projected': projected_units / 4.0,
                'rawpca': raw_units / 4.0,
            })
    result.append({'stage29': 1/3, 'projected': 1/3, 'rawpca': 1/3})
    return result


def stack_paths_from_lock(lock):
    class Paths:
        pass
    paths = Paths()
    for key in (
        'old_script', 'projected_script', 'raw_script',
        'old_model', 'projected_model', 'raw_model',
        'old_lock', 'projected_lock', 'raw_lock',
    ):
        setattr(paths, key, lock[key])
        assert sha256(lock[key]) == lock[key + '_sha256'], key
    return paths


def train(args):
    os.makedirs(args.output_dir, exist_ok=False)
    payload = torch.load(args.train_dump, map_location='cpu')
    rows = [
        row for row in payload['rows']
        if 85 <= scene_bucket(str(row['scene_id'])) <= 99
    ]
    stack = load_stack(args)
    data = build_data(rows, stack, require_scene=True)
    data['thresholds'] = {
        'stage29': float(stack['old_lock']['gate']['threshold']),
        'projected': float(stack['projected_lock']['gate']['threshold']),
        'rawpca': float(stack['raw_lock']['gate']['threshold']),
    }
    buckets = np.asarray([scene_bucket(scene) for scene in data['scenes']])
    dev = subset_data(data, (buckets >= 90) & (buckets <= 94))
    test = subset_data(data, (buckets >= 95) & (buckets <= 99))
    candidates = []
    for weights in weight_grid():
        gate = choose_threshold(dev, weights, args.preserve_acc025_drop)
        test_summary = apply_locked(test, weights, gate['threshold'])
        candidates.append({
            'weights': weights,
            'threshold': gate['threshold'],
            'dev': gate['summary'],
            'test': test_summary,
        })
        print('blend_candidate', json.dumps(candidates[-1], sort_keys=True), flush=True)
    selected_index = max(range(len(candidates)), key=lambda index: (
        candidates[index]['dev']['selected']['acc050'],
        candidates[index]['dev']['selected']['acc025'],
        candidates[index]['dev']['selected']['mean_iou'],
        -candidates[index]['dev']['override_ratio'],
    ))
    selected = candidates[selected_index]
    lock = {
        'protocol': 'three_frozen_ranker_normalized_score_blend_train_only',
        'script_sha256': sha256(os.path.abspath(__file__)),
        'train_dump': os.path.abspath(args.train_dump),
        'train_dump_sha256': sha256(args.train_dump),
        'weights': selected['weights'],
        'threshold': selected['threshold'],
        'preserve_acc025_drop': float(args.preserve_acc025_drop),
        'selected_index': selected_index,
        'selected': selected,
        'candidates': candidates,
        'split_bucket_ranges': {'dev': [90, 94], 'test': [95, 99]},
    }
    for key in (
        'old_script', 'projected_script', 'raw_script',
        'old_model', 'projected_model', 'raw_model',
        'old_lock', 'projected_lock', 'raw_lock',
    ):
        path = os.path.abspath(getattr(args, key))
        lock[key] = path
        lock[key + '_sha256'] = sha256(path)
    output = os.path.join(args.output_dir, 'locked_three_ranker_blend.json')
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
    print(json.dumps(lock, indent=2, sort_keys=True))


def evaluate(args):
    lock = load_policy(args.lock_json)
    paths = stack_paths_from_lock(lock)
    stack = load_stack(paths)
    rows = torch.load(args.dump, map_location='cpu')['rows']
    data = build_data(rows, stack, require_scene=False)
    data['thresholds'] = {
        'stage29': float(stack['old_lock']['gate']['threshold']),
        'projected': float(stack['projected_lock']['gate']['threshold']),
        'rawpca': float(stack['raw_lock']['gate']['threshold']),
    }
    result = apply_locked(data, lock['weights'], lock['threshold'])
    result.update({
        'dump': os.path.abspath(args.dump),
        'dump_sha256': sha256(args.dump),
        'lock_sha256': sha256(args.lock_json),
        'weights': lock['weights'],
        'threshold': lock['threshold'],
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
    parser.add_argument('raw_script')
    parser.add_argument('old_model')
    parser.add_argument('projected_model')
    parser.add_argument('raw_model')
    parser.add_argument('old_lock')
    parser.add_argument('projected_lock')
    parser.add_argument('raw_lock')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    train_parser = sub.add_parser('train')
    train_parser.add_argument('train_dump')
    add_stack_args(train_parser)
    train_parser.add_argument('output_dir')
    train_parser.add_argument('--preserve-acc025-drop', type=float, default=0.0)
    eval_parser = sub.add_parser('evaluate')
    eval_parser.add_argument('dump')
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
