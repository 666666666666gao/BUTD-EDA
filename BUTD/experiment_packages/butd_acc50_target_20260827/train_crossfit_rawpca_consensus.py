#!/usr/bin/env python
"""Cross-fit raw-PCA ranker and lock a semantic-consensus override policy."""

import argparse
import gc
import hashlib
import importlib.util
import json
import math
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


def selected_feature_match(left, right, atol=1e-6):
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    assert left.shape == right.shape
    return np.all(np.isclose(left, right, rtol=0.0, atol=atol), axis=1)


def consensus_summary(old_iou, projected_iou, raw_iou,
                      raw_eq_old, raw_eq_projected):
    old_iou = np.asarray(old_iou, dtype=np.float32)
    projected_iou = np.asarray(projected_iou, dtype=np.float32)
    raw_iou = np.asarray(raw_iou, dtype=np.float32)
    raw_eq_old = np.asarray(raw_eq_old, dtype=bool)
    raw_eq_projected = np.asarray(raw_eq_projected, dtype=bool)
    assert old_iou.shape == projected_iou.shape == raw_iou.shape
    override = raw_eq_projected & ~raw_eq_old
    if raw_eq_projected.any():
        max_delta = float(np.max(np.abs(
            raw_iou[raw_eq_projected] - projected_iou[raw_eq_projected]
        )))
        assert max_delta <= 1e-5, max_delta
    selected = np.where(override, projected_iou, old_iou)
    old25 = old_iou > 0.25
    projected25 = projected_iou > 0.25
    old50 = old_iou > 0.50
    projected50 = projected_iou > 0.50
    return {
        'stage29': metrics(old_iou),
        'projected': metrics(projected_iou),
        'rawpca': metrics(raw_iou),
        'oracle_three': metrics(np.maximum.reduce([
            old_iou, projected_iou, raw_iou
        ])),
        'selected': metrics(selected),
        'semantic_agreement_count': int(raw_eq_projected.sum()),
        'semantic_agreement_ratio': float(raw_eq_projected.mean()),
        'override_count': int(override.sum()),
        'override_ratio': float(override.mean()),
        'fix025_count': int((override & ~old25 & projected25).sum()),
        'break025_count': int((override & old25 & ~projected25).sum()),
        'fix050_count': int((override & ~old50 & projected50).sum()),
        'break050_count': int((override & old50 & ~projected50).sum()),
    }


def load_oof(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def train(args):
    os.makedirs(args.output_dir, exist_ok=False)
    stack = load_module(args.stack_script, 'consensus_stack_module')
    raw_module = load_module(args.raw_script, 'consensus_raw_module')
    stage88_lock = json.load(open(
        args.stage88_lock, 'r', encoding='utf-8'
    ))
    raw_lock = json.load(open(args.raw_lock, 'r', encoding='utf-8'))
    assert sha256(args.raw_model) == raw_lock['model_sha256']
    assert raw_lock['feature_names'] == raw_module.FEATURE_NAMES
    assert sha256(args.stage88_oof) == stage88_lock['oof_sha256']
    old_lock = json.load(open(stage88_lock['old_lock'], encoding='utf-8'))
    projected_lock = json.load(open(
        stage88_lock['projected_lock'], encoding='utf-8'
    ))
    old_names = old_lock['feature_names']
    projected_names = projected_lock['feature_names']
    raw_positions = {
        name: index for index, name in enumerate(raw_module.FEATURE_NAMES)
    }
    old_indices = np.asarray(
        [raw_positions[name] for name in old_names], dtype=np.int64
    )
    oof = load_oof(args.stage88_oof)
    n = len(oof['old_iou'])
    assert oof['x'].shape == (
        n, len(stage88_lock['feature_names'])
    )
    rows = torch.load(args.raw_dump, map_location='cpu')['rows']
    assert len(rows) == n
    raw_example_ids = np.asarray([
        int(row.get('example_id', index)) for index, row in enumerate(rows)
    ], dtype=np.int64)
    assert np.array_equal(raw_example_ids, oof['example_ids'])
    groups = stack.build_groups(
        rows, raw_module, int(raw_lock['max_candidates'])
    )
    raw_groups, iou_groups, baselines, scenes, _ = groups
    buckets = np.asarray(oof['buckets'], dtype=np.int32)
    assert np.array_equal(
        buckets,
        np.asarray([stack.scene_bucket(scene) for scene in scenes], dtype=np.int32),
    )
    projected_dim = len(projected_names)
    old_dim = len(old_names)
    old_choice_features = oof['x'][:, projected_dim:projected_dim + old_dim]
    projected_choice_features = oof['x'][
        :, projected_dim + old_dim:projected_dim + 2 * old_dim
    ]
    assert old_choice_features.shape == (n, old_dim)
    assert projected_choice_features.shape == (n, old_dim)
    raw_iou = np.empty(n, dtype=np.float32)
    raw_choice_features = np.empty((n, old_dim), dtype=np.float32)
    raw_gate_margin = np.empty(n, dtype=np.float32)
    all_indices = np.arange(n, dtype=np.int64)
    folds = buckets % int(args.num_folds)
    fold_train_receipts = []
    for fold in range(int(args.num_folds)):
        held_indices = all_indices[folds == fold].tolist()
        train_indices = all_indices[folds != fold].tolist()
        train_raw, _, train_iou, train_sizes, _ = stack.materialize(
            raw_groups, iou_groups, baselines, train_indices, old_indices
        )
        held_raw, held_old, held_iou, held_sizes, held_baselines = stack.materialize(
            raw_groups, iou_groups, baselines, held_indices, old_indices
        )
        labels = (train_iou >= 0.50).astype(np.int32)
        ranker = stack.make_ranker(
            raw_lock['selected_config'], raw_lock['best_iteration'],
            raw_lock['selected_config_index'], args.num_threads,
        )
        ranker.fit(
            train_raw, labels, group=train_sizes.tolist(),
            feature_name=raw_module.FEATURE_NAMES,
            callbacks=[lgb.log_evaluation(0)],
        )
        scores = ranker.booster_.predict(
            held_raw, num_iteration=int(raw_lock['best_iteration'])
        ).astype(np.float32)
        choices, gaps = stack.choice_indices(
            scores, held_sizes, held_baselines,
            float(raw_lock['gate']['threshold']),
        )
        raw_iou[held_indices] = held_iou[choices]
        raw_choice_features[held_indices] = held_old[choices]
        raw_gate_margin[held_indices] = (
            gaps - float(raw_lock['gate']['threshold'])
        )
        fold_train_receipts.append({
            'fold': fold,
            'train_groups': len(train_indices),
            'heldout_groups': len(held_indices),
            'rawpca': metrics(held_iou[choices]),
        })
        print('RAW_OOF_FOLD_COMPLETE', json.dumps(
            fold_train_receipts[-1], sort_keys=True
        ), flush=True)
        del train_raw, train_iou, train_sizes, held_raw, held_old
        del held_iou, held_sizes, held_baselines, labels, ranker, scores
        gc.collect()
    raw_eq_old = selected_feature_match(
        raw_choice_features, old_choice_features
    )
    raw_eq_projected = selected_feature_match(
        raw_choice_features, projected_choice_features
    )
    all_summary = consensus_summary(
        oof['old_iou'], oof['projected_iou'], raw_iou,
        raw_eq_old, raw_eq_projected,
    )
    fold_summaries = []
    for fold in range(int(args.num_folds)):
        mask = folds == fold
        item = consensus_summary(
            oof['old_iou'][mask], oof['projected_iou'][mask], raw_iou[mask],
            raw_eq_old[mask], raw_eq_projected[mask],
        )
        item['fold'] = fold
        item['net025'] = item['fix025_count'] - item['break025_count']
        item['net050'] = item['fix050_count'] - item['break050_count']
        fold_summaries.append(item)
    net025 = all_summary['fix025_count'] - all_summary['break025_count']
    net050 = all_summary['fix050_count'] - all_summary['break050_count']
    required_scaled_net050 = int(math.ceil(32.0 * n / 9508.0))
    fold_nets050 = [item['net050'] for item in fold_summaries]
    external_eval_worthy = bool(
        net025 >= 0
        and net050 >= required_scaled_net050
        and sum(value >= 0 for value in fold_nets050) >= 4
        and min(fold_nets050) >= -5
    )
    evidence_path = os.path.join(
        args.output_dir, 'rawpca_oof_consensus_evidence.npz'
    )
    with open(evidence_path + '.tmp', 'wb') as handle:
        np.savez(
            handle,
            raw_iou=raw_iou,
            raw_choice_features=raw_choice_features,
            raw_gate_margin=raw_gate_margin,
            raw_eq_old=raw_eq_old,
            raw_eq_projected=raw_eq_projected,
            buckets=buckets,
            example_ids=oof['example_ids'],
        )
    os.replace(evidence_path + '.tmp', evidence_path)
    lock = {
        'protocol': 'five_fold_scene_oof_rawpca_semantic_consensus',
        'rule': (
            'override_stage29_only_when_projected_and_rawpca_select_the_same_'
            'non_stage29_candidate'
        ),
        'num_folds': int(args.num_folds),
        'script_sha256': sha256(os.path.abspath(__file__)),
        'raw_dump': os.path.abspath(args.raw_dump),
        'raw_dump_sha256': sha256(args.raw_dump),
        'stage88_oof': os.path.abspath(args.stage88_oof),
        'stage88_oof_sha256': sha256(args.stage88_oof),
        'stage88_lock': os.path.abspath(args.stage88_lock),
        'stage88_lock_sha256': sha256(args.stage88_lock),
        'raw_script': os.path.abspath(args.raw_script),
        'raw_script_sha256': sha256(args.raw_script),
        'raw_model': os.path.abspath(args.raw_model),
        'raw_model_sha256': sha256(args.raw_model),
        'raw_lock': os.path.abspath(args.raw_lock),
        'raw_lock_sha256': sha256(args.raw_lock),
        'stack_script': os.path.abspath(args.stack_script),
        'stack_script_sha256': sha256(args.stack_script),
        'evidence_path': os.path.abspath(evidence_path),
        'evidence_sha256': sha256(evidence_path),
        'fold_train_receipts': fold_train_receipts,
        'fold_summaries': fold_summaries,
        'all_oof': all_summary,
        'net025': net025,
        'net050': net050,
        'required_scaled_net050': required_scaled_net050,
        'external_eval_worthy': external_eval_worthy,
    }
    lock_path = os.path.join(
        args.output_dir, 'locked_rawpca_consensus_policy.json'
    )
    with open(lock_path, 'w', encoding='utf-8') as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
    print(json.dumps(lock, indent=2, sort_keys=True))


def evaluate(args):
    lock = json.load(open(args.lock_json, 'r', encoding='utf-8'))
    assert lock['external_eval_worthy'] is True
    for key in ('stage88_lock', 'raw_script', 'raw_model', 'raw_lock', 'stack_script'):
        assert sha256(lock[key]) == lock[key + '_sha256'], key
    stack = load_module(lock['stack_script'], 'consensus_eval_stack')
    raw_module = load_module(lock['raw_script'], 'consensus_eval_raw')
    stage88 = json.load(open(lock['stage88_lock'], encoding='utf-8'))
    old_module = load_module(stage88['old_script'], 'consensus_eval_old')
    projected_module = load_module(
        stage88['projected_script'], 'consensus_eval_projected'
    )
    old_policy = json.load(open(stage88['old_lock'], encoding='utf-8'))
    projected_policy = json.load(open(
        stage88['projected_lock'], encoding='utf-8'
    ))
    raw_policy = json.load(open(lock['raw_lock'], encoding='utf-8'))
    for path, policy in (
        (stage88['old_model'], old_policy),
        (stage88['projected_model'], projected_policy),
        (lock['raw_model'], raw_policy),
    ):
        assert sha256(path) == policy['model_sha256']
    rows = torch.load(args.dump, map_location='cpu')['rows']
    groups = stack.build_groups(
        rows, raw_module, int(raw_policy['max_candidates'])
    )
    raw_positions = {
        name: index for index, name in enumerate(raw_module.FEATURE_NAMES)
    }
    old_indices = np.asarray([
        raw_positions[name] for name in old_module.FEATURE_NAMES
    ], dtype=np.int64)
    projected_indices = np.asarray([
        raw_positions[name] for name in projected_module.FEATURE_NAMES
    ], dtype=np.int64)
    indices = list(range(len(rows)))
    raw_x, old_x, ious, sizes, baselines = stack.materialize(
        groups[0], groups[1], groups[2], indices, old_indices
    )
    projected_x = raw_x[:, projected_indices]
    old_scores = lgb.Booster(model_file=stage88['old_model']).predict(
        old_x, num_iteration=int(old_policy['best_iteration'])
    ).astype(np.float32)
    projected_scores = lgb.Booster(
        model_file=stage88['projected_model']
    ).predict(
        projected_x, num_iteration=int(projected_policy['best_iteration'])
    ).astype(np.float32)
    raw_scores = lgb.Booster(model_file=lock['raw_model']).predict(
        raw_x, num_iteration=int(raw_policy['best_iteration'])
    ).astype(np.float32)
    old_choices, _ = stack.choice_indices(
        old_scores, sizes, baselines, old_policy['gate']['threshold']
    )
    projected_choices, _ = stack.choice_indices(
        projected_scores, sizes, baselines,
        projected_policy['gate']['threshold'],
    )
    raw_choices, _ = stack.choice_indices(
        raw_scores, sizes, baselines, raw_policy['gate']['threshold']
    )
    raw_eq_old = selected_feature_match(
        old_x[raw_choices], old_x[old_choices]
    )
    raw_eq_projected = selected_feature_match(
        old_x[raw_choices], old_x[projected_choices]
    )
    result = consensus_summary(
        ious[old_choices], ious[projected_choices], ious[raw_choices],
        raw_eq_old, raw_eq_projected,
    )
    result.update({
        'dump': os.path.abspath(args.dump),
        'dump_sha256': sha256(args.dump),
        'lock_sha256': sha256(args.lock_json),
        'goal_achieved_offline': bool(
            result['selected']['acc025'] > 0.5391
            and result['selected']['acc050'] > 0.4241
        ),
        'diagnostic_only': True,
    })
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    train_parser = sub.add_parser('train')
    train_parser.add_argument('raw_dump')
    train_parser.add_argument('stage88_oof')
    train_parser.add_argument('stage88_lock')
    train_parser.add_argument('stack_script')
    train_parser.add_argument('raw_script')
    train_parser.add_argument('raw_model')
    train_parser.add_argument('raw_lock')
    train_parser.add_argument('output_dir')
    train_parser.add_argument('--num-folds', type=int, default=5)
    train_parser.add_argument('--num-threads', type=int, default=32)
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
