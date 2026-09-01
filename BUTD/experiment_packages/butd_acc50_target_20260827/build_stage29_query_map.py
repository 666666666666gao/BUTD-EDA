#!/usr/bin/env python
"""Build a train-only, stable-key map from the locked Stage29 policy.

The LightGBM policy and its gate are fixed before this script sees any
validation examples.  Each ScanRefer training annotation is mapped to the
query, detector match, and interpolation action selected by that policy.
Ground-truth IoU is used only for receipt diagnostics, never for selection.
"""

import argparse
import collections
import hashlib
import json
import os

import lightgbm as lgb
import numpy as np
import torch

from train_joint_option_ranker import (
    ACTIONS,
    MATCH_POWERS,
    aligned_iou,
    candidate_positions,
    pair_iou,
    row_options,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stable_key(row):
    values = (
        str(row.get('scene_id', '')),
        str(row.get('object_id', '')),
        str(row.get('ann_id', '')),
    )
    if not all(values):
        raise ValueError('missing stable ScanRefer key: {}'.format(values))
    return '|'.join(values)


def decode_option(row, option_index, max_candidates):
    positions = candidate_positions(row, max_candidates)
    options_per_candidate = len(MATCH_POWERS) * len(ACTIONS)
    candidate_slot, remainder = divmod(int(option_index), options_per_candidate)
    match_slot, action_slot = divmod(remainder, len(ACTIONS))
    if candidate_slot >= len(positions):
        raise IndexError((option_index, candidate_slot, len(positions)))
    position = int(positions[candidate_slot])
    query_index = int(row['adapter_candidate_query'][position])
    match_power = float(MATCH_POWERS[match_slot])
    alpha = float(ACTIONS[action_slot])

    pred = np.asarray(row['adapter_box_at_candidate'][position], np.float32)
    det_boxes = np.asarray(row.get('detected_box', []), np.float32)
    det_conf = np.asarray(
        row.get('detected_target_confidence', []), np.float32
    )
    if det_boxes.ndim != 2 or det_boxes.shape[-1] != 6:
        det_boxes = np.zeros((0, 6), np.float32)
    if len(det_conf) != len(det_boxes):
        det_conf = np.zeros(len(det_boxes), np.float32)
    det_conf = np.clip(np.nan_to_num(det_conf), 0.0, 1.0)
    if len(det_boxes):
        support = pair_iou(pred, det_boxes)
        match_score = support * np.power(det_conf + 1e-6, match_power)
        detector_index = int(np.argmax(match_score))
        matched = det_boxes[detector_index]
    else:
        detector_index = -1
        matched = pred.copy()
    option = pred + alpha * (matched - pred)
    option[3:] = np.maximum(np.abs(option[3:]), 1e-5)
    return {
        'query_index': query_index,
        'candidate_position': position,
        'candidate_slot': candidate_slot,
        'match_power': match_power,
        'match_slot': match_slot,
        'alpha': alpha,
        'action_slot': action_slot,
        'detector_index': detector_index,
        'detector_box': matched.astype(np.float32).tolist(),
        'source_query_box': pred.astype(np.float32).tolist(),
        'source_option_box': option.astype(np.float32).tolist(),
    }


def metrics(values):
    values = np.asarray(values, np.float32)
    return {
        'count': int(len(values)),
        'acc025': float(np.mean(values >= 0.25)),
        'acc050': float(np.mean(values >= 0.50)),
        'mean_iou': float(np.mean(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump')
    parser.add_argument('model')
    parser.add_argument('lock_json')
    parser.add_argument('output_map')
    parser.add_argument('output_json')
    parser.add_argument('--predict-batch-size', type=int, default=256)
    args = parser.parse_args()

    if os.path.exists(args.output_map) or os.path.exists(args.output_json):
        raise FileExistsError('refusing to overwrite Stage29 query-map outputs')
    lock = json.load(open(args.lock_json, encoding='utf-8'))
    if sha256(args.model) != lock['model_sha256']:
        raise ValueError('Stage29 model SHA256 does not match lock')
    rows = torch.load(args.dump, map_location='cpu')['rows']
    if not rows:
        raise ValueError('empty geometry dump')
    max_candidates = int(lock.get('max_candidates', 8))
    threshold = float(lock['gate']['threshold'])
    iteration = int(lock['best_iteration'])
    booster = lgb.Booster(model_file=args.model)

    entries = {}
    baseline_ious = []
    selected_ious = []
    selected_raw_ious = []
    action_counts = collections.Counter()
    match_counts = collections.Counter()
    query_changed = 0
    policy_changed = 0
    predict_batch_size = max(1, int(args.predict_batch_size))
    for start in range(0, len(rows), predict_batch_size):
        pending = []
        flat_features = []
        for index in range(start, min(start + predict_batch_size, len(rows))):
            row = rows[index]
            key = stable_key(row)
            if key in entries:
                raise ValueError('duplicate stable key: {}'.format(key))
            features, ious, baseline_index = row_options(
                row, max_candidates=max_candidates
            )
            pending.append((index, row, key, ious, int(baseline_index), len(features)))
            flat_features.append(features)
        flat_scores = booster.predict(
            np.concatenate(flat_features, axis=0), num_iteration=iteration
        )
        cursor = 0
        for index, row, key, ious, baseline_index, option_count in pending:
            scores = flat_scores[cursor:cursor + option_count]
            cursor += option_count
            best_index = int(np.argmax(scores))
            gap = float(scores[best_index] - scores[baseline_index])
            selected_index = best_index if gap >= threshold else baseline_index
            decoded = decode_option(row, selected_index, max_candidates)
            baseline_decoded = decode_option(
                row, baseline_index, max_candidates
            )
            gt_box = np.asarray(row['gt_box'], np.float32)
            raw_iou = float(aligned_iou(
                np.asarray(decoded['source_query_box'], np.float32)[None],
                gt_box[None],
            )[0])
            decoded.update({
                'policy_gap': gap,
                'policy_gate_passed': bool(gap >= threshold),
                'policy_option_index': int(selected_index),
                'baseline_option_index': int(baseline_index),
                'baseline_query_index': int(baseline_decoded['query_index']),
            })
            entries[key] = decoded
            baseline_ious.append(float(ious[baseline_index]))
            selected_ious.append(float(ious[selected_index]))
            selected_raw_ious.append(raw_iou)
            action_counts['{:.1f}'.format(decoded['alpha'])] += 1
            match_counts['{:.1f}'.format(decoded['match_power'])] += 1
            policy_changed += int(selected_index != baseline_index)
            query_changed += int(
                decoded['query_index'] != baseline_decoded['query_index']
            )
        if cursor != len(flat_scores):
            raise AssertionError((cursor, len(flat_scores)))
        print(
            'mapped={}/{}'.format(start + len(pending), len(rows)),
            flush=True,
        )

    source_hashes = {
        'dump_sha256': sha256(args.dump),
        'model_sha256': sha256(args.model),
        'lock_sha256': sha256(args.lock_json),
    }
    summary = {
        'stage': '134_map',
        'status': 'complete',
        'protocol': 'train_only_locked_stage29_stable_key_query_action_map',
        'rows': len(rows),
        'unique_keys': len(entries),
        'max_candidates': max_candidates,
        'threshold': threshold,
        'best_iteration': iteration,
        'predict_batch_size': predict_batch_size,
        'baseline': metrics(baseline_ious),
        'selected_option': metrics(selected_ious),
        'selected_raw_query': metrics(selected_raw_ious),
        'policy_changed_ratio': float(policy_changed / len(rows)),
        'query_changed_ratio': float(query_changed / len(rows)),
        'action_counts': dict(sorted(action_counts.items())),
        'match_power_counts': dict(sorted(match_counts.items())),
        'source_hashes': source_hashes,
    }
    payload = {
        'format': 'stage29_query_action_map_v1',
        'metadata': summary,
        'entries': entries,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_map)), exist_ok=True)
    torch.save(payload, args.output_map)
    summary['output_map'] = os.path.abspath(args.output_map)
    summary['output_map_sha256'] = sha256(args.output_map)
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
