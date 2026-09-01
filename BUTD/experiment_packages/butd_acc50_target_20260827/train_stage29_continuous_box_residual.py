#!/usr/bin/env python
"""Train a conservative continuous box residual on locked Stage29 choices."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CENTER_LIMIT = 0.50
LOG_SIZE_LIMIT = 0.70
GATE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
RESIDUAL_SCALES = (0.25, 0.50, 0.75, 1.00)


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


def decode_projected_queries(row):
    shape = tuple(int(value) for value in row[
        'adapter_last_proj_query_f16_shape'
    ])
    values = np.frombuffer(
        row['adapter_last_proj_query_f16'], dtype=np.float16
    ).reshape(shape).astype(np.float32)
    assert shape[0] == len(row['adapter_candidate_query'])
    assert shape[1] == 64
    assert np.isfinite(values).all()
    return values


def reconstruct_option(row, option_index, joint, max_candidates):
    positions = joint.candidate_positions(row, max_candidates)
    actions_per_candidate = len(joint.MATCH_POWERS) * len(joint.ACTIONS)
    candidate_slot = int(option_index) // actions_per_candidate
    remainder = int(option_index) % actions_per_candidate
    match_slot = remainder // len(joint.ACTIONS)
    action_slot = remainder % len(joint.ACTIONS)
    assert 0 <= candidate_slot < len(positions)
    position = int(positions[candidate_slot])
    match_power = float(joint.MATCH_POWERS[match_slot])
    alpha = float(joint.ACTIONS[action_slot])
    pred = np.asarray(row['adapter_box_at_candidate'][position], np.float32)
    pred[3:] = np.maximum(np.abs(pred[3:]), 1e-5)
    det_boxes = np.asarray(row.get('detected_box', []), dtype=np.float32)
    det_conf = np.asarray(
        row.get('detected_target_confidence', []), dtype=np.float32
    )
    if det_boxes.ndim != 2 or det_boxes.shape[-1] != 6:
        det_boxes = np.zeros((0, 6), dtype=np.float32)
    if len(det_conf) != len(det_boxes):
        det_conf = np.zeros(len(det_boxes), dtype=np.float32)
    det_conf = np.clip(np.nan_to_num(det_conf), 0.0, 1.0)
    support = joint.pair_iou(pred, det_boxes)
    if len(det_boxes):
        match_score = support * np.power(det_conf + 1e-6, match_power)
        match_index = int(np.argmax(match_score))
        matched = det_boxes[match_index].astype(np.float32, copy=True)
    else:
        match_index = -1
        matched = pred.copy()
    matched[3:] = np.maximum(np.abs(matched[3:]), 1e-5)
    option = pred + alpha * (matched - pred)
    option[3:] = np.maximum(np.abs(option[3:]), 1e-5)
    return {
        'position': position,
        'query_id': int(row['adapter_candidate_query'][position]),
        'match_power': match_power,
        'match_index': match_index,
        'alpha': alpha,
        'box': option,
    }


def collect_locked_choices(rows, joint, booster, policy, chunk_size=256):
    feature_rows = []
    boxes = []
    gt_boxes = []
    base_ious = []
    scenes = []
    example_ids = []
    query_ids = []
    for start in range(0, len(rows), int(chunk_size)):
        chunk = rows[start:start + int(chunk_size)]
        group_x = []
        group_iou = []
        baselines = []
        for row in chunk:
            x, ious, baseline = joint.row_options(
                row, max_candidates=int(policy['max_candidates'])
            )
            group_x.append(x)
            group_iou.append(ious)
            baselines.append(int(baseline))
        merged = np.concatenate(group_x, axis=0)
        scores = booster.predict(
            merged, num_iteration=int(policy['best_iteration'])
        )
        cursor = 0
        for local_index, (row, x, ious, baseline) in enumerate(zip(
                chunk, group_x, group_iou, baselines)):
            group_scores = scores[cursor:cursor + len(x)]
            top = int(np.argmax(group_scores))
            gap = float(group_scores[top] - group_scores[baseline])
            choice = top if gap >= float(policy['gate']['threshold']) else baseline
            record = reconstruct_option(
                row, choice, joint, int(policy['max_candidates'])
            )
            gt = np.asarray(row['gt_box'], dtype=np.float32)
            calculated_iou = float(joint.aligned_iou(
                record['box'][None], gt[None]
            )[0])
            assert abs(calculated_iou - float(ious[choice])) <= 2e-5
            projected = decode_projected_queries(row)
            feature_rows.append(np.concatenate([
                np.asarray(x[choice], dtype=np.float32),
                projected[record['position']],
            ]))
            boxes.append(record['box'])
            gt_boxes.append(gt)
            base_ious.append(calculated_iou)
            scene = str(row.get('scene_id', ''))
            scenes.append(scene)
            example_ids.append(int(row.get('example_id', start + local_index)))
            query_ids.append(record['query_id'])
            cursor += len(x)
        assert cursor == len(scores)
        if min(start + len(chunk), len(rows)) % 1024 < int(chunk_size):
            print('residual_built_groups={}/{}'.format(
                min(start + len(chunk), len(rows)), len(rows)
            ), flush=True)
    result = {
        'x': np.asarray(feature_rows, dtype=np.float32),
        'boxes': np.asarray(boxes, dtype=np.float32),
        'gt_boxes': np.asarray(gt_boxes, dtype=np.float32),
        'base_iou': np.asarray(base_ious, dtype=np.float32),
        'scenes': scenes,
        'example_ids': np.asarray(example_ids, dtype=np.int64),
        'query_ids': np.asarray(query_ids, dtype=np.int64),
    }
    assert result['x'].shape == (len(rows), 120)
    assert np.isfinite(result['x']).all()
    return result


def residual_targets(boxes, gt_boxes, base_iou):
    size = np.maximum(np.abs(boxes[:, 3:]), 1e-5)
    gt_size = np.maximum(np.abs(gt_boxes[:, 3:]), 1e-5)
    center = (gt_boxes[:, :3] - boxes[:, :3]) / size
    log_size = np.log(gt_size / size)
    raw_target = np.concatenate([center, log_size], axis=1).astype(np.float32)
    reachable = (
        (np.abs(center) <= CENTER_LIMIT).all(axis=1)
        & (np.abs(log_size) <= LOG_SIZE_LIMIT).all(axis=1)
    )
    active = (
        (base_iou >= 0.25) & (base_iou < 0.50) & reachable
    )
    target = np.zeros_like(raw_target)
    target[active, :3] = np.clip(
        raw_target[active, :3], -CENTER_LIMIT, CENTER_LIMIT
    )
    target[active, 3:] = np.clip(
        raw_target[active, 3:], -LOG_SIZE_LIMIT, LOG_SIZE_LIMIT
    )
    return target, active.astype(np.float32), reachable


class ResidualNet(nn.Module):
    def __init__(self, input_dim=120, hidden_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.gate = nn.Linear(hidden_dim // 2, 1)
        self.delta = nn.Linear(hidden_dim // 2, 6)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x):
        hidden = self.backbone(x)
        gate = self.gate(hidden).squeeze(-1)
        raw = torch.tanh(self.delta(hidden))
        limits = raw.new_tensor([
            CENTER_LIMIT, CENTER_LIMIT, CENTER_LIMIT,
            LOG_SIZE_LIMIT, LOG_SIZE_LIMIT, LOG_SIZE_LIMIT,
        ])
        return gate, raw * limits


def split_masks(scenes, scene_bucket):
    buckets = np.asarray([scene_bucket(scene) for scene in scenes], np.int32)
    return buckets, {
        'train': buckets < 70,
        'dev': (buckets >= 70) & (buckets < 85),
        'test': buckets >= 85,
    }


def predict(model, x, device, batch_size=2048):
    gates = []
    deltas = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            values = torch.from_numpy(x[start:start + batch_size]).to(device)
            gate, delta = model(values)
            gates.append(torch.sigmoid(gate).cpu().numpy())
            deltas.append(delta.cpu().numpy())
    return np.concatenate(gates), np.concatenate(deltas)


def apply_residual(boxes, delta, gate, threshold, scale):
    output = np.asarray(boxes, dtype=np.float32).copy()
    active = np.asarray(gate) >= float(threshold)
    scaled = np.asarray(delta, dtype=np.float32) * float(scale)
    output[active, :3] += (
        scaled[active, :3] * np.maximum(np.abs(output[active, 3:]), 1e-5)
    )
    output[active, 3:] *= np.exp(scaled[active, 3:])
    output[:, 3:] = np.maximum(np.abs(output[:, 3:]), 1e-5)
    return output, active


def aligned_iou(joint, boxes, gt_boxes):
    return np.asarray(joint.aligned_iou(boxes, gt_boxes), dtype=np.float32)


def metrics(ious):
    values = np.asarray(ious, dtype=np.float32)
    return {
        'count': int(len(values)),
        'mean_iou': float(values.mean()),
        'acc025': float((values > 0.25).mean()),
        'acc050': float((values > 0.50).mean()),
    }


def policy_summary(joint, boxes, gt_boxes, base_iou, gate, delta,
                   threshold, scale):
    corrected, override = apply_residual(
        boxes, delta, gate, threshold, scale
    )
    selected_iou = aligned_iou(joint, corrected, gt_boxes)
    old25 = base_iou > 0.25
    new25 = selected_iou > 0.25
    old50 = base_iou > 0.50
    new50 = selected_iou > 0.50
    return {
        'baseline': metrics(base_iou),
        'selected': metrics(selected_iou),
        'override_count': int(override.sum()),
        'override_ratio': float(override.mean()),
        'fix025_count': int((~old25 & new25).sum()),
        'break025_count': int((old25 & ~new25).sum()),
        'fix050_count': int((~old50 & new50).sum()),
        'break050_count': int((old50 & ~new50).sum()),
        'threshold': float(threshold),
        'scale': float(scale),
    }


def choose_policy(joint, arrays, mask, gate, delta):
    candidates = []
    for threshold in GATE_THRESHOLDS:
        for scale in RESIDUAL_SCALES:
            summary = policy_summary(
                joint, arrays['boxes'][mask], arrays['gt_boxes'][mask],
                arrays['base_iou'][mask], gate[mask], delta[mask],
                threshold, scale,
            )
            if summary['selected']['acc025'] + 1e-12 < summary['baseline']['acc025']:
                continue
            candidates.append(summary)
    assert candidates
    return max(candidates, key=lambda item: (
        item['selected']['acc050'], item['selected']['acc025'],
        item['selected']['mean_iou'], -item['override_ratio'],
    )), candidates


def train(args):
    os.makedirs(args.output_dir, exist_ok=False)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    joint = load_module(args.joint_script, 'box_residual_joint')
    policy = json.load(open(args.stage29_lock, encoding='utf-8'))
    assert sha256(args.stage29_model) == policy['model_sha256']
    assert policy['feature_names'] == joint.FEATURE_NAMES
    rows = torch.load(args.train_dump, map_location='cpu')['rows']
    booster = lgb.Booster(model_file=args.stage29_model)
    arrays = collect_locked_choices(rows, joint, booster, policy)
    target, active, reachable = residual_targets(
        arrays['boxes'], arrays['gt_boxes'], arrays['base_iou']
    )
    buckets, masks = split_masks(arrays['scenes'], joint.scene_bucket)
    mean = arrays['x'][masks['train']].mean(axis=0).astype(np.float32)
    std = arrays['x'][masks['train']].std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-5)
    normalized = ((arrays['x'] - mean) / std).astype(np.float32)
    device = torch.device(args.device)
    model = ResidualNet(input_dim=120, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    train_indices = np.nonzero(masks['train'])[0]
    dev_indices = np.nonzero(masks['dev'])[0]
    positive = max(1, int(active[train_indices].sum()))
    negative = max(1, len(train_indices) - positive)
    pos_weight = torch.tensor(
        [min(20.0, negative / positive)], device=device
    )
    best = None
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        rng = np.random.RandomState(epoch)
        order = train_indices.copy()
        rng.shuffle(order)
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            x_t = torch.from_numpy(normalized[index]).to(device)
            target_t = torch.from_numpy(target[index]).to(device)
            active_t = torch.from_numpy(active[index]).to(device)
            gate_logit, predicted = model(x_t)
            gate_loss = F.binary_cross_entropy_with_logits(
                gate_logit, active_t, pos_weight=pos_weight
            )
            active_denom = active_t.sum().clamp(min=1.0)
            residual_loss = (
                F.smooth_l1_loss(predicted, target_t, reduction='none').mean(1)
                * active_t
            ).sum() / active_denom
            inactive = 1.0 - active_t
            zero_loss = (
                predicted.pow(2).mean(1) * inactive
            ).sum() / inactive.sum().clamp(min=1.0)
            loss = gate_loss + 2.0 * residual_loss + 0.10 * zero_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(index)
            train_count += len(index)
        model.eval()
        with torch.no_grad():
            x_d = torch.from_numpy(normalized[dev_indices]).to(device)
            target_d = torch.from_numpy(target[dev_indices]).to(device)
            active_d = torch.from_numpy(active[dev_indices]).to(device)
            gate_d, predicted_d = model(x_d)
            dev_gate = F.binary_cross_entropy_with_logits(
                gate_d, active_d, pos_weight=pos_weight
            )
            dev_residual = (
                F.smooth_l1_loss(
                    predicted_d, target_d, reduction='none'
                ).mean(1) * active_d
            ).sum() / active_d.sum().clamp(min=1.0)
            dev_loss = float(dev_gate + 2.0 * dev_residual)
        item = {
            'epoch': epoch,
            'train_loss': train_loss_sum / max(1, train_count),
            'dev_loss': dev_loss,
        }
        history.append(item)
        print('BOX_RESIDUAL_EPOCH', json.dumps(item, sort_keys=True), flush=True)
        if best is None or dev_loss < best['dev_loss'] - 1e-6:
            best = {
                'epoch': epoch,
                'dev_loss': dev_loss,
                'state_dict': {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break
    assert best is not None
    model.load_state_dict(best['state_dict'])
    gate, delta = predict(model, normalized, device)
    chosen, policy_candidates = choose_policy(
        joint, arrays, masks['dev'], gate, delta
    )
    test_summary = policy_summary(
        joint, arrays['boxes'][masks['test']],
        arrays['gt_boxes'][masks['test']], arrays['base_iou'][masks['test']],
        gate[masks['test']], delta[masks['test']],
        chosen['threshold'], chosen['scale'],
    )
    dev_net050 = chosen['fix050_count'] - chosen['break050_count']
    test_net025 = test_summary['fix025_count'] - test_summary['break025_count']
    test_net050 = test_summary['fix050_count'] - test_summary['break050_count']
    external_eval_worthy = bool(
        dev_net050 > 0 and test_net025 >= 0 and test_net050 >= 19
    )
    model_path = os.path.join(args.output_dir, 'box_residual_model.pt')
    torch.save({
        'state_dict': best['state_dict'],
        'input_mean': mean,
        'input_std': std,
        'input_dim': 120,
        'hidden_dim': int(args.hidden_dim),
        'center_limit': CENTER_LIMIT,
        'log_size_limit': LOG_SIZE_LIMIT,
    }, model_path)
    evidence_path = os.path.join(args.output_dir, 'internal_evidence.npz')
    with open(evidence_path + '.tmp', 'wb') as handle:
        np.savez(
            handle,
            gate=gate,
            delta=delta,
            base_iou=arrays['base_iou'],
            buckets=buckets,
            active=active,
            reachable=reachable,
            example_ids=arrays['example_ids'],
            query_ids=arrays['query_ids'],
        )
    os.replace(evidence_path + '.tmp', evidence_path)
    lock = {
        'protocol': 'stage29_locked_choice_scene70_15_15_continuous_box_residual',
        'script_sha256': sha256(os.path.abspath(__file__)),
        'train_dump': os.path.abspath(args.train_dump),
        'train_dump_sha256': sha256(args.train_dump),
        'joint_script': os.path.abspath(args.joint_script),
        'joint_script_sha256': sha256(args.joint_script),
        'stage29_model': os.path.abspath(args.stage29_model),
        'stage29_model_sha256': sha256(args.stage29_model),
        'stage29_lock': os.path.abspath(args.stage29_lock),
        'stage29_lock_sha256': sha256(args.stage29_lock),
        'model_path': os.path.abspath(model_path),
        'model_sha256': sha256(model_path),
        'evidence_path': os.path.abspath(evidence_path),
        'evidence_sha256': sha256(evidence_path),
        'feature_dim': 120,
        'hidden_dim': int(args.hidden_dim),
        'center_limit': CENTER_LIMIT,
        'log_size_limit': LOG_SIZE_LIMIT,
        'gate_threshold_grid': list(GATE_THRESHOLDS),
        'residual_scale_grid': list(RESIDUAL_SCALES),
        'best_epoch': int(best['epoch']),
        'best_dev_loss': float(best['dev_loss']),
        'active_counts': {
            name: int(active[mask].sum()) for name, mask in masks.items()
        },
        'split_counts': {
            name: int(mask.sum()) for name, mask in masks.items()
        },
        'selected_policy': {
            'threshold': chosen['threshold'],
            'scale': chosen['scale'],
        },
        'internal': {
            'dev': chosen,
            'test': test_summary,
        },
        'test_net025': test_net025,
        'test_net050': test_net050,
        'external_eval_worthy': external_eval_worthy,
        'history': history,
        'dev_policy_candidates': policy_candidates,
    }
    lock_path = os.path.join(
        args.output_dir, 'locked_box_residual_policy.json'
    )
    with open(lock_path, 'w', encoding='utf-8') as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
    print(json.dumps(lock, indent=2, sort_keys=True))


def load_model_artifact(path, device):
    artifact = torch.load(path, map_location='cpu')
    assert artifact['input_dim'] == 120
    assert artifact['center_limit'] == CENTER_LIMIT
    assert artifact['log_size_limit'] == LOG_SIZE_LIMIT
    model = ResidualNet(
        input_dim=artifact['input_dim'], hidden_dim=artifact['hidden_dim']
    ).to(device)
    model.load_state_dict(artifact['state_dict'])
    return model, artifact


def evaluate(args):
    lock = json.load(open(args.lock_json, encoding='utf-8'))
    assert lock['external_eval_worthy'] is True
    for key in ('joint_script', 'stage29_model', 'stage29_lock', 'model_path'):
        assert sha256(lock[key]) == lock[key + '_sha256'], key
    joint = load_module(lock['joint_script'], 'box_residual_eval_joint')
    policy = json.load(open(lock['stage29_lock'], encoding='utf-8'))
    rows = torch.load(args.dump, map_location='cpu')['rows']
    booster = lgb.Booster(model_file=lock['stage29_model'])
    arrays = collect_locked_choices(rows, joint, booster, policy)
    device = torch.device(args.device)
    model, artifact = load_model_artifact(lock['model_path'], device)
    normalized = (
        arrays['x'] - np.asarray(artifact['input_mean'], np.float32)
    ) / np.asarray(artifact['input_std'], np.float32)
    gate, delta = predict(model, normalized.astype(np.float32), device)
    selected = lock['selected_policy']
    result = policy_summary(
        joint, arrays['boxes'], arrays['gt_boxes'], arrays['base_iou'],
        gate, delta, selected['threshold'], selected['scale'],
    )
    result.update({
        'dump': os.path.abspath(args.dump),
        'dump_sha256': sha256(args.dump),
        'lock_sha256': sha256(args.lock_json),
        'model_sha256': sha256(lock['model_path']),
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
    train_parser.add_argument('train_dump')
    train_parser.add_argument('joint_script')
    train_parser.add_argument('stage29_model')
    train_parser.add_argument('stage29_lock')
    train_parser.add_argument('output_dir')
    train_parser.add_argument('--device', default='cuda')
    train_parser.add_argument('--hidden-dim', type=int, default=128)
    train_parser.add_argument('--lr', type=float, default=1e-3)
    train_parser.add_argument('--epochs', type=int, default=80)
    train_parser.add_argument('--patience', type=int, default=15)
    train_parser.add_argument('--batch-size', type=int, default=512)
    eval_parser = sub.add_parser('evaluate')
    eval_parser.add_argument('dump')
    eval_parser.add_argument('lock_json')
    eval_parser.add_argument('output_json')
    eval_parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        raise AssertionError(args.command)


if __name__ == '__main__':
    main()
