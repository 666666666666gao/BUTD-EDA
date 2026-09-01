#!/usr/bin/env python
"""Cross-fitted audit of learned query-to-detector box matching."""

import argparse
import json

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor


ALPHAS = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])


def xyzxyz(boxes):
    center = boxes[..., :3]
    size = boxes[..., 3:].abs().clamp(min=1e-6)
    return torch.cat([center - size / 2, center + size / 2], dim=-1)


def aligned_iou(boxes_a, boxes_b):
    a = xyzxyz(boxes_a)
    b = xyzxyz(boxes_b)
    intersection = (
        torch.minimum(a[..., 3:], b[..., 3:])
        - torch.maximum(a[..., :3], b[..., :3])
    ).clamp(min=0).prod(dim=-1)
    volume_a = (a[..., 3:] - a[..., :3]).prod(dim=-1)
    volume_b = (b[..., 3:] - b[..., :3]).prod(dim=-1)
    return intersection / (volume_a + volume_b - intersection + 1e-9)


def selected_candidate(row):
    hit50 = torch.tensor(row['adapter_hit50_logit_at_candidate'])
    fused = torch.tensor(row['adapter_fused_at_candidate'])
    standardized = (fused - fused.mean()) / (
        fused.std(unbiased=False) + 1e-6
    )
    position = int((hit50 + 1e-3 * standardized).argmax())
    return position, int(row['adapter_candidate_query'][position])


def metric(ious, mask=None):
    if mask is None:
        mask = np.ones(len(ious), dtype=bool)
    values = ious[mask]
    return {
        'acc025': float((values >= 0.25).mean()),
        'acc050': float((values >= 0.50).mean()),
        'count': int(mask.sum()),
    }


def build_examples(e6_rows, geometry_rows):
    geometry_by_id = {int(row['example_id']): row for row in geometry_rows}
    examples = []
    for e6_row in e6_rows:
        geometry_row = geometry_by_id[int(e6_row['example_id'])]
        position, query = selected_candidate(e6_row)
        geometry_position = geometry_row['adapter_candidate_query'].index(query)
        pred = torch.tensor(
            geometry_row['adapter_box_at_candidate'][geometry_position],
            dtype=torch.float32,
        )
        gt = torch.tensor(geometry_row['gt_box'], dtype=torch.float32)
        detected = geometry_row.get('detected_box') or []
        confidence = geometry_row.get('detected_target_confidence') or []
        class_ids = geometry_row.get('detected_class_id') or []
        target_cid = geometry_row.get('text_target_cid', -1)
        target_cid = -1 if target_cid is None else int(target_cid)
        hit50 = np.asarray(
            e6_row['adapter_hit50_logit_at_candidate'], dtype=np.float32
        )
        fused = np.asarray(
            e6_row['adapter_fused_at_candidate'], dtype=np.float32
        )
        hit25 = np.asarray(
            e6_row['adapter_hit25_logit_at_candidate'], dtype=np.float32
        )
        rescue = np.asarray(
            e6_row['adapter_rescue_logit_at_candidate'], dtype=np.float32
        )
        constants = np.asarray([
            float(hit50[position]),
            float(np.sort(hit50)[-1] - np.sort(hit50)[-2]),
            float(hit25[position]),
            float(rescue[position]),
            float(fused[position]),
            float(np.sort(fused)[-1] - np.sort(fused)[-2]),
            float(geometry_row.get('is_unique_label_only', False)),
            min(len(detected), 50) / 50.0,
        ], dtype=np.float32)
        example = {
            'id': int(e6_row['example_id']),
            'pred': pred,
            'gt': gt,
            'features': np.zeros((0, 22), dtype=np.float32),
            'pair_iou': np.zeros(0, dtype=np.float32),
            'blend_iou': np.zeros(0, dtype=np.float32),
            'all_action_iou': np.zeros((0, len(ALPHAS)), dtype=np.float32),
            'overlap_conf_index': -1,
        }
        if detected and len(detected) == len(confidence):
            det = torch.tensor(detected, dtype=torch.float32)
            conf = torch.tensor(confidence, dtype=torch.float32)
            pair_iou = aligned_iou(pred.view(1, 6).expand_as(det), det)
            center_delta = (
                (det[:, :3] - pred[:3])
                / pred[3:].abs().clamp(min=1e-3)
            )
            size_ratio = torch.log(
                (det[:, 3:].abs() + 1e-4)
                / (pred[3:].abs() + 1e-4)
            ).clamp(min=-4.0, max=4.0)
            center_distance = center_delta.square().sum(dim=1).sqrt()
            size_distance = size_ratio.square().sum(dim=1).sqrt()
            class_match = torch.zeros(len(det))
            if len(class_ids) == len(detected):
                class_match = torch.tensor(
                    [float(int(value) == target_cid) for value in class_ids]
                )
            repeated = torch.tensor(constants).view(1, -1).expand(len(det), -1)
            pair_features = torch.cat([
                pair_iou[:, None],
                conf[:, None],
                (pair_iou * conf)[:, None],
                class_match[:, None],
                center_distance[:, None],
                size_distance[:, None],
                center_delta,
                size_ratio,
                repeated,
            ], dim=1)
            action_boxes = (
                pred.view(1, 1, 6)
                + ALPHAS.view(1, -1, 1)
                * (det[:, None, :] - pred.view(1, 1, 6))
            )
            action_iou = aligned_iou(
                action_boxes, gt.view(1, 1, 6)
            )
            blend_iou = action_iou[:, 3]
            example.update({
                'features': pair_features.numpy(),
                'pair_iou': pair_iou.numpy(),
                'blend_iou': blend_iou.numpy(),
                'all_action_iou': action_iou.numpy(),
                'overlap_conf_index': int((pair_iou * conf).argmax()),
            })
        examples.append(example)
    return examples


def evaluate_indices(examples, chosen_indices, choose_action=False):
    selected_ious = []
    selected_actions = []
    for example, chosen in zip(examples, chosen_indices):
        if chosen < 0 or example['pair_iou'][chosen] < 0.30:
            iou = float(aligned_iou(
                example['pred'].view(1, 6), example['gt'].view(1, 6)
            )[0])
            selected_actions.append(0)
        elif choose_action:
            utilities = (
                4.0 * (example['all_action_iou'][chosen] >= 0.50)
                + 1.0 * (example['all_action_iou'][chosen] >= 0.25)
                + example['all_action_iou'][chosen]
            )
            action = int(utilities.argmax())
            iou = float(example['all_action_iou'][chosen, action])
            selected_actions.append(action)
        else:
            iou = float(example['blend_iou'][chosen])
            selected_actions.append(3)
        selected_ious.append(iou)
    return np.asarray(selected_ious), np.asarray(selected_actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('e6_dump')
    parser.add_argument('geometry_dump')
    parser.add_argument('output_json')
    args = parser.parse_args()
    e6_rows = torch.load(args.e6_dump, map_location='cpu')['rows']
    geometry_rows = torch.load(
        args.geometry_dump, map_location='cpu'
    )['rows']
    examples = build_examples(e6_rows, geometry_rows)
    ids = np.asarray([example['id'] for example in examples])
    even = (ids % 2) == 0
    odd = ~even

    static_indices = [example['overlap_conf_index'] for example in examples]
    static_iou, _ = evaluate_indices(examples, static_indices)
    oracle_indices = []
    oracle_joint_indices = []
    for example in examples:
        eligible = np.where(example['pair_iou'] >= 0.30)[0]
        if len(eligible) == 0:
            oracle_indices.append(-1)
            oracle_joint_indices.append(-1)
        else:
            oracle_indices.append(int(eligible[
                example['blend_iou'][eligible].argmax()
            ]))
            joint_utility = (
                4.0 * (example['all_action_iou'][eligible] >= 0.50)
                + 1.0 * (example['all_action_iou'][eligible] >= 0.25)
                + example['all_action_iou'][eligible]
            ).max(axis=1)
            oracle_joint_indices.append(int(eligible[joint_utility.argmax()]))
    oracle_iou, _ = evaluate_indices(examples, oracle_indices)
    oracle_joint_iou, _ = evaluate_indices(
        examples, oracle_joint_indices, choose_action=True
    )

    crossfit_indices = np.full(len(examples), -1, dtype=np.int64)
    for train_mask, test_mask in ((even, odd), (odd, even)):
        train_features = []
        train_labels = []
        train_weights = []
        for use, example in zip(train_mask, examples):
            if not use or len(example['features']) == 0:
                continue
            eligible = example['pair_iou'] >= 0.30
            if not eligible.any():
                continue
            train_features.append(example['features'][eligible])
            train_labels.append(example['blend_iou'][eligible])
            train_weights.append(np.full(
                int(eligible.sum()), 1.0 / float(eligible.sum()),
                dtype=np.float32,
            ))
        train_features = np.concatenate(train_features)
        train_labels = np.concatenate(train_labels)
        train_weights = np.concatenate(train_weights)
        regressor = HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=2.0,
            random_state=0,
        )
        regressor.fit(
            train_features, train_labels, sample_weight=train_weights
        )
        for index in np.where(test_mask)[0]:
            example = examples[index]
            eligible = np.where(example['pair_iou'] >= 0.30)[0]
            if len(eligible) == 0:
                continue
            prediction = regressor.predict(example['features'][eligible])
            crossfit_indices[index] = int(eligible[prediction.argmax()])
    crossfit_iou, _ = evaluate_indices(examples, crossfit_indices)

    result = {
        'num_examples': len(examples),
        'static_overlap_conf_alpha03': metric(static_iou),
        'oracle_detector_alpha03': metric(oracle_iou),
        'oracle_detector_and_alpha': metric(oracle_joint_iou),
        'crossfit_match_alpha03': {
            'overall': metric(crossfit_iou),
            'even': metric(crossfit_iou, even),
            'odd': metric(crossfit_iou, odd),
        },
    }
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
