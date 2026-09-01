#!/usr/bin/env python
"""Cross-fit semantic override gates jointly with fixed geometry actions."""

import argparse
import json

import numpy as np
import torch


ALPHAS = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
COMBOS = {
    'hit50': (1.0, 0.0, 0.0, 0.0),
    'combo_025': (1.0, 0.25, 0.25, 0.25),
    'combo_hit25_05': (1.0, 0.50, 0.25, 0.25),
    'combo_fused_05': (1.0, 0.25, 0.50, 0.25),
    'combo_rescue_05': (1.0, 0.25, 0.25, 0.50),
    'combo_hit25_neg025': (1.0, -0.25, 0.25, 0.25),
}
GATE_THRESHOLDS = (-1e9, 0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)
SUPPORT_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50)


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


def standardize(values):
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean()) / (values.std() + 1e-6)


def build_records(e6_rows, geometry_rows):
    geometry_by_id = {int(row['example_id']): row for row in geometry_rows}
    records = []
    for e6_row in e6_rows:
        geometry_row = geometry_by_id[int(e6_row['example_id'])]
        queries = list(e6_row['adapter_candidate_query'])
        geometry_positions = {
            int(query): index
            for index, query in enumerate(
                geometry_row['adapter_candidate_query']
            )
        }
        fallback_query = int(e6_row['adapter_fallback_query'])
        fallback_position = queries.index(fallback_query)
        components = np.stack([
            standardize(e6_row['adapter_hit50_logit_at_candidate']),
            standardize(e6_row['adapter_hit25_logit_at_candidate']),
            standardize(e6_row['adapter_fused_at_candidate']),
            standardize(e6_row['adapter_rescue_logit_at_candidate']),
        ], axis=1)
        gt = torch.tensor(geometry_row['gt_box'], dtype=torch.float32)
        detected = geometry_row.get('detected_box') or []
        confidence = geometry_row.get('detected_target_confidence') or []
        det = (
            torch.tensor(detected, dtype=torch.float32)
            if detected and len(detected) == len(confidence)
            else None
        )
        conf = (
            torch.tensor(confidence, dtype=torch.float32)
            if det is not None else None
        )
        candidate_action_iou = []
        candidate_support = []
        for query in queries:
            pred = torch.tensor(
                geometry_row['adapter_box_at_candidate'][
                    geometry_positions[int(query)]
                ],
                dtype=torch.float32,
            )
            matched = pred
            support = 0.0
            if det is not None:
                pair_iou = aligned_iou(
                    pred.view(1, 6).expand_as(det), det
                )
                matched_index = int((pair_iou * conf).argmax())
                matched = det[matched_index]
                support = float(pair_iou[matched_index])
            action_boxes = (
                pred.view(1, 6)
                + torch.tensor(ALPHAS).view(-1, 1)
                * (matched - pred).view(1, 6)
            )
            candidate_action_iou.append(
                aligned_iou(action_boxes, gt.view(1, 6)).numpy()
            )
            candidate_support.append(support)
        records.append({
            'id': int(e6_row['example_id']),
            'components': components,
            'fallback_position': fallback_position,
            'action_iou': np.stack(candidate_action_iou),
            'support': np.asarray(candidate_support, dtype=np.float32),
        })
    return records


def decisions(records, policy):
    combo = np.asarray(COMBOS[policy['combo']], dtype=np.float32)
    chosen_iou = np.zeros(len(records), dtype=np.float32)
    for index, record in enumerate(records):
        scores = record['components'] @ combo
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        fallback = record['fallback_position']
        delta_fallback = float(scores[best] - scores[fallback])
        top_margin = float(scores[best] - scores[order[1]])
        gate_value = {
            'fallback_delta': delta_fallback,
            'top_margin': top_margin,
            'minimum': min(delta_fallback, top_margin),
        }[policy['gate']]
        selected = (
            best if gate_value >= policy['gate_threshold'] else fallback
        )
        action = policy['action']
        if record['support'][selected] < policy['support_threshold']:
            action = 0
        chosen_iou[index] = record['action_iou'][selected, action]
    return chosen_iou


def metrics(ious, mask):
    values = ious[mask]
    return {
        'acc025': float((values >= 0.25).mean()),
        'acc050': float((values >= 0.50).mean()),
        'count': int(mask.sum()),
    }


def policy_rank(result):
    feasible = result['acc025'] > 0.5391
    return (int(feasible), result['acc050'], result['acc025'])


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
    records = build_records(e6_rows, geometry_rows)
    ids = np.asarray([record['id'] for record in records])
    masks = {
        'all': np.ones(len(records), dtype=bool),
        'even': (ids % 2) == 0,
        'odd': (ids % 2) == 1,
    }

    evaluated = []
    for combo_name in COMBOS:
        for gate in ('fallback_delta', 'top_margin', 'minimum'):
            for gate_threshold in GATE_THRESHOLDS:
                for action in range(len(ALPHAS)):
                    for support_threshold in SUPPORT_THRESHOLDS:
                        policy = {
                            'combo': combo_name,
                            'gate': gate,
                            'gate_threshold': gate_threshold,
                            'action': action,
                            'alpha': float(ALPHAS[action]),
                            'support_threshold': support_threshold,
                        }
                        ious = decisions(records, policy)
                        result = {
                            'policy': policy,
                            'all': metrics(ious, masks['all']),
                            'even': metrics(ious, masks['even']),
                            'odd': metrics(ious, masks['odd']),
                        }
                        evaluated.append((result, ious))

    top = sorted(
        (result for result, _ in evaluated),
        key=lambda value: policy_rank(value['all']),
        reverse=True,
    )[:20]
    best_even_index = max(
        range(len(evaluated)),
        key=lambda idx: policy_rank(evaluated[idx][0]['even']),
    )
    best_odd_index = max(
        range(len(evaluated)),
        key=lambda idx: policy_rank(evaluated[idx][0]['odd']),
    )
    even_selected = evaluated[best_odd_index][1]
    odd_selected = evaluated[best_even_index][1]
    crossfit_iou = np.where(masks['even'], even_selected, odd_selected)
    crossfit = {
        'overall': metrics(crossfit_iou, masks['all']),
        'even': metrics(crossfit_iou, masks['even']),
        'odd': metrics(crossfit_iou, masks['odd']),
        'policy_trained_on_even': evaluated[best_even_index][0]['policy'],
        'policy_trained_on_odd': evaluated[best_odd_index][0]['policy'],
    }
    result = {
        'num_records': len(records),
        'num_policies': len(evaluated),
        'best_full': top[0],
        'top20': top,
        'crossfit': crossfit,
    }
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
