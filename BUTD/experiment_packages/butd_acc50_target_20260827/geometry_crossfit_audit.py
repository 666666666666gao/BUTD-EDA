#!/usr/bin/env python
"""Cross-fitted diagnostic for deployable query/detector box calibration."""

import argparse
import json

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier


ALPHAS = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)


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


def build_arrays(e6_rows, geometry_rows):
    geometry_by_id = {int(row['example_id']): row for row in geometry_rows}
    pred_boxes = []
    matched_boxes = []
    gt_boxes = []
    supports = []
    features = []
    example_ids = []
    missing_detector = 0

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
        matched = pred.clone()
        support = 0.0
        matched_confidence = 0.0
        matched_class = 0.0
        if detected and len(detected) == len(confidence):
            det = torch.tensor(detected, dtype=torch.float32)
            conf = torch.tensor(confidence, dtype=torch.float32)
            pred_repeat = pred.view(1, 6).expand(det.shape[0], -1)
            overlap = aligned_iou(pred_repeat, det)
            matched_index = int((overlap * conf).argmax())
            matched = det[matched_index]
            support = float(overlap[matched_index])
            matched_confidence = float(conf[matched_index])
            if len(class_ids) == len(detected):
                matched_class = float(int(class_ids[matched_index]) == target_cid)
        else:
            missing_detector += 1

        hit50 = np.asarray(
            e6_row['adapter_hit50_logit_at_candidate'], dtype=np.float32
        )
        hit25 = np.asarray(
            e6_row['adapter_hit25_logit_at_candidate'], dtype=np.float32
        )
        rescue = np.asarray(
            e6_row['adapter_rescue_logit_at_candidate'], dtype=np.float32
        )
        fused = np.asarray(
            e6_row['adapter_fused_at_candidate'], dtype=np.float32
        )
        center_delta = (
            (matched[:3] - pred[:3]) / pred[3:].abs().clamp(min=1e-3)
        )
        size_ratio = torch.log(
            (matched[3:].abs() + 1e-4) / (pred[3:].abs() + 1e-4)
        ).clamp(min=-4.0, max=4.0)
        sorted_hit50 = np.sort(hit50)[::-1]
        sorted_fused = np.sort(fused)[::-1]
        features.append([
            support,
            matched_confidence,
            min(len(detected), 50) / 50.0,
            matched_class,
            float(hit50[position]),
            float(sorted_hit50[0] - sorted_hit50[1]),
            float(hit25[position]),
            float(rescue[position]),
            float(fused[position]),
            float(sorted_fused[0] - sorted_fused[1]),
            float(geometry_row.get('is_unique_label_only', False)),
            *center_delta.tolist(),
            *size_ratio.tolist(),
        ])
        pred_boxes.append(pred)
        matched_boxes.append(matched)
        gt_boxes.append(gt)
        supports.append(support)
        example_ids.append(int(e6_row['example_id']))

    pred_boxes = torch.stack(pred_boxes)
    matched_boxes = torch.stack(matched_boxes)
    gt_boxes = torch.stack(gt_boxes)
    supports = torch.tensor(supports)
    action_boxes = (
        pred_boxes[:, None, :]
        + torch.tensor(ALPHAS)[None, :, None]
        * (matched_boxes - pred_boxes)[:, None, :]
    )
    action_boxes = torch.where(
        (supports >= 0.30)[:, None, None],
        action_boxes,
        pred_boxes[:, None, :],
    )
    action_iou = aligned_iou(action_boxes, gt_boxes[:, None, :]).numpy()
    return {
        'ids': np.asarray(example_ids),
        'features': np.asarray(features, dtype=np.float32),
        'supports': supports.numpy(),
        'action_iou': action_iou,
        'missing_detector': missing_detector,
    }


def metrics(action_iou, actions, mask=None):
    if mask is None:
        mask = np.ones(len(actions), dtype=bool)
    selected = action_iou[np.arange(len(actions)), actions][mask]
    return {
        'acc025': float((selected >= 0.25).mean()),
        'acc050': float((selected >= 0.50).mean()),
        'count': int(mask.sum()),
    }


def oracle_labels(action_iou):
    utility = (
        4.0 * (action_iou >= 0.50)
        + 1.0 * (action_iou >= 0.25)
        + action_iou
    )
    return utility.argmax(axis=1), utility


def fit_piecewise(train, test, support, confidence, utility):
    support_edges = np.asarray([0.30, 0.45, 0.60, 0.75, 0.90, 2.0])
    confidence_edges = np.asarray([0.0, 0.50, 0.90, 1.01])
    actions = np.full(len(train), 3, dtype=np.int64)
    for support_low, support_high in zip(
        support_edges[:-1], support_edges[1:]
    ):
        for conf_low, conf_high in zip(
            confidence_edges[:-1], confidence_edges[1:]
        ):
            train_cell = (
                train
                & (support >= support_low)
                & (support < support_high)
                & (confidence >= conf_low)
                & (confidence < conf_high)
            )
            test_cell = (
                test
                & (support >= support_low)
                & (support < support_high)
                & (confidence >= conf_low)
                & (confidence < conf_high)
            )
            if train_cell.sum() < 40:
                continue
            action = int(utility[train_cell].mean(axis=0).argmax())
            actions[test_cell] = action
    actions[test & (support < 0.30)] = 0
    return actions


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
    arrays = build_arrays(e6_rows, geometry_rows)
    ids = arrays['ids']
    features = arrays['features']
    support = arrays['supports']
    action_iou = arrays['action_iou']
    confidence = features[:, 1]
    labels, utility = oracle_labels(action_iou)
    even = (ids % 2) == 0
    odd = ~even

    static = {
        str(float(alpha)): metrics(
            action_iou, np.full(len(ids), index, dtype=np.int64)
        )
        for index, alpha in enumerate(ALPHAS)
    }
    oracle = metrics(action_iou, labels)

    crossfit_hgb = np.full(len(ids), 3, dtype=np.int64)
    crossfit_piecewise = np.full(len(ids), 3, dtype=np.int64)
    for train, test in ((even, odd), (odd, even)):
        classifier = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=2.0,
            random_state=0,
        )
        classifier.fit(features[train], labels[train])
        crossfit_hgb[test] = classifier.predict(features[test])
        piecewise_actions = fit_piecewise(
            train, test, support, confidence, utility
        )
        crossfit_piecewise[test] = piecewise_actions[test]

    result = {
        'num_rows': int(len(ids)),
        'missing_detector_rows': int(arrays['missing_detector']),
        'static': static,
        'oracle': oracle,
        'crossfit_hgb': {
            'overall': metrics(action_iou, crossfit_hgb),
            'even': metrics(action_iou, crossfit_hgb, even),
            'odd': metrics(action_iou, crossfit_hgb, odd),
            'action_counts': np.bincount(
                crossfit_hgb, minlength=len(ALPHAS)
            ).tolist(),
        },
        'crossfit_piecewise': {
            'overall': metrics(action_iou, crossfit_piecewise),
            'even': metrics(action_iou, crossfit_piecewise, even),
            'odd': metrics(action_iou, crossfit_piecewise, odd),
            'action_counts': np.bincount(
                crossfit_piecewise, minlength=len(ALPHAS)
            ).tolist(),
        },
    }
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
