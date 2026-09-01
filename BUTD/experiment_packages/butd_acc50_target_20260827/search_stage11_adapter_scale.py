import json
import sys

import torch


dump_path, output_path = sys.argv[1:]
rows = torch.load(dump_path, map_location='cpu')['rows']
scales = [index / 10.0 for index in range(0, 201)]
gaps = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]
count_caps = [None, 1, 2, 3, 5]
policies = []

def summarize(selected, scale, gap, cap):
    hit25 = [float(iou >= 0.25) for iou in selected]
    hit50 = [float(iou >= 0.50) for iou in selected]
    even = list(range(0, len(rows), 2)); odd = list(range(1, len(rows), 2))
    mean = lambda values, ids: sum(values[i] for i in ids) / len(ids)
    return {
        'scale': scale, 'min_score_gain': gap, 'detector_count_cap': cap,
        'overall_acc0.25': sum(hit25) / len(hit25),
        'overall_acc0.50': sum(hit50) / len(hit50),
        'even_acc0.25': mean(hit25, even), 'even_acc0.50': mean(hit50, even),
        'odd_acc0.25': mean(hit25, odd), 'odd_acc0.50': mean(hit50, odd),
    }

for scale in scales:
    per_row = []
    for row in rows:
        fused = torch.tensor(row['adapter_fused_at_candidate'])
        delta = torch.tensor(row['adapter_delta_at_candidate'])
        iou = row['adapter_iou_at_candidate']
        fallback_iou = float(row['fused_top']['iou'])
        fallback_query = int(row['fused_top']['query'])
        candidate_queries = row['adapter_candidate_query']
        if fused.numel() == 0 or fallback_query not in candidate_queries:
            per_row.append((
                [fallback_iou], 0, 0, 0.0,
                row['detector_class_count'] if row['detector_class_count'] is not None else 999
            ))
            continue
        fallback = candidate_queries.index(fallback_query)
        scores = fused + (scale / 4.0) * delta
        chosen = int(scores.argmax())
        gain = float(scores[chosen] - scores[fallback])
        count = row['detector_class_count']
        per_row.append((iou, fallback, chosen, gain, count if count is not None else 999))
    for gap in gaps:
        for cap in count_caps:
            selected = []
            for iou, fallback, chosen, gain, count in per_row:
                use = gain >= gap and (cap is None or count <= cap)
                selected.append(float(iou[chosen if use else fallback]))
            policies.append(summarize(selected, scale, gap, cap))

for row in policies:
    row['pass_acc0.25'] = row['overall_acc0.25'] > 0.5391
    row['pass_acc0.50'] = row['overall_acc0.50'] > 0.4241
    row['goal_achieved'] = row['pass_acc0.25'] and row['pass_acc0.50']
    row['joint_margin'] = min(
        row['overall_acc0.25'] - 0.5391,
        row['overall_acc0.50'] - 0.4241,
    )
    row['robust_margin'] = min(
        row['even_acc0.25'] - 0.5391, row['even_acc0.50'] - 0.4241,
        row['odd_acc0.25'] - 0.5391, row['odd_acc0.50'] - 0.4241,
    )
feasible = [row for row in policies if row['goal_achieved']]
best = max(policies, key=lambda row: (
    row['goal_achieved'], row['robust_margin'], row['joint_margin'],
    row['overall_acc0.25'], row['overall_acc0.50']))
payload = {
    'num_examples': len(rows), 'num_policies': len(policies),
    'num_feasible': len(feasible), 'best': best,
    'best_feasible': max(feasible, key=lambda row: (
        row['robust_margin'], row['overall_acc0.25'], row['overall_acc0.50']))
        if feasible else None,
}
with open(output_path, 'w') as f:
    json.dump(payload, f, indent=2, sort_keys=True); f.write('\n')
print(json.dumps(payload, indent=2, sort_keys=True))
