"""Candidate-ranking metrics on the frozen Full42 (Stage150 E13) candidate pool.

For each score source, over all 9508 ScanRefer val samples and the 256 predicted
queries per sample: mean target-candidate rank r_tau (no valid candidate -> Q+1),
Top-k target inclusion, observed Acc (Top-1), oracle Acc and the
positive-negative score margin (raw and z-scored within the source).
A candidate is valid at tau when IoU > tau, matching the official Acc counting.
"""

import json
import sys

import torch

SOURCES = ('base', 'structured', 'fused', 'quality', 'detector_policy_adapter')
TAUS = (0.25, 0.5)
KS = (1, 3, 5)
NEG_IOU = 0.10


def main(dump_path, output_path):
    rows = torch.load(dump_path, map_location='cpu')['rows']
    assert len(rows) == 9508, len(rows)
    iou = torch.stack([row['all_query_iou'] for row in rows])
    unique = torch.tensor([row['is_unique_label_only'] for row in rows])
    groups = {'overall': torch.ones_like(unique), 'unique': unique, 'multiple': ~unique}
    Q = iou.shape[1]
    result = {'count': len(rows), 'queries': Q, 'valid_rule': 'iou > tau',
              'groups': {name: int(mask.sum()) for name, mask in groups.items()},
              'sources': {}}
    for source in SOURCES:
        scores = torch.stack([row['all_query_scores'][source] for row in rows])
        order = scores.argsort(dim=1, descending=True)
        sorted_iou = iou.gather(1, order)
        z = (scores - scores.mean()) / scores.std()
        neg_mask = iou <= NEG_IOU
        per_group = {}
        for group, mask in groups.items():
            stats = {}
            for tau in TAUS:
                valid = sorted_iou[mask] > tau
                has = valid.any(dim=1)
                rank = torch.where(has, valid.float().argmax(dim=1) + 1,
                                   torch.full_like(has, Q + 1, dtype=torch.long))
                pos = torch.where(iou[mask] > tau, scores[mask], torch.full_like(scores[mask], float('-inf'))).max(dim=1).values
                neg = torch.where(neg_mask[mask], scores[mask], torch.full_like(scores[mask], float('-inf'))).max(dim=1).values
                pos_z = torch.where(iou[mask] > tau, z[mask], torch.full_like(z[mask], float('-inf'))).max(dim=1).values
                neg_z = torch.where(neg_mask[mask], z[mask], torch.full_like(z[mask], float('-inf'))).max(dim=1).values
                both = torch.isfinite(pos) & torch.isfinite(neg)
                stats[str(tau)] = {
                    'acc_observed': 100.0 * valid[:, 0].float().mean().item(),
                    'acc_oracle': 100.0 * has.float().mean().item(),
                    'no_valid_candidate_pct': 100.0 * (~has).float().mean().item(),
                    'mean_rank_all': rank.float().mean().item(),
                    'mean_rank_valid_only': rank[has].float().mean().item(),
                    'median_rank_valid_only': rank[has].float().median().item(),
                    **{f'top{k}_inclusion': 100.0 * valid[:, :k].any(dim=1).float().mean().item() for k in KS},
                    'margin_raw_mean': (pos - neg)[both].mean().item(),
                    'margin_z_mean': (pos_z - neg_z)[both].mean().item(),
                    'margin_samples': int(both.sum()),
                }
            per_group[group] = stats
        result['sources'][source] = per_group
    with open(output_path, 'w') as handle:
        json.dump(result, handle, indent=2)
    for source in SOURCES:
        for tau in TAUS:
            s = result['sources'][source]['overall'][str(tau)]
            print(f"{source:24s} tau={tau}: acc={s['acc_observed']:.2f} oracle={s['acc_oracle']:.2f} "
                  f"rank={s['mean_rank_all']:.2f} rank_valid={s['mean_rank_valid_only']:.2f} "
                  f"top1/3/5={s['top1_inclusion']:.2f}/{s['top3_inclusion']:.2f}/{s['top5_inclusion']:.2f} "
                  f"margin_z={s['margin_z_mean']:.3f}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
