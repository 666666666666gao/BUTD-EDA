#!/usr/bin/env python3
"""Register the external baseline and the accepted optimized Full result."""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


BASELINE = {
    'source': 'BUTD-DETR published paper values corrected by the user on 2026-08-24; external baseline, not retrained',
    'unique_025': 82.88,
    'unique_050': 64.98,
    'multiple_025': 44.73,
    'multiple_050': 33.97,
    'overall_025': 50.42,
    'overall_050': 38.60,
}

FULL = {
    'source': (
        'Accepted optimized M3 Full checkpoint reused without retraining; '
        'paper-facing micro-tune continued from the prior complete Full model'
    ),
    'run_dir': '/home/gb/new butd/butd_detr-main/logs/butd_universal_target/three_targets_20260820/scanrefer_microtune_lr2e5_e6/scanrefer_spacy/1787171156',
    'checkpoint': '/home/gb/new butd/butd_detr-main/logs/butd_universal_target/three_targets_20260820/scanrefer_microtune_lr2e5_e6/scanrefer_spacy/1787171156/ckpt_best_primary.pth',
    'checkpoint_sha256': 'a60000dfc1163b4b80ef51b68d8fd985448f3604d319891146edd8640e31b8d1',
    'epoch': 3,
    'unique_025': 87.4559549,
    'unique_050': 67.01902748,
    'multiple_025': 48.59685993,
    'multiple_050': 35.14649524,
    'overall_025': 54.3962978544,
    'overall_050': 39.9032393379,
    'sacr_rank_loss_weight': 0.0,
    'rapf_quality_weight': 0.25,
    'retrained_for_current_ablation_plan': False,
}

MATCHED_PROTOCOL_FULL = {
    'source': (
        'Independent official-init 65-epoch Full checkpoint retained as the '
        'matched-protocol control for the SACR/RAPF internal tables'
    ),
    'run_dir': '/home/gb/new butd/butd_detr-main/logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_from_official_init/02_full_sacr_rapf_qahnl/scanrefer_spacy/1786908904',
    'checkpoint': '/home/gb/new butd/butd_detr-main/logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_from_official_init/02_full_sacr_rapf_qahnl/scanrefer_spacy/1786908904/ckpt_best_primary.pth',
    'checkpoint_sha256': 'f12c34ce2a94c43694759101c4f5b5c9c43eb5b639787e818df8a5072dfa0ee4',
    'epoch': 65,
    'unique_025': 85.97603946,
    'unique_050': 65.89147287,
    'multiple_025': 48.10236123,
    'multiple_050': 35.19594511,
    'overall_025': 53.7547328565,
    'overall_050': 39.77702987,
    'sacr_rank_loss_weight': 0.0,
    'rapf_quality_weight': 0.75,
    'retrained_for_current_ablation_plan': False,
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt-dir', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = Path(FULL['checkpoint'])
    matched_checkpoint = Path(MATCHED_PROTOCOL_FULL['checkpoint'])
    for label, path, expected in (
        ('accepted optimized Full', checkpoint, FULL['checkpoint_sha256']),
        (
            'matched-protocol Full',
            matched_checkpoint,
            MATCHED_PROTOCOL_FULL['checkpoint_sha256'],
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise AssertionError('{} checkpoint SHA mismatch'.format(label))
    receipt = {
        'status': 'PASS',
        'recorded_at': datetime.now().astimezone().isoformat(),
        'baseline': BASELINE,
        'full': dict(FULL, checkpoint_size=checkpoint.stat().st_size),
        'matched_protocol_full': dict(
            MATCHED_PROTOCOL_FULL,
            checkpoint_size=matched_checkpoint.stat().st_size,
        ),
        'causal_boundary': (
            'M0 is external. M1/M2 start independently from the official detector and use '
            'SACR rank supervision because QAHNL is absent. Main-table M3 is the user-accepted '
            'optimized Full checkpoint continued from a prior complete Full model. SACR/RAPF '
            'internal rows instead use the separate official-init 65-epoch Full checkpoint as '
            'their matched-protocol control. Both checkpoints are reused without retraining, '
            'and the two evidence tiers must not be silently merged.'
        ),
    }
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    output = args.receipt_dir / 'known_baseline_and_formal_full.json'
    temporary = output.with_name(output.name + '.tmp.{}'.format(os.getpid()))
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    os.replace(str(temporary), str(output))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
