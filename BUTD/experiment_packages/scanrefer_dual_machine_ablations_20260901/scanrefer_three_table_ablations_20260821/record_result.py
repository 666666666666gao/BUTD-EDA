#!/usr/bin/env python3
"""Audit one completed three-table row and update the remote handoff."""

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path


METRIC_KEYS = {
    'u25': 'last__bbs_unique_acc0.25_top1',
    'u50': 'last__bbs_unique_acc0.50_top1',
    'm25': 'last__bbs_multiple_acc0.25_top1',
    'm50': 'last__bbs_multiple_acc0.50_top1',
    'o25': 'last__bbs_acc0.25_top1',
    'o50': 'last__bbs_acc0.50_top1',
}

MAIN_LABELS = {'M1': '(b) +SACR', 'M2': '(c) +RAPF'}
SACR_LABELS = {
    'S0': 'S0 w/o target-attribute',
    'S1': 'S1 w/o relation-anchor',
    'S2': 'S2 w/o pairwise geometry',
    'S3': 'S3 hard top-1 anchor',
}
RAPF_LABELS = {
    'R0': 'R0 fixed fusion (g=0.1)',
    'R1': 'R1 w/o query-quality cue',
    'R2': 'R2 w/o parser/anchor gate cues',
    'R3': 'R3 w/o gate supervision',
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_scalars(path):
    scalars = {}
    pattern = re.compile(r'^([^:]+):\s*([-+0-9.eE]+)\s*$')
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        match = pattern.match(line.strip())
        if match:
            try:
                scalars[match.group(1)] = float(match.group(2))
            except ValueError:
                pass
    return scalars


def percent(value):
    return 100.0 * value if abs(value) <= 1.5 else value


def f2(value):
    return '{:.2f}'.format(percent(value))


def assert_close(name, actual, expected, tol=1e-9):
    if actual is None or abs(float(actual) - expected) > tol:
        raise AssertionError('{} expected {}, got {}'.format(name, expected, actual))


def assert_flag(config, key, expected=True):
    if bool(config.get(key)) != bool(expected):
        raise AssertionError('{} expected {}, got {}'.format(key, expected, config.get(key)))


def validate_config(row, config):
    if config.get('rng_seed') != 0 or config.get('batch_size') != 24:
        raise AssertionError('seed/batch mismatch')
    if config.get('max_epoch') != 65 or config.get('val_freq') != 5:
        raise AssertionError('epoch/validation protocol mismatch')
    if config.get('lr_decay_epochs') != [55, 60]:
        raise AssertionError('LR decay mismatch')
    assert_close('lr', config.get('lr'), 1e-4)
    assert_close('lr_backbone', config.get('lr_backbone'), 1e-3)
    if config.get('pp_checkpoint') != '/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth':
        raise AssertionError('not official detector initialization')
    if config.get('checkpoint_path') not in (None, ''):
        raise AssertionError('row did not train independently from official init')
    assert_flag(config, 'best_checkpoint_only', True)
    if config.get('best_checkpoint_metric') != 'last__bbs_acc0.25_top1':
        raise AssertionError('best-checkpoint metric mismatch')
    for forbidden in ('use_source_pool_selector', 'use_detector_policy_adapter'):
        assert_flag(config, forbidden, False)

    if row == 'M1':
        assert_flag(config, 'use_sacr', True)
        assert_flag(config, 'use_rapf', False)
        assert_flag(config, 'use_qahnl', False)
        assert_flag(config, 'eval_use_structured_scores', True)
        assert_close('sacr_rank_loss_weight', config.get('sacr_rank_loss_weight'), 0.2)
        return
    if row == 'M2':
        assert_flag(config, 'use_sacr', True)
        assert_flag(config, 'use_rapf', True)
        assert_flag(config, 'use_qahnl', False)
        assert_flag(config, 'use_reliability_gate', True)
        assert_flag(config, 'eval_use_fused_scores', True)
        assert_close('sacr_rank_loss_weight', config.get('sacr_rank_loss_weight'), 0.2)
        assert_close('rapf_quality_weight', config.get('rapf_quality_weight'), 0.75)
        return

    assert_flag(config, 'use_sacr', True)
    assert_flag(config, 'use_rapf', True)
    assert_flag(config, 'use_qahnl', True)
    assert_flag(config, 'eval_use_fused_scores', True)
    if config.get('qahnl_score_source') != 'fused':
        raise AssertionError('QAHNL source must remain fused')
    assert_close('sacr_rank_loss_weight', config.get('sacr_rank_loss_weight'), 0.0)

    if row in SACR_LABELS:
        assert_close('rapf_quality_weight', config.get('rapf_quality_weight'), 0.75)
        if row == 'S0':
            assert_flag(config, 'sacr_disable_target_attr', True)
            assert_flag(config, 'sacr_disable_relation', False)
        elif row == 'S1':
            assert_flag(config, 'sacr_disable_relation', True)
            assert_flag(config, 'sacr_disable_target_attr', False)
        elif row == 'S2':
            if config.get('sacr_geo_dim') != 0:
                raise AssertionError('S2 must use sacr_geo_dim=0')
        elif row == 'S3':
            if config.get('sacr_anchor_aggregation') != 'hard':
                raise AssertionError('S3 must use hard anchor aggregation')
        return

    if row == 'R0':
        assert_flag(config, 'use_reliability_gate', False)
        assert_flag(config, 'use_quality_head', True)
        assert_flag(config, 'rapf_use_quality', True)
        assert_close('rapf_fixed_alpha', config.get('rapf_fixed_alpha'), 0.1)
        assert_close('rapf_gate_loss_weight', config.get('rapf_gate_loss_weight'), 0.0)
        assert_flag(config, 'rapf_disable_safety', False)
        assert_flag(config, 'rapf_disable_residual_clipping', False)
    elif row == 'R1':
        assert_flag(config, 'use_quality_head', False)
        assert_flag(config, 'rapf_use_quality', False)
        assert_flag(config, 'use_reliability_gate', True)
    elif row == 'R2':
        assert_flag(config, 'rapf_disable_parser_anchor_features', True)
        assert_flag(config, 'rapf_disable_agreement_features', False)
        assert_flag(config, 'rapf_disable_safety', False)
        assert_flag(config, 'use_quality_head', True)
    elif row == 'R3':
        assert_flag(config, 'use_reliability_gate', True)
        assert_close('rapf_gate_loss_weight', config.get('rapf_gate_loss_weight'), 0.0)
    else:
        raise AssertionError('unsupported row {}'.format(row))


def replace_row(text, label, replacement):
    begin_marker = '<!-- SIMPLIFIED_ABLATION_20260821_BEGIN -->'
    end_marker = '<!-- SIMPLIFIED_ABLATION_20260821_END -->'
    begin = text.find(begin_marker)
    end = text.find(end_marker)
    if begin < 0 or end < 0 or end <= begin:
        raise AssertionError('authoritative three-table handoff section missing')
    prefix = text[:begin]
    section = text[begin:end]
    suffix = text[end:]
    pattern = re.compile(r'^\| ' + re.escape(label) + r' \|.*$', re.MULTILINE)
    section, count = pattern.subn(replacement, section, count=1)
    if count != 1:
        raise AssertionError('expected exactly one table row {}, got {}'.format(label, count))
    return prefix + section + suffix


def update_handoff(path, row, metrics, receipt):
    text = path.read_text(encoding='utf-8')
    if row in MAIN_LABELS:
        label = MAIN_LABELS[row]
        marks = ('✓', '', '') if row == 'M1' else ('✓', '✓', '')
        values = ' | '.join(f2(metrics[key]) for key in ('u25', 'u50', 'm25', 'm50', 'o25', 'o50'))
        replacement = '| {} | {} | {} | {} | {} |'.format(label, marks[0], marks[1], marks[2], values)
    else:
        labels = dict(SACR_LABELS)
        labels.update(RAPF_LABELS)
        label = labels[row]
        values = ' | '.join(f2(metrics[key]) for key in ('m25', 'm50', 'o25', 'o50'))
        replacement = '| {} | {} |'.format(label, values)
    text = replace_row(text, label, replacement)

    marker = '<!-- THREE_TABLE_RESULT_{}_{} -->'.format(row, receipt['checkpoint_sha256'])
    if marker not in text:
        metric_text = ', '.join('{}={:.10f}'.format(key, metrics[key]) for key in (
            'u25', 'u50', 'm25', 'm50', 'o25', 'o50'
        ))
        text += (
            '\n\n{}\n### Three-table row {} completed - {}\n\n'
            '- Run directory: `{}`\n'
            '- Selected epoch / Overall Acc@0.25: {} / {:.10f}\n'
            '- Official BBS metrics: {}\n'
            '- Best checkpoint: `{}`\n'
            '- Checkpoint/config SHA256: `{}` / `{}`\n'
            '- Final reload, fatal scan, protocol, and one-weight gates: PASS.\n'
        ).format(
            marker, row, receipt['recorded_at'], receipt['run_dir'],
            receipt['selected_epoch'], receipt['selected_score'], metric_text,
            receipt['checkpoint'], receipt['checkpoint_sha256'],
            receipt['config_sha256'],
        )
    temporary = path.with_name(path.name + '.tmp.{}'.format(os.getpid()))
    temporary.write_text(text, encoding='utf-8')
    os.replace(str(temporary), str(path))


def main():
    rows = sorted(list(MAIN_LABELS) + list(SACR_LABELS) + list(RAPF_LABELS))
    parser = argparse.ArgumentParser()
    parser.add_argument('row', choices=rows)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--receipt-dir', type=Path, required=True)
    args = parser.parse_args()

    run = args.run_dir.resolve()
    selected_path = run / 'best_primary.json'
    config_path = run / 'config.json'
    final_eval = run / 'eval_epoch_last.log'
    for required in (selected_path, config_path, final_eval):
        if not required.is_file():
            raise FileNotFoundError(required)
    selected = json.loads(selected_path.read_text(encoding='utf-8'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_config(args.row, config)

    checkpoint = Path(selected['checkpoint']).resolve()
    if not checkpoint.is_file() or checkpoint.parent != run:
        raise AssertionError('selected checkpoint missing or outside run')
    weights = sorted(
        path for path in run.rglob('*')
        if path.is_file() and path.suffix.lower() in ('.pth', '.pt', '.ckpt')
    )
    if weights != [checkpoint]:
        raise AssertionError('expected exactly one selected weight, found {}'.format(weights))

    scalars = parse_scalars(final_eval)
    metrics = {short: scalars[key] for short, key in METRIC_KEYS.items()}
    if abs(metrics['o25'] - float(selected['score'])) > 5e-8:
        raise AssertionError('final reload does not match selected best score')
    fatal = re.compile(
        r'Traceback|CUDA out of memory|RuntimeError|AssertionError|Killed|No space left|\bnan\b|\binf\b',
        re.IGNORECASE,
    )
    scanned = []
    for candidate in (run / 'log.txt', final_eval):
        if candidate.is_file():
            scanned.append(str(candidate))
            if fatal.search(candidate.read_text(encoding='utf-8', errors='replace')):
                raise AssertionError('fatal signature in {}'.format(candidate))

    receipt = {
        'status': 'PASS',
        'row': args.row,
        'run_dir': str(run),
        'selected_epoch': int(selected['epoch']),
        'selected_score': float(selected['score']),
        'metrics': metrics,
        'checkpoint': str(checkpoint),
        'checkpoint_size': checkpoint.stat().st_size,
        'checkpoint_sha256': sha256(checkpoint),
        'config_sha256': sha256(config_path),
        'fatal_logs_scanned': scanned,
        'retained_weight_count': 1,
        'recorded_at': datetime.now().astimezone().isoformat(),
    }
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    output = args.receipt_dir / '{}.json'.format(args.row)
    temporary = output.with_name(output.name + '.tmp.{}'.format(os.getpid()))
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    os.replace(str(temporary), str(output))
    update_handoff(args.handoff, args.row, metrics, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
