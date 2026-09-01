#!/usr/bin/env python3
"""Static audit of the ten trainable launch commands."""

import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ROWS = ('M1', 'M2', 'S1', 'S2', 'S0', 'S3', 'R1', 'R3', 'R0', 'R2')


def one_value(tokens, flag):
    indexes = [i for i, value in enumerate(tokens) if value == flag]
    if len(indexes) != 1:
        raise AssertionError('{} occurs {} times'.format(flag, len(indexes)))
    return tokens[indexes[0] + 1]


def has(tokens, flag):
    return flag in tokens


def command_for(row):
    env = {'DRY_RUN': '1'}
    proc = subprocess.run(
        ['bash', str(ROOT / 'launch' / 'run_row.sh'), row],
        cwd=str(ROOT.parents[1]), env=dict(__import__('os').environ, **env),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        check=True,
    )
    return shlex.split(proc.stdout.strip())


def main():
    seen_log_dirs = set()
    for row in ROWS:
        tokens = command_for(row)
        assert one_value(tokens, '--batch_size') == '24'
        assert one_value(tokens, '--max_epoch') == '65'
        assert one_value(tokens, '--val_freq') == '5'
        assert one_value(tokens, '--rng_seed') == '0'
        assert one_value(tokens, '--lr_decay_epochs') == '55'
        decay_index = tokens.index('--lr_decay_epochs')
        assert tokens[decay_index + 1:decay_index + 3] == ['55', '60']
        assert one_value(tokens, '--pp_checkpoint') == '/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth'
        assert one_value(tokens, '--best_checkpoint_metric') == 'last__bbs_acc0.25_top1'
        assert has(tokens, '--best_checkpoint_only')
        assert not has(tokens, '--checkpoint_path')
        assert not has(tokens, '--use_source_pool_selector')
        assert not has(tokens, '--use_detector_policy_adapter')
        log_dir = one_value(tokens, '--log_dir')
        assert log_dir not in seen_log_dirs
        seen_log_dirs.add(log_dir)

        if row == 'M1':
            assert has(tokens, '--use_sacr') and not has(tokens, '--use_rapf')
            assert not has(tokens, '--use_qahnl') and has(tokens, '--eval_use_structured_scores')
            assert one_value(tokens, '--sacr_rank_loss_weight') == '0.2'
        elif row == 'M2':
            assert has(tokens, '--use_sacr') and has(tokens, '--use_rapf')
            assert not has(tokens, '--use_qahnl') and has(tokens, '--eval_use_fused_scores')
            assert one_value(tokens, '--sacr_rank_loss_weight') == '0.2'
            assert one_value(tokens, '--rapf_quality_weight') == '0.75'
        else:
            assert has(tokens, '--use_sacr') and has(tokens, '--use_rapf')
            assert has(tokens, '--use_qahnl') and has(tokens, '--eval_use_fused_scores')
            assert one_value(tokens, '--qahnl_score_source') == 'fused'
            assert not has(tokens, '--sacr_rank_loss_weight')
            assert one_value(tokens, '--rapf_quality_weight') == '0.75' if row != 'R1' else True

        if row == 'S0':
            assert has(tokens, '--sacr_disable_target_attr')
        elif row == 'S1':
            assert has(tokens, '--sacr_disable_relation')
        elif row == 'S2':
            assert one_value(tokens, '--sacr_geo_dim') == '0'
        elif row == 'S3':
            assert one_value(tokens, '--sacr_anchor_aggregation') == 'hard'
        elif row == 'R0':
            assert one_value(tokens, '--rapf_fixed_alpha') == '0.1'
            assert one_value(tokens, '--rapf_gate_loss_weight') == '0'
            assert not has(tokens, '--use_reliability_gate')
            assert not has(tokens, '--rapf_disable_safety')
            assert not has(tokens, '--rapf_disable_residual_clipping')
        elif row == 'R1':
            assert not has(tokens, '--use_quality_head')
            assert not has(tokens, '--rapf_use_quality')
            assert has(tokens, '--use_reliability_gate')
        elif row == 'R2':
            assert has(tokens, '--rapf_disable_parser_anchor_cues')
            assert not has(tokens, '--rapf_disable_agreement_features')
            assert not has(tokens, '--rapf_disable_safety')
        elif row == 'R3':
            assert one_value(tokens, '--rapf_gate_loss_weight') == '0'
            assert has(tokens, '--use_reliability_gate')

    assert len(seen_log_dirs) == len(ROWS)
    print('THREE_TABLE_DRY_RUN_AUDIT_PASS rows={}'.format(len(ROWS)))


if __name__ == '__main__':
    main()
