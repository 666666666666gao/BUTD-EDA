"""Compile row-specific training CLI from verified historical effective configs.

No training is started here. Native parse_option still validates the compiled
arguments in the actual training process. Missing row implementations fail.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path


STAGE_NAMES = {
    'base': 'base', 'continuation': 'continuation', 'microtune': 'microtune',
    '14c': 'stage14c_rescue_gate_stable',
    '16': 'stage16_hit50_rank_augmented',
    '18': 'stage18_geometry_action_head',
    '94': 'stage94_last_box_head_lr1e5',
    '95': 'stage95_targeted_last_box_nojitter',
    '135c': 'stage135c_stage29_option_last_box_noaug_jointmask',
    '150': 'stage150b_rerank_only_tier3_from_stage135c',
}
EPOCHS = {'base': 65, 'continuation': 2, 'microtune': 3,
          '14c': 3, '16': 2, '18': 2, '94': 2, '95': 1, '135c': 1, '150': 1}
ALIASES = {
    'qahnl_pos_iou_threshold', 'qahnl_neg_iou_threshold', 'qahnl_topk_iou',
    'qahnl_margin', 'qahnl_entity_hardneg', 'qahnl_attr_hardneg', 'qahnl_rel_hardneg',
}
ROWS = ('Full', 'M0-local', 'M1', 'M2', 'S0', 'S1', 'S2', 'S3',
        'R0', 'R1', 'R2', 'R3', 'Q1', 'Q2', 'Q3')


def read_native_parser(runtime):
    """Read the existing argparse declarations without importing CUDA modules."""
    tree = ast.parse((runtime / 'models/detector_policy_sources.py').read_text(encoding='utf-8'))
    constants = [ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name)
                         and target.id == 'DETECTOR_POLICY_SOURCE_NAMES'
                         for target in node.targets)]
    assert len(constants) == 1
    source = (runtime / 'main_utils.py').read_text(encoding='utf-8')
    assert source.count('def parse_option():') == 1
    declaration = source.split('def parse_option():', 1)[1].split(
        '    args, unknown = parser.parse_known_args()', 1)[0]
    namespace = {'argparse': argparse,
                 'DETECTOR_POLICY_SOURCE_NAMES': constants[0]}
    exec('def build_parser():' + declaration + '    return parser\n', namespace)
    return namespace['build_parser']()


def serialize_config(parser, config):
    actions = {action.dest: action for action in parser._actions
               if action.dest != 'help'}
    unknown = set(config) - set(actions) - ALIASES
    assert not unknown, sorted(unknown)
    argv = []
    for key, value in config.items():
        if key in ALIASES:
            continue
        action = actions[key]
        if value is None or value == action.default:
            continue
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            assert value is True, (key, value, action.default)
            argv.append(flag)
        elif isinstance(action, argparse._StoreFalseAction):
            assert value is False, (key, value, action.default)
            argv.append(flag)
        elif isinstance(action, argparse._StoreAction):
            argv.append(flag)
            if isinstance(value, list):
                argv.extend(str(item) for item in value)
            else:
                argv.append(str(value))
        else:
            raise ValueError('Unsupported native argparse action: ' + key)
    parsed = vars(parser.parse_args(argv))
    for key, value in config.items():
        if key not in ALIASES:
            assert parsed[key] == value, (key, value, parsed[key])
    return argv, parsed


def apply_row(config, row, stage):
    downstream = stage not in ('base', 'continuation', 'microtune')
    if row in ('M0-local', 'M1', 'R1') and downstream:
        raise ValueError('No-quality downstream export/schema validation is incomplete: ' + row)
    if row == 'S0':
        config['sacr_disable_target_attr'] = True
    elif row == 'S1':
        config['sacr_disable_relation'] = True
    elif row == 'S2':
        config['sacr_geo_dim'] = 0
    elif row == 'S3':
        config['sacr_anchor_aggregation'] = 'hard'
    elif row == 'R0':
        config.update(use_reliability_gate=False, rapf_fixed_alpha=0.1,
                      rapf_gate_loss_weight=0.0, rapf_initial_gate_bias=-2.1972246)
    elif row == 'R2':
        config['rapf_disable_parser_anchor_features'] = True
    elif row == 'R3':
        config['rapf_gate_loss_weight'] = 0.0
    elif row == 'Q1':
        config['qahnl_disable_hungarian_pos_rescue'] = True
    elif row == 'Q2':
        config.update(qahnl_disable_hungarian_pos_rescue=True,
                      qahnl_disable_top_iou_pos=True)
    elif row == 'Q3':
        config['qahnl_loss_form'] = 'hinge'
    if row in ('M0-local', 'M1', 'R1'):
        config.update(use_quality_head=False, rapf_use_quality=False,
                      quality_loss_weight=0.0)
    if row in ('M0-local', 'M1', 'M2'):
        config.update(use_qahnl=False, qahnl_loss_weight=0.0,
                      qahnl_tiered_quality=False)
    if row in ('M1', 'M2'):
        config['sacr_rank_loss_weight'] = 0.0 if downstream else 0.2
    if row in ('M0-local', 'M1'):
        config.update(use_rapf=False, use_reliability_gate=False,
                      rapf_gate_loss_weight=0.0, eval_use_fused_scores=False)
        config['eval_use_structured_scores'] = row == 'M1'
    if row == 'M0-local':
        config.update(use_sacr=False, use_structured_slots=False)


def compile_stage(records, runtime, row, stage, parent, parent_epoch,
                  log_dir, data_root, query_map=None):
    matches = [item for item in records if item['stage'] == STAGE_NAMES[stage]]
    assert len(matches) == 1, stage
    reference = matches[0]
    config = dict(reference['config'])
    if stage == 'base':
        assert parent is None and parent_epoch == 0
    else:
        assert parent and parent_epoch >= 0
    if stage == 'microtune':
        assert parent_epoch == 0, 'Microtune requires a verified epoch-0 model-only initialization.'
    endpoint = parent_epoch + EPOCHS[stage]
    config.update(
        checkpoint_path=parent, pp_checkpoint=str(data_root / 'gf_detector_l6o256.pth'),
        data_root=str(data_root), log_dir=str(log_dir),
        start_epoch=1, max_epoch=endpoint, rng_seed=0, batch_size=24,
        eval=False, eval_train=False, eval_max_samples=-1,
        early_stopping=False, best_checkpoint_only=False,
        best_checkpoint_constraint_lower=None, best_checkpoint_constraint_upper=None,
        best_checkpoint_constraint_epsilon=0.0, save_freq=endpoint + 1,
        val_freq=5 if stage == 'base' else 1,
        dense_val_start_epoch=31 if stage == 'base' else 0,
        dense_val_end_epoch=endpoint if stage == 'base' else 0, dense_val_freq=1,
        eval_results_json_path='',
    )
    if stage == '135c':
        assert query_map, 'Stage135c requires this row\'s train-only query/action map.'
        config['last_box_target_query_map'] = str(query_map)
    apply_row(config, row, stage)
    for key, value in config.items():
        if isinstance(value, str):
            assert 'butd_acc50_target_20260827/' not in value, (key, value)
    if stage == '150' and row in ('M0-local', 'M1', 'M2'):
        return {'status': 'COMPILED_IDENTITY_STAGE', 'row': row, 'stage': stage,
                'runtime': str(runtime), 'parent_checkpoint': parent,
                'parent_epoch': parent_epoch, 'endpoint_epoch': parent_epoch,
                'stage_epochs': 0, 'log_dir': str(log_dir),
                'runtime_main_utils_sha256': hashlib.sha256((runtime / 'main_utils.py').read_bytes()).hexdigest(),
                'optimizer_steps': 0, 'reason': 'QAHNL removed; no replacement loss.'}
    parser = read_native_parser(runtime)
    argv, effective = serialize_config(parser, config)
    for key, value in effective.items():
        if isinstance(value, str):
            assert 'butd_acc50_target_20260827/' not in value, (key, value)
    return {
        'status': 'COMPILED_NOT_EXECUTED', 'row': row, 'stage': stage,
        'runtime': str(runtime), 'parent_checkpoint': parent,
        'parent_epoch': parent_epoch, 'stage_epochs': EPOCHS[stage],
        'endpoint_epoch': endpoint, 'expected_checkpoint_name': 'ckpt_epoch_last.pth',
        'reference_config_sha256': reference['config_sha256'],
        'runtime_main_utils_sha256': hashlib.sha256((runtime / 'main_utils.py').read_bytes()).hexdigest(),
        'argv': argv, 'effective_config': effective,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=Path, nargs='+', required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--row', choices=ROWS, required=True)
    parser.add_argument('--stage', choices=STAGE_NAMES, required=True)
    parser.add_argument('--parent')
    parser.add_argument('--parent-epoch', type=int, required=True)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--query-map', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [record for path in args.history for record in json.loads(path.read_text(encoding='utf-8'))]
    result = compile_stage(records, args.runtime, args.row, args.stage, args.parent,
                           args.parent_epoch, args.log_dir, args.data_root, args.query_map)
    assert not args.output.exists(), args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'row': args.row, 'stage': args.stage,
                      'endpoint_epoch': result['endpoint_epoch'], 'output': str(args.output)}))
