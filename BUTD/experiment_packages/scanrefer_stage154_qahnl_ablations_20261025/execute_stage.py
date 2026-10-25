"""Execute one compiled stage, verifying actual parent identity before training."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


PARENT_STAGES = {'continuation': 'base', 'microtune': 'microtune_init',
                 '14c': 'microtune', '16': '14c', '18': '14c',
                 '94': '18', '95': '94', '135c': '95', '150': '135c'}
OFFICIAL_SHA256 = '9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2'


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_parent(plan, receipt_path):
    if plan['stage'] == 'base':
        assert receipt_path is None and plan['parent_checkpoint'] is None
        assert plan['parent_epoch'] == 0
        init = plan['effective_config']['pp_checkpoint']
        assert sha256(init) == OFFICIAL_SHA256
        return {'official_detector': init, 'sha256': OFFICIAL_SHA256}
    assert receipt_path, 'A completed same-row parent receipt is required.'
    receipt = read_json(receipt_path)
    assert receipt['row'] == plan['row'], 'Parent belongs to a different ablation row.'
    assert receipt['stage'] == PARENT_STAGES[plan['stage']]
    expected_status = ('MODEL_ONLY_INIT_READY' if plan['stage'] == 'microtune'
                       else 'TRAINING_STAGE_COMPLETE')
    assert receipt['status'] == expected_status
    path = Path(plan['parent_checkpoint'])
    assert path.resolve() == Path(receipt['checkpoint']).resolve()
    digest = sha256(path)
    assert digest == receipt['checkpoint_sha256']
    checkpoint = torch.load(str(path), map_location='cpu')
    epoch = checkpoint['epoch']
    assert isinstance(epoch, int), 'Parent epoch must be an actual numeric checkpoint field.'
    assert epoch == plan['parent_epoch'] == receipt['epoch'], 'Parent epoch does not match the compiled budget.'
    assert plan['endpoint_epoch'] - epoch == plan['stage_epochs']
    if plan['stage'] == 'continuation':
        assert checkpoint['optimizer']['state']
        assert 'last_epoch' in checkpoint['scheduler']
    if plan['stage'] == 'microtune':
        assert epoch == 0
        assert 'optimizer' not in checkpoint and 'scheduler' not in checkpoint
    assert checkpoint['model']
    del checkpoint
    return {'receipt': str(receipt_path), 'receipt_sha256': sha256(receipt_path),
            'checkpoint': str(path), 'checkpoint_sha256': digest, 'epoch': epoch}


def verify_runtime(plan):
    runtime = Path(plan['runtime'])
    assert sha256(runtime / 'main_utils.py') == plan['runtime_main_utils_sha256']
    manifest = read_json(runtime / 'STAGE154_RUNTIME_MANIFEST.json')
    for item in manifest['copied_files']:
        assert sha256(runtime / item['path']) == item['runtime_sha256'], item['path']
    return sha256(runtime / 'STAGE154_RUNTIME_MANIFEST.json')


def execute(plan_path, parent_receipt, data_manifest, master_port, preflight_only):
    plan = read_json(plan_path)
    assert plan['status'] in ('COMPILED_NOT_EXECUTED', 'COMPILED_IDENTITY_STAGE')
    runtime_hash = verify_runtime(plan)
    parent = verify_parent(plan, parent_receipt)
    subprocess.check_call(['sha256sum', '-c', str(data_manifest)])
    proof = {'plan': str(plan_path.resolve()), 'plan_sha256': sha256(plan_path), 'runtime_manifest_sha256': runtime_hash,
             'data_manifest_sha256': sha256(data_manifest), 'parent': parent}
    identity = plan['status'] == 'COMPILED_IDENTITY_STAGE'
    if identity:
        assert plan['row'] in ('M0-local', 'M1', 'M2') and plan['stage'] == '150'
        assert plan['optimizer_steps'] == plan['stage_epochs'] == 0
        source_receipt = read_json(parent['receipt'])
        assert source_receipt['downstream_checkpoint_policy'] == 'fixed_stage_endpoint'
        source_plan_path = Path(source_receipt['proof']['plan'])
        assert sha256(source_plan_path) == source_receipt['proof']['plan_sha256']
        source_plan = read_json(source_plan_path)
        assert source_plan['status'] == 'COMPILED_NOT_EXECUTED'
        assert source_plan['row'] == plan['row'] and source_plan['stage'] == '135c'
        assert source_plan['runtime'] == plan['runtime']
        assert source_receipt['proof']['runtime_manifest_sha256'] == runtime_hash
        assert source_receipt['proof']['data_manifest_sha256'] == proof['data_manifest_sha256']
        assert source_plan['endpoint_epoch'] == parent['epoch'] == plan['endpoint_epoch']
        assert not source_plan['effective_config']['use_qahnl']
        assert source_plan['effective_config']['qahnl_loss_weight'] == 0.0
    if plan['stage'] == '135c':
        from query_map_provenance import verify
        proof['query_map'] = verify(plan, parent, proof['data_manifest_sha256'])
    if preflight_only:
        return {'status': 'STAGE_PREFLIGHT_PASS_NOT_EXECUTED', 'proof': proof}
    output = Path(plan['log_dir'] if identity else plan['effective_config']['log_dir'])
    assert not output.exists(), 'Refusing to rerun or overwrite an existing stage.'
    output.parent.mkdir(parents=True, exist_ok=True)
    if not identity:
        memory = subprocess.check_output([
            'nvidia-smi', '--id=0', '--query-gpu=memory.used',
            '--format=csv,noheader,nounits']).decode().strip()
        assert int(memory) < 500, 'GPU is still busy; do not launch concurrently.'
        assert shutil.disk_usage(str(output.parent)).free >= 2600000 * 1024
    output.mkdir()
    (output / 'preflight.json').write_text(json.dumps(proof, indent=2) + '\n')
    if identity:
        receipt = {'status': 'TRAINING_STAGE_COMPLETE', 'row': plan['row'],
                   'stage': plan['stage'], 'checkpoint': parent['checkpoint'],
                   'checkpoint_sha256': parent['checkpoint_sha256'],
                   'epoch': plan['endpoint_epoch'], 'optimizer_steps_this_stage': 0,
                   'identity_stage': True, 'identity_source_stage': '135c',
                   'identity_source_plan': str(source_plan_path.resolve()),
                   'identity_source_plan_sha256': sha256(source_plan_path),
                   'downstream_checkpoint_policy': 'fixed_stage_endpoint',
                   'stage_training_metrics_only': True,
                   'full_stage154_ablation_complete': False, 'proof': proof}
    else:
        runtime = Path(plan['runtime'])
        environment = dict(os.environ)
        environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment['PATH']
        environment.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           TOKENIZERS_PARALLELISM='false',
                           PYTHONPATH=str(runtime) + ':' + str(runtime / 'pointnet2'))
        command = [str(Path(sys.executable).parent / 'torchrun'),
                   '--nproc_per_node=1', '--master_port=' + str(master_port),
                   'train_dist_mod.py'] + plan['argv']
        with (output / 'launcher.log').open('w') as handle:
            result = subprocess.run(command, cwd=str(runtime), env=environment,
                                    stdout=handle, stderr=subprocess.STDOUT)
        (output / 'exit_code.txt').write_text(str(result.returncode) + '\n')
        assert result.returncode == 0, 'Training failed; see launcher.log. No automatic restart.'
        paths = list(output.glob('scanrefer_spacy/*/ckpt_epoch_last.pth'))
        assert len(paths) == 1, paths
        checkpoint_path = paths[0]
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        assert checkpoint['epoch'] == plan['endpoint_epoch']
        assert checkpoint['optimizer']['state'], 'No optimizer update was saved.'
        final_log = checkpoint_path.parent / 'eval_epoch_last.log'
        assert final_log.is_file()
        best_path = checkpoint_path.parent / 'ckpt_best_primary.pth'
        best_selection = read_json(checkpoint_path.parent / 'best_primary.json')
        assert best_path.is_file()
        receipt = {'status': 'TRAINING_STAGE_COMPLETE', 'row': plan['row'],
                   'stage': plan['stage'], 'checkpoint': str(checkpoint_path),
                   'checkpoint_sha256': sha256(checkpoint_path),
                   'epoch': checkpoint['epoch'],
                   'scheduler_last_epoch': checkpoint['scheduler']['last_epoch'],
                   'final_stage_eval_log': str(final_log),
                   'final_stage_eval_sha256': sha256(final_log),
                   'observed_best_checkpoint': str(best_path),
                   'observed_best_sha256': sha256(best_path),
                   'observed_best_selection': best_selection,
                   'downstream_checkpoint_policy': 'fixed_stage_endpoint',
                   'stage_training_metrics_only': True,
                   'full_stage154_ablation_complete': False, 'proof': proof}
    (output / 'stage_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path)
    parser.add_argument('--parent-receipt', type=Path)
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--master-port', type=int, default=33541)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    result = execute(args.plan, args.parent_receipt, args.data_manifest,
                     args.master_port, args.preflight_only)
    print(json.dumps(result, indent=2))
