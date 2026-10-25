"""32-batch training witness; allow native AMP scaling to reach optimizer updates."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from compile_stage import read_native_parser, serialize_config
from execute_stage import read_json, verify_parent, verify_runtime, sha256


def smoke(plan_path, output):
    plan = read_json(plan_path)
    assert plan['stage'] == 'base' and plan['status'] == 'COMPILED_NOT_EXECUTED'
    runtime_hash = verify_runtime(plan)
    verify_parent(plan, None)
    assert not output.exists()
    runtime = Path(plan['runtime'])
    sys.path.insert(0, str(runtime / 'pointnet2'))
    sys.path.insert(0, str(runtime))
    os.chdir(str(runtime))
    from train_dist_mod import TrainTester, configure_reproducibility
    from main_utils import parse_option

    config = dict(plan['effective_config'])
    config.update(log_dir=str(output), max_epoch=1, save_freq=2,
                  val_freq=1, dense_val_start_epoch=0, dense_val_end_epoch=0,
                  print_freq=1, eval_max_samples=2, num_workers=0)
    argv, _ = serialize_config(read_native_parser(runtime), config)
    sys.argv = ['smoke_base.py'] + argv
    args = parse_option()
    configure_reproducibility(args.rng_seed)
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    class SmokeTrainTester(TrainTester):
        @staticmethod
        def get_datasets(options, include_train=True):
            train, validation = TrainTester.get_datasets(options, include_train)
            assert len(train) >= 768 and len(validation) == 2
            return torch.utils.data.Subset(train, list(range(768))), validation

        def train_one_epoch(self, epoch, loader, model, criterion, set_criterion,
                            optimizer, scheduler, options):
            prefixes = []
            if options.use_sacr:
                prefixes.append('sacr_head.')
            if options.use_rapf:
                prefixes.append('reliability_fusion.')
            if options.use_quality_head:
                prefixes.append('quality_head.')
            base_model = model.module
            before = {name: parameter.detach().cpu().clone()
                      for name, parameter in base_model.named_parameters()
                      if name.startswith(tuple(prefixes)) and parameter.requires_grad}
            super().train_one_epoch(epoch, loader, model, criterion, set_criterion,
                                    optimizer, scheduler, options)
            witness = {}
            for prefix in prefixes:
                changed = 0
                count = 0
                for name, parameter in base_model.named_parameters():
                    if name.startswith(prefix) and name in before:
                        current = parameter.detach().cpu()
                        assert torch.isfinite(current).all().item(), name
                        changed += int(torch.count_nonzero(current != before[name]).item())
                        count += current.numel()
                witness[prefix] = {'parameters': count, 'changed_elements': changed}
            (output / 'optimizer_witness.json').write_text(json.dumps({
                'modules': witness, 'scheduler_last_epoch': scheduler.last_epoch,
                'optimizer_state_entries': len(optimizer.state),
                'optimizer_steps': sorted(set(float(state['step']) for state in
                                              optimizer.state.values() if 'step' in state)),
            }, indent=2) + '\n')
            for prefix in prefixes:
                assert witness[prefix]['changed_elements'] > 0, 'No actual optimizer update in ' + prefix
            assert scheduler.last_epoch > 0
            self.training_witness = witness

    tester = SmokeTrainTester(args)
    checkpoint_path = Path(tester.main(args))
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    assert checkpoint['epoch'] == 1
    assert checkpoint['optimizer']['state']
    best_path = checkpoint_path.parent / 'ckpt_best_primary.pth'
    assert best_path.is_file()
    receipt = {
        'status': 'BASE_TRAINING_SMOKE_PASS', 'row': plan['row'],
        'stage': 'base_smoke', 'formal_training': False,
        'plan_sha256': sha256(plan_path), 'runtime_manifest_sha256': runtime_hash,
        'train_examples': 768, 'validation_examples': 2,
        'module_optimizer_updates': tester.training_witness,
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': sha256(checkpoint_path), 'epoch': 1,
        'observed_best_checkpoint': str(best_path),
        'observed_best_sha256': sha256(best_path),
        'forbidden_as_formal_parent': True,
    }
    (output / 'smoke_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    smoke(args.plan.absolute(), args.output.absolute())
