import json
from types import SimpleNamespace

import torch

from main_utils import save_best_checkpoint


def _objects(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2], gamma=0.1
    )
    args = SimpleNamespace(
        log_dir=str(tmp_path),
        best_checkpoint_metric='last__bbs_acc0.25_top1',
        best_checkpoint_min_delta=0.0,
    )
    return args, model, optimizer, scheduler


def test_best_checkpoint_is_created_and_embeds_selection(tmp_path):
    args, model, optimizer, scheduler = _objects(tmp_path)
    assert save_best_checkpoint(
        args, 5, model, optimizer, scheduler,
        {'last__bbs_acc0.25_top1': 0.51},
    )
    checkpoint = torch.load(
        tmp_path / 'ckpt_best_primary.pth', map_location='cpu'
    )
    assert checkpoint['epoch'] == 5
    assert checkpoint['best_checkpoint_selection']['score'] == 0.51
    receipt = json.loads((tmp_path / 'best_primary.json').read_text())
    assert receipt['metric'] == 'last__bbs_acc0.25_top1'
    assert receipt['comparison'] == 'strict_greater_than'


def test_best_checkpoint_only_changes_on_strict_improvement(tmp_path):
    args, model, optimizer, scheduler = _objects(tmp_path)
    save_best_checkpoint(
        args, 5, model, optimizer, scheduler,
        {'last__bbs_acc0.25_top1': 0.51},
    )
    assert not save_best_checkpoint(
        args, 10, model, optimizer, scheduler,
        {'last__bbs_acc0.25_top1': 0.51},
    )
    assert not save_best_checkpoint(
        args, 15, model, optimizer, scheduler,
        {'last__bbs_acc0.25_top1': 0.50},
    )
    assert save_best_checkpoint(
        args, 20, model, optimizer, scheduler,
        {'last__bbs_acc0.25_top1': 0.52},
    )
    checkpoint = torch.load(
        tmp_path / 'ckpt_best_primary.pth', map_location='cpu'
    )
    assert checkpoint['epoch'] == 20
    assert checkpoint['best_checkpoint_selection']['score'] == 0.52


def test_best_checkpoint_rejects_missing_metric(tmp_path):
    args, model, optimizer, scheduler = _objects(tmp_path)
    try:
        save_best_checkpoint(args, 5, model, optimizer, scheduler, {})
    except KeyError as exc:
        assert 'last__bbs_acc0.25_top1' in str(exc)
    else:
        raise AssertionError('missing metric must fail closed')
