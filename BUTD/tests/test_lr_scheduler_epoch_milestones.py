from types import SimpleNamespace

import pytest
import torch

from utils.lr_scheduler import get_scheduler


def test_step_milestones_land_on_requested_epoch_boundaries_without_warmup():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    args = SimpleNamespace(
        lr_scheduler="step",
        lr_decay_epochs=[55, 60],
        lr_decay_rate=0.1,
        warmup_epoch=-1,
        warmup_multiplier=100,
        max_epoch=65,
    )
    iterations_per_epoch = 2
    scheduler = get_scheduler(optimizer, iterations_per_epoch, args)

    assert sorted(scheduler.milestones.elements()) == [110, 120]
    for _ in range(54 * iterations_per_epoch):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    for _ in range(iterations_per_epoch):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    for _ in range(5 * iterations_per_epoch):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
