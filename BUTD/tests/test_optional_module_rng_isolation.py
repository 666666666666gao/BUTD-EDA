import torch
import torch.nn as nn

from models.bdetr import _build_seeded_optional_module


def _state(module):
    return {
        key: value.detach().clone() for key, value in module.state_dict().items()
    }


def test_optional_module_does_not_shift_shared_initialization():
    torch.manual_seed(0)
    shared_without_optional = nn.Linear(16, 8)

    torch.manual_seed(0)
    _build_seeded_optional_module(101, lambda: nn.Linear(32, 7))
    shared_with_optional = nn.Linear(16, 8)

    for key, expected in _state(shared_without_optional).items():
        assert torch.equal(expected, shared_with_optional.state_dict()[key])


def test_same_optional_module_has_identical_initialization_across_configs():
    torch.manual_seed(0)
    first = _build_seeded_optional_module(107, lambda: nn.Linear(12, 5))

    torch.manual_seed(999)
    _build_seeded_optional_module(103, lambda: nn.Linear(50, 50))
    second = _build_seeded_optional_module(107, lambda: nn.Linear(12, 5))

    for key, expected in _state(first).items():
        assert torch.equal(expected, second.state_dict()[key])
