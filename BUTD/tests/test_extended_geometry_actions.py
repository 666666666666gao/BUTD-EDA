from types import SimpleNamespace

import torch
import torch.nn as nn

from main_utils import (
    _freeze_non_detector_policy_adapter_parameters,
    _seed_geometry_extension_checkpoint,
    _upgrade_extended_geometry_checkpoint,
)
from models.detector_policy_sources import DetectorPolicyAdapterHead


class _Wrapper(nn.Module):
    def __init__(self, extended, extension_head=False):
        super().__init__()
        self.detector_policy_adapter = DetectorPolicyAdapterHead(
            hidden_dim=64,
            query_dim=288,
            extended_geometry_actions=extended,
            geometry_extension_head=extension_head,
        )


def test_extended_geometry_action_values():
    head = DetectorPolicyAdapterHead(extended_geometry_actions=True)
    expected = head.geometry_actions.new_tensor(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    )
    assert torch.equal(head.geometry_actions, expected)
    assert head.geometry_head[-1].out_features == len(expected)


def test_six_action_checkpoint_upgrade_preserves_legacy_decisions():
    legacy = nn.DataParallel(_Wrapper(False))
    extended = nn.DataParallel(_Wrapper(True))
    torch.manual_seed(47)
    with torch.no_grad():
        for parameter in (
            legacy.module.detector_policy_adapter.geometry_scalar_mlp.parameters()
        ):
            parameter.normal_(0, 0.1)
        for parameter in (
            legacy.module.detector_policy_adapter.geometry_head.parameters()
        ):
            parameter.normal_(0, 0.1)

    checkpoint = {
        key: value.detach().clone()
        for key, value in legacy.state_dict().items()
    }
    checkpoint, changed = _upgrade_extended_geometry_checkpoint(
        SimpleNamespace(detector_policy_geometry_extended_actions=True),
        extended.state_dict(),
        checkpoint,
    )
    assert changed
    missing, unexpected = extended.load_state_dict(checkpoint, strict=False)
    assert not missing
    assert not unexpected

    old_head = legacy.module.detector_policy_adapter.geometry_head[-1]
    new_head = extended.module.detector_policy_adapter.geometry_head[-1]
    assert torch.equal(new_head.weight[:6], old_head.weight)
    assert torch.equal(new_head.bias[:6], old_head.bias)
    assert torch.equal(new_head.weight[6], old_head.weight[5])
    assert torch.equal(new_head.weight[7], old_head.weight[5])
    assert torch.allclose(new_head.bias[6:], old_head.bias[5] - 1.0)

    hidden = torch.randn(4096, 64, device=new_head.weight.device)
    legacy_logits = old_head(hidden)
    extended_logits = new_head(hidden)
    assert torch.equal(extended_logits[:, :6], legacy_logits)
    assert torch.equal(
        extended_logits.argmax(dim=-1), legacy_logits.argmax(dim=-1)
    )


def test_residual_extension_checkpoint_seed_preserves_legacy_decisions():
    legacy = nn.DataParallel(_Wrapper(False))
    extension = nn.DataParallel(_Wrapper(False, extension_head=True))
    torch.manual_seed(48)
    with torch.no_grad():
        for parameter in legacy.module.detector_policy_adapter.parameters():
            parameter.normal_(0, 0.1)

    checkpoint = {
        key: value.detach().clone()
        for key, value in legacy.state_dict().items()
    }
    checkpoint, changed = _seed_geometry_extension_checkpoint(
        SimpleNamespace(detector_policy_geometry_extension_head=True),
        extension.state_dict(),
        checkpoint,
    )
    assert changed
    missing, unexpected = extension.load_state_dict(checkpoint, strict=False)
    assert not missing
    assert not unexpected

    old = legacy.module.detector_policy_adapter
    new = extension.module.detector_policy_adapter
    assert torch.equal(
        new.geometry_extension_head[0].weight,
        old.geometry_head[0].weight,
    )
    assert torch.equal(
        new.geometry_extension_head[0].bias,
        old.geometry_head[0].bias,
    )
    assert torch.equal(
        new.geometry_extension_head[2].weight[0],
        old.geometry_head[2].weight[5],
    )
    assert torch.allclose(
        new.geometry_extension_head[2].bias,
        old.geometry_head[2].bias[5] - 1.0,
    )

    geometry_input = torch.randn(
        4096, 192, device=new.geometry_head[0].weight.device
    )
    legacy_logits = old.geometry_head(geometry_input)
    extension_logits = new.geometry_extension_head(geometry_input)
    assert torch.allclose(
        extension_logits,
        legacy_logits[:, 5:6].expand(-1, 2) - 1.0,
        atol=1e-6,
        rtol=1e-6,
    )
    combined = torch.cat([legacy_logits, extension_logits], dim=-1)
    assert torch.equal(
        combined.argmax(dim=-1), legacy_logits.argmax(dim=-1)
    )


def test_residual_extension_train_only_freezes_all_legacy_parameters():
    model = nn.DataParallel(_Wrapper(False, extension_head=True))
    count = _freeze_non_detector_policy_adapter_parameters(
        SimpleNamespace(
            detector_policy_adapter_train_only=True,
            source_pool_selector_train_only=False,
            detector_policy_geometry_train_only=False,
            detector_policy_geometry_extension_train_only=True,
        ),
        model,
    )
    trainable = [name for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    assert count > 0
    assert trainable
    assert all('geometry_extension_head' in name for name in trainable)
    assert not any('geometry_head.' in name for name in trainable)
