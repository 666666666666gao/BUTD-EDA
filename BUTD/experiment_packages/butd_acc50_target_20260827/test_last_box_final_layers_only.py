#!/usr/bin/env python3
import sys
from pathlib import Path
from types import SimpleNamespace

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import main_utils


class BoxMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(8, 8, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Identity(),
            nn.Conv1d(8, 8, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Identity(),
            nn.Conv1d(8, 3, 1),
        )


class PredictionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.center_residual_head = BoxMLP()
        self.size_pred_head = BoxMLP()
        self.semantic_head = nn.Conv1d(8, 4, 1)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prediction_heads = nn.ModuleList(
            [PredictionHead() for _ in range(6)]
        )
        self.backbone = nn.Conv1d(8, 8, 1)


model = DummyModel()
args = SimpleNamespace(
    last_box_head_train_only=True,
    last_box_head_final_layers_only=True,
)
count = main_utils._freeze_non_last_box_head_parameters(args, model)
trainable = [name for name, p in model.named_parameters() if p.requires_grad]
expected = {
    'prediction_heads.5.center_residual_head.net.8.weight',
    'prediction_heads.5.center_residual_head.net.8.bias',
    'prediction_heads.5.size_pred_head.net.8.weight',
    'prediction_heads.5.size_pred_head.net.8.bias',
}
assert set(trainable) == expected, trainable
assert count == sum(model.get_parameter(name).numel() for name in expected)
assert all('.net.8.' in name for name in trainable)
print('LAST_BOX_FINAL_LAYERS_ONLY_PASS', count, sorted(trainable))
