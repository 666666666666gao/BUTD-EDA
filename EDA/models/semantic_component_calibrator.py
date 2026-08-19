import torch
import torch.nn as nn


class SemanticComponentCalibrator(nn.Module):
    """Learn a tiny calibration over semantic-eval score components."""

    def __init__(self, max_delta=0.25, extra_score_count=0,
                 extra_max_weight=0.25):
        super().__init__()
        self.max_delta = float(max_delta)
        self.extra_score_count = int(extra_score_count)
        self.extra_max_weight = float(extra_max_weight)
        self.logit_delta = nn.Parameter(torch.zeros(5))
        if self.extra_score_count > 0:
            self.extra_logit_weights = nn.Parameter(
                torch.zeros(self.extra_score_count)
            )
        else:
            self.register_parameter('extra_logit_weights', None)
        self.register_buffer(
            'base_weights',
            torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0]),
        )

    @staticmethod
    def _normalize_extra_scores(extra_scores):
        mean = extra_scores.mean(dim=1, keepdim=True)
        std = extra_scores.std(dim=1, keepdim=True, unbiased=False)
        return (extra_scores - mean) / std.clamp(min=1e-6)

    def forward(self, component_scores, extra_scores=None):
        component_scores = torch.nan_to_num(
            component_scores.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        weights = self.base_weights.to(component_scores.device)
        scale = 1.0 + self.max_delta * torch.tanh(self.logit_delta)
        weights = weights * scale.to(component_scores.device)
        scores = (component_scores * weights.view(1, 1, -1)).sum(dim=-1)
        output = {
            'semantic_component_scores': scores,
            'semantic_component_weights': weights,
        }
        if self.extra_score_count > 0:
            if extra_scores is None:
                raise ValueError(
                    "extra_scores must be provided when extra_score_count > 0"
                )
            extra_scores = torch.nan_to_num(
                extra_scores.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            if extra_scores.dim() == 2:
                extra_scores = extra_scores.unsqueeze(-1)
            if extra_scores.shape[-1] != self.extra_score_count:
                raise ValueError(
                    "extra_scores last dimension must match extra_score_count"
                )
            extra_scores = self._normalize_extra_scores(extra_scores)
            extra_weights = (
                self.extra_max_weight
                * torch.tanh(self.extra_logit_weights)
            )
            extra_weights = extra_weights.to(component_scores.device)
            scores = scores + (
                extra_scores * extra_weights.view(1, 1, -1)
            ).sum(dim=-1)
            output['semantic_component_scores'] = scores
            output['semantic_component_extra_weights'] = extra_weights
        return output
