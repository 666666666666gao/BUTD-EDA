"""Query quality head for IoU-aware grounding scores."""

import torch
import torch.nn as nn


class QualityHead(nn.Module):
    """Predict per-query localization quality.

    The predicted boxes are detached before entering the MLP so the quality
    objective cannot train the box regression path through box coordinates.
    """

    def __init__(self, d_model=288, hidden_dim=288):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, query_feats, pred_boxes):
        with torch.cuda.amp.autocast(enabled=False):
            query_feats = query_feats.float()
            pred_boxes = pred_boxes.detach().float()
            quality_logits = self.mlp(
                torch.cat([query_feats, pred_boxes], dim=-1)
            ).squeeze(-1)
            pred_iou = torch.sigmoid(quality_logits)
        return {
            'quality_logits': quality_logits,
            'pred_iou': pred_iou,
        }
