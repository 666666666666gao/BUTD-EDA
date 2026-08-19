import os
import sys
import time
import numpy as np

import torch

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        SummaryWriter = None


class _NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None

class TensorBoard():
    def __init__(self, out_dir, distributed_rank):

        self.init_log_item()

        self.best = {
            "epoch": 0,
            "train_loss": float("inf"),
            "val_loss": float("inf"),
            "iou": float("inf"),
            "recall": -float("inf"),
            "acc": -float("inf")
        }

        if distributed_rank == 0:
            writer_cls = SummaryWriter or _NullSummaryWriter
            self.tensorboard_writer = {
                "train": writer_cls(os.path.join('tensorboard_output/', "tensorboard/train")),
                "val": writer_cls(os.path.join('tensorboard_output/', "tensorboard/val"))
            }
    
    def dump_tensorboard(self, phase, timestamp):
        log = {
            # phase:[value]
            "loss": ["loss", "loss_bbox", "loss_ce", "loss_sem_align", "loss_giou", \
                "query_points_generation_loss"],
            "score": ["soft_token_0.25", "soft_token_0.5", "contrastive_0.25", "contrastive_0.5"],
            "learning_rate": ["lr_base", "lr_pointnet"]
        }

        # train loss
        if phase == "train_loss":
            for val in log["loss"]:
                self.tensorboard_writer["train"].add_scalar(
                    "{}/{}".format("train_loss", val),
                    self.item[phase][val],
                    timestamp
                )
        
        # lr
        if phase == "train_lr":
            for val in log["learning_rate"]:
                self.tensorboard_writer["train"].add_scalar(
                    "{}/{}".format("learning_rate", val),
                    self.item[phase][val],
                    timestamp
                )
        
        # val loss
        if phase == "val_loss":
            for val in log["loss"]:
                self.tensorboard_writer["val"].add_scalar(
                    "{}/{}".format("val_loss", val),
                    self.item[phase][val],
                    timestamp
                )

        # val score
        if phase == "val_score":
            for val in log["score"]:
                self.tensorboard_writer["val"].add_scalar(
                    "{}/{}".format("score", val),
                    self.item[phase][val],
                    timestamp
                )

    def init_log_item(self):
        self.item = {
            "train_lr":{
                "lr_base": [],
                "lr_pointnet": [],
            },

            "train_loss":{
                "loss":[],
                "loss_bbox": [],
                "loss_ce": [],
                "loss_sem_align": [],
                "loss_giou": [],
                "query_points_generation_loss": []
            },

            "val_score":{
                "soft_token_0.25": [],
                "soft_token_0.5": [],
                "contrastive_0.25": [],
                "contrastive_0.5": [],
            },

            "val_loss":{
                "loss":[],
                "loss_bbox": [],
                "loss_ce": [],
                "loss_sem_align": [],
                "loss_giou": [],
                "query_points_generation_loss": []
            }
        }
