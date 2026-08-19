import torch

from train_dist_mod import TrainTester


def test_model_inputs_include_official_bbs_alignment_tensors():
    batch = {
        "point_clouds": torch.zeros(1, 2, 3),
        "utterances": ["the chair"],
        "all_detected_boxes": torch.zeros(1, 2, 6),
        "all_detected_bbox_label_mask": torch.ones(1, 2),
        "all_detected_class_ids": torch.zeros(1, 2, dtype=torch.long),
        "positive_map": torch.zeros(1, 1, 256),
        "box_label_mask": torch.ones(1, 1),
    }
    inputs = TrainTester._get_inputs(batch)
    assert inputs["positive_map"] is batch["positive_map"]
    assert inputs["box_label_mask"] is batch["box_label_mask"]
