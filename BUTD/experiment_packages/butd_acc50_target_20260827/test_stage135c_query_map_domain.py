#!/usr/bin/env python3
"""Regression test for Stage29 train-map IDs in joint ScanNet/ScanRefer batches."""

import argparse

import torch

from models.losses import _last_box_mapped_selection


def _predictions(batch_size):
    boxes = torch.zeros((batch_size, 256, 6), dtype=torch.float32)
    boxes[:, :, 3:] = 1.0
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map_path")
    args = parser.parse_args()

    # The real joint-training dataset appends 10x ScanNet after ScanRefer:
    # ScanRefer occupies global indices 0..36664 and Stage29 uses that exact
    # domain; ScanNet occupies 36665..48654 and must fall back.
    end_points = {
        "_loss_is_training_batch": True,
        "example_id": torch.tensor([0, 36664, 36665, 48654], dtype=torch.long),
        "dataset": [
            "scanrefer_spacy", "scanrefer_spacy", "scannet", "scannet"
        ],
    }
    fallback = torch.tensor([7, 8, 9, 10], dtype=torch.long)
    result = _last_box_mapped_selection(
        end_points, args.map_path, fallback, _predictions(4)
    )
    assert result["mapped"].tolist() == [True, True, False, False]
    assert result["selected_index"][2:].tolist() == [9, 10]
    assert result["alpha"][2:].tolist() == [0.0, 0.0]

    # A ScanRefer-local ID outside the frozen map must still fail closed.
    bad = {
        "_loss_is_training_batch": True,
        "example_id": torch.tensor([48655], dtype=torch.long),
        "dataset": ["scanrefer_spacy"],
    }
    try:
        _last_box_mapped_selection(
            bad, args.map_path, torch.tensor([3]), _predictions(1)
        )
    except ValueError as exc:
        assert "strict training coverage" in str(exc)
    else:
        raise AssertionError("out-of-range ScanRefer ID did not fail closed")

    # Validation must never consume a training-map target, even if IDs collide.
    validation = dict(end_points)
    validation["_loss_is_training_batch"] = False
    val_fallback = torch.tensor([11, 12, 13, 14], dtype=torch.long)
    val_result = _last_box_mapped_selection(
        validation, args.map_path, val_fallback, _predictions(4)
    )
    assert val_result["mapped"].tolist() == [False, False, False, False]
    assert val_result["selected_index"].tolist() == val_fallback.tolist()

    print("STAGE135C_QUERY_MAP_DOMAIN_PASS")


if __name__ == "__main__":
    main()
