import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiment_packages"
    / "butd_acc50_target_20260827"
    / "stage146_tiered_query_rank_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("tiered_diagnostic", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_row():
    return {
        "adapter_score_at_candidate": [4.0, 3.0, 2.0],
        "adapter_fused_at_candidate": [4.1, 3.2, 2.2],
        "adapter_delta_at_candidate": [0.0, 0.1, -0.1],
        "adapter_hit25_logit_at_candidate": [1.0, 2.0, -1.0],
        "adapter_hit50_logit_at_candidate": [0.0, 3.0, -2.0],
        "adapter_rescue_logit_at_candidate": [-1.0, 2.0, -3.0],
        "adapter_iou_at_candidate": [0.30, 0.70, 0.0],
        "adapter_box_at_candidate": [
            [0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1, 1],
            [2, 0, 0, 1, 1, 1],
        ],
        "adapter_candidate_query": [10, 11, 12],
        "quality_topk_query": [11, 10],
        "quality_topk_score": [0.8, 0.4],
        "base_at_quality_topk": [0.5, 0.4],
        "fused_at_quality_topk": [3.2, 4.1],
        "contrastive_at_quality_topk": [0.6, 0.5],
        "detector_class_at_quality_topk": [0.9, 0.8],
        "detector_conf20_at_quality_topk": [0.9, 0.8],
        "detector_class_count": 3,
        "detector_conf20_count": 3,
        "detector_conf50_count": 2,
        "text_target_cid": 1,
        "decomposition_status": "ok",
        "spacy_profile_bucket": "full_natural",
        "scene_id": "scene0000_00",
    }


def test_tiers_and_feature_group_are_consistent():
    group = MODULE.row_group(synthetic_row(), candidate_k=3)
    assert group["labels"].tolist() == [1, 2, 0]
    assert group["features"].shape[0] == 3
    assert group["features"].shape[1] > 40
    assert np.isfinite(group["features"]).all()


def test_reject_threshold_preserves_baseline_when_gap_is_small():
    group = MODULE.row_group(synthetic_row(), candidate_k=3)
    arrays = {
        "features": group["features"],
        "labels": group["labels"],
        "ious": group["ious"],
        "group_sizes": np.asarray([3]),
        "baseline_ious": np.asarray([group["baseline_iou"]]),
    }
    after, changed, _ = MODULE.apply_policy(
        [np.asarray([0.0, 0.1, -1.0])], arrays, threshold=0.2
    )
    assert not changed[0]
    assert after[0] == group["baseline_iou"]
    after, changed, _ = MODULE.apply_policy(
        [np.asarray([0.0, 0.3, -1.0])], arrays, threshold=0.2
    )
    assert changed[0]
    assert after[0] >= 0.5


def test_summary_counts_threshold_fixes_and_breaks():
    before = np.asarray([0.1, 0.6, 0.3, 0.8], dtype=np.float32)
    after = np.asarray([0.6, 0.1, 0.7, 0.8], dtype=np.float32)
    result = MODULE.summarize(before, after, before != after)
    assert result["fix_050"] == 2
    assert result["break_050"] == 1
    assert result["fix_025"] == 1
    assert result["break_025"] == 1


def test_missing_detector_counts_are_treated_as_zero():
    row = synthetic_row()
    row["detector_class_count"] = None
    row["detector_conf20_count"] = None
    group = MODULE.row_group(row, candidate_k=3)
    assert np.isfinite(group["features"]).all()


def test_missing_text_target_id_is_encoded_as_invalid():
    row = synthetic_row()
    row["text_target_cid"] = None
    group = MODULE.row_group(row, candidate_k=3)
    assert np.isfinite(group["features"]).all()
