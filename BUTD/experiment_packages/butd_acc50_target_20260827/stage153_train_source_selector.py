#!/usr/bin/env python3
"""Train/evaluate a train-only Stage142-vs-Stage150 source selector.

The selector is trained and policy-locked on scene-disjoint partitions of the
ScanRefer training split. Ground-truth fields are used only to form training
targets and metrics. Every feature admitted by ``feature_dict`` is available
at inference time. ScanRefer validation must not be consumed until the
internal scene-test authorization gate passes.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import sys

import lightgbm as lgb
import numpy as np
import torch


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

import stage140_train_eval_nested_blend as blend
import stage143_same_checkpoint_complement_gate as complement
import train_joint_option_ranker as option_ranker


FORBIDDEN_SOURCE_FEATURE_TOKENS = (
    "iou", "oracle", "threshold_utility", "ground_truth", "gt_",
    "candidate_hit", "label", "target_box",
)

COMPACT_ARRAY_KEYS = (
    "adapter_score_at_candidate",
    "adapter_fused_at_candidate",
    "adapter_delta_at_candidate",
    "adapter_hit25_logit_at_candidate",
    "adapter_hit50_logit_at_candidate",
    "adapter_rescue_logit_at_candidate",
    "base_at_quality_topk",
    "fused_at_quality_topk",
    "contrastive_at_quality_topk",
    "quality_topk_score",
    "detector_class_at_quality_topk",
    "detector_logit_at_quality_topk",
    "detector_logit_top2_at_quality_topk",
    "detector_logit_top3_at_quality_topk",
    "detector_conf20_at_quality_topk",
    "detector_conf30_at_quality_topk",
    "detector_conf40_at_quality_topk",
    "detector_conf50_at_quality_topk",
    "detected_target_confidence",
)

DECOMP_VALUES = ("ok", "repaired_structured", "global_only", "weak_generic")
SPACY_AUG_VALUES = (
    "spacy_aug_none", "spacy_aug_yaw_only", "spacy_aug_full_natural",
)
SPACY_PROFILE_VALUES = (
    "spacy_profile_none", "spacy_profile_yaw_relation",
    "spacy_profile_full_natural_relation_free",
    "spacy_profile_yaw_relation_free",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def as_array(value):
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    return np.nan_to_num(result, nan=0.0, posinf=50.0, neginf=-50.0)


def scalar(value, default=0.0):
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.number)):
        result = float(value)
        return result if math.isfinite(result) else float(default)
    if torch.is_tensor(value) and value.numel() == 1:
        result = float(value.detach().cpu().item())
        return result if math.isfinite(result) else float(default)
    return float(default)


def entropy(values):
    values = as_array(values)
    if len(values) <= 1:
        return 0.0
    shifted = np.clip(values - float(values.max()), -50.0, 50.0)
    probabilities = np.exp(shifted)
    probabilities /= max(float(probabilities.sum()), 1e-8)
    return float(-(probabilities * np.log(probabilities + 1e-8)).sum())


def margin(values):
    values = as_array(values)
    if len(values) <= 1:
        return 0.0
    top = np.partition(values, -2)[-2:]
    return float(top.max() - top.min())


def correlation(left, right):
    left = as_array(left)
    right = as_array(right)
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    if float(left.std()) < 1e-6 or float(right.std()) < 1e-6:
        return 0.0
    result = float(np.corrcoef(left, right)[0, 1])
    return result if math.isfinite(result) else 0.0


def pair_iou(box_a, box_b):
    box_a = np.asarray(box_a, dtype=np.float32)
    box_b = np.asarray(box_b, dtype=np.float32)
    a_lo = box_a[:3] - np.abs(box_a[3:]) / 2.0
    a_hi = box_a[:3] + np.abs(box_a[3:]) / 2.0
    b_lo = box_b[:3] - np.abs(box_b[3:]) / 2.0
    b_hi = box_b[:3] + np.abs(box_b[3:]) / 2.0
    inter = np.maximum(np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo), 0.0)
    intersection = float(np.prod(inter))
    volume_a = float(np.prod(np.maximum(a_hi - a_lo, 0.0)))
    volume_b = float(np.prod(np.maximum(b_hi - b_lo, 0.0)))
    return intersection / max(volume_a + volume_b - intersection, 1e-8)


def source_feature_allowed(name, value):
    lowered = name.lower()
    if any(token in lowered for token in FORBIDDEN_SOURCE_FEATURE_TOKENS):
        return False
    if "query" in lowered and "same_query" not in lowered:
        return False
    if any(token in lowered for token in ("center_x", "center_y", "center_z")):
        if "delta" not in lowered:
            return False
    return isinstance(value, (bool, int, float, np.number)) or (
        torch.is_tensor(value) and value.numel() == 1
    )


def choose_source_feature_names(row):
    names = sorted(
        name for name, value in row.items()
        if source_feature_allowed(str(name), value)
    )
    assert names
    for name in names:
        lowered = name.lower()
        assert not any(
            token in lowered for token in FORBIDDEN_SOURCE_FEATURE_TOKENS
        ), name
    return names


def compact_stats(features, prefix, values):
    values = as_array(values)
    if len(values) == 0:
        values = np.zeros(1, dtype=np.float32)
    ordered = np.sort(values)[::-1]
    features[prefix + "__count"] = float(len(values))
    features[prefix + "__mean"] = float(values.mean())
    features[prefix + "__std"] = float(values.std())
    features[prefix + "__max"] = float(values.max())
    features[prefix + "__min"] = float(values.min())
    features[prefix + "__range"] = float(values.max() - values.min())
    features[prefix + "__margin"] = (
        float(ordered[0] - ordered[1]) if len(ordered) > 1 else 0.0
    )
    features[prefix + "__entropy"] = entropy(values)


def option_metadata(row, max_candidates=8):
    positions = option_ranker.candidate_positions(row, max_candidates)
    queries = list(row["adapter_candidate_query"])
    pred_boxes = np.asarray(row["adapter_box_at_candidate"], dtype=np.float32)
    det_boxes = np.asarray(row.get("detected_box", []), dtype=np.float32)
    det_conf = np.asarray(
        row.get("detected_target_confidence", []), dtype=np.float32
    )
    if det_boxes.ndim != 2 or det_boxes.shape[-1] != 6:
        det_boxes = np.zeros((0, 6), dtype=np.float32)
    if len(det_conf) != len(det_boxes):
        det_conf = np.zeros(len(det_boxes), dtype=np.float32)
    det_conf = np.clip(np.nan_to_num(det_conf), 0.0, 1.0)
    metadata = []
    for position in positions:
        pred = pred_boxes[position]
        support = option_ranker.pair_iou(pred, det_boxes)
        for match_power in option_ranker.MATCH_POWERS:
            if len(det_boxes):
                match_score = support * np.power(det_conf + 1e-6, match_power)
                match_index = int(np.argmax(match_score))
                matched = det_boxes[match_index]
            else:
                matched = pred.copy()
            for alpha in option_ranker.ACTIONS:
                box = pred + alpha * (matched - pred)
                box[3:] = np.maximum(np.abs(box[3:]), 1e-5)
                metadata.append({
                    "box": box.astype(np.float32),
                    "query": int(queries[position]),
                    "candidate_position": int(position),
                    "match_power": float(match_power),
                    "alpha": float(alpha),
                })
    return metadata


def build_stage142_arrays(
    rows, stage142_lock_path, stage31_lock, stage33_lock,
    require_scene=None,
):
    provenance = blend.verified_sources(stage31_lock, stage33_lock)
    lock = read_json(stage142_lock_path)
    safe_config = complement.validate_stage142_lock(lock, provenance)
    scene_presence = [bool(row.get("scene_id", "")) for row in rows]
    assert scene_presence and (
        all(scene_presence) or not any(scene_presence)
    ), "mixed scene_id availability"
    if require_scene is None:
        require_scene = all(scene_presence)
    group_features, group_ious, metas = option_ranker.build_dataset(
        rows, max_candidates=8, require_scene=require_scene
    )
    features, _, ious, groups, baselines = option_ranker.materialize(
        group_features, group_ious, metas, list(range(len(rows)))
    )
    boosters = blend.load_boosters(provenance)
    inner, pointwise = blend.component_scores(
        features, groups, boosters, provenance
    )
    selected_ious = []
    selected_features = []
    selected_boxes = []
    selected_queries = []
    contexts = []
    cursor = 0
    for row_index, (size_value, baseline_value) in enumerate(zip(groups, baselines)):
        size = int(size_value)
        baseline_index = int(baseline_value)
        group_x = features[cursor:cursor + size]
        group_iou = ious[cursor:cursor + size]
        group_inner = inner[cursor:cursor + size]
        group_pointwise = pointwise[cursor:cursor + size]
        safe_scores = (
            safe_config["inner_weight"] * group_inner
            + safe_config["pointwise_weight"] * group_pointwise
        )
        best = int(np.argmax(safe_scores))
        gap = float(safe_scores[best] - safe_scores[baseline_index])
        selected = best if gap >= safe_config["threshold"] else baseline_index
        metadata = option_metadata(rows[row_index], max_candidates=8)
        assert len(metadata) == size
        selected_ious.append(float(group_iou[selected]))
        selected_features.append(group_x[selected].astype(np.float32))
        selected_boxes.append(metadata[selected]["box"])
        selected_queries.append(metadata[selected]["query"])
        contexts.append({
            "safe_gap": gap,
            "safe_fired": float(selected != baseline_index),
            "safe_selected_score": float(safe_scores[selected]),
            "safe_baseline_score": float(safe_scores[baseline_index]),
            "safe_margin": margin(safe_scores),
            "safe_entropy": entropy(safe_scores),
            "inner_margin": margin(group_inner),
            "inner_entropy": entropy(group_inner),
            "pointwise_margin": margin(group_pointwise),
            "pointwise_entropy": entropy(group_pointwise),
            "inner_pointwise_correlation": correlation(
                group_inner, group_pointwise
            ),
            "inner_selected_score": float(group_inner[selected]),
            "pointwise_selected_score": float(group_pointwise[selected]),
            "inner_baseline_score": float(group_inner[baseline_index]),
            "pointwise_baseline_score": float(group_pointwise[baseline_index]),
            "group_size_norm": float(size / 144.0),
            "selected_alpha": metadata[selected]["alpha"],
            "selected_match_power": metadata[selected]["match_power"],
        })
        cursor += size
    assert cursor == len(features)
    return {
        "ious": np.asarray(selected_ious, dtype=np.float32),
        "features": np.asarray(selected_features, dtype=np.float32),
        "boxes": np.asarray(selected_boxes, dtype=np.float32),
        "queries": np.asarray(selected_queries, dtype=np.int32),
        "contexts": contexts,
        "metas": metas,
        "safe_config": safe_config,
        "provenance": provenance,
    }


def iter_source_rows(dump_path):
    manifest = torch.load(dump_path, map_location="cpu")
    assert manifest["format"] == "source_choice_feature_dump_sharded_v1"
    assert int(manifest["topk"]) == 1
    base = os.path.dirname(dump_path)
    count = 0
    for relative in manifest["shards"]:
        payload = torch.load(os.path.join(base, relative), map_location="cpu")
        for row in payload["rows"]:
            count += 1
            yield row
    assert count == int(manifest["row_count"]), (count, manifest["row_count"])


def selected_stage150(compact_row):
    scores = as_array(compact_row["adapter_score_at_candidate"])
    ious = as_array(compact_row["adapter_iou_at_candidate"])
    boxes = np.asarray(
        compact_row["adapter_box_at_candidate"], dtype=np.float32
    )
    queries = np.asarray(
        compact_row["adapter_candidate_query"], dtype=np.int32
    )
    assert len(scores) == len(ious) == len(boxes) == len(queries)
    selected = int(np.argmax(scores))
    return {
        "index": selected,
        "iou": float(ious[selected]),
        "box": boxes[selected],
        "query": int(queries[selected]),
        "scores": scores,
        "queries": queries,
    }


def feature_dict(
    source_row, source_feature_names, compact_row, raw_row,
    stage142_features, stage142_box, stage142_query, stage142_context,
):
    result = {}
    for name in source_feature_names:
        result["source__" + name] = scalar(source_row.get(name, 0.0))

    for key in COMPACT_ARRAY_KEYS:
        compact_stats(result, "compact__" + key, compact_row.get(key, []))

    selected = selected_stage150(compact_row)
    selected_index = selected["index"]
    result["compact__adapter_selected_index_norm"] = float(
        selected_index / max(len(selected["scores"]) - 1, 1)
    )
    result["compact__adapter_candidate_count"] = float(len(selected["scores"]))
    result["compact__adapter_rescue_gate"] = float(
        bool(compact_row.get("adapter_rescue_gate", False))
    )
    result["compact__selected_is_rescue_query"] = float(
        selected["query"] == int(compact_row.get("adapter_rescue_query", -1))
    )
    result["compact__selected_is_fallback_query"] = float(
        selected["query"] == int(compact_row.get("adapter_fallback_query", -1))
    )
    result["compact__detector_class_count"] = scalar(
        compact_row.get("detector_class_count", 0.0)
    )
    for key in (
        "detector_conf20_count", "detector_conf30_count",
        "detector_conf40_count", "detector_conf50_count",
    ):
        result["compact__" + key] = scalar(compact_row.get(key, 0.0))

    selected_size = np.maximum(np.abs(selected["box"][3:]), 1e-5)
    for axis, value in zip("xyz", np.log(selected_size)):
        result["compact__selected_log_size_" + axis] = float(value)
    result["compact__selected_log_volume"] = float(
        np.log(np.prod(selected_size) + 1e-8)
    )

    detector_classes = np.asarray(
        compact_row.get("detected_class_id", []), dtype=np.int64
    ).reshape(-1)
    detector_confidence = as_array(
        compact_row.get("detected_target_confidence", [])
    )
    text_cid = compact_row.get("text_target_cid", None)
    if text_cid is not None and len(detector_classes):
        matches = detector_classes == int(text_cid)
        result["compact__text_target_detector_match_count"] = float(matches.sum())
        result["compact__text_target_detector_match_ratio"] = float(matches.mean())
        result["compact__text_target_detector_match_conf_max"] = (
            float(detector_confidence[matches].max())
            if bool(matches.any()) and len(detector_confidence) == len(matches)
            else 0.0
        )
    else:
        result["compact__text_target_detector_match_count"] = 0.0
        result["compact__text_target_detector_match_ratio"] = 0.0
        result["compact__text_target_detector_match_conf_max"] = 0.0

    for value in DECOMP_VALUES:
        result["meta__decomp__" + value] = float(
            str(raw_row.get("decomposition_status", "")) == value
        )
    for value in SPACY_AUG_VALUES:
        result["meta__spacy_aug__" + value] = float(
            str(raw_row.get("spacy_augmentation_bucket", "")) == value
        )
    for value in SPACY_PROFILE_VALUES:
        result["meta__spacy_profile__" + value] = float(
            str(raw_row.get("spacy_profile_bucket", "")) == value
        )

    for name, value in zip(option_ranker.FEATURE_NAMES, stage142_features):
        result["stage142_option__" + name] = float(value)
    for name, value in stage142_context.items():
        result["stage142_context__" + name] = float(value)

    result["cross__same_query"] = float(selected["query"] == int(stage142_query))
    result["cross__box_iou"] = pair_iou(selected["box"], stage142_box)
    old_size = np.maximum(np.abs(stage142_box[3:]), 1e-5)
    new_size = np.maximum(np.abs(selected["box"][3:]), 1e-5)
    center_delta = (selected["box"][:3] - stage142_box[:3]) / (
        0.5 * (old_size + new_size) + 1e-5
    )
    for axis, value in zip("xyz", center_delta):
        result["cross__normalized_center_delta_" + axis] = float(value)
    for axis, value in zip("xyz", np.log(new_size / old_size)):
        result["cross__log_size_ratio_" + axis] = float(value)

    for source in (
        "base", "fused", "quality", "contrastive_base", "acd",
        "target_detector", "target_detector_logit",
    ):
        query = int(scalar(source_row.get(source + "_top_query", -1.0), -1.0))
        result["cross__stage150_same_as_" + source] = float(
            selected["query"] == query
        )
        result["cross__stage142_same_as_" + source] = float(
            int(stage142_query) == query
        )

    old_query_positions = np.nonzero(selected["queries"] == int(stage142_query))[0]
    result["cross__stage142_query_in_stage150_pool"] = float(
        len(old_query_positions) > 0
    )
    for key in (
        "adapter_score_at_candidate", "adapter_fused_at_candidate",
        "adapter_hit25_logit_at_candidate", "adapter_hit50_logit_at_candidate",
        "adapter_rescue_logit_at_candidate",
    ):
        values = as_array(compact_row.get(key, []))
        new_value = float(values[selected_index]) if selected_index < len(values) else 0.0
        old_value = (
            float(values[int(old_query_positions[0])])
            if len(old_query_positions) and int(old_query_positions[0]) < len(values)
            else 0.0
        )
        result["cross__" + key + "__new"] = new_value
        result["cross__" + key + "__old"] = old_value
        result["cross__" + key + "__new_minus_old"] = new_value - old_value

    for name, value in result.items():
        assert math.isfinite(float(value)), (name, value)
    return result


def build_matrix(
    source_dump, compact_dump, raw_rows, stage142, locked_feature_names=None
):
    compact_rows = torch.load(compact_dump, map_location="cpu")["rows"]
    assert len(compact_rows) == len(raw_rows) == len(stage142["ious"])
    source_iterator = iter_source_rows(source_dump)
    first_source = next(source_iterator)
    source_feature_names = choose_source_feature_names(first_source)
    if locked_feature_names is not None:
        source_feature_names = list(locked_feature_names)

    source_rows = itertools.chain([first_source], source_iterator)
    vectors = []
    feature_names = None
    new_ious = []
    for index, (source_row, compact_row, raw_row) in enumerate(
        zip(source_rows, compact_rows, raw_rows)
    ):
        assert int(compact_row["example_id"]) == index
        assert int(raw_row.get("example_id", index)) == index
        assert np.allclose(
            np.asarray(compact_row["gt_box"], dtype=np.float32),
            np.asarray(raw_row["gt_box"], dtype=np.float32),
            rtol=0.0, atol=1e-5,
        ), index
        raw_queries = np.asarray(
            raw_row["adapter_candidate_query"], dtype=np.int32
        ).reshape(-1)
        compact_queries = np.asarray(
            compact_row["adapter_candidate_query"], dtype=np.int32
        ).reshape(-1)
        assert len(raw_queries) == len(compact_queries), index
        assert np.array_equal(
            np.sort(raw_queries), np.sort(compact_queries)
        ), (index, raw_queries.tolist(), compact_queries.tolist())
        selected = selected_stage150(compact_row)
        values = feature_dict(
            source_row, source_feature_names, compact_row, raw_row,
            stage142["features"][index], stage142["boxes"][index],
            stage142["queries"][index], stage142["contexts"][index],
        )
        if feature_names is None:
            feature_names = sorted(values)
        assert sorted(values) == feature_names
        vectors.append([values[name] for name in feature_names])
        new_ious.append(selected["iou"])
    assert len(vectors) == len(raw_rows)
    matrix = np.asarray(vectors, dtype=np.float32)
    assert matrix.shape == (len(raw_rows), len(feature_names))
    assert np.isfinite(matrix).all()
    return (
        matrix, np.asarray(new_ious, dtype=np.float32),
        feature_names, source_feature_names,
    )


def hit(values, threshold):
    return np.asarray(values) > float(threshold)


def summarize(values):
    values = np.asarray(values, dtype=np.float32)
    return {
        "count": int(len(values)),
        "hits025": int(hit(values, 0.25).sum()),
        "hits050": int(hit(values, 0.50).sum()),
        "acc025": float(hit(values, 0.25).mean()),
        "acc050": float(hit(values, 0.50).mean()),
        "mean_iou": float(values.mean()),
    }


def fix_break(before, after, changed):
    result = {"changed": int(np.asarray(changed, dtype=bool).sum())}
    for threshold, suffix in ((0.25, "025"), (0.50, "050")):
        old = hit(before, threshold)
        new = hit(after, threshold)
        fixes = int((~old & new).sum())
        breaks = int((old & ~new).sum())
        result["fix_" + suffix] = fixes
        result["break_" + suffix] = breaks
        result["net_" + suffix] = fixes - breaks
    return result


def evaluate_scores(scores, old_ious, new_ious, threshold):
    changed = np.asarray(scores) >= float(threshold)
    selected = np.where(changed, new_ious, old_ious)
    return {
        "selected": summarize(selected),
        "default_stage142": summarize(old_ious),
        "stage150": summarize(new_ious),
        "changed_ratio": float(changed.mean()),
        "threshold": float(threshold),
        "fix_break": fix_break(old_ious, selected, changed),
    }


def choose_threshold(scores, old_ious, new_ious):
    scores = np.asarray(scores, dtype=np.float32)
    finite = scores[np.isfinite(scores)]
    assert len(finite) == len(scores) and len(finite) > 0
    thresholds = list(np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 301))))
    thresholds.extend([float("inf"), float("-inf")])
    default = summarize(old_ious)
    tolerance025 = max(1, int(math.ceil(0.001 * len(old_ious))))
    feasible = []
    all_rows = []
    for threshold in thresholds:
        result = evaluate_scores(scores, old_ious, new_ious, threshold)
        fb = result["fix_break"]
        result["preserves_acc025"] = bool(
            result["selected"]["hits025"] >= default["hits025"] - tolerance025
        )
        result["precision_gate"] = bool(
            fb["fix_050"] >= fb["break_050"] + 2
            and fb["fix_050"] >= 1.20 * max(1, fb["break_050"])
        )
        result["conservative_change_gate"] = bool(
            result["changed_ratio"] <= 0.20
        )
        all_rows.append(result)
        if (
            result["preserves_acc025"]
            and result["precision_gate"]
            and result["conservative_change_gate"]
        ):
            feasible.append(result)
    if not feasible:
        max_score = float(finite.max())
        abstain_threshold = max_score + max(1.0, abs(max_score))
        assert math.isfinite(abstain_threshold)
        abstain = evaluate_scores(
            scores, old_ious, new_ious, abstain_threshold
        )
        abstain.update({
            "preserves_acc025": True,
            "precision_gate": False,
            "conservative_change_gate": True,
            "no_positive_feasible_threshold": True,
        })
        return abstain
    return max(feasible, key=lambda row: (
        row["selected"]["hits050"],
        row["selected"]["hits025"],
        row["selected"]["mean_iou"],
        -row["changed_ratio"],
    ))


def labels(old_ious, new_ious):
    old25, new25 = hit(old_ious, 0.25), hit(new_ious, 0.25)
    old50, new50 = hit(old_ious, 0.50), hit(new_ious, 0.50)
    utility = (
        4.0 * (new50.astype(np.float32) - old50.astype(np.float32))
        + 1.0 * (new25.astype(np.float32) - old25.astype(np.float32))
        + 0.1 * (new_ious - old_ious)
    )
    return {
        "utility": utility.astype(np.float32),
        "beneficial": utility > 0.0,
        "fix050": (~old50) & new50,
        "break050": old50 & (~new50),
        "break025": old25 & (~new25),
        "new_hit050": new50.astype(np.int32),
        "old_hit050": old50.astype(np.int32),
        "disagree050": old50 != new50,
    }


def learner_common(seed):
    return dict(
        n_estimators=500,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=5,
        min_child_samples=160,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=1.0,
        reg_lambda=8.0,
        random_state=seed,
        n_jobs=16,
        verbosity=-1,
    )


def fit_candidates(train_x, train_old, train_new):
    train_labels = labels(train_old, train_new)
    candidates = []

    weights = np.ones(len(train_x), dtype=np.float32)
    weights[train_labels["beneficial"]] = 3.0
    weights[train_labels["fix050"]] = 8.0
    weights[train_labels["break050"]] = 12.0
    weights[train_labels["break025"]] = 20.0
    utility = lgb.LGBMRegressor(objective="huber", **learner_common(15301))
    utility.fit(train_x, train_labels["utility"], sample_weight=weights)
    candidates.append({"name": "utility_regression", "models": [utility]})

    benefit = lgb.LGBMClassifier(objective="binary", **learner_common(15302))
    benefit.fit(
        train_x, train_labels["beneficial"].astype(np.int32),
        sample_weight=weights,
    )
    candidates.append({"name": "benefit_classifier", "models": [benefit]})

    disagreement = train_labels["disagree050"]
    assert int(disagreement.sum()) > 100
    disagree_weights = np.ones(int(disagreement.sum()), dtype=np.float32)
    disagree_target = train_labels["fix050"][disagreement].astype(np.int32)
    disagree_weights[disagree_target == 0] = 1.5
    fix_vs_break = lgb.LGBMClassifier(
        objective="binary", **learner_common(15303)
    )
    fix_vs_break.fit(
        train_x[disagreement], disagree_target,
        sample_weight=disagree_weights,
    )
    candidates.append({"name": "fix_vs_break_classifier", "models": [fix_vs_break]})

    old_hit = lgb.LGBMClassifier(objective="binary", **learner_common(15304))
    new_hit = lgb.LGBMClassifier(objective="binary", **learner_common(15305))
    old_hit.fit(train_x, train_labels["old_hit050"])
    new_hit.fit(train_x, train_labels["new_hit050"])
    candidates.append({"name": "dual_hit050_advantage", "models": [old_hit, new_hit]})
    return candidates, train_labels


def predict_candidate(candidate, features):
    name = candidate["name"]
    models = candidate["models"]
    if name == "utility_regression":
        return models[0].predict(features)
    if name in ("benefit_classifier", "fix_vs_break_classifier"):
        return models[0].predict_proba(features)[:, 1]
    old_probability = np.clip(models[0].predict_proba(features)[:, 1], 1e-5, 1-1e-5)
    new_probability = np.clip(models[1].predict_proba(features)[:, 1], 1e-5, 1-1e-5)
    old_logit = np.log(old_probability / (1.0 - old_probability))
    new_logit = np.log(new_probability / (1.0 - new_probability))
    return new_logit - old_logit


def save_candidate(candidate, output_dir):
    paths = []
    for index, learner in enumerate(candidate["models"]):
        path = os.path.join(output_dir, "selector_model_{:02d}.txt".format(index))
        learner.booster_.save_model(path, num_iteration=500)
        paths.append({
            "path": os.path.abspath(path),
            "sha256": sha256(path),
            "iteration": 500,
        })
    return paths


def load_candidate(lock):
    models = []
    for item in lock["models"]:
        assert sha256(item["path"]) == item["sha256"]
        models.append(lgb.Booster(model_file=item["path"]))
    return {"name": lock["selected_candidate"], "models": models}


def booster_predict(candidate, features):
    name = candidate["name"]
    models = candidate["models"]
    predictions = [
        model.predict(features, num_iteration=500) for model in models
    ]
    if name in ("utility_regression", "benefit_classifier", "fix_vs_break_classifier"):
        return predictions[0]
    old_probability = np.clip(predictions[0], 1e-5, 1-1e-5)
    new_probability = np.clip(predictions[1], 1e-5, 1-1e-5)
    return (
        np.log(new_probability / (1.0 - new_probability))
        - np.log(old_probability / (1.0 - old_probability))
    )


def train(args):
    assert not os.path.exists(args.output_dir), args.output_dir
    os.makedirs(args.output_dir)
    raw_rows = option_ranker.load_rows(args.stage142_train_dump)
    assert len(raw_rows) == 36665
    stage142 = build_stage142_arrays(
        raw_rows, args.stage142_lock, args.stage31_lock, args.stage33_lock
    )
    matrix, new_ious, feature_names, source_feature_names = build_matrix(
        args.stage150_source_dump, args.stage150_compact_dump,
        raw_rows, stage142,
    )
    splits = option_ranker.split_indices(stage142["metas"])
    train_indices = np.asarray(splits["train"], dtype=np.int64)
    dev_indices = np.asarray(splits["dev"], dtype=np.int64)
    test_indices = np.asarray(splits["test"], dtype=np.int64)
    candidates, train_labels = fit_candidates(
        matrix[train_indices], stage142["ious"][train_indices],
        new_ious[train_indices],
    )
    candidate_reports = []
    for candidate in candidates:
        scores = predict_candidate(candidate, matrix[dev_indices])
        dev = choose_threshold(
            scores, stage142["ious"][dev_indices], new_ious[dev_indices]
        )
        candidate_reports.append({
            "name": candidate["name"], "dev": dev, "candidate": candidate,
        })
        print(json.dumps({"name": candidate["name"], "dev": dev}, sort_keys=True))
    selected = max(candidate_reports, key=lambda item: (
        item["dev"]["selected"]["hits050"],
        item["dev"]["selected"]["hits025"],
        item["dev"]["selected"]["mean_iou"],
        -item["dev"]["changed_ratio"],
    ))
    model_items = save_candidate(selected["candidate"], args.output_dir)
    locked_candidate = load_candidate({
        "selected_candidate": selected["name"], "models": model_items,
    })
    test_scores = booster_predict(locked_candidate, matrix[test_indices])
    internal_test = evaluate_scores(
        test_scores, stage142["ious"][test_indices], new_ious[test_indices],
        selected["dev"]["threshold"],
    )
    test_tolerance025 = max(1, int(math.ceil(0.001 * len(test_indices))))
    fb = internal_test["fix_break"]
    internal_gate_pass = bool(
        internal_test["selected"]["hits050"]
        >= internal_test["default_stage142"]["hits050"] + 5
        and fb["fix_050"] > fb["break_050"]
        and internal_test["selected"]["hits025"]
        >= internal_test["default_stage142"]["hits025"] - test_tolerance025
        and internal_test["changed_ratio"] <= 0.20
    )
    label_counts = {
        name: int(np.asarray(value).sum())
        for name, value in train_labels.items()
        if name != "utility"
    }
    label_counts.update({
        "rows": int(len(train_indices)),
        "utility_positive": int((train_labels["utility"] > 0).sum()),
        "utility_negative": int((train_labels["utility"] < 0).sum()),
    })
    lock = {
        "stage": "153_train_only_stage142_stage150_source_selector",
        "status": "complete_train_only_lock",
        "protocol": (
            "scanrefer_train_scene70_fit_scene15_dev_lock_scene15_test_"
            "rich_cross_checkpoint_conservative_selector_v1"
        ),
        "selection_data_scope": "scanrefer_train_scenes_only",
        "validation_labels_used_for_selection": False,
        "validation_evaluation_authorized": internal_gate_pass,
        "script": os.path.abspath(__file__),
        "script_sha256": sha256(os.path.abspath(__file__)),
        "stage142_train_dump": os.path.abspath(args.stage142_train_dump),
        "stage142_train_dump_sha256": sha256(args.stage142_train_dump),
        "stage150_source_dump": os.path.abspath(args.stage150_source_dump),
        "stage150_source_dump_sha256": sha256(args.stage150_source_dump),
        "stage150_compact_dump": os.path.abspath(args.stage150_compact_dump),
        "stage150_compact_dump_sha256": sha256(args.stage150_compact_dump),
        "stage142_lock": os.path.abspath(args.stage142_lock),
        "stage142_lock_sha256": sha256(args.stage142_lock),
        "stage31_lock_sha256": sha256(args.stage31_lock),
        "stage33_lock_sha256": sha256(args.stage33_lock),
        "provenance": stage142["provenance"],
        "safe_config": stage142["safe_config"],
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "source_feature_names": source_feature_names,
        "selected_candidate": selected["name"],
        "models": model_items,
        "selected_dev": selected["dev"],
        "all_dev_candidates": [
            {"name": item["name"], "dev": item["dev"]}
            for item in candidate_reports
        ],
        "internal_scene_test": internal_test,
        "internal_gate_pass": internal_gate_pass,
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "train_label_counts": label_counts,
    }
    lock_path = os.path.join(args.output_dir, "locked_source_selector.json")
    atomic_json(lock_path, lock)
    print(json.dumps({
        "lock": os.path.abspath(lock_path),
        "lock_sha256": sha256(lock_path),
        "selected_candidate": selected["name"],
        "selected_dev": selected["dev"],
        "internal_scene_test": internal_test,
        "internal_gate_pass": internal_gate_pass,
        "validation_evaluation_authorized": internal_gate_pass,
        "feature_count": len(feature_names),
        "train_label_counts": label_counts,
    }, indent=2, sort_keys=True))


def evaluate(args):
    lock = read_json(args.policy_lock)
    assert lock["stage"] == "153_train_only_stage142_stage150_source_selector"
    assert lock["validation_labels_used_for_selection"] is False
    assert lock["validation_evaluation_authorized"] is True
    assert sha256(lock["script"]) == lock["script_sha256"]
    raw_rows = option_ranker.load_rows(args.stage142_dump)
    stage142 = build_stage142_arrays(
        raw_rows, args.stage142_lock, args.stage31_lock, args.stage33_lock
    )
    matrix, new_ious, feature_names, source_feature_names = build_matrix(
        args.stage150_source_dump, args.stage150_compact_dump,
        raw_rows, stage142, locked_feature_names=lock["source_feature_names"],
    )
    assert feature_names == lock["feature_names"]
    assert source_feature_names == lock["source_feature_names"]
    candidate = load_candidate(lock)
    scores = booster_predict(candidate, matrix)
    result_metrics = evaluate_scores(
        scores, stage142["ious"], new_ious,
        lock["selected_dev"]["threshold"],
    )
    count = len(raw_rows)
    strict025 = math.floor(0.5391 * count) + 1
    strict050 = math.floor(0.4241 * count) + 1
    result = {
        "stage": "153_source_selector_validation_eval",
        "status": "complete",
        "diagnostic_only_until_integrated_and_independently_reloaded": True,
        "policy_lock": os.path.abspath(args.policy_lock),
        "policy_lock_sha256": sha256(args.policy_lock),
        "validation_labels_used_for_selection": False,
        "metrics": result_metrics,
        "strict_goal_hits": {"acc025": strict025, "acc050": strict050},
        "strict_goal_met_offline": bool(
            result_metrics["selected"]["hits025"] >= strict025
            and result_metrics["selected"]["hits050"] >= strict050
        ),
    }
    assert not os.path.exists(args.output_json), args.output_json
    atomic_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("stage142_train_dump")
    train_parser.add_argument("stage150_source_dump")
    train_parser.add_argument("stage150_compact_dump")
    train_parser.add_argument("stage31_lock")
    train_parser.add_argument("stage33_lock")
    train_parser.add_argument("stage142_lock")
    train_parser.add_argument("output_dir")
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("stage142_dump")
    eval_parser.add_argument("stage150_source_dump")
    eval_parser.add_argument("stage150_compact_dump")
    eval_parser.add_argument("stage31_lock")
    eval_parser.add_argument("stage33_lock")
    eval_parser.add_argument("stage142_lock")
    eval_parser.add_argument("policy_lock")
    eval_parser.add_argument("output_json")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
