#!/usr/bin/env python3
"""Train-only high-precision selector between fixed Stage154 and Stage165."""

import argparse
import json
import math
import os

import numpy as np

import stage153_train_source_selector as base
import stage154_oof_source_selector as oof
import stage167_stage154_stage165_meta_selector as stage167


STAGE = "168_train_only_risk_capped_oof_stage154_stage165_selector"
RESULT_STAGE = "168_risk_capped_meta_selector_validation_eval"
MAX_OOF_CHANGED_RATIO = 0.10
MAX_INTERNAL_CHANGED_RATIO = 0.12
MIN_OOF_FIX_BREAK_RATIO = 1.75
MIN_INTERNAL_FIX_BREAK_RATIO = 1.50


def choose_risk_capped_threshold(scores, old_ious, new_ious):
    scores = np.asarray(scores, dtype=np.float32)
    finite = scores[np.isfinite(scores)]
    assert len(finite) == len(scores) and len(finite) > 0
    thresholds = list(np.unique(
        np.quantile(finite, np.linspace(0.0, 1.0, 501))
    ))
    thresholds.append(float("inf"))
    default = base.summarize(old_ious)
    feasible = []
    for threshold in thresholds:
        result = base.evaluate_scores(
            scores, old_ious, new_ious, float(threshold)
        )
        fb = result["fix_break"]
        result["preserves_acc025"] = bool(
            result["selected"]["hits025"] >= default["hits025"]
        )
        result["risk_cap_gate"] = bool(
            result["changed_ratio"] <= MAX_OOF_CHANGED_RATIO + 1e-12
        )
        result["high_precision_gate"] = bool(
            fb["fix_050"] >= fb["break_050"] + 5
            and fb["fix_050"] >= (
                MIN_OOF_FIX_BREAK_RATIO * max(1, fb["break_050"])
            )
        )
        if (
            result["preserves_acc025"]
            and result["risk_cap_gate"]
            and result["high_precision_gate"]
        ):
            feasible.append(result)
    if not feasible:
        threshold = float(finite.max()) + max(1.0, abs(float(finite.max())))
        result = base.evaluate_scores(scores, old_ious, new_ious, threshold)
        result.update({
            "preserves_acc025": True,
            "risk_cap_gate": True,
            "high_precision_gate": False,
            "no_positive_feasible_threshold": True,
        })
        return result
    return max(feasible, key=lambda row: (
        row["fix_break"]["net_050"],
        row["selected"]["hits025"],
        row["selected"]["mean_iou"],
        -row["changed_ratio"],
    ))


def train(args):
    assert not os.path.exists(args.output_dir), args.output_dir
    os.makedirs(args.output_dir)
    data = stage167.build_inputs(args)
    old_ious = data["stage154_ious"]
    new_ious = data["stage165_ious"]
    development, test = oof.development_and_test(data["metas"])
    folds = oof.oof_folds(data["metas"], development, fold_count=5)
    candidate_names = (
        "utility_regression", "benefit_classifier",
        "fix_vs_break_classifier", "dual_hit050_advantage",
    )
    oof_scores = {
        name: np.full(len(old_ious), np.nan, dtype=np.float32)
        for name in candidate_names
    }
    fold_reports = []
    for item in folds:
        fit, heldout = item["fit"], item["heldout"]
        candidates, _ = base.fit_candidates(
            data["features"][fit], old_ious[fit], new_ious[fit]
        )
        for name in candidate_names:
            candidate = oof.candidate_by_name(candidates, name)
            oof_scores[name][heldout] = base.predict_candidate(
                candidate, data["features"][heldout]
            ).astype(np.float32)
        fold_reports.append({
            "fold": int(item["fold"]),
            "fit_rows": int(len(fit)),
            "heldout_rows": int(len(heldout)),
        })

    reports = []
    for name in candidate_names:
        scores = oof_scores[name][development]
        assert np.isfinite(scores).all()
        locked = choose_risk_capped_threshold(
            scores, old_ious[development], new_ious[development]
        )
        reports.append({"name": name, "oof": locked})
        print(json.dumps(reports[-1], sort_keys=True), flush=True)
    feasible = [
        item for item in reports
        if item["oof"]["high_precision_gate"]
    ]
    selected = max(feasible or reports, key=lambda item: (
        item["oof"]["fix_break"]["net_050"],
        item["oof"]["selected"]["hits025"],
        item["oof"]["selected"]["mean_iou"],
        -item["oof"]["changed_ratio"],
    ))

    final_candidates, _ = base.fit_candidates(
        data["features"][development],
        old_ious[development], new_ious[development],
    )
    final_candidate = oof.candidate_by_name(final_candidates, selected["name"])
    models = base.save_candidate(final_candidate, args.output_dir)
    locked_candidate = base.load_candidate({
        "selected_candidate": selected["name"], "models": models,
    })
    test_scores = base.booster_predict(
        locked_candidate, data["features"][test]
    )
    internal_test = base.evaluate_scores(
        test_scores, old_ious[test], new_ious[test],
        selected["oof"]["threshold"],
    )
    fb = internal_test["fix_break"]
    internal_gate_pass = bool(
        selected["oof"]["high_precision_gate"]
        and internal_test["selected"]["hits050"]
        >= internal_test["default_stage142"]["hits050"] + 5
        and fb["fix_050"] >= fb["break_050"] + 3
        and fb["fix_050"] >= (
            MIN_INTERNAL_FIX_BREAK_RATIO * max(1, fb["break_050"])
        )
        and internal_test["selected"]["hits025"]
        >= internal_test["default_stage142"]["hits025"]
        and internal_test["changed_ratio"]
        <= MAX_INTERNAL_CHANGED_RATIO + 1e-12
    )
    train_labels = base.labels(old_ious[development], new_ious[development])
    lock = {
        "stage": STAGE,
        "status": "complete_train_only_risk_capped_oof_lock",
        "protocol": (
            "scanrefer_train_scene85_five_scene_oof_stage154_stage165_"
            "risk_capped_v1"
        ),
        "selection_data_scope": "scanrefer_train_scenes_only",
        "validation_labels_used_for_selection": False,
        "validation_evaluation_authorized": internal_gate_pass,
        "internal_gate_pass": internal_gate_pass,
        "script": os.path.abspath(__file__),
        "script_sha256": base.sha256(os.path.abspath(__file__)),
        "feature_names": data["feature_names"],
        "feature_count": len(data["feature_names"]),
        "source_feature_names": data["source_feature_names"],
        "selected_candidate": selected["name"],
        "models": models,
        "selected_oof": selected["oof"],
        "all_oof_candidates": reports,
        "fold_reports": fold_reports,
        "internal_scene_test": internal_test,
        "risk_protocol": {
            "max_oof_changed_ratio": MAX_OOF_CHANGED_RATIO,
            "max_internal_changed_ratio": MAX_INTERNAL_CHANGED_RATIO,
            "min_oof_fix_break_ratio": MIN_OOF_FIX_BREAK_RATIO,
            "min_internal_fix_break_ratio": MIN_INTERNAL_FIX_BREAK_RATIO,
            "threshold_selection": (
                "max_net050_then_hits025_then_mean_iou_then_lower_change_"
                "under_train_oof_risk_constraints"
            ),
        },
        "stage154_lock": os.path.abspath(args.stage154_lock),
        "stage154_lock_sha256": base.sha256(args.stage154_lock),
        "stage165_policy": os.path.abspath(args.stage165_policy),
        "stage165_policy_sha256": base.sha256(args.stage165_policy),
        "stage31_lock_sha256": base.sha256(args.stage31_lock),
        "stage33_lock_sha256": base.sha256(args.stage33_lock),
        "stage142_lock_sha256": base.sha256(args.stage142_lock),
        "stage142_provenance": data["stage142_provenance"],
        "stage142_safe_config": data["stage142_safe_config"],
        "stage142_dump_sha256": base.sha256(args.stage142_dump),
        "stage150_source_dump_sha256": base.sha256(args.stage150_source_dump),
        "stage150_compact_dump_sha256": base.sha256(args.stage150_compact_dump),
        "threshold154": data["threshold154"],
        "threshold165": data["threshold165"],
        "split_sizes": {
            "development_oof": int(len(development)),
            "internal_test": int(len(test)),
        },
        "development_label_counts": {
            "rows": int(len(development)),
            "fix050": int(train_labels["fix050"].sum()),
            "break050": int(train_labels["break050"].sum()),
            "disagree050": int(train_labels["disagree050"].sum()),
        },
    }
    lock_path = os.path.join(args.output_dir, "locked_risk_capped_selector.json")
    base.atomic_json(lock_path, lock)
    print(json.dumps({
        "lock": os.path.abspath(lock_path),
        "lock_sha256": base.sha256(lock_path),
        "selected_candidate": selected["name"],
        "selected_oof": selected["oof"],
        "internal_scene_test": internal_test,
        "internal_gate_pass": internal_gate_pass,
        "validation_evaluation_authorized": internal_gate_pass,
    }, indent=2, sort_keys=True))


def evaluate(args):
    lock = base.read_json(args.policy_lock)
    assert lock["stage"] == STAGE
    assert lock["validation_labels_used_for_selection"] is False
    assert lock["validation_evaluation_authorized"] is True
    assert base.sha256(lock["script"]) == lock["script_sha256"]
    data = stage167.build_inputs(args)
    assert data["feature_names"] == lock["feature_names"]
    candidate = base.load_candidate(lock)
    scores = base.booster_predict(candidate, data["features"])
    metrics = base.evaluate_scores(
        scores, data["stage154_ious"], data["stage165_ious"],
        lock["selected_oof"]["threshold"],
    )
    count = len(data["raw_rows"])
    goals = {
        "acc025": math.floor(0.5391 * count) + 1,
        "acc050": math.floor(0.4241 * count) + 1,
    }
    result = {
        "stage": RESULT_STAGE,
        "status": "complete",
        "diagnostic_only_until_integrated_and_independently_reloaded": True,
        "policy_lock": os.path.abspath(args.policy_lock),
        "policy_lock_sha256": base.sha256(args.policy_lock),
        "validation_labels_used_for_selection": False,
        "metrics": metrics,
        "strict_goal_hits": goals,
        "strict_goal_met_offline": bool(
            metrics["selected"]["hits025"] >= goals["acc025"]
            and metrics["selected"]["hits050"] >= goals["acc050"]
        ),
    }
    assert not os.path.exists(args.output_json), args.output_json
    base.atomic_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    stage167.add_inputs(train_parser)
    train_parser.add_argument("output_dir")
    eval_parser = sub.add_parser("evaluate")
    stage167.add_inputs(eval_parser)
    eval_parser.add_argument("policy_lock")
    eval_parser.add_argument("output_json")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
