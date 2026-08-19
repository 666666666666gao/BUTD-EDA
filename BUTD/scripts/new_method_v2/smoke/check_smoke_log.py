#!/usr/bin/env python3
"""Validate new_method_v2 smoke log directories."""

import argparse
import json
import re
from pathlib import Path


BAD_PATTERNS = (
    "Traceback",
    "RuntimeError",
    "ValueError",
    "AssertionError",
    "CUDA out of memory",
)
BAD_REGEX = re.compile(r"(?<![A-Za-z0-9_])(?:nan|inf|-inf)(?![A-Za-z0-9_])", re.IGNORECASE)
METRIC_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_@.]*)(?:\s*[:=]\s*)"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

CANONICAL_METRIC_ALIASES = {
    "dbg_quality_corr_pred_iou_target_iou": (
        "dbg_quality_iou_corr",
        "dbg_quality_pred_target_iou_corr",
    ),
    "dbg_quality_top1_quality_improves_ratio": (
        "dbg_quality_top1_iou_improvement",
        "dbg_quality_top1_improves_ratio",
    ),
    "dbg_qahnl_pos_query_ratio": (
        "dbg_qahnl_positive_query_ratio",
    ),
    "dbg_qahnl_neg_query_ratio": (
        "dbg_qahnl_negative_query_ratio",
    ),
    "dbg_qahnl_iou_gap_mean": (
        "dbg_qahnl_iou_gap",
    ),
    "dbg_qahnl_score_gap_mean": (
        "dbg_qahnl_score_gap",
    ),
    "dbg_qahnl_loss_unweighted": (
        "dbg_qahnl_loss_raw",
    ),
}

QUALITY_EXPECTED = {
    "dbg_quality_corr_pred_iou_target_iou",
    "dbg_quality_top1_quality_improves_ratio",
}
QAHNL_EXPECTED = {
    "dbg_qahnl_pos_query_ratio",
    "dbg_qahnl_neg_query_ratio",
    "dbg_qahnl_iou_gap_mean",
    "dbg_qahnl_score_gap_mean",
    "dbg_qahnl_margin_mean",
    "dbg_qahnl_loss_unweighted",
}

MODE_RULES = {
    "baseline": {
        "expected": {"eval_primary_score_source_id"},
        "forbidden_prefixes": ("dbg_sacr_", "dbg_rapf_", "dbg_quality_", "dbg_qahnl_"),
        "config": {
            "use_quality_head": False,
            "use_sacr": False,
            "use_rapf": False,
            "use_qahnl": False,
        },
    },
    "quality_only": {
        "expected": QUALITY_EXPECTED,
        "forbidden_prefixes": ("dbg_sacr_", "dbg_rapf_", "dbg_qahnl_"),
        "config": {
            "use_quality_head": True,
            "use_sacr": False,
            "use_rapf": False,
            "use_qahnl": False,
        },
    },
    "sacr_only": {
        "expected": {"dbg_sacr_structured_valid_ratio"},
        "forbidden_prefixes": ("dbg_rapf_", "dbg_qahnl_", "dbg_quality_"),
        "config": {
            "use_structured_slots": True,
            "use_sacr": True,
            "use_rapf": False,
            "use_qahnl": False,
            "use_quality_head": False,
        },
    },
    "rapf_quality": {
        "expected": {"dbg_sacr_structured_valid_ratio", "dbg_rapf_gate_mean"} | QUALITY_EXPECTED,
        "forbidden_prefixes": ("dbg_qahnl_",),
        "config": {
            "use_structured_slots": True,
            "use_sacr": True,
            "use_rapf": True,
            "use_reliability_gate": True,
            "use_quality_head": True,
            "rapf_use_quality": True,
            "use_qahnl": False,
        },
    },
    "full": {
        "expected": {
            "dbg_sacr_structured_valid_ratio",
            "dbg_rapf_gate_mean",
            "eval_target_row_uses_primary_score",
            "eval_anchor_rows_use_baseline_score",
            "eval_bbf_is_diagnostic_only",
        } | QUALITY_EXPECTED | QAHNL_EXPECTED,
        "forbidden_prefixes": (),
        "config": {
            "use_structured_slots": True,
            "use_sacr": True,
            "use_rapf": True,
            "use_reliability_gate": True,
            "use_quality_head": True,
            "rapf_use_quality": True,
            "use_qahnl": True,
        },
        "must_be_zero": {
            "dbg_qahnl_ambiguous_as_negative_ratio",
            "dbg_positive_map_fallback_used_ratio",
            "positive_map_fallback_used_ratio",
            "dbg_warn_global_only_target_positive_map_ratio",
        },
    },
    "full_no_quality": {
        "expected": {
            "dbg_sacr_structured_valid_ratio",
            "dbg_rapf_gate_mean",
        } | QAHNL_EXPECTED,
        "forbidden_prefixes": ("dbg_quality_",),
        "forbidden_terms": ("quality_logits", "pred_iou", "loss_quality"),
        "config": {
            "use_structured_slots": True,
            "use_sacr": True,
            "use_rapf": True,
            "use_reliability_gate": True,
            "use_quality_head": False,
            "rapf_use_quality": False,
            "use_qahnl": True,
        },
        "config_not": {"qahnl_score_source": "quality"},
    },
}


def iter_text_files(root):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".log", ".txt", ".json"}:
            yield path


def collect_metrics(files):
    metrics = {}
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in METRIC_RE.finditer(text):
            key = match.group("key")
            try:
                value = float(match.group("value"))
            except ValueError:
                continue
            metrics.setdefault(key, []).append(value)
    return metrics


def has_metric(metrics, key):
    if key in metrics:
        return True
    return any(alias in metrics for alias in CANONICAL_METRIC_ALIASES.get(key, ()))


def metric_alias_text(key):
    aliases = CANONICAL_METRIC_ALIASES.get(key, ())
    if not aliases:
        return ""
    return " (aliases: {})".format(", ".join(aliases))


def collect_configs(files):
    configs = []
    for path in files:
        if path.name not in {"config.json", "args.json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        data = json.loads(text)
        if isinstance(data, dict):
            configs.append((path, data))
    return configs


def validate_config(mode, configs):
    if mode is None:
        return []
    if not configs:
        return ["no config.json/args.json files found"]
    rules = MODE_RULES[mode]
    failures = []
    for path, config in configs:
        if not bool(config.get("verbose_diagnostics", False)):
            failures.append(f"{path}: verbose_diagnostics must be enabled for smoke")
        if not bool(config.get("eval_report_diagnostic_scores", False)):
            failures.append(f"{path}: eval_report_diagnostic_scores must be enabled for smoke")
        for key, expected in sorted(rules.get("config", {}).items()):
            actual = bool(config.get(key, False))
            if actual != expected:
                failures.append(f"{path}: {key} expected {expected}, saw {actual}")
        for key, forbidden_value in sorted(rules.get("config_not", {}).items()):
            if config.get(key) == forbidden_value:
                failures.append(f"{path}: {key} must not be {forbidden_value}")
    return failures


def validate_mode(mode, metrics, configs, files):
    if mode is None:
        return []
    rules = MODE_RULES[mode]
    failures = validate_config(mode, configs)
    for key in sorted(rules.get("expected", ())):
        if not has_metric(metrics, key):
            failures.append(f"mode {mode}: missing expected metric {key}{metric_alias_text(key)}")
    for key in sorted(metrics):
        for prefix in rules.get("forbidden_prefixes", ()):
            if key.startswith(prefix):
                failures.append(f"mode {mode}: forbidden metric {key}")
    for key in sorted(rules.get("must_be_zero", ())):
        if key not in metrics:
            failures.append(f"mode {mode}: missing hard-check metric {key}")
            continue
        bad_values = [value for value in metrics[key] if abs(value) > 1e-12]
        if bad_values:
            failures.append(f"mode {mode}: {key} must be 0, saw {bad_values[-1]}")
    forbidden_terms = rules.get("forbidden_terms", ())
    if forbidden_terms:
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for term in forbidden_terms:
                if term in text:
                    failures.append(f"mode {mode}: forbidden term {term} in {path}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--mode", choices=sorted(MODE_RULES), default=None)
    args = parser.parse_args()

    if not args.log_dir.exists():
        raise SystemExit(f"log directory does not exist: {args.log_dir}")

    files = list(iter_text_files(args.log_dir))
    if not files:
        raise SystemExit(f"no log/config files found under {args.log_dir}")

    configs = collect_configs(files)
    config_count = len(configs)
    eval_count = 0
    failures = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.name.startswith("eval_epoch_"):
            eval_count += 1
        for pattern in BAD_PATTERNS:
            if pattern in text:
                failures.append(f"{path}: contains {pattern}")
        if BAD_REGEX.search(text):
            failures.append(f"{path}: contains non-finite scalar text")

    if eval_count == 0:
        failures.append("no eval_epoch_*.log files found")
    failures.extend(validate_mode(args.mode, collect_metrics(files), configs, files))
    if failures:
        raise SystemExit("\n".join(failures))
    mode_msg = f", mode={args.mode}" if args.mode else ""
    print(f"check_smoke_log: ok ({len(files)} files, {config_count} configs, {eval_count} eval logs{mode_msg})")


if __name__ == "__main__":
    main()
