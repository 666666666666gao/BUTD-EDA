#!/usr/bin/env python3
"""Audit diagnostic metrics and regenerate the registry artifacts."""

import json
import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "docs" / "DIAGNOSTIC_METRICS_REGISTRY.md"
AUDIT_PATH = ROOT / "docs" / "DIAGNOSTIC_METRICS_AUDIT.md"
JSON_PATH = ROOT / "diagnostics_audit.json"

METRIC_RE = re.compile(
    r"['\"]((?:dbg|diag|eval|loss|positive_map|metadata_conflict)_[A-Za-z0-9_@.]+)['\"]"
)
ALLOWED_WARNING_SEVERITY = {"hard_error", "high_risk", "info"}

CORE_DEBUG = {
    "dbg_metadata_conflict_ratio",
    "metadata_conflict_ratio",
    "positive_map_target_missing_ratio",
    "positive_map_fallback_used_ratio",
    "positive_map_global_only_target_empty_ratio",
    "dbg_sacr_structured_valid_ratio",
    "dbg_sacr_global_only_ratio",
    "dbg_sacr_weak_generic_ratio",
    "dbg_rapf_gate_mean",
    "dbg_rapf_wrong_to_right_ratio",
    "dbg_rapf_right_to_wrong_ratio",
    "dbg_quality_corr_pred_iou_target_iou",
    "dbg_quality_top1_quality_improves_ratio",
    "dbg_qahnl_ambiguous_as_negative_ratio",
    "dbg_qahnl_pos_query_ratio",
    "dbg_qahnl_neg_query_ratio",
    "dbg_qahnl_iou_gap_mean",
    "dbg_qahnl_score_gap_mean",
    "dbg_qahnl_valid_batch_ratio",
    "dbg_qahnl_violation_ratio",
    "dbg_qahnl_global_only_used_ratio",
    "dbg_qahnl_global_only_skipped_structured_ratio",
    "dbg_qahnl_weak_generic_used_ratio",
    "dbg_warn_global_only_target_positive_map_ratio",
    "eval_primary_score_source_id",
    "eval_bbs_score_source_id",
    "eval_bbf_score_source_id",
    "eval_target_row_uses_primary_score",
    "eval_anchor_rows_use_baseline_score",
    "eval_bbf_is_diagnostic_only",
}

TRAINING_LOSS_TERMS = {
    "loss_ce",
    "loss_bbox",
    "loss_giou",
    "loss_contrastive_align",
    "loss_constrastive_align",
    "loss_s2s_aux",
    "loss_acd_rank",
    "loss_quality",
    "loss_sacr_rank",
    "loss_rapf_gate",
    "loss_qahnl",
    "loss_dhc_consistency",
    "loss_dhc_ent_hardneg",
    "loss_dhc_attr_hardneg",
    "loss_dhc_rel_hardneg",
}

LEGACY_ALIASES = {
    "dbg_quality_iou_corr": "dbg_quality_corr_pred_iou_target_iou",
    "dbg_quality_pred_target_iou_corr": "dbg_quality_corr_pred_iou_target_iou",
    "dbg_quality_top1_iou_improvement": "dbg_quality_top1_quality_improves_ratio",
    "dbg_quality_top1_improves_ratio": "dbg_quality_top1_quality_improves_ratio",
    "dbg_qahnl_positive_query_ratio": "dbg_qahnl_pos_query_ratio",
    "dbg_qahnl_negative_query_ratio": "dbg_qahnl_neg_query_ratio",
    "dbg_qahnl_iou_gap": "dbg_qahnl_iou_gap_mean",
    "dbg_qahnl_score_gap": "dbg_qahnl_score_gap_mean",
    "dbg_qahnl_global_only_skipped_ratio": "dbg_qahnl_global_only_skipped_structured_ratio",
    "positive_map_fallback_used_ratio": "dbg_positive_map_fallback_used_ratio",
}
BACKUP_SUFFIXES = (".bak", ".backup", ".orig", ".tmp", ".swp", "~")
EXCLUDED_PARTS = {"__pycache__"}

EXPECTED_METRICS = {
    "dbg_rapf_gate_std",
    "dbg_rapf_gate_min",
    "dbg_rapf_gate_max",
    "dbg_rapf_global_only_gate_mean",
    "dbg_rapf_generic_gate_mean",
    "dbg_rapf_residual_abs_mean",
    "dbg_rapf_residual_clip_ratio",
    "dbg_rapf_delta_mean",
    "dbg_rapf_delta_abs_mean",
    "dbg_rapf_entropy_mean",
    "dbg_rapf_margin_mean",
    "dbg_rapf_disagree_ratio",
    "dbg_rapf_js_mean",
    "dbg_rapf_quality_max_mean",
    "dbg_rapf_top1_correct_ratio",
    "dbg_rapf_iou_delta_mean",
    "dbg_quality_iou50_positive_ratio",
    "dbg_quality_target_iou_mean",
    "dbg_quality_target_iou_std",
    "dbg_quality_pred_iou_mean",
    "dbg_quality_pred_iou_std",
    "dbg_quality_corr_pred_iou_target_iou",
    "dbg_quality_pred_target_iou_corr",
    "dbg_quality_top1_base_iou",
    "dbg_quality_top1_quality_iou",
    "dbg_quality_top1_improves_ratio",
    "dbg_quality_top1_quality_improves_ratio",
    "dbg_qahnl_positive_query_ratio",
    "dbg_qahnl_pos_query_ratio",
    "dbg_qahnl_negative_query_ratio",
    "dbg_qahnl_neg_query_ratio",
    "dbg_qahnl_hard_negative_query_ratio",
    "dbg_qahnl_ambiguous_ignore_ratio",
    "dbg_qahnl_ambiguous_as_negative_ratio",
    "dbg_qahnl_pos_iou",
    "dbg_qahnl_neg_iou",
    "dbg_qahnl_hardneg_iou",
    "dbg_qahnl_iou_gap",
    "dbg_qahnl_iou_gap_mean",
    "dbg_qahnl_score_gap",
    "dbg_qahnl_score_gap_mean",
    "dbg_qahnl_margin_mean",
    "dbg_qahnl_margin_min",
    "dbg_qahnl_margin_max",
    "dbg_qahnl_loss_unweighted",
    "dbg_qahnl_loss_weighted",
    "loss_sacr_rank_raw",
    "dbg_sacr_rank_loss_raw",
    "loss_rapf_gate_raw",
    "dbg_rapf_gate_loss_raw",
    "loss_qahnl_raw",
    "dbg_qahnl_loss_raw",
    "dbg_warn_qahnl_no_positive_ratio",
    "dbg_warn_qahnl_no_hard_negative_ratio",
    "dbg_sacr_relation_active_ratio",
}


def should_skip(path, include_archive=False):
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return True
    if any(path.name.endswith(suffix) for suffix in BACKUP_SUFFIXES):
        return True
    if not include_archive and "archive" in rel.parts:
        return True
    return False


def iter_source_files(include_archive=False):
    for rel in ("main_utils.py", "train_dist_mod.py"):
        path = ROOT / rel
        if not should_skip(path, include_archive=include_archive):
            yield path
    for base in ("models", "src", "scripts"):
        root = ROOT / base
        if root.exists():
            for path in root.rglob("*.py"):
                if path == Path(__file__).resolve():
                    continue
                if should_skip(path, include_archive=include_archive):
                    continue
                yield path
    if include_archive:
        archive_root = ROOT / "archive"
        if archive_root.exists():
            for path in archive_root.rglob("*.py"):
                if should_skip(path, include_archive=include_archive):
                    continue
                yield path


def module_for(path):
    rel = path.relative_to(ROOT)
    if len(rel.parts) == 1:
        return rel.stem
    return "/".join(rel.parts[:2])


def collect_metrics(include_archive=False):
    metrics = {}
    for path in iter_source_files(include_archive=include_archive):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        for match in METRIC_RE.finditer(text):
            name = match.group(1)
            if name.endswith("_"):
                continue
            entry = metrics.setdefault(name, {"name": name, "modules": set()})
            entry["modules"].add(module_for(path))
    for name in EXPECTED_METRICS | CORE_DEBUG | set(TRAINING_LOSS_TERMS):
        metrics.setdefault(name, {"name": name, "modules": {"expected_contract"}})
    return metrics


def metric_type(name):
    if name.startswith("dbg_warn_"):
        return "warning"
    if name.startswith("dbg_") and "_loss_" in name:
        return "diagnostic"
    if name.startswith("loss_"):
        return "loss"
    if name.startswith("eval_"):
        return "eval_metadata"
    return "diagnostic"


def status_for(name):
    if name in LEGACY_ALIASES:
        return "alias"
    legacy_prefixes = (
        "dbg_acd_",
        "loss_acd_",
        "dbg_dhc_",
        "loss_dhc_",
        "dbg_s2s_",
    )
    if name.startswith(legacy_prefixes):
        return "legacy_only"
    if name.startswith("loss_") and name.endswith("_raw"):
        return "alias"
    if name.startswith("loss_"):
        return "training_loss"
    return "active"


def severity_for(name):
    if not name.startswith("dbg_warn_"):
        return "info"
    hard_tokens = ("nan", "inf", "unknown", "global_only_target_positive_map")
    high_tokens = (
        "no_valid",
        "ambiguous",
        "right_to_wrong",
        "fallback",
        "global_only",
    )
    if any(token in name for token in hard_tokens):
        return "hard_error"
    if any(token in name for token in high_tokens):
        return "high_risk"
    return "info"


def expected_range_for(name):
    if name.endswith("_ratio") or "_ratio" in name or name.endswith("_enabled"):
        return "[0, 1]"
    if name.endswith("_corr"):
        return "[-1, 1]"
    if name.startswith("loss_"):
        return ">= 0"
    if name.endswith("_id"):
        return "categorical numeric id"
    return "finite scalar"


def enabled_when_for(name):
    if "qahnl" in name:
        return "--use_qahnl"
    if "rapf" in name:
        return "--use_rapf"
    if "quality" in name:
        return "--use_quality_head"
    if "sacr" in name:
        return "--use_sacr"
    if "_dhc_" in name or name.startswith("loss_dhc_"):
        return "legacy --use_dhc"
    if "_acd_" in name or name.startswith("loss_acd_"):
        return "legacy --use_late_acd"
    if "_s2s_" in name:
        return "legacy --use_s2s_aux_loss"
    if name.startswith("eval_") or name.startswith("diag_"):
        return "evaluation"
    return "always when producer runs"


def replacement_for(name):
    if name in LEGACY_ALIASES:
        return LEGACY_ALIASES[name]
    if name.startswith("loss_") and name.endswith("_raw"):
        stem = name[len("loss_"):-len("_raw")]
        return f"dbg_{stem}_loss_raw"
    if "_dhc_" in name or name.startswith("loss_dhc_"):
        return "QA-HNL / RAPF diagnostics"
    if "_acd_" in name or name.startswith("loss_acd_"):
        return "SACR / RAPF diagnostics"
    if "_s2s_" in name:
        return "coverage_stats / SACR diagnostics"
    return ""


def description_for(name):
    if name.startswith("dbg_warn_"):
        return "Warning indicator emitted for diagnostic review."
    if name.startswith("loss_"):
        return "Training objective term or raw loss diagnostic."
    if name.startswith("eval_"):
        return "Evaluation score-source metadata."
    if name.startswith("diag_"):
        return "Evaluation diagnostic accuracy from an alternate score source."
    return "Runtime diagnostic scalar."


def core_debug_for(name):
    return (
        name in CORE_DEBUG
        or name.startswith("dbg_warn_")
        or name.startswith("dbg_data_decomp_")
        or name.startswith("dbg_positive_map_")
        or name.startswith("dbg_metadata_conflict_")
    )


def total_loss_block():
    path = ROOT / "models" / "losses.py"
    text = path.read_text(encoding="utf-8")
    start = text.rfind("    loss = (")
    end = text.find("    end_points['loss_ce']", start)
    return text[start:end] if start >= 0 and end >= 0 else ""


def build_registry(metrics):
    rows = []
    for name, raw in sorted(metrics.items()):
        status = status_for(name)
        row = {
            "metric_name": name,
            "module": ", ".join(sorted(raw["modules"])),
            "type": metric_type(name),
            "status": status,
            "description": description_for(name),
            "expected_range": expected_range_for(name),
            "enabled_when": enabled_when_for(name),
            "used_in_total_loss": name in TRAINING_LOSS_TERMS,
            "severity": severity_for(name),
            "replacement_if_renamed": replacement_for(name),
        }
        rows.append(row)
    return rows


def write_registry(rows):
    header = [
        "# Diagnostic Metrics Registry",
        "",
        "Generated by `python scripts/audit_diagnostics.py`.",
        "",
        "| metric_name | module | type | status | description | expected_range | enabled_when | used_in_total_loss | severity | replacement_if_renamed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    body = []
    for row in rows:
        body.append(
            "| {name} | {module} | {type} | {status} | {description} | "
            "{expected_range} | {enabled_when} | {used_in_total_loss} | "
            "{severity} | {replacement_if_renamed} |".format(
                name=row["metric_name"],
                **row,
            )
        )
    REGISTRY_PATH.write_text("\n".join(header + body) + "\n", encoding="utf-8")


def write_audit(rows, diagnostic_terms_in_total_loss, missing_warning_severity):
    lines = [
        "# Diagnostic Metrics Audit",
        "",
        "Generated by `python scripts/audit_diagnostics.py`.",
        "",
        "## Summary",
        "",
        f"- Metrics discovered: {len(rows)}",
        f"- Warning metrics without allowed severity: {len(missing_warning_severity)}",
        f"- Diagnostic terms found in total loss block: {len(diagnostic_terms_in_total_loss)}",
        "",
        "## Total Loss Check",
        "",
    ]
    if diagnostic_terms_in_total_loss:
        lines.extend(
            f"- `{name}` appears in the total loss block."
            for name in diagnostic_terms_in_total_loss
        )
    else:
        lines.append("- No `dbg_*`, `diag_*`, or `eval_*` terms appear in total loss aggregation.")
    lines.extend(["", "## Warning Severity Check", ""])
    if missing_warning_severity:
        lines.extend(
            f"- `{name}` has invalid severity `{severity}`."
            for name, severity in missing_warning_severity
        )
    else:
        lines.append("- All `dbg_warn_*` metrics use allowed severity values.")
    AUDIT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Include archived legacy code in the diagnostics scan.",
    )
    args = parser.parse_args()

    metrics = collect_metrics(include_archive=args.include_archive)
    rows = build_registry(metrics)
    block = total_loss_block()
    diagnostic_terms_in_total_loss = sorted(
        set(re.findall(r"\b(?:dbg|diag|eval)_[A-Za-z0-9_@.]+", block))
    )
    missing_warning_severity = sorted(
        (row["metric_name"], row["severity"])
        for row in rows
        if row["metric_name"].startswith("dbg_warn_")
        and row["severity"] not in ALLOWED_WARNING_SEVERITY
    )
    write_registry(rows)
    write_audit(rows, diagnostic_terms_in_total_loss, missing_warning_severity)
    JSON_PATH.write_text(
        json.dumps(
            {
                "metrics": rows,
                "diagnostic_terms_in_total_loss": diagnostic_terms_in_total_loss,
                "missing_warning_severity": missing_warning_severity,
                "allowed_warning_severity": sorted(ALLOWED_WARNING_SEVERITY),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if diagnostic_terms_in_total_loss or missing_warning_severity:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
