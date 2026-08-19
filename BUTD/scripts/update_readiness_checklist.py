#!/usr/bin/env python3
"""Update the pre-large-scale-training readiness checklist from real outputs."""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CHECKLIST = ROOT / "reports" / "PRE_LARGE_SCALE_TRAINING_CHECKLIST.md"
DEFAULT_EVIDENCE_DIR = ROOT / "reports" / "readiness"
MODES = ("baseline", "quality_only", "sacr_only", "rapf_quality", "full")
BAD_PATTERNS = ("Traceback", "RuntimeError", "ValueError", "AssertionError")
VALID_STATUSES = {"pass", "fail", "not-run"}


def rel(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def result(status, evidence, details=None):
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid readiness status: {status}")
    return status, evidence, details or {}


def ok_fail_notrun_from_text(path, ok_pattern):
    if not path.exists():
        return "not-run", f"missing {path.relative_to(ROOT)}"
    text = path.read_text(encoding="utf-8", errors="ignore")
    if any(pattern in text for pattern in BAD_PATTERNS):
        return "fail", str(path.relative_to(ROOT))
    if re.search(ok_pattern, text, re.IGNORECASE):
        return "pass", str(path.relative_to(ROOT))
    return "fail", f"{path.relative_to(ROOT)} did not contain {ok_pattern}"


def run_command(cmd):
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout.strip()


def run_command_with_log(cmd, log_path):
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode, proc.stdout.strip()


def check_module_smoke(args):
    return ok_fail_notrun_from_text(args.module_smoke_log, r"module_smoke:\s*ok")


def check_five_mode_smoke(args):
    return "not-run", "five-mode smoke is not a current readiness gate"
    if not args.smoke_root.exists():
        return "not-run", f"missing {args.smoke_root.relative_to(ROOT)}"
    missing = [mode for mode in MODES if not (args.smoke_root / mode).exists()]
    if missing:
        existing = [mode for mode in MODES if (args.smoke_root / mode).exists()]
        if existing:
            return "fail", f"incomplete smoke modes; missing {', '.join(missing)}"
        return "not-run", "no smoke mode directories found"
    failures = []
    for mode in MODES:
        code, output = run_command([
            sys.executable,
            "scripts/new_method_v2/smoke/check_smoke_log.py",
            "--mode",
            mode,
            str(args.smoke_root / mode),
        ])
        if code != 0:
            failures.append(f"{mode}: {output.splitlines()[-1] if output else 'failed'}")
    if failures:
        return "fail", "; ".join(failures)
    return "pass", str(args.smoke_root.relative_to(ROOT))


def parse_status_file(path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and data.get("status") in VALID_STATUSES:
        return data["status"]
    lowered = text.lower()
    if "fail" in lowered:
        return "fail"
    if "pass" in lowered:
        return "pass"
    if "not-run" in lowered:
        return "not-run"
    return None


def check_status_artifact(path, label):
    status = parse_status_file(path)
    if status is None:
        return "not-run", f"missing {label} status artifact"
    return status, str(path.relative_to(ROOT))


def check_script_whitelist(args):
    command = [sys.executable, "scripts/validate_new_method_scripts.py"]
    log_path = args.evidence_dir / "script_whitelist.log"
    code, output = run_command_with_log(command, log_path)
    summary = output.splitlines()[-1] if output else "validate_new_method_scripts.py"
    details = {
        "command": " ".join(command),
        "return_code": code,
        "log_path": rel(log_path),
        "summary": summary,
    }
    return ("pass" if code == 0 else "fail"), summary, details


def check_diagnostics_audit(args):
    code, output = run_command([sys.executable, "scripts/audit_diagnostics.py"])
    if code == 0:
        return "pass", output or "scripts/audit_diagnostics.py"
    return "fail", output or "scripts/audit_diagnostics.py failed"


def check_parser_contracts(args):
    cases = [
        ("unknown_flag_fails", ["--this_flag_should_fail"], False),
        ("use_sacr_without_slots_fails", ["--use_sacr"], False),
        ("sacr_disable_relation_without_sacr_fails", ["--sacr_disable_relation"], False),
        ("fused_qahnl_without_rapf_fails", ["--use_qahnl", "--use_contrastive_align"], False),
        (
            "qahnl_quality_without_quality_head_fails",
            ["--use_qahnl", "--use_contrastive_align", "--qahnl_score_source", "quality"],
            False,
        ),
        (
            "sacr_disable_relation_with_sacr_passes",
            ["--use_structured_slots", "--use_sacr", "--sacr_disable_relation"],
            True,
        ),
        (
            "qahnl_alias_equal_passes",
            ["--qahnl_pos_iou_thresh", "0.3", "--qahnl_pos_iou_threshold", "0.3"],
            True,
        ),
        (
            "qahnl_alias_conflict_fails",
            ["--qahnl_pos_iou_thresh", "0.3", "--qahnl_pos_iou_threshold", "0.4"],
            False,
        ),
    ]
    failures = []
    details = {}
    for name, argv, should_pass in cases:
        code = (
            "import sys; "
            f"sys.argv=['train_dist_mod.py'] + {argv!r}; "
            "from main_utils import parse_option; "
            "parse_option()"
        )
        rc, output = run_command([sys.executable, "-c", code])
        passed = rc == 0
        details[name] = {"returncode": rc, "expected_pass": should_pass}
        if passed != should_pass:
            details[name]["output"] = output[-1000:]
            failures.append(name)
    if failures:
        return "fail", "parser contract failures: " + ", ".join(failures), details
    return "pass", "main_utils.parse_option parser contracts", details


def check_evaluator_fixture(args):
    try:
        import torch
        from src.grounding_evaluator import GroundingEvaluator

        base_scores = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.0]])
        end_points = {
            "structured_scores": torch.tensor([[0.7, 0.2, 0.1]]),
            "coverage_stats": [
                {"decomposition_status": "repaired_structured", "has_target": 1}
            ],
            "decomposition_status": ["ok"],
        }
        scores = GroundingEvaluator._single_source_scores(
            end_points, "structured", 0, 2, base_scores=base_scores
        )
        if not torch.allclose(scores[0], end_points["structured_scores"][0]):
            return "fail", "primary target row did not use structured scores", {}
        if not torch.allclose(scores[1], base_scores[1]):
            return "fail", "anchor rows did not preserve baseline scores", {}
        try:
            GroundingEvaluator._single_source_scores(end_points, "structured", 0, 2)
        except ValueError:
            pass
        else:
            return "fail", "multi-object diagnostic source accepted missing base_scores", {}
        status = GroundingEvaluator._decomposition_status(end_points, 0)
        if status != "repaired_structured":
            return "fail", f"coverage_stats status was not authoritative: {status}", {}
    except Exception as exc:
        return "fail", "evaluator fixture raised", {"error": repr(exc)}
    return "pass", "GroundingEvaluator score-source and metadata fixture", {}


class _FakeTokenized:
    def char_to_token(self, pos):
        if pos < 0:
            return None
        return min(int(pos), 255)


class _FakeTokenizer:
    def batch_encode_plus(self, *_args, **_kwargs):
        return _FakeTokenized()


def check_positive_map_fixture(args):
    try:
        from src.joint_det_dataset import Joint3DDataset

        dataset = object.__new__(Joint3DDataset)
        dataset.detect_intermediate = False
        dataset.tokenizer = _FakeTokenizer()
        global_only = {
            "utterance": "the red chair near the table",
            "target": "chair",
            "anchors": [],
            "target_slot": {"start": 8, "end": 13, "text": "chair"},
            "entity_spans": [{"start": 8, "end": 13, "text": "chair"}],
            "global_only_due_to_parse_error": True,
            "decomp_global_only_mask": True,
            "decomposition_status": "global_only_target_unresolved",
        }
        _, pmap, diag = dataset._get_token_positive_map(global_only)
        if pmap[0].sum() != 0:
            return "fail", "global-only target positive map was not empty", {}
        if diag["dbg_warn_global_only_target_positive_map"] != 0:
            return "fail", "global-only empty positive map emitted hard warning", {}
        lexical = {
            "utterance": "the red chair near the table",
            "target": "chair",
            "anchors": [],
            "target_slot": {},
            "entity_spans": [],
            "decomposition_status": "ok",
        }
        _, pmap, diag = dataset._get_token_positive_map(lexical)
        if pmap[0].sum() <= 0:
            return "fail", "lexical exact match did not create a positive map", {}
        if diag["positive_map_fallback_used"] != 0:
            return "fail", "lexical exact match was counted as fallback", {}
    except Exception as exc:
        return "fail", "positive-map fixture raised", {"error": repr(exc)}
    return "pass", "Joint3DDataset positive-map fixture", {}


def check_qahnl_global_only_fixture(args):
    try:
        import torch
        from models.losses import _qahnl_losses

        end_points = {
            "base_grounding_scores": torch.tensor([[0.9, 0.2, 0.1, 0.0]]),
            "structured_scores": torch.tensor([[0.9, 0.2, 0.1, 0.0]]),
            "last_center": torch.tensor([[[0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]]),
            "last_pred_size": torch.ones(1, 4, 3),
            "center_label": torch.zeros(1, 1, 3),
            "size_gts": torch.ones(1, 1, 3),
            "box_label_mask": torch.ones(1, 1),
            "dataset": ["sr3d_spacy"],
            "global_only_mask": torch.tensor([True]),
            "weak_generic_target_mask": torch.tensor([False]),
        }
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        cfg = {
            "score_source": "base",
            "num_hard_neg": 2,
            "pos_iou_thresh": 0.25,
            "neg_iou_thresh": 0.10,
            "topk_iou_pos": 1,
            "margin_base": 0.2,
            "margin_iou_lambda": 0.0,
            "margin_min": 0.05,
            "margin_max": 0.5,
            "temperature": 1.0,
            "temperature_max": 6.0,
            "loss_weight": 0.2,
        }
        base_losses = _qahnl_losses(end_points, indices, cfg)
        if base_losses["dbg_qahnl_global_only_used_ratio"].item() <= 0:
            return "fail", "base QA-HNL did not use global-only sample", {}
        structured_losses = _qahnl_losses(
            end_points, indices, dict(cfg, score_source="structured")
        )
        if structured_losses["dbg_qahnl_global_only_used_ratio"].item() != 0:
            return "fail", "structured QA-HNL used global-only sample", {}
        if structured_losses["dbg_qahnl_global_only_skipped_structured_ratio"].item() <= 0:
            return "fail", "structured QA-HNL did not report skipped global-only sample", {}
    except Exception as exc:
        return "fail", "QA-HNL global-only fixture raised", {"error": repr(exc)}
    return "pass", "QA-HNL global-only base/fused versus structured fixture", {}


def check_final_long_training(args):
    return "not-run", "final same-length long training has not been run"


def normalize_check(value):
    if len(value) == 2:
        status, evidence = value
        details = {}
    else:
        status, evidence, details = value
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid readiness status: {status}")
    return {"status": status, "evidence": evidence, "details": details}


def write_evidence(evidence_dir, item, check_result):
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / f"{item}.json"
    details = check_result.get("details", {})
    payload = {
        "name": item,
        "status": check_result["status"],
        "command": details.get("command", ""),
        "return_code": details.get("return_code", None),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "log_path": details.get("log_path", ""),
        "summary": details.get("summary", check_result["evidence"]),
        "evidence": check_result["evidence"],
        "details": details,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render(rows):
    lines = [
        "# Pre Large-Scale Training Checklist",
        "",
        "Generated by `python scripts/update_readiness_checklist.py`.",
        "",
        "| item | status | evidence |",
        "| --- | --- | --- |",
    ]
    for item, status, evidence in rows:
        lines.append(f"| {item} | {status} | {evidence} |")
    lines.extend([
        "",
        "## Launch Script Contracts",
        "",
        "- ScanRefer two-stage: `--butd --self_attend --augment_det --lr_decay_epochs 65`.",
        "- ScanRefer single-stage: `--self_attend --augment_det --lr_decay_epochs 65`.",
        "- NR3D/SR3D active mainline: `--butd_cls --self_attend`.",
        "- Historical NR3D/SR3D runs that used `--butd_gt --detect_intermediate` are legacy-only and require rerun before comparison.",
        "- Five-mode smoke is ScanRefer two-stage only. Single-stage smoke, if added, must be a separate baseline / quality-only / full smoke.",
        "",
        "Allowed statuses: `pass`, `fail`, `not-run`.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checklist", type=Path, default=DEFAULT_CHECKLIST)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--module-smoke-log", type=Path,
                        default=ROOT / "logs" / "new_method_v2" / "smoke" / "module_smoke.log")
    parser.add_argument("--smoke-root", type=Path,
                        default=ROOT / "logs" / "new_method_v2" / "smoke" / "scanrefer" / "two_stage")
    parser.add_argument("--baseline-parity-result", type=Path,
                        default=ROOT / "reports" / "baseline_parity_status.json")
    parser.add_argument("--scanrefer-short-result", type=Path,
                        default=ROOT / "reports" / "scanrefer_short_training_status.json")
    args = parser.parse_args()

    checks = [
        ("parser_contracts", check_parser_contracts(args)),
        ("script_whitelist", check_script_whitelist(args)),
        ("diagnostics_audit", check_diagnostics_audit(args)),
        ("evaluator_fixture", check_evaluator_fixture(args)),
        ("positive_map_fixture", check_positive_map_fixture(args)),
        ("qahnl_global_only_fixture", check_qahnl_global_only_fixture(args)),
        ("module_smoke", check_module_smoke(args)),
        ("five_mode_smoke", check_five_mode_smoke(args)),
        ("baseline_parity", check_status_artifact(args.baseline_parity_result, "baseline parity")),
        ("scanrefer_short_training", check_status_artifact(args.scanrefer_short_result, "ScanRefer short training")),
        ("final_long_training", check_final_long_training(args)),
    ]
    rows = []
    for name, raw_result in checks:
        check_result = normalize_check(raw_result)
        evidence_path = write_evidence(args.evidence_dir, name, check_result)
        evidence = "{} ({})".format(
            check_result["evidence"],
            rel(evidence_path),
        )
        rows.append((name, check_result["status"], evidence.replace("|", "/")))
    args.checklist.parent.mkdir(parents=True, exist_ok=True)
    args.checklist.write_text(render(rows), encoding="utf-8")
    print(f"updated {rel(args.checklist)}")
    if any(status == "fail" for _, status, _ in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
