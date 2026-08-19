#!/usr/bin/env python
"""Eval-only RAPF/fusion sweep for ScanRefer two-stage full checkpoints.

This is exploratory tuning infrastructure. It does not train a model and it is
not an official ablation entrypoint. Acc@0.25 and Acc@0.5 are both primary
metrics; the default objective weights them equally.
"""

from __future__ import print_function

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from parse_eval_metric import csv_value
from parse_eval_metric import contains_failure
from parse_eval_metric import markdown_table
from parse_eval_metric import metric_row
from parse_eval_metric import sort_rows


CURRENT_CLIP = 2.0
CURRENT_QUALITY_WEIGHT = 0.25
CURRENT_GENERIC_GATE_CAP = 0.35
MISSING_CHECKPOINT_MESSAGE = (
    "Eval-only sweep requires an existing checkpoint. No fallback checkpoint is used."
)


def repo_root():
    return Path(__file__).resolve().parents[3]


def parse_float_list(raw):
    return [float(item) for item in raw.split(",") if item.strip()]


def shell_join(command):
    try:
        import shlex
        return " ".join(shlex.quote(str(part)) for part in command)
    except Exception:
        return subprocess.list2cmdline([str(part) for part in command])


def bool_label(value):
    return "1" if bool(value) else "0"


def normalize_checkpoint_args(args):
    checkpoint = getattr(args, "checkpoint", None)
    checkpoint_path = getattr(args, "checkpoint_path", None)
    requested = checkpoint or checkpoint_path
    if not requested:
        raise SystemExit(MISSING_CHECKPOINT_MESSAGE)

    if checkpoint and checkpoint_path:
        canonical = Path(checkpoint).expanduser().resolve()
        alias = Path(checkpoint_path).expanduser().resolve()
        if canonical != alias:
            raise SystemExit(
                "ERROR: --checkpoint and --checkpoint_path resolve to different paths."
            )

    resolved = Path(requested).expanduser().resolve()
    exists = resolved.exists()
    args.checkpoint = requested
    args.requested_checkpoint_path = os.fspath(requested)
    args.resolved_checkpoint_path = os.fspath(resolved)
    args.checkpoint_exists = bool(exists)
    if not args.dry_run and not exists:
        raise SystemExit(MISSING_CHECKPOINT_MESSAGE)
    return args


def combo_name(clip, quality_weight, generic_gate_cap, quality_anchor):
    def fmt(value):
        return ("{:.2f}".format(float(value))).replace(".", "p")

    return (
        "clip_{clip}_qw_{qw}_cap_{cap}_anchor_{anchor}".format(
            clip=fmt(clip),
            qw=fmt(quality_weight),
            cap=fmt(generic_gate_cap),
            anchor=bool_label(quality_anchor),
        )
    )


def distributed_launch(master_port):
    torchrun = shutil.which("torchrun")
    if torchrun:
        return [torchrun, "--nproc_per_node", "1", "--master_port", str(master_port)]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "1",
        "--master_port",
        str(master_port),
    ]


def build_eval_command(args, combo, combo_log_dir):
    clip, quality_weight, generic_gate_cap, quality_anchor = combo
    cmd = distributed_launch(args.master_port)
    cmd.extend([
        "train_dist_mod.py",
        "--eval",
        "--num_decoder_layers",
        "6",
        "--checkpoint_path",
        args.resolved_checkpoint_path,
        "--dataset",
        args.dataset,
        "--test_dataset",
        args.dataset,
        "--data_root",
        args.data_root,
        "--butd",
        "--self_attend",
        "--augment_det",
        "--batch_size",
        "24",
        "--lr_decay_epochs",
        "65",
        "--joint_det",
        "--use_color",
        "--weight_decay",
        "0.0005",
        "--use_soft_token_loss",
        "--use_contrastive_align",
        "--use_structured_slots",
        "--use_sacr",
        "--use_rapf",
        "--use_reliability_gate",
        "--use_quality_head",
        "--rapf_use_quality",
        "--use_qahnl",
        "--qahnl_score_source",
        "fused",
        "--eval_use_fused_scores",
        "--eval_report_diagnostic_scores",
        "--rapf_struct_residual_clip",
        str(clip),
        "--rapf_quality_weight",
        str(quality_weight),
        "--rapf_generic_gate_cap",
        str(generic_gate_cap),
        "--log_dir",
        os.fspath(combo_log_dir),
    ])
    if quality_anchor:
        cmd.append("--rapf_quality_anchor_structured_residual")
    return cmd


def load_verified(text, checkpoint):
    marker = "=> loading checkpoint"
    if marker not in text:
        return False
    expected = os.fspath(Path(checkpoint).expanduser().resolve()).replace("\\", "/")
    observed = text.replace("\\", "/")
    return expected in observed


def read_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def run_combo(args, combo, index):
    clip, quality_weight, generic_gate_cap, quality_anchor = combo
    output_dir = Path(args.output_dir)
    combo_dir = output_dir / combo_name(
        clip, quality_weight, generic_gate_cap, quality_anchor
    )
    combo_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = combo_dir / "eval_stdout.log"
    command = build_eval_command(args, combo, combo_dir)
    command_text = shell_join(command)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("TORCH_DISTRIBUTED_DEBUG", "INFO")

    base_row = {
        "eval_only_status": "pending",
        "checkpoint": args.requested_checkpoint_path,
        "requested_checkpoint_path": args.requested_checkpoint_path,
        "resolved_checkpoint_path": args.resolved_checkpoint_path,
        "checkpoint_exists": bool(args.checkpoint_exists),
        "rapf_struct_residual_clip": clip,
        "rapf_quality_weight": quality_weight,
        "rapf_generic_gate_cap": generic_gate_cap,
        "rapf_quality_anchor_structured_residual": bool(quality_anchor),
        "rapf_gate_loss_weight_eval_effective": False,
        "rapf_initial_gate_bias_eval_effective": False,
        "log_dir": os.fspath(combo_dir),
        "stdout_log_path": os.fspath(stdout_log),
        "command": command_text,
        "return_code": None,
        "status": "pending",
        "error_message": "",
    }

    if args.dry_run:
        base_row.update({
            "status": "dry_run",
            "eval_only_status": "dry_run",
            "return_code": 0,
        })
        print("[dry-run] combination {}: {}".format(index, combo_name(*combo)))
        print(command_text)
        return base_row

    proc = subprocess.run(
        command,
        cwd=os.fspath(repo_root()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    metrics = {}
    try:
        from parse_eval_metric import parse_eval_text
        metrics = parse_eval_text(proc.stdout)
    except Exception as exc:
        base_row["error_message"] = "metric parse failed: {}".format(exc)
    base_row.update(metric_row(
        metrics,
        weight_025=args.objective_weight_025,
        weight_050=args.objective_weight_050,
    ))
    base_row["return_code"] = proc.returncode

    errors = []
    if proc.returncode != 0:
        errors.append("return_code={}".format(proc.returncode))
    if contains_failure(proc.stdout):
        errors.append("failure marker in stdout")
    if not load_verified(proc.stdout, args.checkpoint):
        errors.append("checkpoint load verification failed")
    if base_row.get("balanced_objective") is None:
        errors.append("missing balanced objective metrics")
    if errors:
        base_row["status"] = "failed"
        base_row["eval_only_status"] = "failed"
        base_row["error_message"] = "; ".join(errors)
    else:
        base_row["status"] = "completed"
        base_row["eval_only_status"] = "completed"
    return base_row


def fieldnames():
    return [
        "eval_only_status",
        "checkpoint",
        "requested_checkpoint_path",
        "resolved_checkpoint_path",
        "checkpoint_exists",
        "rapf_struct_residual_clip",
        "rapf_quality_weight",
        "rapf_generic_gate_cap",
        "rapf_quality_anchor_structured_residual",
        "rapf_gate_loss_weight_eval_effective",
        "rapf_initial_gate_bias_eval_effective",
        "last_bbs_acc025",
        "last_bbs_acc050",
        "last_bbf_acc025",
        "last_bbf_acc050",
        "bbs_minus_bbf_acc025",
        "bbs_minus_bbf_acc050",
        "balanced_objective",
        "objective_with_disagree_penalty",
        "last_bbs_vs_bbf_top1_disagree_ratio",
        "last_bbs_top1_iou",
        "last_bbf_top1_iou",
        "eval_primary_score_source",
        "eval_bbs_score_source",
        "eval_bbf_score_source",
        "warn_025_drop",
        "warn_050_drop",
        "log_dir",
        "stdout_log_path",
        "return_code",
        "status",
        "error_message",
        "command",
    ]


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames()})


def completed(rows):
    return [row for row in rows if row.get("status") == "completed"]


def find_current_row(rows):
    for row in rows:
        if (
            row.get("status") == "completed"
            and float(row.get("rapf_struct_residual_clip", -1.0)) == CURRENT_CLIP
            and float(row.get("rapf_quality_weight", -1.0)) == CURRENT_QUALITY_WEIGHT
            and float(row.get("rapf_generic_gate_cap", -1.0)) == CURRENT_GENERIC_GATE_CAP
            and not bool(row.get("rapf_quality_anchor_structured_residual"))
        ):
            return row
    return None


def add_warnings(rows):
    current = find_current_row(rows)
    if current is None:
        for row in rows:
            row["warn_025_drop"] = False
            row["warn_050_drop"] = False
        return
    current025 = current.get("last_bbs_acc025")
    current050 = current.get("last_bbs_acc050")
    for row in rows:
        acc025 = row.get("last_bbs_acc025")
        acc050 = row.get("last_bbs_acc050")
        row["warn_025_drop"] = (
            acc025 is not None and current025 is not None and acc025 < current025 - 0.01
        )
        row["warn_050_drop"] = (
            acc050 is not None and current050 is not None and acc050 < current050 - 0.01
        )


def write_best_json(path, rows, args):
    ok = completed(rows)
    best = sort_rows(ok, "balanced_objective")[:1]
    payload = {
        "note": "Exploratory eval-only tuning result; not an official paper result.",
        "objective": (
            "{} * last__bbs_acc0.25_top1 + {} * last__bbs_acc0.50_top1".format(
                args.objective_weight_025,
                args.objective_weight_050,
            )
        ),
        "best": best[0] if best else None,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def ranking_section(title, rows, sort_key):
    columns = [
        ("clip", "rapf_struct_residual_clip"),
        ("quality_weight", "rapf_quality_weight"),
        ("generic_cap", "rapf_generic_gate_cap"),
        ("quality_anchor", "rapf_quality_anchor_structured_residual"),
        ("Acc@0.25", "last_bbs_acc025"),
        ("Acc@0.5", "last_bbs_acc050"),
        ("balanced", "balanced_objective"),
        ("penalty_obj", "objective_with_disagree_penalty"),
        ("warn_025_drop", "warn_025_drop"),
        ("warn_050_drop", "warn_050_drop"),
        ("log_dir", "log_dir"),
    ]
    top = sort_rows(completed(rows), sort_key)[:5]
    return "## {}\n\n{}\n".format(title, markdown_table(top, columns))


def write_markdown(path, rows, args):
    ok = completed(rows)
    failed = [row for row in rows if row.get("status") == "failed"]
    lines = [
        "# ScanRefer Two-Stage Eval-Only Fusion Sweep",
        "",
        "Exploratory tuning report. These results are not official ablations or paper results.",
        "",
        "- Objective: `{:.3g} * Acc@0.25 + {:.3g} * Acc@0.5`".format(
            args.objective_weight_025,
            args.objective_weight_050,
        ),
        "- Completed combinations: {}".format(len(ok)),
        "- Failed combinations: {}".format(len(failed)),
        "- Requested checkpoint: `{}`".format(args.requested_checkpoint_path),
        "- Resolved checkpoint: `{}`".format(args.resolved_checkpoint_path),
        "- Checkpoint exists: `{}`".format(args.checkpoint_exists),
        "- Eval-effective parameters: `rapf_struct_residual_clip`, `rapf_quality_weight`, `rapf_generic_gate_cap`, `rapf_quality_anchor_structured_residual`.",
        "- Non eval-effective checkpoint parameters not swept: `rapf_gate_loss_weight`, `rapf_initial_gate_bias`.",
        "",
    ]
    lines.append(ranking_section("Top 5 by Balanced Objective", rows, "balanced_objective"))
    lines.append(ranking_section("Top 5 by Acc@0.25", rows, "last_bbs_acc025"))
    lines.append(ranking_section("Top 5 by Acc@0.5", rows, "last_bbs_acc050"))

    warnings = [
        row for row in ok
        if row.get("warn_025_drop") or row.get("warn_050_drop")
    ]
    lines.append("## Metric Trade-Off Warnings\n")
    if warnings:
        columns = [
            ("clip", "rapf_struct_residual_clip"),
            ("quality_weight", "rapf_quality_weight"),
            ("Acc@0.25", "last_bbs_acc025"),
            ("Acc@0.5", "last_bbs_acc050"),
            ("warn_025_drop", "warn_025_drop"),
            ("warn_050_drop", "warn_050_drop"),
        ]
        lines.append(markdown_table(warnings, columns))
    else:
        lines.append("No completed combination crossed the configured 0.01 drop warning threshold.")
    lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exploratory eval-only RAPF/fusion sweep for ScanRefer two-stage."
    )
    parser.add_argument("--checkpoint", dest="checkpoint", default=None)
    parser.add_argument(
        "--checkpoint_path",
        dest="checkpoint_path",
        default=None,
        help="Backward-compatible alias for --checkpoint.",
    )
    parser.add_argument("--base_log_dir", "--base-log-dir", default=None)
    parser.add_argument("--data_root", "--data-root", dest="data_root", required=True)
    parser.add_argument(
        "--output_dir", "--output-dir", dest="output_dir",
        default="logs/new_method_v2/tuning/eval_fusion_sweep",
    )
    parser.add_argument("--stage", default="two_stage", choices=["two_stage"])
    parser.add_argument("--dataset", default="scanrefer_spacy")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--master-port", type=int, default=29581)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-combinations", type=int, default=None)
    parser.add_argument(
        "--save-csv",
        default="reports/tuning/scanrefer_two_stage_eval_fusion_sweep.csv",
    )
    parser.add_argument(
        "--save-md",
        default="reports/tuning/scanrefer_two_stage_eval_fusion_sweep.md",
    )
    parser.add_argument(
        "--save-json",
        default="reports/tuning/scanrefer_two_stage_eval_fusion_sweep_best.json",
    )
    parser.add_argument("--objective-weight-025", type=float, default=0.5)
    parser.add_argument("--objective-weight-050", type=float, default=0.5)
    parser.add_argument(
        "--rapf-struct-residual-clips",
        default="0.0,0.25,0.5,1.0,2.0",
    )
    parser.add_argument("--rapf-quality-weights", default="0.25,0.5,1.0")
    parser.add_argument("--rapf-generic-gate-caps", default="0.35")
    parser.add_argument(
        "--include-quality-anchor",
        action="store_true",
        help="Also sweep the experimental default-off quality-anchor RAPF variant.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    normalize_checkpoint_args(args)
    if args.dry_run:
        print("dry_run = true")
        print("checkpoint existence not required")
    clips = parse_float_list(args.rapf_struct_residual_clips)
    quality_weights = parse_float_list(args.rapf_quality_weights)
    generic_caps = parse_float_list(args.rapf_generic_gate_caps)
    quality_anchors = [False, True] if args.include_quality_anchor else [False]
    combos = list(itertools.product(
        clips,
        quality_weights,
        generic_caps,
        quality_anchors,
    ))
    if args.max_combinations is not None:
        combos = combos[:args.max_combinations]

    rows = []
    for index, combo in enumerate(combos, start=1):
        row = run_combo(args, combo, index)
        rows.append(row)
        if (
            not args.dry_run
            and row.get("status") == "failed"
            and "checkpoint load verification failed" in row.get("error_message", "")
        ):
            raise SystemExit(
                "ERROR: eval-only sweep could not verify requested checkpoint load: {}".format(
                    row.get("resolved_checkpoint_path")
                )
            )
    add_warnings(rows)

    if args.dry_run:
        print("[dry-run] eval-only sweep planned {} combinations.".format(len(rows)))
        print("[dry-run] gate loss and initial gate bias are not swept for eval-only.")
        return

    write_csv(args.save_csv, rows)
    write_best_json(args.save_json, rows, args)
    write_markdown(args.save_md, rows, args)
    print("Saved CSV: {}".format(args.save_csv))
    print("Saved best JSON: {}".format(args.save_json))
    print("Saved Markdown: {}".format(args.save_md))


if __name__ == "__main__":
    main()
