#!/usr/bin/env python
"""Rerun top Optuna RAPF candidates for medium ScanRefer two-stage training.

The default candidate source is the top5 JSON exported by the Optuna short-run.
Candidates are ranked by balanced objective unless explicitly overridden.
"""

from __future__ import print_function

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

from parse_eval_metric import csv_value
from parse_eval_metric import markdown_table
from parse_eval_metric import metric_row
from parse_eval_metric import parse_eval_file
from parse_eval_metric import sort_rows


PARAM_KEYS = [
    "rapf_struct_residual_clip",
    "rapf_quality_weight",
    "rapf_gate_loss_weight",
    "rapf_initial_gate_bias",
    "rapf_generic_gate_cap",
]


def repo_root():
    return Path(__file__).resolve().parents[3]


def default_full_script():
    return repo_root() / "scripts" / "new_method_v2" / "scanrefer" / "two_stage" / "05_full_sacr_rapf_qahnl_scanrefer_2stage.sh"


def bash_path(path):
    path = Path(path)
    try:
        rel = path.resolve().relative_to(repo_root())
        return rel.as_posix()
    except Exception:
        return os.fspath(path).replace("\\", "/")


def shell_join(command):
    try:
        import shlex
        return " ".join(shlex.quote(str(part)) for part in command)
    except Exception:
        return subprocess.list2cmdline([str(part) for part in command])


def load_candidates(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        candidates = payload
    else:
        candidates = payload.get("top5", [])
    return [candidate for candidate in candidates if candidate.get("status") == "completed"]


def rank_value(row, key):
    if key == "balanced_objective":
        key = "objective"
    value = row.get(key)
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return float("-inf")


def select_candidates(candidates, top_k, rank_by):
    ranked = sorted(candidates, key=lambda row: rank_value(row, rank_by), reverse=True)
    return ranked[:top_k]


def run_env(args):
    env = os.environ.copy()
    env.pop("EXTRA_ARGS", None)
    env.pop("USER_ARGS", None)
    env.pop("TUNE_ARGS", None)
    for key in (
        "NMV2_BATCH_SIZE",
        "NMV2_MAX_EPOCH",
        "NMV2_VAL_FREQ",
        "NMV2_SAVE_FREQ",
        "NMV2_PRINT_FREQ",
        "NMV2_LR_DECAY_EPOCHS",
        "NMV2_LOG_ROOT",
        "PP_CHECKPOINT",
    ):
        env.pop(key, None)
    env["DATA_ROOT"] = args.data_root
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    return env


def candidate_log_dir(args, index, candidate):
    trial_number = candidate.get("trial_number", index)
    return Path(args.output_root) / "rank_{:02d}_trial_{}".format(
        index,
        trial_number,
    )


def override_args(args, candidate, index):
    log_dir = candidate_log_dir(args, index, candidate)
    seed = candidate.get("seed", args.seed)
    overrides = [
        "--max_epoch",
        str(args.max_epoch),
        "--save_freq",
        "5",
        "--val_freq",
        "5",
        "--data_root",
        args.data_root,
        "--log_dir",
        os.fspath(log_dir),
        "--rng_seed",
        str(seed),
    ]
    for key in PARAM_KEYS:
        overrides.extend(["--" + key, str(candidate[key])])
    if candidate.get("rapf_quality_anchor_structured_residual", False):
        overrides.append("--rapf_quality_anchor_structured_residual")
    if getattr(args, "resume_from_trial_checkpoint", False):
        checkpoint = candidate.get("checkpoint_path")
        if checkpoint:
            overrides.extend([
                "--checkpoint_path",
                os.fspath(Path(checkpoint).expanduser().resolve()),
            ])
    return overrides


def wrapper_command(args, candidate, index):
    return ["bash", bash_path(args.full_script)] + override_args(args, candidate, index)


def effective_command(args, candidate, index):
    cmd = ["bash", bash_path(args.full_script), "--dry-run"] + override_args(
        args, candidate, index
    )
    proc = subprocess.run(
        cmd,
        cwd=os.fspath(repo_root()),
        env=run_env(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "full-script dry-run failed; cannot verify effective command:\n{}"
            .format(proc.stdout)
        )
    return proc.stdout.strip()


def strict_run_dir(log_root, dataset):
    pattern = os.path.join(os.fspath(log_root), dataset, "*", "config.json")
    configs = glob.glob(pattern)
    if not configs:
        return None, "missing config.json under {}".format(pattern)
    if len(configs) > 1:
        return None, "ambiguous run directory; found {} config.json files under {}".format(
            len(configs),
            os.path.join(os.fspath(log_root), dataset),
        )
    return Path(configs[0]).parent, ""


def find_eval_log(run_dir, max_epoch):
    if run_dir is None:
        return None
    return run_dir / "eval_epoch_{}.log".format(max_epoch)


def find_checkpoint(run_dir, max_epoch):
    if run_dir is None:
        return None
    return run_dir / "ckpt_epoch_{}.pth".format(max_epoch)


def base_record(args, candidate, index, command):
    record = {
        "rank": index,
        "source_trial_number": candidate.get("trial_number"),
        "status": "pending",
        "resume_from_trial_checkpoint": bool(args.resume_from_trial_checkpoint),
        "objective": None,
        "balanced_objective": None,
        "objective_with_disagree_penalty": None,
        "last_bbs_acc025": None,
        "last_bbs_acc050": None,
        "last_bbf_acc025": None,
        "last_bbf_acc050": None,
        "last_bbs_vs_bbf_top1_disagree_ratio": None,
        "log_dir": os.fspath(candidate_log_dir(args, index, candidate)),
        "checkpoint_dir": None,
        "checkpoint_path": None,
        "checkpoint_status": "not_required",
        "config_path": None,
        "eval_log_path": None,
        "command": command,
        "effective_command": command,
        "seed": candidate.get("seed", args.seed),
        "return_code": None,
        "error_message": "",
    }
    for key in PARAM_KEYS:
        record[key] = candidate.get(key)
    record["rapf_quality_anchor_structured_residual"] = bool(
        candidate.get("rapf_quality_anchor_structured_residual", False)
    )
    return record


def source_trial_checkpoint_status(candidate):
    checkpoint = candidate.get("checkpoint_path")
    if not checkpoint:
        return "missing"
    return "exists" if Path(checkpoint).expanduser().resolve().exists() else "missing"


def run_candidate(args, candidate, index):
    if args.resume_from_trial_checkpoint:
        status = source_trial_checkpoint_status(candidate)
        if status != "exists":
            command_text = shell_join(wrapper_command(args, candidate, index))
            record = base_record(args, candidate, index, command_text)
            record["status"] = "failed"
            record["checkpoint_status"] = "missing"
            record["return_code"] = -1
            record["error_message"] = (
                "missing source trial checkpoint for explicit resume; no fallback checkpoint is used"
            )
            if args.dry_run:
                print("[dry-run] rerun candidate rank {} cannot resume: missing trial checkpoint".format(index))
            return record
    try:
        command_text = effective_command(args, candidate, index)
    except RuntimeError as exc:
        command_text = shell_join(wrapper_command(args, candidate, index))
        record = base_record(args, candidate, index, command_text)
        record["status"] = "failed"
        record["return_code"] = -1
        record["checkpoint_status"] = "failed"
        record["error_message"] = str(exc)
        if args.dry_run:
            print("[dry-run] failed to verify candidate rank {}: {}".format(
                index,
                exc,
            ))
        return record
    record = base_record(args, candidate, index, command_text)
    if args.dry_run:
        print("[dry-run] rerun candidate rank {} source trial {}".format(
            index,
            candidate.get("trial_number"),
        ))
        print(command_text)
        record["status"] = "dry_run"
        record["checkpoint_status"] = (
            source_trial_checkpoint_status(candidate)
            if args.resume_from_trial_checkpoint else
            "not_required"
        )
        record["return_code"] = 0
        return record

    cmd = wrapper_command(args, candidate, index)
    proc = subprocess.run(
        cmd,
        cwd=os.fspath(repo_root()),
        env=run_env(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    record["return_code"] = proc.returncode
    stdout_path = candidate_log_dir(args, index, candidate) / "rerun_stdout.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(proc.stdout, encoding="utf-8")

    run_dir, run_dir_error = strict_run_dir(record["log_dir"], "scanrefer_spacy")
    config_path = run_dir / "config.json" if run_dir else None
    eval_log = find_eval_log(run_dir, args.max_epoch)
    checkpoint = find_checkpoint(run_dir, args.max_epoch)
    record["checkpoint_dir"] = os.fspath(run_dir) if run_dir else None
    record["checkpoint_path"] = os.fspath(checkpoint) if checkpoint else None
    record["config_path"] = os.fspath(config_path) if config_path else None
    record["eval_log_path"] = os.fspath(eval_log) if eval_log else None
    if run_dir_error:
        record["checkpoint_status"] = "failed"
    elif checkpoint is not None and Path(checkpoint).exists():
        record["checkpoint_status"] = "exists"
    else:
        record["checkpoint_status"] = "missing"

    errors = []
    if proc.returncode != 0:
        errors.append("return_code={}".format(proc.returncode))
    if run_dir_error:
        errors.append(run_dir_error)
    if eval_log is None or not Path(eval_log).exists():
        errors.append("missing exact eval log for epoch {}".format(args.max_epoch))
    else:
        metrics = parse_eval_file(eval_log)
        values = metric_row(
            metrics,
            weight_025=args.objective_weight_025,
            weight_050=args.objective_weight_050,
        )
        record.update(values)
        record["objective"] = values.get("balanced_objective")
        record["balanced_objective"] = values.get("balanced_objective")
    if record.get("objective") is None:
        errors.append("missing objective metrics")
    if checkpoint is None or not Path(checkpoint).exists():
        errors.append("missing exact checkpoint for epoch {}".format(args.max_epoch))
    if errors:
        record["status"] = "failed"
        record["error_message"] = "; ".join(errors)
    else:
        record["status"] = "completed"
    return record


def output_stem(top_k, max_epoch):
    return "scanrefer_two_stage_top{}_{}epoch".format(top_k, max_epoch)


def fieldnames():
    return [
        "rank",
        "source_trial_number",
        "status",
        "resume_from_trial_checkpoint",
        "objective",
        "balanced_objective",
        "objective_with_disagree_penalty",
        "last_bbs_acc025",
        "last_bbs_acc050",
        "last_bbf_acc025",
        "last_bbf_acc050",
        "last_bbs_vs_bbf_top1_disagree_ratio",
        "rapf_struct_residual_clip",
        "rapf_quality_weight",
        "rapf_gate_loss_weight",
        "rapf_initial_gate_bias",
        "rapf_generic_gate_cap",
        "rapf_quality_anchor_structured_residual",
        "log_dir",
        "checkpoint_dir",
        "checkpoint_path",
        "checkpoint_status",
        "config_path",
        "eval_log_path",
        "command",
        "effective_command",
        "seed",
        "return_code",
        "error_message",
    ]


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames()})


def write_markdown(path, rows, args):
    columns = [
        ("rank", "rank"),
        ("source_trial", "source_trial_number"),
        ("resume_from_trial_checkpoint", "resume_from_trial_checkpoint"),
        ("objective", "objective"),
        ("Acc@0.25", "last_bbs_acc025"),
        ("Acc@0.5", "last_bbs_acc050"),
        ("penalty_obj", "objective_with_disagree_penalty"),
        ("clip", "rapf_struct_residual_clip"),
        ("quality_weight", "rapf_quality_weight"),
        ("gate_loss", "rapf_gate_loss_weight"),
        ("gate_bias", "rapf_initial_gate_bias"),
        ("generic_cap", "rapf_generic_gate_cap"),
        ("quality_anchor", "rapf_quality_anchor_structured_residual"),
        ("checkpoint_status", "checkpoint_status"),
        ("status", "status"),
        ("log_dir", "log_dir"),
    ]
    ranked = sort_rows(
        [row for row in rows if row.get("status") == "completed"],
        "objective",
    )
    lines = [
        "# ScanRefer Two-Stage Top-K Medium Rerun",
        "",
        "Exploratory medium-run report. This is not a final paper result.",
        "",
        "- Source top5 JSON: `{}`".format(args.top5_json),
        "- Requested top-k: {}".format(args.top_k),
        "- Max epoch: {}".format(args.max_epoch),
        "- Resume from trial checkpoint: `{}`".format(bool(args.resume_from_trial_checkpoint)),
        "- Objective: `{:.3g} * Acc@0.25 + {:.3g} * Acc@0.5`".format(
            args.objective_weight_025,
            args.objective_weight_050,
        ),
        "",
        "## Completed Candidates by Balanced Objective",
        "",
        markdown_table(ranked, columns) if ranked else "No completed reruns.",
        "",
        "## All Candidates",
        "",
        markdown_table(rows, columns) if rows else "No candidates.",
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rerun top Optuna RAPF candidates for 25/30 epoch screening."
    )
    parser.add_argument(
        "--top5-json",
        default="reports/tuning/optuna_scanrefer_two_stage_full_top5.json",
    )
    parser.add_argument(
        "--best-json",
        default="reports/tuning/optuna_scanrefer_two_stage_full_best.json",
        help="Accepted for compatibility; top5-json is the default candidate source.",
    )
    parser.add_argument(
        "--trials-csv",
        default="reports/tuning/optuna_scanrefer_two_stage_full_trials.csv",
        help="Accepted for compatibility; top5-json is the default candidate source.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-epoch", type=int, default=30)
    parser.add_argument(
        "--output-root",
        default="logs/new_method_v2/tuning/scanrefer_two_stage_top3_30epoch",
    )
    parser.add_argument("--rank-by", default="balanced_objective")
    parser.add_argument("--data-root", "--data_root", dest="data_root",
                        default="/root/autodl-tmp/DATA_ROOT")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-from-trial-checkpoint",
        action="store_true",
        default=False,
        help="Explicitly resume each rerun from its source Optuna trial checkpoint.",
    )
    parser.add_argument("--report-dir", default="reports/tuning")
    parser.add_argument("--objective-weight-025", type=float, default=0.5)
    parser.add_argument("--objective-weight-050", type=float, default=0.5)
    parser.add_argument("--full-script", default=os.fspath(default_full_script()))
    args = parser.parse_args()
    args.full_script = Path(args.full_script)
    return args


def main():
    args = parse_args()
    if args.top_k < 1 or args.top_k > 5:
        print("ERROR: --top-k must be between 1 and 5 for top5 candidates.", file=sys.stderr)
        return 2
    candidates = load_candidates(args.top5_json)
    selected = select_candidates(candidates, args.top_k, args.rank_by)
    if not selected:
        print("ERROR: no completed candidates found in {}".format(args.top5_json), file=sys.stderr)
        return 2
    rows = []
    for index, candidate in enumerate(selected, start=1):
        rows.append(run_candidate(args, candidate, index))
    if args.dry_run:
        print("[dry-run] planned {} reruns ranked by {}.".format(
            len(rows),
            args.rank_by,
        ))
        return 0
    stem = output_stem(args.top_k, args.max_epoch)
    csv_path = Path(args.report_dir) / (stem + ".csv")
    md_path = Path(args.report_dir) / (stem + ".md")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, args)
    print("Saved rerun CSV: {}".format(csv_path))
    print("Saved rerun Markdown: {}".format(md_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
