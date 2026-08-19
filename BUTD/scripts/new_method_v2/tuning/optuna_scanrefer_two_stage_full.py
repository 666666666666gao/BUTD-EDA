#!/usr/bin/env python
"""Optuna short-run search for ScanRefer two-stage full RAPF parameters.

This script is exploratory tuning infrastructure. It searches only RAPF/fusion
parameters and optimizes the balanced Acc@0.25/Acc@0.5 objective.
"""

from __future__ import print_function

import argparse
import csv
import datetime
import glob
import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path
try:
    from urllib.parse import urlsplit, urlunsplit
except ImportError:
    from urlparse import urlsplit, urlunsplit

from parse_eval_metric import csv_value
from parse_eval_metric import markdown_table
from parse_eval_metric import metric_row
from parse_eval_metric import parse_eval_file
from parse_eval_metric import sort_rows


SEARCH_SPACE = {
    "rapf_struct_residual_clip": [0.0, 0.25, 0.5, 1.0],
    "rapf_quality_weight": [0.25, 0.5, 0.75, 1.0],
    "rapf_gate_loss_weight": [0.0, 0.05, 0.1, 0.2],
    "rapf_initial_gate_bias": [-3.0, -2.5, -2.0],
    "rapf_generic_gate_cap": [0.1, 0.2, 0.35],
}

PARAM_KEYS = [
    "rapf_struct_residual_clip",
    "rapf_quality_weight",
    "rapf_gate_loss_weight",
    "rapf_initial_gate_bias",
    "rapf_generic_gate_cap",
]

SQLITE_MULTI_WORKER_ERROR = (
    "SQLite storage is only supported for single-machine debugging. "
    "Use PostgreSQL/MySQL for multi-machine Optuna."
)


def repo_root():
    return Path(__file__).resolve().parents[3]


def default_full_script():
    return repo_root() / "scripts" / "new_method_v2" / "scanrefer" / "two_stage" / "05_full_sacr_rapf_qahnl_scanrefer_2stage.sh"


def shell_join(command):
    try:
        import shlex
        return " ".join(shlex.quote(str(part)) for part in command)
    except Exception:
        return subprocess.list2cmdline([str(part) for part in command])


def bash_path(path):
    path = Path(path)
    try:
        rel = path.resolve().relative_to(repo_root())
        return rel.as_posix()
    except Exception:
        return os.fspath(path).replace("\\", "/")


def storage_backend(storage):
    scheme = urlsplit(storage).scheme.lower()
    if scheme.startswith("sqlite"):
        return "sqlite"
    if scheme.startswith("postgresql") or scheme.startswith("postgres"):
        return "postgresql"
    if scheme.startswith("mysql"):
        return "mysql"
    return scheme


def mask_storage_url(storage):
    parsed = urlsplit(storage)
    if not parsed.password:
        return storage
    netloc = parsed.netloc
    if "@" in netloc:
        userinfo, hostport = netloc.rsplit("@", 1)
        if ":" in userinfo:
            username = userinfo.split(":", 1)[0]
            netloc = "{}:***@{}".format(username, hostport)
        else:
            netloc = "{}:***@{}".format(userinfo, hostport)
    return urlunsplit((
        parsed.scheme,
        netloc,
        parsed.path,
        parsed.query,
        parsed.fragment,
    ))


def validate_storage_policy(args):
    if args.multi_worker and storage_backend(args.storage) == "sqlite":
        raise SystemExit(SQLITE_MULTI_WORKER_ERROR)
    if args.multi_worker and not args.worker_id:
        raise SystemExit("ERROR: --worker-id is required when --multi-worker is enabled.")


def ensure_sqlite_parent(storage):
    prefix = "sqlite:///"
    if storage.startswith(prefix):
        db_path = Path(storage[len(prefix):])
        if db_path.parent:
            db_path.parent.mkdir(parents=True, exist_ok=True)


def verify_full_script_forwarding(script_path):
    text = Path(script_path).read_text(encoding="utf-8", errors="replace")
    has_user_args_array = "USER_ARGS=()" in text and 'for arg in "$@"' in text
    user_args_pos = text.find('"${USER_ARGS[@]}"')
    diag_pos = text.find('"${DIAG_ARGS[@]}"')
    extra_pos = text.find('"${EXTRA_ARGS_ARR[@]}"')
    if not has_user_args_array or user_args_pos < 0:
        return False, "full script does not collect and forward USER_ARGS"
    if diag_pos >= 0 and user_args_pos < diag_pos:
        return False, "USER_ARGS appear before default/diagnostic args"
    if extra_pos >= 0 and user_args_pos > extra_pos:
        return False, "USER_ARGS appear after EXTRA_ARGS; override order is unclear"
    return True, "USER_ARGS are forwarded after default args"


def params_from_trial(trial):
    return {
        name: trial.suggest_categorical(name, values)
        for name, values in SEARCH_SPACE.items()
    }


def dry_params(index):
    params = {}
    for offset, name in enumerate(PARAM_KEYS):
        values = SEARCH_SPACE[name]
        params[name] = values[(index + offset) % len(values)]
    return params


def default_worker_id():
    return socket.gethostname()


def trial_log_dir(args, trial_number):
    if getattr(args, "multi_worker", False):
        return Path(args.output_root) / args.worker_id / "trial_{:04d}".format(
            int(trial_number)
        )
    return Path(args.output_root) / "trial_{:04d}".format(int(trial_number))


def trial_override_args(args, params, trial_number):
    log_dir = trial_log_dir(args, trial_number)
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
        str(args.seed),
        "--rapf_struct_residual_clip",
        str(params["rapf_struct_residual_clip"]),
        "--rapf_quality_weight",
        str(params["rapf_quality_weight"]),
        "--rapf_gate_loss_weight",
        str(params["rapf_gate_loss_weight"]),
        "--rapf_initial_gate_bias",
        str(params["rapf_initial_gate_bias"]),
        "--rapf_generic_gate_cap",
        str(params["rapf_generic_gate_cap"]),
    ]
    if params.get("rapf_quality_anchor_structured_residual", False):
        overrides.append("--rapf_quality_anchor_structured_residual")
    return overrides


def wrapper_command(args, params, trial_number):
    return ["bash", bash_path(args.full_script)] + trial_override_args(
        args, params, trial_number
    )


def run_env(args):
    env = os.environ.copy()
    env.pop("EXTRA_ARGS", None)
    env.pop("USER_ARGS", None)
    env.pop("TUNE_ARGS", None)
    for key in list(env.keys()):
        if key.startswith("NMV2_"):
            env.pop(key, None)
    for key in ("PP_CHECKPOINT",):
        env.pop(key, None)
    env["DATA_ROOT"] = args.data_root
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    python_bin = os.fspath(Path(sys.executable).parent)
    env["PATH"] = python_bin + os.pathsep + env.get("PATH", "")
    return env


def effective_command(args, params, trial_number):
    cmd = [
        "bash",
        bash_path(args.full_script),
        "--dry-run",
    ] + trial_override_args(args, params, trial_number)
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
    return proc.stdout.strip(), proc.stdout


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


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def close_enough(actual, expected):
    try:
        return abs(float(actual) - float(expected)) < 1e-8
    except Exception:
        return actual == expected


def validate_config(config_path, args, params):
    if not config_path or not Path(config_path).exists():
        return False, "missing config.json"
    config = load_json(config_path)
    errors = []
    for key in PARAM_KEYS:
        if not close_enough(config.get(key), params[key]):
            errors.append("{} expected {} got {}".format(key, params[key], config.get(key)))
    if not close_enough(config.get("max_epoch"), args.max_epoch):
        errors.append("max_epoch expected {} got {}".format(args.max_epoch, config.get("max_epoch")))
    if int(config.get("num_decoder_layers", -1)) != 6:
        errors.append("num_decoder_layers changed from 6")
    forbidden = {
        "lr": 1e-4,
        "lr_backbone": 1e-3,
        "batch_size": 24,
    }
    for key, expected in forbidden.items():
        if not close_enough(config.get(key), expected):
            errors.append("{} expected {} got {}".format(key, expected, config.get(key)))
    return not errors, "; ".join(errors)


def code_version_record():
    root = repo_root()
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.fspath(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode == 0:
            return proc.stdout.strip(), "git"
    except Exception:
        pass
    timestamp = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return "no_git; time_utc={}; project_path={}".format(timestamp, root), "no_git"


def conda_env_name():
    return os.environ.get("CONDA_DEFAULT_ENV", "")


def base_record(args, trial_number, params, command_text, effective_command_text=None):
    if effective_command_text is None:
        effective_command_text = command_text
    code_version, code_version_source = code_version_record()
    worker_id = getattr(args, "worker_id", "local")
    gpus = getattr(args, "gpus", "")
    study_name = getattr(args, "study_name", "")
    storage = getattr(args, "storage", "")
    return {
        "trial_number": int(trial_number),
        "rank": None,
        "worker_id": worker_id,
        "hostname": socket.gethostname(),
        "gpu": str(gpus),
        "cuda_visible_devices": str(gpus),
        "study_name": study_name,
        "storage_url_masked": mask_storage_url(storage),
        "params": json.dumps(params, sort_keys=True),
        "code_version": code_version,
        "code_version_source": code_version_source,
        "project_path": os.fspath(repo_root()),
        "python_executable": sys.executable,
        "conda_env": conda_env_name(),
        "status": "pending",
        "objective": None,
        "balanced_objective": None,
        "objective_with_disagree_penalty": None,
        "last_bbs_acc025": None,
        "last_bbs_acc050": None,
        "last_bbf_acc025": None,
        "last_bbf_acc050": None,
        "last_bbs_vs_bbf_top1_disagree_ratio": None,
        "rapf_struct_residual_clip": params["rapf_struct_residual_clip"],
        "rapf_quality_weight": params["rapf_quality_weight"],
        "rapf_gate_loss_weight": params["rapf_gate_loss_weight"],
        "rapf_initial_gate_bias": params["rapf_initial_gate_bias"],
        "rapf_generic_gate_cap": params["rapf_generic_gate_cap"],
        "rapf_quality_anchor_structured_residual": bool(
            params.get("rapf_quality_anchor_structured_residual", False)
        ),
        "log_dir": os.fspath(trial_log_dir(args, trial_number)),
        "checkpoint_dir": None,
        "checkpoint_path": None,
        "checkpoint_status": "not_required",
        "config_path": None,
        "eval_log_path": None,
        "command": command_text,
        "effective_command": effective_command_text,
        "seed": args.seed,
        "return_code": None,
        "error_message": "",
    }


def run_trial_command(args, trial, params):
    try:
        effective, _ = effective_command(args, params, trial.number)
    except RuntimeError as exc:
        record = base_record(
            args,
            trial.number,
            params,
            shell_join(wrapper_command(args, params, trial.number)),
            "",
        )
        record["status"] = "failed"
        record["return_code"] = -1
        record["checkpoint_status"] = "failed"
        record["error_message"] = str(exc)
        safe_set_trial_record(trial, record)
        return record
    record = base_record(
        args,
        trial.number,
        params,
        shell_join(wrapper_command(args, params, trial.number)),
        effective,
    )
    record["status"] = "running"
    safe_set_trial_record(trial, record)

    cmd = wrapper_command(args, params, trial.number)
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
    stdout_path = trial_log_dir(args, trial.number) / "trial_stdout.log"
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

    errors = []
    if proc.returncode != 0:
        errors.append("return_code={}".format(proc.returncode))
    if run_dir_error:
        errors.append(run_dir_error)
    valid_config, config_error = validate_config(config_path, args, params)
    if not valid_config:
        errors.append(config_error)
    if run_dir_error or not valid_config:
        record["checkpoint_status"] = "failed"
    elif checkpoint is not None and Path(checkpoint).exists():
        record["checkpoint_status"] = "exists"
    else:
        record["checkpoint_status"] = "missing"
    if eval_log is None or not Path(eval_log).exists():
        errors.append("missing exact eval log for epoch {}".format(args.max_epoch))
    else:
        metrics = parse_eval_file(eval_log)
        metric_values = metric_row(
            metrics,
            weight_025=args.objective_weight_025,
            weight_050=args.objective_weight_050,
        )
        record.update(metric_values)
        record["objective"] = metric_values.get("balanced_objective")
        record["balanced_objective"] = metric_values.get("balanced_objective")
        if record["objective"] is None:
            errors.append("missing balanced objective metrics")
    if errors:
        record["status"] = "failed"
        record["error_message"] = "; ".join([item for item in errors if item])
    else:
        record["status"] = "completed"
    safe_set_trial_record(trial, record)
    return record


def safe_set_trial_record(trial, record):
    try:
        trial.set_user_attr("record", record)
        return True, ""
    except Exception as exc:
        message = "failed to set trial record: {}".format(exc)
        try:
            local_path = Path(record.get("log_dir", ".")) / "optuna_record_write_error.txt"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(message, encoding="utf-8")
        except Exception:
            pass
        return False, message


def trial_objective_value(record):
    value = record.get("objective")
    if record.get("status") != "completed" or value is None:
        return -1.0
    if isinstance(value, float) and math.isfinite(value):
        return value
    try:
        value = float(value)
    except Exception:
        return -1.0
    return value if math.isfinite(value) else -1.0


def run_optuna_trials(args, study):
    for _ in range(args.n_trials):
        trial = study.ask()
        params = params_from_trial(trial)
        record = run_trial_command(args, trial, params)
        rows = collect_records(study)
        if record not in rows:
            rows.append(record)
        write_all_outputs(args, rows)
        value = trial_objective_value(record)
        try:
            study.tell(trial, value)
        except Exception as exc:
            message = "failed to tell Optuna trial {}: {}".format(
                trial.number,
                exc,
            )
            print("WARNING: {}".format(message), file=sys.stderr)
            record["error_message"] = "; ".join(
                item for item in [record.get("error_message"), message] if item
            )
            rows = collect_records(study)
            if record not in rows:
                rows.append(record)
            write_all_outputs(args, rows)


def fieldnames():
    return [
        "trial_number",
        "rank",
        "worker_id",
        "hostname",
        "gpu",
        "cuda_visible_devices",
        "study_name",
        "storage_url_masked",
        "params",
        "code_version",
        "code_version_source",
        "project_path",
        "python_executable",
        "conda_env",
        "status",
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


def write_csv(path, rows, names=None):
    names = names or fieldnames()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in names})


def successful(rows):
    return [row for row in rows if row.get("status") == "completed"]


def ranked_successful(rows):
    ranked = sort_rows(successful(rows), "objective")
    out = []
    for index, row in enumerate(ranked, start=1):
        copied = dict(row)
        copied["rank"] = index
        out.append(copied)
    return out


def top5_rows(rows):
    return ranked_successful(rows)[:5]


def add_metric_ranks(rows, reference_rows):
    if not rows:
        return rows
    by025 = sort_rows(reference_rows, "last_bbs_acc025")
    by050 = sort_rows(reference_rows, "last_bbs_acc050")
    rank025 = {int(row["trial_number"]): index for index, row in enumerate(by025, 1)}
    rank050 = {int(row["trial_number"]): index for index, row in enumerate(by050, 1)}
    for row in rows:
        trial_number = int(row["trial_number"])
        row["rank_acc025"] = rank025.get(trial_number)
        row["rank_acc050"] = rank050.get(trial_number)
    return rows


def write_json_outputs(args, rows):
    report_dir = Path(args.report_dir)
    ok = successful(rows)
    top5 = add_metric_ranks(top5_rows(rows), ok)
    best = top5[0] if top5 else None
    objective_desc = "{} * last__bbs_acc0.25_top1 + {} * last__bbs_acc0.50_top1".format(
        args.objective_weight_025,
        args.objective_weight_050,
    )
    best_payload = {
        "note": "Exploratory Optuna short-run result; not an official paper result.",
        "objective": objective_desc,
        "study_name": args.study_name,
        "storage_url_masked": mask_storage_url(args.storage),
        "best": best,
    }
    top5_payload = {
        "note": "Top-5 exploratory short-run candidates sorted by balanced objective.",
        "objective": objective_desc,
        "study_name": args.study_name,
        "storage_url_masked": mask_storage_url(args.storage),
        "top5": top5,
    }
    (report_dir / "optuna_scanrefer_two_stage_full_best.json").write_text(
        json.dumps(best_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (report_dir / "optuna_scanrefer_two_stage_full_top5.json").write_text(
        json.dumps(top5_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(
        report_dir / "optuna_scanrefer_two_stage_full_top5.csv",
        top5,
        names=fieldnames() + ["rank_acc025", "rank_acc050"],
    )


def param_distribution(top_rows):
    if not top_rows:
        return "No successful trials available."
    lines = []
    for key in PARAM_KEYS:
        counts = {}
        for row in top_rows:
            value = row.get(key)
            counts[value] = counts.get(value, 0) + 1
        summary = ", ".join(
            "{}: {}".format(csv_value(value), count)
            for value, count in sorted(counts.items(), key=lambda item: str(item[0]))
        )
        lines.append("- `{}`: {}".format(key, summary))
    return "\n".join(lines)


def table_for(rows, columns):
    return markdown_table(rows, columns)


def write_summary(args, rows):
    report_dir = Path(args.report_dir)
    ok = successful(rows)
    failed = [row for row in rows if row.get("status") != "completed"]
    top_balanced = add_metric_ranks(top5_rows(rows), ok)
    top025 = sort_rows(ok, "last_bbs_acc025")[:5]
    top050 = sort_rows(ok, "last_bbs_acc050")[:5]
    basic_columns = [
        ("rank", "rank"),
        ("trial", "trial_number"),
        ("worker", "worker_id"),
        ("objective", "objective"),
        ("Acc@0.25", "last_bbs_acc025"),
        ("Acc@0.5", "last_bbs_acc050"),
        ("clip", "rapf_struct_residual_clip"),
        ("quality_weight", "rapf_quality_weight"),
        ("gate_loss_weight", "rapf_gate_loss_weight"),
        ("gate_bias", "rapf_initial_gate_bias"),
        ("generic_gate_cap", "rapf_generic_gate_cap"),
        ("quality_anchor_enabled", "rapf_quality_anchor_structured_residual"),
        ("checkpoint_status", "checkpoint_status"),
        ("log_dir", "log_dir"),
    ]
    metric_rank_columns = [
        ("rank_balanced", "rank"),
        ("trial", "trial_number"),
        ("worker", "worker_id"),
        ("objective", "objective"),
        ("Acc@0.25", "last_bbs_acc025"),
        ("Acc@0.25 rank", "rank_acc025"),
        ("Acc@0.5", "last_bbs_acc050"),
        ("Acc@0.5 rank", "rank_acc050"),
        ("log_dir", "log_dir"),
    ]
    for index, row in enumerate(top025, start=1):
        row["rank"] = index
    for index, row in enumerate(top050, start=1):
        row["rank"] = index

    lines = [
        "# Optuna ScanRefer Two-Stage Full RAPF Summary",
        "",
        "Exploratory short-run tuning report. These trials are not final paper results.",
        "",
        "- Objective: `{:.3g} * Acc@0.25 + {:.3g} * Acc@0.5`".format(
            args.objective_weight_025,
            args.objective_weight_050,
        ),
        "- Study name: `{}`".format(args.study_name),
        "- Storage: `{}`".format(mask_storage_url(args.storage)),
        "- Successful trials: {}".format(len(ok)),
        "- Failed trials: {}".format(len(failed)),
        "- Recommendation: {}.".format(
            "run top-k 25/30 epoch rerun before any long training"
            if len(ok) >= 3 else
            "run more successful short trials before rerun"
        ),
        "",
        "## Best Trial by Balanced Objective",
        "",
        table_for(top_balanced[:1], basic_columns) if top_balanced else "No successful trials.",
        "",
        "## Top-5 by balanced objective",
        "",
        table_for(top_balanced, basic_columns) if top_balanced else "No successful trials.",
        "",
        "## Top-5 by Acc@0.25",
        "",
        table_for(top025, basic_columns) if top025 else "No successful trials.",
        "",
        "## Top-5 by Acc@0.5",
        "",
        table_for(top050, basic_columns) if top050 else "No successful trials.",
        "",
        "## Top-5 Optuna Parameter Sets",
        "",
        table_for(top_balanced, basic_columns) if top_balanced else "No successful trials.",
        "",
        "## Recommended Top-K Rerun Candidates",
        "",
        table_for(top_balanced[:3], basic_columns) if top_balanced else "No successful trials.",
        "",
        "Default rerun uses top 3 by balanced objective; top 5 are retained for manual review.",
        "",
        "## Acc@0.25 / Acc@0.5 Trade-Off Notes",
        "",
    ]
    lines.append(
        "No threshold-based trade-off labels are computed. Review the metric ranks below."
    )
    lines.append("")
    lines.append(table_for(top_balanced, metric_rank_columns) if top_balanced else "No successful trials.")
    lines.extend([
        "",
        "## Parameter Distribution in Top 5",
        "",
        param_distribution(top_balanced),
        "",
        "## Failed Trials",
        "",
        "Failed trial count: {}".format(len(failed)),
    ])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "optuna_scanrefer_two_stage_full_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def collect_records(study):
    rows = []
    for trial in study.trials:
        record = trial.user_attrs.get("record")
        if record:
            rows.append(record)
    rows.sort(key=lambda row: int(row.get("trial_number", 0)))
    return rows


def write_all_outputs(args, rows):
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "optuna_scanrefer_two_stage_full_trials.csv", rows)
    write_json_outputs(args, rows)
    write_summary(args, rows)


def dry_run(args):
    validate_storage_policy(args)
    ok, message = verify_full_script_forwarding(args.full_script)
    print("[dry-run] forwarding check: {}".format(message))
    if not ok:
        return 2
    print("[dry-run] search space:")
    for key in PARAM_KEYS:
        print("  {} = {}".format(key, SEARCH_SPACE[key]))
    print("[dry-run] objective = {:.3g} * Acc@0.25 + {:.3g} * Acc@0.5".format(
        args.objective_weight_025,
        args.objective_weight_050,
    ))
    print("[dry-run] study_name = {}".format(args.study_name))
    print("[dry-run] storage = {}".format(mask_storage_url(args.storage)))
    print("[dry-run] multi_worker = {}".format(bool(args.multi_worker)))
    print("[dry-run] worker_id = {}".format(args.worker_id))
    for trial_number in range(args.n_trials):
        params = dry_params(trial_number)
        try:
            effective, _ = effective_command(args, params, trial_number)
        except RuntimeError as exc:
            print("[dry-run] ERROR: {}".format(exc), file=sys.stderr)
            return 2
        print("")
        print("[dry-run] trial {} override args appear after full-script defaults:".format(trial_number))
        print(shell_join(trial_override_args(args, params, trial_number)))
        print("[dry-run] final effective command:")
        print(effective)
    print("")
    print("[dry-run] output schemas will be written on real runs:")
    for name in (
        "optuna_scanrefer_two_stage_full_trials.csv",
        "optuna_scanrefer_two_stage_full_best.json",
        "optuna_scanrefer_two_stage_full_top5.json",
        "optuna_scanrefer_two_stage_full_top5.csv",
        "optuna_scanrefer_two_stage_full_summary.md",
    ):
        print("  {}".format(Path(args.report_dir) / name))
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exploratory Optuna short-run search for ScanRefer two-stage full RAPF."
    )
    parser.add_argument("--study-name", default="scanrefer_two_stage_full_rapf")
    parser.add_argument(
        "--storage",
        default="sqlite:///reports/tuning/optuna_scanrefer_two_stage_full.db",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--max-epoch", type=int, default=15)
    parser.add_argument("--data-root", "--data_root", dest="data_root",
                        default="/root/autodl-tmp/DATA_ROOT")
    parser.add_argument(
        "--output-root",
        default="logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna",
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--multi-worker", action="store_true")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--objective-weight-025", type=float, default=0.5)
    parser.add_argument("--objective-weight-050", type=float, default=0.5)
    parser.add_argument("--report-dir", default="reports/tuning")
    parser.add_argument("--full-script", default=os.fspath(default_full_script()))
    args = parser.parse_args()
    args.full_script = Path(args.full_script)
    if not args.worker_id:
        args.worker_id = default_worker_id() if args.multi_worker else "local"
    return args


def main():
    args = parse_args()
    try:
        validate_storage_policy(args)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.dry_run:
        return dry_run(args)

    ok, message = verify_full_script_forwarding(args.full_script)
    print("Forwarding check: {}".format(message))
    if not ok:
        print("ERROR: {}".format(message), file=sys.stderr)
        return 2

    try:
        import optuna
    except ImportError:
        print(
            "ERROR: Optuna is not installed. Install it with `pip install optuna` "
            "or add it to the active environment before running this tuning script.",
            file=sys.stderr,
        )
        return 2

    ensure_sqlite_parent(args.storage)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
    )

    run_optuna_trials(args, study)
    rows = collect_records(study)
    write_all_outputs(args, rows)
    print("Saved Optuna tuning reports under {}".format(args.report_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
