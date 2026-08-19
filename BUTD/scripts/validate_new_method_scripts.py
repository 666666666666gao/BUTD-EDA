#!/usr/bin/env python3
"""Validate active new_method_v2 launch scripts."""

import argparse
import json
import re
import shlex
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = ROOT / "scripts" / "new_method_v2"

IGNORED_PARTS = {
    "__pycache__",
    "archive",
    "backup",
    "obsolete_artifacts",
    "smoke",
    "tuning",
}
BACKUP_SUFFIXES = (".bak", ".backup", ".orig", ".tmp", ".swp", "~")

TOP_ALLOWED = {"run_priority.sh", "run_ablations.sh"}
SR3D_ALLOWED = {
    "01_baseline_sr3d.sh",
    "02_quality_only_sr3d.sh",
    "03_full_sacr_rapf_qahnl_sr3d.sh",
    "run_priority.sh",
}
NR3D_ALLOWED = {
    "01_baseline_nr3d.sh",
    "02_quality_only_nr3d.sh",
    "03_full_sacr_rapf_qahnl_nr3d.sh",
    "run_priority.sh",
}
SCANREFER_ROOT_ALLOWED = {"run_priority.sh", "run_ablations.sh"}
SCANREFER_TWO_STAGE_ALLOWED = {
    "01_baseline_scanrefer_2stage.sh",
    "02_quality_only_scanrefer_2stage.sh",
    "03_sacr_only_scanrefer_2stage.sh",
    "04_rapf_quality_scanrefer_2stage.sh",
    "05_full_sacr_rapf_qahnl_scanrefer_2stage.sh",
    "06_full_no_gate_supervision_scanrefer_2stage.sh",
    "07_full_no_quality_scanrefer_2stage.sh",
    "08_full_no_qahnl_scanrefer_2stage.sh",
    "09_sacr_no_relation_scanrefer_2stage.sh",
    "10_qahnl_base_source_scanrefer_2stage.sh",
    "11_sacr_rank_scanrefer_2stage.sh",
    "12_full_quality_primary_scanrefer_2stage.sh",
    "run_priority.sh",
    "run_ablations.sh",
}
SCANREFER_SINGLE_STAGE_ALLOWED = {
    "01_baseline_scanrefer_1stage.sh",
    "02_quality_only_scanrefer_1stage.sh",
    "05_full_sacr_rapf_qahnl_scanrefer_1stage.sh",
    "run_priority.sh",
}

VALUE_OPTIONS = {
    "--num_decoder_layers",
    "--weight_decay",
    "--data_root",
    "--val_freq",
    "--batch_size",
    "--save_freq",
    "--print_freq",
    "--max_epoch",
    "--lr_backbone",
    "--lr",
    "--dataset",
    "--test_dataset",
    "--log_dir",
    "--lr_decay_epochs",
    "--pp_checkpoint",
    "--rapf_gate_loss_weight",
    "--sacr_rank_loss_weight",
    "--qahnl_score_source",
}

BASE_ARG_ORDER = (
    "--num_decoder_layers",
    "--use_color",
    "--weight_decay",
    "--data_root",
    "--val_freq",
    "--batch_size",
    "--save_freq",
    "--print_freq",
    "--max_epoch",
    "--lr_backbone",
    "--lr",
    "--dataset",
    "--test_dataset",
    "--joint_det",
    "--detect_intermediate",
    "--use_soft_token_loss",
    "--use_contrastive_align",
    "--lr_decay_epochs",
    "--pp_checkpoint",
    "--butd",
    "--butd_gt",
    "--butd_cls",
    "--self_attend",
    "--augment_det",
)
MODULE_ARG_ORDER = (
    "--use_quality_head",
    "--eval_use_quality_scores",
    "--use_structured_slots",
    "--use_sacr",
    "--sacr_disable_relation",
    "--sacr_rank_loss_weight",
    "--use_rapf",
    "--use_reliability_gate",
    "--rapf_gate_loss_weight",
    "--rapf_use_quality",
    "--use_qahnl",
    "--qahnl_score_source",
    "--eval_use_structured_scores",
    "--eval_use_fused_scores",
)


def rel_text(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def is_ignored(path):
    return any(part in IGNORED_PARTS for part in path.parts) or any(
        path.name.endswith(suffix) for suffix in BACKUP_SUFFIXES
    )


def active_shell_scripts():
    for path in sorted(ACTIVE_ROOT.rglob("*.sh")):
        rel = path.relative_to(ACTIVE_ROOT)
        if is_ignored(rel):
            continue
        yield rel, path


def strip_shell_comments(text):
    stripped = []
    for line in text.splitlines():
        in_single = False
        in_double = False
        escaped = False
        cut = len(line)
        for idx, char in enumerate(line):
            if escaped:
                escaped = False
                continue
            if char == "\\" and not in_single:
                escaped = True
                continue
            if char == "'" and not in_double:
                in_single = not in_single
                continue
            if char == '"' and not in_single:
                in_double = not in_double
                continue
            if char == "#" and not in_single and not in_double:
                cut = idx
                break
        stripped.append(line[:cut])
    return "\n".join(stripped)


def shell_tokens(text):
    lexer = shlex.shlex(strip_shell_comments(text), posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def split_shell_words(value):
    if value == "":
        return []
    return shlex.split(value)


def extract_defaults(text):
    defaults = {}
    pattern = re.compile(r"^([A-Z0-9_]+)=\$\{\1:-(.*)\}$")
    for line in strip_shell_comments(text).splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        var, raw_value = match.groups()
        values = split_shell_words(raw_value)
        defaults[var] = values[0] if len(values) == 1 else " ".join(values)
    return defaults


def expand_default(value, defaults):
    expanded = value
    for _ in range(4):
        next_value = expanded
        for var, default in defaults.items():
            next_value = next_value.replace("${" + var + "}", default)
        if next_value == expanded:
            break
        expanded = next_value
    return expanded


def resolve_value_token(token, defaults):
    if token == "${LR_DECAY_EPOCHS_ARR[@]}":
        return defaults.get("NMV2_LR_DECAY_EPOCHS", token).split()
    match = re.fullmatch(r"\$\{([A-Z0-9_]+)\}", token)
    if match:
        var = match.group(1)
        if var in defaults:
            return [expand_default(defaults[var], defaults)]
    return [expand_default(token, defaults)]


def extract_array_tokens(text, name):
    stripped = strip_shell_comments(text)
    lines = stripped.splitlines()
    collected = []
    in_array = False
    for line in lines:
        trimmed = line.strip()
        if not in_array:
            prefix = f"{name}=("
            if trimmed.startswith(prefix):
                rest = trimmed[len(prefix):]
                if rest:
                    collected.append(rest)
                in_array = True
            continue
        if trimmed == ")":
            break
        collected.append(line)
    if not collected:
        return []
    return shlex.split("\n".join(collected), posix=True)


def command_args(cmd_tokens):
    try:
        start = cmd_tokens.index("train_dist_mod.py") + 1
    except ValueError:
        return []
    return cmd_tokens[start:]


def parse_options(args, defaults):
    options = defaultdict(list)
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            i += 1
            continue
        if "=" in token:
            option, value = token.split("=", 1)
            values = resolve_value_token(value, defaults)
        elif token == "--lr_decay_epochs":
            option = token
            values = []
            while i + 1 < len(args) and not args[i + 1].startswith("--"):
                values.extend(resolve_value_token(args[i + 1], defaults))
                i += 1
        elif token in VALUE_OPTIONS:
            option = token
            values = []
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                values = resolve_value_token(args[i + 1], defaults)
                i += 1
        else:
            option = token
            values = []
        options[option].append(tuple(values))
        i += 1
    return options


def canonical_args(options, order):
    canonical = []
    for option in order:
        for values in options.get(option, []):
            canonical.append([option, *values])
    return canonical


def has_option(options, option):
    return option in options


def option_values(options, option):
    values = []
    for item in options.get(option, []):
        values.extend(item)
    return values


def add_failure(failures, per_script, message, rel=None):
    failures.append(message)
    if rel is not None:
        per_script[str(rel)].append(message)


def classify_script(rel, failures, per_script):
    parts = rel.parts
    name = rel.name
    if len(parts) == 1:
        if name not in TOP_ALLOWED:
            add_failure(failures, per_script, f"unexpected top-level active script: {rel}", rel)
        return {"dataset": "all", "stage": "root", "kind": "runner"}

    group = parts[0]
    if group == "sr3d":
        if len(parts) != 2 or name not in SR3D_ALLOWED:
            add_failure(failures, per_script, f"unexpected SR3D active script: {rel}", rel)
        return {
            "dataset": "sr3d",
            "stage": "mainline",
            "kind": "runner" if name == "run_priority.sh" else "launch",
        }
    if group == "nr3d":
        if len(parts) != 2 or name not in NR3D_ALLOWED:
            add_failure(failures, per_script, f"unexpected NR3D active script: {rel}", rel)
        return {
            "dataset": "nr3d",
            "stage": "mainline",
            "kind": "runner" if name == "run_priority.sh" else "launch",
        }
    if group != "scanrefer":
        add_failure(failures, per_script, f"unexpected active script directory: {rel}", rel)
        return {"dataset": group, "stage": "unknown", "kind": "unknown"}

    if len(parts) == 2:
        if re.match(r"0[0-9].*\.sh$", name):
            add_failure(failures, per_script, f"root-level numbered ScanRefer script is active: {rel}", rel)
        if name not in SCANREFER_ROOT_ALLOWED:
            add_failure(failures, per_script, f"unexpected root ScanRefer active script: {rel}", rel)
        return {"dataset": "scanrefer", "stage": "root", "kind": "runner"}

    stage = parts[1]
    if stage == "two_stage":
        if len(parts) != 3 or name not in SCANREFER_TWO_STAGE_ALLOWED:
            add_failure(failures, per_script, f"unexpected ScanRefer two-stage active script: {rel}", rel)
        return {
            "dataset": "scanrefer",
            "stage": "two_stage",
            "kind": "runner" if name.startswith("run_") else "launch",
        }
    if stage == "single_stage":
        if len(parts) != 3 or name not in SCANREFER_SINGLE_STAGE_ALLOWED:
            add_failure(failures, per_script, f"unexpected ScanRefer single-stage active script: {rel}", rel)
        return {
            "dataset": "scanrefer",
            "stage": "single_stage",
            "kind": "runner" if name == "run_priority.sh" else "launch",
        }
    add_failure(failures, per_script, f"unexpected ScanRefer stage directory: {rel}", rel)
    return {"dataset": "scanrefer", "stage": stage, "kind": "unknown"}


def validate_global_tokens(rel, tokens, failures, per_script):
    if rel.name.startswith("common") or re.match(r"block[0-9]", rel.name):
        add_failure(failures, per_script, f"obsolete helper/block script is active: {rel}", rel)
    if rel.name == "run_all_sr3d.sh":
        add_failure(failures, per_script, f"obsolete aggregate runner is active: {rel}", rel)
    for idx, token in enumerate(tokens):
        if token == "source" and idx + 1 < len(tokens) and "common" in tokens[idx + 1]:
            add_failure(failures, per_script, f"{rel}: forbidden source of common helper", rel)
        if token == "run_new_method_v2" or token.endswith("/run_new_method_v2"):
            add_failure(failures, per_script, f"{rel}: forbidden run_new_method_v2 helper call", rel)


def require_options(rel, options, required, failures, per_script):
    for option in required:
        if not has_option(options, option):
            add_failure(failures, per_script, f"{rel}: missing required arg {option}", rel)


def forbid_options(rel, options, forbidden, failures, per_script):
    for option in forbidden:
        if has_option(options, option):
            add_failure(failures, per_script, f"{rel}: forbidden arg {option}", rel)


def require_lr_decay(rel, options, expected, failures, per_script):
    values = option_values(options, "--lr_decay_epochs")
    if values != expected:
        got = " ".join(values) if values else "<missing>"
        add_failure(
            failures,
            per_script,
            f"{rel}: expected --lr_decay_epochs {' '.join(expected)}, got {got}",
            rel,
        )


def validate_scanrefer_launch(rel, options, failures, per_script):
    if rel.parts[1] == "two_stage":
        require_options(rel, options, ("--butd", "--self_attend", "--augment_det"), failures, per_script)
        forbid_options(rel, options, ("--butd_gt", "--butd_cls", "--detect_intermediate"), failures, per_script)
        require_lr_decay(rel, options, ["65"], failures, per_script)
    elif rel.parts[1] == "single_stage":
        require_options(rel, options, ("--self_attend", "--augment_det"), failures, per_script)
        forbid_options(
            rel,
            options,
            ("--butd", "--butd_gt", "--butd_cls", "--detect_intermediate"),
            failures,
            per_script,
        )
        require_lr_decay(rel, options, ["65"], failures, per_script)

    if rel.name == "06_full_no_gate_supervision_scanrefer_2stage.sh":
        require_options(rel, options, ("--use_reliability_gate",), failures, per_script)
        if option_values(options, "--rapf_gate_loss_weight") != ["0"]:
            add_failure(failures, per_script, f"{rel}: must disable only RAPF gate loss with --rapf_gate_loss_weight 0", rel)
    if rel.name == "07_full_no_quality_scanrefer_2stage.sh":
        forbid_options(rel, options, ("--use_quality_head", "--rapf_use_quality"), failures, per_script)
        if option_values(options, "--qahnl_score_source") == ["quality"]:
            add_failure(failures, per_script, f"{rel}: no-quality ablation must not use quality QA-HNL scores", rel)
        require_options(rel, options, ("--use_rapf", "--use_qahnl"), failures, per_script)
    if rel.name == "09_sacr_no_relation_scanrefer_2stage.sh":
        require_options(rel, options, ("--sacr_disable_relation",), failures, per_script)
    if rel.name == "12_full_quality_primary_scanrefer_2stage.sh":
        require_options(
            rel,
            options,
            (
                "--use_structured_slots",
                "--use_sacr",
                "--use_rapf",
                "--use_reliability_gate",
                "--use_quality_head",
                "--rapf_use_quality",
                "--use_qahnl",
                "--eval_use_quality_scores",
            ),
            failures,
            per_script,
        )
        forbid_options(rel, options, ("--eval_use_fused_scores",), failures, per_script)


def validate_referit_launch(rel, options, failures, per_script):
    require_options(rel, options, ("--butd_cls", "--self_attend"), failures, per_script)
    forbid_options(rel, options, ("--butd_gt", "--detect_intermediate"), failures, per_script)


def resolve_bash_target(path, token):
    if token.startswith("${SCRIPT_DIR}/"):
        return (path.parent / token[len("${SCRIPT_DIR}/"):]).resolve()
    if token.startswith("$SCRIPT_DIR/"):
        return (path.parent / token[len("$SCRIPT_DIR/"):]).resolve()
    if token.startswith("scripts/new_method_v2/"):
        return (ROOT / token).resolve()
    return None


def bash_targets(path, text):
    tokens = shell_tokens(text)
    targets = []
    for idx, token in enumerate(tokens[:-1]):
        if token != "bash":
            continue
        target = resolve_bash_target(path, tokens[idx + 1])
        if target is None:
            continue
        try:
            rel = target.relative_to(ACTIVE_ROOT)
        except ValueError:
            continue
        targets.append(rel)
    return targets


def validate_forwarding(script_texts, failures, per_script):
    def targets(rel):
        path = ACTIVE_ROOT / rel
        return bash_targets(path, script_texts.get(str(rel), ""))

    top_ablations = Path("run_ablations.sh")
    scanrefer_ablations = Path("scanrefer/run_ablations.sh")
    two_stage_ablations = Path("scanrefer/two_stage/run_ablations.sh")
    if str(top_ablations) in script_texts and targets(top_ablations) != [scanrefer_ablations]:
        add_failure(
            failures,
            per_script,
            "top-level run_ablations.sh must forward only to scanrefer/run_ablations.sh",
            top_ablations,
        )
    if str(scanrefer_ablations) in script_texts and targets(scanrefer_ablations) != [two_stage_ablations]:
        add_failure(
            failures,
            per_script,
            "scanrefer/run_ablations.sh must forward only to scanrefer/two_stage/run_ablations.sh",
            scanrefer_ablations,
        )
    for rel in (top_ablations, scanrefer_ablations, two_stage_ablations):
        seen = set()
        stack = [rel]
        while stack:
            current = stack.pop()
            if current in seen:
                add_failure(failures, per_script, f"run_ablations forwarding cycle at {current}", current)
                break
            seen.add(current)
            for target in targets(current):
                if target.name == "run_ablations.sh":
                    stack.append(target)

    scanrefer_priority = Path("scanrefer/run_priority.sh")
    allowed_priority = [
        Path("scanrefer/two_stage/run_priority.sh"),
        Path("scanrefer/single_stage/run_priority.sh"),
    ]
    if str(scanrefer_priority) in script_texts and targets(scanrefer_priority) != allowed_priority:
        add_failure(
            failures,
            per_script,
            "scanrefer/run_priority.sh must call only two-stage and single-stage priority runners",
            scanrefer_priority,
        )


def validate_base_consistency(entries, failures, per_script):
    grouped = defaultdict(list)
    for entry in entries:
        if entry["kind"] == "launch":
            grouped[(entry["dataset"], entry["stage"])].append(entry)
    for (dataset, stage), items in grouped.items():
        expected = items[0]["normalized_base_args"]
        for item in items[1:]:
            if item["normalized_base_args"] != expected:
                rel = Path(item["path"]).relative_to("scripts/new_method_v2")
                add_failure(
                    failures,
                    per_script,
                    f"{dataset}/{stage}: normalized base args differ in {rel}",
                    rel,
                )

    for dataset in ("nr3d", "sr3d"):
        items = grouped.get((dataset, "mainline"), [])
        if not items:
            continue
        baseline = next((item for item in items if Path(item["path"]).name.startswith("01_baseline")), None)
        baseline_has_augment = baseline and any(arg[0] == "--augment_det" for arg in baseline["normalized_base_args"])
        for item in items:
            has_augment = any(arg[0] == "--augment_det" for arg in item["normalized_base_args"])
            if bool(has_augment) != bool(baseline_has_augment):
                rel = Path(item["path"]).relative_to("scripts/new_method_v2")
                add_failure(
                    failures,
                    per_script,
                    f"{dataset}: --augment_det must match the active baseline setting in {rel}",
                    rel,
                )


def script_contract(classification):
    dataset = classification["dataset"]
    stage = classification["stage"]
    if classification["kind"] != "launch":
        return [], []
    if dataset == "scanrefer" and stage == "two_stage":
        return (
            ["--butd", "--self_attend", "--augment_det", "--lr_decay_epochs 65"],
            ["--butd_gt", "--butd_cls", "--detect_intermediate"],
        )
    if dataset == "scanrefer" and stage == "single_stage":
        return (
            ["--self_attend", "--augment_det", "--lr_decay_epochs 65"],
            ["--butd", "--butd_gt", "--butd_cls", "--detect_intermediate"],
        )
    if dataset in {"nr3d", "sr3d"}:
        return (
            ["--butd_cls", "--self_attend"],
            ["--butd_gt", "--detect_intermediate"],
        )
    return [], []


def validate_scripts():
    if not ACTIVE_ROOT.exists():
        failure = f"active script root does not exist: {ACTIVE_ROOT}"
        return {"status": "fail", "summary": failure, "failures": [failure], "scripts": []}

    failures = []
    per_script = defaultdict(list)
    entries = []
    script_texts = {}

    for rel, path in active_shell_scripts():
        text = path.read_text(encoding="utf-8", errors="ignore")
        script_texts[str(rel)] = text
        tokens = shell_tokens(text)
        classification = classify_script(rel, failures, per_script)
        validate_global_tokens(rel, tokens, failures, per_script)

        cmd_tokens = extract_array_tokens(text, "CMD")
        args = command_args(cmd_tokens)
        defaults = extract_defaults(text)
        options = parse_options(args, defaults)
        required, forbidden = script_contract(classification)

        if classification["kind"] == "launch":
            if classification["dataset"] == "scanrefer":
                validate_scanrefer_launch(rel, options, failures, per_script)
            elif classification["dataset"] in {"nr3d", "sr3d"}:
                validate_referit_launch(rel, options, failures, per_script)
            if not args:
                add_failure(failures, per_script, f"{rel}: missing train_dist_mod.py CMD array", rel)

        entry = {
            "path": rel_text(path),
            "dataset": classification["dataset"],
            "stage": classification["stage"],
            "kind": classification["kind"],
            "normalized_base_args": canonical_args(options, BASE_ARG_ORDER),
            "module_args": canonical_args(options, MODULE_ARG_ORDER),
            "required_args": required,
            "forbidden_args": forbidden,
            "status": "pass",
        }
        entries.append(entry)

    validate_forwarding(script_texts, failures, per_script)
    validate_base_consistency(entries, failures, per_script)

    for entry in entries:
        rel = str(Path(entry["path"]).relative_to("scripts/new_method_v2"))
        if per_script.get(rel):
            entry["status"] = "fail"
            entry["failures"] = per_script[rel]

    status = "fail" if failures else "pass"
    summary = (
        f"validate_new_method_scripts: ok ({len(entries)} active shell scripts)"
        if status == "pass"
        else f"validate_new_method_scripts: fail ({len(failures)} issues)"
    )
    return {
        "status": status,
        "summary": summary,
        "failures": failures,
        "scripts": entries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable validation details")
    args = parser.parse_args()

    result = validate_scripts()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["status"] == "pass":
        print(result["summary"])
    else:
        print("\n".join(result["failures"]))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
