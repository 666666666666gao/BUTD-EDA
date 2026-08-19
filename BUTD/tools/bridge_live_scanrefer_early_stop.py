#!/usr/bin/env python3
"""Bridge validation-based early stopping into one already-running legacy row.

The full ScanRefer ablation was launched before native early stopping existed.
This controller observes only completed validation logs, applies the locked
policy, gracefully terminates that exact process tree after saturation,
reloads/evaluates the strict-best checkpoint, signs the terminal receipt, and
resumes the queue. Future rows use native early stopping in main_utils.py.
"""

import csv
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/gb/new butd/butd_detr-main")
from main_utils import ValidationEarlyStopper


REPO = Path("/home/gb/new butd/butd_detr-main")
JOB_ID = "02_full_sacr_rapf_qahnl"
RUN_DIR = (
    REPO / "logs/butd_universal_target"
    / "scanrefer_ablation_retrain_20260814_v2_from_official_init"
    / JOB_ID / "scanrefer_spacy" / "1786750440"
)
JOB_LOG_ROOT = RUN_DIR.parents[2] / JOB_ID
QUEUE = (
    REPO / "logs/butd_universal_target"
    / "scanrefer_ablation_retrain_20260814_v2_queue"
)
STATUS = QUEUE / "status"
SUMMARY = QUEUE / "summary.tsv"
PRIMARY = "last__bbs_acc0.25_top1"
QUEUE_SCREEN = "scanrefer_ablation_retrain_20260814_v2"
WATCHDOG_SCREEN = "scanrefer_ablation_watchdog_20260814"
BRIDGE_PROGRESS = QUEUE / "EXPECTED_EARLY_STOP_BRIDGE_IN_PROGRESS"
BRIDGE_COMPLETE = QUEUE / "EXPECTED_EARLY_STOP_BRIDGE_COMPLETED"
POLL_SECONDS = 300


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def atomic_json(path, payload):
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_eval(path):
    values = {}
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"^([^:]+):\s+([-+0-9.eE]+)\s*$", line)
        if match:
            try:
                values[match.group(1)] = float(match.group(2))
            except ValueError:
                pass
    if PRIMARY not in values:
        raise RuntimeError("missing {} in {}".format(PRIMARY, path))
    return values


def validation_history():
    pattern = re.compile(r"^eval_epoch_(\d+)\.log$")
    rows = []
    for path in RUN_DIR.glob("eval_epoch_*.log"):
        match = pattern.match(path.name)
        if match:
            rows.append((int(match.group(1)), parse_eval(path)))
    rows.sort()
    return rows


def build_stopper(rows):
    stopper = ValidationEarlyStopper(
        metric=PRIMARY,
        min_epoch=35,
        patience=4,
        min_delta=0.001,
        max_epoch=100,
        val_freq=5,
    )
    event = None
    for epoch, values in rows:
        event = stopper.update(epoch, values)
    return stopper, event


def screen_exists(name):
    result = subprocess.run(
        ["screen", "-ls"], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, universal_newlines=True,
    )
    return ("." + name) in result.stdout


def stop_screen(name):
    if screen_exists(name):
        subprocess.run(["screen", "-S", name, "-X", "quit"], check=False)


def process_cmdlines():
    found = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
            cmd = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        found[int(proc.name)] = cmd
    return found


def exact_training_pids():
    marker = str(JOB_LOG_ROOT)
    matches = {}
    for pid, cmd in process_cmdlines().items():
        if "train_dist_mod.py" in cmd and "--log_dir" in cmd and marker in cmd:
            matches[pid] = cmd
    return matches


def terminate_exact_training():
    targets = exact_training_pids()
    if not targets:
        raise RuntimeError("no exact training processes found for {}".format(JOB_LOG_ROOT))
    print("{} TERMINATING_EXACT_PIDS {}".format(now(), sorted(targets)), flush=True)
    for pid in sorted(targets, reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 90
    while time.time() < deadline:
        remaining = exact_training_pids()
        if not remaining:
            return
        time.sleep(2)
    remaining = exact_training_pids()
    print("{} FORCE_TERMINATING_EXACT_PIDS {}".format(now(), sorted(remaining)), flush=True)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(5)
    if exact_training_pids():
        raise RuntimeError("exact training processes survived termination")


def wait_queue_exit():
    deadline = time.time() + 180
    while time.time() < deadline and screen_exists(QUEUE_SCREEN):
        time.sleep(2)
    if screen_exists(QUEUE_SCREEN):
        # The trainer is gone; terminate only the exact now-failed queue shell.
        stop_screen(QUEUE_SCREEN)
        time.sleep(3)
    if screen_exists(QUEUE_SCREEN):
        raise RuntimeError("legacy queue screen did not exit")


def start_placeholder():
    if screen_exists(QUEUE_SCREEN):
        raise RuntimeError("cannot start placeholder while queue screen exists")
    command = (
        "cd '{}' && while [ -f '{}' ]; do sleep 30; done"
        .format(REPO, BRIDGE_PROGRESS)
    )
    subprocess.run(
        ["screen", "-dmS", QUEUE_SCREEN, "bash", "-lc", command],
        check=True,
    )
    if not screen_exists(QUEUE_SCREEN):
        raise RuntimeError("placeholder queue screen failed to start")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_reload_eval(best_checkpoint, best_score):
    output_root = QUEUE / "external_early_stop_best_reload"
    output_root.mkdir(parents=True, exist_ok=True)
    result_json = output_root / "eval_results.json"
    if result_json.exists():
        result_json.rename(
            output_root / ("eval_results.previous." + str(int(time.time())) + ".json")
        )
    env = dict(os.environ)
    env.update({
        "PATH": "/root/miniconda3/envs/bdetr/bin:" + env.get("PATH", ""),
        "ABLATION_LOG_ROOT": str(output_root / "runs"),
        "NMV2_MAX_EPOCH": "100",
        "NMV2_VAL_FREQ": "5",
        "NMV2_BATCH_SIZE": "24",
        "NMV2_EARLY_STOP_MIN_EPOCH": "35",
        "NMV2_EARLY_STOP_PATIENCE": "4",
        "NMV2_EARLY_STOP_MIN_DELTA": "0.001",
        "MASTER_PORT": "29777",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    cmd = [
        "bash",
        "scripts/ablations/scanrefer_20260814/02_full_scanrefer_20260814.sh",
        "--eval",
        "--checkpoint_path", str(best_checkpoint),
        "--eval_results_json_path", str(result_json),
        "--print_freq", "1000",
    ]
    print("{} FINAL_RELOAD_EVAL_START {}".format(now(), " ".join(cmd)), flush=True)
    subprocess.run(cmd, cwd=str(REPO), env=env, check=True)
    values = json.loads(result_json.read_text())
    score = float(values[PRIMARY])
    if abs(score - float(best_score)) > 5e-8:
        raise RuntimeError(
            "best reload metric differs: {} vs {}".format(score, best_score)
        )
    lines = [
        "==============================================",
        "Final Evaluation Results - Selected Strict-Best Checkpoint",
        "==============================================",
        "",
    ]
    for key in sorted(values):
        value = values[key]
        if isinstance(value, (float, int)):
            lines.append("{}: {:.10f}".format(key, float(value)))
        else:
            lines.append("{}: {}".format(key, value))
    lines.extend(["", "==============================================", ""])
    atomic_text(RUN_DIR / "eval_epoch_last.log", "\n".join(lines))
    print("{} FINAL_RELOAD_EVAL_PASS score={:.10f}".format(now(), score), flush=True)
    return values


def remove_pilot_from_summary():
    fields = [
        "job_id", "status", "start_time", "end_time", "run_dir",
        "metric", "best_checkpoint", "sha256",
    ]
    rows = []
    if SUMMARY.exists():
        with SUMMARY.open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    rows = [row for row in rows if row.get("job_id") != JOB_ID]
    tmp = SUMMARY.with_name(SUMMARY.name + ".tmp.{}".format(os.getpid()))
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(tmp), str(SUMMARY))


def archive_failed_marker():
    failed = STATUS / (JOB_ID + ".failed")
    if failed.exists():
        destination = QUEUE / (
            JOB_ID + ".failed.expected_early_stop." + str(int(time.time()))
        )
        os.replace(str(failed), str(destination))


def resume_services():
    if BRIDGE_PROGRESS.exists():
        BRIDGE_PROGRESS.unlink()
    deadline = time.time() + 60
    while time.time() < deadline and screen_exists(QUEUE_SCREEN):
        time.sleep(2)
    if screen_exists(QUEUE_SCREEN):
        stop_screen(QUEUE_SCREEN)
        time.sleep(2)
    queue_cmd = (
        "cd '{}' && exec bash scripts/run_scanrefer_ablation_retrain_queue_20260814_v2.sh "
        ">> '{}' 2>&1"
    ).format(REPO, QUEUE / "early_stop_resume_stdout.log")
    subprocess.run(
        ["screen", "-dmS", QUEUE_SCREEN, "bash", "-lc", queue_cmd],
        check=True,
    )
    watchdog_cmd = (
        "cd '{}' && exec bash scripts/scanrefer_ablation_watchdog_20260814.sh "
        ">> '{}' 2>&1"
    ).format(REPO, QUEUE / "watchdog_early_stop_resume.log")
    stop_screen(WATCHDOG_SCREEN)
    subprocess.run(
        ["screen", "-dmS", WATCHDOG_SCREEN, "bash", "-lc", watchdog_cmd],
        check=True,
    )
    time.sleep(3)
    if not screen_exists(QUEUE_SCREEN) or not screen_exists(WATCHDOG_SCREEN):
        raise RuntimeError("queue/watchdog did not resume")


def bridge_stop(stopper, event, rows):
    if BRIDGE_COMPLETE.exists():
        print("{} BRIDGE_ALREADY_COMPLETED".format(now()), flush=True)
        return
    best_path = RUN_DIR / "best_primary.json"
    best = json.loads(best_path.read_text())
    checkpoint = Path(best["checkpoint"]).resolve()
    if checkpoint != (RUN_DIR / "ckpt_best_primary.pth").resolve():
        raise RuntimeError("best checkpoint path mismatch")
    if not checkpoint.is_file():
        raise RuntimeError("best checkpoint is missing")
    if int(rows[-1][0]) != int(event["epoch"]):
        raise RuntimeError("latest validation/event mismatch")

    payload = stopper.receipt(
        "monitoring", mechanism="external_live_process_bridge"
    )
    payload["bridge_state"] = "saturation_confirmed_waiting_for_best_reload"
    atomic_json(RUN_DIR / "early_stopping.json", payload)
    atomic_text(BRIDGE_PROGRESS, now() + "\n")

    # Stop the old watchdog before the intentional torchrun signal can be
    # mistaken for a training failure.
    stop_screen(WATCHDOG_SCREEN)
    terminate_exact_training()
    wait_queue_exit()
    start_placeholder()

    final_reload_eval(checkpoint, best["score"])
    digest = sha256(checkpoint)
    atomic_text(
        RUN_DIR / "ckpt_best_primary.sha256",
        "{}  {}\n".format(digest, checkpoint.name),
    )

    payload = stopper.receipt(
        "early_stopped",
        stop_epoch=event["epoch"],
        mechanism="external_live_process_bridge",
    )
    payload.update({
        "status": "pilot_closed_for_fixed_schedule_transition",
        "reason": "formal_lr55_60_e65_protocol_frozen",
        "bridge_state": "completed",
        "strict_best_epoch": int(best["epoch"]),
        "strict_best_score": float(best["score"]),
        "strict_best_checkpoint": str(checkpoint),
        "strict_best_checkpoint_sha256": digest,
        "final_reload_evaluation": str(RUN_DIR / "eval_epoch_last.log"),
        "completed_at": now(),
    })
    atomic_json(RUN_DIR / "early_stopping.json", payload)

    # This legacy run is a learning-rate tuning pilot, not a paper row. The
    # user froze a new formal protocol (65 epochs, decays after 55 and 60).
    # Preserve logs/metrics/SHA, remove the now-unused 770 MiB pilot weight,
    # move the pilot outside the formal train root, and let the queue rerun the
    # Full row independently from the official initialization.
    pilot_root = (
        REPO / "reports/tuning/scanrefer_ablation_20260815"
        / "pilot_before_fixed_lr55_60_e65"
    )
    pilot_job = pilot_root / JOB_ID
    if pilot_job.exists():
        raise RuntimeError("pilot archive already exists: {}".format(pilot_job))
    pilot_root.mkdir(parents=True, exist_ok=True)
    best["checkpoint_original"] = best.get("checkpoint")
    best["checkpoint"] = None
    best["checkpoint_removed_after_verified_final_reload"] = True
    best["checkpoint_sha256"] = digest
    atomic_json(RUN_DIR / "best_primary.json", best)
    payload["strict_best_checkpoint_removed_as_nonformal_pilot"] = True
    payload["formal_retrain_protocol"] = {
        "start_from_official_initialization": True,
        "max_epoch": 65,
        "lr_decay_epochs": [55, 60],
        "lr_decay_rate": 0.1,
    }
    atomic_json(RUN_DIR / "early_stopping.json", payload)
    checkpoint.unlink()
    if checkpoint.exists():
        raise RuntimeError("pilot checkpoint cleanup failed")
    pilot_receipt = {
        "status": "PILOT_ARCHIVED_FORMAL_RETRAIN_REQUIRED",
        "job_id": JOB_ID,
        "pilot_run_dir_before_move": str(RUN_DIR),
        "pilot_stop_epoch": int(event["epoch"]),
        "pilot_best_epoch": int(best["epoch"]),
        "pilot_best_score": float(best["score"]),
        "removed_pilot_checkpoint_sha256": digest,
        "formal_protocol": payload["formal_retrain_protocol"],
        "archived_at": now(),
    }
    atomic_json(RUN_DIR / "PILOT_ARCHIVE_RECEIPT.json", pilot_receipt)

    archive_failed_marker()
    remove_pilot_from_summary()
    done = STATUS / (JOB_ID + ".done")
    if done.exists():
        os.replace(str(done), str(QUEUE / (JOB_ID + ".done.pilot_transition")))
    os.replace(str(JOB_LOG_ROOT), str(pilot_job))
    atomic_text(
        BRIDGE_COMPLETE,
        json.dumps(pilot_receipt, indent=2, sort_keys=True) + "\n",
    )
    resume_services()
    completed_archive = QUEUE / "PILOT_LR_TRANSITION_COMPLETED.json"
    os.replace(str(BRIDGE_COMPLETE), str(completed_archive))
    print(
        "{} PILOT_ARCHIVE_AND_FIXED_LR_FORMAL_RETRAIN_PASS "
        "pilot_stop_epoch={} pilot_best_epoch={} pilot_best_score={:.10f}".format(
            now(), event["epoch"], best["epoch"], float(best["score"])
        ),
        flush=True,
    )


def main():
    os.chdir(str(REPO))
    if BRIDGE_COMPLETE.exists():
        print("{} BRIDGE_ALREADY_COMPLETED".format(now()))
        return 0
    required = [
        RUN_DIR / "config.json",
        RUN_DIR / "best_primary.json",
        RUN_DIR / "ckpt_best_primary.pth",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing bridge prerequisites: {}".format(missing))
    while True:
        rows = validation_history()
        stopper, event = build_stopper(rows)
        payload = stopper.receipt(
            "monitoring", mechanism="external_live_process_bridge"
        )
        payload["bridge_state"] = "monitoring_live_legacy_process"
        atomic_json(RUN_DIR / "early_stopping.json", payload)
        if event is not None:
            print(
                "{} MONITOR epoch={} score={:.10f} reference={:.10f}@{} "
                "stale={}/{}".format(
                    now(), event["epoch"], event["score"],
                    event["reference_score"], event["reference_epoch"],
                    event["stale_validations"], stopper.patience,
                ),
                flush=True,
            )
            # The user froze a replacement formal schedule while this legacy
            # pilot was in epoch 54. Close the pilot after its complete epoch-55
            # validation regardless of whether that single point rises, then
            # rerun the formal row from the official initialization.
            if event["epoch"] >= 55:
                bridge_stop(stopper, event, rows)
                return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
