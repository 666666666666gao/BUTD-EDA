#!/usr/bin/env python3
"""Fail-closed completion audit for the three ScanRefer extension rows."""

import csv
import hashlib
import json
import os
import re
import subprocess
import time
from io import StringIO
from pathlib import Path


REPO = Path("/home/gb/new butd/butd_detr-main")
QUEUE = REPO / "logs/butd_universal_target/scanrefer_ablation_extension_20260815_queue"
TRAIN_ROOT = REPO / "logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_from_official_init"
INIT = Path("/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth")
INIT_SHA256 = "9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
PRIMARY = "last__bbs_acc0.25_top1"

EXPECTED = {
    "08_sacr_only": {
        "use_structured_slots": True,
        "use_sacr": True,
        "use_rapf": False,
        "use_reliability_gate": False,
        "use_quality_head": False,
        "rapf_use_quality": False,
        "use_qahnl": False,
        "eval_use_structured_scores": True,
        "eval_use_fused_scores": False,
        "sacr_disable_relation": False,
    },
    "09_sacr_qahnl": {
        "use_structured_slots": True,
        "use_sacr": True,
        "use_rapf": False,
        "use_reliability_gate": False,
        "use_quality_head": False,
        "rapf_use_quality": False,
        "use_qahnl": True,
        "qahnl_score_source": "structured",
        "eval_use_structured_scores": True,
        "eval_use_fused_scores": False,
        "sacr_disable_relation": False,
    },
    "10_full_qahnl_base_source": {
        "use_structured_slots": True,
        "use_sacr": True,
        "use_rapf": True,
        "use_reliability_gate": True,
        "use_quality_head": True,
        "rapf_use_quality": True,
        "use_qahnl": True,
        "qahnl_score_source": "base",
        "eval_use_structured_scores": False,
        "eval_use_fused_scores": True,
        "rapf_gate_loss_weight": 0.1,
        "sacr_disable_relation": False,
    },
}

METRICS = (
    "last__bbs_acc0.25_top1", "last__bbs_acc0.50_top1",
    "last__bbs_unique_acc0.25_top1", "last__bbs_unique_acc0.50_top1",
    "last__bbs_multiple_acc0.25_top1", "last__bbs_multiple_acc0.50_top1",
    "last__bbs_unique_count_acc0.25", "last__bbs_unique_count_acc0.50",
    "last__bbs_multiple_count_acc0.25", "last__bbs_multiple_count_acc0.50",
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def close(a, b, tolerance=5e-10):
    return abs(float(a) - float(b)) <= tolerance


def parse_eval(path):
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        if key in METRICS:
            try:
                values[key] = float(raw.strip())
            except ValueError:
                pass
    require(set(METRICS).issubset(values), "incomplete metrics: {}".format(path))
    uc25 = int(round(values["last__bbs_unique_count_acc0.25"]))
    uc50 = int(round(values["last__bbs_unique_count_acc0.50"]))
    mc25 = int(round(values["last__bbs_multiple_count_acc0.25"]))
    mc50 = int(round(values["last__bbs_multiple_count_acc0.50"]))
    require(uc25 == uc50 == 1419, "unexpected Unique count")
    require(mc25 == mc50 == 8089, "unexpected Multiple count")
    return values


def load_summary():
    with (QUEUE / "summary.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result = {}
    for job_id in EXPECTED:
        matches = [row for row in rows if row.get("job_id") == job_id]
        require(len(matches) == 1, "expected one summary row for {}".format(job_id))
        require(matches[0].get("status") == "completed", "{} not completed".format(job_id))
        result[job_id] = matches[0]
    require(len(rows) == len(EXPECTED), "unexpected summary rows")
    return result


def audit_job(job_id, expected, summary):
    require((QUEUE / "status" / (job_id + ".done")).is_file(), "missing done marker")
    require(not (QUEUE / "status" / (job_id + ".failed")).exists(), "failed marker exists")
    receipts = list((TRAIN_ROOT / job_id / "scanrefer_spacy").glob("*/best_primary.json"))
    require(len(receipts) == 1, "{} must have exactly one run".format(job_id))
    receipt_path = receipts[0]
    run_dir = receipt_path.parent.resolve()
    require(str(run_dir) == summary["run_dir"], "run_dir mismatch")
    cfg = json.loads((run_dir / "config.json").read_text())
    common = {
        "checkpoint_path": None,
        "pp_checkpoint": str(INIT),
        "rng_seed": 0,
        "start_epoch": 1,
        "max_epoch": 100,
        "val_freq": 5,
        "batch_size": 24,
        "lr_decay_epochs": [65],
        "best_checkpoint_only": True,
        "best_checkpoint_metric": PRIMARY,
        "best_checkpoint_min_delta": 0.0,
        "dataset": ["scanrefer_spacy"],
        "test_dataset": "scanrefer_spacy",
    }
    for key, wanted in common.items():
        require(cfg.get(key) == wanted, "{} {} mismatch".format(job_id, key))
    for key, wanted in expected.items():
        require(cfg.get(key) == wanted, "{} {}={!r}, expected {!r}".format(job_id, key, cfg.get(key), wanted))

    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("metric") == PRIMARY, "wrong primary")
    require(receipt.get("mode") == "max", "wrong mode")
    require(receipt.get("comparison") == "strict_greater_than", "wrong comparison")
    best_epoch = int(receipt["epoch"])
    checkpoint = Path(receipt["checkpoint"]).resolve()
    require(checkpoint == (run_dir / "ckpt_best_primary.pth").resolve(), "checkpoint path mismatch")
    require(checkpoint.is_file(), "checkpoint missing")
    weights = sorted(
        path.resolve() for path in run_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in (".pth", ".pt", ".ckpt")
    )
    require(weights == [checkpoint], "{} retained weights: {}".format(job_id, weights))
    ckpt_sha = sha256(checkpoint)
    require(ckpt_sha == summary["sha256"], "summary SHA mismatch")
    sidecar = run_dir / "ckpt_best_primary.sha256"
    require(sidecar.is_file(), "checkpoint SHA sidecar missing")
    require(sidecar.read_text().split()[0] == ckpt_sha, "sidecar SHA mismatch")

    validations = []
    pattern = re.compile(r"eval_epoch_(\d+)\.log$")
    for path in run_dir.glob("eval_epoch_*.log"):
        match = pattern.match(path.name)
        if match:
            validations.append((int(match.group(1)), parse_eval(path)))
    require(len(validations) == 20, "{} expected 20 validations".format(job_id))
    require(sorted(epoch for epoch, _ in validations) == list(range(5, 101, 5)), "validation epochs incomplete")
    maximum = max(values[PRIMARY] for _, values in validations)
    winning_epoch = min(epoch for epoch, values in validations if close(values[PRIMARY], maximum))
    require(close(receipt["score"], maximum), "receipt is not strict maximum")
    require(best_epoch == winning_epoch, "best epoch mismatch")

    final_path = run_dir / "eval_epoch_last.log"
    require(final_path.is_file(), "final best reload missing")
    final = parse_eval(final_path)
    require(close(final[PRIMARY], receipt["score"]), "final reload differs from best")
    return {
        "job_id": job_id,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_score": float(receipt["score"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": ckpt_sha,
        "final_eval": str(final_path),
        "final_eval_sha256": sha256(final_path),
        "overall_acc025": final["last__bbs_acc0.25_top1"],
        "overall_acc050": final["last__bbs_acc0.50_top1"],
        "unique_acc025": final["last__bbs_unique_acc0.25_top1"],
        "unique_acc050": final["last__bbs_unique_acc0.50_top1"],
        "multiple_acc025": final["last__bbs_multiple_acc0.25_top1"],
        "multiple_acc050": final["last__bbs_multiple_acc0.50_top1"],
        "unique_count": 1419,
        "multiple_count": 8089,
    }


def main():
    require(not (QUEUE / "WATCHDOG_ALERT").exists(), "watchdog alert exists")
    ids = [line.split("\t", 1)[0] for line in (QUEUE / "manifest.tsv").read_text().splitlines() if line]
    require(ids == list(EXPECTED), "extension manifest differs from locked design")
    subprocess.run(
        ["sha256sum", "-c", str(QUEUE / "code_and_launchers.sha256")],
        cwd=str(REPO), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    require(sha256(INIT) == INIT_SHA256, "official initialization changed")
    require("passed" in (QUEUE / "preflight_pytest.log").read_text(), "preflight did not pass")
    summaries = load_summary()
    rows = [audit_job(job_id, expected, summaries[job_id]) for job_id, expected in EXPECTED.items()]
    require(len({row["run_dir"] for row in rows}) == 3, "run directories are not independent")
    payload = {
        "status": "PASS",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": {
            "dataset": "ScanRefer only",
            "primary_metric": PRIMARY,
            "epochs": "1-100",
            "validation_frequency": 5,
            "seed": 0,
            "independent_retraining": True,
            "resume_checkpoint": None,
            "official_initialization": str(INIT),
            "official_initialization_sha256": INIT_SHA256,
            "best_checkpoint_retained_per_row": True,
            "only_one_weight_file_per_row": True,
            "final_evaluation_reloads_best": True,
        },
        "rows": rows,
    }
    atomic_text(QUEUE / "completion_audit.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fields = list(rows[0].keys())
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    atomic_text(QUEUE / "completion_audit.tsv", buffer.getvalue())
    lines = [
        "# ScanRefer ablation extension completion audit", "", "Overall status: **PASS**", "",
        "All three extension rows independently retrained epochs 1--100 from the verified official initialization. Each retains exactly one strict-best official Overall Acc@0.25 checkpoint and passed final best-checkpoint reload evaluation.", "",
        "| Row | Best epoch | O@0.25 | O@0.50 | U@0.25 | U@0.50 | M@0.25 | M@0.50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {job_id} | {best_epoch} | {overall_acc025:.2%} | {overall_acc050:.2%} | {unique_acc025:.2%} | {unique_acc050:.2%} | {multiple_acc025:.2%} | {multiple_acc050:.2%} |".format(**row))
    atomic_text(QUEUE / "COMPLETION_AUDIT.md", "\n".join(lines) + "\n")
    atomic_text(QUEUE / "COMPLETION_AUDIT_PASS", payload["audited_at"] + "\n")
    print("SCANREFER_ABLATION_EXTENSION_COMPLETION_AUDIT_PASS")


if __name__ == "__main__":
    main()
