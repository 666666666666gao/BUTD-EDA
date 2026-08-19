#!/usr/bin/env python3
"""Wait for and audit the seven ScanRefer ablation retrains."""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time
import traceback
from io import StringIO
from pathlib import Path


REPO = Path("/home/gb/new butd/butd_detr-main")
QUEUE = REPO / "logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_queue"
TRAIN_ROOT = REPO / "logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_from_official_init"
INIT = Path("/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth")
INIT_SHA256 = "9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
PRIMARY = "last__bbs_acc0.25_top1"
PAPER_BASELINE = QUEUE / "butd_paper_baseline.json"
PAPER_METRICS = {
    "unique_acc025": 0.842,
    "unique_acc050": 0.663,
    "multiple_acc025": 0.466,
    "multiple_acc050": 0.351,
    "overall_acc025": 0.522,
    "overall_acc050": 0.398,
}

EXPECTED = {
    "01_baseline": {
        "use_structured_slots": False,
        "use_sacr": False,
        "use_rapf": False,
        "use_reliability_gate": False,
        "use_quality_head": False,
        "rapf_use_quality": False,
        "use_qahnl": False,
        "eval_use_fused_scores": False,
        "sacr_disable_relation": False,
    },
    "02_full_sacr_rapf_qahnl": {
        "use_structured_slots": True,
        "use_sacr": True,
        "use_rapf": True,
        "use_reliability_gate": True,
        "use_quality_head": True,
        "rapf_use_quality": True,
        "use_qahnl": True,
        "qahnl_score_source": "fused",
        "eval_use_fused_scores": True,
        "rapf_gate_loss_weight": 0.1,
        "sacr_disable_relation": False,
    },
    "03_no_sacr_rapf_qahnl_base": {
        "use_structured_slots": False,
        "use_sacr": False,
        "use_rapf": False,
        "use_reliability_gate": False,
        "use_quality_head": False,
        "rapf_use_quality": False,
        "use_qahnl": True,
        "qahnl_score_source": "base",
        "eval_use_fused_scores": False,
        "sacr_disable_relation": False,
    },
    "04_no_qahnl": {
        "use_structured_slots": True,
        "use_sacr": True,
        "use_rapf": True,
        "use_reliability_gate": True,
        "use_quality_head": True,
        "rapf_use_quality": True,
        "use_qahnl": False,
        "eval_use_fused_scores": True,
        "rapf_gate_loss_weight": 0.1,
        "sacr_disable_relation": False,
    },
    "05_no_quality": {
        "use_structured_slots": True,
        "use_sacr": True,
        "use_rapf": True,
        "use_reliability_gate": True,
        "use_quality_head": False,
        "rapf_use_quality": False,
        "use_qahnl": True,
        "qahnl_score_source": "fused",
        "eval_use_fused_scores": True,
        "rapf_gate_loss_weight": 0.1,
        "sacr_disable_relation": False,
    },
    "06_no_gate_supervision": {
        "use_structured_slots": True,
        "use_sacr": True,
        "use_rapf": True,
        "use_reliability_gate": True,
        "use_quality_head": True,
        "rapf_use_quality": True,
        "use_qahnl": True,
        "qahnl_score_source": "fused",
        "eval_use_fused_scores": True,
        "rapf_gate_loss_weight": 0.0,
        "sacr_disable_relation": False,
    },
    "07_no_relation": {
        "use_structured_slots": True,
        "use_sacr": True,
        "use_rapf": True,
        "use_reliability_gate": True,
        "use_quality_head": True,
        "rapf_use_quality": True,
        "use_qahnl": True,
        "qahnl_score_source": "fused",
        "eval_use_fused_scores": True,
        "rapf_gate_loss_weight": 0.1,
        "sacr_disable_relation": True,
    },
}

METRICS = (
    "last__bbs_acc0.25_top1",
    "last__bbs_acc0.50_top1",
    "last__bbs_unique_acc0.25_top1",
    "last__bbs_unique_acc0.50_top1",
    "last__bbs_multiple_acc0.25_top1",
    "last__bbs_multiple_acc0.50_top1",
    "last__bbs_unique_count_acc0.25",
    "last__bbs_unique_count_acc0.50",
    "last__bbs_multiple_count_acc0.25",
    "last__bbs_multiple_count_acc0.50",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, value):
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    temporary.write_text(value, encoding="utf-8")
    os.replace(str(temporary), str(path))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


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
    require(set(METRICS).issubset(values), "incomplete official BBS metrics: {}".format(path))
    uc25 = int(round(values["last__bbs_unique_count_acc0.25"]))
    uc50 = int(round(values["last__bbs_unique_count_acc0.50"]))
    mc25 = int(round(values["last__bbs_multiple_count_acc0.25"]))
    mc50 = int(round(values["last__bbs_multiple_count_acc0.50"]))
    require(uc25 == uc50 == 1419, "unexpected Unique count: {}".format(path))
    require(mc25 == mc50 == 8089, "unexpected Multiple count: {}".format(path))
    require(uc25 + mc25 == 9508, "unexpected official ScanRefer total: {}".format(path))
    return values


def check_manifest():
    lines = [line.split("\t", 1)[0] for line in (QUEUE / "manifest.tsv").read_text().splitlines() if line]
    require(lines == list(EXPECTED), "queue manifest/order differs from locked seven-row design")
    subprocess.run(
        ["sha256sum", "-c", str(QUEUE / "code_and_launchers.sha256")],
        cwd=str(REPO), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    require(sha256(INIT) == INIT_SHA256, "official initialization SHA256 changed")
    parity = (QUEUE / "model_init_parity.log").read_text(errors="replace")
    require("MODEL_INIT_PARITY_PASS" in parity, "shared random-initialization parity gate missing")
    preflight = (QUEUE / "preflight_pytest.log").read_text(errors="replace")
    require("8 passed" in preflight, "preflight 8/8 gate missing")


def load_summary():
    with (QUEUE / "summary.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result = {}
    for job_id in EXPECTED:
        matched = [row for row in rows if row["job_id"] == job_id]
        require(len(matched) == 1, "expected one summary row for {}, got {}".format(job_id, len(matched)))
        expected_status = "external_paper" if job_id == "01_baseline" else "completed"
        require(matched[0]["status"] == expected_status, "{} has status {}, expected {}".format(job_id, matched[0]["status"], expected_status))
        result[job_id] = matched[0]
    require(not [r for r in rows if r.get("job_id") not in EXPECTED], "summary contains unexpected rows")
    return result


def audit_job(job_id, expected, summary):
    require((QUEUE / "status" / (job_id + ".done")).is_file(), "missing done marker for {}".format(job_id))
    require(not (QUEUE / "status" / (job_id + ".failed")).exists(), "failed marker exists for {}".format(job_id))
    if job_id == "01_baseline":
        require(PAPER_BASELINE.is_file(), "external BUTD paper baseline is missing")
        payload = json.loads(PAPER_BASELINE.read_text())
        require(payload.get("job_id") == job_id, "paper baseline job id differs")
        require(payload.get("source_type") == "paper", "baseline is not marked as a paper source")
        require(payload.get("source_url") == "https://arxiv.org/abs/2112.08879", "unexpected paper source URL")
        require(payload.get("source_version") == "arXiv:2112.08879v5", "unexpected paper version")
        metrics = payload.get("metrics", {})
        for key, wanted in PAPER_METRICS.items():
            require(key in metrics and close(metrics[key], wanted), "paper baseline {} differs".format(key))
        paper_sha = sha256(PAPER_BASELINE)
        require(summary["run_dir"] == str(PAPER_BASELINE), "paper baseline summary path differs")
        require(summary["sha256"] == paper_sha, "paper baseline summary hash differs")
        require(summary["best_checkpoint"] == "", "paper baseline must not claim a checkpoint")
        require(summary["metric"] == "paper_overall_acc0.25=0.522", "paper baseline summary metric differs")
        return {
            "job_id": job_id,
            "source_type": "paper",
            "source_url": payload["source_url"],
            "protocol_note": payload["protocol_note"],
            "run_dir": str(PAPER_BASELINE),
            "best_epoch": "paper",
            "best_score": PAPER_METRICS["overall_acc025"],
            "checkpoint": "",
            "checkpoint_sha256": "",
            "final_eval": "",
            "final_eval_sha256": paper_sha,
            **PAPER_METRICS,
            "unique_count": None,
            "multiple_count": None,
        }
    receipts = list((TRAIN_ROOT / job_id / "scanrefer_spacy").glob("*/best_primary.json"))
    require(len(receipts) == 1, "expected exactly one valid receipt for {}, got {}".format(job_id, len(receipts)))
    receipt_path = receipts[0]
    run_dir = receipt_path.parent.resolve()
    require(str(run_dir) == summary["run_dir"], "summary run_dir mismatch for {}".format(job_id))
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
        require(cfg.get(key) == wanted, "{} config {}={!r}, expected {!r}".format(job_id, key, cfg.get(key), wanted))
    for key, wanted in expected.items():
        require(cfg.get(key) == wanted, "{} ablation {}={!r}, expected {!r}".format(job_id, key, cfg.get(key), wanted))

    early_path = run_dir / "early_stopping.json"
    require(early_path.is_file(), "missing early-stopping receipt for {}".format(job_id))
    early = json.loads(early_path.read_text())
    early_policy = {
        "enabled": True,
        "metric": PRIMARY,
        "mode": "max",
        "min_epoch": 35,
        "patience_validations": 4,
        "min_delta": 0.001,
        "validation_frequency_epochs": 5,
        "maximum_epoch": 100,
    }
    for key, wanted in early_policy.items():
        require(early.get(key) == wanted, "{} early-stop {} mismatch".format(job_id, key))
    native_policy = {
        "early_stopping": True,
        "early_stopping_metric": PRIMARY,
        "early_stopping_min_epoch": 35,
        "early_stopping_patience": 4,
        "early_stopping_min_delta": 0.001,
    }
    if cfg.get("early_stopping") is True:
        for key, wanted in native_policy.items():
            require(cfg.get(key) == wanted, "{} config {} mismatch".format(job_id, key))
        require(early.get("mechanism") == "native", "{} expected native early stop".format(job_id))
    else:
        require(job_id == "02_full_sacr_rapf_qahnl", "only the already-running full row may use the external bridge")
        require(early.get("mechanism") == "external_live_process_bridge", "wrong external bridge mechanism")

    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("metric") == PRIMARY, "wrong best metric for {}".format(job_id))
    require(receipt.get("mode") == "max", "wrong best mode for {}".format(job_id))
    require(receipt.get("comparison") == "strict_greater_than", "wrong comparison for {}".format(job_id))
    best_epoch = int(receipt["epoch"])
    require(5 <= best_epoch <= 100 and best_epoch % 5 == 0, "invalid best epoch for {}".format(job_id))
    checkpoint = Path(receipt["checkpoint"]).resolve()
    require(checkpoint == (run_dir / "ckpt_best_primary.pth").resolve(), "checkpoint path mismatch for {}".format(job_id))
    require(checkpoint.is_file(), "missing retained best checkpoint for {}".format(job_id))
    retained_weights = sorted(
        path.resolve()
        for path in run_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in (".pth", ".pt", ".ckpt")
    )
    require(
        retained_weights == [checkpoint],
        "{} must retain only ckpt_best_primary.pth, found {}".format(
            job_id, [str(path) for path in retained_weights]
        ),
    )
    checkpoint_sha = sha256(checkpoint)
    require(checkpoint_sha == summary["sha256"], "summary checkpoint hash mismatch for {}".format(job_id))
    sidecar = (run_dir / "ckpt_best_primary.sha256").read_text().split()[0]
    require(sidecar == checkpoint_sha, "sidecar checkpoint hash mismatch for {}".format(job_id))

    validations = []
    numeric_pattern = re.compile(r"eval_epoch_(\d+)\.log$")
    for path in run_dir.glob("eval_epoch_*.log"):
        match = numeric_pattern.match(path.name)
        if match:
            validations.append((int(match.group(1)), parse_eval(path)))
    require(early.get("status") in ("early_stopped", "max_epoch"), "invalid early-stop status for {}".format(job_id))
    stop_epoch = int(early["stop_epoch"])
    require(35 <= stop_epoch <= 100 and stop_epoch % 5 == 0, "invalid stop epoch for {}".format(job_id))
    if early["status"] == "early_stopped":
        require(stop_epoch < 100, "{} claims early stop at maximum epoch".format(job_id))
        require(int(early.get("stale_validations", -1)) >= 4, "{} stopped before patience".format(job_id))
        require(early.get("reason") == "validation_metric_saturated", "{} wrong stop reason".format(job_id))
    else:
        require(stop_epoch == 100, "{} max-epoch receipt ended early".format(job_id))
    epochs = sorted(epoch for epoch, _ in validations)
    require(epochs == list(range(5, stop_epoch + 1, 5)), "{} validation epochs do not match stop receipt".format(job_id))
    maximum = max(values[PRIMARY] for _, values in validations)
    winning_epoch = min(epoch for epoch, values in validations if close(values[PRIMARY], maximum))
    require(close(receipt["score"], maximum), "receipt is not validation maximum for {}".format(job_id))
    require(best_epoch == winning_epoch, "receipt epoch is not first strict maximum for {}".format(job_id))

    final_path = run_dir / "eval_epoch_last.log"
    require(final_path.is_file(), "missing final best-model evaluation for {}".format(job_id))
    final_metrics = parse_eval(final_path)
    require(close(final_metrics[PRIMARY], receipt["score"]), "final reload metric differs from retained best for {}".format(job_id))
    require(summary["best_checkpoint"] == str(checkpoint), "summary checkpoint path mismatch for {}".format(job_id))
    expected_metric_text = "{}={}".format(PRIMARY, receipt["score"])
    require(summary["metric"] == expected_metric_text, "summary metric mismatch for {}".format(job_id))
    return {
        "job_id": job_id,
        "source_type": "trained",
        "source_url": "",
        "protocol_note": "Independently retrained from the verified official initialization under the ScanRefer spaCy parsing protocol.",
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_score": float(receipt["score"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "final_eval": str(final_path),
        "final_eval_sha256": sha256(final_path),
        "overall_acc025": final_metrics["last__bbs_acc0.25_top1"],
        "overall_acc050": final_metrics["last__bbs_acc0.50_top1"],
        "unique_acc025": final_metrics["last__bbs_unique_acc0.25_top1"],
        "unique_acc050": final_metrics["last__bbs_unique_acc0.50_top1"],
        "multiple_acc025": final_metrics["last__bbs_multiple_acc0.25_top1"],
        "multiple_acc050": final_metrics["last__bbs_multiple_acc0.50_top1"],
        "unique_count": 1419,
        "multiple_count": 8089,
    }


def write_outputs(rows):
    baseline = rows[0]["overall_acc025"]
    for row in rows:
        row["delta_vs_baseline_acc025"] = row["overall_acc025"] - baseline
    payload = {
        "status": "PASS",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": {
            "dataset": "ScanRefer only",
            "primary_metric": PRIMARY,
            "epochs": "1-100 maximum; validation-based early stopping",
            "validation_frequency": 5,
            "early_stopping": {
                "metric": PRIMARY,
                "min_epoch": 35,
                "patience_validations": 4,
                "min_delta": 0.001,
            },
            "seed": 0,
            "independent_retraining": True,
            "independent_retraining_scope": "six trained ablation variants; external paper baseline excluded",
            "external_paper_baseline": True,
            "external_paper_baseline_url": "https://arxiv.org/abs/2112.08879",
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
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(QUEUE / "completion_audit.tsv", buffer.getvalue())

    lines = [
        "# ScanRefer ablation completion audit",
        "",
        "Overall status: **PASS**",
        "",
        "The BUTD-DETR baseline is an external ECCV 2022 paper reference and was not retrained. The other six rows were independently retrained from epoch 1 with a 100-epoch ceiling and the same validation-based early-stopping policy from the verified official initialization; no trained row used `checkpoint_path`. Validation ran every 5 epochs, only the strict maximum official BBS Acc@0.25 checkpoint was retained per trained row, its SHA256 was verified, no additional weight file remained, and final evaluation reloaded that checkpoint.",
        "",
        "| Row | Best epoch | Overall @0.25 | Overall @0.50 | Unique @0.25 | Unique @0.50 | Multiple @0.25 | Multiple @0.50 | Delta @0.25 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {job_id} | {best_epoch} | {overall_acc025:.4%} | {overall_acc050:.4%} | {unique_acc025:.4%} | {unique_acc050:.4%} | {multiple_acc025:.4%} | {multiple_acc050:.4%} | {delta_vs_baseline_acc025:+.4%} |".format(**row)
        )
    lines.extend(["", "The external paper baseline reports 84.2/66.3 Unique, 46.6/35.1 Multiple, and 52.2/39.8 Overall at Acc@0.25/0.50. It used the original BUTD-DETR text-label protocol; trained ablation rows use the repository ScanRefer spaCy protocol. Unique/Multiple counts for every trained final evaluation: 1,419 + 8,089 = 9,508.", ""])
    atomic_text(QUEUE / "COMPLETION_AUDIT.md", "\n".join(lines))
    atomic_text(QUEUE / "COMPLETION_AUDIT_PASS", payload["audited_at"] + "\n")


def audit():
    require(not (QUEUE / "WATCHDOG_ALERT").exists(), "watchdog alert exists")
    check_manifest()
    summaries = load_summary()
    rows = [audit_job(job_id, EXPECTED[job_id], summaries[job_id]) for job_id in EXPECTED]
    trained = [row for row in rows if row["source_type"] == "trained"]
    require(len(trained) == 6, "expected six independently trained original-queue rows")
    require(len({row["run_dir"] for row in trained}) == 6, "trained run directories are not independent")
    write_outputs(rows)
    print("SCANREFER_ABLATION_COMPLETION_AUDIT_PASS")


def completed_count():
    return sum((QUEUE / "status" / (job_id + ".done")).is_file() for job_id in EXPECTED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    if not args.watch:
        print("COMPLETED {}/7".format(completed_count()))
        return 0 if completed_count() == 7 else 2
    while completed_count() < 7:
        if (QUEUE / "WATCHDOG_ALERT").exists():
            raise RuntimeError("watchdog alert before completion")
        print("{} COMPLETED {}/7".format(time.strftime("%Y-%m-%dT%H:%M:%S%z"), completed_count()), flush=True)
        time.sleep(args.poll_seconds)
    audit()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        error = traceback.format_exc()
        atomic_text(QUEUE / "FINALIZER_ALERT", error)
        print(error, flush=True)
        raise
