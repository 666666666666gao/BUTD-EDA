#!/usr/bin/env python3
"""Combine the locked seven-row and extension audits into the final ten-row gate."""

import hashlib
import json
import os
import time
from pathlib import Path


REPO = Path("/home/gb/new butd/butd_detr-main")
Q1 = REPO / "logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_queue"
Q2 = REPO / "logs/butd_universal_target/scanrefer_ablation_extension_20260815_queue"
OUT = REPO / "reports/tuning/scanrefer_ablation_20260815"
EXPECTED = {
    "01_baseline", "02_full_sacr_rapf_qahnl", "03_no_sacr_rapf_qahnl_base",
    "04_no_qahnl", "05_no_quality", "06_no_gate_supervision", "07_no_relation",
    "08_sacr_only", "09_sacr_qahnl", "10_full_qahnl_base_source",
}


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path, text):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def main():
    require((Q1 / "COMPLETION_AUDIT_PASS").is_file(), "original seven-row audit has not passed")
    require((Q2 / "COMPLETION_AUDIT_PASS").is_file(), "extension three-row audit has not passed")
    a1 = json.loads((Q1 / "completion_audit.json").read_text())
    a2 = json.loads((Q2 / "completion_audit.json").read_text())
    require(a1.get("status") == a2.get("status") == "PASS", "child audit status differs")
    rows = list(a1["rows"]) + list(a2["rows"])
    require(len(rows) == 10, "expected one paper baseline plus nine audited ablation rows")
    require({row["job_id"] for row in rows} == EXPECTED, "audited job set differs")
    for row in rows:
        row.setdefault("source_type", "trained")
        row.setdefault("source_url", "")
    paper_rows = [row for row in rows if row["source_type"] == "paper"]
    trained_rows = [row for row in rows if row["source_type"] == "trained"]
    require(len(paper_rows) == 1 and paper_rows[0]["job_id"] == "01_baseline", "expected exactly one external BUTD-DETR paper baseline")
    require(len(trained_rows) == 9, "expected nine independently retrained ablation rows")
    require(len({row["run_dir"] for row in trained_rows}) == 9, "trained run directories are not independent")
    for audit in (a1, a2):
        protocol = audit["protocol"]
        require(protocol["dataset"] == "ScanRefer only", "dataset scope differs")
        require(
            protocol["epochs"] == "1-100 maximum; validation-based early stopping",
            "epoch protocol differs",
        )
        require(protocol["early_stopping"] == {
            "metric": "last__bbs_acc0.25_top1",
            "min_epoch": 35,
            "patience_validations": 4,
            "min_delta": 0.001,
        }, "early-stopping protocol differs")
        require(protocol["seed"] == 0, "seed differs")
        require(protocol["independent_retraining"] is True, "not independent retraining")
        require(protocol["resume_checkpoint"] is None, "resume checkpoint found")
        require(protocol["only_one_weight_file_per_row"] is True, "weight policy differs")
        require(protocol["final_evaluation_reloads_best"] is True, "final reload policy differs")
    original_protocol = a1["protocol"]
    require(original_protocol.get("external_paper_baseline") is True, "original audit did not declare the paper baseline")
    require(original_protocol.get("independent_retraining_scope") == "six trained ablation variants; external paper baseline excluded", "original audit retraining scope differs")
    paper = paper_rows[0]
    require(paper["source_url"] == "https://arxiv.org/abs/2112.08879", "paper baseline URL differs")
    require(paper["checkpoint"] == "" and paper["final_eval"] == "", "paper baseline must not claim trained artifacts")
    require(abs(float(paper["overall_acc025"]) - 0.522) <= 5e-10, "paper baseline O@0.25 differs")
    require(abs(float(paper["overall_acc050"]) - 0.398) <= 5e-10, "paper baseline O@0.50 differs")
    for row in trained_rows:
        checkpoint = Path(row["checkpoint"])
        require(checkpoint.is_file(), "checkpoint missing: {}".format(checkpoint))
        require(sha256(checkpoint) == row["checkpoint_sha256"], "checkpoint SHA mismatch")
        run_dir = Path(row["run_dir"])
        weights = [p for p in run_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".pth", ".pt", ".ckpt")]
        require(weights == [checkpoint], "extra weights found in {}".format(run_dir))
        require(Path(row["final_eval"]).is_file(), "final reload log missing")
    payload = {
        "status": "PASS",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": "nine dependency-aware ScanRefer ablations plus one external BUTD-DETR paper baseline",
        "rows": rows,
        "child_audits": [str(Q1 / "completion_audit.json"), str(Q2 / "completion_audit.json")],
    }
    atomic(OUT / "master_completion_audit.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    atomic(OUT / "MASTER_COMPLETION_AUDIT_PASS", payload["audited_at"] + "\n")
    print("SCANREFER_TEN_ROW_MASTER_COMPLETION_AUDIT_PASS")


if __name__ == "__main__":
    main()
