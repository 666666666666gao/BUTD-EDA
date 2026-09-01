#!/usr/bin/env python3
"""Merge the two fail-closed epoch-20 summaries without changing either queue."""

import argparse
import hashlib
import json
import math
import os
import tempfile


EXPECTED_MACHINES = ("machine35608", "machine50630")
MACHINE_ROWS = {"machine35608": "S2", "machine50630": "R1"}
METRICS = (
    "unique_025",
    "unique_050",
    "multiple_025",
    "multiple_050",
    "overall_025",
    "overall_050",
)
EXPECTED_COUNTS = {"multiple": 8089, "total": 9508, "unique": 1419}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def atomic_text(path, content):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".e20-merged.", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def validate_summary(payload):
    machine = payload.get("machine")
    if payload.get("epoch") != 20:
        raise ValueError("{} summary is not epoch 20".format(machine or "unknown"))
    if payload.get("counts") != EXPECTED_COUNTS:
        raise ValueError("{} sample contract mismatch".format(machine))
    if not is_sha256(payload.get("eval_log_sha256")):
        raise ValueError("{} eval log SHA256 is invalid".format(machine))
    if not payload.get("ready_timestamp"):
        raise ValueError("{} ready timestamp is missing".format(machine))
    metrics = payload.get("metrics", {})
    percent = payload.get("percent", {})
    for name in METRICS:
        if name not in metrics or name not in percent:
            raise ValueError("{} is missing metric {}".format(machine, name))
        value = float(metrics[name])
        rendered = float(percent[name])
        if not math.isfinite(value) or not math.isfinite(rendered):
            raise ValueError("{} metric {} is non-finite".format(machine, name))
        if value < 0.0 or value > 1.0:
            raise ValueError("{} metric {} is outside [0,1]".format(machine, name))
        if abs(rendered - value * 100.0) > 1e-8:
            raise ValueError("{} percent/metric mismatch for {}".format(machine, name))


def render_markdown(rows):
    lines = [
        "# ScanRefer epoch-20 dual-machine snapshot",
        "",
        "This report is provisional and read-only. It does not stop, restart, or modify either queue.",
        "",
        "| Row | Machine | Milestone | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ("S2", "R1"):
        payload = rows[row]
        values = payload["percent"]
        lines.append(
            "| {} | {} | E20 | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                row,
                payload["machine"],
                values["unique_025"],
                values["unique_050"],
                values["multiple_025"],
                values["multiple_050"],
                values["overall_025"],
                values["overall_050"],
            )
        )
    lines.extend(
        [
            "",
            "Checkpoint selection remains strict-best Overall@0.25 under the locked 65-epoch protocol; this E20 snapshot is not a final result.",
            "",
        ]
    )
    return "\n".join(lines)


def merge(summary_paths, output_json, output_markdown):
    payloads = [load_json(path) for path in summary_paths]
    by_machine = {payload.get("machine"): payload for payload in payloads}
    if len(payloads) != 2 or set(by_machine) != set(EXPECTED_MACHINES):
        raise ValueError(
            "expected exactly machines {}, got {}".format(
                ",".join(EXPECTED_MACHINES),
                ",".join(sorted(str(name) for name in by_machine)),
            )
        )
    for machine in EXPECTED_MACHINES:
        validate_summary(by_machine[machine])
    eval_shas = [by_machine[machine]["eval_log_sha256"] for machine in EXPECTED_MACHINES]
    if len(set(eval_shas)) != len(eval_shas):
        raise ValueError("the two E20 evaluation logs have the same SHA256")

    paths_by_machine = {
        payload["machine"]: path for path, payload in zip(summary_paths, payloads)
    }
    rows = {}
    source_shas = {}
    for machine in EXPECTED_MACHINES:
        row = MACHINE_ROWS[machine]
        source_shas[machine] = sha256(paths_by_machine[machine])
        payload = by_machine[machine]
        rows[row] = {
            "counts": dict(payload["counts"]),
            "epoch": 20,
            "eval_log_sha256": payload["eval_log_sha256"],
            "machine": machine,
            "metrics": {name: float(payload["metrics"][name]) for name in METRICS},
            "percent": {name: float(payload["percent"][name]) for name in METRICS},
            "ready_timestamp": payload["ready_timestamp"],
        }
    result = {
        "epoch": 20,
        "provisional_only_no_queue_action": True,
        "rows": rows,
        "source_summary_sha256": source_shas,
        "status": "both_e20_complete",
    }
    atomic_text(output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
    atomic_text(output_markdown, render_markdown(rows))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_json")
    parser.add_argument("output_markdown")
    parser.add_argument("summary_json", nargs="+")
    args = parser.parse_args()
    result = merge(
        args.summary_json,
        args.output_json,
        args.output_markdown,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
