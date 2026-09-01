#!/usr/bin/env python3
"""Fail-closed collector for one completed ScanRefer epoch-20 evaluation."""

import argparse
import hashlib
import json
import os
import re
import tempfile


KEYS = (
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

ALIASES = {
    "overall_025": "last__bbs_acc0.25_top1",
    "overall_050": "last__bbs_acc0.50_top1",
    "unique_025": "last__bbs_unique_acc0.25_top1",
    "unique_050": "last__bbs_unique_acc0.50_top1",
    "multiple_025": "last__bbs_multiple_acc0.25_top1",
    "multiple_050": "last__bbs_multiple_acc0.50_top1",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_ready(path):
    fields = {}
    declared_sha = None
    declared_sha_path = None
    with open(path, "r") as handle:
        for raw in handle:
            line = raw.strip()
            match = re.match(r"^([0-9a-fA-F]{64})\s+(.+)$", line)
            if match:
                declared_sha = match.group(1).lower()
                declared_sha_path = match.group(2)
            elif "=" in line:
                key, value = line.split("=", 1)
                fields[key] = value
    for key in ("timestamp", "machine", "milestone", "required_metric_keys"):
        if key not in fields:
            raise ValueError("missing E20_READY field: {}".format(key))
    if int(fields["required_metric_keys"]) != len(KEYS):
        raise ValueError("E20_READY metric-key count is not {}".format(len(KEYS)))
    if declared_sha is None or declared_sha_path is None:
        raise ValueError("E20_READY is missing its SHA256 line")
    milestone = os.path.abspath(fields["milestone"])
    if os.path.abspath(declared_sha_path) != milestone:
        raise ValueError("E20_READY SHA256 path differs from milestone path")
    if os.path.basename(milestone) != "eval_epoch_20.log":
        raise ValueError("milestone is not eval_epoch_20.log")
    return fields, milestone, declared_sha


def parse_metrics(path):
    values = {}
    with open(path, "r", errors="replace") as handle:
        for raw in handle:
            if ":" not in raw:
                continue
            key, value = raw.split(":", 1)
            key = key.strip()
            if key not in KEYS:
                continue
            parsed = float(value.strip())
            if key in values and values[key] != parsed:
                raise ValueError("conflicting duplicate metric: {}".format(key))
            values[key] = parsed
    missing = sorted(set(KEYS) - set(values))
    if missing:
        raise ValueError("missing metric keys: {}".format(", ".join(missing)))
    return values


def atomic_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".e20-summary.", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def collect(ready_path, output_path):
    fields, milestone, declared_sha = parse_ready(ready_path)
    actual_sha = sha256(milestone)
    if actual_sha != declared_sha:
        raise ValueError(
            "SHA256 mismatch: ready={} actual={}".format(declared_sha, actual_sha)
        )
    raw = parse_metrics(milestone)
    unique_025 = int(round(raw["last__bbs_unique_count_acc0.25"]))
    unique_050 = int(round(raw["last__bbs_unique_count_acc0.50"]))
    multiple_025 = int(round(raw["last__bbs_multiple_count_acc0.25"]))
    multiple_050 = int(round(raw["last__bbs_multiple_count_acc0.50"]))
    if unique_025 != unique_050 or multiple_025 != multiple_050:
        raise ValueError("Unique/Multiple counts differ between IoU thresholds")
    if unique_025 + multiple_025 != 9508:
        raise ValueError(
            "unexpected sample contract: {} + {} != 9508".format(
                unique_025, multiple_025
            )
        )
    metrics = {alias: raw[key] for alias, key in ALIASES.items()}
    result = {
        "counts": {
            "multiple": multiple_025,
            "total": unique_025 + multiple_025,
            "unique": unique_025,
        },
        "epoch": 20,
        "eval_log": milestone,
        "eval_log_sha256": actual_sha,
        "machine": fields["machine"],
        "metrics": metrics,
        "percent": {key: value * 100.0 for key, value in metrics.items()},
        "ready_file": os.path.abspath(ready_path),
        "ready_timestamp": fields["timestamp"],
    }
    atomic_json(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ready_file")
    parser.add_argument("output_json")
    args = parser.parse_args()
    result = collect(args.ready_file, args.output_json)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
