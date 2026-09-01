#!/usr/bin/env python3
"""Collect one completed dual-machine ablation row with provenance checks."""

import argparse
import hashlib
import json
import os
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

EXPECTED_PROTOCOL = {
    "rng_seed": 0,
    "batch_size": 24,
    "max_epoch": 65,
    "val_freq": 5,
    "lr": 0.0001,
    "lr_backbone": 0.001,
    "lr_decay_epochs": [55, 60],
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def parse_metrics(path):
    values = {}
    with open(path, "r", errors="replace") as handle:
        for raw in handle:
            if ":" not in raw:
                continue
            key, value = raw.split(":", 1)
            key = key.strip()
            if key in KEYS:
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
    descriptor, temporary = tempfile.mkstemp(prefix=".row-summary.", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def collect(receipt_path, output_path):
    receipt = load_json(receipt_path)
    run = os.path.abspath(receipt["run"])
    checkpoint = os.path.abspath(receipt["checkpoint"])
    expected_checkpoint = os.path.join(run, "ckpt_best_primary.pth")
    if checkpoint != expected_checkpoint:
        raise ValueError("receipt checkpoint is not the run strict-best checkpoint")
    weights = sorted(
        os.path.join(run, name) for name in os.listdir(run) if name.endswith(".pth")
    )
    if weights != [expected_checkpoint]:
        raise ValueError(
            "expected exactly one weight named ckpt_best_primary.pth, found {}".format(
                len(weights)
            )
        )
    if os.path.getsize(checkpoint) != int(receipt["checkpoint_size"]):
        raise ValueError("checkpoint size mismatch")
    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != receipt["checkpoint_sha256"]:
        raise ValueError(
            "checkpoint SHA256 mismatch: receipt={} actual={}".format(
                receipt["checkpoint_sha256"], checkpoint_sha
            )
        )
    config_path = os.path.join(run, "config.json")
    config = load_json(config_path)
    for key, expected in EXPECTED_PROTOCOL.items():
        if config.get(key) != expected:
            raise ValueError(
                "protocol mismatch: {} expected={!r} actual={!r}".format(
                    key, expected, config.get(key)
                )
            )
    best = load_json(os.path.join(run, "best_primary.json"))
    if receipt.get("best_primary") != best:
        raise ValueError("receipt best_primary differs from run best_primary.json")
    epoch = int(best["epoch"])
    best_log = os.path.join(run, "eval_epoch_{}.log".format(epoch))
    reload_log = os.path.join(run, "eval_epoch_last.log")
    raw = parse_metrics(best_log)
    reload_raw = parse_metrics(reload_log)
    primary_key = "last__bbs_acc0.25_top1"
    if best.get("metric") != primary_key:
        raise ValueError("best metric is not {}".format(primary_key))
    if abs(float(best["score"]) - raw[primary_key]) > 1e-9:
        raise ValueError(
            "best score does not match eval log: best={} eval={}".format(
                best["score"], raw[primary_key]
            )
        )
    if raw != reload_raw:
        changed = sorted(key for key in KEYS if raw[key] != reload_raw[key])
        raise ValueError("reload parity failed for: {}".format(", ".join(changed)))
    unique = int(round(raw["last__bbs_unique_count_acc0.25"]))
    unique_050 = int(round(raw["last__bbs_unique_count_acc0.50"]))
    multiple = int(round(raw["last__bbs_multiple_count_acc0.25"]))
    multiple_050 = int(round(raw["last__bbs_multiple_count_acc0.50"]))
    if unique != unique_050 or multiple != multiple_050:
        raise ValueError("Unique/Multiple counts differ between IoU thresholds")
    if unique != 1419 or multiple != 8089 or unique + multiple != 9508:
        raise ValueError(
            "unexpected sample contract: unique={} multiple={} total={}".format(
                unique, multiple, unique + multiple
            )
        )
    for threshold in ("0.25", "0.50"):
        expected_overall = (
            unique * raw["last__bbs_unique_acc{}_top1".format(threshold)]
            + multiple * raw["last__bbs_multiple_acc{}_top1".format(threshold)]
        ) / float(unique + multiple)
        observed_overall = raw["last__bbs_acc{}_top1".format(threshold)]
        if abs(expected_overall - observed_overall) > 1e-9:
            raise ValueError(
                "Overall/subset mismatch at {}: expected={} observed={}".format(
                    threshold, expected_overall, observed_overall
                )
            )
    metrics = {alias: raw[key] for alias, key in ALIASES.items()}
    result = {
        "best_epoch": epoch,
        "best_eval_log": best_log,
        "best_eval_log_sha256": sha256(best_log),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "config": config_path,
        "config_sha256": sha256(config_path),
        "counts": {"unique": unique, "multiple": multiple, "total": unique + multiple},
        "metrics": metrics,
        "percent": {key: value * 100.0 for key, value in metrics.items()},
        "receipt": os.path.abspath(receipt_path),
        "reload_eval_log": reload_log,
        "reload_eval_log_sha256": sha256(reload_log),
        "reload_parity": True,
        "row": receipt["row"],
        "training_protocol": {key: config[key] for key in EXPECTED_PROTOCOL},
    }
    atomic_json(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt")
    parser.add_argument("output_json")
    args = parser.parse_args()
    print(json.dumps(collect(args.receipt, args.output_json), sort_keys=True))


if __name__ == "__main__":
    main()
