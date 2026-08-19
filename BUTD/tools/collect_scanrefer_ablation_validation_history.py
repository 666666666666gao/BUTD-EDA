#!/usr/bin/env python3
"""Collect complete official-BBS validation logs into atomic audit tables."""

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path


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


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_complete_eval(path):
    values = {}
    for line in path.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        if key in METRICS:
            try:
                values[key] = float(raw.strip())
            except ValueError:
                pass
    if any(key not in values for key in METRICS):
        return None
    unique_count = int(round(values["last__bbs_unique_count_acc0.25"]))
    multiple_count = int(round(values["last__bbs_multiple_count_acc0.25"]))
    if unique_count != int(round(values["last__bbs_unique_count_acc0.50"])):
        raise ValueError("Unique count differs between thresholds in {}".format(path))
    if multiple_count != int(round(values["last__bbs_multiple_count_acc0.50"])):
        raise ValueError("Multiple count differs between thresholds in {}".format(path))
    if unique_count + multiple_count != 9508:
        raise ValueError(
            "unexpected official-BBS subset total in {}: {} + {}".format(
                path, unique_count, multiple_count
            )
        )
    return values


def atomic_write_text(path, text):
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    temporary.write_text(text)
    os.replace(str(temporary), str(path))


def collect(train_root):
    rows = []
    pattern = re.compile(r"^eval_epoch_(\d+|last)\.log$")
    for path in train_root.glob("*/scanrefer_spacy/*/eval_epoch_*.log"):
        match = pattern.match(path.name)
        if not match:
            continue
        values = parse_complete_eval(path)
        if values is None:
            continue
        relative = path.relative_to(train_root)
        job_id, _, run_id, _ = relative.parts
        epoch_text = match.group(1)
        row_type = "final_best" if epoch_text == "last" else "validation"
        epoch_sort = 10 ** 9 if epoch_text == "last" else int(epoch_text)
        rows.append({
            "job_id": job_id,
            "run_id": run_id,
            "row_type": row_type,
            "epoch": epoch_text,
            "overall_acc025": values["last__bbs_acc0.25_top1"],
            "overall_acc050": values["last__bbs_acc0.50_top1"],
            "unique_acc025": values["last__bbs_unique_acc0.25_top1"],
            "unique_acc050": values["last__bbs_unique_acc0.50_top1"],
            "multiple_acc025": values["last__bbs_multiple_acc0.25_top1"],
            "multiple_acc050": values["last__bbs_multiple_acc0.50_top1"],
            "unique_count": int(round(values["last__bbs_unique_count_acc0.25"])),
            "multiple_count": int(round(values["last__bbs_multiple_count_acc0.25"])),
            "eval_log": str(path.resolve()),
            "eval_sha256": file_sha256(path),
            "_epoch_sort": epoch_sort,
        })
    rows.sort(key=lambda row: (row["job_id"], int(row["run_id"]), row["_epoch_sort"]))
    for row in rows:
        row.pop("_epoch_sort")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(args.train_root)
    fields = list(rows[0].keys()) if rows else [
        "job_id", "run_id", "row_type", "epoch", "overall_acc025",
        "overall_acc050", "unique_acc025", "unique_acc050",
        "multiple_acc025", "multiple_acc050", "unique_count",
        "multiple_count", "eval_log", "eval_sha256",
    ]
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(args.output_dir / "validation_history.tsv", buffer.getvalue())
    atomic_write_text(
        args.output_dir / "validation_history.json",
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
    )
    print("VALIDATION_HISTORY_ROWS {}".format(len(rows)))


if __name__ == "__main__":
    main()
