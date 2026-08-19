#!/usr/bin/env python3
"""Atomically render the current ten-row ScanRefer ablation paper table."""

import csv
import json
import os
import time
from io import StringIO
from pathlib import Path


REPO = Path("/home/gb/new butd/butd_detr-main")
TRAIN = REPO / "logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_from_official_init"
QUEUE1 = REPO / "logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_queue"
QUEUE2 = REPO / "logs/butd_universal_target/scanrefer_ablation_extension_20260815_queue"
OUT = REPO / "reports/tuning/scanrefer_ablation_20260815"

ROWS = [
    ("module", "01_baseline", "BUTD baseline", 0, 0, 0, 0, 0, 0),
    ("module", "08_sacr_only", "SACR only", 1, 0, 0, 0, 0, 1),
    ("module", "03_no_sacr_rapf_qahnl_base", "QAHNL only (base source)", 0, 0, 1, 0, 0, 0),
    ("module", "09_sacr_qahnl", "SACR + QAHNL (structured source)", 1, 0, 1, 0, 0, 1),
    ("module", "04_no_qahnl", "SACR + RAPF", 1, 1, 0, 1, 1, 1),
    ("module", "02_full_sacr_rapf_qahnl", "Full model", 1, 1, 1, 1, 1, 1),
    ("internal", "05_no_quality", "Full w/o Quality", 1, 1, 1, 0, 1, 1),
    ("internal", "06_no_gate_supervision", "Full w/o Gate supervision", 1, 1, 1, 1, 0, 1),
    ("internal", "07_no_relation", "Full w/o Relation", 1, 1, 1, 1, 1, 0),
    ("internal", "10_full_qahnl_base_source", "Full with QAHNL base source", 1, 1, 1, 1, 1, 1),
]

KEYS = {
    "overall025": "last__bbs_acc0.25_top1",
    "overall050": "last__bbs_acc0.50_top1",
    "unique025": "last__bbs_unique_acc0.25_top1",
    "unique050": "last__bbs_unique_acc0.50_top1",
    "multiple025": "last__bbs_multiple_acc0.25_top1",
    "multiple050": "last__bbs_multiple_acc0.50_top1",
    "unique_count": "last__bbs_unique_count_acc0.25",
    "multiple_count": "last__bbs_multiple_count_acc0.25",
}


def atomic(path, text):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def parse_eval(path):
    values = {}
    for line in path.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        if key in KEYS.values():
            try:
                values[key] = float(raw.strip())
            except ValueError:
                pass
    if not set(KEYS.values()).issubset(values):
        return None
    if int(round(values[KEYS["unique_count"]])) != 1419:
        return None
    if int(round(values[KEYS["multiple_count"]])) != 8089:
        return None
    return {short: values[key] for short, key in KEYS.items()}


def completed(job_id):
    return (QUEUE1 / "status" / (job_id + ".done")).is_file() or (QUEUE2 / "status" / (job_id + ".done")).is_file()


def current_result(job_id):
    receipts = sorted((TRAIN / job_id / "scanrefer_spacy").glob("*/best_primary.json"), key=lambda p: p.stat().st_mtime)
    if not receipts:
        return {"status": "queued", "best_epoch": "-"}
    receipt_path = receipts[-1]
    receipt = json.loads(receipt_path.read_text())
    epoch = int(receipt["epoch"])
    final_path = receipt_path.parent / "eval_epoch_last.log"
    source = final_path if completed(job_id) and final_path.is_file() else receipt_path.parent / ("eval_epoch_{}.log".format(epoch))
    metrics = parse_eval(source) if source.is_file() else None
    result = {"status": "completed" if completed(job_id) else "training", "best_epoch": epoch}
    if metrics:
        result.update(metrics)
    return result


def pct(value):
    return "–" if value is None else "{:.2f}".format(100.0 * value)


def main():
    data = []
    for group, job_id, label, sacr, rapf, qahnl, quality, gate, relation in ROWS:
        result = current_result(job_id)
        data.append({
            "group": group, "job_id": job_id, "setting": label,
            "sacr": sacr, "rapf": rapf, "qahnl": qahnl,
            "quality": quality, "gate_supervision": gate, "relation": relation,
            **result,
        })

    fields = [
        "group", "job_id", "setting", "status", "best_epoch", "sacr", "rapf", "qahnl",
        "quality", "gate_supervision", "relation", "unique025", "unique050",
        "multiple025", "multiple050", "overall025", "overall050",
    ]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
    writer.writeheader(); writer.writerows(data)
    atomic(OUT / "paper_ablation_table.tsv", buffer.getvalue())

    generated = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    md = [
        "# ScanRefer ablation table", "", "Generated: `{}`".format(generated), "",
        "Results are official BBS percentages. `†` denotes an interim strict-best checkpoint; `–` denotes no result yet.", "",
        "| Group | Setting | SACR | RAPF | QAHNL | Quality | Gate Sup. | Relation | U@0.25 | U@0.50 | M@0.25 | M@0.50 | O@0.25 | O@0.50 |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data:
        suffix = "†" if row["status"] == "training" else ""
        checks = ["✓" if row[key] else "×" for key in ("sacr", "rapf", "qahnl", "quality", "gate_supervision", "relation")]
        vals = [pct(row.get(key)) for key in ("unique025", "unique050", "multiple025", "multiple050", "overall025", "overall050")]
        md.append("| {} | {}{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(row["group"], row["setting"], suffix, *(checks + vals)))
    md.extend([
        "", "All rows use ScanRefer only, seed 0, epochs 1--100, validation every 5 epochs, and the same verified official detector initialization. Each completed row retains only the strict-best official Overall Acc@0.25 checkpoint.",
        "", "RAPF is structurally dependent on SACR; therefore RAPF-only and RAPF+QAHNL-without-SACR are invalid configurations and are intentionally excluded.", "",
    ])
    atomic(OUT / "PAPER_ABLATION_TABLE.md", "\n".join(md))

    tex = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Dependency-aware module and internal ablations on ScanRefer. All values are percentages.}",
        r"\label{tab:scanrefer_ablation}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc|cc|cc|cc}", r"\toprule",
        r"Setting & SACR & RAPF & QAHNL & Quality & Gate Sup. & Relation & U@.25 & U@.50 & M@.25 & M@.50 & O@.25 & O@.50 \\",
        r"\midrule",
    ]
    for index, row in enumerate(data):
        if index == 6:
            tex.append(r"\midrule")
        label = row["setting"].replace("_", r"\_") + (r"$^{\dagger}$" if row["status"] == "training" else "")
        checks = [r"$\checkmark$" if row[key] else r"$\times$" for key in ("sacr", "rapf", "qahnl", "quality", "gate_supervision", "relation")]
        vals = ["--" if row.get(key) is None else "{:.2f}".format(100.0 * row[key]) for key in ("unique025", "unique050", "multiple025", "multiple050", "overall025", "overall050")]
        tex.append("{} & {} \\\\".format(label, " & ".join(checks + vals)))
    tex.extend([
        r"\bottomrule", r"\end{tabular}}", r"\vspace{2pt}",
        r"\parbox{\textwidth}{\footnotesize $^{\dagger}$Interim strict-best checkpoint; -- denotes a pending result. RAPF requires SACR structured scores, so invalid RAPF-without-SACR combinations are excluded.}",
        r"\end{table*}", "",
    ])
    atomic(OUT / "PAPER_ABLATION_TABLE.tex", "\n".join(tex))
    print("PAPER_TABLE_UPDATED {} rows={} completed={}".format(generated, len(data), sum(row["status"] == "completed" for row in data)))


if __name__ == "__main__":
    main()
