#!/usr/bin/env python3
"""Assemble the three final ScanRefer ablation tables from audited rows."""

import argparse
import json
import os
import tempfile


EXPECTED_ROWS = ("S0", "S2", "S3", "R0", "R1", "R2", "R3")
METRIC_NAMES = ("multiple_025", "multiple_050", "overall_025", "overall_050")


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def atomic_text(path, content):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".final-tables.", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def row_metrics(payload):
    return {name: float(payload["percent"][name]) for name in METRIC_NAMES}


def known_metrics(payload):
    return {name: float(payload[name]) for name in METRIC_NAMES}


def format_value(value):
    return "{:.4f}".format(value)


def markdown_table(title, rows):
    lines = [
        "## {}".format(title),
        "",
        "| ID | Variant | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row_id, label, metrics in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                row_id,
                label,
                format_value(metrics["multiple_025"]),
                format_value(metrics["multiple_050"]),
                format_value(metrics["overall_025"]),
                format_value(metrics["overall_050"]),
            )
        )
    return "\n".join(lines)


def assemble(manifest_path, row_paths, output_json, output_markdown):
    manifest = load_json(manifest_path)
    payloads = [load_json(path) for path in row_paths]
    by_row = {payload["row"]: payload for payload in payloads}
    if len(by_row) != len(payloads) or set(by_row) != set(EXPECTED_ROWS):
        raise ValueError(
            "expected exactly rows {}, got {}".format(
                ",".join(EXPECTED_ROWS), ",".join(sorted(by_row))
            )
        )
    for row, payload in by_row.items():
        if not payload.get("reload_parity"):
            raise ValueError("reload parity is false for {}".format(row))
        if payload.get("counts") != {"unique": 1419, "multiple": 8089, "total": 9508}:
            raise ValueError("sample contract mismatch for {}".format(row))
    canonical_protocol = by_row[EXPECTED_ROWS[0]]["training_protocol"]
    for row in EXPECTED_ROWS[1:]:
        if by_row[row]["training_protocol"] != canonical_protocol:
            raise ValueError("training protocol differs for {}".format(row))
    checkpoint_shas = [by_row[row]["checkpoint_sha256"] for row in EXPECTED_ROWS]
    if len(set(checkpoint_shas)) != len(checkpoint_shas):
        raise ValueError("duplicate checkpoint SHA256 across new rows")

    main_modules = {"M0": dict(manifest["baseline"])}
    main_modules.update(
        {row: dict(values) for row, values in manifest["main_rows"].items()}
    )
    full = known_metrics(manifest["matched_internal_full"])
    s1 = known_metrics(manifest["reused_internal_rows"]["S1"])
    sacr_internal = {
        "S0": row_metrics(by_row["S0"]),
        "S1": s1,
        "S2": row_metrics(by_row["S2"]),
        "S3": row_metrics(by_row["S3"]),
        "S4": full,
    }
    rapf_internal = {
        "R0": row_metrics(by_row["R0"]),
        "R1": row_metrics(by_row["R1"]),
        "R2": row_metrics(by_row["R2"]),
        "R3": row_metrics(by_row["R3"]),
        "R4": full,
    }
    result = {
        "main_modules": main_modules,
        "new_row_count": len(payloads),
        "new_row_training_protocol": canonical_protocol,
        "new_row_provenance": {
            row: {
                "best_epoch": payload["best_epoch"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
            }
            for row, payload in sorted(by_row.items())
        },
        "rapf_internal": rapf_internal,
        "sacr_internal": sacr_internal,
    }

    main_lines = [
        "## Table 3. Main modules",
        "",
        "| ID | Setting | Overall@0.25 | Overall@0.50 |",
        "|---|---|---:|---:|",
    ]
    main_labels = {
        "M0": "BUTD-DETR paper baseline",
        "M1": "+ SACR",
        "M2": "+ RAPF",
        "M3": "+ QAHNL + fixed calibration",
    }
    for row in ("M0", "M1", "M2", "M3"):
        values = main_modules[row]
        main_lines.append(
            "| {} | {} | {} | {} |".format(
                row,
                main_labels[row],
                format_value(values["overall_025"]),
                format_value(values["overall_050"]),
            )
        )
    sacr_labels = {
        "S0": "w/o target-attribute",
        "S1": "w/o relation-anchor",
        "S2": "w/o pairwise geometry",
        "S3": "hard top-1 anchor",
        "S4": "matched-protocol Full",
    }
    rapf_labels = {
        "R0": "fixed fusion, g=0.1",
        "R1": "w/o query-quality cue",
        "R2": "w/o parser/anchor cues",
        "R3": "w/o gate supervision",
        "R4": "matched-protocol Full",
    }
    sections = [
        "\n".join(main_lines),
        markdown_table(
            "Table 4. SACR internal design",
            [(row, sacr_labels[row], sacr_internal[row]) for row in ("S0", "S1", "S2", "S3", "S4")],
        ),
        markdown_table(
            "Table 5. RAPF internal design",
            [(row, rapf_labels[row], rapf_internal[row]) for row in ("R0", "R1", "R2", "R3", "R4")],
        ),
    ]
    atomic_text(output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
    atomic_text(output_markdown, "\n\n".join(sections) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output_json")
    parser.add_argument("output_markdown")
    parser.add_argument("row_results", nargs="+")
    args = parser.parse_args()
    result = assemble(
        args.manifest,
        args.row_results,
        args.output_json,
        args.output_markdown,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
