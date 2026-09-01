#!/usr/bin/env python3
"""Verify both machine manifests and atomically build the final table bundle."""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import tempfile

from assemble_final_tables import assemble


MACHINE_ROWS = {
    "machine35608": ("S2", "S0", "S3"),
    "machine50630": ("R1", "R3", "R0", "R2"),
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


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_machine_manifest(path):
    fields = {}
    declared = []
    with open(path, "r") as handle:
        for raw in handle:
            line = raw.strip()
            match = re.match(r"^([0-9a-fA-F]{64})\s+(.+)$", line)
            if match:
                basename = os.path.basename(match.group(2))
                if not basename.endswith(".json"):
                    raise ValueError("machine manifest contains a non-JSON row path")
                declared.append((basename[:-5], match.group(1).lower()))
            elif "=" in line:
                key, value = line.split("=", 1)
                fields[key] = value
    for name in ("timestamp", "machine", "row_count"):
        if name not in fields:
            raise ValueError("machine manifest is missing {}".format(name))
    machine = fields["machine"]
    if machine not in MACHINE_ROWS:
        raise ValueError("unexpected machine manifest: {}".format(machine))
    if int(fields["row_count"]) != len(declared):
        raise ValueError("{} row_count differs from SHA entries".format(machine))
    rows = tuple(row for row, _ in declared)
    if rows != MACHINE_ROWS[machine]:
        raise ValueError(
            "{} row assignment mismatch: expected {}, got {}".format(
                machine,
                ",".join(MACHINE_ROWS[machine]),
                ",".join(rows),
            )
        )
    if len(set(rows)) != len(rows):
        raise ValueError("{} contains duplicate rows".format(machine))
    return machine, dict(declared)


def output_hashes(root):
    hashes = {}
    for directory, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if filename == "FINAL_TABLES_RECEIPT.json":
                continue
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            hashes[relative] = sha256(path)
    return hashes


def finalize(manifest_path, machine_manifest_paths, row_paths, output_dir):
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir):
        raise ValueError("output directory already exists: {}".format(output_dir))

    parsed_manifests = {}
    manifest_paths_by_machine = {}
    for path in machine_manifest_paths:
        machine, declared = parse_machine_manifest(path)
        if machine in parsed_manifests:
            raise ValueError("duplicate machine manifest: {}".format(machine))
        parsed_manifests[machine] = declared
        manifest_paths_by_machine[machine] = path
    if set(parsed_manifests) != set(MACHINE_ROWS):
        raise ValueError(
            "expected both machine manifests {}, got {}".format(
                ",".join(sorted(MACHINE_ROWS)),
                ",".join(sorted(parsed_manifests)),
            )
        )

    row_paths_by_id = {}
    for path in row_paths:
        row = load_json(path).get("row")
        if row in row_paths_by_id:
            raise ValueError("duplicate row JSON: {}".format(row))
        row_paths_by_id[row] = path
    expected_rows = set(row for rows in MACHINE_ROWS.values() for row in rows)
    if set(row_paths_by_id) != expected_rows:
        raise ValueError(
            "expected exactly rows {}, got {}".format(
                ",".join(sorted(expected_rows)),
                ",".join(sorted(str(row) for row in row_paths_by_id)),
            )
        )

    for machine, expected in MACHINE_ROWS.items():
        for row in expected:
            actual = sha256(row_paths_by_id[row])
            declared = parsed_manifests[machine][row]
            if actual != declared:
                raise ValueError(
                    "{} {} declared SHA256 mismatch: declared={} actual={}".format(
                        machine, row, declared, actual
                    )
                )

    parent = os.path.dirname(output_dir)
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix=".final-table-bundle.", dir=parent)
    try:
        ordered_rows = [
            row_paths_by_id[row]
            for machine in ("machine35608", "machine50630")
            for row in MACHINE_ROWS[machine]
        ]
        final_json = os.path.join(temporary, "final_tables.json")
        final_markdown = os.path.join(temporary, "final_tables.md")
        latex_dir = os.path.join(temporary, "latex")
        assembled = assemble(
            manifest_path,
            ordered_rows,
            final_json,
            final_markdown,
            output_latex_dir=latex_dir,
        )
        receipt = {
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "machine_manifest_sha256": {
                machine: sha256(manifest_paths_by_machine[machine])
                for machine in sorted(MACHINE_ROWS)
            },
            "machine_rows": MACHINE_ROWS,
            "internal_claim_summary": assembled["internal_claim_summary"],
            "main_monotonicity": assembled["main_monotonicity"],
            "output_sha256": output_hashes(temporary),
            "plan_manifest_sha256": sha256(manifest_path),
            "row_json_sha256": {
                row: sha256(row_paths_by_id[row]) for row in sorted(row_paths_by_id)
            },
            "status": "complete",
        }
        write_json(os.path.join(temporary, "FINAL_TABLES_RECEIPT.json"), receipt)
        os.rename(temporary, output_dir)
        temporary = None
        return receipt
    finally:
        if temporary is not None and os.path.exists(temporary):
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output_dir")
    parser.add_argument("--machine-manifest", action="append", required=True)
    parser.add_argument("row_json", nargs="+")
    args = parser.parse_args()
    result = finalize(
        args.manifest,
        args.machine_manifest,
        args.row_json,
        args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
