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


def format_latex_value(value, is_best=False):
    rendered = "{:.2f}".format(float(value))
    if is_best:
        return "\\textbf{{{}}}".format(rendered)
    return rendered


def latex_metric_cells(metrics, metric_names, maxima):
    return [
        format_latex_value(
            metrics[name],
            "{:.2f}".format(float(metrics[name]))
            == "{:.2f}".format(float(maxima[name])),
        )
        for name in metric_names
    ]


def is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def validate_manifest_provenance(manifest):
    baseline = manifest.get("baseline", {})
    if (
        baseline.get("source") != "published_BUTD_DETR_not_retrained"
        or not baseline.get("protocol_note")
    ):
        raise ValueError("M0 baseline provenance is missing or invalid")

    main_rows = manifest.get("main_rows", {})
    for row in ("M1", "M2"):
        payload = main_rows.get(row, {})
        if (
            not str(payload.get("status", "")).startswith("reuse_audited")
            or not isinstance(payload.get("best_epoch"), int)
            or not is_sha256(payload.get("checkpoint_sha256"))
        ):
            raise ValueError("{} audited checkpoint provenance is invalid".format(row))
    if main_rows["M1"]["checkpoint_sha256"] == main_rows["M2"]["checkpoint_sha256"]:
        raise ValueError("M1 and M2 checkpoint identities must differ")

    m3 = main_rows.get("M3", {})
    required_m3_shas = (
        "validation_result_sha256",
        "subgroup_report_sha256",
        "subgroup_source_manifest_sha256",
    )
    if (
        m3.get("status") != "reuse_stage154_without_retraining"
        or m3.get("label") != "full_plus_fixed_source_selection_calibration"
        or m3.get("single_checkpoint") is not False
        or not all(is_sha256(m3.get(name)) for name in required_m3_shas)
    ):
        raise ValueError(
            "M3 must be the audited Stage154 fixed two-source non-single-checkpoint result"
        )


def render_main_latex(main_modules):
    order = ("M0", "M1", "M2", "M3")
    metric_names = (
        "unique_025",
        "unique_050",
        "multiple_025",
        "multiple_050",
        "overall_025",
        "overall_050",
    )
    maxima = {
        name: max(float(main_modules[row][name]) for row in order)
        for name in metric_names
    }
    settings = {
        "M0": "BUTD-DETR baseline",
        "M1": "+ SACR",
        "M2": "+ RAPF",
        "M3": "Full + calibration$^{\\dagger}$",
    }
    modules = {
        "M0": ("--", "--", "--"),
        "M1": ("$\\checkmark$", "--", "--"),
        "M2": ("$\\checkmark$", "$\\checkmark$", "--"),
        "M3": ("$\\checkmark$", "$\\checkmark$", "$\\checkmark$"),
    }
    lines = [
        "% Requires booktabs and amssymb.",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{3.2pt}",
        "  \\caption{Main reported configurations on ScanRefer. M0--M2 form the module-addition sequence. M3 is Stage154, a fixed two-source selector over Stage142 and Stage150 predictions, and is not a single-checkpoint result. All values are percentages; the best displayed value in each column is bold.}",
        "  \\label{tab:scanrefer_main_ablation}",
        "  \\begin{tabular}{@{}llccc@{\\hspace{4pt}}rrrrrr@{}}",
        "    \\toprule",
        "    ID & Setting & SACR & RAPF & QAHNL & \\multicolumn{2}{c}{Unique} & \\multicolumn{2}{c}{Multiple} & \\multicolumn{2}{c}{Overall} \\\\",
        "    \\cmidrule(lr){6-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11}",
        "     & & & & & @0.25 & @0.50 & @0.25 & @0.50 & @0.25 & @0.50 \\\\",
        "    \\midrule",
    ]
    for row in order:
        cells = latex_metric_cells(main_modules[row], metric_names, maxima)
        lines.append(
            "    {} & {} & {} & {} & {} & {} \\\\".format(
                row,
                settings[row],
                modules[row][0],
                modules[row][1],
                modules[row][2],
                " & ".join(cells),
            )
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\vspace{2pt}",
            "  \\begin{minipage}{0.99\\textwidth}",
            "    \\footnotesize M0 is the published BUTD-DETR result and is not retrained. M1 and M2 are independently trained strict-best checkpoints under the locked protocol. $^{\\dagger}$M3 uses fixed source-selection calibration; validation labels are not used to select its classifier or threshold.",
            "  \\end{minipage}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_internal_latex(rows, caption, label):
    metric_names = ("multiple_025", "multiple_050", "overall_025", "overall_050")
    maxima = {
        name: max(float(metrics[name]) for _, _, metrics in rows)
        for name in metric_names
    }
    lines = [
        "% Requires booktabs.",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "  \\caption{{{}}}".format(caption),
        "  \\label{{{}}}".format(label),
        "  \\begin{tabular}{@{}llrrrr@{}}",
        "    \\toprule",
        "    ID & Variant & \\multicolumn{2}{c}{Multiple} & \\multicolumn{2}{c}{Overall} \\\\",
        "    \\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        "     & & @0.25 & @0.50 & @0.25 & @0.50 \\\\",
        "    \\midrule",
    ]
    for row_id, variant, metrics in rows:
        cells = latex_metric_cells(metrics, metric_names, maxima)
        lines.append(
            "    {} & {} & {} \\\\".format(row_id, variant, " & ".join(cells))
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_latex_tables(result):
    sacr_labels = {
        "S0": "w/o target-attribute",
        "S1": "w/o relation-anchor",
        "S2": "w/o pairwise geometry",
        "S3": "hard top-1 anchor",
        "S4": "matched-protocol Full",
    }
    rapf_labels = {
        "R0": "fixed fusion ($g=0.1$)",
        "R1": "w/o query-quality cue",
        "R2": "w/o parser/anchor cues",
        "R3": "w/o gate supervision",
        "R4": "matched-protocol Full",
    }
    tables = {
        "TABLE_3_main_modules.tex": render_main_latex(result["main_modules"]),
        "TABLE_4_sacr_internal.tex": render_internal_latex(
            [
                (row, sacr_labels[row], result["sacr_internal"][row])
                for row in ("S0", "S1", "S2", "S3", "S4")
            ],
            "Internal ablation of SACR on ScanRefer. All rows retain RAPF and QAHNL and follow the locked training protocol. Multiple@0.25/0.50 and Overall@0.25/0.50 are percentages; the best displayed value in each column is bold.",
            "tab:scanrefer_sacr_ablation",
        ),
        "TABLE_5_rapf_internal.tex": render_internal_latex(
            [
                (row, rapf_labels[row], result["rapf_internal"][row])
                for row in ("R0", "R1", "R2", "R3", "R4")
            ],
            "Internal ablation of RAPF on ScanRefer. All rows retain SACR and QAHNL and follow the locked training protocol. Multiple@0.25/0.50 and Overall@0.25/0.50 are percentages; the best displayed value in each column is bold.",
            "tab:scanrefer_rapf_ablation",
        ),
    }
    tables["TABLES_scanrefer_ablations_all.tex"] = "\n".join(
        tables[name]
        for name in (
            "TABLE_3_main_modules.tex",
            "TABLE_4_sacr_internal.tex",
            "TABLE_5_rapf_internal.tex",
        )
    )
    return tables


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


def compute_internal_effects(rows, full_row):
    effects = {}
    full = rows[full_row]
    for row, metrics in rows.items():
        if row == full_row:
            continue
        row_effects = {}
        for metric in METRIC_NAMES:
            delta = float(full[metric]) - float(metrics[metric])
            row_effects[metric + "_full_minus_variant"] = delta
            row_effects[metric + "_full_not_lower"] = delta >= -1e-9
        effects[row] = row_effects
    return effects


def summarize_internal_claims(internal_effects):
    summary = {}
    for family, rows in internal_effects.items():
        for metric in METRIC_NAMES:
            summary[
                "{}_all_variants_full_not_lower_{}".format(family, metric)
            ] = all(
                payload[metric + "_full_not_lower"] for payload in rows.values()
            )
    return summary


def evidence_checks_markdown(main_monotonicity, internal_claim_summary):
    label_map = {
        "multiple_025": "Multiple@0.25",
        "multiple_050": "Multiple@0.50",
        "overall_025": "Overall@0.25",
        "overall_050": "Overall@0.50",
    }
    lines = [
        "## Evidence checks",
        "",
        "These checks report the observed evidence and never filter or replace a row.",
        "",
        "- Main Overall@0.25 non-decreasing: {}".format(
            "yes" if main_monotonicity["overall_025_non_decreasing"] else "no"
        ),
        "- Main Overall@0.50 non-decreasing: {}".format(
            "yes" if main_monotonicity["overall_050_non_decreasing"] else "no"
        ),
    ]
    for family in ("sacr", "rapf"):
        for metric in METRIC_NAMES:
            key = "{}_all_variants_full_not_lower_{}".format(family, metric)
            lines.append(
                "- {} Full not lower than every variant on {}: {}".format(
                    family.upper(),
                    label_map[metric],
                    "yes" if internal_claim_summary[key] else "no",
                )
            )
    return "\n".join(lines)


def assemble(
    manifest_path,
    row_paths,
    output_json,
    output_markdown,
    output_latex_dir=None,
):
    manifest = load_json(manifest_path)
    validate_manifest_provenance(manifest)
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
    internal_effects = {
        "rapf": compute_internal_effects(rapf_internal, "R4"),
        "sacr": compute_internal_effects(sacr_internal, "S4"),
    }
    internal_claim_summary = summarize_internal_claims(internal_effects)
    main_order = ("M0", "M1", "M2", "M3")
    main_monotonicity = {}
    for metric in ("overall_025", "overall_050"):
        values = [float(main_modules[row][metric]) for row in main_order]
        main_monotonicity[metric + "_values"] = values
        main_monotonicity[metric + "_deltas"] = [
            values[index] - values[index - 1] for index in range(1, len(values))
        ]
        main_monotonicity[metric + "_non_decreasing"] = all(
            values[index] >= values[index - 1] for index in range(1, len(values))
        )
    result = {
        "main_modules": main_modules,
        "main_monotonicity": main_monotonicity,
        "internal_claim_summary": internal_claim_summary,
        "internal_effects": internal_effects,
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
        "| ID | Setting | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
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
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row,
                main_labels[row],
                format_value(values["unique_025"]),
                format_value(values["unique_050"]),
                format_value(values["multiple_025"]),
                format_value(values["multiple_050"]),
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
        evidence_checks_markdown(main_monotonicity, internal_claim_summary),
    ]
    atomic_text(output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
    atomic_text(output_markdown, "\n\n".join(sections) + "\n")
    if output_latex_dir:
        os.makedirs(output_latex_dir, exist_ok=True)
        for name, content in render_latex_tables(result).items():
            atomic_text(os.path.join(output_latex_dir, name), content)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output_json")
    parser.add_argument("output_markdown")
    parser.add_argument("--output-latex-dir")
    parser.add_argument("row_results", nargs="+")
    args = parser.parse_args()
    result = assemble(
        args.manifest,
        args.row_results,
        args.output_json,
        args.output_markdown,
        output_latex_dir=args.output_latex_dir,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
