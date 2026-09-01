#!/usr/bin/env python3
"""Execute the real Stage-0 inline Python blocks against synthetic receipts."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile


PACKAGE_ROOT = Path(__file__).resolve().parent
RUN_SCRIPT = PACKAGE_ROOT / "run_text_policy_diagnostic.sh"
VERIFY_SCRIPT = PACKAGE_ROOT / "verify_stage0_reload.sh"
FINALIZER_SCRIPT = PACKAGE_ROOT / "watch_finalize_goal.sh"
WEIGHTS = (
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00,
)


def _extract_heredoc(path, needle):
    text = path.read_text(encoding="utf-8")
    anchor = text.index(needle)
    body_start = text.index("<<'PY'", anchor)
    body_start = text.index("\n", body_start) + 1
    body_end = text.index("\nPY\n", body_start)
    return text[body_start:body_end]


def _execute(code, argv, label):
    previous = sys.argv
    sys.argv = list(argv)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, label, "exec"), {"__name__": "__main__"})
    finally:
        sys.argv = previous


def _metric_payload(overrides):
    payload = {}
    for weight in WEIGHTS:
        source = "rapf_qw_{:03d}".format(int(round(weight * 100.0)))
        acc025, acc050 = overrides.get(weight, (0.5300, 0.4100))
        payload["diag_{}@0.25".format(source)] = acc025
        payload["diag_{}@0.50".format(source)] = acc050
    return payload


def main():
    selection_code = _extract_heredoc(
        RUN_SCRIPT,
        '"${PYTHON}" - "${RESULT_JSON}" "${GRID_SELECTION}"',
    )
    negative_receipt_code = _extract_heredoc(
        RUN_SCRIPT,
        '"${PYTHON}" - "${M3_CHECKPOINT}" "${GRID_SELECTION}"',
    )
    verification_code = _extract_heredoc(
        VERIFY_SCRIPT,
        '"${PYTHON}" - "${RESULT_JSON}" "${CHECKPOINT}" "${RECEIPT}"',
    )
    finalizer_code = _extract_heredoc(
        FINALIZER_SCRIPT,
        '"${PYTHON}" - "${WINNER}" "${WINNER_RECEIPT}" "${FINAL_RECEIPT}" "${HANDOFF}"',
    )

    with tempfile.TemporaryDirectory(prefix="stage0_receipt_test_") as tmp:
        root = Path(tmp)
        checkpoint = root / "checkpoint.pth"
        checkpoint.write_bytes(b"stage0-checkpoint-contract\n" * 64)
        expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

        # Both 0.25 and 0.50 are feasible, but 0.25 must win because it
        # preserves Acc@0.25 better. Exact equality at 0.5391 is infeasible.
        result = root / "feasible_results.json"
        selection = root / "feasible_selection.json"
        result.write_text(
            json.dumps(_metric_payload({
                0.00: (0.5391, 0.5000),
                0.25: (0.5440, 0.4250),
                0.50: (0.5410, 0.4500),
            })),
            encoding="utf-8",
        )
        _execute(selection_code, ["-", str(result), str(selection)], "selection")
        chosen = json.loads(selection.read_text(encoding="utf-8"))["selected"]
        assert chosen["goal_achieved"] is True
        assert chosen["rapf_quality_weight"] == 0.25
        assert chosen["overall_acc0.25"] == 0.5440

        # No feasible coefficient: the real negative-receipt block must keep
        # the closest candidate, mark it non-independent, and hash the source.
        failed_result = root / "failed_results.json"
        failed_selection = root / "failed_selection.json"
        failed_receipt = root / "failed_receipt.json"
        failed_result.write_text(
            json.dumps(_metric_payload({
                0.25: (0.5420, 0.4230),
                0.50: (0.5300, 0.4240),
            })),
            encoding="utf-8",
        )
        _execute(
            selection_code,
            ["-", str(failed_result), str(failed_selection)],
            "failed_selection",
        )
        failed_chosen = json.loads(
            failed_selection.read_text(encoding="utf-8")
        )["selected"]
        assert failed_chosen["rapf_quality_weight"] == 0.25
        _execute(
            negative_receipt_code,
            ["-", str(checkpoint), str(failed_selection), str(failed_receipt)],
            "negative_receipt",
        )
        negative = json.loads(failed_receipt.read_text(encoding="utf-8"))
        assert negative["goal_achieved"] is False
        assert negative["independent_full_reload"] is False
        assert negative["checkpoint_sha256"] == expected_sha
        assert negative["rapf_quality_weight"] == 0.25

        # The independent verifier must preserve the chosen coefficient,
        # enforce strict thresholds, and hash the exact checkpoint.
        verify_result = root / "verify_results.json"
        verify_receipt = root / "verify_receipt.json"
        verify_result.write_text(
            json.dumps({
                "last__bbs_acc0.25_top1": 0.5440,
                "last__bbs_acc0.50_top1": 0.4250,
            }),
            encoding="utf-8",
        )
        _execute(
            verification_code,
            [
                "-", str(verify_result), str(checkpoint),
                str(verify_receipt), "0.25", str(selection),
            ],
            "verification_receipt",
        )
        verified = json.loads(verify_receipt.read_text(encoding="utf-8"))
        assert verified["goal_achieved"] is True
        assert verified["independent_full_reload"] is True
        assert verified["checkpoint_sha256"] == expected_sha
        assert verified["rapf_quality_weight"] == 0.25

        equality_result = root / "equality_results.json"
        equality_receipt = root / "equality_receipt.json"
        equality_result.write_text(
            json.dumps({
                "last__bbs_acc0.25_top1": 0.5391,
                "last__bbs_acc0.50_top1": 0.5000,
            }),
            encoding="utf-8",
        )
        _execute(
            verification_code,
            [
                "-", str(equality_result), str(checkpoint),
                str(equality_receipt), "0.25", str(selection),
            ],
            "strict_equality_receipt",
        )
        equality = json.loads(equality_receipt.read_text(encoding="utf-8"))
        assert equality["pass_acc0.25"] is False
        assert equality["goal_achieved"] is False

        # Execute the deployed finalizer block itself. It must retain the
        # coefficient in the atomic receipt and the human-readable handoff.
        source_receipt = root / "source_receipt.json"
        final_receipt = root / "final_receipt.json"
        handoff = root / "handoff.md"
        source_receipt.write_text(
            json.dumps(verified),
            encoding="utf-8",
        )
        handoff.write_text("# synthetic handoff\n", encoding="utf-8")
        _execute(
            finalizer_code,
            [
                "-", "stage0", str(source_receipt),
                str(final_receipt), str(handoff),
            ],
            "finalizer",
        )
        finalized = json.loads(final_receipt.read_text(encoding="utf-8"))
        assert finalized["winner_stage"] == "stage0"
        assert finalized["goal_achieved"] is True
        assert finalized["rapf_quality_weight"] == 0.25
        assert finalized["checkpoint_sha256"] == expected_sha
        handoff_text = handoff.read_text(encoding="utf-8")
        assert "RAPF quality weight: `0.2500`" in handoff_text
        assert "Reloaded Overall Acc@0.25 / Acc@0.50" in handoff_text

        invalid_source = dict(verified)
        invalid_source["overall_acc0.25"] = 0.5391
        invalid_source["goal_achieved"] = True
        invalid_source_receipt = root / "invalid_source_receipt.json"
        invalid_source_receipt.write_text(
            json.dumps(invalid_source),
            encoding="utf-8",
        )
        try:
            _execute(
                finalizer_code,
                [
                    "-", "stage0", str(invalid_source_receipt),
                    str(root / "invalid_final.json"), str(handoff),
                ],
                "invalid_finalizer",
            )
        except SystemExit as exc:
            assert "strict thresholds" in str(exc)
        else:
            raise AssertionError("finalizer accepted an equality-threshold receipt")

    print("STAGE0_RECEIPT_PROTOCOL_PASS")


if __name__ == "__main__":
    main()
