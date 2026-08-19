import os
import importlib.util
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "new_method_v2"
    / "scanrefer"
    / "two_stage"
    / "12_full_quality_primary_scanrefer_2stage.sh"
)


def load_validator():
    path = ROOT / "scripts" / "validate_new_method_scripts.py"
    spec = importlib.util.spec_from_file_location("validate_new_method_scripts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScanReferQualityPrimaryScriptTest(unittest.TestCase):
    def test_dry_run_uses_quality_primary_scoring_for_full_model(self):
        env = os.environ.copy()
        env["MASTER_PORT"] = "29661"
        env["DIAG"] = "1"
        env["NMV2_MAX_EPOCH"] = "110"
        proc = subprocess.run(
            [
                "bash",
                os.fspath(SCRIPT),
                "--dry-run",
                "--rapf_struct_residual_clip",
                "0.25",
                "--rapf_quality_weight",
                "0.75",
                "--rapf_gate_loss_weight",
                "0.1",
                "--rapf_initial_gate_bias",
                "-2.5",
                "--rapf_generic_gate_cap",
                "0.1",
            ],
            cwd=os.fspath(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("--eval_use_quality_scores", proc.stdout)
        self.assertNotIn("--eval_use_fused_scores", proc.stdout)
        for flag in (
            "--use_structured_slots",
            "--use_sacr",
            "--use_rapf",
            "--use_reliability_gate",
            "--use_quality_head",
            "--rapf_use_quality",
            "--use_qahnl",
            "--verbose_diagnostics",
            "--eval_report_diagnostic_scores",
        ):
            self.assertIn(flag, proc.stdout)
        self.assertIn("12_full_quality_primary", proc.stdout)
        self.assertIn("--max_epoch 110", proc.stdout)

    def test_validator_accepts_quality_primary_script_contract(self):
        validator = load_validator()
        result = validator.validate_scripts()
        self.assertEqual(result["status"], "pass", result.get("failures"))
        matches = [
            entry
            for entry in result["scripts"]
            if entry["path"].endswith(
                "scanrefer/two_stage/12_full_quality_primary_scanrefer_2stage.sh"
            )
        ]
        self.assertEqual(len(matches), 1, result["scripts"])
        module_args = [" ".join(arg) for arg in matches[0]["module_args"]]
        self.assertIn("--eval_use_quality_scores", module_args)
        self.assertNotIn("--eval_use_fused_scores", module_args)


if __name__ == "__main__":
    unittest.main()
