import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TUNING_DIR = ROOT / "scripts" / "new_method_v2" / "tuning"


def load_module(name, path):
    if os.fspath(TUNING_DIR) not in sys.path:
        sys.path.insert(0, os.fspath(TUNING_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTrial(object):
    def __init__(self, number):
        self.number = number
        self.user_attrs = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class FakeAskTellStudy(object):
    def __init__(self):
        self.trials = []
        self.tell_calls = 0

    def ask(self):
        trial = FakeTrial(9)
        self.trials.append(trial)
        return trial

    def tell(self, trial, value):
        self.tell_calls += 1
        raise ValueError("Cannot tell a FAIL trial.")


class TuningCheckpointPolicyTest(unittest.TestCase):
    def test_eval_checkpoint_alias_conflict_fails(self):
        module = load_module(
            "eval_sweep_conflict",
            TUNING_DIR / "scanrefer_two_stage_eval_fusion_sweep.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.pth"
            second = Path(tmp) / "second.pth"
            args = types.SimpleNamespace(
                checkpoint=os.fspath(first),
                checkpoint_path=os.fspath(second),
                dry_run=True,
            )
            with self.assertRaises(SystemExit):
                module.normalize_checkpoint_args(args)

    def test_eval_checkpoint_path_alias_matches_canonical(self):
        module = load_module(
            "eval_sweep_alias",
            TUNING_DIR / "scanrefer_two_stage_eval_fusion_sweep.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.pth"
            args = types.SimpleNamespace(
                checkpoint=os.fspath(checkpoint),
                checkpoint_path=os.fspath(checkpoint),
                dry_run=True,
            )
            module.normalize_checkpoint_args(args)
            self.assertEqual(Path(args.resolved_checkpoint_path), checkpoint.resolve())
            self.assertFalse(args.checkpoint_exists)

    def test_eval_missing_checkpoint_real_run_fails_with_exact_message(self):
        module = load_module(
            "eval_sweep_missing",
            TUNING_DIR / "scanrefer_two_stage_eval_fusion_sweep.py",
        )
        args = types.SimpleNamespace(
            checkpoint="/tmp/does-not-exist-for-eval-sweep.pth",
            checkpoint_path=None,
            dry_run=False,
        )
        with self.assertRaises(SystemExit) as raised:
            module.normalize_checkpoint_args(args)
        self.assertEqual(
            str(raised.exception),
            "Eval-only sweep requires an existing checkpoint. No fallback checkpoint is used.",
        )

    def test_eval_dry_run_accepts_missing_checkpoint_and_prints_policy(self):
        script = TUNING_DIR / "scanrefer_two_stage_eval_fusion_sweep.py"
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    os.fspath(script),
                    "--checkpoint",
                    os.fspath(Path(tmp) / "missing.pth"),
                    "--data-root",
                    "/tmp/data-root",
                    "--output-dir",
                    os.fspath(Path(tmp) / "out"),
                    "--dry-run",
                    "--max-combinations",
                    "1",
                ],
                cwd=os.fspath(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("dry_run = true", proc.stdout)
        self.assertIn("checkpoint existence not required", proc.stdout)

    def test_eval_wrapper_skips_without_checkpoint_and_prints_command(self):
        script = TUNING_DIR / "run_scanrefer_two_stage_eval_fusion_sweep.sh"
        env = os.environ.copy()
        env.pop("CHECKPOINT", None)
        proc = subprocess.run(
            ["bash", os.fspath(script), "--dry-run", "--max-combinations", "1"],
            cwd=os.fspath(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("Eval-only sweep skipped: no checkpoint was provided.", proc.stdout)
        self.assertIn("--checkpoint /path/to/existing_checkpoint.pth", proc.stdout)

    def test_optuna_missing_checkpoint_keeps_completed_trial_and_records_status(self):
        module = load_module(
            "optuna_policy",
            TUNING_DIR / "optuna_scanrefer_two_stage_full.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(
                output_root=os.fspath(Path(tmp) / "optuna"),
                max_epoch=15,
                data_root="/tmp/data-root",
                seed=0,
                objective_weight_025=0.5,
                objective_weight_050=0.5,
                full_script=module.default_full_script(),
                gpus="0",
            )
            params = {
                "rapf_struct_residual_clip": 0.25,
                "rapf_quality_weight": 0.5,
                "rapf_gate_loss_weight": 0.05,
                "rapf_initial_gate_bias": -2.5,
                "rapf_generic_gate_cap": 0.2,
            }
            run_dir = Path(args.output_root) / "trial_0000" / "scanrefer_spacy" / "123"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(
                """{
  "rapf_struct_residual_clip": 0.25,
  "rapf_quality_weight": 0.5,
  "rapf_gate_loss_weight": 0.05,
  "rapf_initial_gate_bias": -2.5,
  "rapf_generic_gate_cap": 0.2,
  "max_epoch": 15,
  "num_decoder_layers": 6,
  "lr": 0.0001,
  "lr_backbone": 0.001,
  "batch_size": 24
}""",
                encoding="utf-8",
            )
            (run_dir / "eval_epoch_15.log").write_text(
                "\n".join(
                    [
                        "last__bbs_acc0.25_top1: 0.6",
                        "last__bbs_acc0.50_top1: 0.4",
                        "last__bbf_acc0.25_top1: 0.55",
                        "last__bbf_acc0.50_top1: 0.35",
                        "last_bbs_vs_bbf_top1_disagree_ratio: 0.2",
                    ]
                ),
                encoding="utf-8",
            )
            trial = FakeTrial(0)
            fake_proc = types.SimpleNamespace(returncode=0, stdout="training ok")
            with mock.patch.object(module, "effective_command", return_value=("effective", "")):
                with mock.patch.object(module.subprocess, "run", return_value=fake_proc):
                    record = module.run_trial_command(args, trial, params)
        self.assertEqual(record["status"], "completed", record)
        self.assertEqual(record["checkpoint_status"], "missing")
        self.assertIn("checkpoint_status", module.fieldnames())

    def test_optuna_does_not_synthesize_effective_command_on_dry_run_failure(self):
        module = load_module(
            "optuna_no_synthetic_command",
            TUNING_DIR / "optuna_scanrefer_two_stage_full.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(
                output_root=os.fspath(Path(tmp) / "optuna"),
                multi_worker=False,
                worker_id="local",
                max_epoch=15,
                data_root="/tmp/data-root",
                seed=0,
                objective_weight_025=0.5,
                objective_weight_050=0.5,
                full_script=module.default_full_script(),
                gpus="0",
                study_name="scanrefer",
                storage="sqlite:///reports/tuning/test.db",
            )
            params = {
                "rapf_struct_residual_clip": 0.25,
                "rapf_quality_weight": 0.5,
                "rapf_gate_loss_weight": 0.05,
                "rapf_initial_gate_bias": -2.5,
                "rapf_generic_gate_cap": 0.2,
            }
            trial = FakeTrial(3)
            with mock.patch.object(
                module,
                "effective_command",
                side_effect=RuntimeError("full-script dry-run failed"),
            ):
                record = module.run_trial_command(args, trial, params)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["checkpoint_status"], "failed")
        self.assertEqual(record["effective_command"], "")
        self.assertIn("05_full_sacr_rapf_qahnl_scanrefer_2stage.sh", record["command"])
        self.assertIn("full-script dry-run failed", record["error_message"])

    def test_optuna_manual_loop_continues_when_tell_fails(self):
        module = load_module(
            "optuna_manual_loop",
            TUNING_DIR / "optuna_scanrefer_two_stage_full.py",
        )
        args = types.SimpleNamespace(n_trials=1)
        study = FakeAskTellStudy()
        params = {
            "rapf_struct_residual_clip": 0.25,
            "rapf_quality_weight": 0.5,
            "rapf_gate_loss_weight": 0.05,
            "rapf_initial_gate_bias": -2.5,
            "rapf_generic_gate_cap": 0.2,
        }
        record = {
            "trial_number": 9,
            "status": "completed",
            "objective": 0.42,
            "error_message": "",
        }
        with mock.patch.object(module, "params_from_trial", return_value=params):
            with mock.patch.object(module, "run_trial_command", return_value=record):
                with mock.patch.object(module, "collect_records", return_value=[]):
                    with mock.patch.object(module, "write_all_outputs"):
                        module.run_optuna_trials(args, study)
        self.assertEqual(study.tell_calls, 1)
        self.assertIn("failed to tell Optuna trial 9", record["error_message"])

    def test_rerun_resume_is_explicit_and_records_policy(self):
        module = load_module(
            "rerun_policy",
            TUNING_DIR / "rerun_top_optuna_scanrefer_two_stage.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            candidate = {
                "trial_number": 7,
                "seed": 0,
                "checkpoint_path": os.fspath(Path(tmp) / "trial_ckpt.pth"),
                "rapf_struct_residual_clip": 0.25,
                "rapf_quality_weight": 0.5,
                "rapf_gate_loss_weight": 0.05,
                "rapf_initial_gate_bias": -2.5,
                "rapf_generic_gate_cap": 0.2,
            }
            args = types.SimpleNamespace(
                output_root=os.fspath(Path(tmp) / "rerun"),
                max_epoch=30,
                data_root="/tmp/data-root",
                seed=0,
                resume_from_trial_checkpoint=False,
            )
            default_args = module.override_args(args, candidate, 1)
            forbidden = {"--checkpoint_path", "--resume", "--eval", "--eval_only"}
            self.assertTrue(forbidden.isdisjoint(set(default_args)))
            default_record = module.base_record(args, candidate, 1, "command")
            self.assertFalse(default_record["resume_from_trial_checkpoint"])
            self.assertEqual(default_record["checkpoint_status"], "not_required")

            args.resume_from_trial_checkpoint = True
            resume_args = module.override_args(args, candidate, 1)
            self.assertIn("--checkpoint_path", resume_args)
            self.assertIn(candidate["checkpoint_path"], resume_args)

    def test_storage_url_masking_and_sqlite_multi_worker_rejection(self):
        module = load_module(
            "optuna_multi_policy",
            TUNING_DIR / "optuna_scanrefer_two_stage_full.py",
        )
        self.assertEqual(
            module.mask_storage_url("postgresql://user:secret@db:5432/optuna"),
            "postgresql://user:***@db:5432/optuna",
        )
        self.assertEqual(
            module.mask_storage_url("mysql+pymysql://user:secret@db/optuna"),
            "mysql+pymysql://user:***@db/optuna",
        )
        self.assertEqual(
            module.mask_storage_url("sqlite:///reports/tuning/local.db"),
            "sqlite:///reports/tuning/local.db",
        )
        args = types.SimpleNamespace(
            multi_worker=True,
            storage="sqlite:///reports/tuning/bad.db",
            worker_id="worker01",
        )
        with self.assertRaises(SystemExit) as raised:
            module.validate_storage_policy(args)
        self.assertEqual(
            str(raised.exception),
            "SQLite storage is only supported for single-machine debugging. Use PostgreSQL/MySQL for multi-machine Optuna.",
        )

    def test_run_env_prepends_current_python_bin_to_path(self):
        module = load_module(
            "optuna_run_env_path",
            TUNING_DIR / "optuna_scanrefer_two_stage_full.py",
        )
        args = types.SimpleNamespace(data_root="/tmp/data-root", gpus="0")
        env = module.run_env(args)
        self.assertEqual(
            env["PATH"].split(os.pathsep)[0],
            os.fspath(Path(sys.executable).parent),
        )

    def test_multi_worker_log_dir_and_record_metadata(self):
        module = load_module(
            "optuna_multi_metadata",
            TUNING_DIR / "optuna_scanrefer_two_stage_full.py",
        )
        args = types.SimpleNamespace(
            output_root="logs/optuna",
            multi_worker=True,
            worker_id="worker01",
            gpus="0",
            study_name="scanrefer_multi",
            storage="postgresql://user:secret@db:5432/optuna",
            seed=0,
        )
        log_dir = module.trial_log_dir(args, 7)
        self.assertEqual(log_dir, Path("logs/optuna") / "worker01" / "trial_0007")
        params = {
            "rapf_struct_residual_clip": 0.25,
            "rapf_quality_weight": 0.5,
            "rapf_gate_loss_weight": 0.05,
            "rapf_initial_gate_bias": -2.5,
            "rapf_generic_gate_cap": 0.2,
        }
        record = module.base_record(args, 7, params, "effective")
        self.assertEqual(record["worker_id"], "worker01")
        self.assertTrue(record["hostname"])
        self.assertEqual(record["gpu"], "0")
        self.assertEqual(record["study_name"], "scanrefer_multi")
        self.assertEqual(
            record["storage_url_masked"],
            "postgresql://user:***@db:5432/optuna",
        )
        self.assertIn('"rapf_quality_weight": 0.5', record["params"])
        self.assertTrue(record["effective_command"])
        self.assertIn("python_executable", record)
        self.assertIn("code_version", record)

    def test_sqlite_multi_worker_dry_run_fails_before_optuna_connection(self):
        script = TUNING_DIR / "optuna_scanrefer_two_stage_full.py"
        proc = subprocess.run(
            [
                sys.executable,
                os.fspath(script),
                "--multi-worker",
                "--study-name",
                "bad_sqlite_multi",
                "--storage",
                "sqlite:///reports/tuning/bad.db",
                "--worker-id",
                "worker01",
                "--n-trials",
                "1",
                "--max-epoch",
                "1",
                "--dry-run",
            ],
            cwd=os.fspath(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn(
            "SQLite storage is only supported for single-machine debugging. Use PostgreSQL/MySQL for multi-machine Optuna.",
            proc.stdout,
        )

    def test_export_optuna_top5_dry_run_masks_storage(self):
        script = TUNING_DIR / "export_optuna_top5.py"
        proc = subprocess.run(
            [
                sys.executable,
                os.fspath(script),
                "--study-name",
                "dryrun_multi",
                "--storage",
                "postgresql://USER:PASSWORD@DB_HOST:5432/optuna_scanrefer",
                "--output-dir",
                "reports/tuning",
                "--dry-run",
            ],
            cwd=os.fspath(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn(
            "postgresql://USER:***@DB_HOST:5432/optuna_scanrefer",
            proc.stdout,
        )
        self.assertIn("optuna_scanrefer_two_stage_full_top5.json", proc.stdout)


if __name__ == "__main__":
    unittest.main()
