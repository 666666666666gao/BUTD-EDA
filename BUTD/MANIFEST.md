# Research Output Manifest

> Auto-maintained by ARIS skills. Tracks all generated artifacts across the research lifecycle.

| Timestamp | Skill | File | Stage | Description |
|-----------|-------|------|-------|-------------|
| 2026-08-20 03:25 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260820_032500.md | implementation | BUTD three-dataset strict-target experiment plan |
| 2026-08-20 03:25 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | latest copy |
| 2026-08-20 03:25 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260820_032500.md | implementation | BUTD three-dataset experiment tracker |
| 2026-08-20 03:25 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest copy |
| 2026-08-20 03:37 | /run-experiment | experiment_packages/butd_three_targets_20260820/common.sh | implementation | shared runtime, boundary audit, and exact acceptance recording |
| 2026-08-20 03:37 | /run-experiment | experiment_packages/butd_three_targets_20260820/01_scanrefer_to_target.sh | implementation | ScanRefer strict-target calibration and fallback micro-tune launcher |
| 2026-08-20 03:37 | /run-experiment | experiment_packages/butd_three_targets_20260820/02_nr3d_to_target.sh | implementation | independent Nr3D strict-target launcher |
| 2026-08-20 03:37 | /run-experiment | experiment_packages/butd_three_targets_20260820/03_sr3d_to_target.sh | implementation | independent Sr3D strict-target launcher |
| 2026-08-20 03:37 | /run-experiment | experiment_packages/butd_three_targets_20260820/run_three_targets.sh | implementation | serial three-target orchestrator |
| 2026-08-20 03:37 | /run-experiment | experiment_packages/butd_three_targets_20260820/targets.json | implementation | machine-readable strict targets and registered calibration values |
| 2026-08-20 03:37 | /run-experiment | experiment_packages/butd_three_targets_20260820/README.md | implementation | reproducible launch order and paper boundary |
| 2026-08-20 03:37 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260820_033700.md | implementation | corrected version recording dataset-level calibration values exactly |
| 2026-08-20 03:37 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | latest copy |
| 2026-08-20 10:10 | /run-experiment | experiment_packages/butd_three_targets_20260820/watch_sr3d_target.sh | implementation | 120-second durable receipt watcher that stops Sr3D after strict target and aggregates acceptance |
| 2026-08-20 12:05 | /run-experiment | experiment_packages/butd_three_targets_20260820/final_completion_audit.py | implementation | deterministic verifier for thresholds, SHA, checkpoint loading, module boundary, configs, and weight cleanup |
| 2026-08-20 12:05 | /run-experiment | experiment_packages/butd_three_targets_20260820/state/final_completion_audit.json | implementation | machine-verifiable final three-target completion receipt |
| 2026-08-20 12:16 | /run-experiment | refine-logs/EXPERIMENT_TRACKER_20260820_121654.md | evaluation | final accepted three-target results, scope qualifiers, checkpoints, and cleanup evidence |
| 2026-08-20 12:16 | /run-experiment | refine-logs/EXPERIMENT_TRACKER.md | evaluation | latest accepted three-target experiment tracker |
| 2026-08-20 12:16 | /run-experiment | experiment_packages/butd_three_targets_20260820/README.md | implementation | final reproducibility instructions and exact BUTD/BUTD-CLS result scope |
| 2026-08-20 12:20 | /experiment-audit | refine-logs/EXPERIMENT_AUDIT_20260820_122000.md | evaluation | independent integrity audit with remediated PASS verdict |
| 2026-08-20 12:20 | /experiment-audit | refine-logs/EXPERIMENT_AUDIT.md | evaluation | latest human-readable experiment integrity audit |
| 2026-08-20 12:20 | /experiment-audit | refine-logs/EXPERIMENT_AUDIT_20260820_122000.json | evaluation | machine-readable independent experiment integrity audit |
| 2026-08-20 12:20 | /experiment-audit | refine-logs/EXPERIMENT_AUDIT.json | evaluation | latest machine-readable experiment integrity audit |
| 2026-08-20 12:20 | /experiment-audit | .aris/traces/experiment-audit/2026-08-20_run01/ | evaluation | full independent reviewer request/response trace |
| 2026-08-20 14:10 | /run-experiment | experiment_packages/butd_same_module_optimization_20260819/continue_ablation_queue_from_row03.sh | implementation | fail-closed row03 epoch-10 resume, strict-best seeding, canonical consolidation, and serial queue handoff |
| 2026-08-20 14:10 | /run-experiment | reports/tuning/scanrefer_ablation_20260815/ABLATION_RESUME_PROTOCOL_20260820.md | implementation | registered same-row resume boundary and reproducibility disclosure |
| 2026-08-20 14:10 | /run-experiment | reports/tuning/scanrefer_ablation_20260815/ABLATION_PLAN.md | implementation | corrected fixed 65-epoch/13-validation protocol and row03 resume disclosure |
| 2026-08-20 14:10 | /run-experiment | tools/audit_scanrefer_ablation_completion.py | evaluation | audits disclosed row03 same-run continuation without allowing external warm starts |
| 2026-08-20 14:10 | /run-experiment | tools/audit_scanrefer_ablation_master_completion.py | evaluation | master gate accepts exactly one declared same-row epoch-10 continuation |
| 2026-08-20 14:10 | /run-experiment | experiment_packages/butd_same_module_optimization_20260819/README.md | implementation | formal ablation continuation instructions |
