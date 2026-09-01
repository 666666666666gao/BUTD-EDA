#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/gb/new butd/butd_detr-main}"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"

for script in "${PACKAGE_ROOT}"/*.sh "${PACKAGE_ROOT}"/launch/*.sh; do
  bash -n "${script}"
done

"${PYTHON}" -m py_compile \
  "${PACKAGE_ROOT}/module_ablation_smoke.py" \
  "${PACKAGE_ROOT}/record_result.py" \
  "${PACKAGE_ROOT}/register_known_results.py" \
  "${PACKAGE_ROOT}/validate_dry_runs.py"

cd "${REPO_ROOT}"
"${PYTHON}" "${PACKAGE_ROOT}/validate_dry_runs.py"
"${PYTHON}" "${PACKAGE_ROOT}/module_ablation_smoke.py"
"${PYTHON}" train_dist_mod.py --help | grep -q -- '--rapf_disable_parser_anchor_cues'
"${PYTHON}" - "${PACKAGE_ROOT}/state/plan_manifest.json" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1]))
assert plan['dataset'] == 'scanrefer_spacy'
assert plan['seed_policy'] == 'seed0_only_no_replication'
assert plan['baseline']['overall_025'] == 50.42
assert plan['main_full']['overall_025'] == 54.3962978544
assert plan['main_full']['rapf_quality_weight'] == 0.25
assert plan['matched_protocol_full']['overall_025'] == 53.7547328565
assert plan['matched_protocol_full']['rapf_quality_weight'] == 0.75
assert plan['unique_table_configurations'] == 13
assert plan['trainable_seed0_configurations'] == 10
assert plan['reused_from_old_queue'] == ['S1']
assert plan['main_receipts_from_monotonic_queue'] == ['M1', 'M2']
assert plan['new_internal_queue_rows'] == ['S2', 'S0', 'S3', 'R1', 'R3', 'R0', 'R2']
PY

echo THREE_TABLE_PREFLIGHT_PASS
