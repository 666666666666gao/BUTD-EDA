#!/bin/bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${PACKAGE_ROOT}/external_butd_paper_baseline.json}"
/root/miniconda3/envs/bdetr/bin/python - "${OUT}" <<'PY'
import json
import os
import sys
from pathlib import Path

out = Path(sys.argv[1])
payload = {
    "job_id": "01_baseline",
    "source_type": "external_paper",
    "method": "BUTD-DETR",
    "citation": "Jain et al., ECCV 2022",
    "source_url": "https://arxiv.org/abs/2112.08879",
    "source_tables": ["Table 1", "Supplementary Table 8"],
    "dataset": "ScanRefer validation",
    "protocol_note": "External reference used ground-truth text labels; do not retrain as part of this package.",
    "metrics_percent": {
        "unique_acc025": 84.20,
        "unique_acc050": 66.30,
        "multiple_acc025": 46.60,
        "multiple_acc050": 35.10,
        "overall_acc025": 52.20,
        "overall_acc050": 39.80
    }
}
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_name(out.name + ".tmp.{}".format(os.getpid()))
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(str(tmp), str(out))
print(out)
PY

