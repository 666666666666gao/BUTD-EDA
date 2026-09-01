#!/usr/bin/env python3
import hashlib
import json
import os
import sys
from pathlib import Path


baseline_path, candidate_path, receipt_path = map(Path, sys.argv[1:4])
baseline = json.loads(baseline_path.read_text())
candidate = json.loads(candidate_path.read_text())
missing = sorted(set(baseline) - set(candidate))
different = {
    key: [baseline[key], candidate.get(key)]
    for key in baseline
    if key in candidate and baseline[key] != candidate[key]
}
if missing or different:
    raise AssertionError({'missing': missing, 'different': different})


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


receipt = {
    'stage': 130,
    'status': 'zero_init_smoke_parity_pass',
    'samples': 96,
    'shared_metric_count': len(baseline),
    'acc025': candidate['last__bbs_acc0.25_top1'],
    'acc050': candidate['last__bbs_acc0.50_top1'],
    'baseline_metrics_sha256': sha256(baseline_path),
    'candidate_metrics_sha256': sha256(candidate_path),
    'all_baseline_metrics_exact_match': True,
    'new_candidate_fields': sorted(set(candidate) - set(baseline)),
}
receipt_path.parent.mkdir(parents=True, exist_ok=True)
tmp = receipt_path.with_suffix(receipt_path.suffix + '.tmp')
tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
os.replace(tmp, receipt_path)
print(json.dumps(receipt, indent=2, sort_keys=True))
