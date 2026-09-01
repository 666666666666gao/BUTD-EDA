#!/usr/bin/env python3
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/root/autodl-tmp/logs/butd_acc50_target_20260827').resolve()
RECEIPT = Path(
    '/home/gb/new butd/butd_detr-main/experiment_packages/'
    'butd_acc50_target_20260827/state/'
    'stage129_failed_weight_cleanup_20260830.json'
)
TARGETS = [
    ROOT / 'stage1_qahnl_iou50_universal_only/scanrefer_spacy/1787788445/ckpt_best_primary.pth',
    ROOT / 'stage2_qahnl_iou50_full_finetune/scanrefer_spacy/1787805248/ckpt_best_primary.pth',
    ROOT / 'stage4_quality_head_top5/scanrefer_spacy/1787813736/ckpt_best_primary.pth',
    ROOT / 'stage5_quality_head_logits_top5/scanrefer_spacy/1787816459/ckpt_best_primary.pth',
    ROOT / 'stage8_candidate_reranker/scanrefer_spacy/1787835118/ckpt_best_primary.pth',
    ROOT / 'stage9_candidate_reranker_wide/scanrefer_spacy/1787837476/ckpt_best_primary.pth',
    ROOT / 'stage10_dualthreshold_reranker/scanrefer_spacy/1787842259/ckpt_best_primary.pth',
    ROOT / 'stage12_textcid_reranker/scanrefer_spacy/1787846799/ckpt_best_primary.pth',
    ROOT / 'stage14c_rescue_gate_stable/scanrefer_spacy/1787858656/ckpt_best_primary.pth',
    ROOT / 'stage16_hit50_rank_augmented/scanrefer_spacy/1787883778/ckpt_best_primary.pth',
    ROOT / 'stage19_clean_conservative_geometry/scanrefer_spacy/1787903793/ckpt_best_primary.pth',
    ROOT / 'stage20_selected_query_geometry/scanrefer_spacy/1787907404/ckpt_best_primary.pth',
]
PRESERVED = [
    ROOT / 'stage18_geometry_action_head/scanrefer_spacy/1787892530/ckpt_best_primary.pth',
    ROOT / 'stage95_targeted_last_box_nojitter/scanrefer_spacy/1788017622/ckpt_best_primary.pth',
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_receipt(payload):
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    tmp = RECEIPT.with_suffix(RECEIPT.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, RECEIPT)


for path in TARGETS + PRESERVED:
    resolved = path.resolve(strict=True)
    if os.path.commonpath([str(ROOT), str(resolved)]) != str(ROOT):
        raise RuntimeError(f'path escapes cleanup root: {resolved}')
    if resolved.name != 'ckpt_best_primary.pth':
        raise RuntimeError(f'unexpected target filename: {resolved}')
if set(TARGETS) & set(PRESERVED):
    raise RuntimeError('cleanup target overlaps preserved checkpoint')

deleted = []
for path in TARGETS:
    deleted.append({
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256(path),
        'reason': 'failed optimization checkpoint; logs/config/metrics retained',
    })
preserved = []
for path in PRESERVED:
    preserved.append({
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256(path),
        'reason': 'Stage18 reproduction source' if 'stage18_' in str(path)
                  else 'current formal best Stage95',
    })

payload = {
    'status': 'planned',
    'created_at': datetime.now(timezone.utc).astimezone().isoformat(),
    'cleanup_root': str(ROOT),
    'deleted': deleted,
    'preserved': preserved,
    'bytes_recovered': 0,
}
atomic_receipt(payload)
for item in deleted:
    Path(item['path']).unlink()
payload['status'] = 'complete'
payload['completed_at'] = datetime.now(timezone.utc).astimezone().isoformat()
payload['bytes_recovered'] = sum(item['bytes'] for item in deleted)
payload['all_deleted_absent'] = all(not Path(item['path']).exists() for item in deleted)
payload['all_preserved_present'] = all(Path(item['path']).is_file() for item in preserved)
atomic_receipt(payload)
print(json.dumps({
    'status': payload['status'],
    'deleted_count': len(deleted),
    'bytes_recovered': payload['bytes_recovered'],
    'all_deleted_absent': payload['all_deleted_absent'],
    'all_preserved_present': payload['all_preserved_present'],
    'receipt': str(RECEIPT),
}, sort_keys=True))
