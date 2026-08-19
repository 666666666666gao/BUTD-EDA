import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path("scripts/ablations/scanrefer_20260815_extension")
SCRIPTS = {
    "sacr": ROOT / "08_sacr_only_scanrefer_20260815.sh",
    "sacr_qahnl": ROOT / "09_sacr_qahnl_scanrefer_20260815.sh",
    "full_qahnl_base": ROOT / "10_full_qahnl_base_source_scanrefer_20260815.sh",
}


def tokens(path):
    env = dict(os.environ, DRY_RUN="1", NMV2_MAX_EPOCH="100")
    return shlex.split(subprocess.check_output(
        ["bash", str(path)], env=env, universal_newlines=True
    ))


def has(row, flag):
    return flag in row


def value(row, flag):
    return row[row.index(flag) + 1]


def test_shared_protocol_is_independent_from_official_initialization():
    for path in SCRIPTS.values():
        row = tokens(path)
        assert "--checkpoint_path" not in row
        assert value(row, "--pp_checkpoint") == "/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth"
        assert value(row, "--rng_seed") == "0"
        assert value(row, "--max_epoch") == "100"
        assert value(row, "--val_freq") == "5"
        assert value(row, "--dataset") == "scanrefer_spacy"
        assert value(row, "--test_dataset") == "scanrefer_spacy"
        assert has(row, "--best_checkpoint_only")
        assert value(row, "--best_checkpoint_metric") == "last__bbs_acc0.25_top1"
        assert has(row, "--early_stopping")
        assert value(row, "--early_stopping_metric") == "last__bbs_acc0.25_top1"
        assert value(row, "--early_stopping_min_epoch") == "35"
        assert value(row, "--early_stopping_patience") == "4"
        assert value(row, "--early_stopping_min_delta") == "0.001"


def test_dependency_aware_component_matrix():
    sacr = tokens(SCRIPTS["sacr"])
    sacr_q = tokens(SCRIPTS["sacr_qahnl"])
    full_base = tokens(SCRIPTS["full_qahnl_base"])

    assert has(sacr, "--use_structured_slots")
    assert has(sacr, "--use_sacr")
    assert has(sacr, "--eval_use_structured_scores")
    assert not has(sacr, "--use_rapf")
    assert not has(sacr, "--use_qahnl")

    assert has(sacr_q, "--use_sacr")
    assert has(sacr_q, "--use_qahnl")
    assert value(sacr_q, "--qahnl_score_source") == "structured"
    assert has(sacr_q, "--eval_use_structured_scores")
    assert not has(sacr_q, "--use_rapf")

    for flag in (
        "--use_structured_slots", "--use_sacr", "--use_rapf",
        "--use_reliability_gate", "--use_quality_head",
        "--rapf_use_quality", "--use_qahnl", "--eval_use_fused_scores",
    ):
        assert has(full_base, flag)
    assert value(full_base, "--qahnl_score_source") == "base"
    assert value(full_base, "--rapf_gate_loss_weight") == "0.1"


def test_no_invalid_rapf_without_sacr_configuration():
    for path in SCRIPTS.values():
        row = tokens(path)
        assert not has(row, "--use_rapf") or has(row, "--use_sacr")
