import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path("scripts/ablations/scanrefer_20260814")
SCRIPTS = {
    "baseline": ROOT / "01_baseline_scanrefer_20260814.sh",
    "full": ROOT / "02_full_scanrefer_20260814.sh",
    "no_sacr_rapf": ROOT / "03_no_sacr_rapf_scanrefer_20260814.sh",
    "no_qahnl": ROOT / "04_no_qahnl_scanrefer_20260814.sh",
    "no_quality": ROOT / "05_no_quality_scanrefer_20260814.sh",
    "no_gate": ROOT / "06_no_gate_supervision_scanrefer_20260814.sh",
    "no_relation": ROOT / "07_no_relation_scanrefer_20260814.sh",
}


def _tokens(script):
    env = dict(os.environ, DRY_RUN="1", NMV2_MAX_EPOCH="1")
    output = subprocess.check_output(
        ["bash", str(script)], env=env, universal_newlines=True
    )
    return shlex.split(output)


def _has(tokens, flag):
    return flag in tokens


def _value(tokens, flag):
    return tokens[tokens.index(flag) + 1]


def test_every_ablation_uses_identical_official_init_and_no_resume():
    for script in SCRIPTS.values():
        tokens = _tokens(script)
        assert "--checkpoint_path" not in tokens
        assert _value(tokens, "--pp_checkpoint") == (
            "/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth"
        )
        assert _value(tokens, "--rng_seed") == "0"
        assert _value(tokens, "--dataset") == "scanrefer_spacy"
        assert _value(tokens, "--test_dataset") == "scanrefer_spacy"
        assert _has(tokens, "--best_checkpoint_only")
        assert _value(tokens, "--best_checkpoint_metric") == (
            "last__bbs_acc0.25_top1"
        )
        assert _has(tokens, "--early_stopping")
        assert _value(tokens, "--early_stopping_metric") == (
            "last__bbs_acc0.25_top1"
        )
        assert _value(tokens, "--early_stopping_min_epoch") == "35"
        assert _value(tokens, "--early_stopping_patience") == "4"
        assert _value(tokens, "--early_stopping_min_delta") == "0.001"


def test_component_matrix_isolated_as_documented():
    rows = {name: _tokens(path) for name, path in SCRIPTS.items()}
    module_flags = {
        "--use_sacr", "--use_rapf", "--use_reliability_gate",
        "--use_quality_head", "--rapf_use_quality", "--use_qahnl",
        "--sacr_disable_relation", "--eval_use_fused_scores",
    }
    assert not module_flags.intersection(rows["baseline"])

    for name in ("full", "no_qahnl", "no_quality", "no_gate", "no_relation"):
        assert _has(rows[name], "--use_sacr")
        assert _has(rows[name], "--use_rapf")
        assert _has(rows[name], "--eval_use_fused_scores")

    assert _has(rows["full"], "--use_qahnl")
    assert not _has(rows["no_qahnl"], "--use_qahnl")
    assert _has(rows["no_qahnl"], "--use_quality_head")

    assert not _has(rows["no_quality"], "--use_quality_head")
    assert not _has(rows["no_quality"], "--rapf_use_quality")
    assert _has(rows["no_quality"], "--use_qahnl")

    assert _has(rows["no_gate"], "--use_reliability_gate")
    assert _value(rows["no_gate"], "--rapf_gate_loss_weight") == "0"
    assert _value(rows["full"], "--rapf_gate_loss_weight") == "0.1"

    full_flags = set(rows["full"])
    relation_flags = set(rows["no_relation"])
    assert "--sacr_disable_relation" not in full_flags
    assert "--sacr_disable_relation" in relation_flags

    assert _has(rows["no_sacr_rapf"], "--use_qahnl")
    assert _value(rows["no_sacr_rapf"], "--qahnl_score_source") == "base"
    assert not _has(rows["no_sacr_rapf"], "--use_sacr")
    assert not _has(rows["no_sacr_rapf"], "--use_rapf")
