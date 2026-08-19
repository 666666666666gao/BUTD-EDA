import hashlib
import sys
from types import SimpleNamespace

import torch

import train_dist_mod


def _args(**overrides):
    values = dict(
        use_color=True,
        use_height=False,
        use_multiview=False,
        use_soft_token_loss=True,
        num_target=256,
        num_decoder_layers=6,
        self_position_embedding="loc_learned",
        use_contrastive_align=True,
        butd=True,
        butd_gt=False,
        butd_cls=False,
        pp_checkpoint="/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth",
        self_attend=True,
        use_structured_slots=False,
        use_late_acd=False,
        slot_pooling="attention",
        max_rel_anchor_pairs=3,
        acd_top_m_targets=32,
        acd_top_k_anchors=16,
        acd_geo_dim=16,
        acd_hidden_dim=288,
        acd_global_residual_alpha=0.5,
        acd_use_confidence_fusion=False,
        acd_warmup_steps=5000,
        acd_initial_alpha=0.05,
        acd_ea_scale=1.0,
        acd_pool_ea_multiplier=1.0,
        acd_final_ea_multiplier=1.0,
        acd_disable_struct_rerank=False,
        acd_base_score_source="contrastive",
        dhc_margin_min=0.0,
        dhc_temperature_max=0.0,
        structured_debug=False,
        use_quality_head=False,
        use_sacr=False,
        sacr_top_m_targets=32,
        sacr_top_k_anchors=16,
        sacr_hidden_dim=288,
        sacr_geo_dim=16,
        sacr_disable_relation=False,
        use_rapf=False,
        rapf_hidden_dim=128,
        rapf_initial_gate_bias=-2.5,
        rapf_use_quality=False,
        rapf_quality_weight=0.75,
        rapf_struct_residual_clip=0.25,
        rapf_generic_gate_cap=0.1,
        rapf_quality_anchor_structured_residual=False,
        use_qahnl=False,
        qahnl_score_source="fused",
        use_source_pool_selector=False,
        source_pool_selector_hidden_dim=288,
        source_pool_selector_candidate_aware=False,
        source_pool_selector_direct_choice=False,
        source_pool_selector_include_contrastive_choice=False,
        source_pool_selector_rank_features=False,
        source_pool_selector_pairdelta_features=False,
        source_pool_selector_candidate_context=False,
        source_pool_selector_candidate_context_k=5,
        source_pool_selector_include_detector_policy_choice=False,
        source_pool_selector_choice_sources=None,
        source_pool_selector_text_context=False,
        source_pool_selector_metadata_context=False,
        source_pool_selector_context_features=False,
        source_pool_selector_separate_override_head=False,
        source_pool_selector_override_initial_bias=-1.5,
        use_detector_policy_adapter=False,
        use_detector_policy_teacher=False,
        detector_policy_adapter_context=False,
        detector_policy_adapter_hidden_dim=32,
        detector_policy_adapter_delta_scale=0.25,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _digest(state, keys):
    sha = hashlib.sha256()
    for key in sorted(keys):
        sha.update(key.encode())
        sha.update(state[key].detach().cpu().numpy().tobytes())
    return sha.hexdigest()


def main():
    configs = {
        "baseline": _args(),
        "full": _args(
            use_structured_slots=True,
            use_sacr=True,
            use_rapf=True,
            use_quality_head=True,
            rapf_use_quality=True,
            use_qahnl=True,
        ),
        "no_quality": _args(
            use_structured_slots=True,
            use_sacr=True,
            use_rapf=True,
            use_quality_head=False,
            use_qahnl=True,
        ),
        "no_relation": _args(
            use_structured_slots=True,
            use_sacr=True,
            sacr_disable_relation=True,
            use_rapf=True,
            use_quality_head=True,
            rapf_use_quality=True,
            use_qahnl=True,
        ),
    }
    states = {}
    for name, args in configs.items():
        train_dist_mod.configure_reproducibility(0)
        states[name] = train_dist_mod.TrainTester.get_model(args).state_dict()

    optional_prefixes = (
        "structured_slot_builder.", "sacr_head.",
        "reliability_fusion.", "quality_head.",
    )
    shared = {
        key for key in states["baseline"]
        if not key.startswith(optional_prefixes)
    }
    baseline_digest = _digest(states["baseline"], shared)
    for name, state in states.items():
        common = shared.intersection(state)
        assert common == shared
        assert _digest(state, common) == baseline_digest, name

    for prefix in optional_prefixes:
        keys = {key for key in states["full"] if key.startswith(prefix)}
        for name in ("no_quality", "no_relation"):
            common = keys.intersection(states[name])
            if common:
                assert _digest(states["full"], common) == _digest(
                    states[name], common
                ), (prefix, name)
    print("MODEL_INIT_PARITY_PASS", baseline_digest, len(shared))


if __name__ == "__main__":
    main()
