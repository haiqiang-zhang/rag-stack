"""deployment_presets: parse / expand / budget-gate semantics (pure, no GPU).

The expander compiles a policy-level template into the raw measured dims
vocabulary for the ACTUAL stage list; its strict budget check is the only
gate on the cost-model path, and unfittable presets must raise (penalize),
never silently project onto a different deployment.
"""

from __future__ import annotations

import pytest

from rag_stack.search_space.deployment_presets import (
    DeploymentPresetError,
    expand_preset,
    parse_deployment_presets,
    preset_fits_stages,
)

GPUS = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
BOUNDS = {"min_num_gpus": 1, "max_num_gpus": 4, "max_num_gpus_per_stage": 4}

# sequential w/ reranker vs react-stripped shapes
FULL = [
    "semantic_retrieval_encode", "passage_reranker",
    "generator_prefill", "generator_decode",
]
NO_RERANK = ["semantic_retrieval_encode", "generator_prefill", "generator_decode"]


def _parse_one(spec: dict) -> dict:
    return parse_deployment_presets(
        {"deployment_presets": {"p": spec}}
    )["p"]


# ---------------------------------------------------------------------------
# parse_deployment_presets
# ---------------------------------------------------------------------------

def test_parse_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        _parse_one({"placement": "pd_disagg",
                    "num_gpus": {"generator_prefill": 3}})


def test_parse_rejects_unknown_placement_and_keys():
    with pytest.raises(ValueError, match="placement"):
        _parse_one({"placement": "sharded", "num_gpus": {}})
    with pytest.raises(ValueError, match="unknown keys"):
        _parse_one({"placement": "collocated", "tensor_parallel": {"generator": 2}})


def test_disaggregated_cuts_every_stage():
    preset = _parse_one({
        "placement": "disaggregated",
        "num_gpus": {"generator_prefill": 1, "generator_decode": 2},
    })
    # FULL (with reranker) fully-disaggregated needs 5 slots > 4 — use the
    # react-stripped shape (1 enc + 1 prefill + 2 decode = 4).
    dims = expand_preset(preset, NO_RERANK, GPUS, BOUNDS, name="disagg")
    assert all(
        v == 1 for k, v in dims.items() if k.startswith("cut_after_")
    ), dims
    assert dims["num_gpus_generator_prefill"] == 1
    assert dims["num_gpus_generator_decode"] == 2


def test_parse_rejects_collocated_asymmetric_roles():
    with pytest.raises(ValueError, match="pd_disagg"):
        _parse_one({
            "placement": "collocated",
            "num_gpus": {"generator_prefill": 1, "generator_decode": 2},
        })


def test_parse_rejects_pp_not_dividing_chips():
    with pytest.raises(ValueError, match="does not divide"):
        _parse_one({
            "placement": "collocated",
            "num_gpus": {"generator": 4},
            "pipeline_parallel": {"generator": 3},
        })


def test_parse_rejects_multi_slot_semantic_retrieval():
    with pytest.raises(ValueError, match="semantic_retrieval"):
        _parse_one({"placement": "collocated",
                    "num_gpus": {"semantic_retrieval": 2}})


# ---------------------------------------------------------------------------
# expand_preset — vocabulary, aux policy, shape adaptation
# ---------------------------------------------------------------------------

def test_collocated_tp4_dims():
    preset = _parse_one({"placement": "collocated", "num_gpus": {"generator": 4}})
    dims = expand_preset(preset, FULL, GPUS, BOUNDS, name="collocated_tp4")
    # all cut bits zero → one group; aux stages ride the generator's GPUs
    assert all(v == 0 for k, v in dims.items() if k.startswith("cut_after_"))
    assert dims["num_gpus_generator"] == 4
    assert dims["available_gpus"] == GPUS


def test_pd_disagg_cut_only_after_generator_prefill():
    preset = _parse_one({
        "placement": "pd_disagg",
        "num_gpus": {"generator_prefill": 2, "generator_decode": 2},
    })
    dims = expand_preset(preset, FULL, GPUS, BOUNDS, name="disagg_2p2d")
    assert dims["cut_after_generator_prefill"] == 1
    assert dims["cut_after_semantic_retrieval_encode"] == 0
    assert dims["cut_after_passage_reranker"] == 0
    # symmetric roles collapse onto the engine key
    assert dims["num_gpus_generator"] == 2


def test_asymmetric_1p2d_emits_role_keys():
    preset = _parse_one({
        "placement": "pd_disagg",
        "num_gpus": {"generator_prefill": 1, "generator_decode": 2},
    })
    dims = expand_preset(preset, NO_RERANK, GPUS, BOUNDS, name="disagg_1p2d")
    assert dims["num_gpus_generator_prefill"] == 1
    assert dims["num_gpus_generator_decode"] == 2
    assert dims["num_gpus_generator"] == 2  # engine fallback = max(roles)


def test_absent_stage_keys_are_dropped():
    preset = _parse_one({
        "placement": "collocated",
        "num_gpus": {"generator": 2, "passage_reranker": 2},
    })
    dims = expand_preset(preset, NO_RERANK, GPUS, BOUNDS, name="p")
    assert "num_gpus_passage_reranker" not in dims
    assert "cut_after_passage_reranker" not in dims


def test_role_pp_pins_sibling_role_to_one():
    preset = _parse_one({
        "placement": "pd_disagg",
        "num_gpus": {"generator_prefill": 2, "generator_decode": 2},
        "pipeline_parallel": {"generator_prefill": 2},
    })
    dims = expand_preset(preset, NO_RERANK, GPUS, BOUNDS, name="p")
    assert dims["pipeline_parallel_generator_prefill"] == 2
    # the sibling role must be explicitly pp=1 — never inherit via the
    # resolver's engine-key fallback
    assert dims["pipeline_parallel_generator_decode"] == 1
    assert "pipeline_parallel_generator" not in dims


def test_parse_rejects_role_pp_under_collocated():
    with pytest.raises(ValueError, match="pd_disagg"):
        _parse_one({
            "placement": "collocated",
            "num_gpus": {"generator": 4},
            "pipeline_parallel": {"generator_prefill": 2},
        })


# ---------------------------------------------------------------------------
# budget gate — raise, never project
# ---------------------------------------------------------------------------

def test_over_budget_raises():
    preset = _parse_one({
        "placement": "pd_disagg",
        "num_gpus": {"generator_prefill": 4, "generator_decode": 4},
    })
    with pytest.raises(DeploymentPresetError, match="outside budget"):
        expand_preset(preset, NO_RERANK, GPUS, BOUNDS, name="p8")


def test_under_min_budget_raises():
    preset = _parse_one({"placement": "collocated", "num_gpus": {"generator": 1}})
    bounds = dict(BOUNDS, min_num_gpus=4)
    with pytest.raises(DeploymentPresetError, match="outside budget"):
        expand_preset(preset, NO_RERANK, GPUS, bounds, name="tp1")


def test_per_stage_cap_raises():
    preset = _parse_one({"placement": "collocated", "num_gpus": {"generator": 4}})
    bounds = dict(BOUNDS, max_num_gpus_per_stage=2)
    with pytest.raises(DeploymentPresetError, match="max_num_gpus_per_stage"):
        expand_preset(preset, NO_RERANK, GPUS, bounds, name="tp4")


def test_pd_disagg_without_generator_pair_raises():
    preset = _parse_one({"placement": "pd_disagg",
                         "num_gpus": {"generator": 2}})
    with pytest.raises(DeploymentPresetError, match="prefill\\+decode"):
        expand_preset(preset, ["semantic_retrieval_encode"], GPUS, BOUNDS)


def test_preset_fits_stages_mirrors_expand():
    ok = _parse_one({"placement": "collocated", "num_gpus": {"generator": 2}})
    bad = _parse_one({
        "placement": "pd_disagg",
        "num_gpus": {"generator_prefill": 4, "generator_decode": 4},
    })
    assert preset_fits_stages(ok, NO_RERANK, GPUS, BOUNDS)
    assert not preset_fits_stages(bad, NO_RERANK, GPUS, BOUNDS)


# ---------------------------------------------------------------------------
# adaptive left family (aux: left) — aux on leftover cards, pow2 growth
# ---------------------------------------------------------------------------

FULL_QE = [
    "query_expansion_prefill", "query_expansion_decode",
    "semantic_retrieval_encode", "passage_reranker", "passage_compressor",
    "generator_prefill", "generator_decode",
]
ENC_ONLY = ["semantic_retrieval_encode", "generator_prefill", "generator_decode"]


def _left(spec):
    return _parse_one({**spec, "aux": "left"})


def test_left_pd_1p1d_splits_aux_into_leftover_groups():
    preset = _left({"placement": "pd_disagg",
                    "num_gpus": {"generator_prefill": 1, "generator_decode": 1}})
    dims = expand_preset(preset, FULL_QE, GPUS, BOUNDS, name="pd_1p1d_left")
    # 4 aux engines -> 2 contiguous groups (qe+enc | rerank+compress)
    assert dims["cut_after_semantic_retrieval_encode"] == 1
    assert dims["cut_after_passage_compressor"] == 1
    assert dims["cut_after_generator_prefill"] == 1
    assert dims["cut_after_query_expansion_decode"] == 0  # qe stays with enc
    assert dims["num_gpus_generator_prefill"] == 1
    assert dims["num_gpus_generator_decode"] == 1


def test_left_pd_1p1d_grows_decode_when_single_aux():
    preset = _left({"placement": "pd_disagg",
                    "num_gpus": {"generator_prefill": 1, "generator_decode": 1}})
    dims = expand_preset(preset, ENC_ONLY, GPUS, BOUNDS, name="pd_1p1d_left")
    # encoder takes 1 leftover card; the spare card doubles decode -> 1P2D
    assert dims["num_gpus_generator_prefill"] == 1
    assert dims["num_gpus_generator_decode"] == 2
    assert dims["cut_after_semantic_retrieval_encode"] == 1


def test_left_tp2_idles_one_card_on_encoder_only_shape():
    preset = _left({"placement": "collocated", "num_gpus": {"generator": 2}})
    dims = expand_preset(preset, ENC_ONLY, GPUS, BOUNDS, name="tp2_left")
    from rag_stack.search_space.placement import derive_layout
    cuts = {k: v for k, v in dims.items() if k.startswith("cut_after_")}
    nums = {k[9:]: v for k, v in dims.items() if k.startswith("num_gpus_")}
    lay = derive_layout(ENC_ONLY, cuts, nums, GPUS)
    # tp2 cannot grow to 3 (pow2) and the encoder cannot take a 2nd card:
    # 3 slots used, 1 idles — documented, never projected.
    assert lay["total_slots"] == 3
    assert dims["num_gpus_generator"] == 2


def test_left_needs_leftover_cards():
    preset = _left({"placement": "collocated", "num_gpus": {"generator": 4}})
    with pytest.raises(DeploymentPresetError, match="leftover"):
        expand_preset(preset, ENC_ONLY, GPUS, BOUNDS, name="tp4_left")


def test_parse_rejects_left_with_disaggregated():
    with pytest.raises(ValueError, match="redundant"):
        _parse_one({"placement": "disaggregated", "aux": "left",
                    "num_gpus": {"generator": 1}})
