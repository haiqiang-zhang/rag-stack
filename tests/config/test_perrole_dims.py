"""Per-role (asymmetric PD) num_gpus vocabulary — pure layout geometry.

``num_gpus_<engine>`` stays the symmetric default; a role-qualified key
(``num_gpus_generator_prefill`` — the PerformanceStage's full name) wins
inside ``derive_layout``, making asymmetric PD (1P2D / 2P1D) expressible
through raw dims (deployment presets, CM-selected rows) while the
``gpu_layout`` enumeration stays per-engine.
"""

from __future__ import annotations

import pytest

from rag_stack.search_space.placement import (
    _stage_num_gpus,
    cut_after_key,
    derive_layout,
    valid_layout_options,
)

GPUS = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
STAGES = ["semantic_retrieval_encode", "generator_prefill", "generator_decode"]


def _pd_cut() -> dict:
    """encode+prefill collocated, decode split off (the canonical PD cut)."""
    return {
        cut_after_key("semantic_retrieval_encode"): 0,
        cut_after_key("generator_prefill"): 1,
    }


# ---------------------------------------------------------------------------
# _stage_num_gpus precedence
# ---------------------------------------------------------------------------

def test_stage_key_wins_over_engine_key():
    num_gpus = {"generator": 2, "generator_prefill": 1}
    assert _stage_num_gpus("generator_prefill", num_gpus) == 1
    assert _stage_num_gpus("generator_decode", num_gpus) == 2  # engine fallback


@pytest.mark.parametrize(("counts", "expected"), [
    ({}, 1),
    ({"semantic_retrieval": 4}, 4),
    ({"semantic_retrieval": 4, "semantic_retrieval_encode": 2}, 2),
])
def test_semantic_retrieval_encoder_uses_resolved_replica_count(counts, expected):
    assert _stage_num_gpus("semantic_retrieval_encode", counts) == expected


# ---------------------------------------------------------------------------
# derive_layout — symmetric behaviour unchanged, asymmetric via role keys
# ---------------------------------------------------------------------------

def test_symmetric_engine_key_unchanged():
    layout = derive_layout(STAGES, _pd_cut(), {"generator": 2}, GPUS)
    assert layout["total_slots"] == 4  # prefill group 2 + decode group 2
    gen = layout["engines"]["generator"]
    assert gen["pd_serving"] == "disagg_pd"
    assert gen["prefill_devices"] == ["cuda:0", "cuda:1"]
    assert gen["decode_devices"] == ["cuda:2", "cuda:3"]


def test_asymmetric_1p2d_via_role_keys():
    num_gpus = {
        "generator": 2,               # engine max (compat fallback)
        "generator_prefill": 1,       # role-qualified: asymmetric
    }
    layout = derive_layout(STAGES, _pd_cut(), num_gpus, GPUS)
    # prefill-side group (encode+prefill) sized 1, decode group sized 2
    assert layout["total_slots"] == 3
    gen = layout["engines"]["generator"]
    assert gen["prefill_devices"] == ["cuda:0"]
    assert gen["decode_devices"] == ["cuda:1", "cuda:2"]
    assert gen["devices"] == ["cuda:0", "cuda:1", "cuda:2"]


def test_asymmetric_2p1d_via_role_keys():
    num_gpus = {
        "generator": 2,
        "generator_decode": 1,
    }
    layout = derive_layout(STAGES, _pd_cut(), num_gpus, GPUS)
    assert layout["total_slots"] == 3
    gen = layout["engines"]["generator"]
    assert gen["prefill_devices"] == ["cuda:0", "cuda:1"]
    assert gen["decode_devices"] == ["cuda:2"]


def test_unified_engine_takes_max_of_role_keys():
    # No PD cut: generator unified; role keys disagreeing → max wins.
    cuts = {
        cut_after_key("semantic_retrieval_encode"): 1,
        cut_after_key("generator_prefill"): 0,
    }
    num_gpus = {"generator_prefill": 1, "generator_decode": 2}
    layout = derive_layout(STAGES, cuts, num_gpus, GPUS)
    gen = layout["engines"]["generator"]
    assert gen["pd_serving"] == "collocated_pd"
    assert len(gen["devices"]) == 2


# ---------------------------------------------------------------------------
# gpu_layout enumeration cardinality is untouched by the role vocabulary
# ---------------------------------------------------------------------------

def test_valid_layout_options_stay_per_engine():
    options = valid_layout_options(
        STAGES, GPUS, min_slots=1, max_slots=4, max_num_gpus_per_stage=2,
    )
    assert options, "enumeration must not be empty"
    for opt in options:
        for key in opt["num_gpus"]:
            # engine keys only — no role-qualified stage names
            assert key in ("semantic_retrieval", "generator")
