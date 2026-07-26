"""Host deployment configuration, placement geometry and CM consumers."""
from tests import REPO_ROOT

import pytest
import yaml

from rag_stack.config_validator import ConfigValidator
from rag_stack.search_space.placement import (
    stage_engine,
    possible_measured_gpu_stage_lists,
    project_layout_to_slot_bounds,
    valid_layout_options,
)


_CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "baselines"
    / "sgs_h100x4_dragonball_mixedgp_measured_s43.yaml"
)

_FULL_STAGES = [
    "query_expansion_prefill",
    "query_expansion_decode",
    "passage_reranker",
    "generator_prefill",
    "generator_decode",
]


def _assert_layouts_fit_budget(
    *,
    n_gpus: int,
    min_slots: int,
    max_slots: int,
    max_num_gpus_per_stage: int,
):
    gpus = [f"cuda:{i}" for i in range(n_gpus)]
    options = valid_layout_options(
        _FULL_STAGES,
        gpus,
        min_slots=min_slots,
        max_slots=max_slots,
        max_num_gpus_per_stage=max_num_gpus_per_stage,
    )
    assert options

    effective_hi = min(max_slots, n_gpus)
    totals = {option["layout"]["total_slots"] for option in options}
    assert min(totals) >= min_slots
    assert max(totals) <= effective_hi

    for option in options:
        num_gpus = option["num_gpus"]
        layout = option["layout"]
        assert all(1 <= n <= max_num_gpus_per_stage for n in num_gpus.values())
        for group in layout["groups"]:
            expected_size = max(num_gpus[stage_engine(stage)] for stage in group)
            group_idx = layout["groups"].index(group)
            assert len(layout["group_devices"][group_idx]) == expected_size
        assert {
            device
            for devices in layout["group_devices"]
            for device in devices
        }.issubset(set(gpus))


def _measured_config_for_gpu_bounds(n_gpus: int, min_gpus: int, max_gpus: int) -> dict:
    return {
        "global": {"eval_backend": "static_gt"},
        "system": {
            "performance_source": "measured",
            "cm_search_space": {
                "min_num_gpus": min_gpus,
                "max_num_gpus": max_gpus,
                "max_num_gpus_per_stage": 2,
            },
            "system_design_space": {
                "available_gpus": [f"cuda:{i}" for i in range(n_gpus)],
                "batch_size_request": [4, 8],
            },
        },
    }


def test_measured_gpu_bounds_validator_uses_available_gpu_count():
    # GPU-bounds checks live on the system-space-owner axis now (they are
    # about WHO owns the deployment space, not where perf numbers come from).
    ok = ConfigValidator(_measured_config_for_gpu_bounds(8, 2, 8))
    ok._check_system_space_owner()
    assert not any("max_num_gpus" in e for e in ok.errors)

    bad = ConfigValidator(_measured_config_for_gpu_bounds(2, 1, 4))
    bad._check_system_space_owner()
    assert any("len(system.available_gpus)=2" in e for e in bad.errors)


def test_cost_model_gpu_bounds_are_not_limited_by_available_gpus():
    cfg = _measured_config_for_gpu_bounds(2, 8, 16)
    cfg["system"]["performance_source"] = "cost_model"

    validator = ConfigValidator(cfg)
    validator._check_system_space_owner()

    assert not any("available_gpus" in e for e in validator.errors)


def test_measured_system_search_space_includes_dynamic_batch_timeout_only():
    from rag_stack.search_space.system import extract_system_design_params

    cfg = _measured_config_for_gpu_bounds(2, 1, 2)
    cfg["system"]["system_design_space"].update({
        "batch_size_decode": [8, 32],
        "dynamic_batch_timeout_s": [0.002, 0.01],
        "vllm_kv_cache_dtype": "auto",
    })

    params = extract_system_design_params(
        cfg,
        lambda name, value: (name, value),
    )
    names = [name for name, _value in params]

    assert "system__dynamic_batch_timeout_s" in names
    assert "system__available_gpus" not in names


def test_disaggregated_generator_parallelism_is_role_local():
    from rag_stack.system_layout import (
        engine_parallelism,
        engine_role_parallelism,
        generator_pd_info,
    )

    system_config = {
        "layout": {
            "engines": {
                "generator": {
                    "pd_serving": "disagg_pd",
                    "devices": [
                        "cuda:0", "cuda:1", "cuda:2",
                        "cuda:3", "cuda:4", "cuda:5",
                    ],
                    "num_chips": 6,
                    "generator_prefill": {
                        "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                        "num_chips": 4,
                        "tp": 2,
                        "pp": 2,
                    },
                    "generator_decode": {
                        "devices": ["cuda:4", "cuda:5"],
                        "num_chips": 2,
                        "tp": 1,
                        "pp": 2,
                    },
                }
            }
        }
    }

    with pytest.raises(ValueError):
        engine_parallelism(system_config, "generator")

    assert engine_role_parallelism(system_config, "generator", "prefill") == (2, 2)
    assert engine_role_parallelism(system_config, "generator", "decode") == (1, 2)
    assert (
        generator_pd_info(system_config)["pd_layout"]
        == "1p1d_prefill_tp2pp2_decode_tp1pp2"
    )


def test_cm_fixed_generator_parallelism_reads_disagg_stage_roles():
    from rag_stack.cost_model.llm_sim.genz_adapter import (
        resolve_fixed_generator_parallelism,
    )

    pipeline_config = {
        "system": {
            "layout": {
                "engines": {
                    "generator": {
                        "pd_serving": "disagg_pd",
                        "devices": [
                            "cuda:0", "cuda:1", "cuda:2",
                            "cuda:3", "cuda:4", "cuda:5",
                        ],
                        "generator_prefill": {
                            "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                            "tp": 2,
                            "pp": 2,
                        },
                        "generator_decode": {
                            "devices": ["cuda:4", "cuda:5"],
                            "tp": 1,
                            "pp": 2,
                        },
                    }
                }
            }
        }
    }

    assert resolve_fixed_generator_parallelism(pipeline_config) is None
    assert resolve_fixed_generator_parallelism(pipeline_config, "generator_prefill") == (2, 2)
    assert resolve_fixed_generator_parallelism(pipeline_config, "generator_decode") == (1, 2)


def test_system_layout_decode_batch_defaults_and_floor():
    from rag_stack.system_layout import decode_batch, dynamic_batch_timeout_s

    assert decode_batch({"batch_size_request": 8}) == 8
    assert decode_batch({
        "batch_size_request": 8,
        "batch_size_decode": 32,
    }) == 32
    assert decode_batch({
        "batch_size_request": 8,
        "batching": {"decode": 4},
    }) == 8
    assert dynamic_batch_timeout_s({
        "batch_size_request": 8,
        "batching": {"dynamic_timeout_s": 0.01},
    }) == pytest.approx(0.01)
    assert dynamic_batch_timeout_s({
        "batch_size_request": 8,
        "batching": {"dynamic_timeout_s": -1},
    }) == 0.0


def test_measured_layout_options_respect_exact_4gpu_budget():
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    ss = cfg["system"]["cm_search_space"]
    ss["min_num_gpus"] = 4
    ss["max_num_gpus"] = 4
    gpus = cfg["system"]["system_design_space"]["available_gpus"]
    counts = []
    for stages in possible_measured_gpu_stage_lists(cfg):
        options = valid_layout_options(
            stages,
            gpus,
            min_slots=ss["min_num_gpus"],
            max_slots=ss["max_num_gpus"],
            max_num_gpus_per_stage=ss["max_num_gpus_per_stage"],
        )
        counts.append(len(options))
        assert options, f"no valid measured layouts for stages={stages!r}"
        assert {
            option["layout"]["total_slots"] for option in options
        } == {4}

    assert max(counts) > 1


def test_measured_layout_options_handle_various_gpu_budgets():
    cases = [
        {"n_gpus": 1, "min_slots": 1, "max_slots": 1, "max_num_gpus_per_stage": 1},
        {"n_gpus": 2, "min_slots": 1, "max_slots": 2, "max_num_gpus_per_stage": 2},
        {"n_gpus": 4, "min_slots": 2, "max_slots": 4, "max_num_gpus_per_stage": 2},
        {"n_gpus": 8, "min_slots": 4, "max_slots": 8, "max_num_gpus_per_stage": 4},
    ]
    for case in cases:
        _assert_layouts_fit_budget(**case)


def test_impossible_gpu_budget_has_no_layout_options():
    assert valid_layout_options(
        _FULL_STAGES,
        ["cuda:0", "cuda:1"],
        min_slots=4,
        max_slots=4,
        max_num_gpus_per_stage=2,
    ) == []


def test_over_budget_raw_layout_projects_to_exact_4gpu_budget():
    stages = ["passage_reranker", "generator_prefill", "generator_decode"]
    gpus = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    raw_cuts = {
        "cut_after_passage_reranker": 1,
        "cut_after_generator_prefill": 1,
    }
    raw_nums = {
        "passage_reranker": 1,
        "generator": 2,
    }

    cut_bits, num_gpus, layout, changed = project_layout_to_slot_bounds(
        stages,
        raw_cuts,
        raw_nums,
        gpus,
        min_slots=4,
        max_slots=4,
        max_num_gpus_per_stage=2,
    )

    assert changed
    assert layout["total_slots"] == 4
    assert len(layout["group_devices"]) == 2
    assert num_gpus == raw_nums
    assert cut_bits["cut_after_generator_prefill"] == 1


def test_layout_projection_scales_to_8gpu_budget():
    gpus = [f"cuda:{i}" for i in range(8)]
    raw_cuts = {
        "cut_after_query_expansion_prefill": 0,
        "cut_after_query_expansion_decode": 0,
        "cut_after_passage_reranker": 0,
        "cut_after_generator_prefill": 0,
    }
    raw_nums = {
        "query_expansion": 1,
        "passage_reranker": 1,
        "generator": 1,
    }

    cut_bits, num_gpus, layout, changed = project_layout_to_slot_bounds(
        _FULL_STAGES,
        raw_cuts,
        raw_nums,
        gpus,
        min_slots=8,
        max_slots=8,
        max_num_gpus_per_stage=4,
    )

    assert changed
    assert layout["total_slots"] == 8
    assert set(num_gpus) == set(raw_nums)
    assert set(cut_bits) == set(raw_cuts)


def test_valid_layout_options_order_is_pinned_for_full_pipeline():
    """The gpu_layout categorical ("layout_0007") indexes into
    valid_layout_options — the enumeration ORDER is load-bearing. Pin the
    first/last options for a full pipeline (reranker + compressor + generator)
    so a refactor that reorders stages or engines fails loudly instead of
    silently remapping persisted layout choices."""
    stages = [
        "semantic_retrieval_encode",
        "passage_reranker",
        "passage_compressor",
        "generator_prefill",
        "generator_decode",
    ]
    gpus = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    options = valid_layout_options(
        stages, gpus, min_slots=1, max_slots=4, max_num_gpus_per_stage=2,
    )
    assert len(options) == 120
    first = options[0]
    assert first["num_gpus"] == {
        "semantic_retrieval": 1, "passage_reranker": 1,
        "passage_compressor": 1, "generator": 1,
    }
    assert first["cut_bits"] == {
        "cut_after_semantic_retrieval_encode": 0,
        "cut_after_passage_reranker": 0,
        "cut_after_passage_compressor": 0,
        "cut_after_generator_prefill": 0,
    }
    assert first["layout"]["groups"] == [stages]
    last = options[-1]
    assert last["num_gpus"] == {
        "semantic_retrieval": 2, "passage_reranker": 2,
        "passage_compressor": 2, "generator": 2,
    }
    assert last["cut_bits"] == {
        "cut_after_semantic_retrieval_encode": 1,
        "cut_after_passage_reranker": 0,
        "cut_after_passage_compressor": 0,
        "cut_after_generator_prefill": 0,
    }
    assert last["layout"]["engines"]["generator"] == {
        "pd_serving": "collocated_pd", "devices": ["cuda:2", "cuda:3"],
    }
