"""Fixed CM deployment reconstruction and trace handoff, without calibration data.

The real resolver and assembly bridge preserve the sampled preset, batches and
CPU/GPU resources. Pricing itself belongs to the shared dynamic pipeline tests;
two-role LLM serving QPS requires an observed trace, including for fixed layouts.
"""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from rag_stack.cost_model.assembly import RAGCMAssembly
from rag_stack.cost_model.fixed_placement_provider import CmFixedPlacementProvider
from rag_stack.performance_context import PerformanceContext
from rag_stack.rag_ir import make_quality_trace_envelope
from rag_stack.search_space.deployment_presets import expand_preset, parse_deployment_presets
from rag_stack.search_space.placement import measured_gpu_stages
from rag_stack.types import SystemConfig


BATCH = 8
DECODE_BATCH = 16
GPUS = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
PRESETS = {
    "disaggregated_1p2d": {
        "placement": "disaggregated",
        "num_gpus": {"generator_prefill": 1, "generator_decode": 2},
    },
    "disaggregated_2p1d": {
        "placement": "disaggregated",
        "num_gpus": {"generator_prefill": 2, "generator_decode": 1},
    },
}


@pytest.fixture
def ctx():
    # Exercise deployment resolution without binding a dataset, tokenizer,
    # project cache or machine-specific calibration profiles.
    context = PerformanceContext.__new__(PerformanceContext)
    context.hardware = SystemConfig(num_cores=8, cpu_hardware_key="fixture_cpu")
    context.config = {"system": {
        "performance_source": "cost_model",
        "system_space_owner": "optimizer",
        "cm_search_space": {"min_num_gpus": 1, "max_num_gpus": 4,
                            "max_num_gpus_per_stage": 4},
        "retrieval": {"faiss_num_threads": 1, "num_servers": 3},
        "deployment_presets": deepcopy(PRESETS),
        "system_design_space": {
            "deployment_preset": list(PRESETS), "available_gpus": list(GPUS),
            "batch_size_request": BATCH, "batch_size_decode": DECODE_BATCH,
        },
    }}
    context.assembly = RAGCMAssembly(context.hardware)
    return context


@pytest.fixture
def pipeline_config():
    return {"node_lines": [{"node_line_name": "main", "nodes": [
        {"stage": "semantic_retrieval", "top_k": 4,
         "modules": [{"component": "vectordb", "vectordb": "fixture_index"}]},
        {"stage": "generator", "modules": [{"component": "vllm", "model": "fixture_generator"}]},
    ]}], "vectordb": [{"name": "fixture_index", "db_type": "faiss_hnsw",
                      "embedding_model": "fixture_encoder"}]}


def _fixed_system_config(ctx, pipeline_config, name):
    preset = parse_deployment_presets(ctx.config["system"])[name]
    dims = expand_preset(
        preset, measured_gpu_stages(pipeline_config), GPUS,
        ctx.config["system"]["cm_search_space"], name=name,
    )
    dims.update(batch_size_request=BATCH, batch_size_decode=DECODE_BATCH)
    system = ctx.resolve_system_config(
        {}, resolved_config=pipeline_config, system_dims_override=dims,
    )
    system["deployment_preset"] = name
    return system


def _trace():
    calls = [
        {"stage": stage, "step_idx": index, "input_tokens": 12,
         "output_tokens": 2 if stage == "generator" else 0,
         "model_id": "fixture_generator" if stage == "generator" else "fixture_encoder"}
        for index, stage in enumerate([
            "semantic_retrieval_encode", "semantic_retrieval_vectorsearch", "generator",
        ])
    ]
    return make_quality_trace_envelope([calls], question_ids=["q1"])


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_fixed_provider_preserves_deployment_at_dynamic_boundary(ctx, pipeline_config, name, monkeypatch):
    system = _fixed_system_config(ctx, pipeline_config, name)
    original_pipeline, original_system = deepcopy(pipeline_config), deepcopy(system)
    trace = _trace()
    dynamic = Mock(return_value=object())
    monkeypatch.setattr(ctx.assembly, "evaluate_dynamic", dynamic)
    monkeypatch.setattr(ctx.assembly, "_run_sweep", lambda *a, **kw: pytest.fail("fixed deployment must not sweep"))

    result = CmFixedPlacementProvider(ctx.assembly).evaluate(pipeline_config, system, trace=trace)

    assert result is dynamic.return_value
    dynamic.assert_called_once()
    folded = dynamic.call_args.args[0]
    kwargs = dynamic.call_args.kwargs
    mapping = kwargs["fixed_placement"]
    expected_roles = PRESETS[name]["num_gpus"]
    assert mapping.num_chips == {"semantic_retrieval_encode": 1, **expected_roles}
    assert mapping.num_retrieval_servers == 3
    assert mapping.collocation_strategy == [
        ["semantic_retrieval_encode"], ["semantic_retrieval_vectorsearch"],
        ["generator_prefill"], ["generator_decode"],
    ]
    assert kwargs["fixed_batch_size_request"] == BATCH
    assert kwargs["fixed_batch_size_decode"] == DECODE_BATCH
    assert kwargs["trace"] is trace
    assert folded["system"] == system
    assert folded["system"]["layout"] == original_system["layout"]
    assert kwargs["workflow_schema"].stages == [
        "semantic_retrieval_encode", "semantic_retrieval_vectorsearch",
        "generator_prefill", "generator_decode",
    ]
    assert pipeline_config == original_pipeline
    assert system == original_system


def test_fixed_two_role_llm_without_trace_rejects_serving_qps(ctx, pipeline_config, monkeypatch):
    system = _fixed_system_config(ctx, pipeline_config, "disaggregated_1p2d")
    monkeypatch.setattr(ctx.assembly, "_run_sweep", lambda *a, **kw: pytest.fail("invalid no-trace QPS reached sweep"))
    with pytest.raises(RuntimeError, match="no scheduler-cycle composition"):
        CmFixedPlacementProvider(ctx.assembly).evaluate(pipeline_config, system)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_resolve_trial_system_config_schema_parity(ctx, pipeline_config, name):
    via_funnel = ctx.resolve_trial_system_config(
        {"system": {"deployment_preset": name}}, pipeline_config,
    )
    assert via_funnel == _fixed_system_config(ctx, pipeline_config, name)


def test_provider_is_thin_shell_over_evaluate_fixed():
    assembly = Mock()
    provider = CmFixedPlacementProvider(assembly)
    pipeline = {"pipeline": True}
    system = {"batch_size_request": 8, "deployment_preset": "fixture"}
    trace = {"queries": []}
    aggregate = {"shape": "aggregate", "token_stats": {}}

    assert provider.evaluate(pipeline, system, trace=trace) is assembly.evaluate_fixed.return_value
    assembly.evaluate_fixed.assert_called_once_with(pipeline, system, trace=trace, runtime_stats=None)
    assembly.evaluate_fixed.reset_mock()
    assert provider.evaluate(pipeline, system, runtime_stats=aggregate) is assembly.evaluate_fixed.return_value
    assembly.evaluate_fixed.assert_called_once_with(pipeline, system, trace=None, runtime_stats=aggregate)
