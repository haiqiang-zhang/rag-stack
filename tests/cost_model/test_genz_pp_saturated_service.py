import importlib
from pathlib import Path

import pandas as pd
import pytest

from GenZ.LLM_inference import decode_moddeling
from GenZ.LLM_inference.llm_spec_decode import (
    spec_decode_modeling,
    spec_prefill_modeling,
)
from GenZ.LLM_inference.utils import get_inference_system
from GenZ.Models import create_full_decode_model, remove_layer_file
from GenZ.Models.get_language_model import vllm_pipeline_layer_partitions
from GenZ.analyse_model import (
    get_model_df,
    get_pipeline_stage_latencies,
    get_pipeline_stage_memory_requirements,
)
from rag_stack.cost_model.llm_sim import genz_cache


def test_vllm_default_pipeline_partition_matches_floor_remainder_policy(
    monkeypatch,
):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)

    assert vllm_pipeline_layer_partitions(10, 4) == (2, 3, 3, 2)
    assert vllm_pipeline_layer_partitions(12, 5) == (2, 2, 3, 3, 2)
    assert vllm_pipeline_layer_partitions(28, 2) == (14, 14)


def test_vllm_pipeline_partition_override_is_exact_and_fail_closed(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "1,2,3,4")
    assert vllm_pipeline_layer_partitions(10, 4) == (1, 2, 3, 4)
    assert vllm_pipeline_layer_partitions(10, 4, "4,3,2,1") == (4, 3, 2, 1)

    with pytest.raises(ValueError, match="sums to"):
        vllm_pipeline_layer_partitions(10, 4, "1,1,1,1")
    with pytest.raises(ValueError, match="every PP rank"):
        vllm_pipeline_layer_partitions(10, 4, "0,2,3,5")


def test_pipeline_stage_decomposition_expands_layers_and_charges_send_to_sender():
    frame = pd.DataFrame(
        [
            ("embeddings", "GEMM", (), 1.0),
            ("Repeat", "Repeat", 2, 0.0),
            ("QKV", "GEMM", (), 2.0),
            ("End Repeat", "EndRepeat", 2, 0.0),
            ("Message Pass", "Sync", (), 0.5),
            ("Repeat", "Repeat", 3, 0.0),
            ("QKV", "GEMM", (), 1.0),
            ("End Repeat", "EndRepeat", 3, 0.0),
            ("classifier", "GEMM", (), 0.25),
        ],
        columns=["Layer Name", "Op Type", "Dimension", "Latency (msec)"],
    )

    assert get_pipeline_stage_latencies(frame, 2) == pytest.approx((5.5, 3.25))


def test_decode_reports_saturated_service_separately_from_full_traversal(
    monkeypatch,
):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    output = decode_moddeling(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        batch_size=2,
        input_tokens=16,
        output_tokens=1,
        Bb=1,
        system_name="H100_GPU",
        tensor_parallel=1,
        pipeline_parallel=2,
        model_offload=True,
    )

    stages = output["PipelineStageLatencies"]
    assert len(stages) == 2
    assert output["Latency"] == pytest.approx(max(stages))
    assert output["SaturatedServiceLatency"] == pytest.approx(max(stages))
    assert output["TraversalLatency"] == pytest.approx(sum(stages))
    assert output["Latency"] < output["TraversalLatency"]
    assert output["Throughput"] == pytest.approx(1000 * 2 / output["Latency"])

    # RAG-CM's current SimEngine owns one in-flight SchedulerOutput.  A single
    # batch cannot consume the max-rank interdeparture merely because vLLM's
    # queue has PP slots; those slots require disjoint batches.
    assert genz_cache.scheduler_batch_traversal_latency_ms(
        output,
        pipeline_parallel=2,
    ) == pytest.approx(sum(stages))


def test_single_scheduler_batch_pp_fails_closed_without_traversal() -> None:
    with pytest.raises(ValueError, match="requires GenZ TraversalLatency"):
        genz_cache.scheduler_batch_traversal_latency_ms(
            {
                "Latency": 1.0,
                "SaturatedServiceLatency": 1.0,
            },
            pipeline_parallel=2,
        )

    # PP=1 has no distinction and remains compatible with the historical
    # scalar result returned by small self-contained test doubles.
    assert genz_cache.scheduler_batch_traversal_latency_ms(
        {"Latency": 1.25},
        pipeline_parallel=1,
    ) == pytest.approx(1.25)


def test_decode_graph_materializes_nonuniform_vllm_stage_partition(monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    model_file = create_full_decode_model(
        name="gpt-2",
        input_sequence_length=16,
        output_gen_tokens=1,
        tensor_parallel=1,
        pipeline_parallel=5,
    )
    try:
        frame = pd.read_csv(Path("/tmp/genz/data/model") / model_file)
        layer_repeats = frame.loc[frame["Name"] == "Repeat", "M"].tolist()
        assert layer_repeats == [2, 2, 3, 3, 2]
        assert (frame["Name"] == "Message Pass").sum() == 4
    finally:
        remove_layer_file(model_file)


def test_override_memory_fit_uses_largest_physical_rank(monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    model_file = create_full_decode_model(
        name="gpt-2",
        input_sequence_length=16,
        output_gen_tokens=1,
        tensor_parallel=1,
        pipeline_parallel=2,
        pipeline_layer_partition="1,11",
    )
    try:
        frame = get_model_df(
            model_file,
            system=get_inference_system("H100_GPU"),
            batch_size=2,
            intermediate_on_chip=True,
            model_characterstics=True,
        )
        requirements = get_pipeline_stage_memory_requirements(frame, 2)
        assert len(requirements) == 2
        assert requirements[1] > requirements[0]
        # The old aggregate/PP approximation lies strictly below the rank
        # that owns eleven of twelve layers plus the output embedding.
        assert max(requirements) > sum(requirements) / 2
    finally:
        remove_layer_file(model_file)


@pytest.mark.parametrize(
    ("module_name", "entrypoint_name", "factory_name"),
    [
        (
            "GenZ.LLM_inference.llm_prefill",
            "prefill_moddeling",
            "create_full_prefill_model",
        ),
        (
            "GenZ.LLM_inference.llm_decode",
            "decode_moddeling",
            "create_full_decode_model",
        ),
    ],
)
def test_prefill_decode_fit_gate_rejects_largest_rank_not_average(
    monkeypatch,
    module_name,
    entrypoint_name,
    factory_name,
):
    module = importlib.import_module(module_name)

    class _System:
        @staticmethod
        def get_off_chip_mem_size():
            return 80.0

    summary = pd.DataFrame({
        f"Total Weights ({module.unit.unit_mem})": [80.0],
        f"KV Cache ({module.unit.unit_mem})": [0.0],
        f"Unused Weights ({module.unit.unit_mem})": [0.0],
    })
    monkeypatch.setattr(module, "get_inference_system", lambda **_kw: _System())
    monkeypatch.setattr(module, factory_name, lambda **_kw: "fixture-model")
    monkeypatch.setattr(module, "get_model_df", lambda *_a, **_kw: object())
    monkeypatch.setattr(module, "get_summary_table", lambda *_a, **_kw: summary)
    monkeypatch.setattr(
        module,
        "get_pipeline_stage_memory_requirements",
        lambda *_a, **_kw: (40.0, 90.0),
    )

    entrypoint = getattr(module, entrypoint_name)
    kwargs = {
        "model": "fixture",
        "batch_size": 2,
        "input_tokens": 16,
        "pipeline_parallel": 2,
        "pipeline_layer_partition": "1,11",
        "model_offload": False,
    }
    if entrypoint_name == "decode_moddeling":
        kwargs.update(output_tokens=1, Bb=1)
    with pytest.raises(ValueError, match="largest PP rank"):
        entrypoint(**kwargs)


@pytest.mark.parametrize(
    ("modeling", "extra_kwargs"),
    [
        (spec_prefill_modeling, {}),
        (
            spec_decode_modeling,
            {
                "output_tokens": 1,
                "num_parallel_tokens": 2,
                "token_acceptance_rate": 0.7,
            },
        ),
    ],
)
def test_speculative_pp_reports_service_and_traversal_separately(
    monkeypatch,
    modeling,
    extra_kwargs,
):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    output = modeling(
        model="gpt-2",
        draft_model="gpt-2",
        batch_size=2,
        input_tokens=16,
        system_name="H100_GPU",
        tensor_parallel=1,
        pipeline_parallel=2,
        model_offload=True,
        **extra_kwargs,
    )

    stages = output["PipelineStageLatencies"]
    assert len(stages) == 2
    assert output["Latency"] == pytest.approx(max(stages))
    assert output["SaturatedServiceLatency"] == pytest.approx(max(stages))
    assert output["TraversalLatency"] == pytest.approx(sum(stages))
    assert output["Latency"] < output["TraversalLatency"]


def test_genz_cache_formula_and_partition_identity_include_partition(monkeypatch):
    expected = "v5-single-scheduler-batch-pp-traversal"
    assert genz_cache._GENZ_FORMULA_VERSION == expected

    kwargs = {"model": "gpt-2", "pipeline_parallel": 2}
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    no_override = genz_cache._key("decode", kwargs)
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "6,6")
    assert genz_cache._key("decode", kwargs) != no_override

    explicit = {**kwargs, "pipeline_layer_partition": "6,6"}
    explicit_key = genz_cache._key("decode", explicit)
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "5,7")
    assert genz_cache._key("decode", explicit) == explicit_key
