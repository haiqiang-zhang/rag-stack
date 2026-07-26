"""DES contracts for measured one-model-per-device auxiliary stages."""

import pytest

from rag_stack.cost_model.closed_loop_sim import (
    SimStageItem,
    _data_parallel_global_batch_cap,
    build_sim_inputs,
)
from rag_stack.cost_model.engines import CallLatency
from rag_stack.types import SystemConfig


@pytest.mark.parametrize(
    ("local_cap", "devices", "expected_global_cap"),
    [
        (8, ("cuda:0", "cuda:1", "cuda:2", "cuda:3"), 32),
        (4, ("cuda:0", "cuda:1", "cuda:2", "cuda:3"), 16),
        (8, ("cuda:0",), 8),
        (8, (), 8),
    ],
)
def test_data_parallel_global_batch_cap_matches_measured_gang_batcher(
    local_cap,
    devices,
    expected_global_cap,
):
    assert _data_parallel_global_batch_cap(local_cap, devices) == (
        expected_global_cap
    )


@pytest.mark.parametrize("invalid_cap", [0, -1])
def test_data_parallel_global_batch_cap_rejects_invalid_local_cap(invalid_cap):
    with pytest.raises(ValueError, match="invalid local data-parallel batch cap"):
        _data_parallel_global_batch_cap(invalid_cap, ("cuda:0",))


def test_reranker_gang_cap_expands_but_forward_policy_stays_local(monkeypatch):
    class _AnalyticalOnlyCurve:
        @staticmethod
        def corrected_service_s(_batch, component_service_s):
            return float(component_service_s)

    monkeypatch.setattr(
        "rag_stack.cost_model.stage_batch_curve.load_stage_batch_curve",
        lambda _hardware, _stage: _AnalyticalOnlyCurve(),
    )
    devices = ["cuda:0", "cuda:1"]
    generator_device = "cuda:2"
    layout = {
        "available_devices": devices + [generator_device],
        "resource_groups": [
            {
                "id": "gpu:reranker",
                "kind": "gpu",
                "devices": devices,
                "performance_stages": ["passage_reranker"],
            },
            {
                "id": "gpu:generator",
                "kind": "gpu",
                "devices": [generator_device],
                "performance_stages": [
                    "generator_prefill",
                    "generator_decode",
                ],
            },
        ],
        "performance_stages": {
            "passage_reranker": {
                "resource": "gpu:reranker",
                "devices": devices,
                "num_chips": 2,
            },
            "generator_prefill": {
                "resource": "gpu:generator",
                "devices": [generator_device],
                "num_chips": 1,
                "engine": "generator",
                "role": "prefill",
                "tp": 1,
                "pp": 1,
                "replicas": 1,
                "replica_devices": [[generator_device]],
            },
            "generator_decode": {
                "resource": "gpu:generator",
                "devices": [generator_device],
                "num_chips": 1,
                "engine": "generator",
                "role": "decode",
                "tp": 1,
                "pp": 1,
                "replicas": 1,
                "replica_devices": [[generator_device]],
            },
        },
        "engines": {
            "generator": {
                "pd_serving": "collocated_pd",
                "devices": [generator_device],
                "tp": 1,
                "pp": 1,
                "replicas": 1,
                "replica_devices": [[generator_device]],
            },
        },
    }
    observed = []

    def _forward_tiles(input_tokens, request_batch_size=None):
        observed.append((tuple(input_tokens), int(request_batch_size)))
        return (1e-4,)

    latency = CallLatency(
        prefill_s=lambda tokens, batch, tp=1, pp=1: 1e-4,
        decode_s=lambda inp, out, batch, tp=1, pp=1: 1e-4,
        rerank_s=lambda tokens, batch: 1e-4,
        rerank_forward_tiles_s=_forward_tiles,
    )
    trace = [[
        {"stage": "passage_reranker", "input_tokens": 32},
        {"stage": "generator", "input_tokens": 64, "output_tokens": 1},
    ]]
    inputs, reason = build_sim_inputs(
        {
            "system": {
                "batching": {"dynamic_timeout_s": 0.0},
                "communication": {"include_non_pd_comm": False},
            },
        },
        SystemConfig(num_cores=8),
        trace,
        [["passage_reranker"], ["generator_prefill", "generator_decode"]],
        batch_size_request=4,
        batch_size_decode=4,
        candidate_row={"candidate_system_layout": layout},
        latency=latency,
    )

    assert reason is None
    stages, _engines, _queries, *_ = inputs
    reranker = next(stage for stage in stages if stage.name == "passage_reranker")
    assert reranker.batch_cap == 8
    assert reranker.batch_ops is not None
    reranker.batch_ops([
        SimStageItem(stage="passage_reranker", input_tokens=index)
        for index in range(8)
    ])
    assert observed == [
        ((0, 1, 2, 3), 4),
        ((4, 5, 6, 7), 4),
    ]
