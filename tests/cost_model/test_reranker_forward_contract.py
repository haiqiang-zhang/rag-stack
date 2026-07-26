"""CPU-only contracts for reranker forward policy and executed microbatches."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("component", "request_batch", "expected"),
    [
        ("monot5", 32, 8),
        ("monot5", 4, 4),
        ("colbert_reranker", 32, 32),
        ("sentence_transformer_reranker", 32, 32),
    ],
)
def test_system_resolver_applies_one_concrete_reranker_policy(
    component: str,
    request_batch: int,
    expected: int,
) -> None:
    from rag_stack.cost_model.reranker_policy import (
        reranker_forward_batch_from_system,
    )

    assert reranker_forward_batch_from_system(
        component,
        request_batch,
        {},
    ) == expected
    assert reranker_forward_batch_from_system(
        component,
        request_batch,
        {"reranker_max_batch_size": 3},
    ) == min(request_batch, 3)


class _RerankerComponent:
    def estimate(self, config, hardware):
        del hardware
        module = next(
            node["modules"][0]
            for line in config["node_lines"]
            for node in line["nodes"]
            if node["stage"] == "passage_reranker"
        )
        return SimpleNamespace(latency_s=0.001 * int(module["batch_size"]))


def _monot5_pipeline() -> dict:
    return {
        "_token_stats": {
            "avg_chunk_tokens": 64,
            "avg_query_tokens": 8,
            "avg_output_tokens": 16,
            "prompt_template_tokens": 1,
            "bytes_per_token": 4.0,
        },
        "node_lines": [{"nodes": [
            {
                "stage": "semantic_retrieval",
                "top_k": 4,
                "modules": [{"component": "vectordb"}],
            },
            {
                "stage": "passage_reranker",
                "top_k": 2,
                "modules": [{"component": "monot5"}],
            },
        ]}],
        "system": {"reranker_max_batch_size": 3},
    }


def test_measured_and_cost_model_consume_the_same_system_resolver(
    monkeypatch,
) -> None:
    from rag_stack.cost_model import sweep_inputs
    from rag_stack.cost_model.engines import build_call_latency
    from rag_stack.cost_model.reranker_policy import (
        reranker_forward_batch_from_system,
    )
    from rag_stack_evaluator.static_rag_evaluator.measured.evaluator import (
        _measured_reranker_forward_batch,
    )
    from rag_stack.types import SystemConfig

    monkeypatch.setattr(
        sweep_inputs,
        "build_performance_stage_cost_models",
        lambda config: {"passage_reranker": _RerankerComponent()},
    )
    config = _monot5_pipeline()
    latency = build_call_latency(
        config,
        SystemConfig(num_cores=8),
    )

    class Module:
        component = "monot5"

    class Node:
        module = Module()

    expected = reranker_forward_batch_from_system(
        "monot5", 32, config["system"]
    )
    assert expected == 3
    assert _measured_reranker_forward_batch(config["system"], Node(), 32) == expected
    assert latency.rerank_forward_batch is not None
    assert latency.rerank_forward_batch(32) == expected


class _Encoding(dict):
    def to(self, device):
        del device
        return self


class _Tokenizer:
    def __call__(self, texts, **kwargs):
        del kwargs
        return _Encoding(
            input_ids=list(texts),
            attention_mask=[1] * len(texts),
        )


class _Scores:
    def __init__(self, size: int):
        self.size = size

    def __getitem__(self, key):
        del key
        return self


class _Probabilities:
    def __init__(self, size: int):
        self.size = size

    def __getitem__(self, key):
        del key
        return self

    def tolist(self):
        return [0.5] * self.size


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        return False


def test_monot5_reports_actual_microbatches_and_clears_after_pop(
    monkeypatch,
) -> None:
    from rag_stack.cost_model.reranker_policy import (
        MONOT5_FORWARD_EXECUTION_SCHEMA,
    )
    from rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker import monot5

    class FakeOutOfMemoryError(RuntimeError):
        pass

    empty_cache_calls: list[None] = []

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            OutOfMemoryError=FakeOutOfMemoryError,
            empty_cache=lambda: empty_cache_calls.append(None),
        ),
        no_grad=lambda: _NoGrad(),
        nn=SimpleNamespace(
            functional=SimpleNamespace(
                softmax=lambda scores, dim: _Probabilities(scores.size),
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Model:
        def generate(self, *, input_ids, **kwargs):
            del kwargs
            if len(input_ids) > 2:
                raise FakeOutOfMemoryError("synthetic CPU-only OOM")
            return SimpleNamespace(scores=[_Scores(len(input_ids))])

    instance = monot5.MonoT5.__new__(monot5.MonoT5)
    instance.model = Model()
    instance.tokenizer = _Tokenizer()
    instance.device = "cpu"
    instance._replicas = [(instance.tokenizer, instance.model)]
    instance._replica_devices = [instance.device]
    instance.token_false_id = 0
    instance.token_true_id = 1

    contents = [["d0", "d1", "d2", "d3", "d4"]]
    ids = [["0", "1", "2", "3", "4"]]
    instance._pure(["q"], contents, ids, top_k=1, batch=4)

    report = instance.pop_last_forward_execution_report()
    assert report is not None
    assert report["schema"] == MONOT5_FORWARD_EXECUTION_SCHEMA
    assert report["requested_forward_microbatch"] == 4
    assert report["successful_forward_microbatches"] == [2, 2, 1]
    assert report["actual_forward_microbatch"] == 2
    assert report["oom_fallback_count"] == 1
    assert report["failed_forward_microbatches"] == [4]
    assert empty_cache_calls == [None]
    assert instance.pop_last_forward_execution_report() is None
