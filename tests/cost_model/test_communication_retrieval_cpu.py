import pandas as pd
import pytest

from rag_stack.cost_model.communication import (
    CommunicationPayloadModel,
    apply_communication_costs,
)
from rag_stack.types import SystemConfig


def test_retrieval_edges_are_host_transfers_even_inside_same_group():
    df = pd.DataFrame(
        [
            {
                "latency_s": 1.0,
                "qps": 1e12,
                "qps_per_chip": 1e12,
                "collocation_strategy": [
                    ["semantic_retrieval_encode", "generator_prefill"],
                    ["semantic_retrieval_vectorsearch"],
                ],
                "pipeline_stage_order": [
                    "semantic_retrieval_encode",
                    "semantic_retrieval_vectorsearch",
                    "generator_prefill",
                ],
                "candidate_system_layout": {
                    "performance_stages": {
                        "semantic_retrieval_encode": {"devices": ["cuda:0"]},
                        "semantic_retrieval_vectorsearch": {"devices": []},
                        "generator_prefill": {"devices": ["cuda:1"]},
                    },
                },
            }
        ]
    )
    hardware = SystemConfig(interconnect={
        "gpu_gpu": {"bandwidth_gbps": 300, "latency_us": 0.4},
        "gpu_cpu": {"bandwidth_gbps": 20, "latency_us": 2},
    })
    pipeline_config = {
        "system": {"interconnect": hardware.interconnect},
        "_token_stats": {
            "avg_chunk_tokens": 100,
            "prompt_template_tokens": 10,
            "avg_output_tokens": 20,
            "avg_query_tokens": 5,
            "bytes_per_token": 4.0,
        },
        "node_lines": [
            {
                "nodes": [
                    {
                        "stage": "semantic_retrieval",
                        "modules": [{"component": "vectordb", "vectordb": "vdb"}],
                    }
                ]
            }
        ],
        "vectordb": [{
            "name": "vdb", "embedding_dim": 768,
            "embedding_dtype_bytes": 4,
        }],
    }

    out = apply_communication_costs(
        df,
        hardware=hardware,
        pipeline_config=pipeline_config,
        seq_len_prefill=128,
        qe_seq_len=None,
        reranker_input_fanout=1,
        query_expansion_fanout=1,
    )

    boundaries = out["comm_boundaries"].iloc[0]
    assert "semantic_retrieval_encode\u2192semantic_retrieval_vectorsearch" in boundaries
    assert "semantic_retrieval_vectorsearch\u2192generator_prefill" in boundaries
    assert out["comm_latency_s"].iloc[0] == pytest.approx(
        2 * 2e-6 + (768 * 4 + 128 * 4) / 20e9
    )


def test_des_payload_prefers_trace_input_bytes_with_shape_edge_exceptions(
    monkeypatch,
):
    monkeypatch.setattr(
        "rag_stack.cost_model.communication._genz_model_for",
        lambda *_args, **_kwargs: None,
    )
    config = {
        "_token_stats": {
            "avg_chunk_tokens": 100,
            "prompt_template_tokens": 10,
            "avg_output_tokens": 20,
            "avg_query_tokens": 5,
            "bytes_per_token": 4.0,
        },
        "node_lines": [{"nodes": [{
            "stage": "semantic_retrieval",
            "modules": [{"vectordb": "vdb"}],
        }]}],
        "vectordb": [{
            "name": "vdb", "embedding_dim": 768,
            "embedding_dtype_bytes": 4,
        }],
    }
    payload = CommunicationPayloadModel(
        config,
        seq_len_prefill=128,
        qe_seq_len=None,
        reranker_input_fanout=16,
        query_expansion_fanout=1,
    )

    assert payload.payload_bytes(
        "passage_reranker",
        "generator_prefill",
        destination_input_tokens=100,
        destination_input_bytes=1234,
    ) == 1234.0
    # Old traces without input_bytes retain the token-based fallback.
    assert payload.payload_bytes(
        "passage_reranker",
        "generator_prefill",
        destination_input_tokens=100,
    ) == 400.0
    # Embedding/KV tensor edges are shape-derived, never serialized text.
    assert payload.payload_bytes(
        "semantic_retrieval_encode",
        "semantic_retrieval_vectorsearch",
        destination_input_bytes=9999,
    ) == 768 * 4
