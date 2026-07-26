"""Unit test for the reranker fanout semantic fix.

Locks in the contract: ``_get_reranker_input_fanout`` returns the per-user
work multiplier at the reranker stage, which equals the retrieval
``top_k`` (= number of pairs scored), NOT the YAML's
``passage_reranker.top_k`` (= post-rerank kept count).

The two differ whenever the optimizer picks retrieval top_k > rerank
top_k (the typical "retrieve broad, rerank narrow" pattern). Before this
fix the cost model used the kept count and severely under-estimated the
reranker workload — locking in here so we don't regress.
"""

from __future__ import annotations

import pytest

from rag_stack.cost_model.sweep_inputs import _get_reranker_input_fanout


def _make_pipeline(retrieval_top_k, rerank_top_k):
    nodes = []
    if retrieval_top_k is not None:
        nodes.append({
            "stage": "semantic_retrieval",
            "top_k": retrieval_top_k,
            "modules": [{"component": "vectordb", "vectordb": "idx"}],
        })
    nodes.append({
        "stage": "passage_reranker",
        "top_k": rerank_top_k,
        "modules": [{"component": "monot5"}],
    })
    return {"node_lines": [{"node_line_name": "x", "nodes": nodes}]}


def test_returns_retrieval_top_k_not_rerank_top_k():
    """The textbook bug: retrieve 16, rerank 4. Reranker work = 16, not 4."""
    cfg = _make_pipeline(retrieval_top_k=16, rerank_top_k=4)
    assert _get_reranker_input_fanout(cfg) == 16


def test_retrieve_narrow_rerank_kept_high_does_not_matter():
    """Reranker kept count never affects rerank workload."""
    cfg = _make_pipeline(retrieval_top_k=4, rerank_top_k=16)
    # Even though rerank.top_k=16, reranker only saw 4 retrieved passages
    # so workload = 4. (Real reranker would still only score 4 pairs.)
    assert _get_reranker_input_fanout(cfg) == 4


def test_swept_list_uses_max():
    """When top_k is a swept list (sweep didn't get scalarified yet) we
    pick the worst case (max). Defensive — by the time the cost model
    runs, ``resolve_pipeline_block`` should have scalarified, but the
    helper handles the lossy path safely."""
    cfg = _make_pipeline(retrieval_top_k=[1, 4, 16], rerank_top_k=4)
    assert _get_reranker_input_fanout(cfg) == 16


def test_missing_retrieval_node_falls_back_to_default():
    """No retrieval node → no way to know; use the documented default."""
    cfg = {"node_lines": [{"nodes": [{"stage": "passage_reranker", "top_k": 4}]}]}
    assert _get_reranker_input_fanout(cfg, default=8) == 8


def test_empty_node_lines():
    """No nodes at all → default."""
    assert _get_reranker_input_fanout({}, default=16) == 16
    assert _get_reranker_input_fanout({"node_lines": []}, default=16) == 16
    assert _get_reranker_input_fanout({"node_lines": [{"nodes": []}]}, default=16) == 16


# ---------------------------------------------------------------------------
# Query-expansion fanout (N) — module-type defaults + DAG override
# ---------------------------------------------------------------------------


from rag_stack.cost_model.sweep_inputs import (
    _get_query_expansion_fanout,
    _get_retrieval_backend_fanout,
)


def _with_qe(component, retrieval_top_k=8):
    """Build a pipeline_config that has a query_expansion node + retrieval."""
    return {"node_lines": [{
        "node_line_name": "x",
        "nodes": [
            {"stage": "query_expansion",
             "modules": [{"component": component}]},
            {"stage": "semantic_retrieval", "top_k": retrieval_top_k,
             "modules": [{"component": "vectordb", "vectordb": "idx"}]},
        ],
    }]}


def test_qe_fanout_no_query_expansion_returns_one():
    """No query_expansion node → fanout = 1 (no rewrite happens)."""
    cfg = {"node_lines": [{"nodes": [
        {"stage": "semantic_retrieval", "top_k": 8,
         "modules": [{"component": "vectordb", "vectordb": "idx"}]},
    ]}]}
    assert _get_query_expansion_fanout(cfg) == 1


def test_qe_fanout_module_type_defaults():
    """Known components use their static defaults; unknown → 1.

    See ``_QUERY_EXPANSION_MODULE_DEFAULT_FANOUT`` in ``static_assembly`` for
    the rationale behind each number (hyde:1 is code-enforced;
    query_decompose:2 and multi_query_expansion:4 are prompt-derived
    heuristics).
    """
    assert _get_query_expansion_fanout(_with_qe("hyde")) == 1
    assert _get_query_expansion_fanout(_with_qe("query_decompose")) == 2
    assert _get_query_expansion_fanout(_with_qe("multi_query_expansion")) == 4
    # Unknown component — should NOT crash, falls through to 1.
    assert _get_query_expansion_fanout(_with_qe("some_unknown_expander")) == 1


def test_qe_fanout_dag_overrides_default():
    """Aggregate DAG payload from a quality run wins over the static
    module-type default. Ceil(measured) — fanout is the workload value
    (avg sub-queries per user), NOT a batching/scheduling decision."""
    cfg = _with_qe("hyde")  # default would be 1
    dag = {"shape": "aggregate", "query_expansion_fanout": 3.2}
    assert _get_query_expansion_fanout(cfg, dag=dag) == 4  # ceil(3.2)


def test_qe_fanout_dag_minimum_one():
    """Pathological measurement (0.0 or negative) clamps to 1."""
    cfg = _with_qe("query_decompose")
    assert _get_query_expansion_fanout(
        cfg, dag={"shape": "aggregate", "query_expansion_fanout": 0.0},
    ) == 1


def test_qe_fanout_dag_ignored_when_not_aggregate():
    """Per-query DAG (FlashRAG list shape) is not consumed by this helper —
    falls through to module-type default."""
    cfg = _with_qe("multi_query_expansion")
    # multi_query_expansion default = 4 (see provenance comment in static_assembly).
    assert _get_query_expansion_fanout(cfg, dag=[{"query_id": "q1"}]) == 4


# ---------------------------------------------------------------------------
# Reranker fanout interaction with query expansion
# ---------------------------------------------------------------------------


def test_reranker_fanout_not_multiplied_by_query_expansion():
    """Retrieval FUSES the N query-expansion sub-queries back down to top_k
    (RRF / even-distribute) BEFORE the reranker, so the reranker scores exactly
    retrieval_top_k (query, passage) pairs per user — NOT top_k × N. The
    expansion fanout N is already charged at retrieval/encode and must not be
    double-counted at the reranker."""
    cfg = _make_pipeline(retrieval_top_k=8, rerank_top_k=4)
    # Inject a query_expansion node
    cfg["node_lines"][0]["nodes"].insert(0, {
        "stage": "query_expansion",
        "modules": [{"component": "multi_query_expansion"}],
    })
    # multi_query_expansion default N = 4, but retrieval pre-merges to top_k → 8.
    assert _get_reranker_input_fanout(cfg) == 8

    # A measured DAG carrying the QE fanout does NOT change the reranker input
    # count — still top_k, because the merge happens inside the retrieval node.
    dag = {"shape": "aggregate", "query_expansion_fanout": 3.2}
    assert _get_reranker_input_fanout(cfg, dag=dag) == 8

    dag2 = {"shape": "aggregate", "query_expansion_fanout": 5.5}
    assert _get_reranker_input_fanout(cfg, dag=dag2) == 8


def test_retrieval_backend_mean_fanout_is_not_rounded_for_demand():
    cfg = _with_qe("hyde")
    dag = {
        "shape": "aggregate",
        "retrieval_backend_fanout": 1.5,
    }

    assert _get_retrieval_backend_fanout(cfg, dag=dag) == 1.5


def test_reranker_fanout_no_query_expansion_unchanged():
    """Without query_expansion, reranker fanout reduces to retrieval top_k
    (previous fix's behavior)."""
    cfg = _make_pipeline(retrieval_top_k=16, rerank_top_k=4)
    assert _get_reranker_input_fanout(cfg) == 16  # × 1 implicit


# ---------------------------------------------------------------------------
# RAGSweep._apply_fanout math (the work-multiplier transformation)
# ---------------------------------------------------------------------------


def test_apply_fanout_identity_for_one_or_none():
    """fanout in {None, 0, 1} → no transformation."""
    import pandas as pd
    from rago.rago_sweep import RAGSweep

    df = pd.DataFrame({
        "batch_size": [1, 2, 4, 8],
        "latency_s": [0.01, 0.02, 0.04, 0.08],
        "qps": [100, 200, 400, 800],
        "qps_per_chip": [50, 100, 200, 400],
    })
    for fanout in (None, 0, 1):
        out = RAGSweep._apply_fanout(df.copy(), fanout)
        pd.testing.assert_frame_equal(out, df)


def test_apply_fanout_large_batch_divides():
    """When max_batch_size >= fanout: filter rows >= fanout, divide batch by fanout."""
    import pandas as pd
    from rago.rago_sweep import RAGSweep

    df = pd.DataFrame({
        "batch_size": [1, 2, 4, 8, 16, 32],
        "latency_s": [0.01, 0.02, 0.04, 0.08, 0.16, 0.32],
        "qps": [100, 200, 400, 800, 1600, 3200],
        "qps_per_chip": [50, 100, 200, 400, 800, 1600],
    })
    out = RAGSweep._apply_fanout(df, fanout=4)
    # rows < 4 dropped, kept: batch_size in {4,8,16,32} → user units {1,2,4,8}
    assert list(out["batch_size"]) == [1, 2, 4, 8]
    # Latency / qps unchanged at this stage (the CALL latency is the same,
    # we just relabeled what each call services).
    assert list(out["latency_s"]) == [0.04, 0.08, 0.16, 0.32]


def test_apply_fanout_nonpow2_buckets_to_pow2_user_batch():
    """fanout=3 (non-pow2 — real measured workload). User-batch axis
    must stay pow2 (latency lookup invariant). Work-batch / fanout
    floored to the nearest pow2 bucket; same-bucket collisions keep
    the lower-latency row."""
    import pandas as pd
    from rago.rago_sweep import RAGSweep

    df = pd.DataFrame({
        "batch_size": [1, 2, 4, 8, 16, 32, 64],
        "latency_s": [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64],
        "qps": [100, 200, 400, 800, 1600, 3200, 6400],
        "qps_per_chip": [50, 100, 200, 400, 800, 1600, 3200],
    })
    out = RAGSweep._apply_fanout(df, fanout=3)
    # Drop rows < 3, then for each remaining row: user_batch = floor_pow2(work / 3)
    #   4/3=1.33 → 1, 8/3=2.67 → 2, 16/3=5.33 → 4, 32/3=10.67 → 8, 64/3=21.33 → 16
    # All buckets distinct → 5 rows.
    assert list(out["batch_size"]) == [1, 2, 4, 8, 16]
    # All values are pow2 (the invariant get_latency_from_pareto needs).
    assert all((b & (b - 1)) == 0 for b in out["batch_size"])


def test_apply_fanout_small_batch_scales_latency():
    """When max_batch_size < fanout: clamp to max row, latency × (fanout/max),
    qps × (max/fanout), batch_size = 1."""
    import pandas as pd
    from rago.rago_sweep import RAGSweep

    df = pd.DataFrame({
        "batch_size": [1, 2, 4],
        "latency_s": [0.01, 0.02, 0.04],
        "qps": [100, 200, 400],
        "qps_per_chip": [50, 100, 200],
    })
    out = RAGSweep._apply_fanout(df, fanout=8)  # max_batch=4 < 8
    # Clamp to the max_batch row (batch_size=4) and scale.
    assert len(out) == 1
    row = out.iloc[0]
    assert row["batch_size"] == 1                            # 1 user/logical-call
    assert row["latency_s"] == 0.04 * (8 / 4)                # 2× as many calls
    assert row["qps"] == int(400 * 4 / 8)                    # halved
    assert row["qps_per_chip"] == int(200 * 4 / 8)


# ---------------------------------------------------------------------------
# RetrievalPolicy plumbing — verify static_assembly hands fanout to RAGO
# ---------------------------------------------------------------------------


def test_retrieval_policy_carries_query_expansion_fanout():
    """The new RAGO field is recognized + survives construction."""
    from rago.config import RetrievalPolicy

    p = RetrievalPolicy(query_expansion_fanout=3)
    assert p.query_expansion_fanout == 3

    p = RetrievalPolicy()
    assert p.query_expansion_fanout is None


# ---------------------------------------------------------------------------
# Per-stage fanout helper used to extend sweep range so user-unit Pareto
# reaches max_batch_size_request even with query_expansion > 1.
# ---------------------------------------------------------------------------


from rag_stack.cost_model.sweep_inputs import _get_stage_fanout


def test_stage_fanout_no_query_expansion():
    """Without query_expansion, every stage has fanout = 1 (call-unit ==
    user-unit, no transform needed)."""
    cfg = {"node_lines": [{"nodes": [
        {"stage": "semantic_retrieval", "top_k": 16,
         "modules": [{"component": "vectordb", "vectordb": "idx"}]},
        {"stage": "passage_reranker", "top_k": 4,
         "modules": [{"component": "monot5"}]},
    ]}]}
    assert _get_stage_fanout("semantic_retrieval_vectorsearch", cfg) == 1
    assert _get_stage_fanout("semantic_retrieval_encode", cfg) == 1
    # Reranker: no q_e, so N=1, but reranker fanout = retrieval_top_k × N = 16.
    assert _get_stage_fanout("passage_reranker", cfg) == 16
    # LLM stages: always 1.
    assert _get_stage_fanout("generator_prefill", cfg) == 1
    assert _get_stage_fanout("generator_decode", cfg) == 1
    assert _get_stage_fanout("query_expansion_prefill", cfg) == 1


@pytest.mark.parametrize("stage,expected", [
    ("semantic_retrieval_vectorsearch", 4),
    ("semantic_retrieval_encode", 4),
    ("passage_reranker", 8),
    ("generator_prefill", 1),
])
def test_stage_fanout_with_multi_query_expansion(stage, expected):
    """Expansion scales dense retrieval; fused top-k bounds downstream work."""
    cfg = {"node_lines": [{"nodes": [
        {"stage": "query_expansion",
         "modules": [{"component": "multi_query_expansion"}]},
        {"stage": "semantic_retrieval", "top_k": 8,
         "modules": [{"component": "vectordb", "vectordb": "idx"}]},
        {"stage": "passage_reranker", "top_k": 4,
         "modules": [{"component": "monot5"}]},
    ]}]}
    assert _get_stage_fanout(stage, cfg) == expected


def test_stage_fanout_dag_measurement_overrides():
    """When the quality run emits an aggregate DAG, measured fanout wins
    over the static module-type default."""
    cfg = {"node_lines": [{"nodes": [
        {"stage": "query_expansion",
         "modules": [{"component": "hyde"}]},   # default 1
        {"stage": "semantic_retrieval", "top_k": 8,
         "modules": [{"component": "vectordb", "vectordb": "idx"}]},
    ]}]}
    # No DAG: fanout = static default for hyde = 1.
    assert _get_stage_fanout("semantic_retrieval_vectorsearch", cfg) == 1
    # Batch-ceiling sizing rounds observed fanout up; it does not reject a
    # valid multi-list observation or keep the component's default of one.
    dag = {"shape": "aggregate", "query_expansion_fanout": 2.7}
    assert _get_stage_fanout("semantic_retrieval_vectorsearch", cfg, dag=dag) == 3
    assert _get_stage_fanout("semantic_retrieval_encode", cfg, dag=dag) == 3


def test_stage_fanout_unknown_stage_is_one():
    """Stages outside the fanout-aware set default to 1."""
    cfg = {"node_lines": [{"nodes": [
        {"stage": "query_expansion",
         "modules": [{"component": "hyde"}]},
    ]}]}
    # Even with q_e present, prefill/decode are 1 (LLM call is one shot).
    assert _get_stage_fanout("generator_prefill", cfg) == 1
    assert _get_stage_fanout("generator_decode", cfg) == 1
    assert _get_stage_fanout("query_expansion_decode", cfg) == 1
    assert _get_stage_fanout("passage_filter", cfg) == 1
    assert _get_stage_fanout("passage_compressor_prefill", cfg) == 1
    assert _get_stage_fanout("some_random_stage", cfg) == 1
