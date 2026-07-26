"""Trace contract, derived workload summaries, and analytical per-call pricing.

These tests inject simple component costs to check the trace walk without
GenZ/FAISS. Candidate construction and simulation have separate integration
coverage. A summary supplies representative component inputs, not final QPS.
"""

from __future__ import annotations

import pytest

from rag_stack.rag_ir.trace import (
    is_trace_payload,
    is_quality_trace_candidate,
    make_quality_trace_envelope,
    normalize_quality_trace,
)
from rag_stack.cost_model.engines import (
    CallLatency,
    ChainPerf,
    _closed_loop_load_concurrency,
    _closed_loop_mean_latency,
    _profile,
    _compute_chain_latency,
)


# ---------------------------------------------------------------------------
# Analytic call-latency — linear in tokens, batch-independent. decode_s is a
# constant PER-STEP latency (GenZ semantics); the walk multiplies it by the
# output-token count. Keeps every expected latency hand-checkable.
# ---------------------------------------------------------------------------

def _analytic_latency() -> CallLatency:
    # prefill/decode accept (tp, pp) — the walk passes the deployment's
    # parallelism. TP shrinks per-call compute linearly here (1/tp) so a TP-aware
    # test can observe it; pp ignored.
    return CallLatency(
        prefill_s=lambda inp, b, tp=1, pp=1: inp * 1e-6 / tp,
        decode_s=lambda inp, out, b, tp=1, pp=1: 2e-6 / tp,   # per decode step
        retrieve_s=lambda b: 1e-3,
        bm25_retrieve_s=lambda b: 3e-4,
        rerank_s=lambda inp, b: inp * 5e-7,
    )


def _gen(inp, out):
    return {"stage": "generator", "input_tokens": inp, "output_tokens": out, "step_idx": 0}


def _ret(model_id=None, stage="semantic_retrieval_vectorsearch"):
    return {"stage": stage, "input_tokens": 16, "output_tokens": 0,
            "step_idx": 0, "model_id": model_id}


# ---------------------------------------------------------------------------
# Trace contract: normalize_quality_trace / is_trace_payload
# ---------------------------------------------------------------------------



def test_is_trace_payload_recognizes_only_the_canonical_envelope():
    envelope = make_quality_trace_envelope(
        [[_quality_gen(10, 3, input_bytes=40)]], question_ids=["q0"],
    )
    assert is_trace_payload(envelope) is True
    # legacy/bare forms are NOT trace payloads — routers must fail them
    # closed (directed converter error), never route them anywhere silently.
    assert is_trace_payload([]) is False
    assert is_trace_payload([[{"stage": "generator"}]]) is False
    assert is_trace_payload({"shape": "aggregate"}) is False  # static_gt aggregate
    assert is_trace_payload([{"query_id": "q"}]) is False
    assert is_trace_payload(None) is False
    assert is_trace_payload({
        "trace_scope": "measured_only", "queries": [],
    }) is False
    assert is_quality_trace_candidate({"trace_scope": "unknown", "queries": []})


def _quality_gen(
    inp: int,
    out: int,
    *,
    input_bytes: int,
    step_idx: int = 0,
) -> dict:
    return {
        "stage": "generator",
        "input_tokens": inp,
        "output_tokens": out,
        "input_bytes": input_bytes,
        "step_idx": step_idx,
        "model_id": "generator-model",
    }


def test_canonical_envelope_keeps_distinct_same_shape_questions():
    # Equal token/byte lengths are workload shapes, not question identities.
    # MSMARCO has 100 unique questions but only 75 such first-call shapes —
    # the canonical envelope keys on question_id, so nothing is shape-deduped.
    calls_a = [_quality_gen(10, 3, input_bytes=40)]
    calls_b = [_quality_gen(10, 7, input_bytes=40)]
    payload = make_quality_trace_envelope(
        [calls_a, calls_b], question_ids=["q-a", "q-b"],
    )

    normalized = normalize_quality_trace(payload)

    assert normalized.traces == [calls_a, calls_b]
    assert normalized.normalization == "producer_projection"
    assert normalized.kept_queries == 2
    assert normalized.lossy is False
    assert is_trace_payload(payload) is True


def test_normalize_rejects_bare_list_with_converter_direction():
    calls = [_quality_gen(10, 3, input_bytes=40)]
    with pytest.raises(ValueError, match="convert_legacy_traces.py"):
        normalize_quality_trace([calls, [dict(calls[0])]])


def test_normalize_rejects_scoped_envelope_fail_closed():
    payload = {
        "trace_scope": "measured_only",
        "queries": [[_quality_gen(10, 3, input_bytes=40)]],
    }
    with pytest.raises(ValueError, match="unsupported fields"):
        normalize_quality_trace(payload)
    payload["trace_scope"] = "dataset_quality_first_completion"
    with pytest.raises(ValueError, match="unsupported fields"):
        normalize_quality_trace(payload)
    with pytest.raises(TypeError, match="convert_legacy_traces.py"):
        normalize_quality_trace("not a trace at all")


def test_quality_trace_rejects_unknown_stage_and_envelope_metadata():
    with pytest.raises(ValueError, match="unsupported stage"):
        make_quality_trace_envelope([[{
            "stage": "measured_generator_worker",
            "input_tokens": 10,
            "output_tokens": 3,
            "step_idx": 0,
        }]], question_ids=["q0"])

    with pytest.raises(ValueError, match="unsupported fields"):
        normalize_quality_trace({
            "queries": [{
                "question_id": "q0",
                "calls": [_quality_gen(10, 3, input_bytes=40)],
            }],
            "performance": {"qps": 999.0},
        })


def test_canonical_envelope_rejects_incomplete_duplicate_disorder_and_empty_id():
    incomplete = [{
        "stage": "semantic_retrieval_vectorsearch",
        "input_tokens": 5,
        "output_tokens": 0,
        "step_idx": 0,
    }]
    with pytest.raises(ValueError, match="incomplete"):
        normalize_quality_trace({
            "queries": [{"question_id": "q0", "calls": incomplete}],
        })

    complete = [_quality_gen(10, 3, input_bytes=40)]
    with pytest.raises(ValueError, match="duplicate"):
        normalize_quality_trace({
            "queries": [
                {"question_id": "same", "calls": complete},
                {"question_id": "same", "calls": [dict(complete[0])]},
            ],
        })

    bad_order = [
        _quality_gen(10, 3, input_bytes=40, step_idx=1),
        _quality_gen(20, 4, input_bytes=80, step_idx=0),
    ]
    with pytest.raises(ValueError, match="step order"):
        normalize_quality_trace({
            "queries": [{"question_id": "q0", "calls": bad_order}],
        })

    for empty_identity in ("", "   "):
        with pytest.raises(ValueError, match="question_id"):
            normalize_quality_trace({
                "queries": [{
                    "question_id": empty_identity,
                    "calls": [_quality_gen(10, 3, input_bytes=40)],
                }],
            })


# ---------------------------------------------------------------------------
# Routing contract (mirrors the controller's CM dispatch)
# ---------------------------------------------------------------------------

def test_routing_contract_envelope_aggregate_and_legacy():
    """The controller routes: canonical envelope → DynamicAssembly; static_gt
    aggregate (dict) / None → StaticAssembly; legacy bare lists are never
    routed anywhere — they get a directed converter error. Routing never
    inspects pipeline.mode, so it is mode-independent."""
    from rag_stack.cost_model import compute_trace_performance, RAGCMAssembly
    envelope = make_quality_trace_envelope(
        [[_quality_gen(500, 50, input_bytes=2000)]], question_ids=["0"],
    )
    aggregate = {"shape": "aggregate", "query_expansion_fanout": 1.0}
    legacy_list = [[_gen(500, 50)], [_gen(500, 50), _gen(900, 50)]]
    assert is_trace_payload(envelope) and not is_quality_trace_candidate(aggregate)
    assert not is_trace_payload(aggregate)
    assert not is_trace_payload(None)
    assert not is_trace_payload(legacy_list)   # → directed error arm
    assert callable(compute_trace_performance)
    assert RAGCMAssembly is not None


def test_closed_loop_analytical_load_ignores_measured_harness_fields():
    assert _closed_loop_load_concurrency({}, 32) == 32
    assert _closed_loop_load_concurrency(
        {"measured_load_concurrency": 640}, 32,
    ) == 32
    assert _closed_loop_load_concurrency(
        {"batching": {"load_factor": 20}}, 32,
    ) == 32
    assert _closed_loop_load_concurrency(
        {"batching": {"load_factor": 20, "max_load_concurrency": 256}}, 32,
    ) == 32


def test_closed_loop_mean_latency_is_queue_aware_mean():
    assert _closed_loop_mean_latency(1.5, qps=26.0, load_concurrency=128) == pytest.approx(128 / 26.0)
    assert _closed_loop_mean_latency(10.0, qps=26.0, load_concurrency=128) == pytest.approx(10.0)
    assert _closed_loop_mean_latency(1.5, qps=0.0, load_concurrency=128) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# _profile compatibility alias — representative component inputs from traces
# ---------------------------------------------------------------------------

def test_profile_prefill_fanout_is_mean_generate_rounds_and_float():
    """prefill_fanout = mean generate calls/query, kept a FLOAT (no ceil)."""
    traces = [
        [_ret(), _gen(800, 50), _ret(), _gen(1200, 40)],  # 2 generate
        [_ret(), _gen(850, 60)],                          # 1 generate
    ]
    agg = _profile(traces)
    assert agg["prefill_fanout"] == pytest.approx(1.5)     # (2+1)/2, FLOAT not ceiled
    assert agg["num_retrievals"] == pytest.approx(1.5)     # FLOAT mean handoffs (not rounded)
    assert agg["shape"] == "aggregate"
    assert agg["rag_ir_mode"] == "dynamic"                # so StaticAssembly consumes token_stats


def test_profile_decode_aggregates_output_sum_prefill_keeps_percall():
    """Preserve the component adapter's distinct input/output averaging units."""
    traces = [
        [_gen(800, 50), _gen(1200, 40)],   # Σ out = 90 ; inputs 800,1200
        [_gen(850, 60)],                    # Σ out = 60 ; input 850
    ]
    ts = _profile(traces)["token_stats"]
    assert ts["avg_generator_output_tokens"] == pytest.approx((90 + 60) / 2)   # 75, per-query SUM
    assert ts["avg_generator_input_tokens"] == int((800 + 1200 + 850) / 3)     # 950, per-CALL mean


def test_profile_retrieval_invocations_count_dense_worker_visits_only():
    """R = mean fused DENSE worker visits/query; bm25 is excluded."""
    traces = [
        [_ret(), _ret(), _gen(500, 50)],            # 2 dense
        [_ret(), _ret(model_id="bm25", stage="lexical_retrieval"), _gen(500, 50)],  # 1 dense + 1 bm25
    ]
    agg = _profile(traces)
    assert agg["retrieval_invocations_per_query"] == pytest.approx(1.5)


def test_profile_pure_retrieval_clamps_fanout_to_one():
    """No generate calls → prefill_fanout clamps to 1.0 (no negative/zero)."""
    agg = _profile([[_ret(), _ret()] for _ in range(3)])
    assert agg["prefill_fanout"] == pytest.approx(1.0)
    assert agg["num_retrievals"] == 1


# ---------------------------------------------------------------------------
# _compute_chain_latency — the per-query serial compute-chain walk
# ---------------------------------------------------------------------------

def test_chain_latency_sums_ops_serially():
    """One query: retrieve + (prefill + decode×out). Hand-checked."""
    traces = [[_ret(), _gen(800, 50)]]
    e2e, ttft, tpot = _compute_chain_latency(traces, _analytic_latency())
    # retrieve 1e-3 + prefill 800e-6 + decode 2e-6×50=1e-4  = 1.9e-3
    assert e2e == pytest.approx(1e-3 + 800e-6 + 2e-6 * 50)
    # TTFT (mean) = retrieve + answer prefill × the avg-queue fraction (the average
    # request's first token follows ~half the batch prefill under continuous batching).
    from rag_stack.cost_model.engines import _TTFT_PREFILL_FRAC
    assert ttft == pytest.approx(1e-3 + 800e-6 * _TTFT_PREFILL_FRAC)
    assert tpot == pytest.approx(2e-6)


def test_chain_latency_k2_batch16_uses_structured_retrieval_worker_units():
    """The analytical cap uses the DES worker plan for a K=2, B=16 visit."""
    seen = {"encode": [], "search": [], "backend": [], "fusion": []}

    def encode_s(batch: int) -> float:
        seen["encode"].append(batch)
        return 0.01

    def search_s(batch: int) -> float:
        seen["search"].append(batch)
        return 0.02

    def backend_s(batch: int) -> float:
        seen["backend"].append(batch)
        return 0.05 + 0.003 * batch

    def fusion_s(batch: int) -> float:
        seen["fusion"].append(batch)
        return 0.001 * batch

    latency = CallLatency(
        encode_s=encode_s,
        retrieve_s=lambda batch: 0.02 + 0.05 + 0.004 * batch,
        retrieve_kernel_s=search_s,
        retrieve_backend_overhead_s=backend_s,
        rrf_s=fusion_s,
        retrieval_fanout=2,
    )
    traces = [[
        {"stage": "semantic_retrieval_encode", "input_tokens": 7},
        {"stage": "semantic_retrieval_vectorsearch", "input_tokens": 7},
        {"stage": "retrieval_fusion", "model_id": "rrf_pure", "input_tokens": 0,
         "input_profile": {"items": 4, "list_sizes": [2, 2], "unique_items": 3},
         "output_profile": {"items": 2, "unique_items": 2}},
    ]]

    perf = _compute_chain_latency(traces, latency, batch=16)

    assert seen == {
        "encode": [16, 16],
        "search": [16, 16],
        "backend": [16, 16],
        "fusion": [16],
    }
    assert perf.svc_encode == pytest.approx(0.02)
    assert perf.svc_retrieval == pytest.approx(0.252)
    assert perf.svc_fusion == pytest.approx(0.016)
    assert perf.e2e == pytest.approx(0.272)


def test_chain_latency_is_tp_aware():
    """The walk passes the per-stage (tp,pp) through to GenZ — single-query
    latency shrinks with TP (it is NOT pinned to tp=1)."""
    traces = [[_gen(1000, 20)]]
    lat = _analytic_latency()
    tp1 = _compute_chain_latency(traces, lat, 1, 1, 1, 1)[0]
    tp2 = _compute_chain_latency(traces, lat, 2, 1, 2, 1)[0]
    # prefill 1000e-6/tp + decode (2e-6/tp)×20 ; tp=2 halves both → tp2 == tp1/2
    assert tp2 == pytest.approx(tp1 / 2)


def test_chain_latency_decode_burst_scales_with_output():
    """decode contributes per-step × output_tokens (the burst)."""
    short = _compute_chain_latency([[_gen(500, 10)]], _analytic_latency())[0]
    long = _compute_chain_latency([[_gen(500, 100)]], _analytic_latency())[0]
    # both share prefill 500e-6; decode differs by (100-10)×2e-6
    assert long - short == pytest.approx((100 - 10) * 2e-6)


def test_chain_latency_more_rounds_raises_latency():
    """Variable agentic round count is captured purely via the trace length."""
    one = [[_gen(500, 50)] for _ in range(8)]
    three = [[_gen(500, 50), _gen(900, 50), _gen(1300, 50)] for _ in range(8)]
    lat = _analytic_latency()
    assert _compute_chain_latency(three, lat)[0] > _compute_chain_latency(one, lat)[0]


def test_chain_latency_larger_prefill_raises_latency():
    small = _compute_chain_latency([[_gen(100, 50)]], _analytic_latency())[0]
    large = _compute_chain_latency([[_gen(4000, 50)]], _analytic_latency())[0]
    assert large > small
    assert large - small == pytest.approx((4000 - 100) * 1e-6)  # prefill is linear here


def test_chain_latency_bm25_cheaper_than_dense_retrieve():
    """bm25 keyword retrieve is the flat CPU cost, not the GPU encode+FAISS path."""
    dense = _compute_chain_latency([[_ret()]], _analytic_latency())[0]
    bm25 = _compute_chain_latency([[_ret(model_id="bm25", stage="lexical_retrieval")]], _analytic_latency())[0]
    assert dense == pytest.approx(1e-3)
    assert bm25 == pytest.approx(3e-4)
    assert bm25 < dense


def test_chain_latency_empty_trace_is_zero():
    e2e, ttft, tpot = _compute_chain_latency([], _analytic_latency())
    assert e2e == 0.0 and ttft == 0.0 and tpot == 0.0


# ---------------------------------------------------------------------------
# WorkflowSchema — rag_ir structure / dataflow dispatch (continuous-batching only)
# ---------------------------------------------------------------------------

from rag_stack.rag_ir.workflow_schema import (
    WorkflowSchema, dataflow_type_of, build_workflow_schema,
)


def test_dataflow_type_sequential_vs_agentic():
    assert dataflow_type_of({}) == "sequential"
    assert dataflow_type_of({"pipeline_runtime": {"mode": "sequential"}}) == "sequential"
    assert dataflow_type_of({"pipeline_runtime": {"mode": "ircot"}}) == "agentic"
    assert dataflow_type_of(
        {"pipeline_runtime": {"mode": "sequential", "rag_dataflow": "react"}}
    ) == "agentic"


def test_build_workflow_schema_flags_and_dispatch():
    schema = build_workflow_schema(
        {"pipeline_runtime": {"rag_dataflow": "react"}},
        stages=["semantic_retrieval_encode", "semantic_retrieval_vectorsearch", "generator_prefill", "generator_decode"],
    )
    assert schema.is_agentic and schema.dataflow_type == "agentic"
    assert schema.has_encode and not schema.has_reranker


def test_decode_batch_independent_of_request_batch():
    """batch_size_decode (decode concurrency) is independent of the request batch:
    pricing decode at a LARGER batch raises the decode burst (slower per-step) while
    prefill stays at the request batch. Default (decode_batch=0) → decode at request."""
    traces = [[_gen(800, 50)]]
    lat = CallLatency(
        prefill_s=lambda inp, b, tp=1, pp=1: inp * 1e-6,
        # decode per-step RISES with batch b (more sequences share the step).
        decode_s=lambda inp, out, b, tp=1, pp=1: 1e-6 * b,
    )
    base = _compute_chain_latency(traces, lat, batch=8)                    # decode at 8
    bigdec = _compute_chain_latency(traces, lat, batch=8, decode_batch=32)  # decode at 32
    # prefill identical (request batch 8); decode burst 4× (b 8→32) → svc_gen rises by
    # the decode delta only.
    dec8 = 1e-6 * 8 * 50
    dec32 = 1e-6 * 32 * 50
    assert bigdec.svc_gen - base.svc_gen == pytest.approx(dec32 - dec8)
    # decode_batch < request is floored to request (decode holds ≥ the request cohort).
    assert _compute_chain_latency(traces, lat, batch=8, decode_batch=4).svc_gen == \
        pytest.approx(base.svc_gen)


def test_chain_perf_services_sum_to_critical_path():
    """ChainPerf returns BOTH the critical-path latency and the per-resource
    services; for a serial (lock-step) chain the services sum to e2e exactly —
    the invariant that lets qps be built from svc_* yet equal batch/latency."""
    traces = [[_ret(), _gen(800, 50)]]
    p = _compute_chain_latency(traces, _analytic_latency())
    assert isinstance(p, ChainPerf)
    assert p.svc_gen + p.svc_retrieval + p.svc_rerank + p.svc_compress == pytest.approx(p.e2e)
    # generator service = prefill + decode burst; retrieval service = FAISS only.
    assert p.svc_gen == pytest.approx(800e-6 + 50 * 2e-6)


def test_retrieval_service_uses_request_batch():
    """Retrieval has no independent batch axis; it is priced at request batch."""
    traces = [[_ret(), _gen(800, 50)]]
    lat = CallLatency(
        prefill_s=lambda inp, b, tp=1, pp=1: 0.0,
        decode_s=lambda inp, out, b, tp=1, pp=1: 0.0,
        retrieve_s=lambda b: b * 1e-3,
    )
    b1 = _compute_chain_latency(traces, lat, batch=1)
    b8 = _compute_chain_latency(traces, lat, batch=8)
    assert b1.svc_retrieval == pytest.approx(1e-3)
    assert b8.svc_retrieval == pytest.approx(8e-3)


def test_batched_pure_stages_use_request_batch():
    """Dynamic batched pure stages must see the deployed request batch."""
    seen = {"rerank": [], "compress": []}
    traces = [[
        {"stage": "passage_reranker", "input_tokens": 128, "output_tokens": 0},
        {"stage": "passage_compressor", "input_tokens": 256, "output_tokens": 0},
    ]]
    lat = CallLatency(
        rerank_s=lambda inp, b: seen["rerank"].append(b) or 0.1,
        compress_s=lambda inp, b: seen["compress"].append(b) or 0.2,
    )
    perf = _compute_chain_latency(traces, lat, batch=32)

    assert seen["rerank"] == [32]
    assert seen["compress"] == [32]
    assert perf.svc_rerank == pytest.approx(0.1)
    assert perf.svc_compress == pytest.approx(0.2)


def test_generate_stage_dispatch_uses_query_expansion_latency():
    """A trace generate call with stage=query_expansion uses the aux vLLM cost
    model, not the main generator model."""
    traces = [[
        {
            "stage": "query_expansion",
            "input_tokens": 10,
            "output_tokens": 2,
        },
        _gen(10, 2),
    ]]
    lat = CallLatency(
        prefill_by_engine={
            "query_expansion": lambda inp, b, tp=1, pp=1: 0.10,
            "generator": lambda inp, b, tp=1, pp=1: 1.00,
        },
        decode_by_engine={
            "query_expansion": lambda inp, out, b, tp=1, pp=1: 0.01,
            "generator": lambda inp, out, b, tp=1, pp=1: 0.20,
        },
    )
    perf = _compute_chain_latency(
        traces,
        lat,
        overlap_aux_generates=True,
    )
    assert perf.svc_prefill_by_stage["query_expansion_prefill"] == pytest.approx(0.10)
    assert perf.svc_decode_by_stage["query_expansion_decode"] == pytest.approx(0.02)
    assert perf.svc_prefill_by_stage["generator_prefill"] == pytest.approx(1.00)
    assert perf.svc_decode_by_stage["generator_decode"] == pytest.approx(0.40)


def test_sequential_aux_generate_decode_counts_in_service():
    """Sequential query expansion is not an answer stream, but its decode still
    consumes vLLM capacity and must be present in service demand."""
    traces = [[
        {
            "stage": "query_expansion",
            "input_tokens": 10,
            "output_tokens": 5,
        },
        _gen(20, 7),
    ]]
    lat = CallLatency(
        prefill_s=lambda inp, b, tp=1, pp=1: 0.0,
        decode_s=lambda inp, out, b, tp=1, pp=1: 0.01,
    )
    perf = _compute_chain_latency(
        traces,
        lat,
        overlap_aux_generates=True,
    )
    assert perf.svc_decode_by_stage["query_expansion_decode"] == pytest.approx(0.05)
    assert perf.svc_decode_by_stage["generator_decode"] == pytest.approx(0.07)


def test_chain_latency_models_light_stage_collocation_contention():
    """When query encode shares the generator GPU, light-stage service adds
    contention to the decode critical path. CPU retrieval itself does not."""
    traces = [[{"stage": "semantic_retrieval_encode", "input_tokens": 16, "output_tokens": 0},
               _ret(), _gen(800, 50)] for _ in range(4)]
    lat = _analytic_latency()
    lat.encode_s = lambda b: 5e-4
    colloc = _compute_chain_latency(traces, lat, gpu_collocated_with_gen=True)[0]
    solo = _compute_chain_latency(traces, lat, gpu_collocated_with_gen=False)[0]
    assert colloc > solo
