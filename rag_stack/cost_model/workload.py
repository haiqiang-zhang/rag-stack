"""Prepare the trace cohort and its derived component-model inputs.

The summary is not a performance estimate. Representative lengths and call
counts feed component sweeps, feasibility checks and communication models;
analytical scoring and simulation also consume the retained ordered traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any

import numpy as np

from rag_stack.rag_ir.trace import (
    QualityTraceInput,
    QualityTraceNormalization,
    TracePayload,
    normalize_quality_trace,
)
from rag_stack.rag_ir.execution_path import compile_execution_paths


MAX_TRACE_QUERIES = 2048


@dataclass(frozen=True, init=False)
class PreparedExecutionPaths:
    """Owned, validated call snapshot borrowed read-only by CM consumers.

    Construction validates and copies raw paths. Analytical walks, physical
    lowering and pricing callbacks must not mutate ``paths`` or nested call
    profiles. The wrapper contains only the selected paths so it can travel to
    DES workers without carrying the original, possibly much larger envelope.
    It is an internal runtime contract, not a persisted execution artifact.
    """

    paths: TracePayload

    def __init__(self, traces: TracePayload, *, retrieval_fanout: int = 1):
        object.__setattr__(self, "paths", compile_execution_paths(
            traces, retrieval_fanout=retrieval_fanout,
        ))


def resolve_execution_paths(
    traces: TracePayload,
    prepared_paths: PreparedExecutionPaths | None = None,
    *,
    retrieval_fanout: int = 1,
) -> PreparedExecutionPaths:
    """Validate raw input once or borrow the exact prepared snapshot.

    Checking identity is constant-time and prevents attaching another cohort's
    validation to arbitrary traces. Pickle memoization preserves this identity
    when both references travel in one worker task. No hashes or global cache
    are involved.
    """
    if prepared_paths is None:
        return PreparedExecutionPaths(traces, retrieval_fanout=retrieval_fanout)
    if not isinstance(prepared_paths, PreparedExecutionPaths):
        raise TypeError("prepared_paths must be PreparedExecutionPaths")
    if prepared_paths.paths is not traces:
        raise ValueError("prepared_paths must own the supplied traces")
    return prepared_paths


@dataclass(frozen=True)
class TraceWorkload:
    """One cohort shared by candidate preparation, analysis and simulation."""

    traces: TracePayload
    source_question_ids: list[Any]
    normalization: QualityTraceNormalization
    summary: dict
    prepared_paths: PreparedExecutionPaths

    @property
    def execution_paths(self) -> TracePayload:
        return self.prepared_paths.paths


def _integer_mean(values: list[int]) -> int:
    return int(np.mean(values)) if values else 0


def summarize_trace(traces: TracePayload) -> dict:
    """Derive representative lengths and invocation counts without call order.

    Generator input length is averaged per call; generator output length is
    averaged over each query's total output. These are RAGO/component adapter
    conventions, not a claim that repeated calls equal one longer call.
    Invocation factors retain fractional means, with the historical minimum
    of one expected by the component tables. ``num_retrievals`` is a legacy
    RAGO key for generator prefill/decode handoffs, not retrieval calls.

    The compatibility dictionary also retains ``source="static_gt"`` and
    ``rag_ir_mode="dynamic"`` for existing token-override consumers. Neither
    field changes the provenance or execution order of the original trace.
    """
    generator_inputs: list[int] = []
    expansion_inputs: list[int] = []
    expansion_outputs: list[int] = []
    retrieval_inputs: list[int] = []
    retrieval_outputs: list[int] = []
    generator_counts: list[int] = []
    dense_retrieval_counts: list[int] = []
    generator_outputs_per_query: list[int] = []
    fusion_calls: list[dict] = []
    fusion_counts: list[int] = []

    for calls in traces:
        generators = dense_retrievals = output_tokens = fusions = 0
        for call in calls:
            stage = call.get("stage")
            if not stage:
                raise KeyError(f"trace call missing required 'stage' field: {call!r}")
            if stage == "query_expansion":
                expansion_inputs.append(int(call.get("input_tokens", 0)))
                expansion_outputs.append(int(call.get("output_tokens", 0)))
            elif stage == "generator":
                generator_inputs.append(int(call.get("input_tokens", 0)))
                output_tokens += int(call.get("output_tokens", 0))
                generators += 1
            elif stage in ("semantic_retrieval_vectorsearch", "lexical_retrieval"):
                retrieval_inputs.append(int(call.get("input_tokens", 0)))
                retrieval_outputs.append(int(call.get("output_tokens", 0)))
                dense_retrievals += stage == "semantic_retrieval_vectorsearch"
            elif stage == "retrieval_fusion":
                fusion_calls.append(deepcopy(call))
                fusions += 1
        generator_counts.append(generators)
        dense_retrieval_counts.append(dense_retrievals)
        generator_outputs_per_query.append(output_tokens)
        fusion_counts.append(fusions)

    generator_visits = max(1.0, float(np.mean(generator_counts))) if traces else 1.0
    retrieval_visits = max(1.0, float(np.mean(dense_retrieval_counts))) if traces else 1.0
    return {
        "shape": "aggregate",
        "source": "static_gt",
        "rag_ir_mode": "dynamic",
        "n_user_queries": len(traces),
        "token_stats": {
            "avg_query_tokens": _integer_mean(retrieval_inputs),
            "avg_retrieved_context_tokens": _integer_mean(retrieval_outputs),
            "avg_generator_input_tokens": _integer_mean(generator_inputs),
            "avg_generator_output_tokens": _integer_mean(generator_outputs_per_query),
            "avg_query_expansion_input_tokens": _integer_mean(expansion_inputs),
            "avg_query_expansion_output_tokens": _integer_mean(expansion_outputs),
        },
        "retrieval_invocations_per_query": retrieval_visits,
        "prefill_fanout": generator_visits,
        "num_retrievals": generator_visits,
        "retrieval_fusion_calls": fusion_calls,
        "fusion_invocations_per_query": float(np.mean(fusion_counts)) if traces else 0.0,
    }


def prepare_trace_workload(
    trace: QualityTraceInput,
    *,
    query_expansion_fanout: int = 1,
    max_queries: int = MAX_TRACE_QUERIES,
) -> TraceWorkload:
    """Normalize once and apply the same deterministic cohort to every phase.

    Source question IDs follow the stride sample. Summary statistics are
    computed after sampling. Dense retrieval visits per query and expanded
    subqueries per visit stay separate. Fusion demand comes exclusively from
    observed fusion calls and their input/output profiles, never from fanout.
    """
    if max_queries < 1:
        raise ValueError("max_queries must be positive")
    if query_expansion_fanout < 1:
        raise ValueError("query_expansion_fanout must be positive")
    normalization = normalize_quality_trace(trace)
    traces = normalization.traces
    from rag_stack.cost_model.support import require_cm_trace_support
    require_cm_trace_support(traces)
    source_question_ids = list(normalization.source_question_ids)
    if len(traces) > max_queries:
        stride = (len(traces) + max_queries - 1) // max_queries
        traces = traces[::stride]
        source_question_ids = source_question_ids[::stride]
    summary = summarize_trace(traces)
    summary["query_expansion_fanout"] = query_expansion_fanout
    summary["retrieval_backend_fanout"] = (
        summary["retrieval_invocations_per_query"] * query_expansion_fanout
    )
    prepared_paths = PreparedExecutionPaths(
        traces, retrieval_fanout=query_expansion_fanout,
    )
    return TraceWorkload(
        traces, source_question_ids, normalization, summary, prepared_paths,
    )
