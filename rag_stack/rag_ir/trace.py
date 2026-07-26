"""The generic, pipeline-shape-agnostic cost-model trace input.

A *trace* is one query's complete execution: an ordered list of stage calls,
each carrying the algorithm-layer stage name and its real input/output token
counts. Sequential, iterative (``flare`` / ``ircot``) and agentic
(``corag`` / ``searchr1``) pipelines all serialize to the SAME shape — the
cost model replays the trace and never needs to know which orchestrator
produced it. The "agentic-ness" (variable step count, growing prefill) lives
entirely in the data: more ``generator`` calls, larger ``input_tokens``.

Stage vocabulary (:mod:`rag_stack.taxonomy`): calls reference user stages
(``generator`` / ``query_expansion`` — one call spans the LLM's prefill AND
decode; the cost walk splits it — ``passage_reranker`` /
``passage_compressor`` / ``lexical_retrieval``) with ONE exception:
**semantic retrieval is always TWO calls** — the embedding forward
(``semantic_retrieval_encode``) and the vector search
(``semantic_retrieval_vectorsearch``). There is no folded single-call form.

Evaluator backends adapt their own runtime events to this shared contract.
Backend-specific monitor conversion belongs to the evaluator package.

The call lists remain plain JSON. At the CM boundary they arrive wrapped in
the quality-trace envelope — exactly one complete invocation per dataset
question. Nothing else is accepted: bare lists and retired ``trace_scope`` envelopes
(``trace_scope``) are rejected with an error naming the one-off converter
(``benchmarks/rag_cm_accuracy/tools/convert_legacy_traces.py``), which owns
the legacy projection semantics. The ``TypedDict`` below documents the pure
per-call record consumed after that single normalization boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, List, NotRequired, Optional, TypedDict

from rag_stack.rag_ir.io_profile import validate_io_profile
from rag_stack.rag_ir.call_identity import validate_call_identity

__all__ = [
    "TraceCall",
    "Trace",
    "TracePayload",
    "QualityTraceEnvelope",
    "QualityTraceInput",
    "QualityTraceNormalization",
    "TRACE_CALL_REQUIRED_FIELDS",
    "TRACE_CALL_OPTIONAL_FIELDS",
    "is_quality_trace_candidate",
    "make_quality_trace_envelope",
    "normalize_quality_trace",
    "is_trace_payload",
    "validate_quality_trace_envelope",
]


class TraceCall(TypedDict):
    """One stage call within a query's execution trace.

    Attributes:
        stage: Taxonomy stage name (REQUIRED): ``"generator"`` |
            ``"query_expansion"`` | ``"semantic_retrieval_encode"`` |
            ``"semantic_retrieval_vectorsearch"`` | ``"lexical_retrieval"`` |
            ``"passage_reranker"`` | ``"passage_compressor"``.
        input_tokens: Quality-invocation input/context token count (FORMAL
            tokenizer — the call's own model). The cost model flows on these.
        output_tokens: Quality-invocation output token count (``0`` for
            encode/search/rerank).
        input_bytes: Exact UTF-8 byte count of the input — tokenizer-independent ground
            truth, recorded ALONGSIDE tokens (the system still flows on tokens; bytes are
            for cross-checking / byte-based models, not consumed by the token cost path).
        output_bytes: Exact UTF-8 byte count of the output (``0`` when no text output).
        step_idx: Sequential position of this call within the query.
        model_id: Backing model id (LLM/embedder/db type); ``None`` if absent.
        input_profile: Optional exact structured-item summary, with no raw IDs.
        output_profile: Optional exact output-item summary, with no raw IDs.
        The CM boundary deliberately strips runtime observations such as
            padding groups, forward-batch shape, latency, QPS and timestamps.
            They are not properties of a quality invocation.
    """

    stage: str
    input_tokens: int
    output_tokens: int
    input_bytes: int
    output_bytes: int
    step_idx: int
    model_id: Optional[str]
    input_profile: NotRequired[dict]
    output_profile: NotRequired[dict]
    call_id: NotRequired[str]
    node_id: NotRequired[str]
    input_refs: NotRequired[dict[str, list[str]]]


# One query's ordered calls; and the full payload (one Trace per query).
Trace = List[TraceCall]
TracePayload = List[Trace]


_QUALITY_TRACE_STAGES = frozenset({
    "generator",
    "query_expansion",
    "semantic_retrieval_encode",
    "semantic_retrieval_vectorsearch",
    "lexical_retrieval",
    "retrieval_fusion",
    "passage_reranker",
    "passage_compressor",
})

# ---------------------------------------------------------------------------
# Frozen contract — the one quality-trace shape.
#
# The contract has a single semantic: exactly one COMPLETE invocation DAG per dataset
# question, produced by the quality run itself. There is no trace_scope field
# (one schema needs no scope axis); ``provenance`` is an optional free-form
# block the CM passes through untouched (the legacy converter stamps its
# lossy diagnostics there). Field whitelists are enforced HARD at every
# layer — an unknown key is an error, not something to strip silently —
# by ``validate_quality_trace_envelope``, which both ends share: producers
# call it after projection (traces leave the factory already valid), the CM
# calls it again at ``normalize_quality_trace`` as defence in depth.
# ---------------------------------------------------------------------------
TRACE_CALL_REQUIRED_FIELDS = ("stage", "input_tokens", "output_tokens", "step_idx")
# Bytes stay optional because some evaluator backends carry no byte counts; the
# communication model has token-fallback semantics for ABSENT bytes — filling
# 0 would silently zero the comm volume instead.
TRACE_CALL_OPTIONAL_FIELDS = (
    "model_id", "input_bytes", "output_bytes", "input_profile", "output_profile",
    "call_id", "node_id", "input_refs",
)
_QUALITY_TRACE_ENVELOPE_FIELDS = frozenset({
    "queries",
    "provenance",
})
_QUALITY_TRACE_QUERY_FIELDS = frozenset({
    "question_id",
    "source_question_id",
    "invocation_id",
    "calls",
})


class QualityTraceEnvelope(TypedDict, total=False):
    """The quality-trace input for the cost model (the frozen contract).

    ``{"queries": [{"question_id", "invocation_id"?, "calls": [...]}],
    "provenance"?: {...}}`` — exactly one complete
    invocation per dataset question; no scope axis. Anything else (bare call
    lists, retired ``trace_scope`` envelopes) is rejected with an
    error naming the one-off converter.
    """

    queries: List[Any]
    provenance: dict          # optional, CM passes it through untouched


QualityTraceInput = TracePayload | QualityTraceEnvelope


@dataclass(frozen=True)
class QualityTraceNormalization:
    """One normalized cohort plus auditable, non-performance diagnostics."""

    traces: TracePayload
    question_ids: tuple[Any, ...]
    source_question_ids: tuple[Any, ...]
    invocation_ids: tuple[Any | None, ...]
    scope: str
    normalization: str
    identity: str
    input_queries: int
    complete_queries: int
    kept_queries: int
    dropped_incomplete: int
    dropped_repeats: int
    lossy: bool

    def diagnostics(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "normalization": self.normalization,
            "identity": self.identity,
            "input_queries": self.input_queries,
            "complete_queries": self.complete_queries,
            "kept_queries": self.kept_queries,
            "dropped_incomplete": self.dropped_incomplete,
            "dropped_repeats": self.dropped_repeats,
            "lossy": self.lossy,
        }


def _project_call(call: Any, query_position: int, call_position: int) -> dict:
    """Producer-side projection to the frozen call whitelist.

    Keeps exactly the contract fields that are present and drops everything
    else (recorder-internal diagnostics such as padding/batch bookkeeping
    must never leave the producer). Validation of the KEPT fields happens in
    :func:`validate_quality_trace_envelope` — projection only shapes."""
    if not isinstance(call, dict):
        raise TypeError(
            f"quality trace query {query_position} call {call_position} "
            "is not an object"
        )
    return {
        key: deepcopy(call[key])
        for key in TRACE_CALL_REQUIRED_FIELDS + TRACE_CALL_OPTIONAL_FIELDS
        if key in call
    }


def _validate_call(call: Any, query_position: int, call_position: int) -> None:
    where = f"quality trace query {query_position} call {call_position}"
    if not isinstance(call, dict):
        raise TypeError(f"{where} is not an object")
    allowed = set(TRACE_CALL_REQUIRED_FIELDS) | set(TRACE_CALL_OPTIONAL_FIELDS)
    unknown = set(call) - allowed
    if unknown:
        raise ValueError(f"{where} has unsupported fields: {sorted(unknown)}")
    missing = [key for key in TRACE_CALL_REQUIRED_FIELDS if key not in call]
    if missing:
        raise ValueError(f"{where} missing required fields: {missing}")
    stage = call["stage"]
    if not isinstance(stage, str) or stage not in _QUALITY_TRACE_STAGES:
        raise ValueError(f"{where} has unsupported stage={stage!r}")
    for key in ("input_tokens", "output_tokens", "step_idx"):
        count = call[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{where} has invalid {key}={count!r}")
    for key in ("input_bytes", "output_bytes"):
        if key not in call:
            continue
        count = call[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{where} has invalid {key}={count!r}")
    model_id = call.get("model_id")
    if model_id is not None and not isinstance(model_id, str):
        raise ValueError(f"{where} has invalid model_id={model_id!r}")
    for key in ("input_profile", "output_profile"):
        if key in call:
            validate_io_profile(call[key], where=f"{where}.{key}")
    if stage == "retrieval_fusion" and model_id in {"rrf", "rrf_pure"}:
        inputs, outputs = call.get("input_profile"), call.get("output_profile")
        if inputs is not None and outputs is not None:
            bound = inputs.get("unique_items", sum(inputs["list_unique_items"])
                               if "list_unique_items" in inputs else inputs["items"])
            if outputs["items"] > bound:
                raise ValueError(f"{where} RRF output items exceed the input union bound")
            if "unique_items" in outputs and outputs["unique_items"] != outputs["items"]:
                raise ValueError(f"{where} RRF output must contain unique items")


def validate_quality_trace_envelope(envelope: Any) -> None:
    """The frozen-contract validator shared by producers and consumers.

    Raises on any violation: envelope/query/call field whitelists (unknown
    keys are hard errors), required fields, non-empty unique ``question_id``
    per query, per-query completeness (terminal generator call) and step
    order. Producers run it before shipping; the CM runs it again on intake.
    """
    if not isinstance(envelope, dict):
        raise TypeError("quality trace envelope must be a dict")
    unknown = set(envelope) - _QUALITY_TRACE_ENVELOPE_FIELDS
    if unknown:
        raise ValueError(
            f"quality trace envelope has unsupported fields: {sorted(unknown)}"
        )
    provenance = envelope.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise TypeError("quality trace envelope 'provenance' must be a dict")
    queries = envelope.get("queries")
    if not isinstance(queries, list):
        raise TypeError("quality trace envelope 'queries' must be a list")
    if not queries:
        raise ValueError("quality trace is empty")
    seen_ids: set[Any] = set()
    for position, query in enumerate(queries):
        if not isinstance(query, dict):
            raise TypeError(
                f"quality trace query {position} must be a query record"
            )
        unknown = set(query) - _QUALITY_TRACE_QUERY_FIELDS
        if unknown:
            raise ValueError(
                f"quality trace query {position} has unsupported fields: "
                f"{sorted(unknown)}"
            )
        question_id = query.get("question_id")
        if question_id is None or (
            isinstance(question_id, str) and not question_id.strip()
        ):
            raise ValueError(
                f"quality trace query {position} requires a non-empty question_id"
            )
        try:
            duplicate = question_id in seen_ids
        except TypeError as exc:
            raise TypeError(
                "quality trace question_id must be hashable"
            ) from exc
        if duplicate:
            raise ValueError(
                f"quality trace query {position} has duplicate "
                f"question_id={question_id!r}"
            )
        seen_ids.add(question_id)
        source_question_id = query.get("source_question_id")
        if source_question_id is not None:
            if isinstance(source_question_id, str) and not source_question_id.strip():
                raise ValueError(
                    f"quality trace query {position} has an empty "
                    "source_question_id"
                )
            try:
                hash(source_question_id)
            except TypeError as exc:
                raise TypeError(
                    "quality trace source_question_id must be hashable"
                ) from exc
        calls = query.get("calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError(
                f"quality trace query {position} must carry a non-empty "
                "'calls' list"
            )
        prior_calls = {}
        for call_position, call in enumerate(calls):
            _validate_call(call, position, call_position)
            validate_call_identity(
                call, prior_calls, where=f"quality trace query {position} call {call_position}",
            )
            if "call_id" in call:
                prior_calls[call["call_id"]] = call
        if not _terminal_complete(calls):
            raise ValueError(
                f"quality trace query {position} is incomplete "
                "(must end at a generator call)"
            )
        if not _steps_in_order(calls):
            raise ValueError(
                f"quality trace query {position} has invalid step order"
            )


def make_quality_trace_envelope(
    traces: TracePayload,
    *,
    question_ids: List[Any],
    source_question_ids: Optional[List[Any]] = None,
    invocation_ids: Optional[List[Any]] = None,
    provenance: Optional[dict] = None,
) -> QualityTraceEnvelope:
    """Produce the quality-trace envelope — the only shipping form.

    ``traces`` is one ordered call list per dataset question; ``question_ids``
    supplies the per-question stable identity (cardinality must match — the
    dataset-row invariant is the producer's responsibility). Calls are
    projected to the frozen whitelist (recorder-internal diagnostics never
    leave the factory) and the shared validator runs before return, so an
    envelope in flight is valid by construction.
    """
    if len(question_ids) != len(traces):
        raise ValueError(
            "make_quality_trace_envelope: question_ids/traces cardinality "
            f"mismatch ({len(question_ids)} != {len(traces)})"
        )
    if invocation_ids is not None and len(invocation_ids) != len(traces):
        raise ValueError(
            "make_quality_trace_envelope: invocation_ids/traces cardinality "
            f"mismatch ({len(invocation_ids)} != {len(traces)})"
        )
    if source_question_ids is not None and len(source_question_ids) != len(traces):
        raise ValueError(
            "make_quality_trace_envelope: source_question_ids/traces "
            f"cardinality mismatch ({len(source_question_ids)} != {len(traces)})"
        )
    queries: List[dict] = []
    for position, calls in enumerate(traces):
        if not isinstance(calls, list):
            raise TypeError(
                f"quality trace query {position} must be a call list"
            )
        record: dict = {
            "question_id": question_ids[position],
            "calls": [
                _project_call(call, position, call_position)
                for call_position, call in enumerate(calls)
            ],
        }
        if invocation_ids is not None and invocation_ids[position] is not None:
            record["invocation_id"] = invocation_ids[position]
        if (
            source_question_ids is not None
            and source_question_ids[position] is not None
        ):
            record["source_question_id"] = source_question_ids[position]
        queries.append(record)
    envelope: QualityTraceEnvelope = {"queries": queries}
    if provenance is not None:
        envelope["provenance"] = provenance
    validate_quality_trace_envelope(envelope)
    return envelope


def _terminal_complete(calls: Trace) -> bool:
    return bool(calls and calls[-1].get("stage") == "generator")


def _steps_in_order(calls: Trace) -> bool:
    steps = [call.get("step_idx") for call in calls]
    if not any(step is not None for step in steps):
        return True
    if any(step is None for step in steps):
        return False
    # FlashRAG expands one dense-retrieve event into encode + vectorsearch at
    # the same logical step and can leave gaps after dropping no-cost monitor
    # events. List order is authoritative; step_idx must only agree with it.
    return all(left <= right for left, right in zip(steps, steps[1:]))


_LEGACY_TRACE_DIRECTION = (
    "run benchmarks/rag_cm_accuracy/tools/convert_legacy_traces.py to "
    "convert the artifact to the current envelope once, offline"
)


def normalize_quality_trace(payload: QualityTraceInput) -> QualityTraceNormalization:
    """Canonicalize the sole dynamic CM workload input (current envelope only).

    The envelope is producer-projected — exactly one complete invocation per
    dataset question — so normalization is a validated pass-through: the
    shared validator runs as defence in depth, then every query's calls are
    defensively copied onto the frozen whitelist. The operation happens once
    at the CM boundary so RAGO profiling, the analytical walk, and the DES
    all see the identical cohort.

    Every legacy form is rejected with a directed error: the projection
    semantics for bare lists and retired ``trace_scope`` envelopes now live in
    the one-off converter, never in the CM.
    """
    if isinstance(payload, dict) and "queries" in payload:
        validate_quality_trace_envelope(payload)
        provenance = payload.get("provenance") or {}
        traces: TracePayload = [
            [
                TraceCall(**{
                    key: deepcopy(call[key])
                    for key in (
                        TRACE_CALL_REQUIRED_FIELDS + TRACE_CALL_OPTIONAL_FIELDS
                    )
                    if key in call
                })
                for call in query["calls"]
            ]
            for query in payload["queries"]
        ]
        n = len(traces)

        # Diagnostics describe the PROJECTION HISTORY of the workload, so the
        # converter-stamped provenance (offline legacy projection) wins over
        # the trivial passthrough view whenever its fields are present — a
        # replayed converted artifact reports the SAME audit trail the CM
        # printed when it still ran that projection at its own boundary.
        # Pricing never reads these numbers; kept_queries is always the
        # envelope's actual (priced) cohort size.
        def _provenance_count(name: str, default: int) -> int:
            value = provenance.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return default
            return value

        return QualityTraceNormalization(
            traces=traces,
            question_ids=tuple(
                query["question_id"] for query in payload["queries"]
            ),
            # Canonical quality traces already use the stable dataset identity
            # as question_id.  A completion-window performance sample instead
            # has one unique sample identity plus a possibly repeated stable
            # source_question_id.  Keeping both prevents DES prefix-cache keys
            # from depending on reservoir order or sample cardinality.
            source_question_ids=tuple(
                query.get("source_question_id", query["question_id"])
                for query in payload["queries"]
            ),
            invocation_ids=tuple(
                query.get("invocation_id") for query in payload["queries"]
            ),
            scope=str(provenance.get("scope") or "quality"),
            normalization=str(
                provenance.get("normalization") or "producer_projection"
            ),
            identity=str(provenance.get("identity") or "question_id"),
            input_queries=_provenance_count("input_queries", n),
            complete_queries=_provenance_count("complete_queries", n),
            kept_queries=n,
            dropped_incomplete=_provenance_count("dropped_incomplete", 0),
            dropped_repeats=_provenance_count("dropped_repeats", 0),
            lossy=bool(provenance.get("lossy", False)),
        )

    if isinstance(payload, list):
        raise ValueError(
            "legacy bare-list quality trace payloads are no longer accepted; "
            + _LEGACY_TRACE_DIRECTION
        )
    if isinstance(payload, dict):
        raise ValueError(
            "legacy or malformed quality-trace envelopes are no longer "
            "accepted; " + _LEGACY_TRACE_DIRECTION
        )
    raise TypeError(
        "quality trace must be the current quality-trace envelope; "
        + _LEGACY_TRACE_DIRECTION
    )


def is_trace_payload(dag: Any) -> bool:
    """True only for the current quality-trace envelope.

    Distinguishes the envelope from the ``static_gt`` aggregate dict
    (``{"shape": "aggregate", ...}``) and from ``None`` so the trace path is
    picked only when a quality trace is present. Bare lists and retired envelopes
    are deliberately NOT recognized: routers must send them to
    :func:`normalize_quality_trace` (via :func:`is_quality_trace_candidate`
    or an explicit legacy check) for the directed converter error, never
    silently down the static path.
    """
    if not isinstance(dag, dict) or "queries" not in dag:
        return False
    try:
        validate_quality_trace_envelope(dag)
    except (TypeError, ValueError):
        return False
    return True


def is_quality_trace_candidate(value: Any) -> bool:
    """Whether a mapping is trying to be a quality-trace envelope.

    Unlike :func:`is_trace_payload`, this intentionally recognizes malformed
    candidates so routers send them to trace validation and fail closed instead
    of silently treating them as a static aggregate.
    """
    return isinstance(value, dict) and (
        "trace_scope" in value
        or "queries" in value
    )
