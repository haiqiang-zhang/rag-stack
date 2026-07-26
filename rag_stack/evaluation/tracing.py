"""RAG-Stack-owned hooks for tracing an external RAG pipeline.

``TraceSession`` records exact component I/O sizes, not model execution or judge
calls. Importing this module does not import a model runtime or load a tokenizer.
Only :meth:`TraceSession.finish` lazily imports the host's shared RAG-IR validator.

Example::

    session = TraceSession()
    with session.query("task-1"):
        with session.stage("generator", model_id="my-model") as call:
            call.input(token_ids=prompt_ids, text=rendered_prompt)
            answer = generate(prompt_ids)
            call.output(token_ids=answer.token_ids, text=answer.text)
    trace = session.finish(expected_question_ids=["task-1"])

Components can instead import :func:`trace_stage` directly, without receiving a
recorder argument. Activate :func:`capture_trace` around the evaluator's serving
invocation and :func:`trace_query` around each request. The same exact I/O hooks
and validation apply. Global hooks also work inside explicit ``session.query``
and ``session.batch`` scopes; missing active scopes are errors, never silent
no-ops.

The contract is an ordered call list per request. Concurrent different requests
are supported; overlapping stages of the same request are rejected. Instrument
the serving pipeline only, and run quality judges after ``finish``. These hooks
capture the dynamic trace; the adapter must also describe the workflow structure.
Use these hooks on uninstrumented boundaries. Mixing deferred legacy ``record_io``
wrappers into a public session is rejected to avoid implicit tokenization and
accidental double recording.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import threading
from typing import Any

from . import _recording
from rag_stack.rag_ir.io_profile import copy_io_profile

__all__ = [
    "TraceSession", "StageCall", "SUPPORTED_STAGES", "capture_trace",
    "get_trace_session", "trace_query", "trace_batch", "trace_stage",
]

SUPPORTED_STAGES = frozenset({
    "generator", "query_expansion", "semantic_retrieval_encode",
    "semantic_retrieval_vectorsearch", "lexical_retrieval",
    "retrieval_fusion", "passage_reranker", "passage_compressor",
})

Tokenizer = Callable[[str], Sequence[int]]

_current_session: ContextVar[TraceSession | None] = ContextVar(
    "rag_stack_public_trace_session", default=None,
)


def _sequence(value: Any, name: str) -> list:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return list(value)


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _qids(values: Sequence[str]) -> tuple[str, ...]:
    result = _sequence(values, "question IDs")
    if not result or any(not isinstance(q, str) or not q.strip() for q in result):
        raise ValueError("question IDs must be non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("duplicate question IDs are not allowed")
    return tuple(result)


@dataclass(frozen=True)
class _Payload:
    tokens: tuple[int, ...]
    bytes: tuple[int | None, ...]
    profiles: tuple[dict | None, ...]


def _measure(
    size: int,
    *,
    token_ids: Sequence[int] | Sequence[Sequence[int]] | None = None,
    token_count: int | None = None,
    token_counts: Sequence[int] | None = None,
    text: str | None = None,
    texts: Sequence[str] | None = None,
    byte_count: int | None = None,
    byte_counts: Sequence[int] | None = None,
    tokenizer: Tokenizer | None = None,
    profile: dict | None = None,
    profiles: Sequence[dict] | None = None,
) -> _Payload:
    if profile is not None and profiles is not None:
        raise ValueError("provide profile or profiles, not both")
    if profile is not None:
        if size != 1:
            raise ValueError("profile requires a single query; use profiles for batches")
        structured = [copy_io_profile(profile)]
    elif profiles is not None:
        values = _sequence(profiles, "profiles")
        if len(values) != size:
            raise ValueError("profiles must contain exactly one entry per question ID")
        structured = [copy_io_profile(value) for value in values]
    else:
        structured = [None] * size
    if text is not None and texts is not None:
        raise ValueError("provide text or texts, not both")
    if text is not None:
        if size != 1 or not isinstance(text, str):
            raise ValueError("text requires a single query and a string")
        text_values = [text]
    elif texts is not None:
        text_values = _sequence(texts, "texts")
        if len(text_values) != size or any(not isinstance(t, str) for t in text_values):
            raise ValueError("texts must contain one string per question ID")
    else:
        text_values = None

    choices = sum(v is not None for v in (token_ids, token_count, token_counts))
    if choices > 1:
        raise ValueError("provide exactly one of token_ids, token_count, token_counts")
    if tokenizer is not None and choices:
        raise ValueError("tokenizer cannot be combined with explicit token IDs or counts")
    if token_count is not None:
        if size != 1:
            raise ValueError("token_count requires a single query; use token_counts for batches")
        counts = [_count(token_count, "token_count")]
    elif token_counts is not None:
        counts = [_count(n, "token_counts") for n in _sequence(token_counts, "token_counts")]
    elif token_ids is not None:
        # One-query calls use a flat ID list; batches use a list per query.
        batches = [token_ids] if size == 1 else _sequence(token_ids, "token_ids")
        counts = []
        for ids in batches:
            values = _sequence(ids, "token_ids")
            for value in values:
                _count(value, "token ID")
            counts.append(len(values))
    elif tokenizer is not None and text_values is not None:
        if not callable(tokenizer):
            raise TypeError("tokenizer must be a callable returning token IDs")
        counts = []
        for value in text_values:
            ids = _sequence(tokenizer(value), "tokenizer result")
            for token_id in ids:
                _count(token_id, "token ID")
            counts.append(len(ids))
    elif (profile is not None or profiles is not None) and text_values is None and tokenizer is None:
        # Structured-only operators have no token work. A text payload still
        # requires the existing exact-token contract, even with a profile.
        counts = [0] * size
    else:
        raise ValueError("exact token IDs/counts or text with an explicit tokenizer are required")
    if len(counts) != size:
        raise ValueError("token payload must contain exactly one entry per question ID")

    if byte_count is not None and byte_counts is not None:
        raise ValueError("provide byte_count or byte_counts, not both")
    if byte_count is not None:
        if size != 1:
            raise ValueError("byte_count requires a single query; use byte_counts for batches")
        sizes: list[int | None] = [_count(byte_count, "byte_count")]
    elif byte_counts is not None:
        sizes = [_count(n, "byte_counts") for n in _sequence(byte_counts, "byte_counts")]
        if len(sizes) != size:
            raise ValueError("byte_counts must contain exactly one entry per question ID")
    else:
        sizes = [None] * size
    if text_values is not None:
        actual_sizes = [len(t.encode("utf-8")) for t in text_values]
        if (byte_count is not None or byte_counts is not None) and sizes != actual_sizes:
            raise ValueError("explicit byte counts disagree with the provided text")
        sizes = actual_sizes
    # Keep only exact sizes; neither raw text nor token IDs survive this call.
    return _Payload(tuple(counts), tuple(sizes), tuple(structured))


class StageCall:
    """I/O hooks yielded by :meth:`TraceSession.stage`.

    For token payloads, ``input`` and ``output`` take one token representation:
    ``token_ids`` (flat for one query; nested for a batch), ``token_count`` for
    one query, ``token_counts`` for a batch, or ``text``/``texts`` plus an explicit
    ``tokenizer(text) -> token_ids`` callable. Optional ``text``/``texts`` also
    provide exact UTF-8 byte sizes. Without text, optional ``byte_count`` or
    ``byte_counts`` can supply exact byte sizes; otherwise bytes are omitted.

    Structured operators can supply ``profile`` (one query) or ``profiles``
    (one summary per query). A structured-only payload has zero token work;
    profiles may also accompany explicit token and byte measurements.

    Both hooks are mandatory, including ``output(token_count=0)`` for non-text
    output. Repeated or invalid hooks fail the session even if their error is
    caught by the adapter. Capture the actual rendered prompt and actual model
    tokens; never derive an estimate from whitespace or a different tokenizer.
    """

    def __init__(self, size: int) -> None:
        self._size = size
        self._input: _Payload | None = None
        self._output: _Payload | None = None
        self._failed = False
        self._closed = False
        self.call_ids: tuple[str, ...] = ()

    @property
    def call_id(self) -> str:
        """The completed single-query call identity, for explicit input references."""
        if len(self.call_ids) != 1:
            raise RuntimeError("call_id requires a successfully completed single-query stage")
        return self.call_ids[0]

    def _set(self, direction: str, **payload: Any) -> None:
        try:
            if self._closed:
                raise RuntimeError("stage hooks cannot be used after their scope exits")
            if getattr(self, f"_{direction}") is not None:
                raise ValueError(f"{direction} was already recorded")
            if direction == "output" and self._input is None:
                raise ValueError("record input before output")
            setattr(self, f"_{direction}", _measure(self._size, **payload))
        except BaseException:
            self._failed = True
            raise

    def input(self, **payload: Any) -> None:
        """Capture exact input sizes before invoking the component."""
        self._set("input", **payload)

    def output(self, **payload: Any) -> None:
        """Capture exact output sizes after the component returns."""
        self._set("output", **payload)


class TraceSession:
    """A strict trace recorder shared by one evaluator invocation.

    Use ``query(qid)`` or ``batch(qids)`` in each worker. ContextVars restore the
    prior recorder and query IDs on exit, including nested sessions and asyncio
    tasks; threads explicitly enter their own scopes using this shared object.
    Distinct requests may execute concurrently. Each request must execute stages
    serially because the current RAG-IR does not represent parallel dependencies.

    Any failed scope makes the session unpublishable. ``finish`` requires exact
    request coverage and a terminal generator call for every request, and closes
    the session to further writes. Repeated ``finish`` with identical IDs returns
    the same envelope object (treat it as read-only when publishing a callback).
    """

    def __init__(self) -> None:
        self._recorder = _recording.TraceRecorder()
        self._lock = threading.RLock()
        self._current_qids: ContextVar[tuple[str, ...] | None] = ContextVar(
            f"public_trace_qids_{id(self)}", default=None,
        )
        self._active_queries: set[str] = set()
        self._active_stages: set[str] = set()
        self._failed = False
        self._finished: dict | None = None
        self._finished_ids: tuple[str, ...] | None = None

    def _check_open(self) -> None:
        if self._failed:
            raise RuntimeError("trace session contains a failed scope and cannot be published")
        if self._finished is not None:
            raise RuntimeError("trace session is already finished")

    @contextmanager
    def query(self, question_id: str) -> Iterator[TraceSession]:
        """Bind one stable dataset/task ID for the enclosed component calls."""
        with self.batch([question_id]):
            yield self

    @contextmanager
    def batch(self, question_ids: Sequence[str]) -> Iterator[TraceSession]:
        """Bind IDs in the exact input/output batch order; never truncate rows."""
        qids = _qids(question_ids)
        with self._lock:
            self._check_open()
            if self._active_queries.intersection(qids):
                raise RuntimeError("overlapping query scopes for the same question ID")
            self._active_queries.update(qids)
        local_token = self._current_qids.set(qids)
        session_token = _current_session.set(self)
        recorder_token = _recording._current_recorder.set(self._recorder)
        qids_token = _recording._current_qids.set(list(qids))
        try:
            yield self
        except BaseException:
            with self._lock:
                self._failed = True
            raise
        finally:
            _recording._current_qids.reset(qids_token)
            _recording._current_recorder.reset(recorder_token)
            _current_session.reset(session_token)
            self._current_qids.reset(local_token)
            with self._lock:
                self._active_queries.difference_update(qids)

    @contextmanager
    def stage(
        self, stage: str, *, model_id: str, node_id: str | None = None,
        input_refs: dict[str, list[str]] | Sequence[dict[str, list[str]]] | None = None,
    ) -> Iterator[StageCall]:
        """Wrap one real serving component call with exact input/output hooks.

        Dense retrieval is two calls: ``semantic_retrieval_encode`` followed by
        ``semantic_retrieval_vectorsearch``. ``model_id`` identifies the real model
        or index implementation; it is required for every public stage.
        """
        if not isinstance(stage, str) or stage not in SUPPORTED_STAGES:
            raise ValueError(f"unsupported trace stage: {stage!r}")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        qids = self._current_qids.get()
        if qids is None:
            raise RuntimeError("enter a query() or batch() scope before a stage")
        with self._lock:
            self._check_open()
            if not set(qids).issubset(self._active_queries):
                raise RuntimeError("query scope has already exited")
            if self._active_stages.intersection(qids):
                raise RuntimeError("overlapping stages for one query are not representable")
            self._active_stages.update(qids)
        call = StageCall(len(qids))
        try:
            if isinstance(input_refs, dict):
                if len(qids) != 1:
                    raise ValueError("batched input_refs requires one mapping per question ID")
                refs = [input_refs]
            elif input_refs is None:
                refs = [None] * len(qids)
            else:
                refs = list(input_refs)
                if len(refs) != len(qids):
                    raise ValueError("input_refs must contain exactly one mapping per question ID")
            yield call
            if call._failed or call._input is None or call._output is None:
                raise RuntimeError("stage requires one successful input hook and output hook")
            with self._lock:
                self._check_open()
                call_ids = []
                for index, qid in enumerate(qids):
                    call_ids.append(self._recorder.record(
                        qid, stage, model_id=model_id,
                        input_tokens=call._input.tokens[index],
                        output_tokens=call._output.tokens[index],
                        input_bytes=call._input.bytes[index],
                        output_bytes=call._output.bytes[index],
                        input_profile=call._input.profiles[index],
                        output_profile=call._output.profiles[index],
                        node_id=node_id,
                        input_refs=refs[index],
                    ))
                call.call_ids = tuple(call_ids)
        except BaseException:
            with self._lock:
                self._failed = True
            raise
        finally:
            call._closed = True
            with self._lock:
                self._active_stages.difference_update(qids)

    def finish(self, *, expected_question_ids: Sequence[str]) -> dict:
        """Validate exact coverage and return the canonical quality envelope.

        The expected order is the dataset order, independent of worker completion
        order. Missing/extra IDs, active scopes, or incomplete traces are errors.
        This validates the host contract but does not run the cost model.
        """
        expected = _qids(expected_question_ids)
        with self._lock, self._recorder._lock:
            if self._finished is not None:
                if expected != self._finished_ids:
                    raise ValueError("finish question IDs differ from the published trace")
                return self._finished
            self._check_open()
            if self._active_queries or self._active_stages:
                raise RuntimeError("exit all query and stage scopes before finish")
            try:
                actual = set(self._recorder.qids)
                wanted = set(expected)
                if actual != wanted:
                    raise ValueError(
                        f"trace coverage mismatch: missing={sorted(wanted - actual)!r}, "
                        f"extra={sorted(actual - wanted)!r}"
                    )
                if any(
                    any(key.startswith("_pend_") for key in call)
                    for calls in self._recorder._by_qid.values()
                    for call in calls
                ):
                    raise ValueError(
                        "deferred legacy record_io entries cannot be published by "
                        "TraceSession; use exact public input/output hooks on "
                        "uninstrumented boundaries instead"
                    )
                from rag_stack.rag_ir.trace import make_quality_trace_envelope

                self._recorder.select_qids(expected)
                self._finished = make_quality_trace_envelope(
                    self._recorder.to_trace_v1(), question_ids=list(expected),
                )
                self._finished_ids = expected
            except BaseException:
                self._failed = True
                raise
            return self._finished


def get_trace_session() -> TraceSession | None:
    """Return the session active in this context, or ``None`` outside capture.

    This accessor permits an explicit opt-in instrumentation policy. The global
    recording hooks themselves require an active session and never silently
    discard component calls.
    """
    return _current_session.get()


def _require_trace_session() -> TraceSession:
    session = get_trace_session()
    if session is None:
        raise RuntimeError(
            "no active trace session; enter capture_trace() or an explicit "
            "TraceSession.query()/batch() scope first"
        )
    return session


@contextmanager
def capture_trace(session: TraceSession | None = None) -> Iterator[TraceSession]:
    """Activate a session for hooks anywhere in the current evaluator call.

    A new session is created by default. Pass an existing session when several
    workers contribute different requests to one evaluation. ContextVars isolate
    asyncio tasks and restore nested captures on exit, including exceptions;
    threads must explicitly activate their session in each worker.

    This scope does not publish or finish the trace. After all workers/scopes
    exit, call ``session.finish(expected_question_ids=...)`` and return it with
    the evaluator's quality scores and workflow. An exception escaping capture
    makes this session unpublishable, even if it occurs outside a query scope.
    """
    if session is None:
        session = TraceSession()
    if not isinstance(session, TraceSession):
        raise TypeError("session must be a TraceSession")
    with session._lock:
        session._check_open()
    token = _current_session.set(session)
    try:
        yield session
    except BaseException:
        with session._lock:
            session._failed = True
        raise
    finally:
        _current_session.reset(token)


@contextmanager
def trace_query(question_id: str) -> Iterator[TraceSession]:
    """Bind a request ID on the active session without passing a recorder."""
    with _require_trace_session().query(question_id) as session:
        yield session


@contextmanager
def trace_batch(question_ids: Sequence[str]) -> Iterator[TraceSession]:
    """Bind a batch in exact input/output order on the active session."""
    with _require_trace_session().batch(question_ids) as session:
        yield session


@contextmanager
def trace_stage(
    stage: str, *, model_id: str, node_id: str | None = None,
    input_refs: dict[str, list[str]] | Sequence[dict[str, list[str]]] | None = None,
) -> Iterator[StageCall]:
    """Capture component I/O using the active session and query/batch scope.

    This is the global equivalent of ``session.stage(...)``. A component can
    import it directly without accepting or retaining a recorder argument.
    The yielded hooks have the same exact-token and byte-size contract as
    :class:`StageCall`. Missing capture or query scopes raise ``RuntimeError``.
    """
    with _require_trace_session().stage(
        stage, model_id=model_id, node_id=node_id, input_refs=input_refs,
    ) as call:
        yield call
