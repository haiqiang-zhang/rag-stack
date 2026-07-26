"""rag_cm engines — the per-config latency+tp computers, dispatched by
RAGCMAssembly. TWO engines (the Static vs Dynamic computational split, NOT
sequential vs agentic):

  * :class:`StaticEngine` — config-driven path. Sequential ONLY (asserts it):
    RAGO's single-pass latency_s + qps are final; nothing to walk.
  * :class:`DynamicEngine` — trace-driven analytical model. ``analyze`` builds
    per-call service demands and predicts candidate throughput and latency for
    shortlisting. ``cost`` is a compatibility entry into ``dynamic_pipeline``,
    which owns analysis, shortlisting, and authoritative closed-loop simulation.
    Both sequential and agentic traces use the same analytical implementation.

This module owns the walk primitives (CallLatency / ChainPerf /
build_call_latency / _compute_chain_latency). Workload normalization and summary
statistics live in workload.py; orchestration lives in dynamic_pipeline.py.
The walk is testable without GenZ/FAISS by injecting component service functions.
RAGO is not called here: engines consume candidate rows and per-stage tables.
"""

from __future__ import annotations

import logging
import ast
import math
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from rag_stack.types import SystemConfig
from rag_stack.rag_ir.trace import TracePayload
from rag_stack.rag_ir.workflow_schema import WorkflowSchema
from rag_stack.cost_model.retrieval_worker import build_retrieval_worker_plan
from rag_stack.cost_model.workload import PreparedExecutionPaths, resolve_execution_paths

logger = logging.getLogger("RAG-Stack")

# GPU light stages that, when collocated with the generator AND interleaved
# (agentic rounds), time-multiplex the card → the per-round contention term.
# NOTE: compressors are deliberately NOT in this set — the pre-refactor literal
# ("passage_compressor") never matched a group name (groups carried
# passage_compressor_prefill/_decode), so the contention term never applied to
# them; the flat rename must not silently change that (calibration follow-up).
_GPU_LIGHT_STAGES = {"semantic_retrieval_encode", "passage_reranker"}
_GENERATOR_STAGES = {"generator_prefill", "generator_decode"}


def _groups_collocate(groups) -> bool:
    """True iff some GPU light stage shares a chip group with the generator
    (prefill/decode) — the interleaved time-multiplexing condition. ``groups`` is
    a collocation_strategy: a list of groups, each a list of PerformanceStage names."""
    for grp in (groups or []):
        gs = set(grp)
        if (gs & _GPU_LIGHT_STAGES) and (gs & _GENERATOR_STAGES):
            return True
    return False

_INPUT_GRID = 128
_OUTPUT_GRID = 32

# Compatibility names; workload preparation belongs to workload.py.
from rag_stack.cost_model.workload import (
    MAX_TRACE_QUERIES as _QUERY_CAP,
    summarize_trace as _profile,
)


def _closed_loop_load_concurrency(
    system_config,
    batch_size_request: int,
    batch_size_decode: int | None = None,
) -> int:
    """Structural lower bound for legacy analytical diagnostics.

    Authoritative DynamicEngine throughput comes from the DES's own
    saturation-to-plateau search.  This helper survives only for analytical
    cap/latency diagnostics, and deliberately ignores every measured/load
    harness field in ``system_config``.  Reading those fields anywhere in CM
    would make benchmark population an illicit performance side channel.
    """
    del system_config
    return max(
        1,
        int(batch_size_request),
        int(batch_size_decode or 0),
    )


def _closed_loop_mean_latency(
    trace_mean_s: float,
    qps: float,
    load_concurrency: int,
) -> float:
    """STEADY-STATE mean response latency for the saturated closed loop.

    Little's law: response time = C/μ (client population / bottleneck rate).
    Deliberately steady-state ONLY — a finite measured run's mean is dragged
    ~(1 - 1/(2*waves)) below this by its final drain wave (analytically
    derived, verified ≤0.5% across waves 5→100 on the e2e benchmark), but that
    is a MEASUREMENT-PROTOCOL artifact, not serving behavior, so it is
    compensated on the measured side at benchmark-compare time
    (run_accuracy ``_steady_state_latency_mean``), never modeled here.
    """
    raw = max(0.0, float(trace_mean_s or 0.0))
    rate = float(qps or 0.0)
    if rate <= 0.0 or not np.isfinite(rate):
        return raw
    queued = max(1, int(load_concurrency)) / rate
    return max(raw, queued)


def _expand_deployment_axes(
    df,
    cm_search_space: dict,
    *,
    agentic: bool,
    decode_pinned: bool,
    timeout_pinned: bool,
):
    """Complete missing deployment axes for compatibility callers.

    Concrete RAGO candidates already carry their decode axis. For older rows,
    agentic workloads enumerate configured decode depths, while sequential
    workloads derive the depth from request and generator-stage batch fields.
    Missing batching timeouts expand over ``system.cm_search_space`` (default
    2, 10, and 50 ms). Pinned values suppress expansion. Timeouts affect both
    analytical batch-fill estimates and the simulator's count/timeout queues.
    """
    css = cm_search_space or {}

    def _pow2_grid(cap) -> list:
        vals, v = [], 1
        while v <= max(1, int(cap)):
            vals.append(v)
            v *= 2
        return vals

    if (
        agentic
        and not decode_pinned
        and "batch_size_request" in df.columns
        and "batch_size_decode" not in df.columns
    ):
        dec_grid = css.get("batch_size_decode")
        if dec_grid is None:
            dec_grid = _pow2_grid(css.get("max_batch_size_request", 256))
        dec_grid = sorted({int(v) for v in dec_grid})
        rows, vals = [], []
        for i in range(len(df)):
            b = int(df["batch_size_request"].iloc[i])
            for dv in sorted({b, *[v for v in dec_grid if v > b]}):
                rows.append(i)
                vals.append(int(dv))
        df = df.iloc[rows].reset_index(drop=True)
        df["batch_size_decode"] = vals

    if (
        not agentic
        and not decode_pinned
        and "batch_size_decode" not in df.columns
        and "batch_size_request" in df.columns
    ):
        # Sequential/static rows: RAGO prices the generator stages at their
        # OWN batch axes (batch_size_generator_*), independent of the request
        # batch. The deployed decode concurrency (max_num_seqs) must be the
        # concurrency that pricing assumed — otherwise the saved deployment
        # describes a machine that can never realize the scored qps
        # (s45_eval_0047 postmortem: scored at generator stage batch 128,
        # deployment saved with decode=2 → measured 0.598 vs CM 5.70; CM's
        # own service times at the deployed concurrency reproduce 0.62-0.68).
        # Derive decode = max(request, generator stage batches). Rows whose
        # stage batches ≤ request keep decode = request — bit-identical to
        # the previous fallback. Pinned decode and the fixed-placement
        # benchmark path (always pinned) are untouched.
        gen_cols = [
            c for c in ("batch_size_generator_prefill", "batch_size_generator_decode")
            if c in df.columns
        ]
        if gen_cols:
            dec = df["batch_size_request"].fillna(1).astype(int)
            for c in gen_cols:
                dec = np.maximum(dec, df[c].fillna(0).astype(int))
            df = df.copy()
            df["batch_size_decode"] = dec.astype(int)

    if not timeout_pinned and "dynamic_batch_timeout_s" not in df.columns:
        to_grid = css.get("dynamic_batch_timeout_s")
        if to_grid is None:
            to_grid = [0.002, 0.01, 0.05]
        to_grid = sorted({float(v) for v in to_grid})
        if to_grid:
            n_before = len(df)
            df = df.loc[df.index.repeat(len(to_grid))].reset_index(drop=True)
            df["dynamic_batch_timeout_s"] = to_grid * n_before
    return df


# Mean-TTFT queue fraction — ANALYTICALLY DERIVED, not calibrated: under continuous
# batching a batch of B requests is prefilled in scheduling order, so the AVERAGE
# request's first token follows ~half the batch's prefill (its queue position
# averages (B-1)/2 → 1/2 as B grows). So the TTFT charges the answer prefill × this
# fraction (pre-answer stage times unscaled). 0.5 is the mean queue position, a pure
# scheduling constant (dimensionless, device-independent) — hence a code constant,
# NOT a calibration-profile entry.
_TTFT_PREFILL_FRAC = 0.5

# Prefix caching (vLLM enable_prefix_caching): an agentic generator re-prompts a
# GROWING context each round (round N ⊇ round N-1's prompt + output), so vLLM
# reuses the cached KV of that prefix and re-prefills ONLY the new tokens. The
# walk therefore charges round N's prefill as the INCREMENTAL GenZ cost
# prefill(seq_N) - prefill(cached_prefix) instead of the full prefill(seq_N).
# This is the EXACT incremental causal-prefill cost (GenZ's prefill is causal, so
# the difference = the new tokens' MLP + their attention over the full context) —
# the right way to interact with GenZ without re-deriving attention. Scope: ONLY
# the SAME generator engine across ITS OWN rounds (caching never spans engines —
# embedding/reranker/a different LLM each have their own KV). This serving
# contract is part of the simulator implementation, not an import-time
# environment coefficient.
_PREFIX_CACHING = True

# Retrieval uses the global request batch. The dynamic trace walk prices
# encode/search calls at ``batch_size_request`` and does not expose a separate
# retrieval serving width.


# vLLM chunked-prefill per-step token budget (the max_num_batched_tokens the
# harness launches with) — a serving-protocol constant, not a fitted value.
_PREFILL_CHUNK_TOKENS = 8192


def engine_step_price(
    prefill_fn, decode_fn, *, in_mean: int = 1024, tp: int = 1, pp: int = 1,
):
    """ONE engine step price shared by the fixed-placement caps and the
    closed-loop sim (r26, generator-capacity cut). A continuous-batching
    engine advances in steps: every step serves each of the ``d`` decoding
    sequences one token AND — chunked prefill — a ``c``-token slice of the
    prompt backlog from the same ``max_num_batched_tokens`` budget, so

        step_s(d, c) = decode_s(in_mean, 1, d, tp, pp) + c / chunk_rate
        chunk_rate   = 8192 / prefill_s(8192, 1, tp, pp)

    Both terms are existing GenZ queries — pure mechanism, no fitted
    constants and no measured-grid lookup (the r19c parquet lookup this
    replaces was a per-box benchmark dependency, the opposite of a cost
    model's job). Steady-state identity used by the caps path: a query with
    ``in`` computed prompt tokens and ``out`` output tokens rides steps
    whose prompt slice is ``c = in*d/out``, so its engine occupancy is

        out * step_s(d, in*d/out) / d  ==  out * dec_step(d) / d
                                           + in / chunk_rate.

    Returns ``(step_s, chunk_rate)``; ``chunk_rate`` is inf when
    ``prefill_fn`` cannot price the chunk (callers then fall back to the
    batch-amortized prefill price)."""
    chunk_rate = float("inf")
    if prefill_fn is not None:
        try:
            _chunk_s = float(prefill_fn(
                _PREFILL_CHUNK_TOKENS, 1, int(tp), int(pp)) or 0.0)
        except Exception:  # noqa: BLE001 — unpriceable chunk → fallback
            _chunk_s = 0.0
        if _chunk_s > 0:
            chunk_rate = _PREFILL_CHUNK_TOKENS / _chunk_s

    def step_s(d: int, c: float) -> float:
        s = 0.0
        if d > 0 and decode_fn is not None:
            s += float(decode_fn(int(in_mean), 1, int(d), int(tp), int(pp)))
        if c > 0 and np.isfinite(chunk_rate) and chunk_rate > 0:
            s += float(c) / chunk_rate
        return s

    return step_s, chunk_rate


def _bucket(t: int, grid: int) -> int:
    """Round ``t`` up to the nearest positive multiple of ``grid``."""
    t = max(0, int(t))
    return max(grid, ((t + grid - 1) // grid) * grid)


# ---------------------------------------------------------------------------
# CallLatency — per-component call → latency (seconds).
# ---------------------------------------------------------------------------

@dataclass
class CallLatency:
    """Per-component call-latency functions. ``None`` = stage unavailable.

    Each callable returns the wall-clock seconds to process a *batch* of
    ``batch`` homogeneous calls of the given size (so per-call latency within
    a saturated batch ≈ the returned value). This is purely a latency lookup
    into the existing cost models — no notion of price/money. The latency walk
    passes the deployed stage batch for the component being priced.
    """

    prefill_s: Optional[Callable[..., float]] = None                 # (input_tokens, batch, tp=1, pp=1)
    decode_s: Optional[Callable[..., float]] = None                  # (input_tokens, output_tokens, batch, tp=1, pp=1)
    prefill_by_engine: Dict[str, Callable[..., float]] = field(default_factory=dict)
    decode_by_engine: Dict[str, Callable[..., float]] = field(default_factory=dict)
    # DES compatibility decomposition for the LLM component profile.  Current
    # two-role component records are zero-intercept by contract, so
    # ``call_overhead_by_engine`` must remain exactly zero.  Real scheduler-cycle
    # overhead has one owner: the hardware x collocated-stage x resident-batch
    # curve applied after prefill/decode component work is composed.
    prefill_kernel_by_engine: Dict[str, Callable[..., float]] = field(
        default_factory=dict,
    )
    call_overhead_by_engine: Dict[str, float] = field(default_factory=dict)
    encode_s: Optional[Callable[[int], float]] = None                # (batch) query embedding (GPU)
    # ``retrieve_s`` is the K=1 compatibility service excluding encode.  The
    # structured fields below let the worker plan repeat backend service for
    # flattened query-expansion chunks. RRF is a separate compiled invocation.
    retrieve_s: Optional[Callable[[int], float]] = None              # (user batch) K=1 search + worker overhead
    retrieve_kernel_s: Optional[Callable[[int], float]] = None       # (backend batch) one FAISS chunk
    retrieve_backend_overhead_s: Optional[Callable[[int], float]] = None  # (backend batch) dispatch/copy/glue
    retrieve_outer_overhead_s: Optional[Callable[[int], float]] = None  # legacy non-fusion worker glue
    retrieve_fused_overhead_s: Optional[Callable[[int], float]] = None  # legacy K=1 backend + outer sum
    retrieval_fanout: int = 1                                       # backend rows per user row
    rrf_s: Optional[Callable[[int], float]] = None                   # explicit count-only test/service injection
    rrf_call_s: Optional[Callable[[dict, int], float]] = None        # observed call + repeated user batch
    bm25_retrieve_s: Optional[Callable[[int], float]] = None         # (batch) CPU keyword search
    rerank_s: Optional[Callable[..., float]] = None                  # (input_tokens, batch) DP-only (no TP)
    rerank_batch_s: Optional[Callable[[Sequence[int]], float]] = None  # heterogeneous user-row tokens
    rerank_forward_tiles_s: Optional[
        Callable[..., Sequence[float]]
    ] = None                                                          # heterogeneous GPU forward durations
    rerank_glue_s: Optional[Callable[[int], float]] = None           # (batch) host-side share of rerank_s (excluded from GPU aggregates)
    rerank_fanout: int = 1                                            # pairs per query row (reranker input fanout)
    # r26 padded-shape policy hooks (None = component without them):
    rerank_pad_excess: Optional[Callable[[int], Optional[int]]] = None  # (batch) E[padded seq]-E[pair] from corpus order stats; None = distribution absent
    rerank_forward_batch: Optional[Callable[[int], int]] = None      # (batch) pairs per forward (shared runtime policy, e.g. monot5 -> 8)
    compress_s: Optional[Callable[[int, int], float]] = None          # (input_tokens, batch) passage compressor (e.g. llmlingua2)
    pricing_binding: dict = field(default_factory=dict)
    validate_pricing_binding: Optional[Callable[[], None]] = None
@dataclass
class ChainPerf:
    """Per-request performance from ONE trace walk — two views of the same work:

      * **Critical-path latency** (``e2e`` / ``ttft`` / ``tpot``): the serial sum
        over a query's ordered op chain — what a single request experiences.
      * **Per-RESOURCE service times** (``svc_*``): the mean per-query work that
        lands on each physical resource (generator GPU / retrieval CPU+embed /
        reranker / compressor). Throughput is bounded by the BOTTLENECK resource
        (max of these), so emitting them lets qps be computed DIRECTLY from the
        bottleneck instead of inheriting the full critical-path latency error.

    Latency sums ALL stages on the path (accumulates every stage's error);
    throughput uses each stage's own service demand divided by its serving batch.
    Computing both from the same walk keeps them consistent while decoupling their
    error surfaces.
    """

    e2e: float = 0.0            # median per-request critical-path latency (s)
    e2e_mean: float = 0.0       # mean per-request critical-path latency (s)
    ttft: float = 0.0           # mean time-to-first-token (s)
    tpot: float = 0.0           # mean per-step decode (s)
    svc_gen: float = 0.0        # mean per-query generator GPU service (prefill+decode+PD)
    svc_prefill: float = 0.0    # mean per-query prefill batch service
    svc_decode: float = 0.0     # mean per-query decode batch service (+PD residual)
    svc_encode: float = 0.0     # mean per-query query-embedding service (GPU)
    svc_retrieval: float = 0.0  # CPU retrieval service, including local fusion
    svc_fusion: float = 0.0     # CPU fusion share of svc_retrieval (not added twice)
    svc_rerank: float = 0.0     # mean per-query reranker service
    # Card-side share of svc_rerank (kernel forwards × scale, EXCLUDING the
    # host-side worker glue) — what GPU-group demand / decode contention may
    # charge; the glue is host work during which the card is free.
    svc_rerank_gpu: float = 0.0
    rerank_calls_per_query: float = 0.0
    svc_compress: float = 0.0   # mean per-query compressor service
    svc_gen_by_stage: dict[str, float] = field(default_factory=dict)
    svc_prefill_by_stage: dict[str, float] = field(default_factory=dict)
    svc_decode_by_stage: dict[str, float] = field(default_factory=dict)
    decode_batch: int = 1       # deployed decode-engine sequence concurrency
    retrieval_calls_per_query: float = 0.0
    gen_in_tokens_per_query: float = 0.0   # mean prompt tokens across a query's generate calls (main engine)
    gen_out_tokens_per_query: float = 0.0  # mean output tokens across a query's generate calls (main engine)
    retrieval_calls_before_ttft: float = 0.0

    def __iter__(self):
        # Back-compat: legacy callers unpack ``e2e, ttft, tpot = walk(...)``.
        return iter((self.e2e, self.ttft, self.tpot))

    def __getitem__(self, i):
        # Back-compat: legacy callers index ``walk(...)[0]`` for e2e/ttft/tpot.
        return (self.e2e, self.ttft, self.tpot)[i]


def _unpriced_bm25_service(_batch: int) -> float:
    """Reject BM25 until an exact standalone CPU/corpus profile exists."""

    raise ValueError(
        "BM25 is not priceable: the retired 3 ms constant was not standalone "
        "calibration. Measure and register an exact CPU/corpus BM25 component "
        "profile before using lexical_retrieval in production CM."
    )


def build_call_latency(
    pipeline_config: dict,
    hardware: SystemConfig,
    *,
    bind_profiles: bool = False,
) -> CallLatency:
    """Build a GenZ/FAISS-backed :class:`CallLatency` from the resolved config.

    Reuses :func:`rag_stack.cost_model.sweep_inputs.build_performance_stage_cost_models`
    for component→cost-model dispatch. The LLM (``generate``) latency resolves
    the model + calibration ONCE and then queries GenZ ``prefill_moddeling`` /
    ``decode_moddeling`` at arbitrary token counts, memoized on the bucketed
    ``(tokens, batch)`` key. Retrieval / rerank reuse their cost model's
    ``estimate()`` per call.

    The LLM ``prefill_s``/``decode_s`` accept ``(tp, pp)`` and query GenZ at
    that parallelism; the walk passes the SELECTED deployment's
    ``winning_tp/winning_pp`` per stage (see ``RAGCMAssembly.evaluate_dynamic``), so the
    walk's single-query latency is TP-aware exactly as RAGO's own latency is.
    Retrieval (FAISS, CPU) and the reranker (encoder, DP-only) have no TP axis,
    so those stay tp-agnostic.
    """
    from rag_stack.cost_model.sweep_inputs import (
        build_performance_stage_cost_models,
        _get_compressor_input_fanout,
        _get_query_expansion_fanout,
        _get_reranker_input_fanout,
    )

    stage_models = build_performance_stage_cost_models(pipeline_config)
    profile_bindings = None
    if bind_profiles:
        from rag_stack.cost_model.pricing_binding import capture_pricing_bindings
        profile_bindings = capture_pricing_bindings(stage_models, pipeline_config, hardware)
    latency = CallLatency()
    latency.pricing_binding["components"] = {
        stage: f"{type(model).__module__}.{type(model).__qualname__}"
        for stage, model in stage_models.items()
    }

    def _with_request_batch(batch: int) -> dict:
        cfg = deepcopy(pipeline_config)
        selected_vdbs = set()
        for node_line in cfg.get("node_lines", []):
            for node in node_line.get("nodes", []):
                if node.get("stage") not in ("semantic_retrieval", "hybrid_retrieval"):
                    continue
                for module in node.get("modules", []):
                    module["batch_size"] = max(1, int(batch))
                    if module.get("vectordb"):
                        selected_vdbs.add(str(module["vectordb"]))
        for vdb in cfg.get("vectordb", []):
            if not selected_vdbs or str(vdb.get("name")) in selected_vdbs:
                vdb["batch_size"] = max(1, int(batch))
        return cfg

    def _with_node_module_batch(stage: str, batch: int) -> dict:
        cfg = deepcopy(pipeline_config)
        for node_line in cfg.get("node_lines", []):
            for node in node_line.get("nodes", []):
                if node.get("stage") != stage:
                    continue
                for module in node.get("modules", []):
                    module["batch_size"] = max(1, int(batch))
        return cfg

    def _module_type(stage: str) -> str | None:
        return _component_of(pipeline_config, stage)

    # --- generate (LLM prefill + decode) via GenZ, calibrated, cached -------
    # The trace component stays "generate", but each call belongs to a logical
    # vLLM engine (main generator, query_expansion, ...). Keep one GenZ cache per
    # engine so query-expansion calls do not accidentally use the main generator
    # model/topology.
    from rag_stack.cost_model.llm_sim.genz_cache import (
        cached_prefill_moddeling as prefill_moddeling,
        cached_decode_moddeling as decode_moddeling,
        scheduler_batch_traversal_latency_ms,
    )
    from rag_stack.cost_model.llm_sim.tp_efficiency import (
        pipeline_fill_drain_factor,
    )

    def _install_llm_engine(engine: str, prefill_stage: str, decode_stage: str) -> None:
        llm = stage_models.get(prefill_stage) or stage_models.get(decode_stage)
        if llm is None:
            return

        model = llm._get_model(pipeline_config)
        # _resolve_calibration returns 5 values since the affine per-call
        # overhead landed (e7f15d16); this dynamic-engine call site was left
        # on 4-tuple unpacking and crashed every react/QE eval (07-08,
        # eval_0039 "too many values to unpack"). The overhead is charged
        # ONCE per model call — one prefill per call — mirroring
        # calibration._predict; records without the field resolve to 0.0 and
        # keep every number bit-identical.
        system_name, ceff, meff, bits, call_overhead_s = llm._resolve_calibration(
            hardware, pipeline_config,
        )
        latency.pricing_binding.setdefault("llm_engines", {})[engine] = {
            "model": str(model), "system": str(system_name),
            "compute_efficiency": float(ceff), "memory_efficiency": float(meff),
            "bits": bits, "call_overhead_s": float(call_overhead_s or 0.0),
        }
        # _resolve_calibration invokes the shared strict two-role component
        # validator before returning this compatibility field.  Do not clamp a
        # stale intercept into something apparently admissible here.
        _call_oh_s = float(call_overhead_s or 0.0)
        if _call_oh_s != 0.0:
            raise ValueError(
                f"two-role LLM engine {engine!r} received a non-zero component "
                "per_call_overhead_s after strict record validation"
            )
        _pref_raw_cache: Dict[tuple, float] = {}
        _pref_kernel_cache: Dict[tuple, float] = {}
        _pref_cache: Dict[tuple, float] = {}
        _dec_raw_cache: Dict[tuple, float] = {}
        _dec_cache: Dict[tuple, float] = {}

        def _pipeline_adjusted(raw_s: float, batch: int, pp: int) -> float:
            """Apply the same resident-batch PP law as sweep/exact pricing."""
            if raw_s <= 0:
                return raw_s
            return raw_s * pipeline_fill_drain_factor(int(batch), int(pp))

        def _candidate_devices(stage: str, tp: int, pp: int) -> list[str] | None:
            """One exact model-parallel replica from a materialized layout.

            Candidate layouts retain every DP replica in ``devices``; a single
            GenZ service curve prices one replica, so select its explicit rank
            order instead of passing the flattened DP set to a TP*PP query.
            """
            system = (pipeline_config.get("system") or {})
            layout = system.get("layout") if isinstance(system, dict) else None
            if not isinstance(layout, dict):
                return None
            engines = layout.get("engines") or {}
            info = engines.get(engine) if isinstance(engines, dict) else None
            if not isinstance(info, dict):
                return None
            record = info.get(stage) if info.get("pd_serving") == "disagg_pd" else info
            if not isinstance(record, dict):
                return None
            replicas = record.get("replica_devices")
            if isinstance(replicas, list) and replicas and isinstance(replicas[0], list):
                devices = [str(device) for device in replicas[0] if device]
            else:
                devices = [str(device) for device in (record.get("devices") or []) if device]
            # A complete ``replica_devices`` record may contain more than the
            # exact model-parallel group requested by this analytical call.
            # Preserve candidate rank order and consume only its TP*PP prefix;
            # never substitute or reallocate physical devices here.
            width = max(1, int(tp) * int(pp))
            return devices[:width] or None

        def _system_for_parallelism(role: str, tp: int, pp: int):
            if not hasattr(llm, "_system_config_for_parallelism"):
                return system_name
            stage = prefill_stage if role == "prefill" else decode_stage
            return llm._system_config_for_parallelism(
                system_name,
                hardware,
                pipeline_config,
                stage,
                int(tp),
                int(pp),
                devices_override=_candidate_devices(stage, tp, pp),
            )

        def _raw_prefill_s(input_tokens: int, batch: int, tp: int = 1, pp: int = 1) -> float:
            key = (_bucket(input_tokens, _INPUT_GRID), int(batch), int(tp), int(pp))
            if key not in _pref_raw_cache:
                parallel_system = _system_for_parallelism("prefill", key[2], key[3])
                out = prefill_moddeling(
                    model=model, batch_size=key[1], input_tokens=key[0],
                    system_name=parallel_system, bits=bits, ceff=ceff, meff=meff,
                    tensor_parallel=int(tp), pipeline_parallel=int(pp),
                    model_offload=True, debug=False,
                )
                _pref_raw_cache[key] = scheduler_batch_traversal_latency_ms(
                    out,
                    pipeline_parallel=int(pp),
                ) / 1000.0
            return _pref_raw_cache[key]

        def _raw_decode_s(input_tokens: int, output_tokens: int, batch: int,
                          tp: int = 1, pp: int = 1) -> float:
            key = (
                _bucket(input_tokens, _INPUT_GRID),
                _bucket(output_tokens, _OUTPUT_GRID),
                int(batch), int(tp), int(pp),
            )
            if key not in _dec_raw_cache:
                parallel_system = _system_for_parallelism("decode", key[3], key[4])
                out = decode_moddeling(
                    Bb=1,  # vLLM serves beam=1; GenZ default Bb=4 quadrupled KV accounting (fake offload at 14B b256) and decode bytes — r13
                model=model, batch_size=key[2], input_tokens=key[0],
                    output_tokens=key[1], system_name=parallel_system, bits=bits,
                    ceff=ceff, meff=meff, tensor_parallel=int(tp),
                    pipeline_parallel=int(pp), model_offload=True, debug=False,
                )
                _dec_raw_cache[key] = scheduler_batch_traversal_latency_ms(
                    out,
                    pipeline_parallel=int(pp),
                ) / 1000.0
            return _dec_raw_cache[key]

        def prefill_kernel_s(
            input_tokens: int,
            batch: int,
            tp: int = 1,
            pp: int = 1,
        ) -> float:
            key = (_bucket(input_tokens, _INPUT_GRID), int(batch), int(tp), int(pp))
            if key not in _pref_kernel_cache:
                raw = _raw_prefill_s(input_tokens, batch, tp, pp)
                _pref_kernel_cache[key] = _pipeline_adjusted(raw, batch, pp)
            return _pref_kernel_cache[key]

        def prefill_s(input_tokens: int, batch: int, tp: int = 1, pp: int = 1) -> float:
            key = (_bucket(input_tokens, _INPUT_GRID), int(batch), int(tp), int(pp))
            if key not in _pref_cache:
                # Analytical full-call pricing keeps the affine component
                # intercept once.  DES consumes ``prefill_kernel_s`` instead
                # and schedules this intercept as separate request work.
                _pref_cache[key] = (
                    prefill_kernel_s(input_tokens, batch, tp, pp)
                    + _call_oh_s
                )
            return _pref_cache[key]

        def decode_s(input_tokens: int, output_tokens: int, batch: int,
                     tp: int = 1, pp: int = 1) -> float:
            key = (
                _bucket(input_tokens, _INPUT_GRID),
                _bucket(output_tokens, _OUTPUT_GRID),
                int(batch), int(tp), int(pp),
            )
            if key not in _dec_cache:
                raw = _raw_decode_s(input_tokens, output_tokens, batch, tp, pp)
                _dec_cache[key] = _pipeline_adjusted(raw, batch, pp)
            return _dec_cache[key]

        latency.prefill_by_engine[engine] = prefill_s
        latency.decode_by_engine[engine] = decode_s
        latency.prefill_kernel_by_engine[engine] = prefill_kernel_s
        latency.call_overhead_by_engine[engine] = _call_oh_s
        if engine == "generator":
            latency.prefill_s = prefill_s
            latency.decode_s = decode_s

    _install_llm_engine("generator", "generator_prefill", "generator_decode")
    _install_llm_engine(
        "query_expansion",
        "query_expansion_prefill",
        "query_expansion_decode",
    )

    # --- retrieval = encode (query embedding) + vectordb search, cached -----
    # A real retrieve call embeds the (sub-)query on GPU, then searches FAISS
    # on CPU; both cost models contribute, keeping the trace path at the same
    # granularity as the sequential CM ("encode" stage) and the measured split.
    # ``encode`` (query embedding, GPU) and ``retrieve`` (FAISS, CPU) are
    # separate trace components.  The analytical walk expands their config-
    # derived K backend rows through the shared physical-worker plan below;
    # the trace itself remains one quality invocation pair per user-row visit.
    rcm = stage_models.get("semantic_retrieval_vectorsearch")
    encode_cm = stage_models.get("semantic_retrieval_encode")
    if rcm is not None:
        _ret_kernel_cache: Dict[int, float] = {}
        def retrieve_kernel_s(batch: int) -> float:
            batch = max(1, int(batch))
            if batch not in _ret_kernel_cache:
                res = rcm.estimate(_with_request_batch(batch), hardware)
                _ret_kernel_cache[batch] = max(
                    float(res.latency_s), 1e-9,
                )
            return _ret_kernel_cache[batch]

        def retrieve_backend_overhead_s(batch: int) -> float:
            # The retired assembly profile fitted whole-stage pandas/dispatch
            # residuals against end-to-end runs. Until those worker operators
            # have isolated component profiles, do not invent service demand.
            del batch
            return 0.0

        def retrieve_fused_overhead_s(batch: int) -> float:
            """Legacy non-fusion worker glue; RRF has its own service."""
            return retrieve_backend_overhead_s(batch)

        def retrieve_s(batch: int) -> float:
            return (
                retrieve_kernel_s(batch)
                + retrieve_fused_overhead_s(batch)
            )

        latency.retrieve_s = retrieve_s
        latency.retrieve_kernel_s = retrieve_kernel_s
        latency.retrieve_backend_overhead_s = retrieve_backend_overhead_s
        latency.retrieve_outer_overhead_s = None
        latency.retrieve_fused_overhead_s = retrieve_fused_overhead_s
        latency.retrieval_fanout = max(
            1, int(_get_query_expansion_fanout(pipeline_config)),
        )

    if encode_cm is not None:
        _enc_cache: Dict[int, float] = {}

        def encode_s(batch: int) -> float:
            if batch not in _enc_cache:
                _enc_cache[batch] = max(
                    float(
                        encode_cm.estimate(
                            _with_request_batch(batch), hardware,
                        ).latency_s
                    ),
                    1e-9,
                )
            return _enc_cache[batch]

        latency.encode_s = encode_s

    # --- reranker (encoder) — per-call latency from estimate(), cached ------
    rerank_cm = stage_models.get("passage_reranker")
    if rerank_cm is not None:
        from rag_stack.cost_model.reranker_policy import (
            reranker_forward_batch_from_system,
            reranker_padding_group_pairs,
        )
        from rag_stack.cost_model.token_stats import TokenStats

        _rr_cache: Dict[int, float] = {}
        _rr_forward_cache: Dict[tuple[int, int], float] = {}
        _rr_fanout = max(1, int(_get_reranker_input_fanout(pipeline_config)))
        _rr_component = _module_type("passage_reranker")
        _rr_system_config = pipeline_config.get("system") or {}

        def _rr_forward_batch(batch: int) -> int:
            return reranker_forward_batch_from_system(
                _rr_component,
                max(1, int(batch)),
                _rr_system_config,
            )
        # Padded-shape source (r26): the corpus chunk-length distribution.
        # ``chunk_pad_excess_tokens`` is pure order statistics (E[max over the
        # padding group] - E[chunk]); legacy ``_token_stats`` dicts without
        # the percentile grid return None and keep the pre-r26 mean shape.
        try:
            _rr_token_stats = TokenStats.resolve(pipeline_config)
        except Exception:  # noqa: BLE001 — stats absent → mean-shape fallback
            _rr_token_stats = None

        def _rr_pad_excess(batch: int) -> int | None:
            if _rr_token_stats is None:
                return None
            group = reranker_padding_group_pairs(
                _rr_component, max(1, int(batch)), _rr_fanout,
                forward_batch=_rr_forward_batch(batch),
            )
            return _rr_token_stats.chunk_pad_excess_tokens(group)

        latency.rerank_fanout = _rr_fanout
        latency.rerank_pad_excess = _rr_pad_excess
        latency.rerank_forward_batch = _rr_forward_batch

        def rerank_s(input_tokens: int, batch: int) -> float:
            # Per-pair tokens from the ACTUAL text that flowed through this
            # call (the trace records the whole per-query rerank input =
            # fanout pairs), bucketed to 32-token steps to bound the price
            # cache. r14: input_tokens was accepted-but-ignored, so every
            # reranker priced at the config token stats' shape — tart 4.6x
            # under (95-token stat vs ~2.3k-token real pairs), minilm 8x
            # over. The record's max_seq_len clamp inside estimate() caps
            # this at what the production tokenizer actually feeds the model.
            #
            # Predict padding only from the quality invocation's mean pair
            # shape plus the declared corpus/config distribution. Runtime
            # padding groups and observed forward-batch shapes are not CM
            # inputs.
            per_pair = (
                int(math.ceil(float(input_tokens) / _rr_fanout))
                if input_tokens and input_tokens > 0 else 0
            )
            if per_pair > 0:
                _excess = _rr_pad_excess(batch)
                if _excess:
                    per_pair += int(_excess)
            bucket = ((per_pair + 31) // 32) * 32 if per_pair > 0 else 0
            key = (int(batch), bucket)
            if key not in _rr_cache:
                # Rerankers score one query-passage pair per model item.
                # measured mode dynamically batches user requests, then tiles
                # the resulting B * top_k pair workload through a per-forward
                # cap. Price one forward at the SHARED runtime forward
                # microbatch (rag_stack.cost_model.reranker_policy —
                # monot5 forwards 8 pairs regardless of the request batch;
                # single source with the measured evaluator) and multiply
                # by the number of tiled forwards.
                cap = _rr_forward_batch(batch)
                total_items = max(1, int(batch)) * _rr_fanout
                forwards = max(1, math.ceil(total_items / cap))
                _cfg = _with_node_module_batch("passage_reranker", cap)
                if bucket > 0:
                    # Deep-copied config: route the observed shape through
                    # TokenStats' official override field instead of the
                    # avg_chunk_tokens estimate.
                    _cfg.setdefault("_token_stats", {})[
                        "avg_passage_reranker_input_tokens"] = bucket
                res = rerank_cm.estimate(_cfg, hardware)
                _rr_cache[key] = max(
                    float(res.latency_s) * forwards,
                    1e-9,
                )
            return _rr_cache[key]

        latency.rerank_s = rerank_s

        def _rerank_forward_kernel_s(items: int, bucket: int) -> float:
            key = (max(1, int(items)), max(0, int(bucket)))
            if key not in _rr_forward_cache:
                cfg = _with_node_module_batch("passage_reranker", key[0])
                if key[1] > 0:
                    cfg.setdefault("_token_stats", {})[
                        "avg_passage_reranker_input_tokens"
                    ] = key[1]
                result = rerank_cm.estimate(cfg, hardware)
                _rr_forward_cache[key] = max(
                    float(result.latency_s), 1e-9,
                )
            return _rr_forward_cache[key]

        def rerank_forward_tiles_s(
            input_tokens: Sequence[int],
            request_batch_size: Optional[int] = None,
        ) -> Sequence[float]:
            """Price the real heterogeneous dynamic batch by forward tiles.

            Quality trace tokens remain in user-row units.  Expand each row by
            the declared retrieval top-k, apply the public padding-group policy,
            and price every runtime forward at that tile's largest pair shape.
            No measured forward-batch shape or fitted contention coefficient is
            consumed here.
            """
            rows = [max(0, int(tokens)) for tokens in input_tokens]
            if not rows:
                return ()
            # DP replicas receive contiguous input shards, but the runtime's
            # public forward-cap/padding policy is selected from the original
            # user-request batch before that split.
            batch = max(1, int(request_batch_size or len(rows)))
            excess = int(_rr_pad_excess(batch) or 0)
            pair_shapes: List[int] = []
            for tokens in rows:
                per_pair = (
                    int(math.ceil(float(tokens) / _rr_fanout))
                    if tokens > 0 else 0
                )
                per_pair = max(0, per_pair + excess)
                pair_shapes.extend([per_pair] * _rr_fanout)
            cap = _rr_forward_batch(batch)
            tiles: List[float] = []
            for start in range(0, len(pair_shapes), cap):
                tile = pair_shapes[start:start + cap]
                padded = max(tile, default=0)
                bucket = (
                    ((padded + 31) // 32) * 32 if padded > 0 else 0
                )
                tiles.append(
                    _rerank_forward_kernel_s(len(tile), bucket)
                )
            return tuple(tiles)

        def rerank_batch_s(input_tokens: Sequence[int]) -> float:
            rows = [max(0, int(tokens)) for tokens in input_tokens]
            if not rows:
                return 0.0
            return max(
                sum(rerank_forward_tiles_s(rows)),
                1e-9,
            )

        latency.rerank_batch_s = rerank_batch_s
        latency.rerank_forward_tiles_s = rerank_forward_tiles_s

    # --- passage compressor (e.g. llmlingua2 prune) — per-call latency ------
    # Pruning compressors own the bare ``passage_compressor`` PerformanceStage;
    # generative ones (tree_summarize/refine) the ``passage_compressor_prefill``
    # /_decode pair — a pipeline has exactly one kind. Pruning input shape is
    # structural (effective top_k plus corpus/chunker units), so the dedicated
    # request-batch entry below must not infer it from measured/quality-call
            # token fields. This is the stage the trace walk would otherwise drop —
    # adding it makes the latency chain complete for compress pipelines.
    comp_cm = (stage_models.get("passage_compressor")
               or stage_models.get("passage_compressor_prefill"))
    if comp_cm is not None:
        _comp_cache: Dict[int, float] = {}
        _comp_fanout = max(1, int(_get_compressor_input_fanout(pipeline_config)))
        _estimate_request_batch = getattr(
            comp_cm, "estimate_request_batch", None,
        )

        def compress_s(input_tokens: int, batch: int) -> float:
            # Compressor trace token/byte fields are deliberately not consumed:
            # LLMLingua input work is defined in schema BERT-unit statistics +
            # effective top_k, and archived producers did not reliably record
            # the pre-compression/component-tokenizer shape at this call.
            del input_tokens
            if batch not in _comp_cache:
                cap = max(1, int(batch))
                if _estimate_request_batch is not None:
                    # Authoritative USER-request-batch entry shared with
                    # LLMLingua's RAGO sweep. It applies effective top_k and
                    # corpus elems-per-passage exactly once.
                    res = _estimate_request_batch(
                        pipeline_config, hardware, cap,
                    )
                    _comp_cache[batch] = max(
                        float(res.latency_s), 1e-9,
                    )
                elif getattr(comp_cm, "linear_in_batch", False):
                    total_items = cap * _comp_fanout
                    # Linear-law compressor (llmlingua2): price the whole
                    # request batch in ONE call so fixed_ms is charged
                    # once per pure() — tiling through the cap charged it
                    # ceil(total/cap) times (+4.3% at the corrected elem
                    # shape, r12).
                    res = comp_cm.estimate(
                        _with_node_module_batch(
                            "passage_compressor", total_items,
                        ),
                        hardware,
                    )
                    _comp_cache[batch] = max(
                        float(res.latency_s), 1e-9,
                    )
                else:
                    total_items = cap * _comp_fanout
                    # Non-linear models: tile through the per-forward cap,
                    # matching measured dynamic batching.
                    forwards = max(1, math.ceil(total_items / cap))
                    res = comp_cm.estimate(
                        _with_node_module_batch("passage_compressor", cap),
                        hardware,
                    )
                    _comp_cache[batch] = max(
                        float(res.latency_s) * forwards,
                        1e-9,
                    )
            return _comp_cache[batch]

        latency.compress_s = compress_s
    fusion_cm = stage_models.get("retrieval_fusion")
    if fusion_cm is not None:
        from rag_stack.cost_model.rrf import fusion_service_for_call
        fusion_services: dict[str, float] = {}

        def observed_rrf_service(call, batch):
            key = json.dumps({name: call.get(name) for name in
                              ("stage", "model_id", "input_profile", "output_profile")},
                             sort_keys=True)
            if key not in fusion_services:
                fusion_services[key] = fusion_service_for_call(call, pipeline_config, hardware)
            return max(1, int(batch)) * fusion_services[key]

        latency.rrf_call_s = observed_rrf_service

    # BM25 is distinct from dense retrieval, but no portable standalone
    # CPU/corpus calibration exists yet.  Fail closed if a quality trace invokes
    # it instead of injecting the historical hard-coded 3 ms service.
    latency.bm25_retrieve_s = _unpriced_bm25_service

    if profile_bindings is not None:
        profile_bindings.attach(latency)
    return latency


def _component_of(pipeline_config: dict, stage: str) -> str | None:
    for node_line in pipeline_config.get("node_lines", []) or []:
        for node in node_line.get("nodes", []) or []:
            if node.get("stage") != stage:
                continue
            modules = node.get("modules", []) or []
            if modules:
                return str(modules[0].get("component") or "")
    return None


def _llm_stage_names(engine: str) -> tuple[str, str]:
    """The (prefill, decode) PerformanceStage names of an LLM engine (= its
    owning stage) — a taxonomy table lookup, no string composition."""
    from rag_stack.taxonomy import TAXONOMY
    prefill = decode = None
    for ps in TAXONOMY.performance_stages_of(engine):
        if ps.kind == "prefill":
            prefill = ps.name
        elif ps.kind == "decode":
            decode = ps.name
    if prefill is None or decode is None:
        raise KeyError(f"{engine!r} is not an llm-shaped stage")
    return prefill, decode


def _fixed_llm_tp_pp_from_layout(
    pipeline_config: dict,
    stage: str,
) -> tuple[int, int] | None:
    system = pipeline_config.get("system") or {}
    layout = system.get("layout")
    if layout is None:
        return None
    if not isinstance(layout, dict):
        raise ValueError("system.layout must be a mapping")
    engines = layout.get("engines")
    if not isinstance(engines, dict):
        raise KeyError("system.layout.engines is required")

    from rag_stack.taxonomy import TAXONOMY

    ps = TAXONOMY.performance_stages.get(stage)
    if ps is None or TAXONOMY.user_stage_of(stage) not in TAXONOMY.llm_stages:
        return None
    engine = TAXONOMY.user_stage_of(stage).value

    info = engines.get(engine)
    if not isinstance(info, dict):
        return None
    # Disagg role records are keyed by the FULL PerformanceStage name (`stage`
    # here IS that name); collocated engines carry tp/pp on the entry itself.
    record = (info.get(stage)
              if info.get("pd_serving") == "disagg_pd" else info)
    if not isinstance(record, dict):
        raise KeyError(f"system.layout.engines.{engine}.{stage} is required")
    try:
        devices = record["devices"]
        tp = max(1, int(record["tp"]))
        pp = max(1, int(record["pp"]))
    except KeyError as exc:
        raise KeyError(
            f"system.layout.engines.{engine}.{stage} requires devices/tp/pp"
        ) from exc
    if not isinstance(devices, list) or not devices:
        raise ValueError(
            f"system.layout.engines.{engine}.{stage}.devices must be non-empty"
        )
    if tp * pp != len(devices):
        raise ValueError(
            f"system.layout.engines.{engine}.{stage} TP*PP={tp}*{pp} "
            f"does not match devices={len(devices)}"
        )
    return tp, pp


# The trace's ``stage`` field IS the engine identity for LLM calls — the old
# _generate_engine_from_trace heuristics (model-id matching, final-generate
# position) are gone: every producer stamps the stage, and a missing stage is
# a hard error, not a guess.
_LLM_TRACE_STAGES = ("generator", "query_expansion")
_RETRIEVAL_TRACE_STAGES = ("semantic_retrieval_vectorsearch", "lexical_retrieval")


def _trace_stage(call: dict) -> str:
    stage = call.get("stage")
    if not stage:
        raise KeyError(f"trace call missing required 'stage' field: {call!r}")
    return str(stage)


# ---------------------------------------------------------------------------
# Latency — our own analytical serial-chain walk (compute, placement-free).
# ---------------------------------------------------------------------------

def _compute_chain_latency(
    traces: TracePayload, latency: CallLatency,
    prefill_tp: int = 1, prefill_pp: int = 1,
    decode_tp: int = 1, decode_pp: int = 1,
    batch: int = 1,
    overlap_aux_generates: bool = False,
    gpu_collocated_with_gen: bool = False,
    prefix_caching: bool = False,
    decode_batch: int = 0,
    llm_parallelism: Optional[dict[str, dict[str, tuple[int, int]]]] = None,
    *,
    prepared_paths: PreparedExecutionPaths | None = None,
) -> "ChainPerf":
    """Compute per-request service paths and resource demands from ordered calls.

    Each invocation uses its recorded input/output token lengths and the
    candidate's deployed batch axes. LLM calls combine prefill with generated
    token steps; retrieval, encoding, reranking, and compression use their
    component service functions. Prefix reuse and auxiliary-call overlap follow
    the supplied workflow/serving policy.

    ``llm_parallelism`` supplies the concrete candidate's TP/PP per engine and
    role. The scalar prefill/decode TP/PP arguments support legacy callers.
    Returns ``ChainPerf`` with mean and median request service-path latency,
    mean TTFT/TPOT, and mean per-request resource demands. These are analytical
    service quantities, not multi-request queue simulation. The caller adds
    inter-stage communication separately through ``comm_latency_s``.
    """
    # Direct callers retain validation and input isolation. Candidate walks
    # borrow the same prepared snapshot; neither this walk nor its production
    # service functions mutate the observed calls.
    prepared_paths = resolve_execution_paths(
        traces, prepared_paths,
        retrieval_fanout=max(1, int(latency.retrieval_fanout)),
    )
    traces = prepared_paths.paths
    if latency.rrf_s is None and latency.rrf_call_s is None and any(
        _trace_stage(call) == "retrieval_fusion" for q in traces for call in q
    ):
        raise ValueError("compiled retrieval_fusion requires an RRF service model")
    batch = max(1, int(batch))
    # Per-request latency is priced at the deployed PER-ENGINE batch. Data-parallel
    # replicas do NOT reduce it (each replica still serves the full batch) — DP
    # raises THROUGHPUT (×replicas) not latency, applied to qps in evaluate. TP/PP
    # (which DO lower latency) are already in the prefill/decode (tp,pp) args.
    # Independent LLM batch axes:
    #   _pb = batch_size_request — the per-round prefill/request cohort.
    #   _db = batch_size_decode   — the decode-engine concurrency (the deployed
    #         max_num_seqs). Defaults to the request batch; when set independently
    #         (decode_batch ≥ batch) the decode steps are priced at that larger
    #         concurrency.
    # Retrieval uses the global request batch.
    _pb = batch
    _db = max(int(decode_batch), batch) if decode_batch else batch
    _retrieval_request_batch = batch

    def _llm_tp_pp(engine: str, role: str) -> tuple[int, int]:
        if llm_parallelism and engine in llm_parallelism:
            roles = llm_parallelism[engine]
            if role in roles:
                return roles[role]
        if role == "prefill":
            return prefill_tp, prefill_pp
        return decode_tp, decode_pp

    def _prefill_fn(engine: str):
        return latency.prefill_by_engine.get(engine) or latency.prefill_s

    def _decode_fn(engine: str):
        return latency.decode_by_engine.get(engine) or latency.decode_s

    e2e: List[float] = []
    ttft: List[float] = []
    tpot_steps: List[float] = []
    # Per-RESOURCE service accumulators (one entry per query) — the SAME priced
    # ops as the critical-path sum, bucketed by the physical resource they run on,
    # so throughput can be bounded by the bottleneck resource (see ChainPerf).
    svc_gen: List[float] = []
    svc_prefill: List[float] = []
    svc_decode: List[float] = []
    svc_encode: List[float] = []
    svc_retr: List[float] = []
    svc_fusion: List[float] = []
    svc_rerank: List[float] = []
    svc_rerank_gpu: List[float] = []
    rerank_calls: List[int] = []
    gen_in_toks: List[float] = []
    gen_out_toks: List[float] = []
    svc_compress: List[float] = []
    svc_gen_by_stage: List[dict[str, float]] = []
    svc_prefill_by_stage: List[dict[str, float]] = []
    svc_decode_by_stage: List[dict[str, float]] = []
    retrieval_calls: List[int] = []
    retrieval_calls_before_ttft: List[int] = []

    # GPU light stages (encode / rerank / compress) are charged their REAL solo
    # cost — NOT a time-multiplexing "wait a decode burst" idleness. That tmux term
    # was a SYNC-era artifact: under lock-step react the per-round retrieval blocked
    # for ~a generator cycle (~0.74s). Under CONTINUOUS batching the retrieval runs
    # in the background while the generator keeps decoding OTHER queries, so the only
    # cost on the request's critical path is the retrieval's own latency (~0.15s/round
    # measured) — the encode forward + FAISS. Charging the tmux instead over-predicted
    # 7B/14B latency ~1.4× uniformly (the retrieval does NOT serialize behind a full
    # decode under continuous batching). ``gpu_collocated_with_gen`` is retained for
    # signature compatibility but no longer drives a contention wait.

    # Batched-worker padding semantics: the measured reranker worker tiles
    # the live queries' pairs into shared forwards, each forward padded to
    # its padding group's longest item — so a call's own token count does
    # not set its price.
    #
    # The padded forward shape is predicted from the call's quality-token
    # shape plus corpus/config padding excess (order statistics over the
    # declared padding group — whole call for colbert-type
    # tokenize-then-slice, one forward microbatch otherwise). The r14 tile walk
    # below survives ONLY as the legacy fallback for
    # archived ``_token_stats`` without the percentile grid: cut tiles of
    # one FORWARD microbatch (shared runtime policy — monot5 forwards 8
    # pairs, not the request batch), take each tile's max per-pair tokens
    # (cross-call heterogeneity only — it cannot see within-call spread),
    # and price every forward at the MEAN tile padding. The record's
    # max_seq_len clamp inside estimate() still caps the result at what
    # the production tokenizer feeds the model.
    _rr_pad_tokens = 0
    _rr_fanout_pad = max(1, int(getattr(latency, "rerank_fanout", 1) or 1))
    _rr_fwd_fn = getattr(latency, "rerank_forward_batch", None)
    _rr_excess_fn = getattr(latency, "rerank_pad_excess", None)
    _rr_dist_pad = bool(
        _rr_excess_fn is not None and _rr_excess_fn(batch) is not None
    )
    if not _rr_dist_pad:
        _tile_cap_pairs = max(1, _rr_fwd_fn(batch) if _rr_fwd_fn else batch)
        _tiles: List[int] = []
        _tile_max = 0
        _tile_pairs = 0
        for _q in traces:
            for _c in _q:
                if _trace_stage(_c) != "passage_reranker":
                    continue
                _per_pair = int(math.ceil(
                    float(_c.get("input_tokens", 0) or 0) / _rr_fanout_pad
                ))
                _tile_max = max(_tile_max, _per_pair)
                _tile_pairs += _rr_fanout_pad
                if _tile_pairs >= _tile_cap_pairs:
                    _tiles.append(_tile_max)
                    _tile_max = 0
                    _tile_pairs = 0
        if _tile_pairs > 0:
            _tiles.append(_tile_max)
        if _tiles:
            # mean tile padding, back in per-CALL units (rerank_s divides by
            # fanout again internally).
            _rr_pad_tokens = int(
                math.ceil(sum(_tiles) / len(_tiles)) * _rr_fanout_pad
            )

    # Completeness filter (r25b, autopsy lesion 2): the measured harness
    # records a trace row at ADMISSION, so rows admitted inside the window
    # but still queued at teardown carry only their upstream calls (eval_0038:
    # 1008/1296 rows stopped at retrieval behind a 274 s reranker backlog).
    # Averaging per-query service over those half-rows dilutes every
    # downstream stage's mean — 4.5x on the reranker — and the deeper the
    # backlog the stronger the dilution, which is exactly the heavy-reranker
    # regime. Measured qps counts COMPLETIONS, so the walk must too: a
    # completed row has reached the terminal LLM stage at least once.
    _completed = [q for q in traces
                  if any(_trace_stage(c) == "generator" for c in q)]
    if _completed:
        traces = _completed

    # Production retrieval latencies expose the worker's structural pieces.
    # Price one deployed full-batch visit with the same flatten/chunk helper as
    # the DES: K backend rows per user row and backend glue on every chunk.
    # The compiled RRF call is priced separately. Test doubles that expose only
    # ``retrieve_s`` retain their original one-call behavior.
    _structured_retrieval = (
        latency.retrieve_kernel_s is not None
        and (
            latency.retrieve_backend_overhead_s is not None
            or latency.retrieve_outer_overhead_s is not None
        )
    )
    _retrieval_plan = None
    if _structured_retrieval:
        _retrieval_plan = build_retrieval_worker_plan(
            (max(1, int(latency.retrieval_fanout)),) * batch,
            chunk_cap=batch,
            encode_s=latency.encode_s,
            vectorsearch_s=latency.retrieve_kernel_s,
            backend_glue_s=latency.retrieve_backend_overhead_s,
            outer_s=latency.retrieve_outer_overhead_s,
        )

    for q in traces:
        t = 0.0
        # Per-resource service accumulators for THIS query (bucketed view of t).
        _g = _enc = _ret_svc = _rr = _cc = _fusion = 0.0
        _rr_gpu = 0.0
        _rr_calls = 0
        _g_prefill = _g_decode_cap = 0.0
        _gen_in_tok = _gen_out_tok = 0.0
        # Decode-only portion of _g (the part that time-multiplexes the card with
        # light GPU stages under collocation → the contention base).
        _g_decode = 0.0
        _svc_gen_stage: dict[str, float] = {}
        _svc_prefill_stage: dict[str, float] = {}
        _svc_decode_stage: dict[str, float] = {}
        first_token_t: Optional[float] = None
        # Prefix-caching state: the previous generate's FULL sequence length
        # (input+output), which vLLM has cached. The next round (same generator
        # engine) prefills only the tokens beyond it. Reset per query; 0 = nothing
        # cached yet (the first generate pays the full prefill).
        _cached_prefix_by_engine: dict[str, int] = {}
        # Retrieval is ALWAYS explicit two-call form (encode + vector search) —
        # there is no folded single-call shape, so no fold accounting either.
        _retrieval_calls = sum(
            1 for c in q if _trace_stage(c) in _RETRIEVAL_TRACE_STAGES
        )
        # Index of the FINAL generate = the user-visible answer. In a SEQUENTIAL
        # pipeline any PRECEDING generate is auxiliary preprocessing (e.g.
        # query_expansion); empirically its decode OVERLAPS the request pipeline
        # (measured: a full pipeline's per-request e2e ≈ the base pipeline's, the
        # extra qexp generate adds ~0), so under ``overlap_aux_generates`` only the
        # final generate's decode is counted on the critical path. Iterative /
        # agentic rounds (react, ircot) are data-dependent and strictly serial →
        # overlap_aux_generates=False → every round's decode counts.
        _final_gen = -1
        _first_gen = -1
        for _i, _c in enumerate(q):
            if _trace_stage(_c) in _LLM_TRACE_STAGES:
                if _first_gen < 0:
                    _first_gen = _i
                _final_gen = _i
        _ttft_gen = _final_gen if overlap_aux_generates else _first_gen
        if _ttft_gen >= 0:
            _retrieval_calls_before_ttft = sum(
                1 for j, c in enumerate(q)
                if j < _ttft_gen and _trace_stage(c) in _RETRIEVAL_TRACE_STAGES
            )
        else:
            _retrieval_calls_before_ttft = _retrieval_calls
        for i, c in enumerate(q):
            stage = _trace_stage(c)
            if stage in _LLM_TRACE_STAGES:
                pin = int(c.get("input_tokens", 0))
                pout = int(c.get("output_tokens", 0))
                engine = stage  # the trace stage IS the LLM engine identity
                # (main-engine token accounting happens below, AFTER the
                # prefix-caching branch resolves the EFFECTIVE prefill
                # tokens — full pin over-counts APC-hit continuations 2-5x
                # and over-punished 0046/0042 in the token-shared cap.)
                prefill_stage, decode_stage = _llm_stage_names(engine)
                pref_fn = _prefill_fn(engine)
                dec_fn = _decode_fn(engine)
                pf_tp, pf_pp = _llm_tp_pp(engine, "prefill")
                dec_tp, dec_pp = _llm_tp_pp(engine, "decode")
                _cached_prefix = int(_cached_prefix_by_engine.get(engine, 0))
                if pref_fn is None:
                    _pf = 0.0
                elif (prefix_caching and not overlap_aux_generates
                      and _cached_prefix > 0 and pin > _cached_prefix):
                    # Prefix caching: round N re-prompts the previous round's
                    # sequence (cached) + new tokens → charge ONLY the incremental
                    # GenZ prefill. prefill is causal, so prefill(seq_N) -
                    # prefill(cached) = the new tokens' MLP + their attention over
                    # the full context (the exact incremental cost). Gated to
                    # agentic (overlap_aux=False) + a genuine continuation
                    # (pin > cached) so a sequential qexp→answer (different prompts,
                    # NOT a shared prefix) never wrongly subtracts.
                    _pf = (
                        pref_fn(pin, _pb, pf_tp, pf_pp)
                        - pref_fn(_cached_prefix, _pb, pf_tp, pf_pp)
                    )
                    _pf = max(0.0, _pf)
                else:
                    _pf = pref_fn(pin, _pb, pf_tp, pf_pp)
                t += _pf
                _g += _pf
                _g_prefill += _pf
                _svc_gen_stage[prefill_stage] = (
                    _svc_gen_stage.get(prefill_stage, 0.0) + _pf
                )
                _svc_prefill_stage[prefill_stage] = (
                    _svc_prefill_stage.get(prefill_stage, 0.0) + _pf
                )
                if engine == "generator":
                    # Effective NEW prompt tokens this call feeds the chunk
                    # budget (APC-consistent: continuations pay only the
                    # increment, same convention as _pf above).
                    _eff_in = (
                        pin - _cached_prefix
                        if (prefix_caching and not overlap_aux_generates
                            and _cached_prefix > 0 and pin > _cached_prefix)
                        else pin
                    )
                    _gen_in_tok += max(0, _eff_in)
                    _gen_out_tok += pout
                # This generate's full sequence (input+output) is now KV-cached for
                # the next round of the SAME generator engine.
                _cached_prefix_by_engine[engine] = pin + pout
                # TTFT = first token of the first USER-VISIBLE generation. For a
                # SEQUENTIAL pipeline that is the FINAL generate = the answer (any
                # preceding generate is internal preprocessing like query_expansion),
                # so TTFT includes all the pre-answer stages + the answer's prefill —
                # matching measured (pre_gen_time + answer-generator first token).
                # An agentic loop streams from round 1, so its TTFT is the FIRST
                # generate's prefill. The answer prefill is charged at the avg-queue
                # FRACTION (mean TTFT ≈ half-batch prefill); pre-answer stages unscaled.
                _is_ttft_gen = (i == _final_gen) if overlap_aux_generates else (first_token_t is None)
                if _is_ttft_gen:
                    first_token_t = t - _pf * (1.0 - _TTFT_PREFILL_FRAC)
                if dec_fn is not None and pout > 0:
                    # GenZ's decode ``output_tokens`` parameter is the generated-token
                    # position used to size the KV attention work for ONE decode step.
                    # A full response of ``pout`` tokens visits positions 1..pout, so
                    # price the burst at the mean position instead of charging the final
                    # (largest-KV) step ``pout`` times.
                    avg_decode_pos = max(1, int(math.ceil(float(pout) * 0.5)))
                    step = dec_fn(pin, avg_decode_pos, _db, dec_tp, dec_pp)
                    _burst = step * pout
                    t += _burst
                    _g += _burst
                    _g_decode += _burst
                    _svc_gen_stage[decode_stage] = (
                        _svc_gen_stage.get(decode_stage, 0.0) + _burst
                    )
                    _svc_decode_stage[decode_stage] = (
                        _svc_decode_stage.get(decode_stage, 0.0) + _burst
                    )
                    tpot_steps.append(step)
                    _g_decode_cap += _burst
            elif stage == "semantic_retrieval_encode":
                if _retrieval_plan is not None:
                    _e = _retrieval_plan.encode_s
                    t += _e
                    _enc += _e
                elif latency.encode_s is not None:
                    _e = latency.encode_s(_retrieval_request_batch)
                    t += _e
                    _enc += _e
            elif stage == "semantic_retrieval_vectorsearch":
                if _retrieval_plan is not None:
                    _ret = _retrieval_plan.retrieval_s
                else:
                    _ret = (
                        latency.retrieve_s(_retrieval_request_batch)
                        if latency.retrieve_s is not None else 0.0
                    )
                t += _ret
                _ret_svc += _ret
            elif stage == "retrieval_fusion":
                _fusion_call = (
                    latency.rrf_call_s(c, _retrieval_request_batch)
                    if latency.rrf_call_s is not None
                    else latency.rrf_s(_retrieval_request_batch)
                )
                t += _fusion_call
                _fusion += _fusion_call
                _ret_svc += _fusion_call
            elif stage == "lexical_retrieval":
                _ret = (
                    latency.bm25_retrieve_s(_retrieval_request_batch)
                    if latency.bm25_retrieve_s is not None else 0.0
                )
                t += _ret
                _ret_svc += _ret
            elif stage == "passage_reranker":
                if latency.rerank_s is not None:
                    _rk = latency.rerank_s(
                        _rr_pad_tokens or int(c.get("input_tokens", 0)),
                        _retrieval_request_batch,
                    )
                    t += _rk
                    _rr += _rk
                    _rk_glue = (
                        latency.rerank_glue_s(_retrieval_request_batch)
                        if latency.rerank_glue_s is not None else 0.0
                    )
                    _rr_gpu += max(0.0, _rk - _rk_glue)
                    _rr_calls += 1
            elif stage == "passage_compressor":
                if latency.compress_s is not None:
                    _cp = latency.compress_s(
                        int(c.get("input_tokens", 0)),
                        _retrieval_request_batch,
                    )
                    t += _cp
                    _cc += _cp
        # GPU time-multiplex contention (open-saturation, data-flow agnostic). When the
        # generator's DECODE shares its card(s) with light GPU stages (encode / rerank /
        # compress) — a COLLOCATED placement, gated by gpu_collocated_with_gen which is
        # FALSE for PD-split disagg (dedicated decode engine) — the decode time-shares
        # the card with those stages' work for the OTHER in-flight requests under
        # saturation, so it runs slower. Modeled as the decode burst stretched by the
        # light stages' relative GPU load (svc_light / svc_gen) — a pure time-multiplex
        # fraction, no fitted coefficient. This is exactly why measured collocated tp4
        # (decode⊕encode on cuda:0) is SLOWER than disagg (decode on its own cards):
        # the CM previously gave the opposite order because it never charged this.
        if gpu_collocated_with_gen and _g > 0 and _g_decode > 0:
            # _rr_gpu, not _rr: the reranker's host-side worker glue does not
            # occupy the card, so it must not stretch the decode.
            _cont = _g_decode * (_enc + _rr_gpu + _cc) / _g
            t += _cont
            _g += _cont
            _g_decode_cap += _cont
        e2e.append(t)
        ttft.append(first_token_t if first_token_t is not None else t)
        svc_gen.append(_g)
        svc_prefill.append(_g_prefill)
        svc_decode.append(_g_decode_cap)
        svc_encode.append(_enc)
        svc_retr.append(_ret_svc)
        svc_fusion.append(_fusion)
        svc_rerank.append(_rr)
        svc_rerank_gpu.append(_rr_gpu)
        rerank_calls.append(_rr_calls)
        gen_in_toks.append(_gen_in_tok)
        gen_out_toks.append(_gen_out_tok)
        svc_compress.append(_cc)
        svc_gen_by_stage.append(dict(_svc_gen_stage))
        svc_prefill_by_stage.append(dict(_svc_prefill_stage))
        svc_decode_by_stage.append(dict(_svc_decode_stage))
        retrieval_calls.append(_retrieval_calls)
        retrieval_calls_before_ttft.append(_retrieval_calls_before_ttft)
    # e2e exposes BOTH reductions. The benchmark comparison targets
    # ``latency_s['mean']`` (see run_accuracy), while some legacy callers still
    # inspect the median-like ``e2e`` field. TTFT stays a mean: per-request TTFT is
    # queue-noise dominated, so the stable target is the average queue wait, not
    # the front-of-queue median. TPOT stays a mean (a per-step rate).
    #
    # Throughput is a work-conservation quantity: a saturated resource's qps is
    # serving_width / E[service demand per query]. Using the median here made
    # long-tailed output distributions look too cheap when median < mean, and
    # capped long-answer distributions look too expensive when median > mean.
    _med = lambda xs: float(np.median(xs)) if xs else 0.0
    _mean = lambda xs: float(np.mean(xs)) if xs else 0.0

    def _mean_dict(dicts: List[dict[str, float]]) -> dict[str, float]:
        keys = sorted({k for d in dicts for k in d})
        return {
            k: float(np.mean([float(d.get(k, 0.0)) for d in dicts]))
            for k in keys
        }

    return ChainPerf(
        e2e=_med(e2e),
        e2e_mean=_mean(e2e),
        ttft=float(np.mean(ttft)) if ttft else 0.0,
        tpot=float(np.mean(tpot_steps)) if tpot_steps else 0.0,
        # Per-resource service = mean per-query work. Throughput later divides
        # each component by its own serving width (prefill request batch, decode
        # batch, request batch).
        svc_gen=_mean(svc_gen),
        svc_prefill=_mean(svc_prefill),
        svc_decode=_mean(svc_decode),
        svc_encode=_mean(svc_encode),
        svc_retrieval=_mean(svc_retr),
        svc_fusion=_mean(svc_fusion),
        svc_rerank=_mean(svc_rerank),
        svc_rerank_gpu=_mean(svc_rerank_gpu),
        rerank_calls_per_query=_mean(rerank_calls),
        gen_in_tokens_per_query=_mean(gen_in_toks),
        gen_out_tokens_per_query=_mean(gen_out_toks),
        svc_compress=_mean(svc_compress),
        svc_gen_by_stage=_mean_dict(svc_gen_by_stage),
        svc_prefill_by_stage=_mean_dict(svc_prefill_by_stage),
        svc_decode_by_stage=_mean_dict(svc_decode_by_stage),
        decode_batch=int(_db),
        retrieval_calls_per_query=_mean(retrieval_calls),
        retrieval_calls_before_ttft=_mean(retrieval_calls_before_ttft),
    )


# ---------------------------------------------------------------------------
# Engines — per-config latency + tp (dispatched by RAGCMAssembly)
# ---------------------------------------------------------------------------
class StaticEngine:
    """Config-driven, SEQUENTIAL-only. Identity: RAGO's single-pass latency_s +
    qps are final (the static path only applies comm + selection on RAGO values).
    Asserts sequential — no non-trace iterative config reaches the config path
    (static_gt is sequential-only; agentic always emits a trace → DynamicEngine)."""

    def __init__(self, schema: WorkflowSchema):
        self.schema = schema

    def cost(self, df, sweep_dfs=None, *, pipeline_config=None, hardware=None,
             traces=None, fixed_placement=None):
        assert getattr(self.schema, "dataflow_type", "sequential") == "sequential", (
            "StaticEngine handles only sequential config-driven pipelines "
            "(agentic/iterative go through DynamicEngine via a trace)")
        return df


class DynamicEngine:
    """Trace-aware analytical scoring of concrete deployment candidates.

    ``analyze`` never invokes the simulator. Its scores support shortlisting;
    ``dynamic_pipeline`` obtains final performance from closed-loop simulation.
    """

    def __init__(self, schema: WorkflowSchema):
        self.schema = schema
        self.is_agentic = bool(getattr(schema, "is_agentic", False))
        # Sequential overlaps the aux-generate (query_expansion) decode with the
        # request pipeline; agentic puts every round's decode on the critical path.
        self.overlap_aux = not self.is_agentic

    def collocates_groups(self, groups) -> bool:
        """Whether the generator's DECODE time-multiplexes its card(s) with light GPU
        stages (encode / rerank / compress) → contention. True when the generator
        collocates with a light stage AND is NOT PD-split (disagg's dedicated decode
        engine sits on its own cards, free of the light stages).

        DATA-FLOW AGNOSTIC: applies to BOTH sequential and agentic. Under the
        open-system SATURATION model the in-flight requests' stages overlap — the
        encode of request N+1 runs on the shared card while request N decodes — so a
        collocated sequential pipeline DOES contend, not just an agentic loop. (The
        old `if self.is_agentic else False` was the single-request view: 'sequential
        is retrieval-then-generation, no interleave' — wrong under saturation.)"""
        from rag_stack.cost_model.communication import pd_split
        return _groups_collocate(groups) and not pd_split(groups)

    def gpu_collocated_with_gen(self, fixed_placement) -> bool:
        """Collocation for a single pinned placement (benchmark/replay path)."""
        if fixed_placement is None:
            return False
        return self.collocates_groups(
            getattr(fixed_placement, "collocation_strategy", None))

    def cost(self, df, sweep_dfs, *, pipeline_config, hardware, traces,
             trace_source_question_ids=None, fixed_placement=None,
             prepared_paths: PreparedExecutionPaths | None = None):
        """Compatibility entry for the complete two-stage evaluation pipeline."""
        from rag_stack.cost_model.dynamic_pipeline import evaluate_candidates

        return evaluate_candidates(
            engine=self,
            df=df,
            sweep_dfs=sweep_dfs,
            pipeline_config=pipeline_config,
            hardware=hardware,
            traces=traces,
            trace_source_question_ids=trace_source_question_ids,
            fixed_placement=fixed_placement,
            **({"prepared_paths": prepared_paths} if prepared_paths is not None else {}),
        )

    def analyze(self, df, sweep_dfs, *, pipeline_config, hardware, traces,
                trace_source_question_ids=None, fixed_placement=None, compiled_plan=None,
                prepared_paths: PreparedExecutionPaths | None = None):
        """Score candidates from per-query calls, resource demands, and capacity.

        Candidate layout, batch axes, and communication estimates are inputs.
        Incoming coarse throughput/latency are discarded: this method computes
        every analytical score independently and never invokes DES. Per-stage
        sweep tables provide TP/PP only for legacy rows without exact layouts.
        The explicit ``dynamic_analytical_*`` columns preserve shortlist evidence
        when simulation later replaces the public performance columns.
        """
        if df is None or len(df) == 0:
            return df
        if "batch_size_request" not in df.columns:
            raise ValueError("Analytical candidates require batch_size_request")
        df = df.drop(
            columns=[
                "qps", "qps_per_chip", "latency_s", "latency_s_ttft",
                "latency_s_tpot",
            ],
            errors="ignore",
        ).copy()
        # Component service must be resolved against the same concrete device
        # layout which DES will consume.  A single closure built from the base
        # pipeline is not topology-safe: TP/PP may be row-local, and a
        # heterogeneous system can give the same TP/PP a different exact rank
        # order.  Cache one immutable CallLatency per complete layout identity;
        # timeout siblings and batch siblings then reuse it without falling
        # back to a model- or topology-specific calibration coefficient.
        from rag_stack.cost_model.candidate_layout import (
            CANDIDATE_LAYOUT_COLUMN,
            candidate_system_layout_from_row,
        )
        from rag_stack.cost_model.closed_loop_sim import (
            _pipeline_with_frozen_stage_batch_selection,
            _stage_batch_generation_for_pipeline,
            _stable_signature_value,
        )
        from rag_stack.cost_model.stage_batch_curve import (
            EngineCycleCurveApplicator,
        )
        from rag_stack.rag_ir.normalize import with_observed_trace_components

        if compiled_plan is not None:
            if len(df) != 1:
                raise ValueError("A compiled fixed plan analyzes exactly one deployment")
            compiled_plan.validate_candidate(df.iloc[0])
            if prepared_paths is not None:
                resolve_execution_paths(traces, prepared_paths)
            traces = compiled_plan.analysis_paths()
            # Analysis must project the bound physical graph, not substitute
            # an upstream trace merely because it was already validated.
            prepared_paths = resolve_execution_paths(traces)
            pipeline_config = compiled_plan.pipeline_config
            hardware = compiled_plan.hardware
        else:
            prepared_paths = resolve_execution_paths(traces, prepared_paths)
        traces = prepared_paths.paths
        pipeline_config = with_observed_trace_components(pipeline_config, traces)

        pipeline_config = _pipeline_with_frozen_stage_batch_selection(
            pipeline_config,
            hardware,
        )
        _stage_batch_generation = _stage_batch_generation_for_pipeline(
            pipeline_config,
            hardware,
        )
        _engine_cycle_curves = EngineCycleCurveApplicator(
            _stage_batch_generation
        )

        _latency_by_layout: dict[Any, CallLatency] = {}

        def _candidate_layout(row) -> dict | None:
            if row is None or CANDIDATE_LAYOUT_COLUMN not in row.index:
                return None
            raw = row.get(CANDIDATE_LAYOUT_COLUMN)
            if raw is None or (
                isinstance(raw, (float, np.floating)) and pd.isna(raw)
            ):
                return None
            return candidate_system_layout_from_row(row)

        def _latency_for_row(row) -> tuple[CallLatency, Any]:
            layout = _candidate_layout(row)
            key = _stable_signature_value(layout)
            if compiled_plan is not None:
                return compiled_plan.latency, key
            cached = _latency_by_layout.get(key)
            if cached is not None:
                return cached, key
            candidate_pipeline = pipeline_config
            if layout is not None:
                candidate_pipeline = deepcopy(pipeline_config)
                candidate_pipeline.setdefault("system", {})["layout"] = deepcopy(
                    layout
                )
            resolved = build_call_latency(candidate_pipeline, hardware)
            _latency_by_layout[key] = resolved
            return resolved, key

        def _winning_tp_pp(
            stage: str,
            num_chips: int,
            row=None,
        ) -> tuple[int, int]:
            """Return the TP/PP selected for this *concrete candidate*.

            The per-stage sweep table is only a legacy fallback: one chip count
            can have several physical rank orders and can be repriced to a
            different TP/PP on each topology.  ``candidate_system_layout`` is
            therefore authoritative whenever a candidate row carries it.
            """
            if row is not None:
                from rag_stack.cost_model.candidate_layout import (
                    CANDIDATE_LAYOUT_COLUMN,
                    candidate_system_layout_from_row,
                )

                raw_layout = row.get(CANDIDATE_LAYOUT_COLUMN)
                has_layout = raw_layout is not None and not (
                    isinstance(raw_layout, (float, np.floating))
                    and pd.isna(raw_layout)
                )
                if has_layout:
                    layout = candidate_system_layout_from_row(row)
                    info = (layout.get("performance_stages") or {}).get(stage)
                    if not isinstance(info, dict):
                        raise ValueError(
                            f"Concrete candidate layout is missing LLM stage {stage!r}"
                        )
                    if info.get("tp") is None or info.get("pp") is None:
                        raise ValueError(
                            f"Concrete candidate layout stage {stage!r} has no TP/PP"
                        )
                    tp, pp = max(1, int(info["tp"])), max(1, int(info["pp"]))
                    devices = list(info.get("devices") or [])
                    concrete_chips = len(devices) if devices else int(num_chips)
                    if devices and concrete_chips != int(num_chips):
                        raise ValueError(
                            f"Concrete candidate stage {stage!r} has "
                            f"devices={concrete_chips}, row num_chips={num_chips}"
                        )
                    if tp * pp > concrete_chips or concrete_chips % (tp * pp):
                        raise ValueError(
                            f"Concrete candidate stage {stage!r} TP*PP={tp}*{pp} "
                            f"does not divide devices={concrete_chips}"
                        )
                    return tp, pp

                # Transitional row contract for callers which retained only the
                # flattened candidate-layout columns. Do not consult the
                # homogeneous sweep winner when this candidate-specific value
                # is available.
                flat_tp = row.get(f"candidate_tp_{stage}")
                flat_pp = row.get(f"candidate_pp_{stage}")
                if pd.notna(flat_tp) and pd.notna(flat_pp):
                    tp, pp = max(1, int(flat_tp)), max(1, int(flat_pp))
                    if tp * pp > int(num_chips) or int(num_chips) % (tp * pp):
                        raise ValueError(
                            f"Candidate stage {stage!r} TP*PP={tp}*{pp} does not "
                            f"divide num_chips={num_chips}"
                        )
                    return tp, pp

            fixed_tp_pp = _fixed_llm_tp_pp_from_layout(pipeline_config, stage)
            if fixed_tp_pp is not None:
                tp, pp = fixed_tp_pp
                return max(1, int(tp)), max(1, int(pp))
            sdf = sweep_dfs.get(stage)
            cols = getattr(sdf, "columns", [])
            if sdf is None or "winning_tp" not in cols:
                return (1, 1)
            nc_col = ("hardware.num_chips" if "hardware.num_chips" in cols
                      else "num_chips" if "num_chips" in cols else None)
            sub = sdf[sdf[nc_col] == int(num_chips)] if nc_col else sdf
            if not len(sub):
                return (1, 1)
            row = (sub.loc[sub["batch_size"].idxmax()]
                   if "batch_size" in sub.columns else sub.iloc[0])
            return (max(1, int(row["winning_tp"])), max(1, int(row["winning_pp"])))

        # Dataflow policy comes from the engine itself (self): overlap_aux
        # (sequential overlaps the aux-generate decode; agentic doesn't) +
        # the GPU-collocation / PD-split gating below. Per-query walks supply
        # both response-time estimates and resource demand for capacity analysis.
        _overlap_aux = self.overlap_aux

        from rag_stack.cost_model.communication import pd_split

        _sys_cfg = (pipeline_config.get("system") or {})

        # batch_size_decode (system.batch_size_decode): the decode-engine concurrency
        # = deployed max_num_seqs, INDEPENDENT of the request/prefill batch. Absent ⇒
        # = batch_size_request (the closed-loop default). When set (≥ request) the
        # decode steps are priced at it and it is the decode throughput batch axis.
        # Constraint request ≤ decode.
        _cfg_decode = _sys_cfg.get("batch_size_decode")
        _cfg_decode = int(_cfg_decode) if _cfg_decode else None

        def _decode_batch_of(batch: int) -> int:
            return max(_cfg_decode, batch) if _cfg_decode else batch

        # ── CM system design space: deployment axes (DEFAULT-ON) ────────────
        # See _expand_deployment_axes — deployed decode concurrency plus the
        # dynamic-batching timeout axis, when a legacy row omitted either.
        _css = _sys_cfg.get("cm_search_space") or {}
        _batching_cfg = _sys_cfg.get("batching") if isinstance(_sys_cfg, dict) else {}
        _dynamic_timeout_s = (
            _batching_cfg.get("dynamic_timeout_s")
            if isinstance(_batching_cfg, dict) else None
        )
        if fixed_placement is None and compiled_plan is None:
            df = _expand_deployment_axes(
                df, _css,
                agentic=bool(self.is_agentic),
                decode_pinned=_cfg_decode is not None,
                timeout_pinned=_dynamic_timeout_s is not None,
            )

        # GPU time-multiplexing (agentic only): a GPU light stage (encode / rerank /
        # compress) collocated with the generator can't run concurrently — see
        # _compute_chain_latency. Two placement sources:
        #   * PINNED (benchmark / replay, fixed_placement set): the pinned measured
        #     layout is authoritative — use it for EVERY row. (RAGO's assemble_cost
        #     re-normalizes the df's collocation_strategy column — e.g. routing
        #     "encode" onto retrieval's CPU servers — so the column would
        #     under-detect the measured GPU collocation; never use it here.)
        #   * SWEPT (optimizer, fixed_placement is None): each row is a DIFFERENT
        #     placement, so read collocation PER ROW from its collocation_strategy.
        _per_row_colloc = fixed_placement is None and "collocation_strategy" in df.columns
        _gpu_colloc_fixed = self.gpu_collocated_with_gen(fixed_placement)

        _walk_memo: dict = {}

        def _stage_chips(row, stage: str) -> int:
            col = f"num_chips_{stage}"
            if col in row.index and pd.notna(row[col]):
                return max(1, int(row[col]))
            return 1

        def _llm_parallelism_for_row(row) -> dict[str, dict[str, tuple[int, int]]]:
            # Reuse the exact candidate's cached service functions for stage
            # presence as well as pricing; no base-layout closure is needed.
            row_latency, _ = _latency_for_row(row)
            has_query_expansion = bool(
                "query_expansion" in row_latency.prefill_by_engine
                or "query_expansion" in row_latency.decode_by_engine
            )
            out: dict[str, dict[str, tuple[int, int]]] = {}
            for engine in ("generator", "query_expansion"):
                if engine == "query_expansion" and not has_query_expansion:
                    continue
                pre_stage, dec_stage = _llm_stage_names(engine)
                if (
                    f"num_chips_{pre_stage}" not in row.index
                    and f"num_chips_{dec_stage}" not in row.index
                    and engine not in row_latency.prefill_by_engine
                    and engine not in row_latency.decode_by_engine
                ):
                    continue
                out[engine] = {
                    "prefill": _winning_tp_pp(
                        pre_stage, _stage_chips(row, pre_stage), row,
                    ),
                    "decode": _winning_tp_pp(
                        dec_stage, _stage_chips(row, dec_stage), row,
                    ),
                }
            return out

        def _parallelism_key(parallelism: dict[str, dict[str, tuple[int, int]]]) -> tuple:
            return tuple(
                (engine, role, int(tp), int(pp))
                for engine, roles in sorted(parallelism.items())
                for role, (tp, pp) in sorted(roles.items())
            )

        def _walk(row, batch: int, decode_b: int, colloc: bool):
            llm_parallelism = _llm_parallelism_for_row(row)
            row_latency, layout_key = _latency_for_row(row)
            key = (
                layout_key,
                batch,
                decode_b,
                colloc,
                _parallelism_key(llm_parallelism),
            )
            if key not in _walk_memo:
                _walk_memo[key] = _compute_chain_latency(
                    traces, row_latency, batch=batch,
                    overlap_aux_generates=_overlap_aux,
                    gpu_collocated_with_gen=colloc,
                    prefix_caching=_PREFIX_CACHING,
                    # decode_batch: decode-engine concurrency (= max_num_seqs),
                    # independent of the request batch; default = request.
                    decode_batch=decode_b,
                    llm_parallelism=llm_parallelism,
                    prepared_paths=prepared_paths,
                )
            return _walk_memo[key]

        def _decode_replicas(row) -> int:
            """Data-parallel decode replicas = decode chips / (TP×PP). Each replica
            holds a full batch in flight, so the deployment's concurrency is
            replicas × batch → qps scales ×replicas (DP raises throughput, not
            latency). DP=1 for all pinned-benchmark cases."""
            n_dec = _stage_chips(row, "generator_decode")
            dec_tp, dec_pp = _winning_tp_pp("generator_decode", n_dec, row)
            return max(1, n_dec // max(1, dec_tp * dec_pp))

        def _prefill_replicas(row) -> int:
            """Data-parallel prefill replicas, analogous to decode replicas."""
            n_pf = _stage_chips(row, "generator_prefill")
            pf_tp, pf_pp = _winning_tp_pp("generator_prefill", n_pf, row)
            return max(1, n_pf // max(1, pf_tp * pf_pp))

        # Compute the actual per-query chain for every candidate, then add the
        # separately modeled communication term. Aggregate/RAGO latency is not
        # an input to this calculation.
        has_comm = "comm_latency_s" in df.columns
        walks = []
        for i in range(len(df)):
            base = df.iloc[i]
            batch = int(base["batch_size_request"]) if "batch_size_request" in df.columns else 1
            colloc = (self.collocates_groups(base["collocation_strategy"])
                      if _per_row_colloc else _gpu_colloc_fixed)
            dec_b = (
                int(base["batch_size_decode"])
                if "batch_size_decode" in df.columns
                and pd.notna(base["batch_size_decode"])
                else _decode_batch_of(batch)
            )
            walks.append(_walk(base, batch, dec_b, colloc))
        _comm = [
            float(df["comm_latency_s"].iloc[i]) if has_comm else 0.0
            for i in range(len(walks))
        ]
        df["latency_s_trace_median_raw"] = [
            w.e2e + _comm[i] for i, w in enumerate(walks)
        ]
        df["latency_s_trace_mean_raw"] = [
            w.e2e_mean + _comm[i] for i, w in enumerate(walks)
        ]
        df["latency_s_trace_median"] = df["latency_s_trace_median_raw"]
        df["latency_s_trace_mean"] = df["latency_s_trace_mean_raw"]
        df["latency_s"] = df["latency_s_trace_mean"]
        df["latency_s_ttft"] = [w.ttft for w in walks]
        df["latency_s_tpot"] = [w.tpot for w in walks]
        # Per-resource service times (mean per-query demand on each resource) — the
        # bottleneck-throughput inputs + per-stage accuracy diagnostics. Generator
        # service is split into prefill/decode/overhead because continuous batching
        # gives those subresources different serving widths.
        df["svc_gen_s"] = [w.svc_gen for w in walks]
        df["svc_prefill_s"] = [w.svc_prefill for w in walks]
        df["svc_decode_s"] = [w.svc_decode for w in walks]
        df["svc_encode_s"] = [w.svc_encode for w in walks]
        df["svc_retrieval_s"] = [w.svc_retrieval for w in walks]
        df["svc_fusion_s"] = [w.svc_fusion for w in walks]
        df["svc_rerank_s"] = [w.svc_rerank for w in walks]
        df["svc_rerank_gpu_s"] = [w.svc_rerank_gpu for w in walks]
        df["rerank_calls_per_query"] = [w.rerank_calls_per_query for w in walks]
        df["gen_in_tokens_per_query"] = [w.gen_in_tokens_per_query for w in walks]
        df["gen_out_tokens_per_query"] = [w.gen_out_tokens_per_query for w in walks]
        df["svc_compress_s"] = [w.svc_compress for w in walks]
        df["svc_gen_by_stage_s"] = [w.svc_gen_by_stage for w in walks]
        df["svc_prefill_by_stage_s"] = [w.svc_prefill_by_stage for w in walks]
        df["svc_decode_by_stage_s"] = [w.svc_decode_by_stage for w in walks]
        df["batch_size_decode"] = [w.decode_batch for w in walks]
        # Winning TP/PP of the priced deployment, stamped per row — the chips
        # count alone is ambiguous (4 chips could be tp4, tp2×pp2, …), and the
        # SELECTED row's parallelism must survive into best_hw_params so the
        # chosen system design is fully recorded per eval.
        _tp_pp_rows: list[dict] = []
        for i in range(len(df)):
            _par = _llm_parallelism_for_row(df.iloc[i])
            _flat: dict = {}
            for _engine, _roles in _par.items():
                _pre_stage, _dec_stage = _llm_stage_names(_engine)
                for _role, (_tp, _pp) in _roles.items():
                    _stage = _pre_stage if _role == "prefill" else _dec_stage
                    _flat[f"tp_{_stage}"] = int(_tp)
                    _flat[f"pp_{_stage}"] = int(_pp)
            _tp_pp_rows.append(_flat)
        for _col in sorted({c for r in _tp_pp_rows for c in r}):
            df[_col] = [r.get(_col) for r in _tp_pp_rows]
        if _dynamic_timeout_s is not None:
            # pinned scalar (system.batching.dynamic_timeout_s) — stamped for
            # the result record; the swept axis was expanded above instead.
            df["dynamic_batch_timeout_s"] = float(_dynamic_timeout_s)
        df["retrieval_calls_per_query"] = [
            w.retrieval_calls_per_query for w in walks
        ]
        df["retrieval_calls_before_ttft"] = [
            w.retrieval_calls_before_ttft for w in walks
        ]
        # All candidates use the same resource-capacity model. Independent
        # resources overlap across in-flight requests; shared stages contribute
        # to one resource's demand. Prefill/retrieval use request batch, decode
        # uses decode depth. Communication latency contributes to response time,
        # while comm_qps_bound separately constrains throughput. None of these
        # analytical estimates inherits RAGO's coarse performance score.
        _comm_bound_series = (
            df["comm_qps_bound"]
            if "comm_qps_bound" in df.columns
            else pd.Series(float("inf"), index=df.index)
        )
        _pf_reps = [
            _prefill_replicas(df.iloc[i])
            if "num_chips_generator_prefill" in df.columns else 1
            for i in range(len(df))
        ]
        _reps = [
            _decode_replicas(df.iloc[i])
            if "num_chips_generator_decode" in df.columns else 1
            for i in range(len(df))
        ]
        _retr_servers = [
            max(1, int(df["num_retrieval_servers"].iloc[i]))
            if "num_retrieval_servers" in df.columns else 1
            for i in range(len(df))
        ]
        _encode_reps = [
            max(1, int(df["num_chips_semantic_retrieval_encode"].iloc[i]))
            if "num_chips_semantic_retrieval_encode" in df.columns else 1
            for i in range(len(df))
        ]
        qps_prefill = []
        qps_decode = []
        qps_gen = []
        qps_encode = []
        qps_retr = []
        qps_fusion = []
        qps_rerank = []
        qps_compress = []
        qps_comm = []
        qps_gpu_resource = []
        qps_llm_engine_curve = []
        qps_overall = []
        eq_batch_retr = []
        eq_batch_rerank = []
        eq_batch_compress = []
        curve_extra_latency_s = []
        trace_mean_with_curve_s = []
        trace_median_with_curve_s = []
        curve_cycle_counts = []

        def _finite_or_zero(v: float) -> float:
            return float(v) if np.isfinite(v) else 0.0

        def _min_finite(vals) -> float:
            finite = [float(v) for v in vals if np.isfinite(v)]
            return min(finite) if finite else float("inf")

        def _as_groups(value) -> list[list[str]]:
            if isinstance(value, str):
                try:
                    value = ast.literal_eval(value)
                except Exception:  # noqa: BLE001
                    return []
            if not isinstance(value, (list, tuple)):
                return []
            groups = []
            for group in value:
                if isinstance(group, str):
                    groups.append([group])
                elif isinstance(group, (list, tuple, set)):
                    groups.append([str(s) for s in group])
            return groups

        def _dict_value(row, col: str) -> dict[str, float]:
            value = row.get(col, {})
            return value if isinstance(value, dict) else {}

        def _stage_replicas(row, stage: str) -> int:
            col = f"num_chips_{stage}"
            chips = max(1, int(row[col])) if col in row.index and pd.notna(row[col]) else 1
            if stage in {
                "generator_prefill",
                "generator_decode",
                "query_expansion_prefill",
                "query_expansion_decode",
            }:
                tp, pp = _winning_tp_pp(stage, chips, row)
                return max(1, chips // max(1, int(tp) * int(pp)))
            return chips

        _vllm_engines_cfg = (
            ((_sys_cfg.get("vllm") or {}).get("engines") or {})
            if isinstance(_sys_cfg, dict) else {}
        )

        def _engine_depth(
            engine: str,
            role: str,
            request_batch: int,
            decode_batch: int,
        ) -> int:
            cfg = _vllm_engines_cfg.get(engine) or {}
            if role == "prefill":
                default = request_batch
                value = cfg.get("prefill_max_num_seqs")
            elif role == "decode":
                default = (
                    decode_batch if engine == "generator"
                    else request_batch
                )
                value = cfg.get("decode_max_num_seqs")
            else:
                default = (
                    decode_batch if engine == "generator"
                    else request_batch
                )
                value = cfg.get("max_num_seqs")
            return max(1, int(value or default))

        def _engine_pd_mode(row, engine: str) -> str:
            layout = _candidate_layout(row)
            if layout is None:
                raw = (_sys_cfg.get("layout") or {}) if isinstance(
                    _sys_cfg, dict,
                ) else {}
                layout = raw if isinstance(raw, dict) else {}
            engines = layout.get("engines") or {}
            record = engines.get(engine) if isinstance(engines, dict) else None
            if isinstance(record, dict):
                mode = str(record.get("pd_serving") or "collocated_pd")
                if mode in {"collocated_pd", "disagg_pd"}:
                    return mode
            # Legacy/fixed rows without an engines map retain their
            # explicit placement split. Production RAGO rows always take
            # the concrete-layout branch above.
            groups = row.get("collocation_strategy", [])
            return "disagg_pd" if pd_split(groups) else "collocated_pd"

        _curve_trace = [
            query for query in traces
            if any(
                _trace_stage(call) == "generator"
                for call in (
                    (query.get("calls") if isinstance(query, dict) else query)
                    or []
                )
            )
        ] or list(traces)
        _curve_cycle_cache: dict[tuple[int, int, int, int], dict] = {}

        def _mean_engine_cycles(
            request_batch: int,
            generator_decode_batch: int,
            generator_prefill_depth: int,
            generator_decode_depth: int,
        ) -> dict[str, dict[str, float]]:
            key = (
                int(request_batch),
                int(generator_decode_batch),
                int(generator_prefill_depth),
                int(generator_decode_depth),
            )
            if key in _curve_cycle_cache:
                return _curve_cycle_cache[key]
            per_query: list[dict[str, dict[str, float]]] = []
            for query in _curve_trace:
                calls = (
                    query.get("calls") if isinstance(query, dict) else query
                ) or []
                cached_prefix: dict[str, int] = {}
                counts: dict[str, dict[str, float]] = {}
                for call in calls:
                    engine_name = _trace_stage(call)
                    if engine_name not in {"generator", "query_expansion"}:
                        continue
                    pin = max(0, int(call.get("input_tokens", 0) or 0))
                    pout = max(0, int(call.get("output_tokens", 0) or 0))
                    previous = int(cached_prefix.get(engine_name, 0))
                    effective_input = (
                        pin - previous
                        if (
                            _PREFIX_CACHING
                            and not _overlap_aux
                            and previous > 0
                            and pin > previous
                        ) else pin
                    )
                    effective_input = max(0, effective_input)
                    prefill_depth = (
                        generator_prefill_depth
                        if engine_name == "generator"
                        else _engine_depth(
                            engine_name,
                            "prefill",
                            request_batch,
                            request_batch,
                        )
                    )
                    prefill_cycles = (
                        int(math.ceil(
                            prefill_depth * effective_input
                            / float(_PREFILL_CHUNK_TOKENS)
                        ))
                        if effective_input > 0 else 0
                    )
                    decode_cycles = pout
                    record = counts.setdefault(engine_name, {
                        "prefill": 0.0,
                        "prefill_frontiers": 0.0,
                        "decode": 0.0,
                        "unified": 0.0,
                    })
                    record["prefill"] += float(prefill_cycles)
                    # Direct-P calibration times the completion frontier
                    # of one logical prefill cohort, not every production
                    # 8192-token scheduler chunk.  APC may reduce the
                    # component chunks, but each logical LLM call still
                    # crosses this terminal P frontier exactly once.
                    record["prefill_frontiers"] += 1.0
                    record["decode"] += float(decode_cycles)
                    # A collocated scheduler cycle composes P+D and adds
                    # its curve once. Prompt chunks ride decode cycles;
                    # only the longer lane extends the cycle count.
                    record["unified"] += float(max(
                        prefill_cycles,
                        decode_cycles,
                    ))
                    cached_prefix[engine_name] = pin + pout
                per_query.append(counts)

            engines_seen = sorted({
                engine_name
                for counts in per_query
                for engine_name in counts
            })
            means = {
                engine_name: {
                    role: float(np.mean([
                        counts.get(engine_name, {}).get(role, 0.0)
                        for counts in per_query
                    ])) if per_query else 0.0
                    for role in (
                        "prefill",
                        "prefill_frontiers",
                        "decode",
                        "unified",
                    )
                }
                for engine_name in engines_seen
            }
            _curve_cycle_cache[key] = means
            return means

        def _curve_overhead_s(
            engine: str,
            role: str,
            resident_batch: int,
        ) -> float:
            prefill_stage, decode_stage = _llm_stage_names(engine)
            corrected = _engine_cycle_curves.corrected_service_s(
                engine_stage=engine,
                role=role,
                prefill_stage=prefill_stage,
                decode_stage=decode_stage,
                resident_batch=int(resident_batch),
                component_service_s=1.0,
            )
            overhead = float(corrected) - 1.0
            if not np.isfinite(overhead) or overhead < 0.0:
                raise ValueError(
                    "stage-batch curve produced invalid analytical "
                    f"overhead for {engine}/{role}: {overhead!r}"
                )
            return overhead

        def _gpu_resource_caps(row, batch: int, db: int) -> list[float]:
            pref = _dict_value(row, "svc_prefill_by_stage_s")
            dec = _dict_value(row, "svc_decode_by_stage_s")
            enc_s = float(row.get("svc_encode_s", 0.0) or 0.0)
            # Card-side rerank work comes directly from the component
            # forward model. No whole-stage worker residual is added.
            rr_s = float(
                row.get("svc_rerank_gpu_s", row.get("svc_rerank_s", 0.0))
                or 0.0
            )
            cc_s = float(row.get("svc_compress_s", 0.0) or 0.0)
            groups = _as_groups(row.get("collocation_strategy", []))
            caps: list[float] = []
            for group in groups:
                if group == ["semantic_retrieval_vectorsearch"]:
                    continue
                demand = 0.0
                for stage in group:
                    reps = _stage_replicas(row, stage)
                    if stage in pref:
                        demand += float(pref[stage]) / max(1, reps * batch)
                    if stage in dec:
                        demand += float(dec[stage]) / max(1, reps * db)
                    if stage == "semantic_retrieval_encode" and enc_s > 0:
                        demand += enc_s / max(1, reps * batch)
                    elif stage == "passage_reranker" and rr_s > 0:
                        demand += rr_s / max(1, reps * batch)
                    elif stage.startswith("passage_compressor") and cc_s > 0:
                        demand += cc_s / max(1, reps * batch)
                if demand > 0:
                    caps.append(1.0 / demand)
            return caps

        for i in range(len(df)):
            base = df.iloc[i]
            row_latency, _ = _latency_for_row(base)
            batch = max(1, int(df["batch_size_request"].iloc[i]))
            db = max(1, int(df["batch_size_decode"].iloc[i]))
            pf_s = float(df["svc_prefill_s"].iloc[i])
            dec_s = float(df["svc_decode_s"].iloc[i])
            enc_s = float(df["svc_encode_s"].iloc[i])
            retr_s = float(df["svc_retrieval_s"].iloc[i])
            fusion_s = float(df["svc_fusion_s"].iloc[i])
            rr_s = float(df["svc_rerank_s"].iloc[i])
            cc_s = float(df["svc_compress_s"].iloc[i])
            comm_cap = float(_comm_bound_series.iloc[i])

            # r26 — ONE two-lane engine occupancy replaces the
            # pf_cap / dec_cap / r19 token-cap trio (and deletes the
            # r19c measured-grid parquet lookup: a per-box benchmark
            # dependency, the opposite of a portable cost model).
            # Physics (see engine_step_price): a continuous-batching
            # engine advances in steps; each step serves every decoding
            # sequence one token AND — chunked prefill — a slice of the
            # prompt backlog from the same max_num_batched_tokens
            # budget. In steady state the prompt slice riding each step
            # is c̄ = in·db/out, so a query occupies its engine for
            #   occ = out · step_s(db, c̄) / db
            #       = out · dec_step(db) / db  +  in / chunk_rate.
            # DECODE lane: the walk's per-call, context-aware decode
            # service (svc_decode_s = Σ out·dec_step at the deployed
            # decode batch). PREFILL lane: prompts ride the engine's own
            # 8192-token
            # chunk cadence (pure GenZ), not the giant-batch amortized
            # pf_s — serving never sees a 256-query mega-prefill.
            dec_occ = (
                dec_s / (_reps[i] * db)
                if dec_s > 0 else 0.0
            )
            dec_cap = (1.0 / dec_occ) if dec_occ > 0 else float("inf")
            _gin = float(df["gen_in_tokens_per_query"].iloc[i]
                         if "gen_in_tokens_per_query" in df.columns
                         else 0.0) or 0.0
            _row_pd = bool(
                "collocation_strategy" in base.index
                and pd_split(base["collocation_strategy"])
            )
            if _row_pd:
                # PD-disagg: prompt tokens never share a step budget
                # with decode tokens (different engines), so the
                # unified token-shared pricing does not apply — the
                # eval_0021 autopsy showed the one-token-budget cap
                # crushed a PD row to 0.46x while the bare batched
                # prefill price was within 7% of measured. A dedicated
                # prefill engine packs whole prompt batches with no
                # decode interleave — exactly the shape the walk's
                # batched GenZ prefill (svc_prefill_s) prices — and
                # each lane is its own physical engine: gen_cap = min.
                pf_cap = (
                    _pf_reps[i] * batch / pf_s
                    if pf_s > 0 else float("inf")
                )
                gen_cap = _min_finite((pf_cap, dec_cap))
            else:
                # Unified engine: one step budget carries both lanes,
                # so their occupancies ADD (one resource — harmonic
                # combination, not min-of-caps).
                try:
                    _par = _llm_parallelism_for_row(base).get(
                        "generator", {})
                    _ctp, _cpp = _par.get("prefill", (1, 1))
                except Exception:  # noqa: BLE001
                    _ctp, _cpp = 1, 1
                _chunk_rate = float("inf")
                if row_latency.prefill_s is not None:
                    _, _chunk_rate = engine_step_price(
                        row_latency.prefill_s, None, tp=_ctp, pp=_cpp)
                # Aux LLM engines (query_expansion) are separate vLLM
                # servers: their prefill keeps the batch-amortized
                # price on the shared-card demand.
                _pf_by_stage = _dict_value(base, "svc_prefill_by_stage_s")
                _pf_gen = float(
                    _pf_by_stage.get(_llm_stage_names("generator")[0],
                                     pf_s)
                    if _pf_by_stage else pf_s
                )
                _pf_other = max(0.0, pf_s - _pf_gen)
                pf_occ = _pf_other / (_pf_reps[i] * batch)
                if _gin > 0 and np.isfinite(_chunk_rate):
                    pf_occ += _gin / (_pf_reps[i] * _chunk_rate)
                else:
                    pf_occ += _pf_gen / (_pf_reps[i] * batch)
                pf_cap = (1.0 / pf_occ) if pf_occ > 0 else float("inf")
                _occ = dec_occ + pf_occ
                gen_cap = (1.0 / _occ) if _occ > 0 else float("inf")

            # Apply the SAME frozen hardware×stage×resident-batch
            # generation used by DES. Component capacities above remain
            # exact-device GenZ/portable-stage evidence; this block adds
            # only the canonical stage overhead. A collocated P+D cycle
            # gets one bare-engine correction after the lanes are
            # composed; direct-D gets one correction per engine cycle.
            # Direct-P is a completion-frontier curve and is charged once
            # per logical prefill, not once per scheduler chunk. No phase
            # residual or deployment knob is fitted.
            _gen_pf_depth = _engine_depth(
                "generator", "prefill", batch, db,
            )
            _gen_dec_depth = _engine_depth(
                "generator", "decode", batch, db,
            )
            _cycle_means = _mean_engine_cycles(
                batch,
                db,
                _gen_pf_depth,
                _gen_dec_depth,
            )
            _pref_by_stage = _dict_value(
                base, "svc_prefill_by_stage_s",
            )
            _dec_by_stage = _dict_value(
                base, "svc_decode_by_stage_s",
            )
            _llm_curve_caps: list[float] = []
            _curve_extra = 0.0
            _cycle_audit: dict[str, dict[str, Any]] = {}
            _main_pf_curve_cap = pf_cap
            _main_dec_curve_cap = dec_cap

            for _engine_name, _counts in sorted(_cycle_means.items()):
                _pref_stage, _dec_stage = _llm_stage_names(_engine_name)
                _mode = _engine_pd_mode(base, _engine_name)
                _pref_active = _engine_depth(
                    _engine_name, "prefill", batch,
                    db if _engine_name == "generator" else batch,
                )
                _dec_active = _engine_depth(
                    _engine_name, "decode", batch,
                    db if _engine_name == "generator" else batch,
                )
                _unified_active = _engine_depth(
                    _engine_name, "unified", batch,
                    db if _engine_name == "generator" else batch,
                )
                _pref_rep = _stage_replicas(base, _pref_stage)
                _dec_rep = _stage_replicas(base, _dec_stage)

                if _mode == "collocated_pd":
                    _replicas_engine = max(
                        1, min(_pref_rep, _dec_rep),
                    )
                    _overhead = _curve_overhead_s(
                        _engine_name, "unified", _unified_active,
                    )
                    _overhead_demand = (
                        float(_counts["unified"]) * _overhead
                        / (_replicas_engine * _unified_active)
                    )
                    if _engine_name == "generator":
                        _component_cap = gen_cap
                    else:
                        _component_demand = (
                            float(_pref_by_stage.get(_pref_stage, 0.0))
                            / (_pref_rep * _pref_active)
                            + float(_dec_by_stage.get(_dec_stage, 0.0))
                            / (_dec_rep * _dec_active)
                        )
                        _component_cap = (
                            1.0 / _component_demand
                            if _component_demand > 0.0 else float("inf")
                        )
                    _component_demand = (
                        1.0 / _component_cap
                        if np.isfinite(_component_cap)
                        and _component_cap > 0.0 else 0.0
                    )
                    _adjusted_demand = (
                        _component_demand + _overhead_demand
                    )
                    _adjusted_cap = (
                        1.0 / _adjusted_demand
                        if _adjusted_demand > 0.0 else float("inf")
                    )
                    _llm_curve_caps.append(_adjusted_cap)
                    if _engine_name == "generator":
                        # Unified-role diagnostics intentionally share the
                        # same adjusted engine capacity; they are not
                        # independent phase bounds.
                        _main_pf_curve_cap = _adjusted_cap
                        _main_dec_curve_cap = _adjusted_cap
                    _curve_extra += float(_counts["unified"]) * _overhead
                    _cycle_audit[_engine_name] = {
                        "pd_serving": _mode,
                        "resident_batch": _unified_active,
                        "mean_cycles_per_query": float(
                            _counts["unified"]
                        ),
                        "overhead_s_per_cycle": _overhead,
                    }
                else:
                    _pref_overhead = _curve_overhead_s(
                        _engine_name, "prefill", _pref_active,
                    )
                    _dec_overhead = _curve_overhead_s(
                        _engine_name, "decode", _dec_active,
                    )
                    _pref_overhead_demand = (
                        float(_counts["prefill_frontiers"])
                        * _pref_overhead
                        / (_pref_rep * _pref_active)
                    )
                    _dec_overhead_demand = (
                        float(_counts["decode"]) * _dec_overhead
                        / (_dec_rep * _dec_active)
                    )
                    if _engine_name == "generator":
                        _pref_component_cap = pf_cap
                        _dec_component_cap = dec_cap
                    else:
                        _pref_component_demand = (
                            float(_pref_by_stage.get(_pref_stage, 0.0))
                            / (_pref_rep * _pref_active)
                        )
                        _dec_component_demand = (
                            float(_dec_by_stage.get(_dec_stage, 0.0))
                            / (_dec_rep * _dec_active)
                        )
                        _pref_component_cap = (
                            1.0 / _pref_component_demand
                            if _pref_component_demand > 0.0
                            else float("inf")
                        )
                        _dec_component_cap = (
                            1.0 / _dec_component_demand
                            if _dec_component_demand > 0.0
                            else float("inf")
                        )
                    _pref_demand = (
                        (1.0 / _pref_component_cap)
                        if np.isfinite(_pref_component_cap)
                        and _pref_component_cap > 0.0 else 0.0
                    ) + _pref_overhead_demand
                    _dec_demand = (
                        (1.0 / _dec_component_cap)
                        if np.isfinite(_dec_component_cap)
                        and _dec_component_cap > 0.0 else 0.0
                    ) + _dec_overhead_demand
                    _pref_adjusted = (
                        1.0 / _pref_demand
                        if _pref_demand > 0.0 else float("inf")
                    )
                    _dec_adjusted = (
                        1.0 / _dec_demand
                        if _dec_demand > 0.0 else float("inf")
                    )
                    _llm_curve_caps.extend((
                        _pref_adjusted, _dec_adjusted,
                    ))
                    if _engine_name == "generator":
                        _main_pf_curve_cap = _pref_adjusted
                        _main_dec_curve_cap = _dec_adjusted
                    _curve_extra += (
                        float(_counts["prefill_frontiers"])
                        * _pref_overhead
                        + float(_counts["decode"]) * _dec_overhead
                    )
                    _cycle_audit[_engine_name] = {
                        "pd_serving": _mode,
                        "prefill_resident_batch": _pref_active,
                        "decode_resident_batch": _dec_active,
                        "mean_prefill_cycles_per_query": float(
                            _counts["prefill"]
                        ),
                        "mean_prefill_frontiers_per_query": float(
                            _counts["prefill_frontiers"]
                        ),
                        "mean_decode_cycles_per_query": float(
                            _counts["decode"]
                        ),
                        "prefill_overhead_s_per_frontier": _pref_overhead,
                        "decode_overhead_s_per_cycle": _dec_overhead,
                    }

            _llm_curve_cap = _min_finite(_llm_curve_caps)
            curve_extra_latency_s.append(_curve_extra)
            curve_cycle_counts.append(_cycle_audit)
            trace_mean_with_curve_s.append(
                float(df["latency_s_trace_mean"].iloc[i]) + _curve_extra
            )
            trace_median_with_curve_s.append(
                float(df["latency_s_trace_median"].iloc[i]) + _curve_extra
            )
            enc_cap = (
                _encode_reps[i] * batch / enc_s
                if enc_s > 0 else float("inf")
            )
            # Keep the conservative serial retrieval-service envelope for
            # coarse screening. Only encode/search inherit retrieval-server
            # replication; the independent local CPU fusion service does not.
            # DES schedules the explicit services and their resource contention.
            retrieval_demand = (
                (max(0.0, retr_s - fusion_s) + enc_s) / _retr_servers[i]
                + fusion_s
            )
            retr_cap = (
                batch / retrieval_demand if retrieval_demand > 0 else float("inf")
            )
            fusion_cap = batch / fusion_s if fusion_s > 0 else float("inf")
            rr_cap = batch / rr_s if rr_s > 0 else float("inf")
            cc_cap = batch / cc_s if cc_s > 0 else float("inf")
            if comm_cap <= 0:
                comm_cap = float("inf")
            gpu_group_caps = _gpu_resource_caps(base, batch, db)
            resource_cap = _min_finite(gpu_group_caps)
            # rr_cap is the reranker's serialized WORKER bound (measured
            # runs exactly one BatchedPureStageService per stage — extra
            # reranker chips do not multiply the worker loop, so no
            # replica credit here; the GPU-group demand above still
            # divides by replicas for the card-side work).
            # r19b: gen_cap (engine-loop bounds: chunk-cadence token
            # rate, per-call turnaround, KV-handoff-scoped decode) joins
            # the binding set — it was previously diagnostic-only, so
            # every engine-side constraint silently lost to the
            # GPU-group demand model (probe: 0048 priced 238 while its
            # own pf_cap said 155).
            caps = [
                c for c in (resource_cap, _llm_curve_cap, retr_cap, rr_cap,
                            cc_cap, comm_cap)
                if np.isfinite(c)
            ]
            if not caps:
                caps = [
                    c for c in (
                        _llm_curve_cap, enc_cap, retr_cap, rr_cap, cc_cap,
                        comm_cap,
                    )
                    if np.isfinite(c)
                ]
            qps_prefill.append(_finite_or_zero(_main_pf_curve_cap))
            qps_decode.append(_finite_or_zero(_main_dec_curve_cap))
            qps_gen.append(_finite_or_zero(
                min(
                    resource_cap,
                    _llm_curve_cap,
                ) if np.isfinite(resource_cap) else _llm_curve_cap
            ))
            qps_encode.append(_finite_or_zero(enc_cap))
            qps_retr.append(_finite_or_zero(retr_cap))
            qps_fusion.append(_finite_or_zero(fusion_cap))
            qps_rerank.append(_finite_or_zero(rr_cap))
            qps_compress.append(_finite_or_zero(cc_cap))
            qps_comm.append(_finite_or_zero(comm_cap))
            qps_gpu_resource.append(_finite_or_zero(resource_cap))
            qps_llm_engine_curve.append(
                _finite_or_zero(_llm_curve_cap)
            )
            if not caps:
                raise RuntimeError(
                    "Analytical candidate has no finite resource capacity "
                    f"(row position {i}); refusing a coarse-score fallback"
                )
            overall = min(caps)
            # r17 — closed-loop equilibrium (user directive): measured is a
            # CLOSED saturated loop, so each batch-worker stage's effective
            # rows/batch is DETERMINED by the equilibrium, not the deployed
            # cap. Joint fixed point, no fitted parameters:
            #   B_i  = clip(q * v_i * s_i(B_i), 1, cap)   — a worker's next
            #          batch is what arrived while it served the previous one
            #   s_i(B) = b_i*B — component demand scaled from the row's
            #          own priced whole-batch service at the deployed cap
            #   T(q) = walk latency with worker stages repriced at B_i(q)
            #   q    = min(capacity ceiling, N / T(q))    — population identity
            # Ceiling-limited rows (N/T >> ceiling) are bit-identical to the
            # pre-r17 result; population-starved rows (deep chains at b256)
            # land below the ceiling exactly as measured does.
            _eq_batches: dict[str, float] = {}
            _trace_mean_i = float(
                trace_mean_with_curve_s[-1]
                if trace_mean_with_curve_s else 0.0
            ) or 0.0
            if _trace_mean_i > 0 and np.isfinite(overall) and overall > 0:
                _N = max(1, int(_closed_loop_load_concurrency(
                    _sys_cfg, batch,
                    db if db != batch else None,
                )))
                _v_retr = max(0.0, float(
                    df["retrieval_calls_per_query"].iloc[i]
                    if "retrieval_calls_per_query" in df.columns else 0.0
                ) or 0.0)
                _v_rr = max(0.0, float(
                    df["rerank_calls_per_query"].iloc[i]
                    if "rerank_calls_per_query" in df.columns else 0.0
                ) or 0.0) if rr_s > 0 else 0.0
                _stages_eq = []
                if retr_s > 0 and _v_retr > 0:
                    _S_cap = (retr_s + enc_s) / _v_retr
                    _stages_eq.append(
                        ("retr", _v_retr, 0.0, _S_cap / batch))
                if rr_s > 0 and _v_rr > 0 and np.isfinite(rr_cap) and rr_cap > 0:
                    _S_cap = (batch / rr_cap) / _v_rr
                    _stages_eq.append(
                        ("rerank", _v_rr, 0.0, _S_cap / batch))
                if cc_s > 0 and np.isfinite(cc_cap) and cc_cap > 0:
                    _S_cap = batch / cc_cap
                    _stages_eq.append(
                        ("compress", 1.0, 0.0, _S_cap / batch)
                    )
                _full_batch_part = sum(
                    v * (a + b * batch) for _, v, a, b in _stages_eq)
                _timeout_s = float(
                    base.get(
                        "dynamic_batch_timeout_s",
                        _dynamic_timeout_s
                        if _dynamic_timeout_s is not None else 0.002,
                    )
                )
                if not np.isfinite(_timeout_s) or _timeout_s < 0.0:
                    raise ValueError(
                        "dynamic_batch_timeout_s must be finite and "
                        f"non-negative, got {_timeout_s!r}"
                    )

                def _batch_fill_fixed_point(
                    rate: float,
                    intercept_s: float,
                    slope_s: float,
                ) -> float:
                    """Resident rows collected by count-or-time batching.

                    While one batch is in service the next one can fill;
                    otherwise its first row starts the configured timeout.
                    Therefore the collection window is
                    ``max(service(B), timeout)`` and the unfitted physical
                    fixed point is B=clip(rate*window, 1, cap).  This is a
                    coarse-screen mechanism only; DES remains authoritative.
                    """

                    if rate <= 0.0:
                        return 1.0
                    current = min(
                        float(batch),
                        max(1.0, rate * _timeout_s),
                    )
                    for _fill_iteration in range(24):
                        service_s = max(
                            0.0,
                            intercept_s + slope_s * current,
                        )
                        updated = min(
                            float(batch),
                            max(
                                1.0,
                                rate * max(service_s, _timeout_s),
                            ),
                        )
                        if abs(updated - current) <= 1e-6 * max(
                            1.0, current,
                        ):
                            return float(updated)
                        current = 0.5 * (current + updated)
                    return float(current)

                _q = float(overall)
                for _ in range(24):
                    _t = _trace_mean_i - _full_batch_part
                    for name, v, a, b in _stages_eq:
                        _rate = _q * v
                        _B = _batch_fill_fixed_point(
                            _rate, a, b,
                        )
                        _eq_batches[name] = _B
                        _service_s = max(0.0, a + b * _B)
                        # If the worker drains before the timeout, arrivals
                        # in that idle collection window see the exact
                        # deterministic mean residual wait timeout/2.  A
                        # full batch or a service-filled successor has no
                        # additional timer wait here.
                        _timer_wait_s = (
                            0.5 * max(0.0, _timeout_s - _service_s)
                            if _B < float(batch) - 1e-9 else 0.0
                        )
                        _t += v * (_service_s + _timer_wait_s)
                    if _t <= 0:
                        break
                    _q_new = min(float(overall), _N / _t)
                    if abs(_q_new - _q) <= 1e-3 * max(_q, 1e-9):
                        _q = _q_new
                        break
                    _q = 0.5 * (_q + _q_new)
                overall = min(float(overall), _q)
            eq_batch_retr.append(_eq_batches.get("retr", float("nan")))
            eq_batch_rerank.append(_eq_batches.get("rerank", float("nan")))
            eq_batch_compress.append(_eq_batches.get("compress", float("nan")))
            qps_overall.append(overall)
        df["qps_prefill_capacity"] = qps_prefill
        df["qps_decode_capacity"] = qps_decode
        df["qps_gen_capacity"] = qps_gen
        df["qps_encode_capacity"] = qps_encode
        df["qps_retrieval_capacity"] = qps_retr
        df["qps_fusion_capacity"] = qps_fusion
        df["qps_rerank_capacity"] = qps_rerank
        df["qps_compress_capacity"] = qps_compress
        df["qps_comm_capacity"] = qps_comm
        df["qps_gpu_resource_capacity"] = qps_gpu_resource
        df["qps_llm_engine_curve_capacity"] = qps_llm_engine_curve
        df["analytical_engine_cycle_curve_extra_latency_s"] = (
            curve_extra_latency_s
        )
        df["analytical_engine_cycle_curve_cycles"] = curve_cycle_counts
        df["latency_s_trace_mean"] = trace_mean_with_curve_s
        df["latency_s_trace_median"] = trace_median_with_curve_s
        df["latency_s"] = trace_mean_with_curve_s
        df["dynamic_analytical_stage_batch_generation_sha256"] = (
            _stage_batch_generation.generation_sha256
        )
        df["dynamic_analytical_stage_batch_profile_content_sha256"] = [
            dict(_stage_batch_generation.profile_content_sha256)
            for _ in range(len(df))
        ]
        df["dynamic_analytical_stage_batch_enabled"] = (
            _stage_batch_generation.enabled
        )
        # r17 diagnostics: the equilibrium rows/batch the closed-loop
        # fixed point predicts for each dynamic-batch worker — directly
        # comparable to measured `<stage>_dynamic_batch_avg_size`.
        df["stage_batch_eq_retrieval"] = eq_batch_retr
        df["stage_batch_eq_rerank"] = eq_batch_rerank
        df["stage_batch_eq_compress"] = eq_batch_compress
        df["qps"] = qps_overall
        if "num_chips_e2e" in df.columns:
            df["qps_per_chip"] = [
                float(qps_overall[i])
                / max(1, int(df["num_chips_e2e"].iloc[i]))
                for i in range(len(df))
            ]

        # Approximate mean response time using the service path and analytical
        # load derived from deployed batch depths. The simulator independently
        # searches for a saturated population; no measured load is an input.
        if "qps" in df.columns and "batch_size_request" in df.columns:
            _loads = [
                _closed_loop_load_concurrency(
                    _sys_cfg,
                    int(df["batch_size_request"].iloc[i]),
                    (
                        int(df["batch_size_decode"].iloc[i])
                        if "batch_size_decode" in df.columns
                        and pd.notna(df["batch_size_decode"].iloc[i])
                        else None
                    ),
                )
                for i in range(len(df))
            ]
            _queue_mean = [
                (
                    max(1, int(_loads[i])) / float(df["qps"].iloc[i])
                    if float(df["qps"].iloc[i]) > 0.0
                    and np.isfinite(float(df["qps"].iloc[i]))
                    else float(df["latency_s_trace_mean"].iloc[i])
                )
                for i in range(len(df))
            ]
            _trace_mean = [
                float(df["latency_s_trace_mean"].iloc[i])
                for i in range(len(df))
            ]
            _final_mean = [
                _closed_loop_mean_latency(
                    _trace_mean[i], float(df["qps"].iloc[i]), _loads[i],
                )
                for i in range(len(df))
            ]
            df["load_concurrency"] = _loads
            df["latency_s_queue_mean"] = _queue_mean
            df["latency_s_queue_wait"] = [
                max(0.0, _final_mean[i] - _trace_mean[i])
                for i in range(len(df))
            ]
            df["latency_s"] = _final_mean
            # Legacy diagnostic: stage batch divided by qps is not a response-time
            # target, but it is still useful for spotting batch/QPS scale errors.
            df["latency_s_batch_qps_proxy"] = [
                float(df["batch_size_request"].iloc[i]) / max(1e-9, float(df["qps"].iloc[i]))
                for i in range(len(df))
            ]
        if "qps" not in df.columns or "latency_s" not in df.columns:
            raise RuntimeError(
                "DynamicEngine analytical prepass did not publish qps/latency"
            )
        # Immutable shortlist evidence.  apply_closed_loop_simulation will
        # overwrite only the optimizer-facing qps/latency columns with DES;
        # these explicit names survive as audit diagnostics and are never a
        # fallback result for screened or infeasible rows.
        df["dynamic_analytical_qps"] = df["qps"].astype(float)
        df["dynamic_analytical_latency_s"] = df["latency_s"].astype(float)
        for column in ("latency_s_ttft", "latency_s_tpot", "qps_per_chip"):
            if column in df.columns:
                df["dynamic_analytical_" + column] = df[column].astype(float)
        df["dynamic_analytical_score_is_final"] = False
        return df
