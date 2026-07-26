"""Closed-saturated discrete-event simulator for RAG CM.

The simulator consumes only the frozen workflow schema, canonical quality trace
(ordered QA invocations plus input/output token counts), algorithm config, and
the candidate hardware/layout.  It never reads measured-runtime latency,
padding groups, forward batches, host wall time, QPS, or performance traces.
Portable component service comes from the existing analytical cost models
(GenZ, FAISS, reranker/compressor models).

Each light-stage worker has an explicit count/timeout queue.  Each vLLM engine
has resident prefill/decode state; a scheduler step freezes the current prompt
slices and every sequence's full input/generated position.  Engine steps and
light-stage CUDA kernels enter one per-card FCFS arbiter, acquire all TP/PP
cards atomically, and run non-preemptively.  CPU operators use the box's
logical cores as one processor-sharing resource.  Query-expansion retrieval
flattens every live row's K sub-queries, chunks them by the deployed request
cap, and runs ordered ``encode GPU -> FAISS CPU -> backend glue`` chunks plus
one outer worker tail. Multi-list RRF is a separately priced CPU operation
inside that same physical worker, sharing the host's CPU capacity.

PD layouts use independent prefill/decode queues plus an explicit pair-aware
KV handoff.  Placement, TP/PP/DP, queue depths, batching timeout, FAISS thread
width and vLLM batched-token budget come from the candidate config (with
documented config defaults).  Closed-loop population starts from those
structural deployment capacities and grows to a numerical throughput plateau;
measured-client load policy is not consumed.  No end-to-end correction
coefficient is fitted in this simulation.
"""

from __future__ import annotations

import heapq
import ast
import json
import logging
import math
import os
from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from rag_stack.cost_model.workload import PreparedExecutionPaths

# Structural work-type split: fraction of a stage's SOLO service that
# occupies its GPU (the rest is host work in the stage's own process — r20).
# Shared with the fixed-placement capacity model so the two paths use the
# same decomposition (rag_stack/cost_model/stage_work_split.py).
from rag_stack.cost_model.retrieval_worker import (
    build_retrieval_worker_plan,
    retrieval_chunk_sizes,
)

_BATCH_TIMEOUT_S = 0.002       # configured serving default when not materialized
_PREFILL_BUDGET = 8192         # vLLM chunked-prefill per-step token budget
_VLLM_PREFIX_BLOCK_TOKENS = 16  # vLLM's public default cache block size

# Saturation is a numerical fixed point of the closed network, not a measured
# client-load policy.  Starting from the public deployment capacities below,
# double the population until another doubling changes throughput by at most
# this tolerance.  The tolerance affects only convergence detection; it never
# rescales a component service or the reported QPS.
_SATURATION_QPS_REL_TOL = 5e-3
_SATURATION_QPS_ABS_TOL = 1e-9
_SATURATION_MAX_DOUBLINGS = 4
_SATURATION_MAX_POPULATION = 4096

# Safety bound counts logical DES loop iterations.  Adjacent no-op event
# coalescing below must debit omitted events against this same budget; otherwise
# an optimized run could cross a convergence boundary that the reference event
# stream would have stopped before reaching.
_SIM_MAX_EVENTS = 5_000_000

# The simulator is deterministic, so production sweeps use two warmup and two
# measured population turnovers.  This is a bounded deterministic observation
# horizon, not a fitted service coefficient: every queue/kernel event inside
# it is unchanged.  Callers of ``simulate`` can still request a longer window
# explicitly for convergence audits.
_SIM_WARMUP_TURNOVERS = 2
_SIM_WINDOW_TURNOVERS = 2

# The Python event loop is GIL-bound, but exact deployment signatures are
# independent. The adapter therefore prices signatures in processes and leaves
# two affinity-visible CPUs for the parent plus host services.
_DES_RESERVED_CPUS = 2

# Keep several ready tasks behind every DES worker.  Candidate runtime is not
# perfectly predictable from its deployment shape, so one static shard per
# process can leave the whole pool waiting for a single coarse tail shard.
# Four-way oversubscription is enough for executor work stealing while bounding
# serialization and scheduling overhead to O(worker_count), never O(rows).
_DES_TASKS_PER_WORKER = 4

logger = logging.getLogger("RAG-Stack")

_STAGE_BATCH_RUNTIME_SELECTION_KEY = "cm_v1_stage_batch_selection"
_STAGE_BATCH_FROZEN_HARDWARE_ATTR = (
    "_rag_cm_v1_frozen_stage_batch_generation"
)
_STAGE_BATCH_PROFILES_UNSET = object()


def _rag_cm_calibration_mode(
    pipeline_config: Mapping[str, Any],
    hardware: Any,
) -> str:
    """Resolve calibration mode with replay hardware policy taking precedence."""

    from rag_stack.cost_model.calibration_mode import (
        resolve_calibration_mode,
    )

    return resolve_calibration_mode(pipeline_config, hardware)


def _pipeline_with_frozen_stage_batch_selection(
    pipeline_config: dict,
    hardware: Any,
) -> dict:
    """Attach one JSON-safe whole-file generation for every CM mode.

    Dynamic analytical screening and every serial/spawn DES worker consume
    this exact document.  Neither phase reopens the profile store, so a
    publication during a run cannot make two candidates use different curve
    evidence or let one topology win by observing a newer generation.
    """

    from rag_stack.cost_model.stage_batch_curve import (
        FrozenStageBatchGeneration,
        resolve_frozen_stage_batch_generation,
    )

    calibration_mode = _rag_cm_calibration_mode(pipeline_config, hardware)

    runtime = pipeline_config.get("_runtime") or {}
    existing = runtime.get(_STAGE_BATCH_RUNTIME_SELECTION_KEY)
    if existing is not None:
        generation = FrozenStageBatchGeneration.from_dict(existing)
        if generation.calibration_mode != calibration_mode:
            raise ValueError(
                "frozen stage-batch generation calibration mode mismatch"
            )
        # Constructor validation already checks the embedded hardware key;
        # resolve the runtime hardware identity as an independent guard.
        from rag_stack.cost_model.stage_batch_curve import (
            _system_hardware,
        )
        if generation.system_hardware != _system_hardware(hardware):
            raise ValueError(
                "frozen stage-batch generation hardware identity mismatch"
            )
        return pipeline_config

    frozen_by_mode = getattr(
        hardware,
        _STAGE_BATCH_FROZEN_HARDWARE_ATTR,
        None,
    )
    if not isinstance(frozen_by_mode, dict):
        frozen_by_mode = {}
    document = frozen_by_mode.get(calibration_mode)
    if document is None:
        generation = resolve_frozen_stage_batch_generation(
            hardware,
            calibration_mode=calibration_mode,
        )
        document = generation.to_dict()
        frozen_by_mode = {**frozen_by_mode, calibration_mode: deepcopy(document)}
        setattr(
            hardware,
            _STAGE_BATCH_FROZEN_HARDWARE_ATTR,
            frozen_by_mode,
        )
    else:
        generation = FrozenStageBatchGeneration.from_dict(document)
        if generation.calibration_mode != calibration_mode:
            raise ValueError(
                "cached frozen stage-batch generation mode mismatch"
            )

    frozen = deepcopy(pipeline_config)
    frozen.setdefault("_runtime", {})[
        _STAGE_BATCH_RUNTIME_SELECTION_KEY
    ] = deepcopy(document)
    return frozen


def _stage_batch_generation_for_pipeline(
    pipeline_config: Mapping[str, Any],
    hardware: Any,
):
    """Validate and materialize the attached whole-file snapshot."""

    from rag_stack.cost_model.stage_batch_curve import (
        FrozenStageBatchGeneration,
        _system_hardware,
    )

    runtime = pipeline_config.get("_runtime") or {}
    document = runtime.get(_STAGE_BATCH_RUNTIME_SELECTION_KEY)
    if document is None:
        raise RuntimeError(
            "pipeline has no frozen stage-batch generation; call "
            "_pipeline_with_frozen_stage_batch_selection first"
        )
    generation = FrozenStageBatchGeneration.from_dict(document)
    if generation.system_hardware != _system_hardware(hardware):
        raise ValueError(
            "frozen stage-batch generation hardware identity mismatch"
        )
    expected_mode = _rag_cm_calibration_mode(pipeline_config, hardware)
    if generation.calibration_mode != expected_mode:
        raise ValueError(
            "frozen stage-batch generation calibration mode mismatch"
        )
    return generation


def _stage_batch_profiles_for_pipeline(
    pipeline_config: Mapping[str, Any],
    hardware: Any,
) -> Any:
    """Return the exact frozen empty/pair/six profile mapping."""

    generation = _stage_batch_generation_for_pipeline(
        pipeline_config,
        hardware,
    )
    return generation.materialize_profiles()


def _data_parallel_global_batch_cap(
    local_batch_cap: int,
    devices: Sequence[str],
) -> int:
    """Return the gang-batcher cap for one-model-per-device HF stages.

    ``batch_size_request`` is the local request cap of each concrete replica;
    the measured worker collects one shard per replica before dispatch.
    """

    local = int(local_batch_cap)
    if local <= 0:
        raise ValueError(f"invalid local data-parallel batch cap {local}")
    return local * max(1, len(devices))


def _engine_cycle_profile_stage(
    engine_stage: str,
    role: str,
    *,
    prefill_stage: str,
    decode_stage: str,
) -> str:
    """Resolve the exact calibration boundary for one physical engine.

    A collocated engine owns one bare-stage curve.  A disaggregated role owns
    only its direct PerformanceStage curve; callers must not fall back from a
    missing role curve to the collocated curve.
    """

    if role == "unified":
        return str(engine_stage)
    if role == "prefill":
        return str(prefill_stage)
    if role == "decode":
        return str(decode_stage)
    raise ValueError(f"invalid simulator engine role {role!r}")


def _llm_admission_capacity(max_num_seqs: int) -> int:
    """Public serving admission policy for one logical vLLM engine.

    The client permits one fifth of the engine resident cap to wait inside
    vLLM so a completion can be backfilled without draining the scheduler.
    This is a system-runtime policy, not a measured-performance observation:
    the DES derives it exclusively from the configured ``max_num_seqs``.
    Keep this integer form equivalent to runtime ``ceil(0.20 * cap)`` while
    avoiding a floating-point edge at exact multiples of five.
    """
    cap = max(1, int(max_num_seqs))
    return cap + max(1, (cap + 4) // 5)


def _pd_prefill_feeder_slots() -> int:
    """Return the measured runtime's bounded P-side feed-ahead policy.

    This is deliberately imported from the serving runtime instead of fitted
    or duplicated in CM.  The slot is client admission in front of the P
    engine; it is not engine residency and is not a calibration coefficient.
    """

    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_pd_pair import (
        pd_prefill_feeder_slots,
    )

    slots = int(pd_prefill_feeder_slots())
    if slots < 0:
        raise ValueError(
            f"PD prefill feeder policy returned a negative slot count {slots}"
        )
    return slots


def _pd_prefill_admission_capacity(
    engine_max_num_seqs: int,
    *,
    replicas: int = 1,
) -> int:
    """Aggregate P client admission for independent measured P/D pairs."""

    cap = max(1, int(engine_max_num_seqs))
    count = max(1, int(replicas))
    return count * (cap + _pd_prefill_feeder_slots())


def _pd_decode_admission_capacity(
    engine_max_num_seqs: int,
    *,
    replicas: int = 1,
) -> int:
    """Aggregate strict D admission; no generic HTTP slack is permitted."""

    return max(1, int(replicas)) * max(1, int(engine_max_num_seqs))


def _coprime_trace_stride(query_count: int) -> int:
    """Choose a deterministic near-half rotation that visits every query.

    The historical ``N // 2 + 1`` stride shares a divisor with some trace
    lengths (for example 40 with N=78), trapping each closed-loop slot in only
    one subset forever.  The next coprime stride keeps the intended
    decorrelation while making the rotation a full permutation for every N.
    """
    count = max(1, int(query_count))
    if count == 1:
        return 1
    stride = count // 2 + 1
    while math.gcd(stride, count) != 1:
        stride += 1
        if stride >= count:
            stride = 1
    return stride


@dataclass(frozen=True)
class SimStageItem:
    """One quality-work item queued at a light-stage worker.

    ``calls`` retains the ordered quality invocations folded into one physical
    worker visit.  Dense retrieval is the important example: encode, FAISS,
    and local RRF are separate logical calls but one measured worker service.
    The simulator owns
    ``rid``; builders leave it ``None`` and the closed-loop admission path
    stamps the current request slot when it queues the item.
    """

    rid: Optional[int] = None
    stage: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    input_bytes: Optional[float] = None
    multiplicity: int = 1
    calls: Tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class SimStageOp:
    """One non-preemptive operator inside a physical stage-worker batch."""

    resource: str                    # ``gpu`` or ``cpu``
    service_s: float
    host_width: float = 1.0
    name: str = ""


@dataclass
class SimStage:
    name: str
    service_s: Callable[[int], float]   # batch -> SOLO seconds (host+gpu)
    gpu_share: float
    card: Optional[int]                 # card-group id; None = host-only
    batch_cap: int
    timeout_s: float = _BATCH_TIMEOUT_S
    # explicit GPU part (batch -> seconds); overrides gpu_share when set
    gpu_service_s: Optional[Callable[[int], float]] = None
    # Threads the HOST part of the service runs on (r29 CPU resource):
    # faiss vectorsearch is configured to the box's core count on every
    # measured deployment; Python worker glue is single-threaded. The
    # host burst's solo WALL time is unchanged — the width only sets how
    # much of the shared core budget the burst occupies while it runs.
    host_width: float = 1.0
    # Concrete GPU resources. ``card`` remains the legacy single-device form;
    # builders use ``cards`` so a DP stage occupies every active replica and
    # slows at the most-contended device (the lock-step straggler).
    cards: Tuple[int, ...] = ()
    devices: Tuple[str, ...] = ()
    # Shape-aware service hooks.  ``service_s(batch)`` remains the compatibility
    # API used by synthetic tests and older direct callers.  Production builders
    # publish ``batch_ops`` so the worker executes the real operator order
    # (notably K × [encode GPU -> FAISS CPU] followed by one fused glue op).
    batch_service_s: Optional[Callable[[Sequence[SimStageItem]], float]] = None
    batch_gpu_service_s: Optional[
        Callable[[Sequence[SimStageItem]], float]
    ] = None
    batch_ops: Optional[
        Callable[[Sequence[SimStageItem]], Sequence[SimStageOp]]
    ] = None


@dataclass(frozen=True)
class SimPrefillItem:
    """One prompt slice selected for the next engine scheduler step."""

    context_tokens: float
    new_tokens: float


@dataclass(frozen=True)
class SimDecodeItem:
    """Per-sequence decode state visible to the GenZ step pricer."""

    input_tokens: float
    generated_tokens: float
    remaining_tokens: float


@dataclass
class SimEngine:
    name: str
    cards: Tuple[int, ...]
    depth: int                          # max resident requests
    # (decoding_seqs, chunk_tokens) -> seconds for ONE engine step serving
    # each decoding sequence one token plus a chunk_tokens prefill slice
    # (engines.engine_step_price — the same two-lane physics as the caps).
    step_s: Callable[[int, float], float]
    budget: int = _PREFILL_BUDGET
    # Exact component-profile fixed work, charged once per logical LLM call.
    call_overhead_s: float = 0.0
    # A unified engine performs prefill then decode in one resident queue.
    # PD disaggregation materializes two independent engines instead: the
    # prefill engine releases a request to an explicit KV-transfer event and
    # the decode engine admits it only after that event completes.
    role: str = "unified"               # unified | prefill | decode
    devices: Tuple[str, ...] = ()        # concrete device / cache identity
    # Shape-aware hooks used by production CM.  The legacy aggregate ``step_s``
    # stays mandatory for direct tests and compatibility; when these hooks are
    # present the engine never collapses the trace to one global mean shape.
    prefill_batch_s: Optional[
        Callable[[Sequence[SimPrefillItem]], float]
    ] = None
    decode_step_s: Optional[
        Callable[[Sequence[SimDecodeItem]], float]
    ] = None
    # Automatic prefix caching is a property of one concrete vLLM replica.
    # Capacity is expressed in the same fixed-size blocks as the runtime and
    # is derived by the builder from model KV shape + declared GPU memory.
    # A zero capacity deliberately disables APC when those public facts cannot
    # be resolved; the simulator never invents a measured hit-rate fallback.
    prefix_cache_block_tokens: int = _VLLM_PREFIX_BLOCK_TOKENS
    prefix_cache_capacity_blocks: int = 0


@dataclass(frozen=True)
class SimTransfer:
    """One explicit inter-engine transfer.

    Transfers sharing ``resource`` serialize their payload service
    (``payload_bytes / bandwidth_bytes_s``).  Link latency is included in the
    completion event, so both the finite-bandwidth throughput bound and the
    per-call handoff delay emerge from the same event stream.
    """

    name: str
    resource: str
    payload_bytes: float
    bandwidth_bytes_s: float
    latency_s: float = 0.0
    source_calls: Tuple[Mapping[str, Any], ...] = ()

    @property
    def service_s(self) -> float:
        if self.payload_bytes <= 0:
            return max(0.0, float(self.latency_s))
        if self.bandwidth_bytes_s <= 0:
            return math.inf
        return max(0.0, float(self.latency_s)) + (
            float(self.payload_bytes) / float(self.bandwidth_bytes_s)
        )


@dataclass
class SimCall:
    """One engine call inside a round."""
    engine: int
    eff_in: int                          # prefill tokens actually computed
    out: int
    # Full prompt length is distinct from ``eff_in`` under APC.  Decode KV
    # attention needs the full context even when prefill computes only a suffix.
    full_in: Optional[int] = None
    # None means the first engine is unified and completes the whole call.
    # Otherwise ``engine`` is the prefill side and ``decode_engine`` is the
    # independent PD decode side.
    decode_engine: Optional[int] = None
    transfer: Optional[SimTransfer] = None
    # Optional replica pools.  The simulator chooses the least-loaded replica
    # at admission; the scalar indices above remain the backward-compatible
    # single-replica form.
    engine_pool: Tuple[int, ...] = ()
    decode_engine_pool: Tuple[int, ...] = ()
    transfers: Dict[Tuple[int, int], SimTransfer] = field(default_factory=dict)
    # Stable, quality-only prompt identity.  The block identities are built
    # from (query index, logical engine, call ordinal, token-count lineage),
    # never from runtime hit-rate or performance observations.  ``cache_blocks``
    # describes the full prompt blocks; ``result_cache_blocks`` additionally
    # includes full blocks formed by this call's generated tokens and is only
    # retained by a unified replica (PD decode KV does not teleport back to the
    # prefill replica).
    cache_identity: Optional[Tuple[Any, ...]] = None
    cache_blocks: Tuple[Any, ...] = ()
    result_cache_blocks: Tuple[Any, ...] = ()
    # Unified serving has one client-side admission semaphore around the
    # complete logical request.  These fields are derived from public runtime
    # policy after the frozen quality-trace boundary.
    admission_gate: Optional[str] = None
    admission_capacity: Optional[int] = None
    # Measured disaggregated serving instead has two role-local semaphores.
    # P includes only its bounded frontend feeder; after P finishes its permit
    # remains held while acquiring D.  Once D is reserved P is released, then
    # the physical KV transfer and decode execute while D remains held.
    prefill_admission_gate: Optional[str] = None
    prefill_admission_capacity: Optional[int] = None
    decode_admission_gate: Optional[str] = None
    decode_admission_capacity: Optional[int] = None
    # The compiler retains the evaluator invocation which lowered into this
    # prefill/decode pair. Consumers must not reconstruct it from token counts.
    source_calls: Tuple[Mapping[str, Any], ...] = ()


@dataclass
class SimRound:
    stage_idxs: Tuple[int, ...]          # light stages visited, in order
    call: Optional[SimCall]              # engine call closing the round
    # One quality-shaped workload record per ``stage_idxs`` entry.  Empty is
    # the backwards-compatible synthetic form used by direct simulator tests.
    stage_items: Tuple[SimStageItem, ...] = ()
    # Host-link transfers that must complete before route position ``pos``.
    # A position may carry multiple boundaries (e.g. the combined measured
    # retrieval worker contains encode->FAISS, then FAISS->reranker).
    boundary_transfers: Dict[int, Tuple[SimTransfer, ...]] = field(
        default_factory=dict,
    )


@dataclass
class SimResult:
    qps: float
    latency_mean_s: float
    engine_fill_mean: Dict[str, float]
    stage_batch_mean: Dict[str, float]
    window_s: float
    completions: int
    population: int
    transfer_latency_mean: Dict[str, float] = field(default_factory=dict)
    transfer_queue_wait_mean: Dict[str, float] = field(default_factory=dict)
    converged: bool = True


@dataclass(frozen=True)
class _SaturationOutcome:
    """One bounded numerical search for the closed-network throughput limit."""

    result: SimResult
    populations_tried: Tuple[int, ...]
    final_population: int
    plateau_reached: bool
    stop_reason: str


class _Card:
    __slots__ = ("n_active", "stage_busy")

    def __init__(self) -> None:
        self.n_active = 0
        # Light-stage CUDA kernels are discrete non-preemptive operators.  A
        # resident vLLM engine may resume between them, but it cannot make
        # fractional progress through a kernel that currently owns this card.
        self.stage_busy = 0


@dataclass
class _GpuOp:
    """One atomic GPU job submitted to the concrete-card arbiter."""

    cards: Tuple[int, ...]
    service_s: float
    owner: Any
    seq: int
    submitted_at: float
    name: str = ""
    started_at: Optional[float] = None


class _GpuArbiter:
    """Per-card FCFS arbitration for engine steps and light-stage kernels.

    A TP/PP job is queued under one global arrival ticket on every card it
    needs.  It starts only when it is the head of *all* those queues and all
    cards are idle.  Consequently an in-flight engine step is never preempted,
    a stage that arrived during it precedes the engine's next step, disjoint
    cards run concurrently, and multi-card acquisition is atomic.
    """

    def __init__(self, n_cards: int):
        self._queues: List[deque[_GpuOp]] = [
            deque() for _ in range(max(0, int(n_cards)))
        ]
        self._owners: List[Optional[_GpuOp]] = [
            None for _ in range(max(0, int(n_cards)))
        ]
        self._seq = 0
        self._cardless_active: Dict[int, _GpuOp] = {}

    def submit(
        self,
        *,
        cards: Sequence[int],
        service_s: Optional[float],
        owner: Any,
        now: float,
        name: str = "",
    ) -> Tuple[_GpuOp, Tuple[_GpuOp, ...]]:
        normalized = tuple(sorted(set(int(card) for card in cards)))
        if any(card < 0 or card >= len(self._queues) for card in normalized):
            raise ValueError(f"GPU op {name!r} references invalid cards {normalized}")
        deferred_service = service_s is None
        service_s = 0.0 if deferred_service else float(service_s)
        if (
            not deferred_service
            and (not math.isfinite(service_s) or service_s <= 0.0)
        ):
            raise ValueError(
                f"GPU op {name!r} has non-positive/non-finite service "
                f"{service_s!r}"
            )
        op = _GpuOp(
            cards=normalized,
            service_s=service_s,
            owner=owner,
            seq=self._seq,
            submitted_at=float(now),
            name=str(name),
        )
        self._seq += 1
        if not normalized:
            op.started_at = float(now)
            self._cardless_active[op.seq] = op
            return op, (op,)
        # The overwhelmingly common scheduler path has no queued contender.
        # Submitting work cannot make an unrelated queued operation ready, so
        # when every requested card is both idle and queue-empty this ticket is
        # already the FCFS head on all of them.  Acquire it directly without
        # building/sorting the global ready-candidate set.
        if all(
            self._owners[card] is None and not self._queues[card]
            for card in normalized
        ):
            for card in normalized:
                self._owners[card] = op
            op.started_at = float(now)
            return op, (op,)
        for card in normalized:
            self._queues[card].append(op)
        return op, self._start_ready(float(now))

    def release(self, op: _GpuOp, *, now: float) -> Tuple[_GpuOp, ...]:
        if not op.cards:
            if self._cardless_active.pop(op.seq, None) is not op:
                raise RuntimeError(f"GPU op {op.name!r} is not active")
            # Cardless work releases no resource and therefore cannot make a
            # queued GPU ticket newly runnable.
            return ()
        for card in op.cards:
            if self._owners[card] is not op:
                raise RuntimeError(
                    f"GPU op {op.name!r} does not own card {card}"
                )
        for card in op.cards:
            self._owners[card] = None
        if all(not self._queues[card] for card in op.cards):
            # No queued ticket depends on a card whose state just changed.
            # Readiness on every other card is unchanged from the previous
            # arbiter transition, which already ran _start_ready().
            return ()
        return self._start_ready(float(now))

    def _start_ready(self, now: float) -> Tuple[_GpuOp, ...]:
        candidates = {
            queue[0].seq: queue[0]
            for queue in self._queues
            if queue
        }
        started: List[_GpuOp] = []
        for op in sorted(candidates.values(), key=lambda candidate: candidate.seq):
            if op.started_at is not None:
                continue
            if any(self._owners[card] is not None for card in op.cards):
                continue
            if any(
                not self._queues[card] or self._queues[card][0] is not op
                for card in op.cards
            ):
                continue
            for card in op.cards:
                self._queues[card].popleft()
                self._owners[card] = op
            op.started_at = now
            started.append(op)
        return tuple(started)


class _Cpu:
    """The box's logical cores as ONE processor-sharing resource (r29).

    Every light-stage host task declares the thread WIDTH it runs at (faiss
    thread count or single-threaded worker glue).
    While the total demanded width fits into the core budget nothing
    stretches; oversubscription slows every host task by the fair-share
    factor total_width / cores — plain OS time-slicing, no fitted values.
    ``cores`` None disables the resource entirely (slow() == 1).
    """

    __slots__ = ("cores", "width_active")

    def __init__(self, cores: Optional[float]) -> None:
        self.cores = float(cores) if cores else None
        self.width_active = 0.0

    def slow(self) -> float:
        if self.cores is None or self.cores <= 0:
            return 1.0
        return max(1.0, self.width_active / self.cores)


class _BlockLRU:
    """Replica-local, block-granular KV pool with APC lookup state.

    vLLM does not allocate a separate prefix-cache arena: active request KV
    and reusable cached blocks share one physical block pool.  Cached blocks
    with a positive reference count are pinned by running requests and cannot
    be evicted; zero-reference blocks are the LRU eviction candidates.  Blocks
    whose deterministic identity is not yet publishable are represented by
    ``_anonymous_blocks`` and consume the same capacity.

    The historical class name is retained because this private helper is used
    by focused mechanism tests, but ``capacity`` is the *total* KV block
    budget, not an additional cache-only allowance.
    """

    def __init__(self, capacity_blocks: int):
        self.capacity = max(0, int(capacity_blocks))
        self._blocks: OrderedDict[Any, int] = OrderedDict()
        self._anonymous_blocks = 0

    @property
    def anonymous_blocks(self) -> int:
        return self._anonymous_blocks

    @property
    def physical_blocks(self) -> int:
        return len(self._blocks) + self._anonymous_blocks

    @property
    def pinned_blocks(self) -> int:
        return sum(1 for refs in self._blocks.values() if refs > 0)

    def _assert_invariant(self) -> None:
        if self._anonymous_blocks < 0:
            raise RuntimeError("KV block pool has negative anonymous occupancy")
        if self.physical_blocks > self.capacity:
            raise RuntimeError(
                "KV block pool exceeded capacity: "
                f"{self.physical_blocks} > {self.capacity}"
            )

    def _make_room(self, additional_blocks: int) -> bool:
        additional = max(0, int(additional_blocks))
        shortfall = max(
            0,
            self.physical_blocks + additional - self.capacity,
        )
        if shortfall == 0:
            return True
        victims = []
        for block, refs in self._blocks.items():
            if refs == 0:
                victims.append(block)
                if len(victims) == shortfall:
                    break
        if len(victims) != shortfall:
            # Allocation feasibility is checked before mutating the free LRU,
            # matching vLLM's all-or-nothing scheduler admission.
            return False
        for block in victims:
            del self._blocks[block]
        return True

    def longest_prefix(self, blocks: Sequence[Any]) -> int:
        if self.capacity <= 0:
            return 0
        hit = 0
        for block in blocks:
            if block not in self._blocks:
                break
            self._blocks.move_to_end(block)
            hit += 1
        return hit

    def acquire_longest_prefix(self, blocks: Sequence[Any]) -> Tuple[Any, ...]:
        """Pin and return the contiguous cached prefix of ``blocks``."""

        hit = self.longest_prefix(blocks)
        acquired = tuple(blocks[:hit])
        for block in acquired:
            self._blocks[block] += 1
        self._assert_invariant()
        return acquired

    def release(self, blocks: Sequence[Any]) -> None:
        """Release one resident reference to each cached block."""

        for block in blocks:
            refs = self._blocks.get(block)
            if refs is None or refs <= 0:
                raise RuntimeError("released an unpinned KV cache block")
            self._blocks[block] = refs - 1
            if refs == 1:
                # vLLM appends a newly free cached block to the eviction tail.
                self._blocks.move_to_end(block)
        self._assert_invariant()

    def reserve_anonymous(self, blocks: int) -> bool:
        """Reserve unkeyed active KV, evicting only reusable cache blocks."""

        count = max(0, int(blocks))
        if count == 0:
            return True
        if not self._make_room(count):
            return False
        self._anonymous_blocks += count
        self._assert_invariant()
        return True

    def release_anonymous(self, blocks: int) -> None:
        count = max(0, int(blocks))
        if count > self._anonymous_blocks:
            raise RuntimeError("released more anonymous KV blocks than reserved")
        self._anonymous_blocks -= count
        self._assert_invariant()

    def promote_anonymous(self, block: Any) -> bool:
        """Give one anonymous resident block a deterministic pinned identity.

        A concurrently published identical key is deliberately not folded into
        this request: vLLM may retain duplicate physical blocks for the same
        hash.  In that case the caller keeps its block anonymous until request
        completion, preserving the physical occupancy conservatively.
        """

        if block in self._blocks:
            return False
        if self._anonymous_blocks <= 0:
            raise RuntimeError("cannot promote KV without anonymous occupancy")
        self._anonymous_blocks -= 1
        self._blocks[block] = 1
        self._blocks.move_to_end(block)
        self._assert_invariant()
        return True

    def publish(self, blocks: Sequence[Any]) -> None:
        if self.capacity <= 0:
            return
        for block in blocks:
            if block in self._blocks:
                self._blocks.move_to_end(block)
            else:
                if not self._make_room(1):
                    raise RuntimeError(
                        "cannot publish a KV cache block while the pool is "
                        "fully pinned"
                    )
                self._blocks[block] = 0
        self._assert_invariant()

    def __contains__(self, block: Any) -> bool:
        return block in self._blocks

    def __len__(self) -> int:
        return len(self._blocks)


@dataclass
class _DecodeState:
    full_input: float
    generated: float
    remaining: float
    result_cache_blocks: Tuple[Any, ...] = ()
    resident_cache_blocks: Tuple[Any, ...] = ()
    resident_anonymous_blocks: int = 0
    # Production GenZ pricing consumes only resident aggregates.  Those
    # engines advance decode with a shared logical epoch and materialize these
    # scalar fields only at an API/debug boundary or exit.
    lazy_start_epoch: int = 0
    lazy_finish_epoch: int = 0
    lazy_total_tokens: float = 0.0


@dataclass
class _PrefillState:
    rid: int
    context: float
    processed: float
    remaining: float
    out: float
    full_input: float
    cache_blocks: Tuple[Any, ...] = ()
    result_cache_blocks: Tuple[Any, ...] = ()
    resident_cache_blocks: Tuple[Any, ...] = ()
    resident_anonymous_blocks: int = 0


@dataclass(frozen=True)
class _EngineStepPlan:
    """Resident work frozen when one non-preemptive scheduler step launches."""

    service_s: float
    # Decode residents cannot leave or change while their engine step is in
    # flight; later admissions are appended to the insertion-ordered mapping.
    # The frozen prefix length is therefore sufficient to identify exactly the
    # same work without allocating one ``(rid, token)`` tuple per resident on
    # every generated token.  ``decode_work`` below remains as a lazy debug /
    # test view of the old internal representation.
    decode_count: int
    decode_tail_rid: Optional[int]
    decode_epoch: int
    lazy_decode: bool
    _engine: "_Engine" = field(repr=False, compare=False)
    prefill_work: Tuple[Tuple[int, float], ...]

    @property
    def decode_work(self) -> Tuple[Tuple[int, float], ...]:
        """Materialize the legacy frozen-work view only when inspected."""
        work: List[Tuple[int, float]] = []
        for index, (rid, state) in enumerate(self._engine.decoding.items()):
            if index >= self.decode_count:
                break
            remaining = self._engine._decode_remaining_at(
                state, self.decode_epoch,
            ) if self.lazy_decode else state.remaining
            work.append((rid, min(1.0, max(0.0, remaining))))
        return tuple(work)


class _Engine:
    """Resident engine state advanced by discrete scheduler-step plans.

    ``rates``/``advance`` remain only for direct synthetic compatibility.
    Production :func:`simulate` uses ``build_step`` and ``complete_step`` via
    the concrete-card GPU arbiter.
    """

    def __init__(self, spec: SimEngine, cards: List[_Card]):
        if spec.role not in {"unified", "prefill", "decode"}:
            raise ValueError(
                f"SimEngine {spec.name!r} has invalid role {spec.role!r}"
            )
        self.spec = spec
        self.cards = [cards[c] for c in spec.cards]
        self.decoding: Dict[int, _DecodeState] = {}
        # Production decode pricing depends only on these ordered sums.  Keep
        # them alongside the insertion-ordered resident mapping so the common
        # one-token step is O(1) to price.  Set changes are uncommon and use an
        # ordered rescan to retain bit-for-bit equivalence with Python ``sum``.
        self._decode_input_sum = 0.0
        self._decode_generated_sum = 0.0
        disable_lazy_decode = os.environ.get(
            "CM_DES_DISABLE_LAZY_DECODE", "",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._lazy_decode = bool(
            getattr(spec.decode_step_s, "_rag_cm_decode_lazy_safe", False)
            and not disable_lazy_decode
        )
        self._decode_epoch = 0
        self._decode_inflight_count = 0
        self._decode_finish_buckets: Dict[int, List[int]] = {}
        self.prefill: deque[_PrefillState] = deque()
        # Calls beyond the engine's resident depth wait in strict FCFS order.
        self.waiting: deque[Tuple[int, "SimCall"]] = deque()
        self.prefix_cache = _BlockLRU(spec.prefix_cache_capacity_blocks)
        self.active = False                    # occupies its cards?
        self.fill_area = 0.0                   # ∫ residents dt (for mean fill)
        self.busy_area = 0.0                   # ∫ 1{residents>0} dt

    def _append_decode(self, rid: int, state: _DecodeState) -> None:
        """Append one resident and preserve the production aggregate order."""
        if self._lazy_decode:
            total = max(1.0, float(state.remaining))
            if not total.is_integer() or not float(state.full_input).is_integer():
                raise ValueError(
                    "lazy production decode requires integral token counts"
                )
            start_epoch = self._decode_epoch + (
                1 if self._decode_inflight_count > 0 else 0
            )
            finish_epoch = start_epoch + int(total)
            state.lazy_start_epoch = start_epoch
            state.lazy_finish_epoch = finish_epoch
            state.lazy_total_tokens = total
            self._decode_finish_buckets.setdefault(finish_epoch, []).append(rid)
        self.decoding[rid] = state
        self._decode_input_sum += float(state.full_input)
        self._decode_generated_sum += float(state.generated)

    def _recompute_decode_aggregates(self) -> None:
        """Rebuild aggregates in the historical resident iteration order."""
        self._decode_input_sum = sum(
            float(state.full_input) for state in self.decoding.values()
        )
        self._decode_generated_sum = sum(
            self._decode_generated_at(state, self._decode_epoch)
            if self._lazy_decode else float(state.generated)
            for state in self.decoding.values()
        )

    @staticmethod
    def _decode_generated_at(state: _DecodeState, epoch: int) -> float:
        if state.lazy_total_tokens <= 0.0:
            return float(state.generated)
        return min(
            float(state.lazy_total_tokens),
            float(max(0, int(epoch) - int(state.lazy_start_epoch))),
        )

    @classmethod
    def _decode_remaining_at(cls, state: _DecodeState, epoch: int) -> float:
        if state.lazy_total_tokens <= 0.0:
            return float(state.remaining)
        return max(
            0.0,
            float(state.lazy_total_tokens) - cls._decode_generated_at(state, epoch),
        )

    def _materialize_and_disable_lazy_decode(self) -> None:
        """Restore eager per-resident state for the public fluid API."""
        if not self._lazy_decode:
            return
        if self._decode_inflight_count:
            raise RuntimeError("cannot enter fluid decode during an in-flight step")
        for state in self.decoding.values():
            state.generated = self._decode_generated_at(
                state, self._decode_epoch,
            )
            state.remaining = self._decode_remaining_at(
                state, self._decode_epoch,
            )
            state.lazy_start_epoch = 0
            state.lazy_finish_epoch = 0
            state.lazy_total_tokens = 0.0
        self._lazy_decode = False
        self._decode_finish_buckets.clear()

    def residents(self) -> int:
        return len(self.decoding) + len(self.prefill)

    def has_room(self) -> bool:
        return self.residents() < self.spec.depth

    def _resident_blocks_for_call(
        self,
        *,
        full_input: float,
        output_tokens: float,
    ) -> int:
        """Physical KV blocks reserved by one resident logical call.

        The output length is already part of the immutable quality trace.  We
        reserve its eventual KV at engine admission so the simulator never
        relies on vLLM preemption to overcommit memory.  A prefill-only PD
        engine owns prompt KV only; unified and decode engines own prompt plus
        generated KV.
        """

        if self.prefix_cache.capacity <= 0:
            return 0
        tokens = max(0.0, float(full_input))
        if self.spec.role != "prefill":
            tokens += max(1.0, float(output_tokens))
        block_tokens = max(1, int(self.spec.prefix_cache_block_tokens))
        return int(math.ceil(tokens / block_tokens)) if tokens > 0 else 0

    def _acquire_resident_kv(
        self,
        call: SimCall,
        *,
        full_input: float,
    ) -> Optional[Tuple[Tuple[Any, ...], int]]:
        """Acquire shared-pool KV for ``call`` or return ``None`` to queue it."""

        if self.prefix_cache.capacity <= 0:
            return (), 0
        block_tokens = max(1, int(self.spec.prefix_cache_block_tokens))
        # vLLM requires at least one prompt token to be recomputed.  At an
        # exact block boundary that makes the final complete block ineligible
        # for a prefix hit; a partial tail naturally supplies that token.
        max_hit_blocks = max(
            0,
            (max(0, int(full_input)) - 1) // block_tokens,
        )
        candidates = tuple(call.cache_blocks[:max_hit_blocks])
        resident_blocks = self._resident_blocks_for_call(
            full_input=full_input,
            output_tokens=float(call.out),
        )
        if resident_blocks > self.prefix_cache.capacity:
            raise ValueError(
                f"engine {self.spec.name!r} call requires {resident_blocks} "
                "KV blocks but the replica capacity is only "
                f"{self.prefix_cache.capacity}"
            )
        pinned = self.prefix_cache.acquire_longest_prefix(candidates)
        anonymous = resident_blocks - len(pinned)
        if anonymous < 0:
            self.prefix_cache.release(pinned)
            raise RuntimeError("prefix hits exceed resident KV block demand")
        if not self.prefix_cache.reserve_anonymous(anonymous):
            self.prefix_cache.release(pinned)
            return None
        return pinned, anonymous

    def _promote_prefill_cache(self, state: _PrefillState) -> None:
        """Publish newly computed full prompt blocks while they stay pinned."""

        if self.prefix_cache.capacity <= 0 or not state.cache_blocks:
            return
        pinned = list(state.resident_cache_blocks)
        pinned_set = set(pinned)
        for block in state.cache_blocks:
            if block in pinned_set:
                continue
            if self.prefix_cache.promote_anonymous(block):
                state.resident_anonymous_blocks -= 1
                pinned.append(block)
                pinned_set.add(block)
        if state.resident_anonymous_blocks < 0:
            raise RuntimeError("prompt promotion exceeded resident KV blocks")
        state.resident_cache_blocks = tuple(pinned)

    def _retire_resident_kv(
        self,
        *,
        pinned: Sequence[Any],
        anonymous: int,
        publish: Sequence[Any],
    ) -> None:
        """Release one resident and retain its reusable deterministic blocks."""

        if self.prefix_cache.capacity <= 0:
            return
        self.prefix_cache.release(pinned)
        self.prefix_cache.release_anonymous(anonymous)
        self.prefix_cache.publish(publish)

    def share(self) -> float:
        if not self.cards:
            return 1.0
        if any(card.stage_busy for card in self.cards):
            return 0.0
        return min(1.0 / c.n_active for c in self.cards if c.n_active) if any(
            c.n_active for c in self.cards) else 1.0

    def _prefill_items(self, token_budget: float) -> List[SimPrefillItem]:
        """Pack the next FCFS prompt slices without erasing boundaries."""
        remaining_budget = max(0.0, float(token_budget))
        items: List[SimPrefillItem] = []
        for state in self.prefill:
            if remaining_budget <= 1e-12:
                break
            take = min(state.remaining, remaining_budget)
            if take <= 0:
                continue
            items.append(SimPrefillItem(
                context_tokens=state.context + state.processed,
                new_tokens=take,
            ))
            remaining_budget -= take
        return items

    def _component_step_s(
        self,
        decode_items: Sequence[SimDecodeItem],
        prefill_items: Sequence[SimPrefillItem],
    ) -> float:
        """Price only the component work in one scheduler cycle."""
        if self.spec.decode_step_s is None and self.spec.prefill_batch_s is None:
            return float(self.spec.step_s(
                len(decode_items),
                sum(item.new_tokens for item in prefill_items),
            ))

        step = 0.0
        if decode_items:
            if self.spec.decode_step_s is not None:
                step += float(self.spec.decode_step_s(decode_items))
            else:
                step += float(self.spec.step_s(len(decode_items), 0.0))
        if prefill_items:
            if self.spec.prefill_batch_s is not None:
                step += float(self.spec.prefill_batch_s(prefill_items))
            else:
                step += float(self.spec.step_s(
                    0,
                    sum(item.new_tokens for item in prefill_items),
                ))
        return step

    def _cycle_corrected_service_s(
        self,
        component_service_s: float,
        *,
        completed_prefills: Optional[int] = None,
    ) -> float:
        """Apply the calibrated overhead at its canonical engine boundary.

        A mixed vLLM cycle has one active-service interval.  Prefill and decode
        analytical component work are summed first.  The stage-curve batch is
        the complete resident engine population, matching the calibration
        contract; it is not the subset selected by this scheduler cycle's
        token budget.
        Direct-P is not an engine-cycle residual: its acquisition parent window
        ends when a logical prefill cohort completes.  Its dedicated callback
        therefore distributes one frontier overhead as completed logical
        prefills divided by the resident batch.  The callback is attached by
        the production builder; public/synthetic hooks without it retain
        their historical behavior.
        """

        if self.spec.role == "prefill":
            frontier_callbacks = []
            for hook in (self.spec.decode_step_s, self.spec.prefill_batch_s):
                callback = getattr(
                    hook,
                    "_rag_cm_prefill_frontier_corrected_service_s",
                    None,
                )
                if callback is not None and callback not in frontier_callbacks:
                    frontier_callbacks.append(callback)
            if frontier_callbacks:
                if len(frontier_callbacks) != 1:
                    raise RuntimeError(
                        f"engine {self.spec.name!r} has inconsistent "
                        "prefill-frontier hooks"
                    )
                if completed_prefills is None:
                    raise RuntimeError(
                        f"engine {self.spec.name!r} requires an exact "
                        "completed-prefill count"
                    )
                if (
                    isinstance(completed_prefills, bool)
                    or not isinstance(completed_prefills, int)
                    or completed_prefills < 0
                ):
                    raise RuntimeError(
                        f"engine {self.spec.name!r} has invalid completed "
                        f"prefills {completed_prefills!r}"
                    )
                resident_batch = self.residents()
                if resident_batch <= 0:
                    raise RuntimeError(
                        f"engine {self.spec.name!r} cannot price a prefill "
                        "frontier without a resident population"
                    )
                if completed_prefills > resident_batch:
                    raise RuntimeError(
                        f"engine {self.spec.name!r} completed more prefills "
                        "than its resident population"
                    )
                return float(frontier_callbacks[0](
                    resident_batch,
                    completed_prefills,
                    float(component_service_s),
                ))

        callbacks = []
        for hook in (self.spec.decode_step_s, self.spec.prefill_batch_s):
            callback = getattr(
                hook,
                "_rag_cm_cycle_corrected_service_s",
                None,
            )
            if callback is not None and callback not in callbacks:
                callbacks.append(callback)
        if not callbacks:
            return float(component_service_s)
        if len(callbacks) != 1:
            raise RuntimeError(
                f"engine {self.spec.name!r} has inconsistent "
                "cycle-overhead hooks"
            )
        resident_batch = self.residents()
        if resident_batch <= 0:
            raise RuntimeError(
                f"engine {self.spec.name!r} cannot price a scheduler cycle "
                "without a resident population"
            )
        return float(callbacks[0](
            resident_batch,
            float(component_service_s),
        ))

    def _step_s(
        self,
        decode_items: Sequence[SimDecodeItem],
        prefill_items: Sequence[SimPrefillItem],
        *,
        completed_prefills: Optional[int] = None,
    ) -> float:
        """Price one scheduler step at its calibrated stage boundary."""

        component_service_s = self._component_step_s(
            decode_items,
            prefill_items,
        )
        return self._cycle_corrected_service_s(
            component_service_s,
            completed_prefills=completed_prefills,
        )

    def build_step(self) -> Optional[_EngineStepPlan]:
        """Freeze one vLLM scheduler step from the current resident set."""
        if not self.decoding and not self.prefill:
            return None
        if self._lazy_decode and self._decode_inflight_count:
            raise RuntimeError(
                f"engine {self.spec.name!r} already has an in-flight step"
            )
        aggregate_decode_price = getattr(
            self.spec.decode_step_s,
            "_rag_cm_decode_aggregate_s",
            None,
        )
        decode_count = len(self.decoding)
        decode_tail_rid = next(reversed(self.decoding)) if decode_count else None
        decode_items: List[SimDecodeItem] = []
        if aggregate_decode_price is None:
            # Public/synthetic hooks retain the complete per-sequence
            # contract.  Only the production closure below advertises that
            # its price depends on batch size and the maintained sums.
            decode_items = [
                SimDecodeItem(
                    input_tokens=state.full_input,
                    generated_tokens=state.generated,
                    remaining_tokens=state.remaining,
                )
                for state in self.decoding.values()
            ]
        remaining_budget = max(
            0.0, float(self.spec.budget - decode_count),
        )
        prefill_work_list: List[Tuple[int, float]] = []
        prefill_items: List[SimPrefillItem] = []
        completed_prefills = 0
        for state in self.prefill:
            if remaining_budget <= 1e-12:
                break
            take = min(state.remaining, remaining_budget)
            if take <= 0:
                continue
            prefill_work_list.append((state.rid, take))
            prefill_items.append(SimPrefillItem(
                context_tokens=state.context + state.processed,
                new_tokens=take,
            ))
            if self.spec.role == "prefill" and state.remaining - take <= 1e-6:
                completed_prefills += 1
            remaining_budget -= take
        if aggregate_decode_price is not None and decode_count:
            component_service_s = float(aggregate_decode_price(
                decode_count,
                self._decode_input_sum,
                self._decode_generated_sum,
            ))
            if prefill_items:
                # Preserve the original decode-then-prefill addition order.
                component_service_s += self._component_step_s((), prefill_items)
            service_s = self._cycle_corrected_service_s(
                component_service_s,
                completed_prefills=completed_prefills,
            )
        else:
            service_s = self._step_s(
                decode_items,
                prefill_items,
                completed_prefills=completed_prefills,
            )
        if service_s <= 0 or not math.isfinite(service_s):
            raise ValueError(
                f"engine {self.spec.name!r} produced non-positive/non-finite "
                f"step service "
                f"{service_s!r}"
            )
        plan = _EngineStepPlan(
            service_s=float(service_s),
            decode_count=decode_count,
            decode_tail_rid=decode_tail_rid,
            decode_epoch=self._decode_epoch,
            lazy_decode=self._lazy_decode,
            _engine=self,
            prefill_work=tuple(prefill_work_list),
        )
        if self._lazy_decode:
            self._decode_inflight_count = decode_count
        return plan

    def complete_step(self, plan: _EngineStepPlan) -> List[int]:
        """Atomically commit the work frozen by :meth:`build_step`."""
        if plan._engine is not self:
            raise RuntimeError(
                f"engine {self.spec.name!r} received another engine's step"
            )
        if plan.lazy_decode:
            return self._complete_lazy_step(plan)
        done: List[int] = []
        generated_increment = 0.0
        processed_decode = 0
        processed_tail_rid: Optional[int] = None
        for index, (rid, state) in enumerate(self.decoding.items()):
            if index >= plan.decode_count:
                break
            tokens = min(1.0, max(0.0, state.remaining))
            state.generated += tokens
            state.remaining -= tokens
            generated_increment += tokens
            processed_decode += 1
            processed_tail_rid = rid
            if state.remaining <= 1e-6:
                self._retire_resident_kv(
                    pinned=state.resident_cache_blocks,
                    anonymous=state.resident_anonymous_blocks,
                    publish=(
                        state.result_cache_blocks
                        if self.spec.role == "unified" else ()
                    ),
                )
                done.append(rid)

        if (
            processed_decode != plan.decode_count
            or processed_tail_rid != plan.decode_tail_rid
        ):
            raise RuntimeError(
                f"engine {self.spec.name!r} lost decode residents"
            )
        for rid in done:
            del self.decoding[rid]
        if done:
            # A removal changes the ordered reduction surface; rebuild exactly
            # as the previous per-step scan would have done.
            self._recompute_decode_aggregates()
        else:
            # Without removals every frozen resident generated the same exact
            # unit token.  Production state starts at integer token counts, so
            # this is numerically identical to rescanning the updated states.
            self._decode_generated_sum += generated_increment

        # ``build_step`` always packs an FCFS prefix of this deque.  Consume
        # that same prefix directly instead of indexing the whole queue and
        # rebuilding it after every scheduler step.  A partially processed
        # head necessarily exhausts the token budget, so it is the last item
        # in the frozen plan.
        for work_index, (rid, tokens) in enumerate(plan.prefill_work):
            if not self.prefill or self.prefill[0].rid != rid:
                raise RuntimeError(
                    f"engine {self.spec.name!r} lost prefill resident {rid}"
                )
            state = self.prefill[0]
            state.processed += tokens
            state.remaining -= tokens
            if state.remaining <= 1e-6:
                self.prefill.popleft()
                self._promote_prefill_cache(state)
                if self.spec.role == "prefill":
                    # A PD prefill replica owns only the prompt KV it computed.
                    # Generated KV lives on the decode replica and is not a
                    # cache hit for a later prefill request.
                    self._retire_resident_kv(
                        pinned=state.resident_cache_blocks,
                        anonymous=state.resident_anonymous_blocks,
                        publish=state.cache_blocks,
                    )
                    done.append(state.rid)
                else:
                    self._append_decode(state.rid, _DecodeState(
                        full_input=state.full_input,
                        generated=0.0,
                        remaining=max(1.0, float(state.out)),
                        result_cache_blocks=state.result_cache_blocks,
                        resident_cache_blocks=state.resident_cache_blocks,
                        resident_anonymous_blocks=(
                            state.resident_anonymous_blocks
                        ),
                    ))
            elif work_index + 1 != len(plan.prefill_work):
                raise RuntimeError(
                    f"engine {self.spec.name!r} advanced beyond a partial "
                    "FCFS prefill resident"
                )
        return done

    def _complete_lazy_step(self, plan: _EngineStepPlan) -> List[int]:
        """Commit a production decode step without touching every resident."""
        if not self._lazy_decode:
            raise RuntimeError("lazy decode plan completed after mode change")
        if (
            plan.decode_epoch != self._decode_epoch
            or plan.decode_count != self._decode_inflight_count
            or plan.decode_count > len(self.decoding)
            or (
                plan.decode_count > 0
                and plan.decode_tail_rid not in self.decoding
            )
        ):
            raise RuntimeError(
                f"engine {self.spec.name!r} lost lazy decode residents"
            )

        done: List[int] = []
        if plan.decode_count:
            self._decode_epoch += 1
            # Integral quality-trace outputs make every participating sequence
            # contribute exactly one token.  Suffix admissions made while the
            # step was in flight start at this new epoch and are not charged.
            self._decode_generated_sum += float(plan.decode_count)
            finishing = self._decode_finish_buckets.pop(
                self._decode_epoch, (),
            )
            for rid in finishing:
                state = self.decoding.get(rid)
                if state is None:
                    raise RuntimeError(
                        f"engine {self.spec.name!r} lost decode resident {rid}"
                    )
                state.generated = float(state.lazy_total_tokens)
                state.remaining = 0.0
                self._retire_resident_kv(
                    pinned=state.resident_cache_blocks,
                    anonymous=state.resident_anonymous_blocks,
                    publish=(
                        state.result_cache_blocks
                        if self.spec.role == "unified" else ()
                    ),
                )
                # All maintained quantities are integer-valued on this marked
                # production path, so subtraction is bit-identical to the old
                # ordered resident scan while avoiding that O(batch) pass.
                self._decode_input_sum -= float(state.full_input)
                self._decode_generated_sum -= float(state.lazy_total_tokens)
                del self.decoding[rid]
                done.append(rid)
        self._decode_inflight_count = 0

        # Prefill remains an explicit FCFS token-budget walk.  Keep this logic
        # aligned with the eager path; completed prompts join decode at the new
        # epoch and therefore first participate in the following step.
        for work_index, (rid, tokens) in enumerate(plan.prefill_work):
            if not self.prefill or self.prefill[0].rid != rid:
                raise RuntimeError(
                    f"engine {self.spec.name!r} lost prefill resident {rid}"
                )
            state = self.prefill[0]
            state.processed += tokens
            state.remaining -= tokens
            if state.remaining <= 1e-6:
                self.prefill.popleft()
                self._promote_prefill_cache(state)
                if self.spec.role == "prefill":
                    self._retire_resident_kv(
                        pinned=state.resident_cache_blocks,
                        anonymous=state.resident_anonymous_blocks,
                        publish=state.cache_blocks,
                    )
                    done.append(state.rid)
                else:
                    self._append_decode(state.rid, _DecodeState(
                        full_input=state.full_input,
                        generated=0.0,
                        remaining=max(1.0, float(state.out)),
                        result_cache_blocks=state.result_cache_blocks,
                        resident_cache_blocks=state.resident_cache_blocks,
                        resident_anonymous_blocks=(
                            state.resident_anonymous_blocks
                        ),
                    ))
            elif work_index + 1 != len(plan.prefill_work):
                raise RuntimeError(
                    f"engine {self.spec.name!r} advanced beyond a partial "
                    "FCFS prefill resident"
                )
        return done

    def account(self, dt: float) -> None:
        """Accumulate occupancy while the external GPU arbiter advances."""
        residents = self.residents()
        self.fill_area += residents * dt
        if residents:
            self.busy_area += dt

    def rates(self) -> Tuple[float, float]:
        """(decode tokens/s per decoding request, prefill tokens/s total).

        Two-lane step physics (engines.engine_step_price): one engine step
        serves each of the ``d`` decoding sequences one token and a
        ``c``-token slice of the ACTUAL prefill backlog, budget-limited to
        the chunked-prefill headroom (budget − d). The old model charged
        ALL tokens at a decode-derived aggregate rate and billed the FULL
        budget whenever any prefill was queued — an artificial ceiling
        (0046-A: 51 qps vs measured 136)."""
        # ``rates``/``advance`` are the legacy continuous-time public surface.
        # Production DES never calls them; materialize once if a direct caller
        # mixes that API with a production-marked engine.
        self._materialize_and_disable_lazy_decode()
        d = len(self.decoding)
        if d == 0 and not self.prefill:
            return 0.0, 0.0
        backlog = sum(state.remaining for state in self.prefill)
        prefill_items = self._prefill_items(
            min(backlog, max(0.0, float(self.spec.budget - d))),
        )
        c = sum(item.new_tokens for item in prefill_items)
        decode_items = [
            SimDecodeItem(
                input_tokens=state.full_input,
                generated_tokens=state.generated,
                remaining_tokens=state.remaining,
            )
            for state in self.decoding.values()
        ]
        step = self._step_s(decode_items, prefill_items)
        if step <= 0:
            return 0.0, 0.0
        scale = self.share()
        eff_step = step / scale if scale > 0 else math.inf
        if not math.isfinite(eff_step) or eff_step <= 0:
            return 0.0, 0.0
        dec_rate = 1.0 / eff_step               # 1 token per step per request
        pf_rate = c / eff_step
        return dec_rate, pf_rate

    def next_change_dt(self) -> float:
        dec_rate, pf_rate = self.rates()
        dt = math.inf
        if self.decoding and dec_rate > 0:
            dt = min(
                dt,
                min(state.remaining for state in self.decoding.values())
                / dec_rate,
            )
        if self.prefill and pf_rate > 0:
            dt = min(dt, self.prefill[0].remaining / pf_rate)
        return dt

    def advance(self, dt: float) -> List[int]:
        """Advance ``dt``; return request ids whose current ENGINE phase ended.

        For a unified/decode engine this is the end of the model call.  For a
        prefill-only PD engine it is the point at which the KV handoff becomes
        eligible; :func:`simulate` routes it through the link before decode.
        """
        n = self.residents()
        self.fill_area += n * dt
        if n:
            self.busy_area += dt
        dec_rate, pf_rate = self.rates()
        done: List[int] = []
        if self.decoding and dec_rate > 0:
            generated = dec_rate * dt
            for rid, state in list(self.decoding.items()):
                state.remaining -= generated
                state.generated += generated
                # 1e-6 tokens: coarse enough that the dt needed to finish
                # never underflows double resolution (a 2.8e-8-token tail
                # once needed dt=2.3e-16 s at t=4 -> now+dt==now spin)
                if state.remaining <= 1e-6:
                    self._retire_resident_kv(
                        pinned=state.resident_cache_blocks,
                        anonymous=state.resident_anonymous_blocks,
                        publish=(
                            state.result_cache_blocks
                            if self.spec.role == "unified" else ()
                        ),
                    )
                    del self.decoding[rid]
                    done.append(rid)
        if self.prefill and pf_rate > 0:
            # Budget-limited BATCH advance: the whole FCFS queue progresses
            # at the shared chunk rate, popping completed heads — vLLM packs
            # many short prompts into one chunk step; the old single-head
            # advance serialized them (~6 ms per call = a hard qps ceiling).
            tokens = pf_rate * dt
            while self.prefill and tokens > 1e-12:
                head = self.prefill[0]
                take = min(tokens, head.remaining)
                head.remaining -= take
                head.processed += take
                tokens -= take
                if head.remaining <= 1e-6:
                    self.prefill.popleft()
                    rid = int(head.rid)
                    self._promote_prefill_cache(head)
                    if self.spec.role == "prefill":
                        self._retire_resident_kv(
                            pinned=head.resident_cache_blocks,
                            anonymous=head.resident_anonymous_blocks,
                            publish=head.cache_blocks,
                        )
                        done.append(rid)
                    else:
                        self._append_decode(rid, _DecodeState(
                            full_input=head.full_input,
                            generated=0.0,
                            remaining=max(1.0, float(head.out)),
                            result_cache_blocks=head.result_cache_blocks,
                            resident_cache_blocks=head.resident_cache_blocks,
                            resident_anonymous_blocks=(
                                head.resident_anonymous_blocks
                            ),
                        ))
        if self.decoding:
            # Compatibility ``advance`` can apply fractional progress; restore
            # the exact resident-order aggregate before a later discrete step.
            self._recompute_decode_aggregates()
        else:
            self._decode_input_sum = 0.0
            self._decode_generated_sum = 0.0
        return done

    def admit(self, rid: int, call: SimCall) -> bool:
        """Admit work and return True when a zero-token prefill ended now."""
        if self.residents() < self.spec.depth:
            immediate = self._place(rid, call)
            if immediate is None:
                self.waiting.append((rid, call))
                immediate = False
        else:
            self.waiting.append((rid, call))
            immediate = False
        self._sync_cards()
        return immediate

    def _place(self, rid: int, call: SimCall) -> Optional[bool]:
        full_input = float(
            call.full_in if call.full_in is not None else call.eff_in
        )
        computed = max(0.0, float(call.eff_in))
        resident_kv = self._acquire_resident_kv(
            call,
            full_input=full_input,
        )
        if resident_kv is None:
            return None
        pinned_blocks, anonymous_blocks = resident_kv
        if call.cache_blocks and self.prefix_cache.capacity > 0:
            hit_tokens = len(pinned_blocks) * max(
                1, int(self.spec.prefix_cache_block_tokens),
            )
            # Only complete blocks hit.  The uncached partial tail is always
            # recomputed, even when every full prompt block is resident.
            computed = max(0.0, full_input - float(hit_tokens))
        context = max(0.0, full_input - computed)
        if self.spec.role == "prefill":
            if computed <= 0:
                self._retire_resident_kv(
                    pinned=pinned_blocks,
                    anonymous=anonymous_blocks,
                    publish=call.cache_blocks,
                )
                return True
            self.prefill.append(_PrefillState(
                rid=rid,
                context=context,
                processed=0.0,
                remaining=computed,
                out=float(call.out),
                full_input=full_input,
                cache_blocks=call.cache_blocks,
                result_cache_blocks=call.result_cache_blocks,
                resident_cache_blocks=pinned_blocks,
                resident_anonymous_blocks=anonymous_blocks,
            ))
        elif self.spec.role == "decode":
            self._append_decode(rid, _DecodeState(
                full_input=full_input,
                generated=0.0,
                remaining=max(1.0, float(call.out)),
                result_cache_blocks=(),
                resident_cache_blocks=pinned_blocks,
                resident_anonymous_blocks=anonymous_blocks,
            ))
        elif computed > 0:
            self.prefill.append(_PrefillState(
                rid=rid,
                context=context,
                processed=0.0,
                remaining=computed,
                out=float(call.out),
                full_input=full_input,
                cache_blocks=call.cache_blocks,
                result_cache_blocks=call.result_cache_blocks,
                resident_cache_blocks=pinned_blocks,
                resident_anonymous_blocks=anonymous_blocks,
            ))
        else:
            self._append_decode(rid, _DecodeState(
                full_input=full_input,
                generated=0.0,
                remaining=max(1.0, float(call.out)),
                result_cache_blocks=call.result_cache_blocks,
                resident_cache_blocks=pinned_blocks,
                resident_anonymous_blocks=anonymous_blocks,
            ))
        return False

    def promote_and_sync(self) -> List[int]:
        immediate: List[int] = []
        while self.waiting and self.residents() < self.spec.depth:
            rid, call = self.waiting[0]
            placed = self._place(rid, call)
            if placed is None:
                break
            self.waiting.popleft()
            if placed:
                immediate.append(rid)
        self._sync_cards()
        return immediate

    def _sync_cards(self) -> None:
        want = self.residents() > 0
        if want and not self.active:
            for c in self.cards:
                c.n_active += 1
            self.active = True
        elif not want and self.active:
            for c in self.cards:
                c.n_active -= 1
            self.active = False


class _StageWorker:
    """Single-worker count/timeout batcher (the measured stage service)."""

    def __init__(self, spec: SimStage, cards: List[_Card]):
        self.spec = spec
        card_ids = spec.cards or ((spec.card,) if spec.card is not None else ())
        self.cards = [cards[i] for i in card_ids]
        self.card = self.cards[0] if self.cards else None
        self.queue: deque[Tuple[int, SimStageItem, float]] = deque()
        # The production worker caps batching delay from the oldest enqueue.
        # Keep that timestamp on each queued row so time spent behind a busy
        # worker also consumes the partial-batch timeout budget.
        self.batch_deadline_t = math.inf
        self.busy = False
        self.rows = 0
        self.batches = 0

    def gpu_burst_s(
        self, items: Sequence[SimStageItem],
    ) -> float:
        if self.spec.batch_gpu_service_s is not None:
            return max(0.0, float(self.spec.batch_gpu_service_s(items)))
        batch = len(items)
        if self.spec.gpu_service_s is not None:
            return min(self.spec.gpu_service_s(batch),
                       self.service_s(items))
        return self.service_s(items) * self.spec.gpu_share

    def service_s(self, items: Sequence[SimStageItem]) -> float:
        if self.spec.batch_service_s is not None:
            return max(0.0, float(self.spec.batch_service_s(items)))
        return max(0.0, float(self.spec.service_s(len(items))))

    def host_s(self, items: Sequence[SimStageItem]) -> float:
        return max(0.0, self.service_s(items) - self.gpu_burst_s(items))

    def ops(self, items: Sequence[SimStageItem]) -> Tuple[SimStageOp, ...]:
        if self.spec.batch_ops is not None:
            ops: List[SimStageOp] = []
            for op in self.spec.batch_ops(items):
                service_s = float(op.service_s)
                if not math.isfinite(service_s) or service_s < 0.0:
                    raise ValueError(
                        f"stage operator {op.name!r} has invalid service "
                        f"{service_s!r}"
                    )
                if service_s > 0.0:
                    ops.append(op)
            return tuple(ops)
        gpu = self.gpu_burst_s(items)
        host = self.host_s(items)
        ops: List[SimStageOp] = []
        if gpu > 0.0 and self.cards:
            ops.append(SimStageOp(
                resource="gpu", service_s=gpu,
                name=f"{self.spec.name}.gpu",
            ))
        else:
            host += gpu
        if host > 0.0:
            ops.append(SimStageOp(
                resource="cpu", service_s=host,
                host_width=self.spec.host_width,
                name=f"{self.spec.name}.cpu",
            ))
        return tuple(ops)


@dataclass
class _StageBatchExecution:
    worker: _StageWorker
    entries: List[Tuple[int, SimStageItem]]
    ops: Tuple[SimStageOp, ...]
    op_index: int = 0

    @property
    def current(self) -> Optional[SimStageOp]:
        if self.op_index >= len(self.ops):
            return None
        return self.ops[self.op_index]


@dataclass
class _LogicalEngineGate:
    """FCFS client admission state for one complete logical engine call."""

    capacity: int
    active: int = 0
    waiting: deque[Tuple[int, SimCall]] = field(default_factory=deque)


@dataclass(frozen=True)
class _PendingPdDecodeAdmission:
    rid: int
    call: SimCall
    source_engine_idx: int


@dataclass
class _PdRoleAdmissionGate:
    """FCFS client admission state for one physical P or D engine role."""

    capacity: int
    active: int = 0
    prefill_waiting: deque[Tuple[int, SimCall]] = field(default_factory=deque)
    decode_waiting: deque[_PendingPdDecodeAdmission] = field(
        default_factory=deque,
    )


def _stage_kick_time(
    *,
    now: float,
    first_enqueue_t: float,
    queued: int,
    batch_cap: int,
    timeout_s: float,
) -> float:
    """Earliest legal batch-start time for one dynamic-batch worker.

    A full batch starts immediately. A partial backlog starts when the oldest
    queued row reaches its maximum batching delay; if it already aged past the
    deadline behind a busy worker, it starts as soon as the worker is free.
    """
    if queued <= 0:
        return math.inf
    if queued >= max(1, int(batch_cap)):
        return float(now)
    return max(
        float(now),
        float(first_enqueue_t) + max(0.0, float(timeout_s)),
    )


def _retrieval_chunk_sizes(
    items: Sequence[SimStageItem],
    *,
    chunk_cap: int,
) -> Tuple[int, ...]:
    """Runtime-faithful flattened sub-query chunks for one worker batch.

    The retrieval worker receives user-row items, flattens every row's ``K``
    expanded sub-queries, then feeds the resulting list through the deployed
    embedding/request cap.  In particular, a partial user batch of ``b`` rows
    is *not* ``K`` independent batches of size ``b``: when ``b*K <= cap`` it is
    one backend chunk of size ``b*K``.
    """
    return retrieval_chunk_sizes(
        tuple(item.multiplicity for item in items),
        chunk_cap=chunk_cap,
    )


def _apc_block_keys(
    segments: Sequence[Tuple[Any, int]],
    total_tokens: int,
    block_tokens: int,
) -> Tuple[Any, ...]:
    """Build exact full-block identities from an abstract token lineage.

    The quality trace intentionally carries counts, not token IDs.  Builders
    therefore call this helper only while the prompt is structurally known to
    be deterministic.  Segment identities encode the query/engine/call that
    introduced each token range; reusing the same immutable query template on
    a later closed-loop turnover reconstructs the same block keys without
    consulting runtime observations.
    """
    block_tokens = max(1, int(block_tokens))
    total_tokens = max(0, int(total_tokens))
    full_blocks = total_tokens // block_tokens
    if full_blocks <= 0:
        return ()
    normalized: List[Tuple[Any, int, int]] = []
    cursor = 0
    for identity, raw_length in segments:
        length = max(0, int(raw_length))
        if length <= 0:
            continue
        normalized.append((identity, cursor, cursor + length))
        cursor += length
    if cursor < full_blocks * block_tokens:
        raise ValueError(
            "APC token lineage is shorter than the requested full blocks"
        )

    keys: List[Any] = []
    for block_index in range(full_blocks):
        start = block_index * block_tokens
        end = start + block_tokens
        pieces = []
        for identity, seg_start, seg_end in normalized:
            overlap_start = max(start, seg_start)
            overlap_end = min(end, seg_end)
            if overlap_start < overlap_end:
                pieces.append((
                    identity,
                    overlap_start - seg_start,
                    overlap_end - overlap_start,
                ))
            if seg_start >= end:
                break
        keys.append(("apc_block_v1", block_tokens, tuple(pieces)))
    return tuple(keys)


def _dense_decoder_weight_bytes(model_cfg: Any, dtype_bytes: float) -> Optional[float]:
    """Mechanically count dense decoder GEMM weights from model structure.

    This mirrors the matrices GenZ prices: one token embedding/output matrix,
    Q/K/V/O projections, and ``num_ffi`` input projections plus one FFN output
    projection per layer.  Unsupported MoE/Mamba shapes return ``None`` so APC
    is disabled instead of receiving an invented memory budget.
    """
    if bool(getattr(model_cfg, "is_moe", False)) or bool(
        getattr(model_cfg, "is_mamba", False)
    ):
        return None
    try:
        hidden = int(model_cfg.hidden_size)
        layers = int(model_cfg.num_decoder_layers)
        heads = int(model_cfg.num_attention_heads)
        kv_heads = int(getattr(model_cfg, "num_key_value_heads", None) or heads)
        head_dim = int(
            getattr(model_cfg, "head_dim", None) or (hidden // heads)
        )
        intermediate = int(model_cfg.intermediate_size)
        num_ffi = int(getattr(model_cfg, "num_ffi", 2) or 2)
        vocab = int(model_cfg.vocab_size)
        dtype_bytes = float(dtype_bytes)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return None
    if min(
        hidden, layers, heads, kv_heads, head_dim, intermediate, num_ffi,
        vocab,
    ) <= 0 or dtype_bytes <= 0:
        return None
    attention_params = (
        2 * hidden * hidden
        + 2 * hidden * kv_heads * head_dim
    )
    ffn_params = (num_ffi + 1) * hidden * intermediate
    embedding_params = vocab * hidden
    return float(
        (embedding_params + layers * (attention_params + ffn_params))
        * dtype_bytes
    )


def _kv_tp_shard_degree(tp: int, kv_head_count: Optional[int]) -> int:
    """Number of distinct KV shards across TP ranks.

    Attention heads can be replicated when TP exceeds the model's KV-head
    count (GQA/MQA).  Weight matrices still shard across all TP ranks, but KV
    cache memory and a PD handoff only gain ``min(tp, kv_heads)`` distinct
    shards.
    """
    width = max(1, int(tp))
    if kv_head_count is None:
        return width
    try:
        heads = max(1, int(kv_head_count))
    except (TypeError, ValueError, OverflowError):
        return width
    return min(width, heads)


def _prefix_cache_capacity_blocks(
    *,
    gpu_memory_bytes: float,
    gpu_memory_utilization: float,
    model_weight_bytes: float,
    kv_bytes_per_token: float,
    tp: int,
    pp: int,
    block_tokens: int,
    kv_head_count: Optional[int] = None,
) -> int:
    """Derive one replica's APC capacity from declared memory and KV shape.

    Weights and KV are sharded across the concrete TP x PP ranks.  Every rank
    has the same declared memory budget, so the tight per-rank capacity is the
    replica capacity.  No activation reserve or hit-rate coefficient is guessed;
    invalid/unresolved public facts produce a zero-capacity cache.
    """
    try:
        memory = float(gpu_memory_bytes)
        utilization = float(gpu_memory_utilization)
        weights = float(model_weight_bytes)
        kv_per_token = float(kv_bytes_per_token)
        tp_width = max(1, int(tp))
        pp_width = max(1, int(pp))
        weight_width = tp_width * pp_width
        kv_width = _kv_tp_shard_degree(tp_width, kv_head_count) * pp_width
        block_tokens = max(1, int(block_tokens))
    except (TypeError, ValueError, OverflowError):
        return 0
    if not all(math.isfinite(value) for value in (
        memory, utilization, weights, kv_per_token,
    )):
        return 0
    if memory <= 0 or not (0 < utilization <= 1) or weights < 0 or kv_per_token <= 0:
        return 0
    per_rank_budget = memory * utilization
    per_rank_weights = weights / weight_width
    per_rank_block = (kv_per_token / kv_width) * block_tokens
    available = per_rank_budget - per_rank_weights
    if available <= 0 or per_rank_block <= 0:
        return 0
    return max(0, int(available // per_rank_block))


def simulate(
    stages: Sequence[SimStage],
    engines: Sequence[SimEngine],
    queries: Sequence[List[SimRound]],
    *,
    n_cards: int,
    population: int,
    warmup_turnovers: int = _SIM_WARMUP_TURNOVERS,
    window_turnovers: int = _SIM_WINDOW_TURNOVERS,
    n_cpu_cores: Optional[float] = None,
) -> SimResult:
    cards = [_Card() for _ in range(n_cards)]
    # The box's logical cores are one PS resource for light-stage host
    # segments. LLM engine progress is governed solely by its GenZ step price.
    cpu = _Cpu(n_cpu_cores)
    engs = [_Engine(e, cards) for e in engines]
    workers = [_StageWorker(s, cards) for s in stages]

    # Request state: each of `population` slots cycles queries round-robin.
    # progression: (query idx, round idx, position in round)
    heap: List[Tuple[float, int, str, Any]] = []
    seq = 0
    last_pushed_seq: Optional[int] = None
    pending_recompute_seq: Optional[int] = None
    pending_recompute_t: Optional[float] = None
    pending_stage_kicks: Dict[int, Tuple[int, float]] = {}
    omitted_events_by_seq: Dict[int, int] = {}

    def push(t: float, kind: str, payload: Any) -> int:
        nonlocal seq, last_pushed_seq
        event_seq = seq
        heapq.heappush(heap, (t, event_seq, kind, payload))
        last_pushed_seq = event_seq
        seq += 1
        return event_seq

    def request_engine_recompute(t: float) -> None:
        """Queue an engine rescan without adjacent same-time no-op events.

        Only an engine-recompute event that is both still pending and the most
        recently enqueued event may be coalesced.  Therefore no other event can
        occur between the retained and omitted rescans in heap sequence.  The
        retained rescan observes exactly the same engine state; non-adjacent
        rescans remain separate to preserve all same-time causal ordering.
        """
        nonlocal pending_recompute_seq, pending_recompute_t
        if (
            pending_recompute_seq is not None
            and pending_recompute_seq == last_pushed_seq
            and pending_recompute_t == t
        ):
            omitted_events_by_seq[pending_recompute_seq] = (
                omitted_events_by_seq.get(pending_recompute_seq, 0) + 1
            )
            return
        pending_recompute_seq = push(t, "engine_recompute", None)
        pending_recompute_t = t

    def request_stage_kick(t: float, worker: _StageWorker) -> None:
        """Drop only provably adjacent duplicate kicks for one worker."""
        worker_key = id(worker)
        pending = pending_stage_kicks.get(worker_key)
        if (
            pending is not None
            and pending[0] == last_pushed_seq
            and pending[1] == t
        ):
            omitted_events_by_seq[pending[0]] = (
                omitted_events_by_seq.get(pending[0], 0) + 1
            )
            return
        pending_stage_kicks[worker_key] = (
            push(t, "stage_kick", worker), t,
        )

    req_round: List[int] = [0] * population
    query_rotation_stride = _coprime_trace_stride(len(queries))
    req_qidx: List[int] = [i % len(queries) for i in range(population)]
    req_pos: List[int] = [0] * population
    req_admit_t: List[float] = [0.0] * population
    pending_calls: Dict[int, Tuple[SimCall, int]] = {}
    boundary_progress: Dict[Tuple[int, int, int], int] = {}
    latencies: List[float] = []
    completions = 0
    window_completions = 0
    warmup_target = warmup_turnovers * population
    window_target = window_turnovers * population
    window_start_t: Optional[float] = None
    now = 0.0

    # One FCFS bandwidth lane per transfer resource. Payload transmission
    # reserves the lane; propagation latency delays completion without holding
    # bandwidth, allowing pipelined handoffs on the same physical link.
    link_free_at: Dict[str, float] = {}
    transfer_elapsed: Dict[str, List[float]] = {}
    transfer_wait: Dict[str, List[float]] = {}
    pool_cursor: Dict[Tuple[int, ...], int] = {}
    logical_engine_gates: Dict[str, _LogicalEngineGate] = {}
    held_engine_gate: Dict[int, str] = {}
    pd_prefill_gates: Dict[str, _PdRoleAdmissionGate] = {}
    pd_decode_gates: Dict[str, _PdRoleAdmissionGate] = {}
    held_pd_prefill_gate: Dict[int, str] = {}
    held_pd_decode_gate: Dict[int, str] = {}

    def _choose_engine(pool: Tuple[int, ...]) -> int:
        if not pool:
            raise RuntimeError("empty simulator engine replica pool")
        if len(pool) == 1:
            # No load comparison or round-robin state can affect this choice.
            return pool[0]
        loads = {
            idx: (engs[idx].residents() + len(engs[idx].waiting))
            / max(1, engs[idx].spec.depth)
            for idx in pool
        }
        best = min(loads.values())
        tied = [idx for idx in pool if loads[idx] == best]
        cursor = pool_cursor.get(pool, 0)
        chosen = tied[cursor % len(tied)]
        pool_cursor[pool] = cursor + 1
        return chosen

    def _gate_for(call: SimCall) -> Optional[_LogicalEngineGate]:
        gate_name = call.admission_gate
        if gate_name is None:
            if call.admission_capacity is not None:
                raise ValueError(
                    "SimCall admission_capacity requires admission_gate"
                )
            return None
        gate_name = str(gate_name)
        if not gate_name:
            raise ValueError("SimCall admission_gate must be non-empty")
        try:
            capacity = int(call.admission_capacity)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"logical engine gate {gate_name!r} has invalid capacity "
                f"{call.admission_capacity!r}"
            ) from exc
        if capacity <= 0:
            raise ValueError(
                f"logical engine gate {gate_name!r} has non-positive "
                f"capacity {capacity}"
            )
        gate = logical_engine_gates.get(gate_name)
        if gate is None:
            gate = _LogicalEngineGate(capacity=capacity)
            logical_engine_gates[gate_name] = gate
        elif gate.capacity != capacity:
            raise ValueError(
                f"logical engine gate {gate_name!r} has inconsistent "
                f"capacities {gate.capacity} and {capacity}"
            )
        return gate

    def _admit_engine_call(rid: int, call: SimCall, t: float) -> None:
        pool = call.engine_pool or (call.engine,)
        engine_idx = _choose_engine(pool)
        e = engs[engine_idx]
        pending_calls[rid] = (call, engine_idx)
        immediate = e.admit(rid, call)
        request_engine_recompute(t)
        if immediate:
            handle_engine_phase_done(e, rid, t)

    def _acquire_engine_gate(rid: int, call: SimCall, t: float) -> None:
        gate = _gate_for(call)
        if gate is None:
            _admit_engine_call(rid, call, t)
            return
        gate_name = str(call.admission_gate)
        if rid in held_engine_gate:
            raise RuntimeError(
                f"request {rid} attempted nested logical-engine admission"
            )
        if gate.active < gate.capacity:
            gate.active += 1
            held_engine_gate[rid] = gate_name
            _admit_engine_call(rid, call, t)
        else:
            gate.waiting.append((rid, call))

    def _release_engine_gate(rid: int, t: float) -> None:
        gate_name = held_engine_gate.pop(rid, None)
        if gate_name is None:
            return
        gate = logical_engine_gates[gate_name]
        gate.active -= 1
        if gate.active < 0:
            raise RuntimeError(
                f"logical engine gate {gate_name!r} released below zero"
            )
        # Promote in arrival order.  Normally exactly one permit opens, but
        # the loop keeps the state correct if a future policy changes capacity
        # between construction and simulation.
        while gate.waiting and gate.active < gate.capacity:
            next_rid, next_call = gate.waiting.popleft()
            gate.active += 1
            held_engine_gate[next_rid] = gate_name
            _admit_engine_call(next_rid, next_call, t)

    def _uses_pd_role_admission(call: SimCall) -> bool:
        values = (
            call.prefill_admission_gate,
            call.prefill_admission_capacity,
            call.decode_admission_gate,
            call.decode_admission_capacity,
        )
        if all(value is None for value in values):
            return False
        if any(value is None for value in values):
            raise ValueError(
                "PD role admission requires both gate and capacity for "
                "prefill and decode"
            )
        if call.admission_gate is not None or call.admission_capacity is not None:
            raise ValueError(
                "PD role admission cannot also use the unified logical gate"
            )
        if call.decode_engine is None and not call.decode_engine_pool:
            raise ValueError("PD role admission requires a decode engine")
        return True

    def _pd_role_gate_for(
        call: SimCall,
        role: str,
    ) -> Tuple[str, _PdRoleAdmissionGate]:
        if role not in {"prefill", "decode"}:
            raise ValueError(f"invalid PD admission role {role!r}")
        name_raw = getattr(call, f"{role}_admission_gate")
        capacity_raw = getattr(call, f"{role}_admission_capacity")
        name = str(name_raw)
        if not name:
            raise ValueError(f"PD {role} admission gate must be non-empty")
        try:
            capacity = int(capacity_raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"PD {role} gate {name!r} has invalid capacity "
                f"{capacity_raw!r}"
            ) from exc
        if capacity <= 0:
            raise ValueError(
                f"PD {role} gate {name!r} has non-positive capacity {capacity}"
            )
        gates = pd_prefill_gates if role == "prefill" else pd_decode_gates
        gate = gates.get(name)
        if gate is None:
            gate = _PdRoleAdmissionGate(capacity=capacity)
            gates[name] = gate
        elif gate.capacity != capacity:
            raise ValueError(
                f"PD {role} gate {name!r} has inconsistent capacities "
                f"{gate.capacity} and {capacity}"
            )
        return name, gate

    def _acquire_pd_prefill_gate(rid: int, call: SimCall, t: float) -> None:
        # Validate the complete two-role contract before changing either gate.
        if not _uses_pd_role_admission(call):
            raise ValueError("missing PD role admission contract")
        _pd_role_gate_for(call, "decode")
        gate_name, gate = _pd_role_gate_for(call, "prefill")
        if rid in held_pd_prefill_gate or rid in held_pd_decode_gate:
            raise RuntimeError(
                f"request {rid} attempted nested PD prefill admission"
            )
        if gate.active < gate.capacity:
            gate.active += 1
            held_pd_prefill_gate[rid] = gate_name
            _admit_engine_call(rid, call, t)
        else:
            gate.prefill_waiting.append((rid, call))

    def _release_pd_prefill_gate(rid: int, t: float) -> None:
        gate_name = held_pd_prefill_gate.pop(rid, None)
        if gate_name is None:
            raise RuntimeError(
                f"request {rid} acquired D without a held P admission"
            )
        gate = pd_prefill_gates[gate_name]
        gate.active -= 1
        if gate.active < 0:
            raise RuntimeError(
                f"PD prefill gate {gate_name!r} released below zero"
            )
        while gate.prefill_waiting and gate.active < gate.capacity:
            next_rid, next_call = gate.prefill_waiting.popleft()
            gate.active += 1
            held_pd_prefill_gate[next_rid] = gate_name
            _admit_engine_call(next_rid, next_call, t)

    def _grant_pd_decode_gate(
        pending: _PendingPdDecodeAdmission,
        gate_name: str,
        gate: _PdRoleAdmissionGate,
        t: float,
    ) -> None:
        rid = pending.rid
        if rid in held_pd_decode_gate:
            raise RuntimeError(
                f"request {rid} already holds a PD decode admission"
            )
        gate.active += 1
        held_pd_decode_gate[rid] = gate_name
        # Match VllmPdPair exactly: reserve D while still holding P, then
        # release P before starting the physical NIXL transfer/decode request.
        _release_pd_prefill_gate(rid, t)
        _schedule_transfer(
            rid,
            pending.call,
            pending.source_engine_idx,
            t,
        )

    def _acquire_pd_decode_gate(
        rid: int,
        call: SimCall,
        source_engine_idx: int,
        t: float,
    ) -> None:
        gate_name, gate = _pd_role_gate_for(call, "decode")
        pending = _PendingPdDecodeAdmission(
            rid=rid,
            call=call,
            source_engine_idx=source_engine_idx,
        )
        if gate.active < gate.capacity:
            _grant_pd_decode_gate(pending, gate_name, gate, t)
        else:
            # P intentionally remains held while this FCFS D acquisition waits.
            gate.decode_waiting.append(pending)

    def _release_pd_decode_gate(rid: int, t: float) -> None:
        gate_name = held_pd_decode_gate.pop(rid, None)
        if gate_name is None:
            return
        gate = pd_decode_gates[gate_name]
        gate.active -= 1
        if gate.active < 0:
            raise RuntimeError(
                f"PD decode gate {gate_name!r} released below zero"
            )
        while gate.decode_waiting and gate.active < gate.capacity:
            pending = gate.decode_waiting.popleft()
            _grant_pd_decode_gate(pending, gate_name, gate, t)

    def route(rid: int, t: float) -> None:
        """Send request rid to its next hop at time t."""
        q = queries[req_qidx[rid]]
        rnd = q[req_round[rid]]
        pos = req_pos[rid]
        boundary_key = (rid, req_round[rid], pos)
        transfers = rnd.boundary_transfers.get(pos, ())
        transfer_index = boundary_progress.get(boundary_key, 0)
        if transfer_index < len(transfers):
            transfer = transfers[transfer_index]
            _queue_link_transfer(
                transfer,
                t,
                "boundary_transfer_done",
                (rid, boundary_key, transfer_index),
            )
            return
        # Once the last transfer has completed this key is never consulted
        # again.  Retiring it here bounds the map by in-flight boundaries and
        # avoids the former all-population scan at every round completion.
        if transfers:
            boundary_progress.pop(boundary_key, None)
        if pos < len(rnd.stage_idxs):
            w = workers[rnd.stage_idxs[pos]]
            item = (
                rnd.stage_items[pos]
                if pos < len(rnd.stage_items)
                else SimStageItem(stage=w.spec.name)
            )
            w.queue.append((rid, replace(item, rid=rid), t))
            if len(w.queue) == 1:
                if not w.busy:
                    w.batch_deadline_t = _stage_kick_time(
                        now=t,
                        first_enqueue_t=t,
                        queued=len(w.queue),
                        batch_cap=w.spec.batch_cap,
                        timeout_s=w.spec.timeout_s,
                    )
                    request_stage_kick(w.batch_deadline_t, w)
            if len(w.queue) >= w.spec.batch_cap and not w.busy:
                request_stage_kick(t, w)
        elif rnd.call is not None and pos == len(rnd.stage_idxs):
            if _uses_pd_role_admission(rnd.call):
                _acquire_pd_prefill_gate(rid, rnd.call, t)
            else:
                _acquire_engine_gate(rid, rnd.call, t)
        else:
            finish_round(rid, t)

    def finish_round(rid: int, t: float) -> None:
        nonlocal completions, window_completions, window_start_t
        q = queries[req_qidx[rid]]
        req_round[rid] += 1
        req_pos[rid] = 0
        if req_round[rid] < len(q):
            route(rid, t)
            return
        # query complete — recycle the slot (closed loop)
        completions += 1
        if window_start_t is None and completions >= warmup_target:
            window_start_t = t
            for e in engs:
                e.fill_area = 0.0
                e.busy_area = 0.0
            for w in workers:
                w.rows = 0
                w.batches = 0
            transfer_elapsed.clear()
            transfer_wait.clear()
        elif window_start_t is not None:
            window_completions += 1
            latencies.append(t - req_admit_t[rid])
        req_round[rid] = 0
        req_qidx[rid] = (
            req_qidx[rid] + query_rotation_stride
        ) % len(queries)
        req_admit_t[rid] = t
        route(rid, t)

    def _admit_decode(
        rid: int,
        call: SimCall,
        t: float,
        decode_engine_idx: Optional[int] = None,
    ) -> None:
        decode_pool = call.decode_engine_pool or (
            (call.decode_engine,) if call.decode_engine is not None else ()
        )
        if decode_engine_idx is None and decode_pool:
            decode_engine_idx = _choose_engine(decode_pool)
        if decode_engine_idx is None:
            raise RuntimeError("PD prefill completion has no decode engine")
        dec = engs[decode_engine_idx]
        decode_call = SimCall(
            engine=decode_engine_idx,
            eff_in=0,
            out=call.out,
            full_in=call.full_in,
        )
        immediate = dec.admit(rid, decode_call)
        if immediate:  # a decode engine always creates resident decode work
            raise RuntimeError("decode engine completed a call at admission")
        request_engine_recompute(t)

    def _schedule_transfer(
        rid: int,
        call: SimCall,
        source_engine_idx: int,
        t: float,
    ) -> None:
        decode_pool = call.decode_engine_pool or (
            (call.decode_engine,) if call.decode_engine is not None else ()
        )
        if not decode_pool:
            raise RuntimeError("PD prefill completion has no decode engine pool")
        decode_engine_idx = _choose_engine(decode_pool)
        transfer = (
            call.transfers.get((source_engine_idx, decode_engine_idx))
            or call.transfer
        )
        if transfer is None:
            _admit_decode(rid, call, t, decode_engine_idx)
            return

        _queue_link_transfer(
            transfer,
            t,
            "transfer_done",
            (rid, call, decode_engine_idx),
        )

    def _queue_link_transfer(
        transfer: SimTransfer,
        t: float,
        done_kind: str,
        done_payload: Any,
    ) -> None:
        """Reserve one declared link lane and publish its completion event."""
        if transfer.bandwidth_bytes_s <= 0 and transfer.payload_bytes > 0:
            raise RuntimeError(
                f"transfer {transfer.name!r} has non-positive bandwidth"
            )
        start = max(t, link_free_at.get(transfer.resource, t))
        tx_s = (
            float(transfer.payload_bytes) / float(transfer.bandwidth_bytes_s)
            if transfer.payload_bytes > 0 else 0.0
        )
        link_free_at[transfer.resource] = start + tx_s
        done_t = start + tx_s + max(0.0, float(transfer.latency_s))
        transfer_elapsed.setdefault(transfer.name, []).append(done_t - t)
        transfer_wait.setdefault(transfer.name, []).append(start - t)
        push(done_t, done_kind, done_payload)

    def handle_engine_phase_done(e: _Engine, rid: int, t: float) -> None:
        pending = pending_calls.get(rid)
        if pending is None:
            raise RuntimeError(f"engine completed unknown request {rid}")
        call, source_engine_idx = pending
        if e.spec.role == "prefill":
            if rid in held_pd_prefill_gate:
                _acquire_pd_decode_gate(rid, call, source_engine_idx, t)
            else:
                # Backwards-compatible synthetic PD calls without the measured
                # role-admission contract retain the legacy direct handoff.
                _schedule_transfer(rid, call, source_engine_idx, t)
        else:
            pending_calls.pop(rid, None)
            push(t + e.spec.call_overhead_s, "call_done", rid)

    def _complete_engine_events(done: List[Tuple[_Engine, int]], t: float) -> None:
        promoted_engines: set[int] = set()
        for e, rid in done:
            # ``complete_step`` removed every resident in ``done`` atomically.
            # The first promotion for an engine fills all newly available
            # slots, making later calls for the same completed batch no-ops.
            # Retain the original ordering: promote, publish the first done
            # rid, then publish any zero-prefill promotions.
            engine_key = id(e)
            if engine_key not in promoted_engines:
                immediate = e.promote_and_sync()
                promoted_engines.add(engine_key)
            else:
                immediate = ()
            handle_engine_phase_done(e, rid, t)
            for promoted_rid in immediate:
                handle_engine_phase_done(e, promoted_rid, t)

    # Engine scheduler steps and light-stage CUDA kernels share one concrete-
    # card FCFS arbiter.  Both are atomic non-preemptive jobs; CPU stage ops
    # retain the box-wide processor-sharing model.
    gpu_arbiter = _GpuArbiter(n_cards)
    engine_gpu_ops: Dict[int, _GpuOp] = {}
    cpu_bursts: List[List[Any]] = []  # [execution, remaining, width]

    def _schedule_started(ops: Sequence[_GpuOp]) -> None:
        for op in ops:
            if op.started_at is None:
                raise RuntimeError(f"GPU op {op.name!r} has no start time")
            owner_kind = op.owner[0]
            if owner_kind == "engine":
                engine_idx = op.owner[1]
                engine = engs[engine_idx]
                # The ticket is created when the engine first has resident
                # work, but its batch is frozen only when every concrete card
                # grants that ticket. Requests admitted while it waited are
                # therefore part of the same continuous-batching step.
                plan = engine.build_step()
                if plan is None:
                    raise RuntimeError(
                        f"granted engine {engine.spec.name!r} has no residents"
                    )
                op.owner = ("engine", engine_idx, plan)
                op.service_s = plan.service_s
            elif owner_kind == "stage":
                for card_idx_ in op.cards:
                    cards[card_idx_].stage_busy += 1
            push(op.started_at + op.service_s, "gpu_op_done", op)

    def _request_engine_step(engine_idx: int, t: float) -> None:
        if engine_idx in engine_gpu_ops:
            return
        engine = engs[engine_idx]
        if engine.residents() <= 0:
            return
        op, started = gpu_arbiter.submit(
            cards=engine.spec.cards,
            # Service and resident shape are frozen at grant, not ticket
            # submission, so queued arrivals can join this scheduler step.
            service_s=None,
            owner=("engine", engine_idx, None),
            now=t,
            name=f"{engine.spec.name}.scheduler_step",
        )
        engine_gpu_ops[engine_idx] = op
        _schedule_started(started)

    def _request_all_engine_steps(t: float) -> None:
        for engine_idx in range(len(engs)):
            _request_engine_step(engine_idx, t)

    def _start_stage_op(execution: _StageBatchExecution, t: float) -> None:
        op = execution.current
        if op is None:
            push(t, "stage_done", execution)
            return
        if op.resource == "gpu" and execution.worker.cards:
            card_ids = execution.worker.spec.cards or (
                (execution.worker.spec.card,)
                if execution.worker.spec.card is not None else ()
            )
            _submitted, started = gpu_arbiter.submit(
                cards=card_ids,
                service_s=float(op.service_s),
                owner=("stage", execution),
                now=t,
                name=op.name or execution.worker.spec.name,
            )
            _schedule_started(started)
            return
        if op.resource not in {"gpu", "cpu"}:
            raise ValueError(
                f"stage operator {op.name!r} has invalid resource={op.resource!r}"
            )
        if not math.isfinite(float(op.service_s)) or float(op.service_s) <= 0.0:
            raise ValueError(
                f"stage operator {op.name!r} has non-positive/non-finite "
                f"service {op.service_s!r}"
            )
        width = max(1.0, float(op.host_width))
        cpu.width_active += width
        cpu_bursts.append([execution, float(op.service_s), width])

    def start_batch(w: _StageWorker, t: float) -> None:
        if w.busy or not w.queue:
            return
        if (len(w.queue) < w.spec.batch_cap
                and t < w.batch_deadline_t - 1e-12):
            return
        queued_entries = [
            w.queue.popleft()
            for _ in range(min(w.spec.batch_cap, len(w.queue)))
        ]
        entries = [(rid, item) for rid, item, _enqueue_t in queued_entries]
        w.batch_deadline_t = math.inf
        w.busy = True
        w.rows += len(entries)
        w.batches += 1
        items = [item for _rid, item in entries]
        execution = _StageBatchExecution(
            worker=w,
            entries=entries,
            ops=w.ops(items),
        )
        _start_stage_op(execution, t)

    def advance_cpu_bursts(dt: float, t: float) -> None:
        if not cpu_bursts:
            return
        rate = 1.0 / cpu.slow()
        for burst in list(cpu_bursts):
            burst[1] -= rate * dt
            if burst[1] <= 1e-9:
                cpu_bursts.remove(burst)
                cpu.width_active -= burst[2]
                execution = burst[0]
                execution.op_index += 1
                _start_stage_op(execution, t)

    def next_cpu_burst_dt() -> float:
        if not cpu_bursts:
            return math.inf
        slow = cpu.slow()
        return min(burst[1] * slow for burst in cpu_bursts)

    # prime the loop
    for rid in range(population):
        route(rid, 0.0)

    events = 0
    while window_completions < window_target and events < _SIM_MAX_EVENTS:
        events += 1
        if not heap and not cpu_bursts:
            raise RuntimeError(
                "closed-loop simulator deadlock: no timed GPU/link/stage "
                "event and no active CPU work"
            )
        # GPU completions are timed heap events; the only continuously shared
        # work left is the light-stage CPU lane.
        t_event = heap[0][0] if heap else math.inf
        t_state = now + next_cpu_burst_dt()
        if t_event <= t_state:
            t, event_seq, kind, payload = heapq.heappop(heap)
            # The retained event is processed exactly where the first member
            # of its adjacent same-time run appeared.  Its omitted successors
            # are proven no-ops, but the reference loop still charged one
            # safety-budget iteration for each of them.
            events += omitted_events_by_seq.pop(event_seq, 0)
            dt = max(0.0, t - now)
            for e in engs:
                e.account(dt)
            advance_cpu_bursts(dt, t)
            now = t
            if kind == "stage_kick":
                worker_key = id(payload)
                pending = pending_stage_kicks.get(worker_key)
                if pending is not None and pending[0] == event_seq:
                    pending_stage_kicks.pop(worker_key, None)
                start_batch(payload, now)
            elif kind == "gpu_op_done":
                op = payload
                owner_kind = op.owner[0]
                released = gpu_arbiter.release(op, now=now)
                if owner_kind == "stage":
                    execution = op.owner[1]
                    for card_idx_ in op.cards:
                        cards[card_idx_].stage_busy -= 1
                        if cards[card_idx_].stage_busy < 0:
                            raise RuntimeError("negative stage GPU ownership")
                    execution.op_index += 1
                    _start_stage_op(execution, now)
                elif owner_kind == "engine":
                    engine_idx, plan = op.owner[1], op.owner[2]
                    if engine_gpu_ops.pop(engine_idx, None) is not op:
                        raise RuntimeError(
                            f"unknown completed engine GPU op {op.name!r}"
                        )
                    engine = engs[engine_idx]
                    done = engine.complete_step(plan)
                    _complete_engine_events(
                        [(engine, rid) for rid in done], now,
                    )
                    # Submit the next resident scheduler step only after the
                    # completed job releases its cards. Any stage already in
                    # the FCFS queues therefore stays ahead of this ticket.
                    _request_engine_step(engine_idx, now)
                else:
                    raise RuntimeError(f"unknown GPU op owner {owner_kind!r}")
                _schedule_started(released)
            elif kind == "stage_done":
                execution = payload
                w = execution.worker
                w.busy = False
                for rid, _item in execution.entries:
                    req_pos[rid] += 1
                    route(rid, now)
                if w.queue:
                    w.batch_deadline_t = _stage_kick_time(
                        now=now,
                        first_enqueue_t=float(w.queue[0][2]),
                        queued=len(w.queue),
                        batch_cap=w.spec.batch_cap,
                        timeout_s=w.spec.timeout_s,
                    )
                    request_stage_kick(w.batch_deadline_t, w)
            elif kind == "call_done":
                rid = payload
                if rid in held_pd_decode_gate:
                    # D is strict and remains held through NIXL fetch, decode,
                    # and the terminal frontend completion boundary.
                    _release_pd_decode_gate(rid, now)
                else:
                    # Unified serving retains its complete-request gate.
                    _release_engine_gate(rid, now)
                finish_round(rid, now)
            elif kind == "transfer_done":
                rid, call, decode_engine_idx = payload
                _admit_decode(rid, call, now, decode_engine_idx)
            elif kind == "boundary_transfer_done":
                rid, boundary_key, transfer_index = payload
                boundary_progress[boundary_key] = transfer_index + 1
                route(rid, now)
            elif kind == "engine_recompute":
                if pending_recompute_seq == event_seq:
                    pending_recompute_seq = None
                    pending_recompute_t = None
                _request_all_engine_steps(now)
        else:
            if not t_state > now:
                # float-resolution floor: time must advance (see the
                # 1e-6-token completion threshold above)
                t_state = math.nextafter(now, math.inf)
            dt = max(0.0, t_state - now)
            for e in engs:
                e.account(dt)
            advance_cpu_bursts(dt, t_state)
            now = t_state

    window_s = (now - window_start_t) if window_start_t is not None else now
    qps = window_completions / window_s if window_s > 0 else 0.0
    return SimResult(
        qps=qps,
        latency_mean_s=(sum(latencies) / len(latencies)) if latencies else 0.0,
        engine_fill_mean={
            e.spec.name: (e.fill_area / window_s / max(1, e.spec.depth))
            if window_s > 0 else 0.0 for e in engs},
        stage_batch_mean={
            w.spec.name: (w.rows / w.batches if w.batches else 0.0)
            for w in workers},
        window_s=window_s,
        completions=window_completions,
        population=population,
        transfer_latency_mean={
            name: (sum(xs) / len(xs) if xs else 0.0)
            for name, xs in transfer_elapsed.items()
        },
        transfer_queue_wait_mean={
            name: (sum(xs) / len(xs) if xs else 0.0)
            for name, xs in transfer_wait.items()
        },
        converged=window_completions >= window_target,
    )


# ---------------------------------------------------------------------------
# Builder: CM artifacts -> sim inputs
# ---------------------------------------------------------------------------

_TRACE_STAGE_TO_SIM = {
    "semantic_retrieval_encode": "semantic_retrieval",
    "semantic_retrieval_vectorsearch": "semantic_retrieval",
    "lexical_retrieval": "lexical_retrieval",
    "passage_reranker": "passage_reranker",
    "passage_compressor": "passage_compressor",
}


@dataclass(frozen=True)
class _LLMRoleDeployment:
    devices: Tuple[str, ...]
    replica_devices: Tuple[Tuple[str, ...], ...]
    tp: int
    pp: int

    @property
    def replicas(self) -> int:
        return len(self.replica_devices)


def _mapping_value(value: Any) -> Optional[dict]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        for loader in (json.loads, ast.literal_eval):
            try:
                out = loader(value)
            except Exception:  # noqa: BLE001 - try the other persisted form
                continue
            if isinstance(out, dict):
                return out
    return None


def _row_value(row: Optional[Mapping[str, Any]], key: str, default=None):
    if row is None:
        return default
    try:
        value = row[key]
    except (KeyError, TypeError, IndexError):
        return default
    # pandas' scalar NA cannot be used as a boolean. Containers are never NA.
    if not isinstance(value, (dict, list, tuple, str)):
        try:
            if value != value:  # NaN
                return default
        except Exception:  # noqa: BLE001
            pass
    return value


def _candidate_dynamic_timeout_s(
    pipeline_config: Mapping[str, Any],
    candidate_row: Optional[Mapping[str, Any]],
) -> float:
    """Resolve one candidate's public light-stage batching timeout."""
    timeout_raw = _row_value(candidate_row, "dynamic_batch_timeout_s")
    if timeout_raw is None:
        system = pipeline_config.get("system") or {}
        batching = system.get("batching") or {}
        timeout_raw = batching.get("dynamic_timeout_s", _BATCH_TIMEOUT_S)
    return max(0.0, float(timeout_raw))


_LATENCY_CACHE_KEY_UNSET = object()


def _resolved_candidate_layout(
    pipeline_config: dict,
    candidate_row: Optional[Mapping[str, Any]],
) -> Optional[dict]:
    """Consume RAGO's materialized layout; never allocate placement here."""
    if candidate_row is not None:
        raw = _row_value(candidate_row, "candidate_system_layout")
        if raw is not None:
            try:
                from rag_stack.cost_model.candidate_layout import (
                    candidate_system_layout_from_row,
                )
            except ImportError:
                candidate_system_layout_from_row = None
            if candidate_system_layout_from_row is not None:
                # The owner validates schema/version and required sections.
                return candidate_system_layout_from_row(candidate_row)
            # Minimal CSV fallback for environments loading this module without
            # the candidate-layout adapter installed.
            layout = _mapping_value(raw)
            if layout is None:
                raise ValueError("candidate_system_layout must be a dict/JSON mapping")
            return layout
    system = pipeline_config.get("system") or {}
    return _mapping_value(system.get("layout"))


def _parse_groups(value: Any) -> List[List[str]]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:  # noqa: BLE001
            return []
    if not isinstance(value, list):
        return []
    out: List[List[str]] = []
    for group in value:
        if isinstance(group, dict):
            stages = group.get("performance_stages") or []
        else:
            stages = group
        if isinstance(stages, (list, tuple)):
            out.append([str(stage) for stage in stages])
    return out


def _layout_stage_devices(layout: Optional[dict], stage: str) -> Tuple[str, ...]:
    if not layout:
        return ()
    stages = layout.get("performance_stages") or {}
    info = stages.get(stage) if isinstance(stages, dict) else None
    devices = info.get("devices") if isinstance(info, dict) else None
    if not isinstance(devices, (list, tuple)):
        return ()
    return tuple(str(d) for d in devices if d)


def _role_deployment(
    layout: Optional[dict],
    candidate_row: Optional[Mapping[str, Any]],
    *,
    engine: str,
    stage: str,
    fallback_tp_pp: Tuple[int, int],
    fallback_devices: Tuple[str, ...] = (),
) -> _LLMRoleDeployment:
    """Read one LLM role's already-materialized TP/PP/DP deployment."""
    engines = (layout or {}).get("engines") or {}
    engine_info = engines.get(engine) if isinstance(engines, dict) else None
    engine_info = engine_info if isinstance(engine_info, dict) else {}
    if engine_info.get("pd_serving") == "disagg_pd":
        role_info = engine_info.get(stage)
        role_info = role_info if isinstance(role_info, dict) else {}
    else:
        role_info = engine_info
    stage_map = (layout or {}).get("performance_stages") or {}
    stage_info = stage_map.get(stage) if isinstance(stage_map, dict) else None
    stage_info = stage_info if isinstance(stage_info, dict) else {}

    def _pick(name: str, fallback=None):
        if role_info.get(name) is not None:
            return role_info[name]
        if stage_info.get(name) is not None:
            return stage_info[name]
        for key in (f"candidate_{name}_{stage}", f"{name}_{stage}"):
            value = _row_value(candidate_row, key)
            if value is not None:
                return value
        return fallback

    raw_devices = _pick("devices", fallback_devices)
    if isinstance(raw_devices, str):
        try:
            raw_devices = json.loads(raw_devices)
        except Exception:  # noqa: BLE001
            raw_devices = [d.strip() for d in raw_devices.split(",") if d.strip()]
    devices = tuple(str(d) for d in (raw_devices or ()) if d)
    tp = max(1, int(_pick("tp", fallback_tp_pp[0])))
    pp = max(1, int(_pick("pp", fallback_tp_pp[1])))
    width = tp * pp

    raw_replicas = _pick("replicas")
    raw_replica_devices = _pick("replica_devices")
    replica_devices: Tuple[Tuple[str, ...], ...] = ()
    if isinstance(raw_replica_devices, str):
        try:
            raw_replica_devices = json.loads(raw_replica_devices)
        except Exception:  # noqa: BLE001
            raw_replica_devices = None
    if isinstance(raw_replica_devices, (list, tuple)):
        replica_devices = tuple(
            tuple(str(d) for d in replica if d)
            for replica in raw_replica_devices
            if isinstance(replica, (list, tuple))
        )
    replicas = max(1, int(raw_replicas or len(replica_devices) or (
        len(devices) // width if devices and len(devices) % width == 0 else 1
    )))
    if not replica_devices and devices:
        replica_devices = tuple(
            devices[i * width:(i + 1) * width] for i in range(replicas)
        )
    if devices and len(devices) != width * replicas:
        raise ValueError(
            f"{stage} devices={len(devices)} does not match "
            f"tp*pp*replicas={tp}*{pp}*{replicas}"
        )
    if replica_devices and any(len(replica) != width for replica in replica_devices):
        raise ValueError(
            f"{stage} replica_devices must contain exactly tp*pp={width} devices"
        )
    if not devices and replica_devices:
        devices = tuple(d for replica in replica_devices for d in replica)
    return _LLMRoleDeployment(
        devices=devices,
        replica_devices=replica_devices,
        tp=tp,
        pp=pp,
    )


def kv_transfer_for_devices(
    system_config: Mapping[str, Any],
    source_devices: Sequence[str],
    target_devices: Sequence[str],
    *,
    payload_bytes: float,
    name: str = "generator_kv",
) -> SimTransfer:
    """Build the pair-aware explicit KV handoff used by PD simulation."""
    from rag_stack.cost_model.interconnect_topology import (
        effective_sharded_gpu_transfer_bytes_s,
    )

    bandwidth, latency = effective_sharded_gpu_transfer_bytes_s(
        system_config,
        source_devices,
        target_devices,
    )
    return SimTransfer(
        name=name,
        resource="gpu_gpu",
        payload_bytes=max(0.0, float(payload_bytes)),
        bandwidth_bytes_s=float(bandwidth),
        latency_s=float(latency),
    )


def _structural_start_population(
    *,
    batch_size_request: int,
    batch_size_decode: Optional[int],
    queries: Sequence[Sequence[SimRound]],
) -> int:
    """Return the smallest structural population used by saturation search.

    Client load knobs belong to the measured harness and are intentionally not
    arguments to this function.  A closed network must initially be large
    enough to fill the deployed request/decode batch and every *distinct*
    LLM admission gate.  Unified serving contributes one complete-request
    gate; disaggregated serving contributes its independently occupiable P and
    D role gates.  Repeated calls to the same engine reuse those gates.  The
    numerical search grows this lower bound when downstream queues need more
    population to reach their steady-state plateau.
    """
    base = max(
        1,
        int(batch_size_request),
        int(batch_size_decode or 0),
    )
    gate_capacities: Dict[str, int] = {}
    for query in queries:
        for round_ in query:
            call = round_.call
            if call is None:
                continue
            pd_values = (
                call.prefill_admission_gate,
                call.prefill_admission_capacity,
                call.decode_admission_gate,
                call.decode_admission_capacity,
            )
            if any(value is not None for value in pd_values):
                if any(value is None for value in pd_values):
                    raise ValueError(
                        "incomplete PD role admission contract in simulator input"
                    )
                pairs = (
                    (
                        str(call.prefill_admission_gate),
                        int(call.prefill_admission_capacity),
                    ),
                    (
                        str(call.decode_admission_gate),
                        int(call.decode_admission_capacity),
                    ),
                )
            elif (
                call.admission_gate is not None
                and call.admission_capacity is not None
            ):
                pairs = ((str(call.admission_gate), int(call.admission_capacity)),)
            else:
                continue
            for gate, capacity_raw in pairs:
                capacity = max(1, int(capacity_raw))
                gate_capacities[gate] = max(
                    capacity,
                    gate_capacities.get(gate, 0),
                )
    return max(base, sum(gate_capacities.values()))


def build_sim_inputs(
    pipeline_config_cm: dict,
    hardware,
    trace: list,
    collocation_strategy,
    *,
    trace_source_question_ids: Optional[Sequence[Any]] = None,
    batch_size_request: int,
    batch_size_decode: Optional[int] = None,
    prefix_caching: bool = True,
    candidate_row: Optional[Mapping[str, Any]] = None,
    latency=None,
    latency_cache: Optional[Dict[Any, Any]] = None,
    latency_cache_key: Any = _LATENCY_CACHE_KEY_UNSET,
    stage_batch_profiles: Any = _STAGE_BATCH_PROFILES_UNSET,
    prepared_paths: PreparedExecutionPaths | None = None,
    compiled_workload=None,
    compilation_capture: Optional[Dict[str, Any]] = None,
):
    """Assemble DES inputs from one already-materialized deployment row.

    RAGO owns placement allocation and publishes ``candidate_system_layout``;
    this adapter only consumes that contract. Fixed replays fall back to
    ``pipeline_config.system.layout``. The returned tuple is accepted directly
    by :func:`simulate`; ``None, reason`` denotes an unpriceable candidate.

    ``prepared_paths`` borrows an already-validated, read-only trace cohort.
    It does not enable the fixed compiled workload's retained profile binding.
    """
    from rag_stack.cost_model.engines import (
        build_call_latency,
        engine_step_price,
        _fixed_llm_tp_pp_from_layout,
        _INPUT_GRID as _GENZ_INPUT_GRID,
        _llm_stage_names,
        _trace_stage,
    )
    from rag_stack.cost_model.communication import (
        _find_module,
        _genz_model_for,
    )
    from rag_stack.cost_model.stage_work_split import (
        logical_cores,
    )
    from rag_stack.cost_model.stage_batch_curve import (
        COMPLETE_ENGINE_STAGE_KEYS,
        REQUIRED_COLLOCATED_ENGINE_STAGES,
        EngineCycleCurveApplicator,
        StageBatchCurveError,
        load_stage_batch_curve,
    )
    from rag_stack.cost_model.sweep_inputs import (
        _get_query_expansion_fanout,
        _get_reranker_input_fanout,
    )
    from rag_stack.cost_model.workload import resolve_execution_paths
    from rag_stack.rag_ir.normalize import with_observed_trace_components

    if compiled_workload is not None:
        if prepared_paths is not None and prepared_paths is not compiled_workload.prepared_paths:
            raise ValueError("prepared_paths conflicts with compiled_workload")
        prepared_paths = compiled_workload.prepared_paths
    prepared_paths = resolve_execution_paths(trace, prepared_paths)
    trace = prepared_paths.paths
    pipeline_config_cm = with_observed_trace_components(pipeline_config_cm, trace)

    if trace_source_question_ids is None:
        source_question_ids: Tuple[Any | None, ...] = (None,) * len(trace)
    else:
        source_question_ids = tuple(trace_source_question_ids)
        if len(source_question_ids) != len(trace):
            raise ValueError(
                "trace_source_question_ids must align one-for-one with trace "
                f"queries ({len(source_question_ids)} != {len(trace)})"
            )
        for position, source_id in enumerate(source_question_ids):
            if source_id is None or (
                isinstance(source_id, str) and not source_id.strip()
            ):
                raise ValueError(
                    "trace_source_question_ids contains no stable identity at "
                    f"position {position}"
                )
            try:
                hash(source_id)
            except TypeError as exc:
                raise TypeError(
                    "trace_source_question_ids must be hashable"
                ) from exc

    frozen_generation = None
    engine_cycle_applicator = None
    has_frozen_generation = (
        _STAGE_BATCH_RUNTIME_SELECTION_KEY
        in (pipeline_config_cm.get("_runtime") or {})
    )
    if (
        stage_batch_profiles is _STAGE_BATCH_PROFILES_UNSET
        and not has_frozen_generation
        and _rag_cm_calibration_mode(pipeline_config_cm, hardware)
        == "uncalibrated_spec"
    ):
        # Public/direct special-mode builders still need the all-or-nothing
        # fairness decision. Strict direct test helpers retain their historical
        # per-stage compatibility path; production always arrives frozen from
        # DynamicEngine/apply_closed_loop_simulation.
        pipeline_config_cm = _pipeline_with_frozen_stage_batch_selection(
            pipeline_config_cm,
            hardware,
        )
        has_frozen_generation = True
    if stage_batch_profiles is _STAGE_BATCH_PROFILES_UNSET:
        if has_frozen_generation:
            frozen_generation = _stage_batch_generation_for_pipeline(
                pipeline_config_cm,
                hardware,
            )
            stage_batch_profiles = frozen_generation.materialize_profiles()
    elif not isinstance(stage_batch_profiles, Mapping):
        raise TypeError("stage_batch_profiles must be a mapping or strict sentinel")
    elif has_frozen_generation:
        frozen_generation = _stage_batch_generation_for_pipeline(
            pipeline_config_cm,
            hardware,
        )
        frozen_profiles = frozen_generation.materialize_profiles()
        if set(frozen_profiles) != set(stage_batch_profiles):
            raise ValueError(
                "stage_batch_profiles disagree with the frozen generation"
            )
        for stage, profile in frozen_profiles.items():
            supplied = stage_batch_profiles[stage]
            if supplied.content_sha256 != profile.content_sha256:
                raise ValueError(
                    "stage_batch_profiles content disagrees with the frozen "
                    f"generation for {stage!r}"
                )
        stage_batch_profiles = frozen_profiles
    if (
        stage_batch_profiles is not _STAGE_BATCH_PROFILES_UNSET
        and stage_batch_profiles
        and frozenset(stage_batch_profiles) not in {
            frozenset(REQUIRED_COLLOCATED_ENGINE_STAGES),
            frozenset(COMPLETE_ENGINE_STAGE_KEYS),
        }
    ):
        raise ValueError(
            "runtime stage_batch_profiles must be empty, the collocated pair, "
            "or the complete six"
        )
    if frozen_generation is not None and frozen_generation.enabled:
        engine_cycle_applicator = EngineCycleCurveApplicator(
            frozen_generation
        )

    layout = _resolved_candidate_layout(pipeline_config_cm, candidate_row)
    candidate_pipeline = pipeline_config_cm
    if layout is not None:
        candidate_pipeline = deepcopy(pipeline_config_cm)
        candidate_pipeline.setdefault("system", {})["layout"] = deepcopy(layout)
    system_cfg = candidate_pipeline.get("system") or {}
    groups = _parse_groups(
        (layout or {}).get("resource_groups")
        if layout else collocation_strategy
    )
    if not groups:
        groups = _parse_groups(collocation_strategy)
    stage_group: Dict[str, int] = {
        stage: gi for gi, group in enumerate(groups) for stage in group
    }

    # Concrete card resources come exclusively from the materialized layout.
    # A legacy no-layout direct caller receives one logical card per supplied
    # collocation group; no physical device placement is guessed.
    device_names: List[str] = []

    def _remember(devices) -> None:
        for device in devices or ():
            device = str(device)
            if device and device not in device_names:
                device_names.append(device)

    _remember((layout or {}).get("available_devices"))
    _remember(((system_cfg.get("hardware") or {}).get("available_gpus")))
    for group in (layout or {}).get("resource_groups") or []:
        if isinstance(group, dict):
            _remember(group.get("devices"))
    for info in ((layout or {}).get("performance_stages") or {}).values():
        if isinstance(info, dict):
            _remember(info.get("devices"))
    for info in ((layout or {}).get("engines") or {}).values():
        if not isinstance(info, dict):
            continue
        _remember(info.get("devices"))
        _remember(d for replica in (info.get("replica_devices") or []) for d in replica)
        for record in info.values():
            if isinstance(record, dict):
                _remember(record.get("devices"))
                _remember(
                    d for replica in (record.get("replica_devices") or [])
                    for d in replica
                )
    if not device_names:
        device_names = [f"group:{gi}" for gi in range(max(1, len(groups)))]
    card_idx = {device: i for i, device in enumerate(device_names)}
    n_cards = max(1, len(card_idx))

    def _candidate_devices(stage: str) -> Tuple[str, ...]:
        devices = _layout_stage_devices(layout, stage)
        if devices:
            return devices
        raw = _row_value(candidate_row, f"candidate_devices_{stage}")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:  # noqa: BLE001
                raw = [d.strip() for d in raw.split(",") if d.strip()]
        if isinstance(raw, (list, tuple)):
            return tuple(str(d) for d in raw if d)
        gi = stage_group.get(stage)
        if layout is None and gi is not None:
            return (f"group:{gi}",)
        return ()

    def _cards(devices: Sequence[str]) -> Tuple[int, ...]:
        missing = [str(d) for d in devices if str(d) not in card_idx]
        if missing:
            raise ValueError(
                f"candidate layout references devices absent from its device set: {missing}"
            )
        return tuple(card_idx[str(d)] for d in devices)

    def _fallback_tp_pp(stage: str) -> Tuple[int, int]:
        row_tp = _row_value(candidate_row, f"tp_{stage}")
        row_pp = _row_value(candidate_row, f"pp_{stage}")
        if row_tp is not None or row_pp is not None:
            return max(1, int(row_tp or 1)), max(1, int(row_pp or 1))
        try:
            return _fixed_llm_tp_pp_from_layout(
                candidate_pipeline, stage,
            ) or (1, 1)
        except (KeyError, ValueError):
            return (1, 1)

    dynamic_timeout_s = _candidate_dynamic_timeout_s(
        candidate_pipeline, candidate_row,
    )
    decode_depth = max(
        int(batch_size_request), int(batch_size_decode or batch_size_request),
    )
    vllm_cfg = system_cfg.get("vllm") or {}
    vllm_engines = vllm_cfg.get("engines") or {}
    prefill_budget = int(
        vllm_cfg.get("max_num_batched_tokens") or _PREFILL_BUDGET
    )
    sim_cfg = system_cfg.get("simulator") or {}
    if sim_cfg.get("call_overhead_s") is not None:
        raise ValueError(
            "system.simulator.call_overhead_s is retired; fixed LLM call work "
            "must come from the exact standalone component profile"
        )

    def _engine_model_config(engine: str):
        module = _find_module(candidate_pipeline, engine)
        model_name = module.get("model")
        if isinstance(model_name, list):
            model_name = model_name[0] if len(model_name) == 1 else None
        if not model_name:
            return None
        try:
            from GenZ.Models import get_configs
            from rag_stack.cost_model.llm_sim.genz_model_map import (
                resolve_genz_model,
            )
            resolved = resolve_genz_model(str(model_name))
            return get_configs(resolved) if isinstance(resolved, str) else resolved
        except Exception:  # noqa: BLE001 - unresolved structure fails closed
            return None

    def _engine_cache_geometry(
        engine: str,
        *,
        tp: int,
        pp: int,
        role: str,
    ) -> Tuple[int, int]:
        """Return (block tokens, capacity blocks) from public declarations."""
        module = _find_module(candidate_pipeline, engine)
        engine_cfg = vllm_engines.get(engine) or {}
        raw_block = (
            engine_cfg.get("block_size")
            or module.get("block_size")
            or _VLLM_PREFIX_BLOCK_TOKENS
        )
        try:
            block_tokens = max(1, int(raw_block))
        except (TypeError, ValueError, OverflowError):
            block_tokens = _VLLM_PREFIX_BLOCK_TOKENS
        if not prefix_caching or role == "decode":
            return block_tokens, 0

        model_cfg = _engine_model_config(engine)
        if model_cfg is None:
            return block_tokens, 0

        dtype = str(module.get("dtype") or "bfloat16").lower()
        if "32" in dtype:
            dtype_bytes = 4.0
        elif "8" in dtype:
            dtype_bytes = 1.0
        elif "4" in dtype:
            dtype_bytes = 0.5
        else:
            dtype_bytes = 2.0
        weight_bytes = _dense_decoder_weight_bytes(model_cfg, dtype_bytes)
        kv_bytes = _genz_model_for(candidate_pipeline, engine)
        if weight_bytes is None or kv_bytes is None:
            return block_tokens, 0

        raw_util = (
            engine_cfg.get("gpu_memory_utilization")
            if engine_cfg.get("gpu_memory_utilization") is not None
            else module.get("gpu_memory_utilization", 0.9)
        )
        try:
            from rag_stack.cost_model.llm_sim.calibration import gpu_memory_gb
            gpu_spec = getattr(hardware, "gpu", None)
            memory_bytes = float(gpu_memory_gb({"gpu": gpu_spec})) * (1024.0 ** 3)
        except Exception:  # noqa: BLE001 - no hardware capacity, no APC guess
            return block_tokens, 0
        capacity = _prefix_cache_capacity_blocks(
            gpu_memory_bytes=memory_bytes,
            gpu_memory_utilization=raw_util,
            model_weight_bytes=weight_bytes,
            kv_bytes_per_token=kv_bytes,
            tp=tp,
            pp=pp,
            block_tokens=block_tokens,
            kv_head_count=(
                getattr(model_cfg, "num_key_value_heads", None)
                or getattr(model_cfg, "num_attention_heads", None)
            ),
        )
        return block_tokens, capacity

    # CPU processor sharing in the DES applies to explicit light-stage host
    # operators. No aggregate engine-host capacity residual is injected.
    _n_cpu: Optional[float] = float(logical_cores(hardware))
    # The GenZ closures must see THIS candidate's exact devices/rank order.
    # They can price every request/decode batch dynamically, so cache by
    # concrete topology (not by RAGO microbatch row or timeout replica).
    if latency is None:
        latency_key = (
            latency_cache_key
            if latency_cache_key is not _LATENCY_CACHE_KEY_UNSET
            else _stable_signature_value(layout)
        )
        if latency_cache is not None and latency_key in latency_cache:
            latency = latency_cache[latency_key]
        else:
            latency = build_call_latency(
                candidate_pipeline,
                hardware,
                **({"bind_profiles": True} if compiled_workload is not None else {}),
            )
            if latency_cache is not None:
                latency_cache[latency_key] = latency

    quality_retrieval_fanout = max(
        1, int(_get_query_expansion_fanout(candidate_pipeline)),
    )
    rrf_service = getattr(latency, "rrf_s", None)
    rrf_call_service = getattr(latency, "rrf_call_s", None)
    has_local_fusion = False
    observed_fusion_calls = []
    for calls in trace:
        for index, call in enumerate(calls):
            if _trace_stage(call) != "retrieval_fusion":
                continue
            if (
                index < 2
                or _trace_stage(calls[index - 1]) != "semantic_retrieval_vectorsearch"
                or _trace_stage(calls[index - 2]) != "semantic_retrieval_encode"
            ):
                raise ValueError(
                    "retrieval_fusion must follow its local encode/search pair; "
                    "a nonlocal fusion needs an explicit physical-worker policy"
                )
            if rrf_service is None and rrf_call_service is None:
                raise ValueError("compiled retrieval_fusion requires an RRF service model")
            has_local_fusion = True
            observed_fusion_calls.append(call)

    def _price_fusion_calls(calls):
        if rrf_call_service is not None:
            return sum(float(rrf_call_service(call, 1)) for call in calls)
        # Only explicitly injected count-only services use this compatibility
        # path. Production build_call_latency always installs rrf_call_s.
        return float(rrf_service(len(calls))) if calls else 0.0

    retrieval_visits = sum(
        _trace_stage(call) == "semantic_retrieval_vectorsearch"
        for calls in trace for call in calls
    )
    representative_fusion_per_visit = (
        _price_fusion_calls(observed_fusion_calls) / max(1, retrieval_visits)
        if has_local_fusion else 0.0
    )

    # token statistics per stage from the compiled calls (mean per-row input tokens
    # for token-parameterized services)
    tok_sum: Dict[str, List[float]] = {}
    for q in trace:
        for c in (q.get("calls") if isinstance(q, dict) else q) or []:
            st = _trace_stage(c)
            tok_sum.setdefault(st, []).append(float(c.get("input_tokens", 0)))
    def _mean_tok(st: str, default: int = 256) -> int:
        xs = tok_sum.get(st) or []
        return int(sum(xs) / len(xs)) if xs else default

    stage_specs: List[SimStage] = []
    stage_idx: Dict[str, int] = {}

    def add_stage(
        name: str,
        fn: Callable[[int], float],
        *,
        performance_stage: Optional[str] = None,
        batch_cap: Optional[int] = None,
        gpu_service_s: Optional[Callable[[int], float]] = None,
        batch_service_s: Optional[
            Callable[[Sequence[SimStageItem]], float]
        ] = None,
        batch_gpu_service_s: Optional[
            Callable[[Sequence[SimStageItem]], float]
        ] = None,
        batch_ops: Optional[
            Callable[[Sequence[SimStageItem]], Sequence[SimStageOp]]
        ] = None,
        gpu_share: Optional[float] = None,
        host_width: float = 1.0,
    ) -> None:
        perf_stage = performance_stage or name
        devices = _candidate_devices(perf_stage)
        resolved_batch_cap = (
            int(batch_size_request)
            if batch_cap is None else int(batch_cap)
        )
        if resolved_batch_cap <= 0:
            raise ValueError(
                f"stage {name!r} has invalid batch cap {resolved_batch_cap}"
            )
        stage_idx[name] = len(stage_specs)
        stage_specs.append(SimStage(
            name=name,
            service_s=fn,
            gpu_share=(0.0 if gpu_share is None else gpu_share),
            card=None,
            cards=_cards(devices),
            devices=devices,
            batch_cap=resolved_batch_cap,
            timeout_s=dynamic_timeout_s,
            gpu_service_s=gpu_service_s,
            batch_service_s=batch_service_s,
            batch_gpu_service_s=batch_gpu_service_s,
            batch_ops=batch_ops,
            host_width=max(1.0, float(host_width)),
        ))

    def _dp_batch(batch: int, devices: Sequence[str]) -> int:
        return max(1, math.ceil(max(1, int(batch)) / max(1, len(devices))))

    if latency.encode_s is not None or latency.retrieve_s is not None:
        # ONE worker in the measured harness: encode + faiss + fetch happen
        # inside a single batched node service — two chained batchers would
        # double the queueing and distort every batch statistic.
        _enc = latency.encode_s
        _ret = latency.retrieve_s
        _ret_kernel = getattr(latency, "retrieve_kernel_s", None)
        _ret_backend = getattr(latency, "retrieve_backend_overhead_s", None)
        _ret_outer = getattr(latency, "retrieve_outer_overhead_s", None)
        _ret_tail = getattr(latency, "retrieve_fused_overhead_s", None)
        _enc_devices = _candidate_devices("semantic_retrieval_encode")
        _qe_fanout = quality_retrieval_fanout
        _num_retrieval_servers = max(
            1, int(_row_value(candidate_row, "num_retrieval_servers", 1) or 1),
        )
        _structured_retrieval = (
            _ret_kernel is not None
            and (_ret_backend is not None or _ret_outer is not None)
        )

        def _retrieval_plan(multiplicities: Sequence[int]):
            # Production latencies expose backend glue and outer worker work as
            # separate units.  Legacy injected doubles expose only one fused
            # tail; preserve that tail once while using the same chunk planner.
            search_fn = _ret_kernel if _ret_kernel is not None else _ret
            backend_fn = _ret_backend if _structured_retrieval else None
            outer_fn = (
                _ret_outer
                if _structured_retrieval
                else (_ret_tail if _ret_kernel is not None else None)
            )
            return build_retrieval_worker_plan(
                multiplicities,
                chunk_cap=int(batch_size_request),
                encode_s=_enc,
                vectorsearch_s=(
                    (lambda chunk: float(search_fn(chunk))
                     / _num_retrieval_servers)
                    if search_fn is not None else None
                ),
                backend_glue_s=backend_fn,
                outer_s=outer_fn,
                encode_batch_size=(
                    lambda chunk: _dp_batch(chunk, _enc_devices)
                ),
            )

        def _retr_total(b: int) -> float:
            user_batch = max(1, int(b))
            total = _retrieval_plan((_qe_fanout,) * user_batch).total_s
            if has_local_fusion:
                total += user_batch * representative_fusion_per_visit
            return total

        def _retr_gpu(b: int) -> float:
            user_batch = max(1, int(b))
            return _retrieval_plan((_qe_fanout,) * user_batch).encode_s

        def _retr_ops(items: Sequence[SimStageItem]) -> Sequence[SimStageOp]:
            plan = _retrieval_plan(
                tuple(item.multiplicity for item in items),
            )
            # A quality invocation is one user-row worker visit.  The runtime
            # flattens all visits' K sub-queries and chunks that flat list by
            # the deployed request/embedding cap.  Each backend chunk executes
            # encode, FAISS, then its own dispatch/copy glue.  Row-local outer
            # fetch/glue and local RRF run after the backend chunks.  Their
            # costs remain separate, but all operations retain the runtime's
            # one physical worker, one batch queue, and one admission timeout.
            ops: List[SimStageOp] = []
            for chunk_index, chunk in enumerate(plan.chunks):
                if _enc is not None:
                    ops.append(SimStageOp(
                        resource="gpu",
                        service_s=chunk.encode_s,
                        name=f"semantic_retrieval_encode[{chunk_index}]",
                    ))
                # The encode->FAISS tensor movement is internal to this fused
                # worker visit.  It must not be emitted later as B per-request
                # boundary events.  TODO: give SimStageOp a batch-level link
                # lane and place B*K*embedding_dim bytes here.  Until that
                # resource exists, the incremental internal-link charge is
                # explicitly zero rather than a causally misplaced surrogate.
                if _ret_kernel is not None or _ret is not None:
                    ops.append(SimStageOp(
                        resource="cpu",
                        service_s=chunk.vectorsearch_s,
                        # A partial final backend chunk cannot occupy more
                        # FAISS workers than rows.  ``faiss_width`` already
                        # caps configured threads and physical logical cores.
                        host_width=float(min(
                            chunk.backend_rows, faiss_width,
                        )),
                        name=(
                            "semantic_retrieval_vectorsearch"
                            f"[{chunk_index}]"
                        ),
                    ))
                if _structured_retrieval and _ret_backend is not None:
                    ops.append(SimStageOp(
                        resource="cpu",
                        service_s=chunk.backend_glue_s,
                        host_width=1.0,
                        name=(
                            "semantic_retrieval_backend_glue"
                            f"[{chunk_index}]"
                        ),
                    ))
            outer_fn = (
                _ret_outer
                if _structured_retrieval
                else (_ret_tail if _ret_kernel is not None else None)
            )
            if outer_fn is not None:
                ops.append(SimStageOp(
                    resource="cpu",
                    service_s=plan.outer_s,
                    host_width=1.0,
                    name="semantic_retrieval_fused_tail",
                ))
            fusion_calls = [
                call for item in items for call in item.calls
                if _trace_stage(call) == "retrieval_fusion"
            ]
            if fusion_calls:
                ops.append(SimStageOp(
                    resource="cpu",
                    service_s=_price_fusion_calls(fusion_calls),
                    host_width=1.0,
                    name="retrieval_fusion",
                ))
            return tuple(ops)
        retrieval_cfg = system_cfg.get("retrieval") or {}
        faiss_threads = retrieval_cfg.get("faiss_num_threads")
        if faiss_threads is None:
            faiss_threads = logical_cores(hardware)
        faiss_width = min(
            max(1, int(batch_size_request)),
            max(1, int(faiss_threads)),
            max(1, int(logical_cores(hardware))),
        )
        add_stage(
            "semantic_retrieval",
            _retr_total,
            performance_stage="semantic_retrieval_encode",
            batch_cap=_data_parallel_global_batch_cap(
                batch_size_request, _enc_devices,
            ),
            gpu_share=0.0,
            gpu_service_s=(_retr_gpu if _enc else None),
            batch_ops=_retr_ops,
            host_width=float(faiss_width),
        )
    if latency.bm25_retrieve_s is not None:
        add_stage(
            "lexical_retrieval",
            latency.bm25_retrieve_s,
            host_width=float(min(
                int(batch_size_request), int(logical_cores(hardware)),
            )),
        )
    if latency.rerank_s is not None:
        _rr_glue = latency.rerank_glue_s
        _rr_batch = getattr(latency, "rerank_batch_s", None)
        _rr_tiles = getattr(latency, "rerank_forward_tiles_s", None)
        _rr_tiles_accepts_request_batch = False
        if _rr_tiles is not None:
            import inspect
            try:
                parameters = inspect.signature(_rr_tiles).parameters.values()
                _rr_tiles_accepts_request_batch = any(
                    parameter.name == "request_batch_size"
                    or parameter.kind in {
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    }
                    for parameter in parameters
                )
            except (TypeError, ValueError):
                _rr_tiles_accepts_request_batch = True
        _rr_devices = _candidate_devices("passage_reranker")
        def _rr(batch: int) -> float:
            local_batch = _dp_batch(batch, _rr_devices)
            # Compatibility count-only API: token shape 0 delegates to the
            # workflow schema/TokenStats. Production DES uses ``_rr_ops`` and
            # every quality-call member below.
            s = latency.rerank_s(0, local_batch)
            # the worker-glue coefficient family is what this simulation
            # retires — price the kernel, let queueing emerge
            return max(
                0.0,
                s - (_rr_glue(local_batch) if _rr_glue else 0.0),
            )
        def _rr_ops(items: Sequence[SimStageItem]) -> Sequence[SimStageOp]:
            tokens = [max(0, int(item.input_tokens)) for item in items]
            replicas = max(1, len(_rr_devices))
            # Runtime DP preserves flattened input order and partitions it into
            # contiguous, size-balanced shards. Reranker fanout is identical
            # for every user row, so the same structural partition can be
            # expressed in row units without importing evaluator code.
            base, remainder = divmod(len(tokens), replicas)
            shards: List[List[int]] = []
            cursor = 0
            for replica_index in range(replicas):
                size = base + (1 if replica_index < remainder else 0)
                shards.append(tokens[cursor:cursor + size])
                cursor += size
            replica_tiles: List[Tuple[float, ...]] = []
            for shard in shards:
                if not shard:
                    continue
                if _rr_tiles is not None:
                    # Only the gang queue expands to N*B.  The measured node's
                    # forward/padding policy is injected from the configured
                    # per-replica request cap B before DP dispatch.
                    durations = (
                        _rr_tiles(
                            shard,
                            request_batch_size=int(batch_size_request),
                        )
                        if _rr_tiles_accepts_request_batch
                        else _rr_tiles(shard)
                    )
                    replica_tiles.append(tuple(
                        float(duration)
                        for duration in durations
                    ))
                    continue
                if _rr_batch is not None:
                    total_s = float(_rr_batch(shard))
                    glue_s = float(_rr_glue(len(shard))) if _rr_glue else 0.0
                    replica_kernel_s = max(0.0, total_s - glue_s)
                else:
                    # Compatibility for injected latency doubles: a padded
                    # reranker forward is governed by the largest member of
                    # this concrete DP replica's resident shard.
                    total_s = float(latency.rerank_s(max(shard), len(shard)))
                    glue_s = float(_rr_glue(len(shard))) if _rr_glue else 0.0
                    replica_kernel_s = max(0.0, total_s - glue_s)
                replica_tiles.append((replica_kernel_s,))
            ops: List[SimStageOp] = []
            # DP replicas execute corresponding forward tiles in lock-step;
            # each tile is a distinct GPU job so an older engine ticket may
            # arbitrate between forward launches.
            num_tiles = max((len(tiles) for tiles in replica_tiles), default=0)
            for tile_index in range(num_tiles):
                tile_s = max(
                    (
                        tiles[tile_index]
                        if tile_index < len(tiles) else 0.0
                    )
                    for tiles in replica_tiles
                )
                if tile_s > 0.0:
                    ops.append(SimStageOp(
                        resource="gpu",
                        service_s=tile_s,
                        name=f"passage_reranker.forward[{tile_index}]",
                    ))
            if _rr_glue is not None and items:
                glue_s = float(_rr_glue(len(items)))
                if glue_s > 0.0:
                    ops.append(SimStageOp(
                        resource="cpu",
                        service_s=glue_s,
                        host_width=1.0,
                        name="passage_reranker.glue",
                    ))
            return tuple(ops)
        add_stage(
            "passage_reranker", _rr,
            batch_cap=_data_parallel_global_batch_cap(
                batch_size_request, _rr_devices,
            ),
            gpu_share=0.0,
            batch_ops=_rr_ops,
        )
    if latency.compress_s is not None:
        _cc_devices = _candidate_devices("passage_compressor")
        _cc_component = str(
            _find_module(candidate_pipeline, "passage_compressor").get(
                "component", "",
            )
        )
        _cc_is_data_parallel = _cc_component == "llmlingua2"

        def _cc_structural_batch(items: Sequence[SimStageItem]) -> float:
            # LLMLingua's public CM shape is corpus/chunker elem units from
            # TokenStats, not generator-token counts in the quality call.
            # Keep member shapes on the queue, but price only the declared
            # structural batch here instead of inventing a tokenizer mapping.
            return float(latency.compress_s(
                0, _dp_batch(len(items), _cc_devices),
            ))
        add_stage(
            "passage_compressor",
            lambda batch: latency.compress_s(
                0, _dp_batch(batch, _cc_devices),
            ),
            batch_cap=(
                _data_parallel_global_batch_cap(
                    batch_size_request, _cc_devices,
                )
                if _cc_is_data_parallel else int(batch_size_request)
            ),
            batch_service_s=_cc_structural_batch,
            # The aggregate LLMLingua profile measures complete worker
            # service but does not identify phase boundaries. Preserve that
            # stage capacity as one unsplit worker service; do not invent the
            # retired 0.3/0.7 GPU/host split. A future standalone operator-
            # phase profile can replace this scalar with explicit batch_ops.
            gpu_share=0.0,
            # One LLMLingua replica runs per concrete DP card. Their local
            # batches complete in parallel (the wall service above is the
            # maximum/local-replica service), while their Python/tokenization
            # portions consume one host lane apiece. Width=replicas preserves
            # total host core-work without introducing a fitted coefficient.
            host_width=float(max(1, len(_cc_devices))),
        )

    def _engine_step(
        stage_name: str,
        tp_: int,
        pp_: int,
        prefill_fn,
        decode_fn,
        *,
        role: str,
    ):
        # GenZ prices kernel work at the current prompt/KV shape.  The only
        # calibrated serving overhead is a non-negative
        # hardware×serving-stage×resident-batch curve.  It has no model,
        # TP/PP, topology, deployment, or token-shape lookup dimensions.
        if role not in {"unified", "prefill", "decode"}:
            return None, f"invalid simulator engine role {role!r}"
        if role in {"unified", "decode"} and decode_fn is None:
            return None, f"no decode model for tp{tp_}pp{pp_}"
        if role in {"unified", "prefill"} and prefill_fn is None:
            return None, f"no prefill model for tp{tp_}pp{pp_}"
        prefill_stage, decode_stage = _llm_stage_names(stage_name)
        curve_stage = _engine_cycle_profile_stage(
            stage_name,
            role,
            prefill_stage=prefill_stage,
            decode_stage=decode_stage,
        )
        if engine_cycle_applicator is not None:
            try:
                # Validate the exact role boundary before compiling queries.
                # The real callback below receives the actual mixed-cycle
                # population and already-composed component service.
                engine_cycle_applicator.corrected_service_s(
                    engine_stage=stage_name,
                    role=role,
                    prefill_stage=prefill_stage,
                    decode_stage=decode_stage,
                    resident_batch=1,
                    component_service_s=1.0,
                )
            except StageBatchCurveError as exc:
                boundary = (
                    "collocated" if role == "unified" else f"disagg {role}"
                )
                return None, (
                    f"{boundary} engine-cycle profile {curve_stage!r} is "
                    f"unpriceable: {exc}"
                )
            engine_cycle_profile = None
        elif stage_batch_profiles is _STAGE_BATCH_PROFILES_UNSET:
            try:
                engine_cycle_profile = load_stage_batch_curve(
                    hardware, curve_stage,
                )
            except StageBatchCurveError as exc:
                boundary = (
                    "collocated" if role == "unified" else f"disagg {role}"
                )
                return None, (
                    f"{boundary} engine-cycle profile {curve_stage!r} is "
                    f"unpriceable: {exc}"
                )
        else:
            # Compatibility path for direct builder callers without the
            # run-frozen production document.
            engine_cycle_profile = stage_batch_profiles.get(curve_stage)
            if stage_batch_profiles and engine_cycle_profile is None:
                return None, (
                    "frozen stage-batch generation lacks required role "
                    f"{curve_stage!r}"
                )
        _in_mean = _mean_tok(stage_name, 1024)
        step_fn, _chunk_rate = engine_step_price(
            prefill_fn if role != "decode" else None,
            decode_fn if role != "prefill" else None,
            in_mean=_in_mean,
            tp=tp_,
            pp=pp_,
        )
        # A decode-only PD engine never admits prefill work, so it neither uses
        # nor needs a chunk cadence.  Requiring one incorrectly asks the
        # prefill closure to price the decode role's potentially asymmetric
        # topology (e.g. 1P2D), where those ranks do not exist.
        if role != "decode" and not math.isfinite(_chunk_rate):
            return None, f"chunk cadence unpriceable for tp{tp_}pp{pp_}"
        def _prefill_batch_price(
            items: Sequence[SimPrefillItem],
        ) -> float:
            if not items or prefill_fn is None or role == "decode":
                return 0.0
            batch = len(items)
            # GenZ accepts a homogeneous batch shape.  Preserve every prompt
            # boundary until this final adapter, then use the CURRENT step's
            # batch mean (never a trace-global mean or a synthetic 8192x1
            # prompt).  APC suffix work is the causal marginal from the
            # resident context to context+slice.
            context = int(round(sum(
                float(item.context_tokens) for item in items
            ) / batch))
            total = int(round(sum(
                float(item.context_tokens) + float(item.new_tokens)
                for item in items
            ) / batch))
            full_s = float(prefill_fn(
                max(1, total), batch, int(tp_), int(pp_),
            ))
            cached_s = (
                float(prefill_fn(
                    max(1, context), batch, int(tp_), int(pp_),
                ))
                if context > 0 else 0.0
            )
            marginal_s = full_s - cached_s
            mean_new_tokens = sum(
                float(item.new_tokens) for item in items
            ) / batch
            if marginal_s <= 0.0 and mean_new_tokens > 0.0:
                # ``prefill_fn`` caches a 128-token GenZ grid. APC context and
                # context+suffix can occupy the same bucket; treating that as
                # free work would create zero-time scheduler steps. Interpolate
                # the local model slope between the adjacent bucket endpoints
                # and charge exactly this suffix's token distance.
                anchor = max(
                    int(_GENZ_INPUT_GRID),
                    int(math.ceil(max(1, context) / _GENZ_INPUT_GRID))
                    * int(_GENZ_INPUT_GRID),
                )
                next_anchor = anchor + int(_GENZ_INPUT_GRID)
                anchor_s = float(prefill_fn(
                    anchor, batch, int(tp_), int(pp_),
                ))
                next_s = float(prefill_fn(
                    next_anchor, batch, int(tp_), int(pp_),
                ))
                local_slope = (next_s - anchor_s) / float(_GENZ_INPUT_GRID)
                marginal_s = local_slope * mean_new_tokens
            return float(marginal_s)

        def _decode_step_price_from_aggregates(
            batch: int,
            input_token_sum: float,
            generated_token_sum: float,
        ) -> float:
            if batch <= 0 or decode_fn is None or role == "prefill":
                return 0.0
            input_tokens = int(round(float(input_token_sum) / batch))
            # ``decode_moddeling`` returns one-step latency at this generated
            # position.  The old closure hard-coded position=1 for the whole
            # call; this advances with the actual resident sequences.
            generated_position = max(
                1, int(round(float(generated_token_sum) / batch)) + 1,
            )
            genz_step_s = max(0.0, float(decode_fn(
                max(1, input_tokens), generated_position, batch,
                int(tp_), int(pp_),
            )))
            return genz_step_s

        def _decode_step_price(
            items: Sequence[SimDecodeItem],
        ) -> float:
            if not items:
                return 0.0
            return _decode_step_price_from_aggregates(
                len(items),
                sum(float(item.input_tokens) for item in items),
                sum(float(item.generated_tokens) for item in items),
            )

        # _Engine may bypass short-lived SimDecodeItem allocation only for
        # this production closure.  Arbitrary public/test callbacks continue
        # receiving the original per-sequence objects unchanged.
        _decode_step_price._rag_cm_decode_aggregate_s = (  # type: ignore[attr-defined]
            _decode_step_price_from_aggregates
        )
        _decode_step_price._rag_cm_decode_lazy_safe = True  # type: ignore[attr-defined]

        if engine_cycle_applicator is not None:
            if role == "prefill":
                def _prefill_frontier_corrected_service_s(
                    resident_batch: int,
                    completed_prefills: int,
                    component_service_s: float,
                ) -> float:
                    return engine_cycle_applicator.prefill_frontier_corrected_service_s(
                        engine_stage=stage_name,
                        prefill_stage=prefill_stage,
                        decode_stage=decode_stage,
                        resident_batch=int(resident_batch),
                        completed_prefills=int(completed_prefills),
                        component_service_s=float(component_service_s),
                    )

                _prefill_batch_price._rag_cm_prefill_frontier_corrected_service_s = _prefill_frontier_corrected_service_s  # type: ignore[attr-defined]
            else:
                def _cycle_corrected_service_s(
                    resident_batch: int,
                    component_service_s: float,
                ) -> float:
                    return engine_cycle_applicator.corrected_service_s(
                        engine_stage=stage_name,
                        role=role,
                        prefill_stage=prefill_stage,
                        decode_stage=decode_stage,
                        resident_batch=int(resident_batch),
                        component_service_s=float(component_service_s),
                    )

                _prefill_batch_price._rag_cm_cycle_corrected_service_s = _cycle_corrected_service_s  # type: ignore[attr-defined]
                _decode_step_price._rag_cm_cycle_corrected_service_s = _cycle_corrected_service_s  # type: ignore[attr-defined]
        elif engine_cycle_profile is not None:
            if role == "prefill":
                def _prefill_frontier_corrected_service_s(
                    resident_batch: int,
                    completed_prefills: int,
                    component_service_s: float,
                ) -> float:
                    resident = int(resident_batch)
                    component = float(component_service_s)
                    full_frontier = engine_cycle_profile.corrected_service_s(
                        resident,
                        component,
                    )
                    return component + (full_frontier - component) * (
                        int(completed_prefills) / resident
                    )

                _prefill_batch_price._rag_cm_prefill_frontier_corrected_service_s = _prefill_frontier_corrected_service_s  # type: ignore[attr-defined]
            else:
                def _cycle_corrected_service_s(
                    resident_batch: int,
                    component_service_s: float,
                ) -> float:
                    return engine_cycle_profile.corrected_service_s(
                        int(resident_batch),
                        float(component_service_s),
                    )

                # The engine combines component contributions first and
                # invokes this callback once. In a mixed cycle
                # ``resident_batch`` is the complete P+D population owned by
                # the engine, including residents not selected by this cycle's
                # token budget. Unified and direct-D retain engine-cycle
                # semantics; direct-P took the frontier branch above.
                _prefill_batch_price._rag_cm_cycle_corrected_service_s = _cycle_corrected_service_s  # type: ignore[attr-defined]
                _decode_step_price._rag_cm_cycle_corrected_service_s = _cycle_corrected_service_s  # type: ignore[attr-defined]

        def _legacy_step_price(decoding: int, chunk_tokens: float) -> float:
            # Production engines use the shape-aware hooks above.  The legacy
            # compatibility hook still receives only the resident decode
            # count, so it cannot prove a complete P+D active batch.  Keep it
            # analytical-only; production uses the shape-aware hooks above.
            component_service_s = float(step_fn(decoding, chunk_tokens))
            return component_service_s

        return (
            _legacy_step_price,
            (_prefill_batch_price if role != "decode" else None),
            (_decode_step_price if role != "prefill" else None),
        ), None

    engines_out: List[SimEngine] = []
    engine_deployments: Dict[str, dict] = {}
    trace_engines = {
        _trace_stage(call)
        for query in trace
        for call in ((query.get("calls") if isinstance(query, dict) else query) or [])
        if _trace_stage(call) in {"generator", "query_expansion"}
    }

    def _depth(engine: str, role: str) -> int:
        cfg = vllm_engines.get(engine) or {}
        if role == "prefill":
            return max(1, int(cfg.get("prefill_max_num_seqs") or batch_size_request))
        if role == "decode":
            default = decode_depth if engine == "generator" else batch_size_request
            return max(1, int(cfg.get("decode_max_num_seqs") or default))
        default = decode_depth if engine == "generator" else batch_size_request
        return max(1, int(cfg.get("max_num_seqs") or default))

    def _logical_engine_admission_capacity(engine: str) -> int:
        """Client request cap that fronts unified and PD role queues alike."""
        cfg = vllm_engines.get(engine) or {}
        default = decode_depth if engine == "generator" else batch_size_request
        max_num_seqs = max(1, int(cfg.get("max_num_seqs") or default))
        return _llm_admission_capacity(max_num_seqs)

    def _engine_functions(engine: str):
        return (
            latency.prefill_kernel_by_engine.get(engine)
            or latency.prefill_by_engine.get(engine)
            or (latency.prefill_s if engine == "generator" else None),
            latency.decode_by_engine.get(engine)
            or (latency.decode_s if engine == "generator" else None),
            latency.call_overhead_by_engine.get(engine, 0.0),
        )

    def _add_engine_replica(
        *,
        name: str,
        role: str,
        devices: Tuple[str, ...],
        depth: int,
        step_fn,
        prefill_batch_fn,
        decode_step_fn,
        prefix_cache_block_tokens: int,
        prefix_cache_capacity_blocks: int,
    ) -> int:
        if layout is not None and not devices:
            raise ValueError(f"materialized layout is missing devices for {name}")
        cards = _cards(devices) if devices else ()
        idx = len(engines_out)
        engines_out.append(SimEngine(
            name=name,
            role=role,
            devices=devices,
            cards=cards,
            depth=depth,
            step_s=step_fn,
            budget=prefill_budget,
            # Two-role component records are zero-intercept by contract.
            # Engine-cycle overhead has exactly one owner and must not coexist
            # with an absorbable per-call calibration knob.
            call_overhead_s=0.0,
            prefill_batch_s=prefill_batch_fn,
            decode_step_s=decode_step_fn,
            prefix_cache_block_tokens=prefix_cache_block_tokens,
            prefix_cache_capacity_blocks=prefix_cache_capacity_blocks,
        ))
        return idx

    for engine in sorted(trace_engines):
        prefill_stage, decode_stage = _llm_stage_names(engine)
        prefill_fn, decode_fn, raw_component_call_overhead_s = (
            _engine_functions(engine)
        )
        if prefill_fn is None or decode_fn is None:
            return None, f"no simulator LLM price functions for {engine}"
        try:
            component_call_overhead_s = float(raw_component_call_overhead_s)
        except (TypeError, ValueError, OverflowError):
            return None, (
                f"two-role LLM engine {engine!r} has invalid component call "
                f"overhead {raw_component_call_overhead_s!r}; engine-cycle "
                "overhead must have one non-absorbable owner"
            )
        if (
            not math.isfinite(component_call_overhead_s)
            or component_call_overhead_s != 0.0
        ):
            return None, (
                f"two-role LLM engine {engine!r} has forbidden component "
                f"call overhead {raw_component_call_overhead_s!r}; engine-cycle "
                "overhead must have one non-absorbable owner"
            )
        pf_dep = _role_deployment(
            layout, candidate_row,
            engine=engine,
            stage=prefill_stage,
            fallback_tp_pp=_fallback_tp_pp(prefill_stage),
            fallback_devices=_candidate_devices(prefill_stage),
        )
        dec_dep = _role_deployment(
            layout, candidate_row,
            engine=engine,
            stage=decode_stage,
            fallback_tp_pp=_fallback_tp_pp(decode_stage),
            fallback_devices=_candidate_devices(decode_stage),
        )
        info = ((layout or {}).get("engines") or {}).get(engine) or {}
        disagg = (
            info.get("pd_serving") == "disagg_pd"
            or stage_group.get(prefill_stage) != stage_group.get(decode_stage)
        )
        if not disagg:
            if (pf_dep.tp, pf_dep.pp) != (dec_dep.tp, dec_dep.pp):
                return None, (
                    f"unified {engine} has role-mismatched parallelism: "
                    f"prefill={pf_dep.tp}x{pf_dep.pp}, "
                    f"decode={dec_dep.tp}x{dec_dep.pp}"
                )
            if (
                pf_dep.replica_devices
                and dec_dep.replica_devices
                and pf_dep.replica_devices != dec_dep.replica_devices
            ):
                return None, (
                    f"unified {engine} has different prefill/decode replica "
                    "devices; materialize it as PD-disaggregated instead"
                )
            replicas = pf_dep.replica_devices or dec_dep.replica_devices
            if not replicas:
                fallback = pf_dep.devices or dec_dep.devices
                replicas = (fallback,) if fallback else ((),)
            step_prices, err = _engine_step(
                engine, pf_dep.tp, pf_dep.pp, prefill_fn, decode_fn,
                role="unified",
            )
            if step_prices is None:
                return None, err
            step_fn, prefill_batch_fn, decode_step_fn = step_prices
            cache_block_tokens, cache_capacity_blocks = _engine_cache_geometry(
                engine,
                tp=pf_dep.tp,
                pp=pf_dep.pp,
                role="unified",
            )
            pool = tuple(
                _add_engine_replica(
                    name=f"{engine}.unified.r{ri}",
                    role="unified",
                    devices=tuple(devices),
                    depth=_depth(engine, "unified"),
                    step_fn=step_fn,
                    prefill_batch_fn=prefill_batch_fn,
                    decode_step_fn=decode_step_fn,
                    prefix_cache_block_tokens=cache_block_tokens,
                    prefix_cache_capacity_blocks=cache_capacity_blocks,
                )
                for ri, devices in enumerate(replicas)
            )
            engine_deployments[engine] = {
                "disagg": False,
                "prefill_pool": pool,
                "decode_pool": (),
                "admission_capacity": (
                    _logical_engine_admission_capacity(engine)
                ),
            }
            continue

        pf_prices, err = _engine_step(
            engine, pf_dep.tp, pf_dep.pp, prefill_fn, decode_fn,
            role="prefill",
        )
        if pf_prices is None:
            return None, err
        dec_prices, err = _engine_step(
            engine, dec_dep.tp, dec_dep.pp, prefill_fn, decode_fn,
            role="decode",
        )
        if dec_prices is None:
            return None, err
        pf_step, pf_batch_fn, _unused_pf_decode = pf_prices
        dec_step, _unused_dec_prefill, dec_step_fn = dec_prices
        pf_cache_block_tokens, pf_cache_capacity_blocks = _engine_cache_geometry(
            engine,
            tp=pf_dep.tp,
            pp=pf_dep.pp,
            role="prefill",
        )
        dec_cache_block_tokens, dec_cache_capacity_blocks = _engine_cache_geometry(
            engine,
            tp=dec_dep.tp,
            pp=dec_dep.pp,
            role="decode",
        )
        pf_replicas = pf_dep.replica_devices or (
            (pf_dep.devices,) if pf_dep.devices else ((),)
        )
        dec_replicas = dec_dep.replica_devices or (
            (dec_dep.devices,) if dec_dep.devices else ((),)
        )
        pf_depth = _depth(engine, "prefill")
        dec_depth = _depth(engine, "decode")
        pf_pool = tuple(
            _add_engine_replica(
                name=f"{engine}.prefill.r{ri}",
                role="prefill",
                devices=tuple(devices),
                depth=pf_depth,
                step_fn=pf_step,
                prefill_batch_fn=pf_batch_fn,
                decode_step_fn=None,
                prefix_cache_block_tokens=pf_cache_block_tokens,
                prefix_cache_capacity_blocks=pf_cache_capacity_blocks,
            )
            for ri, devices in enumerate(pf_replicas)
        )
        dec_pool = tuple(
            _add_engine_replica(
                name=f"{engine}.decode.r{ri}",
                role="decode",
                devices=tuple(devices),
                depth=dec_depth,
                step_fn=dec_step,
                prefill_batch_fn=None,
                decode_step_fn=dec_step_fn,
                prefix_cache_block_tokens=dec_cache_block_tokens,
                prefix_cache_capacity_blocks=dec_cache_capacity_blocks,
            )
            for ri, devices in enumerate(dec_replicas)
        )
        kv_bytes_per_token = _genz_model_for(candidate_pipeline, engine)
        if not kv_bytes_per_token:
            return None, f"KV bytes/token unresolved for disaggregated {engine}"
        model_cfg = _engine_model_config(engine)
        kv_head_count = (
            (
                getattr(model_cfg, "num_key_value_heads", None)
                or getattr(model_cfg, "num_attention_heads", None)
            )
            if model_cfg is not None else None
        )
        # The topology helper assumes one distinct shard per target rank.  GQA
        # duplicates KV heads when decode TP exceeds kv_head_count, so scale
        # the logical payload by the target-side replication factor; this
        # yields full/(min(tp,kv_heads)*pp) bytes on every decode rank.
        kv_transfer_payload_factor = (
            float(dec_dep.tp)
            / float(_kv_tp_shard_degree(dec_dep.tp, kv_head_count))
        )
        engine_deployments[engine] = {
            "disagg": True,
            "prefill_pool": pf_pool,
            "decode_pool": dec_pool,
            "kv_bytes_per_token": float(kv_bytes_per_token),
            "kv_transfer_payload_factor": kv_transfer_payload_factor,
            "prefill_admission_capacity": (
                _pd_prefill_admission_capacity(
                    pf_depth,
                    replicas=len(pf_pool),
                )
            ),
            "decode_admission_capacity": (
                _pd_decode_admission_capacity(
                    dec_depth,
                    replicas=len(dec_pool),
                )
            ),
        }

    # Non-PD host boundaries use the SAME byte contract as the analytical
    # communication pass, but execute as FCFS link events here.  DynamicEngine
    # subsequently overwrites the analytical qps/latency columns, so omitting
    # these events would silently erase encode->FAISS and text/prompt movement.
    from rag_stack.cost_model.communication import (
        CommunicationPayloadModel,
        _is_pd_edge,
        gpu_cpu_link_bytes_s,
        include_non_pd_communication,
    )
    non_pd_enabled = include_non_pd_communication(candidate_pipeline)
    payload_model: Optional[CommunicationPayloadModel] = None
    host_bandwidth = host_latency = 0.0

    def _ensure_host_link_model() -> CommunicationPayloadModel:
        """Resolve token stats/link data only when a real host edge exists.

        Generator-only traces have no non-PD transfer.  Keeping this lazy lets
        those valid simulator inputs remain independent of retrieval token
        statistics while still making every materialized host boundary use the
        shared analytical byte contract.
        """
        nonlocal payload_model, host_bandwidth, host_latency
        if payload_model is None:
            payload_model = CommunicationPayloadModel(
                candidate_pipeline,
                seq_len_prefill=_mean_tok("generator", 1024),
                qe_seq_len=_mean_tok("query_expansion", 0) or None,
                reranker_input_fanout=(
                    _get_reranker_input_fanout(candidate_pipeline)
                ),
                query_expansion_fanout=(
                    _get_query_expansion_fanout(candidate_pipeline)
                ),
            )
            host_bandwidth, host_latency = gpu_cpu_link_bytes_s(hardware)
        return payload_model

    layout_stages = (layout or {}).get("performance_stages") or {}

    def _resource_of(stage: str) -> Any:
        info = layout_stages.get(stage) if isinstance(layout_stages, dict) else None
        if isinstance(info, dict) and info.get("resource") is not None:
            return info.get("resource")
        return stage_group.get(stage)

    def _needs_host_transfer(src: Optional[str], dst: str) -> bool:
        if not non_pd_enabled or src is None or _is_pd_edge(src, dst):
            return False
        host_stages = {
            "semantic_retrieval_vectorsearch", "lexical_retrieval",
            "retrieval_fusion",
        }
        if src in host_stages and dst in host_stages:
            return False
        if src in host_stages or dst in host_stages:
            return True
        src_resource, dst_resource = _resource_of(src), _resource_of(dst)
        return (
            src_resource is None
            or dst_resource is None
            or src_resource != dst_resource
        )

    def _host_transfer(
        src: Optional[str],
        dst: str,
        *,
        destination_input_tokens: Optional[int],
        destination_input_bytes: Optional[float],
    ) -> Optional[SimTransfer]:
        if not _needs_host_transfer(src, dst):
            return None
        assert src is not None
        model = _ensure_host_link_model()
        payload_bytes = model.payload_bytes(
            src,
            dst,
            destination_input_tokens=destination_input_tokens,
            destination_input_bytes=destination_input_bytes,
        )
        if payload_bytes <= 0:
            return None
        return SimTransfer(
            name=f"{src}_to_{dst}",
            resource="gpu_cpu",
            payload_bytes=float(payload_bytes),
            bandwidth_bytes_s=float(host_bandwidth),
            latency_s=float(host_latency),
        )

    def _engine_output_is_deterministic(engine: str) -> bool:
        raw_temperature = _find_module(candidate_pipeline, engine).get(
            "temperature", 0.0,
        )
        try:
            temperature = float(raw_temperature)
        except (TypeError, ValueError, OverflowError):
            return False
        return math.isfinite(temperature) and temperature == 0.0

    # Queries become immutable workload templates. APC itself is resolved only
    # at admission, after a concrete prefill/unified replica has been selected.
    queries: List[List[SimRound]] = []
    for query_index, q in enumerate(trace):
        source_question_id = source_question_ids[query_index]
        calls = (q.get("calls") if isinstance(q, dict) else q) or []
        rounds: List[SimRound] = []
        cur: List[int] = []
        cur_items: List[SimStageItem] = []
        boundary_transfers: Dict[int, List[SimTransfer]] = {}
        previous_stage: Optional[str] = None
        # A count-only trace cannot establish whether two sampled invocations
        # are the same logical prompt.  Disable APC for that query instead of
        # manufacturing cache identity from reservoir position.  Production
        # quality/performance trace adapters always supply source lineage.
        input_deterministic = source_question_id is not None
        engine_call_ordinals: Dict[str, int] = {}
        cache_lineage: Dict[
            str, Tuple[Tuple[Tuple[Any, int], ...], int]
        ] = {}
        # Index in ``cur_items`` of an encode call waiting for its one matching
        # vectorsearch call.  ``-1`` means the pair is structurally present but
        # this candidate has no priced semantic-retrieval worker.
        pending_retrieval_item: Optional[int] = None

        def _append_boundary(
            dst: str,
            *,
            position: int,
            destination_input_tokens: Optional[int],
            destination_input_bytes: Optional[float],
        ) -> None:
            transfer = _host_transfer(
                previous_stage,
                dst,
                destination_input_tokens=destination_input_tokens,
                destination_input_bytes=destination_input_bytes,
            )
            if transfer is not None:
                transfer = replace(transfer, source_calls=tuple(
                    dict(item) for item in calls[max(0, call_index - 1):call_index + 1]
                ))
                boundary_transfers.setdefault(position, []).append(transfer)

        for call_index, c in enumerate(calls):
            st = _trace_stage(c)
            if (
                pending_retrieval_item is not None
                and st != "semantic_retrieval_vectorsearch"
            ):
                raise ValueError(
                    "quality trace semantic_retrieval_encode must be followed "
                    "immediately by semantic_retrieval_vectorsearch"
                )
            raw_input_tokens = c.get("input_tokens")
            try:
                parsed_input_tokens = int(raw_input_tokens or 0)
            except (TypeError, ValueError, OverflowError):
                parsed_input_tokens = 0
            destination_input_tokens = (
                parsed_input_tokens if parsed_input_tokens > 0 else None
            )
            raw_input_bytes = c.get("input_bytes")
            try:
                parsed_input_bytes = float(raw_input_bytes)
            except (TypeError, ValueError, OverflowError):
                parsed_input_bytes = math.nan
            destination_input_bytes = (
                parsed_input_bytes
                if math.isfinite(parsed_input_bytes) and parsed_input_bytes >= 0
                else None
            )
            if st == "retrieval_fusion":
                # Preserve the logical invocation while keeping the physical
                # encode/search/fusion visit indivisible.  An independent RRF
                # batcher would invent overlap and another queueing timeout.
                if not cur_items or cur_items[-1].stage != "semantic_retrieval":
                    raise ValueError("local retrieval_fusion requires a priced retrieval worker")
                prior = cur_items[-1]
                if tuple(_trace_stage(call) for call in prior.calls) != (
                    "semantic_retrieval_encode", "semantic_retrieval_vectorsearch",
                ):
                    raise RuntimeError("invalid internal local retrieval-fusion state")
                cur_items[-1] = replace(prior, calls=prior.calls + (dict(c),))
                previous_stage = st
                continue
            if st in _TRACE_STAGE_TO_SIM:
                sim_name = _TRACE_STAGE_TO_SIM[st]
                if sim_name == "semantic_retrieval":
                    if st == "semantic_retrieval_encode":
                        # Encode starts a NEW physical worker visit even when
                        # another complete encode/search pair is adjacent.  A
                        # generic "same mapped stage" dedupe would erase R.
                        _append_boundary(
                            st,
                            position=len(cur),
                            destination_input_tokens=destination_input_tokens,
                            destination_input_bytes=destination_input_bytes,
                        )
                        if sim_name in stage_idx:
                            idx = stage_idx[sim_name]
                            cur.append(idx)
                            cur_items.append(SimStageItem(
                                stage=sim_name,
                                input_tokens=max(0, parsed_input_tokens),
                                output_tokens=max(
                                    0, int(c.get("output_tokens", 0) or 0),
                                ),
                                input_bytes=destination_input_bytes,
                                multiplicity=quality_retrieval_fanout,
                                calls=(dict(c),),
                            ))
                            pending_retrieval_item = len(cur_items) - 1
                        else:
                            pending_retrieval_item = -1
                    elif st == "semantic_retrieval_vectorsearch":
                        if pending_retrieval_item is None:
                            raise ValueError(
                                "quality trace semantic_retrieval_vectorsearch "
                                "has no immediately preceding encode"
                            )
                        if pending_retrieval_item >= 0:
                            prior = cur_items[pending_retrieval_item]
                            if (
                                len(prior.calls) != 1
                                or _trace_stage(prior.calls[0])
                                != "semantic_retrieval_encode"
                            ):
                                raise RuntimeError(
                                    "invalid internal semantic-retrieval pair state"
                                )
                            cur_items[pending_retrieval_item] = replace(
                                prior,
                                input_tokens=max(
                                    prior.input_tokens, parsed_input_tokens,
                                ),
                                output_tokens=max(
                                    prior.output_tokens,
                                    int(c.get("output_tokens", 0) or 0),
                                ),
                                input_bytes=(
                                    destination_input_bytes
                                    if destination_input_bytes is not None
                                    else prior.input_bytes
                                ),
                                calls=prior.calls + (dict(c),),
                            )
                        # This GPU->CPU edge is internal to the fused worker's
                        # operator graph.  Do not emit a post-worker, per-rid
                        # boundary transfer; see ``_retr_ops`` for the explicit
                        # zero-incremental-link TODO.
                        pending_retrieval_item = None
                    else:  # defensive: both mapped retrieval stages handled
                        raise RuntimeError(
                            f"unexpected semantic retrieval trace stage {st!r}"
                        )
                    previous_stage = st
                    continue

                # Every other mapped stage is an external worker hop.  Preserve
                # its existing adjacent-call coalescing behavior.
                _append_boundary(
                    st,
                    position=len(cur),
                    destination_input_tokens=destination_input_tokens,
                    destination_input_bytes=destination_input_bytes,
                )
                if sim_name in stage_idx:
                    idx = stage_idx[sim_name]
                    if not cur or cur[-1] != idx:
                        cur.append(idx)
                        cur_items.append(SimStageItem(
                            stage=sim_name,
                            input_tokens=max(0, parsed_input_tokens),
                            output_tokens=max(
                                0, int(c.get("output_tokens", 0) or 0),
                            ),
                            input_bytes=destination_input_bytes,
                            multiplicity=1,
                            calls=(dict(c),),
                        ))
                    else:
                        # Preserve the pre-existing coalescing contract for
                        # other adjacent calls mapped to the same worker. Dense
                        # retrieval and its local fusion are handled above.
                        prior = cur_items[-1]
                        cur_items[-1] = replace(
                            prior,
                            input_tokens=max(
                                prior.input_tokens, parsed_input_tokens,
                            ),
                            output_tokens=max(
                                prior.output_tokens,
                                int(c.get("output_tokens", 0) or 0),
                            ),
                            input_bytes=(
                                destination_input_bytes
                                if destination_input_bytes is not None
                                else prior.input_bytes
                            ),
                            calls=prior.calls + (dict(c),),
                        )
                previous_stage = st
            elif st in engine_deployments:
                pin = int(c.get("input_tokens", 0))
                pout = max(1, int(c.get("output_tokens", 0)))
                prefill_stage, decode_stage = _llm_stage_names(st)
                _append_boundary(
                    prefill_stage,
                    position=len(cur),
                    destination_input_tokens=pin,
                    destination_input_bytes=destination_input_bytes,
                )
                deployment = engine_deployments[st]
                pf_pool = tuple(deployment["prefill_pool"])
                dec_pool = tuple(deployment["decode_pool"])
                ordinal = engine_call_ordinals.get(st, 0)
                engine_call_ordinals[st] = ordinal + 1
                cache_identity = (
                    (source_question_id, st, ordinal, max(0, pin))
                    if input_deterministic else None
                )
                cache_blocks: Tuple[Any, ...] = ()
                result_cache_blocks: Tuple[Any, ...] = ()
                output_deterministic = _engine_output_is_deterministic(st)
                if prefix_caching and input_deterministic:
                    block_tokens = engines_out[
                        pf_pool[0]
                    ].prefix_cache_block_tokens
                    previous = cache_lineage.get(st)
                    if previous is not None and pin >= previous[1]:
                        prompt_segments = previous[0]
                        suffix_tokens = pin - previous[1]
                        if suffix_tokens > 0:
                            prompt_segments = prompt_segments + ((
                                (
                                    source_question_id,
                                    st,
                                    ordinal,
                                    "prompt_suffix",
                                ),
                                suffix_tokens,
                            ),)
                    else:
                        prompt_segments = ((
                            (source_question_id, st, ordinal, "prompt"),
                            max(0, pin),
                        ),)
                    cache_blocks = _apc_block_keys(
                        prompt_segments, max(0, pin), block_tokens,
                    )
                    if output_deterministic:
                        result_segments = prompt_segments + ((
                            (source_question_id, st, ordinal, "output"),
                            pout,
                        ),)
                        result_cache_blocks = _apc_block_keys(
                            result_segments,
                            max(0, pin) + pout,
                            block_tokens,
                        )
                        cache_lineage[st] = (
                            result_segments, max(0, pin) + pout,
                        )
                    else:
                        # This call's input is exact and reusable, but sampled
                        # output bytes are not. Publish prompt blocks only.
                        result_cache_blocks = cache_blocks
                        cache_lineage.pop(st, None)
                else:
                    cache_lineage.pop(st, None)
                transfers: Dict[Tuple[int, int], SimTransfer] = {}
                if deployment["disagg"]:
                    payload_bytes = (
                        pin
                        * deployment["kv_bytes_per_token"]
                        * deployment.get("kv_transfer_payload_factor", 1.0)
                    )
                    for pf_idx in pf_pool:
                        for dec_idx in dec_pool:
                            transfers[(pf_idx, dec_idx)] = kv_transfer_for_devices(
                                system_cfg,
                                engines_out[pf_idx].devices,
                                engines_out[dec_idx].devices,
                                payload_bytes=payload_bytes,
                                name=f"{st}_kv",
                            )
                rounds.append(SimRound(
                    stage_idxs=tuple(cur),
                    stage_items=tuple(cur_items),
                    call=SimCall(
                        source_calls=(dict(c),),
                        engine=pf_pool[0],
                        engine_pool=pf_pool,
                        decode_engine=(dec_pool[0] if dec_pool else None),
                        decode_engine_pool=dec_pool,
                        eff_in=max(0, pin),
                        out=pout,
                        full_in=max(0, pin),
                        transfers=transfers,
                        cache_identity=cache_identity,
                        cache_blocks=cache_blocks,
                        result_cache_blocks=result_cache_blocks,
                        admission_gate=(
                            None if deployment["disagg"] else st
                        ),
                        admission_capacity=(
                            None if deployment["disagg"] else int(
                                deployment["admission_capacity"]
                            )
                        ),
                        prefill_admission_gate=(
                            f"{st}:prefill" if deployment["disagg"] else None
                        ),
                        prefill_admission_capacity=(
                            int(deployment["prefill_admission_capacity"])
                            if deployment["disagg"] else None
                        ),
                        decode_admission_gate=(
                            f"{st}:decode" if deployment["disagg"] else None
                        ),
                        decode_admission_capacity=(
                            int(deployment["decode_admission_capacity"])
                            if deployment["disagg"] else None
                        ),
                    ),
                    boundary_transfers={
                        position: tuple(items)
                        for position, items in boundary_transfers.items()
                    },
                ))
                cur = []
                cur_items = []
                boundary_transfers = {}
                previous_stage = decode_stage
                input_deterministic = (
                    input_deterministic and output_deterministic
                )
        if pending_retrieval_item is not None:
            raise ValueError(
                "quality trace ended after semantic_retrieval_encode without "
                "semantic_retrieval_vectorsearch"
            )
        if cur or boundary_transfers:
            rounds.append(SimRound(
                stage_idxs=tuple(cur),
                stage_items=tuple(cur_items),
                call=None,
                boundary_transfers={
                    position: tuple(items)
                    for position, items in boundary_transfers.items()
                },
            ))
        # Admission-time traces can end upstream when teardown catches a row
        # still queued. Measured QPS counts completed answer requests; mirror
        # DynamicEngine's completeness filter and exclude rows that never
        # reached the terminal generator.
        if rounds and any(_trace_stage(call) == "generator" for call in calls):
            queries.append(rounds)
    if not queries:
        return None, "trace yielded no generator rounds"

    population = _structural_start_population(
        batch_size_request=int(batch_size_request),
        batch_size_decode=decode_depth,
        queries=queries,
    )
    sim_inputs = (stage_specs, engines_out, queries, n_cards, population, _n_cpu)
    if compilation_capture is not None:
        compilation_capture.update(
            pipeline_config=candidate_pipeline,
            latency=latency,
            source_question_ids=source_question_ids,
            calibration=(
                deepcopy((candidate_pipeline.get("_runtime") or {}).get(
                    _STAGE_BATCH_RUNTIME_SELECTION_KEY,
                ))
            ),
        )
    return sim_inputs, None


def _stable_signature_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (str(k), _stable_signature_value(v))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_stable_signature_value(v) for v in value)
    if not isinstance(value, str):
        try:
            if value != value:  # NaN
                return None
        except Exception:  # noqa: BLE001
            pass
    try:
        return value.item()
    except AttributeError:
        return value


def simulation_signature(
    row: Mapping[str, Any],
    pipeline_config: dict,
) -> Tuple[Any, ...]:
    """Exact DES identity for one RAGO row.

    RAGO's generator-decode batch is an independent deployment axis, not an
    internal row that may be collapsed.  Include every row batch axis in the
    identity and require the qualified generator-decode value to agree with the
    deployed ``batch_size_decode`` before simulation. Every other input that
    alters DES mechanics is included as well: concrete layout (devices,
    TP/PP/DP), timeout, retrieval width/server count, and vLLM budget. Trace and
    service-curve objects are constant within one
    :func:`apply_closed_loop_simulation` call, so they are implicit locally.
    """
    layout = _resolved_candidate_layout(pipeline_config, row)
    system = pipeline_config.get("system") or {}
    retrieval = system.get("retrieval") or {}
    batching = system.get("batching") or {}
    vllm = system.get("vllm") or {}
    row_parallelism = {
        str(key): _row_value(row, str(key))
        for key in getattr(row, "index", row.keys())
        if str(key).startswith((
            "tp_", "pp_", "candidate_tp_", "candidate_pp_",
            "candidate_devices_", "candidate_replicas_",
            "candidate_replica_devices_", "replicas_", "replica_devices_",
            "devices_", "num_chips_",
        ))
    }
    row_batch_axes = {
        str(key): _row_value(row, str(key))
        for key in getattr(row, "index", row.keys())
        if str(key).startswith("batch_size_")
    }
    return (
        _stable_signature_value(layout),
        _stable_signature_value(
            _row_value(row, "collocation_strategy") if layout is None else None
        ),
        int(_row_value(row, "batch_size_request", 1)),
        int(_row_value(
            row,
            "batch_size_decode",
            _row_value(row, "batch_size_request", 1),
        )),
        float(_row_value(
            row,
            "dynamic_batch_timeout_s",
            batching.get("dynamic_timeout_s", _BATCH_TIMEOUT_S),
        )),
        int(_row_value(row, "num_retrieval_servers", 1) or 1),
        _stable_signature_value(retrieval.get("faiss_num_threads")),
        _stable_signature_value(vllm),
        _stable_signature_value(row_batch_axes),
        _stable_signature_value(row_parallelism),
    )


_SIMULATION_SIGNATURE_TIMEOUT_INDEX = 4


def _builder_template_signature(signature: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """DES-builder identity with only the light-stage timeout removed.

    ``build_sim_inputs`` uses the timeout solely to stamp ``SimStage`` specs.
    All other candidate mechanics remain in ``simulation_signature``.  Timeout
    siblings can therefore share one immutable input template while still
    running independent simulations with their own exact timeout.
    """
    index = _SIMULATION_SIGNATURE_TIMEOUT_INDEX
    if len(signature) <= index:
        raise ValueError("simulator signature is missing its timeout field")
    return signature[:index] + signature[index + 1:]


@dataclass(frozen=True)
class _CandidateWork:
    """One exact deployment row plus its already-computed DES identities."""

    ordinal: int
    row: Mapping[str, Any]
    signature: Tuple[Any, ...]

    @property
    def latency_key(self) -> Any:
        # This is exactly build_sim_inputs' historical cache key: the stable
        # resolved layout, including ``None`` for legacy no-layout callers.
        return self.signature[0]

    @property
    def builder_key(self) -> Tuple[Any, ...]:
        return _builder_template_signature(self.signature)


def _sim_inputs_with_timeout(
    template: Tuple[Any, ...],
    timeout_s: float,
) -> Tuple[Any, ...]:
    """Copy only stage specs while reusing a read-only simulator template."""
    stages = [
        replace(stage, timeout_s=float(timeout_s))
        for stage in template[0]
    ]
    return (stages, *template[1:])


def _validate_candidate_batch_axes(row: Mapping[str, Any]) -> None:
    """Fail closed if RAGO and the deployed decode axis diverge.

    The RAGO boundary materializes its independently swept
    ``batch_size_generator_decode`` as the one public ``batch_size_decode``.
    A mismatch here means a second decode grid was introduced downstream; it
    must never be hidden by signature grouping.
    """

    row_keys = {str(key) for key in getattr(row, "index", row.keys())}
    has_rago_decode = "batch_size_generator_decode" in row_keys
    has_deployed_decode = "batch_size_decode" in row_keys
    request = _row_value(row, "batch_size_request", None)
    rago_decode = _row_value(row, "batch_size_generator_decode", None)
    deployed_decode = _row_value(row, "batch_size_decode", None)

    def missing(value: Any) -> bool:
        if value is None:
            return True
        try:
            return math.isnan(float(value))
        except (TypeError, ValueError):
            return False

    if has_rago_decode and missing(rago_decode):
        raise RuntimeError("RAGO generator-decode batch is missing")
    if has_rago_decode and (
        not has_deployed_decode or missing(deployed_decode)
    ):
        raise RuntimeError(
            "RAGO generator-decode batch has no deployed batch_size_decode"
        )

    # Ordinary RAGSweep rows carry a qualified generator-decode axis and have
    # the all-pre-decode-stages-share-request invariant. IterativeRAGSweep has
    # a deliberate batch_size_iterative axis and exposes batch_size_decode
    # directly; that mid-generation cohort is not stage microbatching.
    if not missing(request) and has_rago_decode:
        request_int = int(request)
        for key in getattr(row, "index", row.keys()):
            name = str(key)
            if not name.startswith("batch_size_") or name in {
                "batch_size_request",
                "batch_size_decode",
                "batch_size_generator_decode",
            }:
                continue
            value = _row_value(row, name, None)
            if value is None:
                continue
            try:
                if math.isnan(float(value)):
                    continue
            except (TypeError, ValueError):
                pass
            if int(value) != request_int:
                raise RuntimeError(
                    "RAGO pre-decode batch diverged while global "
                    "no_microbatching is enabled: "
                    f"{name}={value}, batch_size_request={request_int}"
                )

    if not has_rago_decode:
        return
    if int(rago_decode) != int(deployed_decode):
        raise RuntimeError(
            "RAGO generator-decode batch does not match the deployed decode "
            f"axis: batch_size_generator_decode={rago_decode}, "
            f"batch_size_decode={deployed_decode}"
        )


def _validate_saturation_result(result: SimResult, population: int) -> None:
    if not result.converged:
        raise RuntimeError(
            "simulator hit its event limit before the steady-state window "
            f"completed (population={population}, "
            f"completions={result.completions})"
        )
    if (
        result.completions <= 0
        or not math.isfinite(float(result.qps))
        or result.qps <= 0
    ):
        raise RuntimeError(
            "simulator completed no finite positive steady-state throughput "
            f"(population={population}, completions={result.completions}, "
            f"qps={result.qps})"
        )


def _simulate_to_saturation(
    sim_inputs: Tuple[Any, ...],
    *,
    simulate_fn=simulate,
) -> _SaturationOutcome:
    """Increase closed population until QPS reaches its numerical plateau.

    ``simulate_fn`` keeps the exact historical injection interface: stage,
    engine, and query positional arguments plus ``n_cards``, ``population``,
    and ``n_cpu_cores`` keyword arguments.  This lets existing deterministic
    test doubles participate in the same search without learning a new API.
    """
    start = max(1, int(sim_inputs[4]))
    relative_bound = start * (2 ** _SATURATION_MAX_DOUBLINGS)
    max_population = max(
        start,
        min(_SATURATION_MAX_POPULATION, relative_bound),
    )
    tried: List[int] = []
    previous: Optional[SimResult] = None
    previous_population = start

    for attempt in range(_SATURATION_MAX_DOUBLINGS + 1):
        population = min(max_population, start * (2 ** attempt))
        if tried and population == tried[-1]:
            break
        tried.append(population)
        try:
            result = simulate_fn(
                *sim_inputs[:3],
                n_cards=sim_inputs[3],
                population=population,
                n_cpu_cores=sim_inputs[5],
            )
            _validate_saturation_result(result, population)
        except Exception as exc:
            if previous is None:
                raise
            # A larger population can exceed the DES event safety budget.  It
            # must not invalidate the last fully converged physical result;
            # retain that result and make the bounded-search condition visible.
            return _SaturationOutcome(
                result=previous,
                populations_tried=tuple(tried),
                final_population=previous_population,
                plateau_reached=False,
                stop_reason=(
                    "higher_population_rejected:"
                    f"{type(exc).__name__}"
                ),
            )

        if previous is not None:
            tolerance = max(
                _SATURATION_QPS_ABS_TOL,
                _SATURATION_QPS_REL_TOL * max(
                    abs(float(previous.qps)),
                    abs(float(result.qps)),
                ),
            )
            if abs(float(result.qps) - float(previous.qps)) <= tolerance:
                return _SaturationOutcome(
                    result=result,
                    populations_tried=tuple(tried),
                    final_population=population,
                    plateau_reached=True,
                    stop_reason="qps_plateau",
                )

        previous = result
        previous_population = population
        if population >= max_population:
            break

    assert previous is not None
    return _SaturationOutcome(
        result=previous,
        populations_tried=tuple(tried),
        final_population=previous_population,
        plateau_reached=False,
        stop_reason="population_bound",
    )


def _des_worker_count(signature_count: int) -> int:
    """Resolve the exact-signature process count for one DES invocation.

    ``CM_DES_WORKERS=1`` (or zero) is the explicit serial opt-out. Without an
    override, use every affinity-visible CPU except two reserved for the parent
    and host services. Nested process pools are deliberately forbidden: if CM
    is itself invoked from a worker, that worker prices signatures serially.
    """
    count = max(0, int(signature_count))
    if count <= 1:
        return 1

    import multiprocessing as mp
    import os

    try:
        if mp.parent_process() is not None:
            return 1
    except (AttributeError, AssertionError):
        # Python implementations without parent_process still have a stable
        # main-process name.  This is only a nested-pool safety check.
        if mp.current_process().name != "MainProcess":
            return 1

    serial = os.environ.get("CM_DES_SERIAL", "").strip().lower()
    if serial in {"1", "true", "yes", "on"}:
        return 1

    try:
        affinity_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_count = os.cpu_count() or 1
    affinity_count = max(1, int(affinity_count))
    usable_workers = max(1, affinity_count - _DES_RESERVED_CPUS)

    raw = os.environ.get("CM_DES_WORKERS", "").strip()
    if raw:
        try:
            requested = int(raw)
        except ValueError:
            logger.warning(
                "Ignoring invalid CM_DES_WORKERS=%r; using serial DES", raw,
            )
            return 1
        if requested <= 1:
            return 1
        worker_cap = min(requested, usable_workers)
    else:
        worker_cap = usable_workers

    return max(1, min(count, worker_cap))


def _evaluate_candidate_rows(
    candidate_rows: Sequence[_CandidateWork],
    *,
    pipeline_config: dict,
    hardware,
    traces: list,
    trace_source_question_ids: Optional[Sequence[Any]] = None,
    prepared_paths: PreparedExecutionPaths | None = None,
    prefix_caching: bool,
    builder_fn=build_sim_inputs,
    simulate_fn=simulate,
) -> List[Tuple[int, Optional[SimResult], Optional[_SaturationOutcome], str]]:
    """Price exact rows with local topology and adjacent-cohort caches."""
    evaluated: List[
        Tuple[int, Optional[SimResult], Optional[_SaturationOutcome], str]
    ] = []
    candidate_latency_cache: Dict[Any, Any] = {}
    # Affinity chunks place timeout siblings next to each other.  Keep only the
    # immediately preceding template: this removes duplicate trace compilation
    # without retaining one large query graph per deployment shape.
    previous_builder_key: Optional[Tuple[Any, ...]] = None
    previous_template: Optional[Tuple[Any, ...]] = None
    production_builder = builder_fn is build_sim_inputs
    if production_builder or prepared_paths is not None:
        from rag_stack.cost_model.workload import resolve_execution_paths

        prepared_paths = resolve_execution_paths(traces, prepared_paths)
        traces = prepared_paths.paths
    stage_batch_profiles = (
        _stage_batch_profiles_for_pipeline(pipeline_config, hardware)
        if production_builder else _STAGE_BATCH_PROFILES_UNSET
    )
    for work in candidate_rows:
        ordinal, candidate_row = work.ordinal, work.row
        try:
            if (
                production_builder
                and previous_template is not None
                and previous_builder_key == work.builder_key
            ):
                sim_inputs = _sim_inputs_with_timeout(
                    previous_template,
                    _candidate_dynamic_timeout_s(
                        pipeline_config, candidate_row,
                    ),
                )
            else:
                builder_kwargs = dict(
                    batch_size_request=int(
                        candidate_row["batch_size_request"]
                    ),
                    batch_size_decode=int(_row_value(
                        candidate_row,
                        "batch_size_decode",
                        candidate_row["batch_size_request"],
                    )),
                    prefix_caching=prefix_caching,
                    candidate_row=candidate_row,
                    # Build candidate-specific GenZ closures after overlaying
                    # the concrete layout.  The analytical-walk closure may
                    # have been built from the base config and is not
                    # topology-safe.
                    latency=None,
                    latency_cache=candidate_latency_cache,
                )
                if production_builder:
                    # Reuse the exact key already computed in the parent.  The
                    # sentinel-aware builder preserves ``None`` as the valid
                    # legacy no-layout cache identity.
                    builder_kwargs.update(
                        latency_cache_key=work.latency_key,
                        stage_batch_profiles=stage_batch_profiles,
                        trace_source_question_ids=trace_source_question_ids,
                        prepared_paths=prepared_paths,
                    )
                sim_inputs, reason = builder_fn(
                    pipeline_config,
                    hardware,
                    traces,
                    _row_value(candidate_row, "collocation_strategy", []),
                    **builder_kwargs,
                )
                if sim_inputs is None:
                    raise RuntimeError(
                        reason or "simulator input builder rejected candidate"
                    )
                if production_builder:
                    previous_builder_key = work.builder_key
                    previous_template = sim_inputs
            saturation = _simulate_to_saturation(
                sim_inputs,
                simulate_fn=simulate_fn,
            )
            evaluated.append((ordinal, saturation.result, saturation, ""))
        except Exception as exc:  # noqa: BLE001 - one candidate is infeasible
            evaluated.append((
                ordinal,
                None,
                None,
                f"{type(exc).__name__}: {exc}",
            ))
    return evaluated


def _simulate_candidate_chunk(task):
    """Spawn-worker entry point for exact RAGO configurations.

    The function is top-level so it is picklable under ``spawn``.  Native math
    runtimes are restricted while the Python DES runs; otherwise an inherited
    optimizer setting such as ``OMP_NUM_THREADS=74`` gives every process a
    large, mostly sleeping thread team and can oversubscribe the host.
    """
    (
        chunk_id,
        pipeline_config,
        hardware,
        traces,
        trace_source_question_ids,
        prefix_caching,
        prepared_paths,
        candidate_rows,
    ) = task

    import os

    native_thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    previous_environment = {
        variable: os.environ.get(variable)
        for variable in native_thread_variables
    }
    for variable in native_thread_variables:
        os.environ[variable] = "1"

    def _run():
        return _evaluate_candidate_rows(
            candidate_rows,
            pipeline_config=pipeline_config,
            hardware=hardware,
            traces=traces,
            trace_source_question_ids=trace_source_question_ids,
            prepared_paths=prepared_paths,
            prefix_caching=prefix_caching,
        )

    try:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            evaluated = _run()
        else:
            with threadpool_limits(limits=1):
                evaluated = _run()
        return chunk_id, evaluated
    finally:
        # The executor is shared with later stage/RAGO sweep tasks.  Limiting
        # native teams is scoped to this GIL-bound DES task; never leak a
        # changed process environment into the next exact pricing phase.
        for variable, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous


def _des_process_pool(workers: int):
    """Reuse assembly's persistent spawn pool instead of creating a second."""
    from rag_stack.cost_model.assembly import _collocation_pool

    return _collocation_pool(workers)


def _candidate_chunks(
    candidate_rows: Sequence[_CandidateWork],
    workers: int,
) -> List[List[_CandidateWork]]:
    """Build deterministic, bounded, cache-aware work-stealing tasks.

    Round-robin rows defeated both levels of worker-local reuse: every process
    rebuilt every topology's ``CallLatency``, and adjacent replicas differing
    only in batching timeout landed in different processes.  Here a builder
    cohort remains atomic when practical and topology groups receive weighted
    task shares.  Large sweeps produce a small multiple of the process count,
    allowing the persistent executor to pull another task when one worker
    finishes early.  Ordinals remain authoritative, so task scheduling is
    never observable in the returned DataFrame.
    """
    if not candidate_rows:
        return [[]]
    worker_count = max(1, int(workers))
    target_chunks = max(1, min(
        len(candidate_rows),
        worker_count * _DES_TASKS_PER_WORKER,
    ))

    topology_cohorts: OrderedDict[
        Any, OrderedDict[Tuple[Any, ...], List[_CandidateWork]]
    ] = OrderedDict()
    for work in candidate_rows:
        topology_cohorts.setdefault(
            work.latency_key, OrderedDict(),
        ).setdefault(work.builder_key, []).append(work)

    def work_weight(work: _CandidateWork) -> int:
        request = max(1, int(_row_value(
            work.row, "batch_size_request", 1,
        )))
        decode = max(1, int(_row_value(
            work.row, "batch_size_decode", request,
        )))
        # A simulation window completes O(population) requests and its
        # structural population is bounded below by these deployment depths.
        # This is scheduling metadata only; it never enters DES physics.
        return max(request, decode)

    def cohort_weight(cohort: Sequence[_CandidateWork]) -> int:
        return sum(work_weight(work) for work in cohort)

    topology_weights = {
        key: sum(cohort_weight(cohort) for cohort in cohorts.values())
        for key, cohorts in topology_cohorts.items()
    }
    topology_count = len(topology_cohorts)
    chunks: List[List[_CandidateWork]] = []

    if topology_count <= target_chunks:
        # Give every topology one task, then repeatedly split the currently
        # heaviest per-shard topology until every whole builder cohort has a
        # task.  Builder cohorts stay atomic while that still fills the
        # requested task budget: adjacent timeout siblings then reuse one
        # compiled simulator template in their worker.
        shares = {key: 1 for key in topology_cohorts}
        while sum(shares.values()) < target_chunks:
            splittable = [
                key for key, cohorts in topology_cohorts.items()
                if shares[key] < len(cohorts)
            ]
            if not splittable:
                break
            key = max(
                splittable,
                key=lambda item: topology_weights[item] / shares[item],
            )
            shares[key] += 1

        # A small number of timeout-free parents can otherwise strand most of
        # a large host.  Split a cohort only after all whole cohorts already
        # have a task, and only as far as needed to fill the bounded task
        # budget.  Every exact row is still simulated independently; the
        # tradeoff is merely rebuilding an immutable template in an occasional
        # later task instead of serializing several long DES runs behind one
        # coarse shard.
        while sum(shares.values()) < target_chunks:
            splittable = [
                key for key, cohorts in topology_cohorts.items()
                if shares[key] < sum(len(cohort) for cohort in cohorts.values())
            ]
            if not splittable:
                break
            key = max(
                splittable,
                key=lambda item: topology_weights[item] / shares[item],
            )
            shares[key] += 1

        for key, cohorts_by_key in topology_cohorts.items():
            shard_count = shares[key]
            bins: List[List[_CandidateWork]] = [
                [] for _ in range(shard_count)
            ]
            loads = [0] * shard_count
            cohorts = list(cohorts_by_key.values())

            # Whole cohorts are still preferred.  If there are more shards
            # than cohorts, split the heaviest cohort into deterministic,
            # contiguous pieces until every shard has useful work.  Contiguous
            # pieces preserve timeout order and template reuse within each
            # piece; ordinal restoration below keeps output row order exact.
            while len(cohorts) < shard_count:
                split_candidates = [
                    index for index, cohort in enumerate(cohorts)
                    if len(cohort) > 1
                ]
                if not split_candidates:
                    break
                split_index = max(
                    split_candidates,
                    key=lambda index: (
                        cohort_weight(cohorts[index]),
                        len(cohorts[index]),
                        -cohorts[index][0].ordinal,
                    ),
                )
                cohort = cohorts[split_index]
                target_weight = cohort_weight(cohort) / 2.0
                running_weight = 0
                cut = 1
                for position, work in enumerate(cohort[:-1], start=1):
                    running_weight += work_weight(work)
                    cut = position
                    if running_weight >= target_weight:
                        break
                cohorts[split_index:split_index + 1] = [
                    cohort[:cut], cohort[cut:],
                ]
            cohorts.sort(
                key=lambda cohort: (-cohort_weight(cohort), cohort[0].ordinal),
            )
            for cohort in cohorts:
                shard = min(
                    range(shard_count),
                    key=lambda index: (loads[index], index),
                )
                bins[shard].extend(cohort)
                loads[shard] += cohort_weight(cohort)
            chunks.extend(chunk for chunk in bins if chunk)
    else:
        # More topology identities than tasks: keep every topology whole
        # and LPT-pack them.  A chunk may cache several topologies, but no one
        # topology is redundantly rebuilt in multiple tasks.
        chunks = [[] for _ in range(target_chunks)]
        loads = [0] * target_chunks
        topology_items = list(topology_cohorts.items())
        topology_items.sort(
            key=lambda item: (-topology_weights[item[0]],),
        )
        for key, cohorts_by_key in topology_items:
            shard = min(
                range(target_chunks),
                key=lambda index: (loads[index], index),
            )
            for cohort in cohorts_by_key.values():
                chunks[shard].extend(cohort)
            loads[shard] += topology_weights[key]

    # Persistent-pool workers pull tasks dynamically.  Submit the expensive
    # shards first to reduce the makespan without changing any candidate order
    # visible after ordinal restoration.
    chunks.sort(
        key=lambda chunk: -sum(work_weight(work) for work in chunk),
    )
    return chunks


def _evaluate_candidates_in_processes(
    candidate_rows: Sequence[_CandidateWork],
    *,
    workers: int,
    pipeline_config: dict,
    hardware,
    traces: list,
    trace_source_question_ids: Optional[Sequence[Any]] = None,
    prepared_paths: PreparedExecutionPaths | None = None,
    prefix_caching: bool,
) -> List[Tuple[int, Optional[SimResult], Optional[_SaturationOutcome], str]]:
    """Submit exact signature chunks and return results by stable ordinal."""
    from concurrent.futures import as_completed
    from concurrent.futures.process import BrokenProcessPool
    from rag_stack.cost_model.workload import resolve_execution_paths

    prepared_paths = resolve_execution_paths(traces, prepared_paths)
    traces = prepared_paths.paths

    chunks = _candidate_chunks(candidate_rows, workers)
    logger.info(
        "Authoritative DES dispatch: %d exact configurations across %d "
        "task chunks (worker cap=%d)",
        len(candidate_rows), len(chunks), workers,
    )
    process_count = min(max(1, int(workers)), len(chunks))
    executor = _des_process_pool(process_count)
    futures = []
    try:
        for chunk_id, chunk in enumerate(chunks):
            futures.append(executor.submit(
                _simulate_candidate_chunk,
                (
                    chunk_id,
                    pipeline_config,
                    hardware,
                    traces,
                    trace_source_question_ids,
                    prefix_caching,
                    prepared_paths,
                    chunk,
                ),
            ))
        by_ordinal: Dict[
            int,
            Tuple[int, Optional[SimResult], Optional[_SaturationOutcome], str],
        ] = {}
        for future in as_completed(futures):
            _chunk_id, chunk_results = future.result()
            for item in chunk_results:
                ordinal = int(item[0])
                if ordinal in by_ordinal:
                    raise RuntimeError(
                        f"duplicate DES signature ordinal {ordinal}"
                    )
                by_ordinal[ordinal] = item
        expected = set(range(len(candidate_rows)))
        if set(by_ordinal) != expected:
            missing = sorted(expected - set(by_ordinal))
            raise RuntimeError(
                f"parallel DES omitted signature ordinals {missing[:8]}"
            )
        return [by_ordinal[index] for index in range(len(candidate_rows))]
    except Exception as exc:
        for future in futures:
            future.cancel()
        # A broken persistent executor rejects every later task. Reset it so
        # this invocation's serial fallback does not poison the next pricing.
        if isinstance(exc, BrokenProcessPool):
            from rag_stack.cost_model.assembly import _collocation_pool_reset

            _collocation_pool_reset()
        raise


def apply_closed_loop_simulation(
    df,
    *,
    pipeline_config: dict,
    hardware,
    traces: list,
    trace_source_question_ids: Optional[Sequence[Any]] = None,
    prepared_paths: PreparedExecutionPaths | None = None,
    latency=None,
    prefix_caching: bool = True,
    builder_fn=build_sim_inputs,
    simulate_fn=simulate,
    screening=None,
):
    """Simulate a prepared shortlist and publish authoritative QPS/latency.

    The analytical min-of-caps result is retained under explicit
    ``*_cap_upper_bound`` diagnostics. Every normalized RAGO row remains one
    real deployment configuration and retains its identity in the returned
    table. ``dynamic_pipeline`` supplies the precomputed ``screening`` plan;
    the adapter executes that plan without ranking candidates again. Legacy
    direct callers may omit the plan to request objective-aware shortlisting
    here. Every shortlisted row is simulated exactly once, while screened rows
    remain explicit audit records. Signatures check identity, never group rows.

    ``latency`` is an ignored compatibility keyword. Simulation builds and
    caches its own service functions for each exact candidate layout.
    """
    if df is None or len(df) == 0:
        return df

    if builder_fn is build_sim_inputs or prepared_paths is not None:
        from rag_stack.cost_model.workload import resolve_execution_paths

        prepared_paths = resolve_execution_paths(traces, prepared_paths)
        traces = prepared_paths.paths

    cm_search_space = (
        (pipeline_config.get("system") or {}).get("cm_search_space") or {}
    )
    if not bool(cm_search_space.get("no_microbatching", True)):
        raise RuntimeError(
            "closed-loop RAG-CM requires no_microbatching=true"
        )

    # Resolve special-mode stage calibration once in the parent process,
    # before screening or candidate dispatch.  The attached snapshot is plain
    # JSON data and therefore reaches every spawned worker unchanged.
    pipeline_config = _pipeline_with_frozen_stage_batch_selection(
        pipeline_config,
        hardware,
    )

    out = df.copy()
    calibration_mode = _rag_cm_calibration_mode(pipeline_config, hardware)
    frozen_generation = _stage_batch_generation_for_pipeline(
        pipeline_config,
        hardware,
    )
    stage_batch_profile_enabled = frozen_generation.enabled
    stage_batch_profile_reason = frozen_generation.reason
    stage_batch_profile_generation_sha256 = (
        frozen_generation.generation_sha256
    )
    # These are policy diagnostics, not candidate-local observations.  Every
    # row receives exactly the same frozen values so downstream scorecards can
    # prove that no topology won by selectively omitting scheduler overhead.
    out["cm_calibration_mode"] = calibration_mode
    out["stage_batch_profile_enabled"] = stage_batch_profile_enabled
    out["stage_batch_profile_reason"] = stage_batch_profile_reason
    out["stage_batch_profile_generation_sha256"] = (
        stage_batch_profile_generation_sha256
    )
    out["stage_batch_profile_content_sha256"] = [
        dict(frozen_generation.profile_content_sha256)
        for _ in range(len(out))
    ]
    out["qps_cap_upper_bound"] = out["qps"].astype(float)
    out["latency_cap_model_s"] = out["latency_s"].astype(float)
    if "qps_per_chip" in out.columns:
        out["qps_per_chip_cap_upper_bound"] = out["qps_per_chip"].astype(float)

    production_functions = (
        builder_fn is build_sim_inputs and simulate_fn is simulate
    )
    if screening is None:
        # Compatibility for direct adapter callers. The explicit pipeline
        # performs this stage before entering the simulation adapter.
        from rag_stack.cost_model.hybrid_screening import screen_des_candidates

        screening = screen_des_candidates(
            out,
            selection=getattr(hardware, "performance", None),
            production=production_functions,
        )
    selected_positions = set(screening.selected_positions)
    if (
        len(screening.diagnostics) != len(out)
        or any(
            not isinstance(pos, Integral)
            or pos < 0 or pos >= len(out)
            for pos in selected_positions
        )
    ):
        raise ValueError("Screening plan does not match the candidate table")
    if not selected_positions:
        raise RuntimeError(
            "DynamicEngine analytical screening retained no DES candidates"
        )
    if len(selected_positions) < len(out):
        logger.info(
            "DynamicEngine analytical screen retained %d/%d exact RAGO "
            "configurations for authoritative DES (%d queue networks)",
            len(selected_positions), len(out),
            int(screening.summary.get("queue_networks_solved", 0)),
        )

    seen_signatures: Dict[Tuple[Any, ...], int] = {}
    candidates: List[Tuple[Tuple[Any, ...], int]] = []
    for pos in range(len(out)):
        _validate_candidate_batch_axes(out.iloc[pos])
        sig = simulation_signature(out.iloc[pos], pipeline_config)
        if sig in seen_signatures:
            previous = seen_signatures[sig]
            raise RuntimeError(
                "RAGO produced duplicate deployment configurations after "
                f"normalization (rows {previous} and {pos}); refusing to "
                "collapse distinct sweep rows behind one simulator signature"
            )
        seen_signatures[sig] = pos
        candidates.append((sig, pos))

    results: List[Optional[SimResult]] = [None] * len(out)
    saturation_outcomes: List[Optional[_SaturationOutcome]] = [None] * len(out)
    errors: List[str] = [""] * len(out)
    signature_ids: List[str] = [""] * len(out)
    import hashlib
    candidate_rows: List[_CandidateWork] = []
    signature_positions: List[int] = []
    for signature, pos in candidates:
        signature_id = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
        signature_ids[pos] = signature_id
        if pos not in selected_positions:
            errors[pos] = "dynamic_analytical_screened_out"
            continue
        ordinal = len(candidate_rows)
        candidate_rows.append(_CandidateWork(
            ordinal=ordinal,
            row=out.iloc[pos],
            signature=signature,
        ))
        signature_positions.append(pos)

    # Only the production builder/simulator pair is process-safe.  Tests and
    # downstream callers that inject either callable retain the historical
    # in-process behavior, including a single latency cache across signatures.
    workers = (
        _des_worker_count(len(candidate_rows))
        if production_functions else 1
    )
    evaluated = None
    if workers > 1:
        try:
            evaluated = _evaluate_candidates_in_processes(
                candidate_rows,
                workers=workers,
                pipeline_config=pipeline_config,
                hardware=hardware,
                traces=traces,
                trace_source_question_ids=trace_source_question_ids,
                prepared_paths=prepared_paths,
                prefix_caching=prefix_caching,
            )
            logger.info(
                "Priced %d exact RAGO configurations with up to %d "
                "worker processes",
                len(candidate_rows), workers,
            )
        except Exception as exc:  # noqa: BLE001 - infrastructure fallback
            logger.warning(
                "Parallel DES unavailable (%s: %s); retrying signatures "
                "serially without changing simulator inputs",
                type(exc).__name__, exc,
            )

    if evaluated is None:
        evaluated = _evaluate_candidate_rows(
            candidate_rows,
            pipeline_config=pipeline_config,
            hardware=hardware,
            traces=traces,
            trace_source_question_ids=trace_source_question_ids,
            prepared_paths=prepared_paths,
            prefix_caching=prefix_caching,
            builder_fn=builder_fn,
            simulate_fn=simulate_fn,
        )

    for ordinal, result, saturation, error in evaluated:
        pos = signature_positions[ordinal]
        results[pos] = result
        saturation_outcomes[pos] = saturation
        errors[pos] = error

    if not any(result is not None for result in results):
        unique_errors = sorted({error for error in errors if error})
        raise RuntimeError(
            "closed-loop simulator rejected every dynamic deployment: "
            + " | ".join(unique_errors[:5])
        )

    simulator_qps = [
        result.qps if result is not None else 0.0 for result in results
    ]
    simulator_latency_s = [
        result.latency_mean_s if result is not None else math.inf
        for result in results
    ]
    qps = list(simulator_qps)
    latency_s = list(simulator_latency_s)

    out["simulator_qps"] = simulator_qps
    out["simulator_latency_mean_s"] = simulator_latency_s
    out["simulator_status"] = [
        "ok" if result is not None
        else "screened_out" if pos not in selected_positions
        else "infeasible"
        for pos, result in enumerate(results)
    ]
    out["simulator_error"] = errors
    out["simulator_signature"] = signature_ids
    out["simulator_unique_deployments"] = len(candidates)
    out["simulator_evaluated_deployments"] = len(candidate_rows)
    out["hybrid_total_candidates"] = len(out)
    out["hybrid_shortlisted_candidates"] = len(candidate_rows)
    out["hybrid_screened_candidates"] = len(out) - len(candidate_rows)
    out["hybrid_screening_mode"] = str(screening.summary.get("mode", "exhaustive"))
    out["hybrid_screening_score"] = str(screening.summary.get("score", ""))
    out["hybrid_screening_selection"] = str(screening.summary.get("selection", ""))
    out["hybrid_screening_metrics"] = [
        tuple(screening.summary.get("screening_metrics", ())) for _ in range(len(out))
    ]
    out["hybrid_parents_per_topology"] = int(
        screening.summary.get("parents_per_topology", 0)
    )
    screening_diagnostics = screening.diagnostics
    screening_columns = {
        "selection_reasons": "hybrid_selection_reasons",
        "exclusion_reason": "hybrid_exclusion_reason",
        "dynamic_analytical_qps": "dynamic_analytical_qps",
        "analytical_screening_qps": "analytical_screening_qps",
        "analytical_screening_score": "analytical_screening_score",
        "analytical_screening_metric": "analytical_screening_metric",
        "analytical_screening_higher_is_better": "analytical_screening_higher_is_better",
        "analytical_screening_score_name": (
            "analytical_screening_score_name"
        ),
        "analytical_screening_is_final": (
            "analytical_screening_is_final"
        ),
        "analytical_dataflow": "analytical_dataflow",
        "analytical_generator_pd": "analytical_generator_pd",
        "analytical_query_expansion_pd": (
            "analytical_query_expansion_pd"
        ),
        "topology_rank_within_stratum": "hybrid_topology_rank_within_stratum",
        "parent_rank_within_topology": "hybrid_parent_rank_within_topology",
        "global_parent_rank": "hybrid_global_parent_rank",
    }
    for source, target in screening_columns.items():
        if source in screening_diagnostics.columns:
            out[target] = screening_diagnostics[source].tolist()
    out["simulator_engine_fill_mean"] = [
        result.engine_fill_mean if result is not None else {}
        for result in results
    ]
    out["simulator_stage_batch_mean"] = [
        result.stage_batch_mean if result is not None else {}
        for result in results
    ]
    out["simulator_transfer_latency_mean"] = [
        result.transfer_latency_mean if result is not None else {}
        for result in results
    ]
    out["simulator_transfer_queue_wait_mean"] = [
        result.transfer_queue_wait_mean if result is not None else {}
        for result in results
    ]
    out["simulator_populations_tried"] = [
        outcome.populations_tried if outcome is not None else ()
        for outcome in saturation_outcomes
    ]
    out["simulator_saturation_plateau"] = [
        outcome.plateau_reached if outcome is not None else False
        for outcome in saturation_outcomes
    ]
    out["simulator_saturation_stop_reason"] = [
        outcome.stop_reason if outcome is not None
        else "screened_out" if pos not in selected_positions
        else "infeasible"
        for pos, outcome in enumerate(saturation_outcomes)
    ]
    out["load_concurrency"] = [
        outcome.final_population if outcome is not None else 0
        for outcome in saturation_outcomes
    ]
    out["qps"] = qps
    out["latency_s"] = latency_s
    if "num_chips_e2e" in out.columns:
        out["qps_per_chip"] = [
            qps[pos] / max(1, int(out["num_chips_e2e"].iloc[pos]))
            for pos in range(len(out))
        ]
    out["dynamic_performance_model"] = "closed_loop_simulator"
    return out
