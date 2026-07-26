"""Global types shared across rag_stack modules."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CMDeviceCapacityError(RuntimeError):
    """One exact-device GenZ pricing attempt exceeds device memory.

    This is deliberately narrower than Python's :class:`MemoryError`: no
    allocation failed in the controller process.  GenZ proved that the
    requested weights plus KV cache cannot fit the target device.  The
    original GenZ exception is retained through exception chaining.
    """


class CMOutOfMemoryError(CMDeviceCapacityError):
    """Every concrete system candidate is memory-infeasible in the CM."""


class EvalType(str, Enum):
    """Outcome category of one optimization evaluation (one sampled config).

    Persisted as a plain string in each eval record and in
    ``all_evaluations.csv`` — this is a ``str`` subclass, so ``EvalType.GT ==
    "GT"`` and it serializes to ``"GT"`` (the ``__str__`` override makes
    ``str(EvalType.GT) == "GT"`` on every Python version, which the CSV writer
    relies on).

    The category is a COARSE bucket of *what happened*; the precise cause of a
    failure is recorded separately in the ``eval_error`` column of the row (so
    you never have to decode it from the type). Two success outcomes and three
    failure outcomes:

    SUCCESS — produced real objective values that the optimizer observes:
      * ``GT`` — ground-truth eval: real quality (LLM judge) and, in measured
        mode, real performance from the SAME run.
      * ``CM`` — cost-model eval: performance only (RAGO prediction), no quality
        (this eval was requested perf-only).

    FAILURE — no real scores for this config (see ``eval_error`` for the reason):
      * ``DUPLICATE`` — this config was already evaluated/known-bad in an earlier
        iteration, so it was not re-run.
      * ``OOM`` — the CM proved that every concrete deployment candidate is
        unable to fit its target device memory (weights / KV cache), or the
        measured evaluator reported a CUDA / launch OOM for the sampled
        deployment.  A static-quality-harness OOM is infrastructure failure,
        not a sampled-config verdict.
      * ``CONTEXT_OVERFLOW`` — the prompt exceeded the model's context window
        ("maximum context length is N tokens ..."). A property of the sampled
        config (top_k × chunk_size × prompt vs the generator's window), so the
        same config overflows every time.
      * ``INVALID`` — any OTHER failure: ``resolve_config`` / chunk-cache build
        failed, the derived layout exceeded the GPU budget, a vLLM launch failed
        (non-OOM), or a GT/CM eval errored after retries. The ``eval_error``
        column says exactly which.

    Retry policy is provenance-aware.  Measured OOM and CONTEXT_OVERFLOW are
    deterministic sampled-config failures, while a static quality-harness OOM
    is retried and aborts if it persists.  CM OOM is accepted only through the
    typed all-candidate capacity proof; every other CM exception is a bug.

    Resume reloads SUCCESS rows and replays exact zero-objective penalties for
    OOM, CONTEXT_OVERFLOW, and typed deployment-preset infeasibility. Other
    INVALID rows and DUPLICATE rows never become optimizer observations.
    """

    GT = "GT"
    CM = "CM"
    # Fingerprint-deduplicated GT (07-07): the generation INPUTS (queries,
    # final retrieved contexts, prompts, generator config) were byte-identical
    # to an earlier eval's, so quality was INHERITED from that donor instead of
    # re-sampling the same distribution (generation + judge skipped).
    # Performance is real (CM prices THIS config; retrieval-side knobs differ).
    # A success row with real objective values — but it consumes NO valid-eval
    # budget seat (the measurement carried zero new quality information).
    GT_FP = "GT_FP"
    DUPLICATE = "DUPLICATE"
    OOM = "OOM"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    INVALID = "INVALID"

    def __str__(self) -> str:  # so str()/CSV write the value, not "EvalType.GT"
        return self.value

    @property
    def is_success(self) -> bool:
        """True for GT/GT_FP/CM — the config yielded real objective values."""
        return self in (EvalType.GT, EvalType.GT_FP, EvalType.CM)

    @property
    def is_retryable_failure(self) -> bool:
        """Whether a failure of this category is worth retrying. Deterministic
        failures (OOM, CONTEXT_OVERFLOW, DUPLICATE) reproduce identically on
        every attempt; only the generic INVALID bucket can hide transient causes
        (engine spawn races, judge API timeouts, retrieval flakes)."""
        return self is EvalType.INVALID

    @classmethod
    def classify_failure(cls, error: object) -> "EvalType":
        """Bucket a failure into ``OOM`` / ``CONTEXT_OVERFLOW`` / ``INVALID``
        from the exception / message text. The full text is kept verbatim in the
        row's ``eval_error`` column regardless."""
        msg = str(error).lower()
        if (
            # vLLM / OpenAI-compat servers: "This model's maximum context length
            # is 32768 tokens. However, you requested ... input_tokens ..."
            "maximum context length" in msg
            # OpenAI-style error code (LLM-judge providers)
            or "context_length_exceeded" in msg
        ):
            return cls.CONTEXT_OVERFLOW
        if (
            "out of memory" in msg
            or "outofmemory" in msg
            or "cuda oom" in msg
            or "alloc_failed" in msg
        ):
            return cls.OOM
        return cls.INVALID

    @classmethod
    def classify_cm_failure(cls, error: object) -> "EvalType":
        """Classify a failure known to originate inside the cost model.

        Only the typed aggregate emitted after *every* concrete candidate is
        proven memory-infeasible is an allowed CM OOM.  Message matching here
        would let one failed TP/PP attempt, or even an unrelated code error
        containing the words "out of memory", hide as a config verdict.
        """
        if isinstance(error, CMOutOfMemoryError):
            return cls.OOM
        return cls.INVALID

    @classmethod
    def excluded_from_data(cls) -> frozenset[str]:
        """Row ``eval_type`` values that are NOT reloaded as optimizer
        observations on resume / into the quality store — i.e. everything that is
        not a SUCCESS (failures carry no objective values). Legacy aliases
        (pre-rename ``SKIPPED`` / ``INVALID_PENALIZED``) and ``""`` are included
        so old ``all_evaluations.csv`` files still filter correctly.

        Checkpoint restore additionally replays deterministic penalty rows as
        GP observations — see ``Controller._load_checkpoint``. The quality
        store keeps using THIS set unchanged because penalty rows carry no real
        judge scores."""
        return frozenset({
            cls.DUPLICATE.value, cls.OOM.value, cls.CONTEXT_OVERFLOW.value,
            cls.INVALID.value, "",
            "SKIPPED", "INVALID_PENALIZED",  # legacy aliases (old CSVs)
        })

    @classmethod
    def reload_as_penalty(cls) -> frozenset[str]:
        """Row ``eval_type`` values whose 0-penalty objectives ARE replayed
        into the optimizer on resume (user rule 07-06: ONLY genuine
        config-property infeasibilities — GT context overflow and CM OOM —
        may feed the GP; every other failure stays excluded). Keeps the
        live-fed GP trough across resumes without ever counting these rows
        toward the valid-eval budget."""
        return frozenset({cls.OOM.value, cls.CONTEXT_OVERFLOW.value})


@dataclass
class MemoryTier:
    """A single level in the memory hierarchy.

    Bandwidth is stored at two explicit granularities (one consistent semantic
    everywhere — no ``memory_efficiency`` knob):

    - ``bandwidth``    : AGGREGATE (all-core sustained STREAM) at this tier.
    - ``bandwidth_1t`` : SINGLE-THREAD (one-core sustained STREAM) at this tier.

    Effective bandwidth at ``t`` threads is ``min(t*bandwidth_1t, bandwidth)``.
    The two are INDEPENDENT hardware facts: single-thread is limited by per-core
    MLP/latency (~constant per core), while aggregate scales with memory channels
    / core count / NUMA sockets. Single-thread is therefore NEVER derived from
    aggregate — it must be measured (STREAM, 1 thread). ``bandwidth_1t`` defaults
    to 0 = "not provided"; the cost-model hardware builder errors if it needs it.
    """
    name: str           # "llc", "ddr", "storage"
    bandwidth: float    # bytes/s — aggregate (all-core sustained)
    capacity: float     # bytes
    bandwidth_1t: float = 0.0   # bytes/s — single-thread (one-core sustained); 0 = derive


@dataclass
class PerformanceObjective:
    """A single performance metric and its direction."""
    metric: str                # "latency_s", "throughput_qps"
    lower_is_better: bool


@dataclass
class PerformanceSelection:
    """Configures how to select the best hw_config from a hardware sweep.

    All returned scores follow the project's higher-is-better convention,
    so MOBO can directly maximize them.

    Selection modes:
        min_latency:   Pick the hw_config with the lowest latency_s (E2E).
        min_ttft:      Pick the hw_config with the lowest time-to-first-token.
        min_tpot:      Pick the hw_config with the lowest time-per-output-token.
        max_throughput: Pick the hw_config with the highest throughput_qps.
        weighted:      Maximize a weighted sum of normalized metrics (each
                       metric is first converted to higher-is-better, then
                       min-max normalized within the current sweep).
                       Requires ``weights`` dict.
        constrained:   Satisfy ``constraint`` (natural sense: ``latency ≤ L*``,
                       ``qps ≥ Q*``), then optimize ``optimize``. If no point
                       is feasible, returns the ``INFEASIBLE_SCORE`` sentinel
                       so MOBO sees the SLO boundary as a hard cliff.

    Supported metric keys (declared in ``cost_model.selection.METRIC_SPEC``):
        ``latency_s``, ``ttft_s``, ``tpot_s``,
        ``throughput_qps``, ``throughput_qps_per_chip``.
    """
    objectives: list[PerformanceObjective] = field(default_factory=list)
    selection: str = "min_latency"
    weights: dict[str, float] | None = None       # for "weighted"
    constraint: dict[str, float] | None = None    # for "constrained"
    optimize: str | None = None                    # for "constrained"

    @classmethod
    def from_dict(cls, d: dict) -> PerformanceSelection:
        objectives = [
            PerformanceObjective(
                metric=m["metric"],
                lower_is_better=m.get("lower_is_better", True),
            )
            for m in d.get("metrics", [])
        ]
        return cls(
            objectives=objectives,
            selection=d.get("selection", "min_latency"),
            weights=d.get("weights"),
            constraint=d.get("constraint"),
            optimize=d.get("optimize"),
        )


@dataclass
class SystemConfig:
    """System configuration: hardware specs + deployment/scheduling params.

    Contains everything that affects performance but NOT quality:
    - CPU specs for retrieval cost models (roofline-based)
    - GPU specs for LLM cost models (GenZ format)
    - search_space: deployment params swept by cost model (not MOBO)
    - performance: how to select the best result from the sweep
    """
    # CPU (for retrieval cost models — roofline)
    # str = preset name (e.g. "epyc_9124"), "" = custom dict specs.
    cpu_preset: str = ""
    # CPU hardware key (sibling of gpu.hardware_key) — for an inline cpu dict, its
    # ``hardware_key``; for a preset, ``cpu_preset`` is the key. Read via ``cpu_key``.
    cpu_hardware_key: str = ""
    num_cores: int = 1
    peak_flops: float = 0.0           # Peak FLOP/s (all cores)
    peak_int_ops: float = 0.0         # Peak INT-OP/s (for PQ lookups)
    mem_bandwidth: float = 0.0        # AGGREGATE DRAM bandwidth (bytes/s, all-core sustained)
    mem_bandwidth_1t: float = 0.0     # SINGLE-THREAD DRAM bandwidth (bytes/s, one-core sustained)
    memory_tiers: list[MemoryTier] = field(default_factory=list)
    # Heterogeneous core topology (big.LITTLE). All zero = homogeneous.
    num_p_cores: int = 0              # Performance cores (0 = use num_cores as homogeneous)
    num_e_cores: int = 0              # Efficiency cores
    p_core_clock_ghz: float = 0.0    # P-core clock speed (GHz)
    e_core_clock_ghz: float = 0.0    # E-core clock speed (GHz)
    p_core_flops_per_cycle: int = 0   # FP ops per cycle per P-core (e.g. 32 for ARM NEON 4×FMA)
    e_core_flops_per_cycle: int = 0   # FP ops per cycle per E-core
    p_core_int_ops_per_cycle: int = 0 # INT ops per cycle per P-core
    e_core_int_ops_per_cycle: int = 0 # INT ops per cycle per E-core
    thread_migration: bool = True     # OS freely migrates threads between P/E cores
    # CPU topology discriminator (hardware fact, NEVER calibrated).
    #   "uniform"        — single socket, homogeneous cores (default)
    #   "numa"           — multi-socket NUMA; ``numa_sockets`` /
    #                      ``numa_cores_per_socket`` carry the physical layout
    #   "heterogeneous"  — single-socket big.LITTLE; ``num_p_cores`` /
    #                      ``num_e_cores`` etc carry the P/E spec
    # These three are mutually exclusive; fused NUMA+big.LITTLE systems are not
    # currently supported.
    topology_type: str = "uniform"
    numa_sockets: int = 1
    numa_cores_per_socket: int = 0    # 0 = derive from num_cores // numa_sockets
    # GPU (for LLM cost models, GenZ format)
    # str = GenZ predefined name (e.g. "A100_80GB_GPU"), dict = custom specs.
    # When a dict, the optional ``hardware_key`` field doubles as the stable
    # calibration registry key (calibration_profiles/llm_sim/). Inline GPU
    # specifications must provide it; local CUDA identity is never consulted.
    gpu: dict[str, Any] | str | None = None
    # Declarative interconnect fabric for the communication model, per
    # device-pair (gpu_gpu / gpu_cpu). A SYSTEM-level sibling of `gpu`. Each pair
    # is a dict {bandwidth_gbps, latency_us?, type?} with the numbers written
    # DIRECTLY in YAML (no preset table; `type` is a documentary label only).
    # Absent / no bandwidth_gbps → ICN fallback. See
    # rag_stack.cost_model.communication._resolve_link.
    interconnect: dict[str, Any] | None = None
    # Performance-only search space (swept by cost model, NOT by MOBO)
    search_space: dict[str, list] | None = None
    # How to select best performance from hardware sweep
    performance: PerformanceSelection | None = None
    # Stable exact HardwareCatalog topology key used by communication and
    # serving-curve provenance. It is physical-system identity, not deployment.
    system_hardware: str = ""
    # Cost-model runtime policy. Keep the mapping losslessly: hardware replay
    # reconstructs SystemConfig from ``to_dict()``, so dropping this block
    # would silently change an explicitly requested calibration mode on the
    # target hardware. Default validation permits only calibration_mode;
    # profile roots are process-wide filesystem contracts.
    rag_cm: dict[str, Any] | None = None

    @property
    def cpu_key(self) -> str:
        """The CPU hardware key: inline ``cpu.hardware_key`` if given, else the preset name."""
        return self.cpu_hardware_key or self.cpu_preset

    def get_bandwidth(self, tier_name: str) -> float:
        """Return bandwidth for a memory tier by name."""
        for t in self.memory_tiers:
            if t.name == tier_name:
                return t.bandwidth
        raise KeyError(f"Unknown memory tier: {tier_name}")

    def get_capacity(self, tier_name: str) -> float:
        """Return capacity for a memory tier by name."""
        for t in self.memory_tiers:
            if t.name == tier_name:
                return t.capacity
        raise KeyError(f"Unknown memory tier: {tier_name}")

    def to_dict(self) -> dict:
        """Serialize to a dict compatible with ``from_dict``."""
        cpu: dict = {
            "num_cores": self.num_cores,
            "peak_flops": self.peak_flops,
            "peak_int_ops": self.peak_int_ops,
            "mem_bandwidth": self.mem_bandwidth,
            "mem_bandwidth_1t": self.mem_bandwidth_1t,
        }
        if self.cpu_hardware_key:
            cpu["hardware_key"] = self.cpu_hardware_key
        if self.memory_tiers:
            cpu["memory_tiers"] = [
                {"name": t.name, "bandwidth": t.bandwidth,
                 "bandwidth_1t": t.bandwidth_1t, "capacity": t.capacity}
                for t in self.memory_tiers
            ]
        # Topology block — emit only when non-uniform. For uniform, the
        # flat heterogeneous fields (num_p_cores etc) are all 0 and
        # nothing is added.
        if self.topology_type == "numa":
            cpu["topology"] = {
                "type": "numa",
                "sockets": self.numa_sockets,
                "cores_per_socket": self.numa_cores_per_socket or (
                    self.num_cores // max(1, self.numa_sockets)
                ),
            }
        elif self.topology_type == "heterogeneous":
            topo: dict = {"type": "heterogeneous"}
            p_block: dict = {"count": self.num_p_cores}
            e_block: dict = {"count": self.num_e_cores}
            for src_attr, dst_key in (
                ("p_core_clock_ghz", "clock_ghz"),
                ("p_core_flops_per_cycle", "flops_per_cycle"),
                ("p_core_int_ops_per_cycle", "int_ops_per_cycle"),
            ):
                v = getattr(self, src_attr)
                if v:
                    p_block[dst_key] = v
            for src_attr, dst_key in (
                ("e_core_clock_ghz", "clock_ghz"),
                ("e_core_flops_per_cycle", "flops_per_cycle"),
                ("e_core_int_ops_per_cycle", "int_ops_per_cycle"),
            ):
                v = getattr(self, src_attr)
                if v:
                    e_block[dst_key] = v
            topo["p_cores"] = p_block
            topo["e_cores"] = e_block
            if not self.thread_migration:
                topo["thread_migration"] = False
            cpu["topology"] = topo
        else:
            # Uniform: legacy flat fields (preserved for backward compat
            # with YAMLs that pre-date the topology block).
            for attr in (
                "num_p_cores", "num_e_cores", "p_core_clock_ghz", "e_core_clock_ghz",
                "p_core_flops_per_cycle", "e_core_flops_per_cycle",
                "p_core_int_ops_per_cycle", "e_core_int_ops_per_cycle",
            ):
                val = getattr(self, attr)
                if val:
                    cpu[attr] = val
            if not self.thread_migration:
                cpu["thread_migration"] = False

        result: dict = {}
        if self.system_hardware:
            result["system_hardware"] = self.system_hardware
        if self.rag_cm is not None:
            result["rag_cm"] = deepcopy(self.rag_cm)
        if self.cpu_preset:
            result["cpu"] = self.cpu_preset
        else:
            result["cpu"] = cpu
        if self.gpu is not None:
            result["gpu"] = self.gpu
        if self.interconnect is not None:
            result["interconnect"] = self.interconnect
        if self.search_space is not None:
            # ``cm_search_space`` is the canonical YAML/wire key.  Emitting the
            # dataclass attribute name (``search_space``) here used to break the
            # documented ``from_dict(to_dict())`` contract because ``from_dict``
            # only consumed ``cm_search_space``.  Keep nested axes isolated from
            # callers as well: lists such as placement_policy and batch grids are
            # mutable and must not alias the live SystemConfig.
            result["cm_search_space"] = deepcopy(self.search_space)
        if self.performance is not None:
            perf: dict = {"selection": self.performance.selection}
            if self.performance.objectives:
                perf["metrics"] = [
                    {"metric": o.metric, "lower_is_better": o.lower_is_better}
                    for o in self.performance.objectives
                ]
            if self.performance.weights:
                perf["weights"] = self.performance.weights
            if self.performance.constraint:
                perf["constraint"] = self.performance.constraint
            if self.performance.optimize:
                perf["optimize"] = self.performance.optimize
            result["performance_objectives"] = perf
        return result

    @classmethod
    def from_dict(cls, d: dict) -> SystemConfig:
        """Parse from YAML system section.

        Supports nested ``cpu:`` / ``gpu:`` structure::

            system:
              cpu:
                num_cores: 12
                peak_flops: 1.04e12
                peak_int_ops: 1.04e12
                mem_bandwidth: 116.5e9
                memory_tiers: [{name: ddr, bandwidth: 100e9, capacity: 32e9}]
              gpu:
                Flops: 312
                Memory_BW: 2039
              cm_search_space:
                min_num_gpus: 4
                max_num_gpus: 16
                gpus_per_server: 4
        """
        # CPU specs (nested under "cpu" key)
        # Supports two formats (same as gpu):
        #   cpu: epyc_9124           # string → preset name
        #   cpu: {num_cores: 16, …}  # dict → custom specs
        raw_cpu = d.get("cpu", {})
        if isinstance(raw_cpu, str):
            cpu_preset = raw_cpu
            cpu: dict = {}
        else:
            cpu_preset = ""
            cpu = raw_cpu or d  # fallback: flat layout for backward compat
        cpu_hardware_key = raw_cpu.get("hardware_key", "") if isinstance(raw_cpu, dict) else ""

        peak_flops = float(cpu.get("peak_flops", 0.0))
        peak_int_ops = float(cpu.get("peak_int_ops", peak_flops))
        mem_bandwidth = float(cpu.get("mem_bandwidth", 0.0))
        # Single-thread DRAM bandwidth is an INDEPENDENT hardware fact (per-core
        # MLP/latency limited, ~constant per core) — it is NOT a fraction of the
        # aggregate (which scales with channels / core count / NUMA sockets), so
        # it is NEVER derived here. Provide it explicitly (measure with STREAM at
        # 1 thread) or use a profiled cpu preset. 0.0 = "not provided"; the cost
        # model's hardware builder raises a clear error if it actually needs it.
        mem_bandwidth_1t = float(cpu.get("mem_bandwidth_1t", 0.0))

        tiers = [
            MemoryTier(
                name=t["name"],
                bandwidth=float(t["bandwidth"]),
                capacity=float(t.get("capacity", 0)),
                bandwidth_1t=float(t.get("bandwidth_1t", 0.0)),
            )
            for t in cpu.get("memory_tiers", [])
        ]

        perf_dict = d.get("performance_objectives")
        performance = (
            PerformanceSelection.from_dict(perf_dict) if perf_dict else None
        )

        # ``cm_search_space`` is the canonical six-section config key.  Accept
        # the historical serialized alias ``search_space`` on input so existing
        # hardware JSON snapshots remain readable, but always canonicalize it on
        # the next :meth:`to_dict` call.
        search_space = d.get("cm_search_space")
        if search_space is None:
            search_space = d.get("search_space")
        if search_space is not None:
            search_space = deepcopy(search_space)

        # ``cpu.topology:`` block — discriminated by ``type``. When absent
        # (or type=uniform), legacy flat heterogeneous fields are read for
        # backward compat with pre-topology YAMLs that used
        # ``cpu.num_p_cores: 6`` etc directly.
        topology = cpu.get("topology") or {}
        topology_type = str(topology.get("type", "uniform")).lower()

        num_cores = int(cpu.get("num_cores", 1))

        # Heterogeneous (big.LITTLE) — prefer new topology shape, fall back
        # to flat fields. Cross-check happens in ConfigValidator.
        if topology_type == "heterogeneous":
            p = topology.get("p_cores") or {}
            e = topology.get("e_cores") or {}
            num_p_cores = int(p.get("count", 0))
            num_e_cores = int(e.get("count", 0))
            p_core_clock_ghz = float(p.get("clock_ghz", 0.0))
            e_core_clock_ghz = float(e.get("clock_ghz", 0.0))
            p_core_flops_per_cycle = int(p.get("flops_per_cycle", 0))
            e_core_flops_per_cycle = int(e.get("flops_per_cycle", 0))
            p_core_int_ops_per_cycle = int(p.get("int_ops_per_cycle", 0))
            e_core_int_ops_per_cycle = int(e.get("int_ops_per_cycle", 0))
            thread_migration = bool(topology.get("thread_migration", True))
        else:
            num_p_cores = int(cpu.get("num_p_cores", 0))
            num_e_cores = int(cpu.get("num_e_cores", 0))
            p_core_clock_ghz = float(cpu.get("p_core_clock_ghz", 0.0))
            e_core_clock_ghz = float(cpu.get("e_core_clock_ghz", 0.0))
            p_core_flops_per_cycle = int(cpu.get("p_core_flops_per_cycle", 0))
            e_core_flops_per_cycle = int(cpu.get("e_core_flops_per_cycle", 0))
            p_core_int_ops_per_cycle = int(cpu.get("p_core_int_ops_per_cycle", 0))
            e_core_int_ops_per_cycle = int(cpu.get("e_core_int_ops_per_cycle", 0))
            thread_migration = bool(cpu.get("thread_migration", True))

        # NUMA fields — only meaningful when type=numa
        if topology_type == "numa":
            numa_sockets = int(topology.get("sockets", 1))
            numa_cores_per_socket = int(topology.get(
                "cores_per_socket", num_cores // max(1, numa_sockets),
            ))
        else:
            numa_sockets = 1
            numa_cores_per_socket = 0

        return cls(
            cpu_preset=cpu_preset,
            cpu_hardware_key=cpu_hardware_key,
            num_cores=num_cores,
            peak_flops=peak_flops,
            peak_int_ops=peak_int_ops,
            mem_bandwidth=mem_bandwidth,
            mem_bandwidth_1t=mem_bandwidth_1t,
            memory_tiers=tiers,
            num_p_cores=num_p_cores,
            num_e_cores=num_e_cores,
            p_core_clock_ghz=p_core_clock_ghz,
            e_core_clock_ghz=e_core_clock_ghz,
            p_core_flops_per_cycle=p_core_flops_per_cycle,
            e_core_flops_per_cycle=e_core_flops_per_cycle,
            p_core_int_ops_per_cycle=p_core_int_ops_per_cycle,
            e_core_int_ops_per_cycle=e_core_int_ops_per_cycle,
            thread_migration=thread_migration,
            topology_type=topology_type,
            numa_sockets=numa_sockets,
            numa_cores_per_socket=numa_cores_per_socket,
            gpu=d.get("gpu"),
            interconnect=d.get("interconnect"),
            search_space=search_space,
            performance=performance,
            system_hardware=str(d.get("system_hardware") or ""),
            rag_cm=(
                deepcopy(d["rag_cm"])
                if isinstance(d.get("rag_cm"), dict)
                else None
            ),
        )
