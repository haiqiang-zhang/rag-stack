"""GenZ LLM Simulator adapters with RAGO-style sweep logic.

Following rago/llm_sim/genz_scripts/llm_perf.py, provides the LLM-shaped cost
model modules:
- MainLLMCostModel:        Generative LLM (prefill + decode)
- QueryExpansionCostModel: Query expansion LLM (prefill + decode, short sequences)
- PassageRerankerCostModel: Passage reranker (prefill only, encoder model)
(The query-embedding encoder lives in rag_stack/cost_model/retrieval_encode.py.)

All share GenZLLMSimBase which handles:
- (TP, PP) parallelism enumeration per num_devices
- Batch size doubling until OOM
- Best latency selection per (num_gpus, batch_size)
- Pareto filtering of dominated points

Token Size Auto-Inference
-------------------------
GenZ requires ``input_tokens`` and ``output_tokens`` per request for each stage.
These are NOT hardcoded — they are auto-computed from the actual dataset by
``Controller._inject_token_stats()`` and stored in ``pipeline_config["_token_stats"]``.

Raw stats computed by Controller (using the generator LLM's tokenizer):
    avg_chunk_tokens        Average tokens per corpus document.
    avg_query_tokens        Average tokens per query in the QA dataset.
    avg_output_tokens       Average tokens per generation ground truth answer.
    prompt_template_tokens  Tokens in the prompt template (excluding placeholders).

Each CostModel subclass reads these raw stats and applies its own formula:

    MainLLMCostModel (generator):
        input_tokens  = prompt_template_tokens + avg_query_tokens
                        + effective_top_k * avg_chunk_tokens
                        (effective_top_k = running min over the retrieval →
                         rerank/filter chain — what actually reaches the prompt)
        output_tokens = avg_output_tokens

    QueryExpansionCostModel (hyde / query_decompose / multi_query_expansion):
        input_tokens  = avg_query_tokens
        output_tokens = avg_chunk_tokens  (hyde: generates a fake passage)
                      = avg_query_tokens  (others: generates rewritten queries)

    PassageRerankerCostModel (cross-encoder):
        input_tokens  = avg_query_tokens + avg_chunk_tokens  (query + passage concat)
        output_tokens = 1  (encoder-only, outputs a relevance score)

    RetrievalEncodeCostModel (query embedding; see retrieval_encode.py):
        input_tokens  = avg_query_tokens  (query-time encode)
        output_tokens = 1  (pooled vector, no generation)
"""

from __future__ import annotations

import logging
from abc import abstractmethod

import pandas as pd

from rag_stack.cost_model.llm_sim.genz_cache import (
    cached_prefill_moddeling as prefill_moddeling,
    cached_decode_moddeling as decode_moddeling,
    scheduler_batch_traversal_latency_ms,
)
from rag_stack.cost_model.llm_sim.tp_efficiency import (
    pipeline_fill_drain_factor,
)
from GenZ.Models.default_models import ModelConfig
from GenZ.parallelism import (
    check_model_parallelism,
    valid_parallelism_configs,
)

from rag_stack.cost_model import register_cost_model
from rag_stack.cost_model.base import CostModelModule
from rag_stack.cost_model.calibration_mode import is_uncalibrated_spec_mode
from rag_stack.cost_model.types import CostModelResult, OperatorBreakdown
from rag_stack.cost_model.utils import _get_pareto_df, _get_powers_of_two_up_to
from rag_stack.types import CMDeviceCapacityError, SystemConfig

logger = logging.getLogger("RAG-Stack")

_SPEC_FALLBACK_WARNED: set[tuple[str, str, str, str]] = set()


class GenZCalibrationError(ValueError):
    """Production GenZ pricing cannot resolve an exact calibration record."""


def _is_exact_device_capacity_failure(exc: BaseException) -> bool:
    """Recognize GenZ's analytic per-chip weights+KV capacity verdict.

    Keep this deliberately exact and local to the GenZ boundary.  Generic
    ``ValueError`` or ``MemoryError`` instances are code/infrastructure bugs,
    not proof that a sampled deployment is memory-infeasible.
    """
    if not isinstance(exc, ValueError):
        return False
    msg = str(exc).lower()
    legacy_total_memory_verdict = (
        "all params would not fit on chip" in msg
        and "system memory cap:" in msg
        and "weights" in msg
        and "kv cache:" in msg
    )
    max_rank_memory_verdict = (
        "all params would not fit on the largest pp rank" in msg
        and "system memory cap:" in msg
        and "max rank resident memory:" in msg
    )
    return legacy_total_memory_verdict or max_rank_memory_verdict


def _validate_calibration_identity(
    record,
    *,
    component: str,
    model_key: str,
    hardware_key: str,
    bits_hint: str | None,
) -> None:
    """Reject a keyed profile whose payload claims a different identity."""
    allowed_components = {"agnostic"}
    if component:
        allowed_components.add(component)
    mismatches = []
    if record.component not in allowed_components:
        mismatches.append(
            f"component={record.component!r} (expected one of "
            f"{sorted(allowed_components)!r})"
        )
    if record.model_key != model_key:
        mismatches.append(
            f"model={record.model_key!r} (expected {model_key!r})"
        )
    if record.hardware_key != hardware_key:
        mismatches.append(
            f"hardware={record.hardware_key!r} (expected {hardware_key!r})"
        )
    if bits_hint is not None and record.bits != bits_hint:
        mismatches.append(
            f"bits={record.bits!r} (expected {bits_hint!r})"
        )
    if mismatches:
        raise GenZCalibrationError(
            "GenZ calibration record identity mismatch: " + "; ".join(mismatches)
        )


def _is_prefill_decode_engine_stage(stage: str) -> bool:
    """Whether ``stage`` belongs to a real two-role LLM serving engine."""
    from rag_stack.taxonomy import TAXONOMY

    # Test the concrete PerformanceStage, not its parent user stage.  The
    # passage-reranker parent contains both forward-only components (for
    # example MonoT5/TART) and LLM-shaped components (RankGPT/UPR).  Folding a
    # bare ``passage_reranker`` through that heterogeneous parent falsely
    # classifies the forward components as two-role LLM engines and makes them
    # consume generator evidence, TP/PP, and stage-batch curves.
    return str(stage) in TAXONOMY.llm_performance_stages


_SWEEP_RESOURCE_BATCH_IDENTITY = (
    "model.name",
    "model.stage",
    "model.dec_steps",
    "model.seq_len_inference_prefill",
    "hardware.name",
    "hardware.num_chips",
    "batch_size",
)


def _reduce_parallelism_sweep(
    frame: pd.DataFrame,
    *,
    preserve_batch_axis: bool,
) -> pd.DataFrame:
    """Choose TP/PP per resource+batch, then optionally Pareto-reduce batches.

    ``preserve_batch_axis`` is required for two-role LLM phase surfaces: their
    component-only time cannot establish cross-batch dominance before DES adds
    the engine-cycle curve.  Forward/single-role components keep the legacy
    cross-batch Pareto behavior.
    """

    winners = frame.loc[
        frame.groupby(list(_SWEEP_RESOURCE_BATCH_IDENTITY)).time_ms.idxmin()
    ]
    if preserve_batch_axis:
        return winners
    return _get_pareto_df(winners)


def validate_two_role_llm_component_record(record: object) -> None:
    """Enforce the one production contract for a two-role LLM component.

    Generator and query-expansion serving may consume only the current,
    independently measured ceff/meff record.  In particular, model-keyed
    serving intercepts and linear-law substitutes must not enter through a
    point, sweep, exact-device, or dynamic/DES call path and then overlap the
    hardware x stage x batch overhead curve.

    Re-running :class:`CalibrationRecord` validation here is intentional.  A
    loaded record is mutable, and every pricing boundary must fail closed if a
    caller mutates the cached object after construction.
    """
    from rag_stack.cost_model.llm_sim.calibration import (
        CalibrationRecord,
        ZERO_INTERCEPT_ROOFLINE_CONTRACT,
    )

    contract = getattr(record, "calibration_contract", None)
    try:
        overhead_s = float(getattr(record, "per_call_overhead_s", 0.0) or 0.0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GenZCalibrationError(
            "two-role LLM component per_call_overhead_s must be exactly zero"
        ) from exc
    if (
        contract != ZERO_INTERCEPT_ROOFLINE_CONTRACT
        or overhead_s != 0.0
    ):
        raise GenZCalibrationError(
            "two-role LLM serving requires a zero-intercept component record; "
            "per_call_overhead_s must be exactly zero because serving overhead "
            "belongs only to the hardware x stage x batch curve"
        )
    if getattr(record, "linear_law", None) is not None:
        raise GenZCalibrationError(
            "two-role LLM serving forbids a linear_law component substitute; "
            "the component must contain only independently fitted ceff/meff"
        )
    if (
        getattr(record, "compute_efficiency", None) is None
        or getattr(record, "memory_efficiency", None) is None
    ):
        raise GenZCalibrationError(
            "two-role LLM component record requires both ceff and meff"
        )

    try:
        if not isinstance(record, CalibrationRecord):
            raise TypeError("record is not a CalibrationRecord")
        record.__post_init__()

        def point_shape(point: dict) -> tuple[int, ...]:
            seq_len = point["seq_len"]
            axes = seq_len if isinstance(seq_len, (list, tuple)) else (seq_len,)
            return tuple(int(value) for value in axes)

        fit_shapes = {
            point_shape(point)
            for point in record.grid
            if int(point["batch_size"]) == 128
        }
        held_out_shapes = {
            point_shape(point)
            for point in record.validation_grid
            if int(point["batch_size"]) == 256
        }
        if not fit_shapes.intersection(held_out_shapes):
            raise ValueError(
                "generator roofline profile requires an exact same-token-shape "
                "B128 fit/B256 held-out pair"
            )
    except (TypeError, ValueError) as exc:
        raise GenZCalibrationError(
            "two-role LLM component record has invalid runtime pricing fields: "
            f"{exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Fixed generator topology
# ---------------------------------------------------------------------------

def resolve_fixed_llm_parallelism(
    pipeline_config: dict,
    stage: str,
) -> tuple[int, int] | None:
    """Return one fixed LLM PerformanceStage's ``(TP, PP)`` from layout.

    This is intentionally config-driven: fixed-placement callers should pass a
    resolved system/pipeline config, not an out-of-band benchmark hint.  It is
    taxonomy-driven and therefore covers generator, query expansion, and any
    future prefill/decode-shaped engine uniformly.
    """
    from rag_stack.taxonomy import TAXONOMY

    try:
        ps = TAXONOMY.performance_stage(stage)
        engine = TAXONOMY.user_stage_of(stage).value
    except (KeyError, ValueError):
        return None
    if ps.kind not in ("prefill", "decode"):
        return None
    system = pipeline_config.get("system") or {}

    layout = system.get("layout")
    if layout is None:
        return None
    if not isinstance(layout, dict):
        raise ValueError("system.layout must be a mapping")
    engines = layout.get("engines")
    if not isinstance(engines, dict):
        raise KeyError("system.layout.engines is required")
    engine_info = engines.get(engine)
    if not isinstance(engine_info, dict):
        raise KeyError(f"system.layout.engines.{engine} is required")
    pd_serving = engine_info.get("pd_serving")
    if pd_serving not in {"collocated_pd", "disagg_pd"}:
        raise ValueError(f"system.layout.engines.{engine}.pd_serving is required")
    if pd_serving == "disagg_pd":
        record = engine_info.get(stage)
        if not isinstance(record, dict):
            raise KeyError(f"system.layout.engines.{engine}.{stage} is required")
    else:
        record = engine_info
    if "tp" not in record and "pp" not in record:
        devices = record.get("devices")
        if not isinstance(devices, list) or not devices:
            return None
        # Older resolved schemas only persisted explicit TP/PP for generator
        # and query expansion.  Other true prefill/decode engines used TP=N,
        # PP=1; recover that deterministic runtime plan from their exact ranks.
        return len(devices), 1
    try:
        devices = record["devices"]
        tp = max(1, int(record["tp"]))
        pp = max(1, int(record["pp"]))
    except KeyError as exc:
        raise KeyError(
            f"system.layout.engines.{engine}.{stage} requires devices/tp/pp"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"system.layout.engines.{engine}.{stage} has invalid tp/pp"
        ) from exc
    if not isinstance(devices, list) or not devices:
        raise ValueError(
            f"system.layout.engines.{engine}.{stage}.devices must be non-empty"
        )
    chips = len(devices)
    if tp * pp != chips:
        raise ValueError(
            "Fixed generator %s TP/PP=%s/%s is incompatible with chips=%s; "
            "current schema requires an exact match."
            % (stage, tp, pp, chips)
        )
    return tp, pp


def resolve_fixed_generator_parallelism(
    pipeline_config: dict,
    stage: str | None = None,
) -> tuple[int, int] | None:
    """Backward-compatible generator-only wrapper."""
    if stage is None:
        system = pipeline_config.get("system") or {}
        layout = system.get("layout")
        if not isinstance(layout, dict):
            return None
        gen = (layout.get("engines") or {}).get("generator")
        if isinstance(gen, dict) and gen.get("pd_serving") == "disagg_pd":
            return None
        stage = "generator_prefill"
    if stage not in ("generator_prefill", "generator_decode"):
        return None
    return resolve_fixed_llm_parallelism(pipeline_config, stage)


# ---------------------------------------------------------------------------
# Model name mapping
# ---------------------------------------------------------------------------

from rag_stack.cost_model.llm_sim.genz_model_map import resolve_genz_model


# ---------------------------------------------------------------------------
# KV-capacity-aware effective decode batch (r13)
# ---------------------------------------------------------------------------

_WEIGHTS_MB_CACHE: dict = {}


def kv_capped_decode_batch(
    model, batch_size: int, ctx_tokens: int,
    hw_dict: dict, bits: str, tp: int, pp: int,
) -> int:
    """The number of sequences the KV pool can actually hold concurrently.

    When batch x context KV exceeds VRAM, real vLLM does NOT offload
    weights to host — it admits fewer sequences into the running batch
    (continuous batching degrades gracefully; the engine gauge probe shows
    exactly this). GenZ's model_offload path instead prices a catastrophic
    host-offload slowdown (5-8x), which faked the 14B batch-256 cliff
    (predicted 94 s vs measured 31 s). Decode is therefore priced at
    B_eff = min(B, KV-fitting batch) and scaled back to the full batch by
    the caller (throughput at the memory ceiling is B_eff per step).

    Analytic accounting per chip (weights and KV both shard across tp;
    layers across pp): kv_per_seq = 2 x layers/pp x kv_heads/tp x head_dim
    x ctx x dtype_bytes. Weights come from GenZ's own model
    characteristics (cached per (model, bits, tp, pp))."""
    from GenZ.Models import get_configs

    cfg = get_configs(model) if isinstance(model, str) else model
    dtype_bytes = 2 if bits in ("bf16", "fp16") else 4 if bits == "f32" else 2
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_per_seq = (
        2 * (cfg.num_decoder_layers / max(1, pp))
        * (cfg.num_key_value_heads / max(1, tp))
        * head_dim * max(1, int(ctx_tokens)) * dtype_bytes
    )  # bytes per sequence per chip
    weights_key = (getattr(cfg, "model", str(model)), bits, tp, pp)
    if weights_key not in _WEIGHTS_MB_CACHE:
        try:
            from GenZ.LLM_inference.llm_decode import decode_moddeling as _dm
            _mdf, _st = _dm(
                model=model, batch_size=1, Bb=1, input_tokens=64,
                output_tokens=8, system_name=dict(hw_dict), bits=bits,
                tensor_parallel=tp, pipeline_parallel=pp,
                model_offload=True, model_profilling=True,
            )
            _WEIGHTS_MB_CACHE[weights_key] = float(
                _st["Total Weights (MB)"].values[0]
            )
        except Exception:  # noqa: BLE001 — no weights info → no clamp
            _WEIGHTS_MB_CACHE[weights_key] = None
    w = _WEIGHTS_MB_CACHE[weights_key]
    if w is None:
        return max(1, int(batch_size))
    mem_mb = float(hw_dict.get("Memory_size", 80)) * 1024.0
    free_mb = mem_mb - w - 2048.0  # 2 GB activation/workspace headroom
    if free_mb <= 0:
        return 1
    b_eff = int(free_mb * 1024 * 1024 // max(1.0, kv_per_seq))
    return max(1, min(int(batch_size), b_eff))


# ---------------------------------------------------------------------------
# Base class for all GenZ-based cost models
# ---------------------------------------------------------------------------

class GenZLLMSimBase(CostModelModule):
    """Base class for GenZ-based cost model modules.

    Subclasses define:
    - _get_model(): which model to simulate
    - _get_params(): input_tokens, output_tokens, bits
    - _find_module(): how to find config in the pipeline

    Which stages a module produces, and a stage's prefill/decode kind, come from
    the central taxonomy (rag_stack.taxonomy) via
    ``_map_stage`` / ``_stage_kinds`` — NOT per-model overrides.
    """

    @classmethod
    def family(cls) -> str:
        return "llm_sim"

    @classmethod
    def capabilities(cls) -> list[str]:
        return ["calibration_profiles", "calibration_runs"]

    @classmethod
    def profiles_subdir(cls) -> str | None:
        return "llm_sim"

    @abstractmethod
    def _get_model(self, pipeline_config: dict) -> str | ModelConfig:
        """Return GenZ model name or ModelConfig."""
        ...

    def parallelism_model(self, pipeline_config: dict) -> str | ModelConfig:
        """Public model identity for GenZ parallelism-feasibility checks.

        Candidate assembly must not infer attention/KV heads from model names.
        It asks GenZ to validate the concrete TP/PP or P/D pair, using the same
        resolved model object that this pricer sends to GenZ for latency.
        """
        return self._get_model(pipeline_config)

    @abstractmethod
    def _get_params(self, pipeline_config: dict) -> dict:
        """Return {"input_tokens": int, "output_tokens": int, "bits": str}."""
        ...

    @abstractmethod
    def _find_module(self, pipeline_config: dict) -> dict:
        """Find this module's config in the pipeline config."""
        ...

    @abstractmethod
    def _get_model_name_for_df(self, pipeline_config: dict) -> str:
        """Return model name string for the sweep DataFrame."""
        ...

    def _calibration_component(self, pipeline_config: dict) -> str:
        """Return the persisted component identity for calibration records.

        Most GenZ-backed components own a normal pipeline module and persist
        that module's ``component`` value.  Models whose configuration lives
        elsewhere (for example retrieval encode on a vectordb entry) override
        this method instead of relying on CLI-specific identity patches.
        """
        component = self._find_module(pipeline_config).get("component")
        if not isinstance(component, str) or not component.strip():
            raise GenZCalibrationError(
                f"{type(self).__name__} did not resolve a calibration "
                "component identity"
            )
        return component.strip()

    def _sweep_max_devices(self, search_space: dict) -> int:
        """Max GPUs (chips) the GenZ TP/PP sweep enumerates for THIS stage.

        Default = ``max_num_gpus_per_stage`` (model-parallel: GenZ simulates
        each num_devices via (TP,PP)). Subclasses for stages that are NOT
        tensor-parallelizable in the real runtime (small BERT encoders →
        reranker) override this to 1, emitting only single-chip rows so RAGO's
        ``scale_up_performance_to_more_chips`` extrapolates multi-chip as
        DATA-PARALLEL (qps×N) — matching how those stages actually scale.
        """
        cap = search_space.get(
            "max_num_gpus_per_stage", search_space.get("max_num_gpus", 1)
        )
        if isinstance(cap, list):
            cap = max(cap)
        return int(cap)

    def _get_system_config(self, hardware: SystemConfig) -> dict | str:
        """Resolve hardware → the GenZ ``system_name`` dict via the SINGLE
        constructor :func:`calibration.genz_system_dict` (shared with the
        calibrated-record path), so the sweep and the latency walk feed GenZ
        identical interconnect/overhead parameters for the same deployment."""
        from rag_stack.cost_model.llm_sim.calibration import genz_system_dict
        return genz_system_dict(hardware)

    def _parallel_devices_for_stage(
        self,
        pipeline_config: dict,
        stage: str,
        *,
        num_devices: int,
        hardware: SystemConfig | None = None,
    ) -> list[str]:
        """Concrete device/rank order for one fixed or swept LLM stage."""
        from rag_stack.taxonomy import TAXONOMY

        system = (pipeline_config or {}).get("system") or {}
        layout = system.get("layout") if isinstance(system, dict) else None
        if isinstance(layout, dict):
            try:
                engine = TAXONOMY.user_stage_of(stage).value
            except (KeyError, ValueError):
                engine = "generator"
            engines = layout.get("engines") or {}
            info = engines.get(engine)
            if not isinstance(info, dict):
                raise KeyError(f"system.layout.engines.{engine} is required")
            record = info.get(stage) if info.get("pd_serving") == "disagg_pd" else info
            if not isinstance(record, dict):
                raise KeyError(f"system.layout.engines.{engine}.{stage} is required")
            devices = record.get("devices")
            if not isinstance(devices, list) or not devices:
                raise KeyError(f"system.layout.engines.{engine}.{stage}.devices is required")
            devices = [str(device) for device in devices]
            if len(devices) != int(num_devices):
                raise ValueError(
                    f"{stage} TP/PP needs {num_devices} devices but resolved layout "
                    f"declares {devices}"
                )
            return devices

        # Swept CM: use the target system's declared physical inventory, never
        # torch.cuda/the current host.  Select the topology-best rank order for
        # this TP/PP width; the final per-candidate repricer supplies the RAGO
        # mapping's exact devices/order instead.
        from rag_stack.cost_model.interconnect_topology import available_gpu_devices
        topology_source = system if system.get("interconnect") else hardware
        devices = available_gpu_devices(topology_source, required_count=num_devices)
        return list(devices[:int(num_devices)])

    def _system_config_for_parallelism(
        self,
        system_name: dict | str,
        hardware: SystemConfig,
        pipeline_config: dict,
        stage: str,
        tp: int,
        pp: int,
        devices_override: list[str] | None = None,
    ) -> dict | str:
        """Collapse pair-aware topology into GenZ's homogeneous ICN fields."""
        if not isinstance(system_name, dict):
            return system_name
        system = (pipeline_config or {}).get("system") or {}
        from rag_stack.cost_model.interconnect_topology import (
            effective_llm_parallel_link_gbps_us,
            has_gpu_topology,
        )
        topology_source = system if system.get("interconnect") else hardware
        if not has_gpu_topology(topology_source):
            return system_name
        devices = list(devices_override or self._parallel_devices_for_stage(
            pipeline_config,
            stage,
            num_devices=int(tp) * int(pp),
            hardware=hardware,
        ))
        if len(devices) != int(tp) * int(pp):
            raise ValueError(
                f"{stage}: devices/order {devices} does not match TP*PP={tp}*{pp}"
            )
        bw_gbps, physical_lat_us = effective_llm_parallel_link_gbps_us(
            topology_source,
            devices,
            tp=tp,
            pp=pp,
            default_gbps=float(system_name.get("ICN", 150.0)),
            default_us=float(system_name.get("ICN_LL", 1.0)),
        )
        out = dict(system_name)
        out["ICN"] = bw_gbps
        out["candidate_devices"] = list(devices)
        from rag_stack.cost_model.llm_sim.tp_efficiency import effective_icn_ll_us
        icn_ll = effective_icn_ll_us(hardware, physical_latency_us=physical_lat_us)
        if icn_ll is not None:
            out["ICN_LL"] = icn_ll
        return out

    # ------------------------------------------------------------------
    # Benchmark + calibration framework
    # ------------------------------------------------------------------

    def _get_benchmark_runner(self, pipeline_config: dict):
        """Return a BaseBenchmarkRunner for the module selected in
        ``pipeline_config``. Default implementation raises — subclasses
        that support calibration override this and dispatch to the right
        runner via a per-cost-model factory.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no benchmark runner. Override "
            f"_get_benchmark_runner() to enable benchmark/calibrate."
        )

    def benchmark(
        self,
        hardware: SystemConfig,
        pipeline_config: dict,
        batch_sizes: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16),
        seq_lens: list[int] | tuple[int, ...] = (128, 256, 512, 1024),
        warmup: int = 5,
        measurements: int = 20,
        compute_eff: float = 1.0,
        memory_eff: float = 1.0,
    ):
        """Measure real-hw latency, predict at the given
        ``(compute_eff, memory_eff)``, and return a diagnostic
        ``BenchmarkReport``. Does NOT persist anything.

        Defaults to (1.0, 1.0) — the pre-calibration roofline check for
        validating ModelConfig shape. Pass saved coefficients to verify
        an existing calibration profile still applies.
        """
        from rag_stack.cost_model.llm_sim.calibration import run_benchmark_report

        runner = self._get_benchmark_runner(pipeline_config)
        measured = runner.run(
            list(batch_sizes), list(seq_lens), warmup, measurements
        )
        # YAML `module.bits` wins if declared; else auto-detect from
        # loaded torch weights.
        bits = self._find_module(pipeline_config).get("bits") or runner.detect_bits()
        return run_benchmark_report(
            cost_model=self,
            hardware=hardware,
            pipeline_config=pipeline_config,
            measured=measured,
            compute_eff=compute_eff,
            memory_eff=memory_eff,
            bits=bits,
        )

    def calibrate(
        self,
        hardware: SystemConfig,
        pipeline_config: dict,
        batch_sizes: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16),
        seq_lens: list[int] | tuple[int, ...] = (128, 256, 512, 1024),
        warmup: int = 5,
        measurements: int = 20,
        output_dir=None,
    ):
        """Run the benchmark, fit (compute_efficiency, memory_efficiency)
        via scipy L-BFGS-B, persist one JSON file per (model, hardware)
        pair under ``calibration_profiles/llm_sim/`` (override via
        ``output_dir``), and return the ``CalibrationRecord``.
        """
        if any(
            _is_prefill_decode_engine_stage(
                self._performance_stage_name(pipeline_config, kind)
            )
            for kind in self._stage_kinds(pipeline_config)
        ):
            raise GenZCalibrationError(
                "two-role LLM serving must use the dedicated generator "
                "component calibrator so the fit is bound to audited raw "
                "telemetry with explicit warmup exclusion"
            )
        from rag_stack.cost_model.llm_sim.calibration import (
            fit_and_save,
            split_fit_validation_evidence,
        )

        runner = self._get_benchmark_runner(pipeline_config)
        measured = runner.run(
            list(batch_sizes), list(seq_lens), warmup, measurements
        )
        training, validation = split_fit_validation_evidence(measured)
        module = self._find_module(pipeline_config)
        # YAML `module.bits` wins if declared; else auto-detect from
        # loaded torch weights.
        bits = module.get("bits") or runner.detect_bits()
        component = self._calibration_component(pipeline_config)
        return fit_and_save(
            cost_model=self,
            hardware=hardware,
            pipeline_config=pipeline_config,
            measured=training,
            component=component,
            model_key=self._get_model_name_for_df(pipeline_config),
            output_dir=output_dir,
            bits=bits,
            validation_points=validation,
        )

    def _resolve_calibration(
        self, hardware: SystemConfig, pipeline_config: dict
    ):
        """Return ``(system_name, ceff, meff, bits, call_overhead_s)`` for the GenZ call.

        ``call_overhead_s`` is retained for single-role legacy component
        compatibility.  It is required to be exactly zero for every two-role
        LLM before this method returns.

        Lookup order:
          1. Calibration record for (component, model_key, hw_key, bits).
          2. Same-model/hardware/bits standalone ``agnostic`` record.

        Strict mode fails closed on a miss.  Explicit ``uncalibrated_spec``
        mode may replace only a missing exact record with the physical
        roofline.  A malformed record, identity mismatch, or unresolved
        physical GPU specification still fails closed in both modes.
        """
        from rag_stack.cost_model.llm_sim.calibration import (
            _default_calibration_dir,
            _hardware_to_dict,
            _load_system,
            detect_hardware_key,
            load_record,
        )

        try:
            model_key = self._get_model_name_for_df(pipeline_config)
            module = self._find_module(pipeline_config)
            component = str(module.get("component", "") or "")
            bits_hint = module.get("bits")   # YAML declaration, None if absent
        except Exception as exc:
            raise GenZCalibrationError(
                "Production GenZ pricing could not resolve the component/model "
                "identity required for an exact calibration record"
            ) from exc

        hw_key = detect_hardware_key(hardware)
        requested_bits = (
            repr(bits_hint) if bits_hint is not None else "<profile-resolved>"
        )
        identity = (
            f"component={component or 'agnostic'!r}, model={model_key!r}, "
            f"hardware={hw_key!r}, bits={requested_bits}"
        )
        profile_path = _default_calibration_dir() / f"{hw_key}.json"

        # ``load_record`` historically treats a false-y JSON value as a miss.
        # Spec mode must distinguish an absent key from a key that is present
        # but malformed, including ``null``, ``{}``, and ``[]``.  Follow the
        # same specialized-then-agnostic lookup order and preflight only the
        # entry that lookup would select; unrelated components in the shared
        # per-hardware file remain independent.
        if profile_path.exists() or profile_path.is_symlink():
            try:
                raw_system = _load_system(profile_path)
            except Exception as exc:
                raise GenZCalibrationError(
                    f"Invalid GenZ calibration profile for {identity}: {exc}"
                ) from exc
            if isinstance(raw_system, dict):
                selected_entries: list[tuple[str, object]] = []
                for lookup_component in dict.fromkeys(
                    (component, "agnostic")
                ):
                    prefix = f"{lookup_component}__{model_key}__"
                    if bits_hint is None:
                        matches = sorted(
                            key for key in raw_system
                            if key.startswith(prefix)
                            and "__" not in key[len(prefix):]
                        )
                    else:
                        key = f"{prefix}{bits_hint}"
                        matches = [key] if key in raw_system else []
                    if matches:
                        selected_entries = [
                            (key, raw_system[key]) for key in matches
                        ]
                        break
                if len(selected_entries) == 1:
                    selected_key, selected_value = selected_entries[0]
                    if not isinstance(selected_value, dict) or not selected_value:
                        raise GenZCalibrationError(
                            "Invalid GenZ calibration record for "
                            f"{identity}: present entry {selected_key!r} must "
                            "be a non-empty object"
                        )
        try:
            rec = load_record(
                component, model_key, hw_key,
                bits=bits_hint,
            )
        except Exception as exc:
            raise GenZCalibrationError(
                f"Invalid GenZ calibration record for {identity}: {exc}"
            ) from exc
        if rec is None:
            if profile_path.is_symlink() and not profile_path.exists():
                raise GenZCalibrationError(
                    "Invalid GenZ calibration profile entry for "
                    f"{identity}: {profile_path} is a broken symlink"
                )
            if is_uncalibrated_spec_mode(pipeline_config, hardware):
                fallback_bits = bits_hint
                if fallback_bits is None:
                    fallback_bits = self._get_params(pipeline_config).get("bits")
                if not isinstance(fallback_bits, str) or not fallback_bits.strip():
                    raise GenZCalibrationError(
                        "uncalibrated_spec GenZ pricing requires a concrete "
                        f"precision for {identity}"
                    )
                warning_key = (
                    str(component or "agnostic"),
                    str(model_key),
                    str(hw_key),
                    fallback_bits,
                )
                if warning_key not in _SPEC_FALLBACK_WARNED:
                    _SPEC_FALLBACK_WARNED.add(warning_key)
                    logger.warning(
                        "No exact GenZ component profile for %s; "
                        "uncalibrated_spec is using the physical roofline "
                        "with ceff=meff=1 and no component overhead.",
                        identity,
                    )
                try:
                    return (
                        _hardware_to_dict(hardware),
                        1.0,
                        1.0,
                        fallback_bits,
                        0.0,
                    )
                except Exception as exc:
                    raise GenZCalibrationError(
                        "Cannot build the physical GenZ system for "
                        f"uncalibrated_spec pricing ({identity}): {exc}"
                    ) from exc
            raise GenZCalibrationError(
                "Production GenZ pricing requires an exact component record "
                "or same-model agnostic record; none was found for "
                f"{identity}. Add/promote the standalone profile under "
                f"{profile_path}. YAML GPU "
                "efficiency defaults and uncalibrated roofline pricing are "
                "diagnostic-only and cannot be used here."
            )

        _validate_calibration_identity(
            rec,
            component=component,
            model_key=model_key,
            hardware_key=hw_key,
            bits_hint=bits_hint,
        )
        if self._requires_two_role_component_contract(pipeline_config):
            # This is also the dynamic/DES boundary: engines.build_call_latency
            # calls _resolve_calibration directly.  Validate before reading or
            # returning the legacy-compatible per_call_overhead_s field.
            validate_two_role_llm_component_record(rec)
        if rec.compute_efficiency is None or rec.memory_efficiency is None:
            raise GenZCalibrationError(
                f"GenZ roofline pricing requires compute/memory efficiencies, "
                f"but the exact record for {identity} is linear-law-only"
            )

        try:
            return (
                _hardware_to_dict(hardware),
                rec.compute_efficiency,
                rec.memory_efficiency,
                rec.bits,
                float(getattr(rec, "per_call_overhead_s", 0.0) or 0.0),
            )
        except Exception as exc:
            raise GenZCalibrationError(
                f"Cannot build the physical GenZ system for the exact "
                f"calibration record ({identity}): {exc}"
            ) from exc

    def _calibration_record(self, hardware, pipeline_config: dict):
        """Load this model's calibration record (None when absent) — used
        by the linear-law prediction branch; the (ceff, meff, overhead)
        path keeps going through _resolve_calibration."""
        from rag_stack.cost_model.llm_sim.calibration import (
            detect_hardware_key,
            load_record,
        )
        try:
            model_key = self._get_model_name_for_df(pipeline_config)
            module = self._find_module(pipeline_config)
            component = module.get("component", "")
            bits_hint = module.get("bits")
        except Exception:  # noqa: BLE001
            return None
        hardware_key = detect_hardware_key(hardware)
        record = load_record(
            component, model_key, hardware_key,
            bits=bits_hint,
        )
        if record is not None:
            _validate_calibration_identity(
                record,
                component=str(component or ""),
                model_key=model_key,
                hardware_key=hardware_key,
                bits_hint=bits_hint,
            )
        return record

    def _requires_two_role_component_contract(
        self,
        pipeline_config: dict,
    ) -> bool:
        """Whether this concrete component owns a prefill+decode engine stage."""

        return any(
            _is_prefill_decode_engine_stage(
                self._performance_stage_name(pipeline_config, kind)
            )
            for kind in self._stage_kinds(pipeline_config)
        )

    def _pricing_record_and_params(
        self,
        hardware: SystemConfig,
        pipeline_config: dict,
        params: dict,
    ) -> tuple[object | None, dict]:
        """Resolve the exact record and apply its tokenizer shape contract.

        This helper is shared by single-point, sweep, and exact-device pricing
        so a compact ``linear_law`` record never falls through to the GenZ
        roofline path merely because the caller used a different public API.
        """
        record = self._calibration_record(hardware, pipeline_config)
        if (
            record is not None
            and self._requires_two_role_component_contract(pipeline_config)
        ):
            # Point/sweep/exact paths pass here before a possible linear-law
            # early return.  _resolve_calibration invokes the same helper for
            # the dynamic/DES path, which historically bypassed this method.
            validate_two_role_llm_component_record(record)
        adjusted = dict(params)
        max_seq_len = (
            getattr(record, "max_seq_len", None)
            if record is not None else None
        )
        if max_seq_len:
            adjusted["input_tokens"] = min(
                int(adjusted["input_tokens"]), int(max_seq_len)
            )
        return record, adjusted

    @staticmethod
    def _linear_law_latency_s(
        record: object | None,
        params: dict,
        batch_size: int,
    ) -> float | None:
        """Return exact linear-law batch service, or ``None`` for roofline records."""
        law = getattr(record, "linear_law", None) if record is not None else None
        if not isinstance(law, dict) or law.get("per_item_ms") is None:
            return None
        batch = max(1, int(batch_size))
        input_tokens = float(params["input_tokens"])
        latency_s = (
            float(law.get("fixed_ms", 0.0))
            + batch * (
                float(law["per_item_ms"])
                + float(law["per_token_ms"]) * input_tokens
            )
        ) / 1000.0
        if "capacity_tokens" in law:
            aggregate_tokens = float(batch) * input_tokens
            latency_s += (
                float(law["overload_step_ms"])
                * float(aggregate_tokens > float(law["capacity_tokens"]))
                + float(law["overload_per_token_ms"])
                * max(0.0, aggregate_tokens - float(law["capacity_tokens"]))
            ) / 1000.0
        return max(float(latency_s), 1e-9)

    def _inject_calibration_shape(
        self, pipeline_config: dict, batch_size: int, seq_len: int
    ) -> dict:
        """Clone ``pipeline_config`` with one benchmark point's
        ``(batch_size, seq_len)`` injected where THIS cost model reads its
        input length and batch. The default (``avg_chunk_tokens`` +
        node-module ``batch_size``) matches rerankers / compressors /
        generators; models whose ``_get_params`` reads a different token
        stat or whose ``_find_module`` returns a non-node dict (e.g. the
        encode stage's vectordb entry) must override — otherwise
        benchmark/calibrate silently predicts a shape-blind constant.
        """
        from rag_stack.cost_model.llm_sim.calibration import _inject_shape
        return _inject_shape(pipeline_config, batch_size, seq_len)

    def _map_stage(self, stage: str) -> str:
        """Map a PerformanceStage name to its GenZ sim kind (prefill/decode)
        via the central taxonomy — NO subclass overrides. GenZ kinds never
        leave this adapter."""
        from rag_stack.taxonomy import TAXONOMY
        ps = TAXONOMY.performance_stages.get(stage)
        return ps.kind if ps is not None else stage

    def _stage_kinds(self, pipeline_config: dict) -> list[str]:
        """GenZ kinds this model contributes (e.g. ``["prefill", "decode"]`` for
        the generator, ``["prefill"]`` for a reranker), derived from the taxonomy
        for the selected component — replaces the old per-model
        ``_get_stages()``."""
        from rag_stack.taxonomy import TAXONOMY
        module = self._find_module(pipeline_config)
        name = str(module.get("component", "") or "")
        comp = TAXONOMY.components.get(name)
        if comp is None or comp.shape is None:
            return ["prefill"]
        kinds: list[str] = []
        for ps in TAXONOMY.performance_stages_of_component(name):
            if ps.kind not in kinds:
                kinds.append(ps.kind)
        return kinds or ["prefill"]

    def _performance_stage_name(self, pipeline_config: dict, kind: str) -> str:
        """Resolve a taxonomy PerformanceStage when this model owns one.

        Most GenZ adapters read a node module carrying a registered
        ``component``. Retrieval encode is intentionally different: its
        configuration owner is the selected vectordb record, which has an
        embedding model but no component name. Such non-LLM models use the
        GenZ ``kind`` directly and must not be routed through LLM topology
        resolution.
        """
        from rag_stack.taxonomy import TAXONOMY

        module = self._find_module(pipeline_config)
        component = str(module.get("component", "") or "")
        if component not in TAXONOMY.components:
            return kind
        for stage in TAXONOMY.performance_stages_of_component(component):
            if stage.kind == kind:
                return stage.name
        return kind

    def estimate(
        self, pipeline_config: dict, hardware: SystemConfig
    ) -> CostModelResult:
        """Coarse analytical component estimate for one complete model call.

        This surface has no scheduler-cycle trace, so it must not consume a
        collocated engine-cycle overhead curve.  The closed-loop simulator is
        the sole boundary that can add that curve once per real cycle.
        """
        module = self._find_module(pipeline_config)
        if not module:
            stages = [
                n.get("stage")
                for nl in pipeline_config.get("node_lines", [])
                for n in nl.get("nodes", [])
            ]
            raise RuntimeError(
                f"{type(self).__name__}.estimate(): "
                f"_find_module() returned empty. "
                f"stages in pipeline_config: {stages}"
            )

        model = self._get_model(pipeline_config)
        params = self._get_params(pipeline_config)
        batch_size = int(module.get("batch_size", 1))

        # Per-item linear-law branch: records carrying linear_law (fit via
        # `cli refit --form linear`) predict t = fixed + B*(c1 + c2*L)
        # instead of the GenZ roofline — the roofline shape under-predicts
        # tiny cross-encoders by -50..-64% at production batches even when
        # fit on the full grid (r12, ms_marco_minilm_l2). L is the same
        # per-item input_tokens the fit consumed.
        _rec, params = self._pricing_record_and_params(
            hardware, pipeline_config, params,
        )
        # Production truncation cap (r14): the module's tokenizer feeds the
        # model at most record.max_seq_len tokens per item, regardless of how
        # much retrieved text arrives. Applies to BOTH prediction forms.
        t_s = self._linear_law_latency_s(_rec, params, batch_size)
        if t_s is not None:
            return CostModelResult(
                latency_s=float(t_s),
                throughput_qps=batch_size / float(t_s),
                operators=[OperatorBreakdown(
                    name="per_item_linear", time_s=float(t_s),
                    compute_time_s=t_s, bottleneck="prefill",
                )],
            )

        # Production GenZ calls require the exact component/model/hardware/bits
        # record (with the standalone same-model ``agnostic`` record allowed).
        system_name, ceff, meff, bits, call_overhead_s = self._resolve_calibration(
            hardware, pipeline_config
        )
        stages = self._stage_kinds(pipeline_config)

        def _fixed_parallelism(kind: str) -> tuple[int, int]:
            name = self._performance_stage_name(pipeline_config, kind)
            if not _is_prefill_decode_engine_stage(name):
                return 1, 1
            return resolve_fixed_llm_parallelism(pipeline_config, stage=name) or (1, 1)

        def _parallel_system(kind: str, tp: int, pp: int):
            stage_name = self._performance_stage_name(pipeline_config, kind)
            if not _is_prefill_decode_engine_stage(stage_name):
                return system_name
            return self._system_config_for_parallelism(
                system_name, hardware, pipeline_config,
                stage_name, tp, pp,
            )

        operators = []
        total_s = 0.0

        if "prefill" in stages:
            tp, pp = _fixed_parallelism("prefill")
            result = prefill_moddeling(
                model=model, batch_size=batch_size,
                input_tokens=params["input_tokens"],
                system_name=_parallel_system("prefill", tp, pp), bits=bits,
                ceff=ceff, meff=meff,
                tensor_parallel=tp, pipeline_parallel=pp,
                model_offload=True,
            )
            component_t = (
                scheduler_batch_traversal_latency_ms(
                    result, pipeline_parallel=int(pp),
                )
                * pipeline_fill_drain_factor(int(batch_size), int(pp))
                / 1000.0
            )
            total_s += component_t
            operators.append(OperatorBreakdown(
                name="llm_prefill", time_s=component_t,
                compute_time_s=component_t, bottleneck="prefill",
            ))

        if "decode" in stages:
            tp, pp = _fixed_parallelism("decode")
            _sysname = _parallel_system("decode", tp, pp)
            # KV-capacity clamp (r13): price at the batch the KV pool can
            # actually hold and scale linearly back to the requested batch
            # — vLLM's graceful concurrency reduction, NOT GenZ's
            # catastrophic host-offload.
            _ctx = int(params["input_tokens"]) + int(params["output_tokens"])
            _b_eff = (
                kv_capped_decode_batch(
                    model, batch_size, _ctx, _sysname, bits, tp, pp,
                )
                if isinstance(_sysname, dict) else batch_size
            )
            result = decode_moddeling(
                Bb=1,  # vLLM serves beam=1; GenZ default Bb=4 quadrupled KV accounting (fake offload at 14B b256) and decode bytes — r13
                model=model, batch_size=_b_eff,
                input_tokens=params["input_tokens"],
                output_tokens=params["output_tokens"],
                system_name=_sysname, bits=bits,
                ceff=ceff, meff=meff,
                tensor_parallel=tp, pipeline_parallel=pp,
                model_offload=True,
            )
            component_t = (
                scheduler_batch_traversal_latency_ms(
                    result, pipeline_parallel=int(pp),
                )
                * pipeline_fill_drain_factor(int(_b_eff), int(pp))
                / 1000.0
                * (float(batch_size) / float(_b_eff))
            )
            total_s += component_t
            operators.append(OperatorBreakdown(
                name="llm_decode", time_s=component_t,
                compute_time_s=component_t, bottleneck="decode",
            ))

        # Calibrated fixed per-call overhead — once per model call (prefill +
        # decode together). 0.0 for pre-affine records → bit-identical.
        if call_overhead_s > 0.0:
            total_s += call_overhead_s
            operators.append(OperatorBreakdown(
                name="per_call_overhead", time_s=call_overhead_s,
                compute_time_s=call_overhead_s, bottleneck="overhead",
            ))

        return CostModelResult(
            latency_s=total_s,
            throughput_qps=batch_size / total_s if total_s > 0 else 0.0,
            operators=operators,
        )

    def generate_sweep_df(
        self,
        pipeline_config: dict,
        hardware: SystemConfig,
        stage: str,
        max_batch_size: int,
    ) -> pd.DataFrame:
        """RAGO-style sweep: num_devices × (TP,PP) × powers-of-2 batch sizes."""
        module = self._find_module(pipeline_config)
        if not module:
            stages = [
                n.get("stage")
                for nl in pipeline_config.get("node_lines", [])
                for n in nl.get("nodes", [])
            ]
            raise RuntimeError(
                f"{type(self).__name__}.generate_sweep_df(stage={stage!r}): "
                f"_find_module() returned empty. "
                f"stages in pipeline_config: {stages}"
            )

        model = self._get_model(pipeline_config)
        params = self._get_params(pipeline_config)
        model_name = self._get_model_name_for_df(pipeline_config)
        record, params = self._pricing_record_and_params(
            hardware, pipeline_config, params,
        )

        search_space = hardware.search_space or {}
        # Per-stage GPU cap, NOT total fleet. ``max_num_gpus`` is the
        # whole-experiment budget shared across stages (retrieval +
        # encode + reranker + LLM …); the single LLM stage swept here
        # may use at most ``max_num_gpus_per_stage``. Reading
        # ``max_num_gpus`` instead caused PP to climb to 8 even though
        # the YAML cap was 4, which then tripped GenZ's
        # ZeroDivisionError on models whose layer count doesn't divide
        # cleanly by 8 (e.g. Qwen2.5-7B: 28 / 8 = 3.5).
        # ``_sweep_max_devices`` is overridable per-stage: reranker caps it to
        # 1 (single-chip GenZ rows → RAGO scale_up does DP), real LLMs keep the
        # full per-stage cap (GenZ TP/PP rows = model-parallel).
        max_gpus_for_stage = self._sweep_max_devices(search_space)
        num_device_list = _get_powers_of_two_up_to(int(max_gpus_for_stage))

        from rago.utils import get_power_of_two_list
        sweep_batch_sizes = get_power_of_two_list(max_batch_size)

        result_list = []

        sim_stage = self._map_stage(stage)

        def _analytical_component_ms(
            batch_size: int,
            component_service_ms: float,
        ) -> float:
            # Stage sweeps price complete phase calls, not scheduler cycles.
            # Engine-cycle overhead is applied only by closed_loop_sim._Engine.
            _ = batch_size
            return float(component_service_ms)

        linear_probe = self._linear_law_latency_s(record, params, 1)
        if linear_probe is not None:
            if sim_stage != "prefill":
                raise ValueError(
                    f"{stage}: linear-law component profiles support prefill-like "
                    "operators only"
                )
            # A linear-law record prices one concrete component replica. RAGO
            # performs data-parallel replica scaling; TP/PP must not be invented.
            for batch_size in sweep_batch_sizes:
                latency_s = self._linear_law_latency_s(
                    record, params, int(batch_size),
                )
                assert latency_s is not None
                corrected_ms = _analytical_component_ms(
                    int(batch_size),
                    float(latency_s) * 1000.0,
                )
                result_list.append({
                    "model.name": model_name,
                    "model.stage": "prefill",
                    "model.dec_steps": 1,
                    "model.seq_len_inference_prefill": params["input_tokens"],
                    "hardware.name": "system",
                    "hardware.num_chips": 1,
                    "batch_size": int(batch_size),
                    "time_ms": corrected_ms,
                    "winning_tp": 1,
                    "winning_pp": 1,
                })
            df = pd.DataFrame(result_list)
            return _reduce_parallelism_sweep(
                df,
                preserve_batch_axis=False,
            )

        # Calibrated (system, ceff, meff, bits) — same resolution estimate()
        # uses. A linear-law-only record returned above and never requires dead
        # roofline coefficients.
        system_name, ceff, meff, cal_bits, call_overhead_s = self._resolve_calibration(
            hardware, pipeline_config
        )
        if cal_bits:
            params = {**params, "bits": cal_bits}

        is_pd_llm_stage = _is_prefill_decode_engine_stage(stage)
        fixed_tp_pp = (
            resolve_fixed_llm_parallelism(pipeline_config, stage=stage)
            if is_pd_llm_stage else None
        )

        for num_devices in num_device_list:
            # GenZ owns the model constraints.  Build the per-batch feasible
            # sets up front so invalid attention/KV/PP layouts never reach the
            # expensive roofline calls.  Keep the old choice-major iteration
            # order to preserve deterministic tie-breaking among legal rows.
            choices = [
                result.parallelism
                for result in valid_parallelism_configs(
                    model,
                    int(num_devices),
                    batch_size=None,
                )
            ]
            if fixed_tp_pp is not None:
                choices = [
                    choice for choice in choices
                    if choice == (int(fixed_tp_pp[0]), int(fixed_tp_pp[1]))
                ]
            valid_by_batch = {
                int(batch_size): {
                    result.parallelism
                    for result in valid_parallelism_configs(
                        model,
                        int(num_devices),
                        batch_size=int(batch_size),
                    )
                }
                for batch_size in sweep_batch_sizes
            }
            for tp, pp in choices:
                # Role = the GenZ phase, but only the generator's own sweep may
                # bind to a PD role record — other CMs resolve the unified
                # generator devices (pre-taxonomy behavior preserved).
                parallel_system_name = (
                    self._system_config_for_parallelism(
                        system_name, hardware, pipeline_config,
                        stage, tp, pp,
                    )
                    if is_pd_llm_stage else system_name
                )
                for batch_size in sweep_batch_sizes:
                    if (int(tp), int(pp)) not in valid_by_batch[int(batch_size)]:
                        continue
                    try:
                        if sim_stage == "decode":
                            output = decode_moddeling(
                                Bb=1,  # vLLM serves beam=1; GenZ default Bb=4 quadrupled KV accounting (fake offload at 14B b256) and decode bytes — r13
                model=model, batch_size=batch_size,
                                input_tokens=params["input_tokens"],
                                output_tokens=params["output_tokens"],
                                system_name=parallel_system_name, bits=params["bits"],
                                ceff=ceff, meff=meff,
                                tensor_parallel=tp, pipeline_parallel=pp,
                                debug=False, model_offload=False,
                            )
                            result_list.append({
                                "model.name": model_name,
                                "model.stage": "decode",
                                "model.dec_steps": params["output_tokens"],
                                "model.seq_len_inference_prefill": params["input_tokens"],
                                "hardware.name": "system",
                                "hardware.num_chips": num_devices,
                                "batch_size": batch_size,
                                # GenZ prices the complete vLLM scheduler-step
                                # resident batch, including TP compute/memory and
                                # physical PP message passes.  The compatibility
                                # factor is analytical identity.
                                "time_ms": _analytical_component_ms(
                                    int(batch_size),
                                    scheduler_batch_traversal_latency_ms(
                                        output, pipeline_parallel=int(pp),
                                    ) * pipeline_fill_drain_factor(
                                        int(batch_size),
                                        int(pp),
                                    ),
                                ),
                                # Recorded so e2e validation can deploy vLLM
                                # with the same (TP, PP) the cost model picked
                                # as the winner. RAGO ignores these columns.
                                "winning_tp": tp,
                                "winning_pp": pp,
                            })
                        elif sim_stage == "prefill":
                            output = prefill_moddeling(
                                model=model, batch_size=batch_size,
                                input_tokens=params["input_tokens"],
                                system_name=parallel_system_name, bits=params["bits"],
                                ceff=ceff, meff=meff,
                                tensor_parallel=tp, pipeline_parallel=pp,
                                debug=False, model_offload=False,
                            )
                            result_list.append({
                                "model.name": model_name,
                                "model.stage": "prefill",
                                "model.dec_steps": 1,
                                "model.seq_len_inference_prefill": params["input_tokens"],
                                "hardware.name": "system",
                                "hardware.num_chips": num_devices,
                                "batch_size": batch_size,
                                # Coarse phase-level component price only;
                                # scheduler-cycle overhead belongs to DES.
                                "time_ms": _analytical_component_ms(
                                    int(batch_size),
                                    scheduler_batch_traversal_latency_ms(
                                        output, pipeline_parallel=int(pp),
                                    ) * pipeline_fill_drain_factor(
                                        int(batch_size),
                                        int(pp),
                                    ),
                                ),
                                "winning_tp": tp,
                                "winning_pp": pp,
                            })
                    except Exception as e:
                        # Distinguish real OOM (a CUDA/memory failure
                        # that scales with batch size — larger batches
                        # would also fail) from structural cost-model
                        # failures (e.g. GenZ ZeroDivisionError when
                        # ``num_layers % pp != 0`` — tied to (model, pp),
                        # not batch_size). The previous handler labelled
                        # everything as "OOM", masking the real bug.
                        msg = str(e).lower()
                        is_oom = (
                            "out of memory" in msg
                            or "cuda oom" in msg
                            or type(e).__name__ in ("OutOfMemoryError",)
                        )
                        if is_oom:
                            logger.warning(
                                f"Model '{model_name}' OOM at "
                                f"batch={batch_size}, tp={tp}, pp={pp}, "
                                f"num_devices={num_devices}: {e}"
                            )
                        else:
                            logger.warning(
                                f"Model '{model_name}' cost-model "
                                f"failure at batch={batch_size}, tp={tp}, "
                                f"pp={pp}, num_devices={num_devices}: "
                                f"{type(e).__name__}: {e} "
                                f"(skipping rows for this (tp, pp); "
                                f"other combos continue)"
                            )
                        # In BOTH cases the remaining batch sweep at
                        # this (tp, pp) is doomed (OOM monotonic in
                        # batch; structural failure independent of
                        # batch). Break out of the batch loop only —
                        # the (tp, pp) and num_devices loops continue.
                        break

        if not result_list:
            raise RuntimeError(
                f"Cost model failed: model '{model_name}' cannot run on "
                f"{num_device_list} GPU(s) with system '{system_name}'. "
                f"Increase max_num_gpus_per_stage in config or use a "
                f"smaller model."
            )

        df = pd.DataFrame(result_list)

        # Keep best latency per (num_gpus, batch_size) across all (TP, PP)
        # A two-role LLM phase sweep is only a component surface.  Its
        # scheduler-cycle overhead is unavailable until prefill and decode are
        # composed by DES, so component-only throughput cannot prove that one
        # resident batch dominates another.  Preserve every enumerated batch
        # after choosing the best TP/PP for that exact resource identity.
        # Single-role components retain the historical Pareto reduction.
        return _reduce_parallelism_sweep(
            df,
            preserve_batch_axis=is_pd_llm_stage,
        )

    def estimate_stage_for_devices(
        self,
        pipeline_config: dict,
        hardware: SystemConfig,
        *,
        stage: str,
        batch_size: int,
        devices: list[str],
        parallelism: tuple[int, int] | None = None,
    ) -> dict:
        """Exact-device analytical component price used by RAGO repricing.

        Unlike ``generate_sweep_df`` this never substitutes the canonical first
        N GPUs: ``devices`` is the concrete RAGO rank order for this row.  A
        supplied parallelism pins a fixed replay or a unified engine's common
        TP/PP choice; otherwise all TP/PP factorizations are considered.  This
        phase-level surface does not know scheduler-cycle composition and
        therefore never consumes an engine-cycle overhead curve.
        """
        devices = [str(device) for device in devices]
        if not devices:
            raise ValueError(f"{stage}: exact-device GenZ pricing requires devices")
        model = self._get_model(pipeline_config)
        params = self._get_params(pipeline_config)
        record, params = self._pricing_record_and_params(
            hardware, pipeline_config, params,
        )
        sim_stage = self._map_stage(stage)

        def _analytical_component_service_s(
            component_service_s: float,
        ) -> float:
            return float(component_service_s)

        linear_latency_s = self._linear_law_latency_s(
            record, params, int(batch_size),
        )
        if linear_latency_s is not None:
            chosen = parallelism or (1, 1)
            tp, pp = max(1, int(chosen[0])), max(1, int(chosen[1]))
            if sim_stage != "prefill" or len(devices) != 1 or (tp, pp) != (1, 1):
                raise ValueError(
                    f"{stage}: linear-law component pricing requires one "
                    "concrete device with TP=PP=1"
                )
            corrected_latency_s = _analytical_component_service_s(
                float(linear_latency_s)
            )
            return {
                "stage": stage,
                "devices": list(devices),
                "tp": 1,
                "pp": 1,
                "replicas": 1,
                "latency_s": corrected_latency_s,
                "throughput_qps": (
                    float(batch_size) / corrected_latency_s
                ),
                "effective_icn_gbps": None,
                "effective_icn_ll_us": None,
            }

        system_name, ceff, meff, cal_bits, call_overhead_s = self._resolve_calibration(
            hardware, pipeline_config
        )
        if cal_bits:
            params = {**params, "bits": cal_bits}
        if sim_stage not in ("prefill", "decode"):
            raise ValueError(f"{stage}: GenZ repricer only supports prefill/decode")

        if parallelism is not None:
            requested = (int(parallelism[0]), int(parallelism[1]))
            check_model_parallelism(
                model,
                tensor_parallel=requested[0],
                pipeline_parallel=requested[1],
                batch_size=int(batch_size),
            ).require()
            choices = [requested]
        else:
            fixed = resolve_fixed_llm_parallelism(pipeline_config, stage)
            if fixed is not None:
                requested = (int(fixed[0]), int(fixed[1]))
                check_model_parallelism(
                    model,
                    tensor_parallel=requested[0],
                    pipeline_parallel=requested[1],
                    batch_size=int(batch_size),
                ).require()
                choices = [requested]
            else:
                choices = [
                    result.parallelism
                    for result in valid_parallelism_configs(
                        model,
                        len(devices),
                        batch_size=int(batch_size),
                    )
                ]

        candidates: list[dict] = []
        capacity_failures: list[CMDeviceCapacityError] = []
        for tp, pp in choices:
            if int(tp) * int(pp) != len(devices):
                raise ValueError(
                    f"{stage}: structurally valid GenZ choice tp={tp},pp={pp} "
                    f"does not use the {len(devices)} concrete devices"
                )
            try:
                exact_system = self._system_config_for_parallelism(
                    system_name,
                    hardware,
                    pipeline_config,
                    stage,
                    int(tp),
                    int(pp),
                    devices_override=devices,
                )
                if sim_stage == "decode":
                    output = decode_moddeling(
                        Bb=1,
                        model=model,
                        batch_size=int(batch_size),
                        input_tokens=params["input_tokens"],
                        output_tokens=params["output_tokens"],
                        system_name=exact_system,
                        bits=params["bits"],
                        ceff=ceff,
                        meff=meff,
                        tensor_parallel=int(tp),
                        pipeline_parallel=int(pp),
                        debug=False,
                        model_offload=False,
                    )
                    component_step_s = (
                        scheduler_batch_traversal_latency_ms(
                            output, pipeline_parallel=int(pp),
                        ) * pipeline_fill_drain_factor(
                            int(batch_size), int(pp),
                        )
                        / 1000.0
                    )
                    corrected_step_s = _analytical_component_service_s(
                        component_step_s
                    )
                    latency_s = (
                        corrected_step_s * float(params["output_tokens"])
                    )
                else:
                    output = prefill_moddeling(
                        model=model,
                        batch_size=int(batch_size),
                        input_tokens=params["input_tokens"],
                        system_name=exact_system,
                        bits=params["bits"],
                        ceff=ceff,
                        meff=meff,
                        tensor_parallel=int(tp),
                        pipeline_parallel=int(pp),
                        debug=False,
                        model_offload=False,
                    )
                    component_service_s = (
                        scheduler_batch_traversal_latency_ms(
                            output, pipeline_parallel=int(pp),
                        )
                        * pipeline_fill_drain_factor(int(batch_size), int(pp))
                        / 1000.0
                    )
                    latency_s = _analytical_component_service_s(
                        component_service_s
                    )
                if latency_s <= 0:
                    raise ValueError(f"non-positive latency {latency_s}")
                candidates.append({
                    "stage": stage,
                    "devices": list(devices),
                    "tp": int(tp),
                    "pp": int(pp),
                    "replicas": 1,
                    "latency_s": float(latency_s),
                    "throughput_qps": float(batch_size) / float(latency_s),
                    "effective_icn_gbps": (
                        float(exact_system.get("ICN"))
                        if isinstance(exact_system, dict) and exact_system.get("ICN") is not None
                        else None
                    ),
                    "effective_icn_ll_us": (
                        float(exact_system.get("ICN_LL"))
                        if isinstance(exact_system, dict) and exact_system.get("ICN_LL") is not None
                        else None
                    ),
                })
            except Exception as exc:
                if not _is_exact_device_capacity_failure(exc):
                    # Every choice reaching this loop has already passed
                    # model-aware structural validation.  A failure other than
                    # GenZ's exact capacity verdict is therefore a modelling
                    # bug and must not be hidden by a successful/OOM sibling.
                    raise
                failure = CMDeviceCapacityError(
                    f"{stage}: exact-device capacity exceeded on {devices}: "
                    f"tp={tp},pp={pp}: {type(exc).__name__}: {exc}"
                )
                failure.__cause__ = exc
                capacity_failures.append(failure)
        if not candidates:
            if capacity_failures:
                aggregate = CMDeviceCapacityError(
                    f"{stage}: every exact-device TP/PP choice exceeds memory "
                    f"for {devices}: "
                    + "; ".join(str(exc) for exc in capacity_failures)
                )
                raise aggregate from capacity_failures[0]
            raise RuntimeError(
                f"{stage}: exact-device GenZ repricing produced no candidate "
                f"for {devices}"
            )
        return min(candidates, key=lambda item: item["latency_s"])


# ---------------------------------------------------------------------------
def _resolve_model_name(module: dict) -> str:
    """Resolve model name from a module config dict.

    Uses 'model' field, then maps to GenZ simulator name via the unified registry.
    """
    name = module.get("model")
    if not name:
        raise ValueError(f"Cannot resolve model name from module config: {module}")
    if isinstance(name, list):
        name = name[0]
    return resolve_genz_model(name)


# ---------------------------------------------------------------------------
# 1. Main LLM (prefill + decode) — matches get_perf_main_llm()
# ---------------------------------------------------------------------------

def _find_generator_module(pipeline_config: dict) -> dict:
    for node_line in pipeline_config.get("node_lines", []):
        for node in node_line.get("nodes", []):
            if node.get("stage") == "generator":
                modules = node.get("modules", [])
                if modules:
                    return modules[0]
    return {}


@register_cost_model("llm_sim")
class MainLLMCostModel(GenZLLMSimBase):
    """Main generative LLM: prefill + decode."""

    def _get_model(self, pipeline_config: dict) -> str:
        module = _find_generator_module(pipeline_config)
        return _resolve_model_name(module)

    def _get_params(self, pipeline_config: dict) -> dict:
        from rag_stack.cost_model.token_stats import TokenStats
        module = _find_generator_module(pipeline_config)
        ts = TokenStats.resolve(pipeline_config)
        # Effective #passages reaching the prompt (running min over the
        # retrieval->rerank/filter chain), NOT the retriever's raw top_k.
        top_k = _effective_context_top_k(pipeline_config)
        return {
            "input_tokens": ts.get_llm_input_tokens(top_k),
            "output_tokens": ts.avg_output_tokens,
            "bits": module.get("bits", "bf16"),
        }

    def _find_module(self, pipeline_config: dict) -> dict:
        return _find_generator_module(pipeline_config)

    def _get_model_name_for_df(self, pipeline_config: dict) -> str:
        module = _find_generator_module(pipeline_config)
        return _resolve_model_name(module)


# ---------------------------------------------------------------------------
# 2. Query Rewriter (prefill + decode, short sequences) — matches get_perf_rewriter()
# ---------------------------------------------------------------------------

def _find_retrieval_top_k(pipeline_config: dict) -> int:
    for node_line in pipeline_config.get("node_lines", []):
        for node in node_line.get("nodes", []):
            if node.get("stage") in ("semantic_retrieval", "hybrid_retrieval"):
                if "top_k" not in node:
                    raise ValueError("Retrieval node missing 'top_k'.")
                return int(node["top_k"])
    raise ValueError("No retrieval node found. Cannot estimate input tokens.")


def _scalar_top_k(value) -> int:
    """Sweep-spec tolerant top_k: lists take their max (upper bound)."""
    if isinstance(value, list):
        value = max(value)
    return int(value)


def _effective_context_top_k(
    pipeline_config: dict, before_node_type: str | None = None
) -> int:
    """#passages actually flowing OUT of the retrieval-side chain — i.e. into
    the generator prompt (or into ``before_node_type`` when given).

    The retriever emits ``retrieval.top_k`` candidates; every downstream
    reranker / filter that declares a ``top_k`` SELECTS, so the count reaching
    the prompt is the running ``min`` along the chain — NOT the retriever's
    top_k. (retrieval 64 → rerank 8 puts 8 passages in the prompt, not 64;
    retrieval 2 → rerank 8 puts 2.) Using the retriever's value alone
    mis-costs the generator prefill in both directions across the space.
    """
    k: int | None = None
    for node_line in pipeline_config.get("node_lines", []):
        for node in node_line.get("nodes", []):
            nt = node.get("stage")
            if before_node_type is not None and nt == before_node_type:
                if k is None:
                    raise ValueError(
                        f"No retrieval node found before {before_node_type!r}."
                    )
                return k
            if nt in ("semantic_retrieval", "hybrid_retrieval"):
                if "top_k" not in node:
                    raise ValueError("Retrieval node missing 'top_k'.")
                k = _scalar_top_k(node["top_k"])
            elif nt in ("passage_reranker", "passage_filter") and k is not None:
                tk = node.get("top_k")
                if tk is not None:
                    k = min(k, _scalar_top_k(tk))
    if k is None:
        raise ValueError("No retrieval node found. Cannot estimate input tokens.")
    return k


def _find_query_expansion_module(pipeline_config: dict) -> dict:
    for node_line in pipeline_config.get("node_lines", []):
        for node in node_line.get("nodes", []):
            if node.get("stage") == "query_expansion":
                modules = node.get("modules", [])
                for m in modules:
                    if m.get("component") in ("hyde", "query_decompose", "multi_query_expansion"):
                        return m
    return {}


@register_cost_model("query_expansion_sim")
class QueryExpansionCostModel(GenZLLMSimBase):
    """Query expansion LLM: prefill + decode, short sequences."""

    def _get_model(self, pipeline_config: dict) -> str:
        module = _find_query_expansion_module(pipeline_config)
        return _resolve_model_name(module)

    def _get_params(self, pipeline_config: dict) -> dict:
        from rag_stack.cost_model.token_stats import TokenStats
        module = _find_query_expansion_module(pipeline_config)
        ts = TokenStats.resolve(pipeline_config)
        mod_type = module.get("component", "")
        return {
            "input_tokens": ts.get_query_expansion_input_tokens(),
            "output_tokens": ts.get_query_expansion_output_tokens(mod_type),
            "bits": module.get("bits", "int8"),
        }

    def _find_module(self, pipeline_config: dict) -> dict:
        return _find_query_expansion_module(pipeline_config)

    def _get_model_name_for_df(self, pipeline_config: dict) -> str:
        module = _find_query_expansion_module(pipeline_config)
        return _resolve_model_name(module)


# ---------------------------------------------------------------------------
# 3. Query-embedding encoder — moved to
#    rag_stack/cost_model/retrieval_encode.py (RetrievalEncodeCostModel,
#    "retrieval_encode_sim"). The removed "encoder_sim" stub here modeled a
#    database encoder off a ``vdb["encoder"]`` config shape nothing produced
#    and was never wired into a stage; deleted to keep ONE encoder model.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. Passage Reranker (prefill only) — matches get_perf_reranker()
# ---------------------------------------------------------------------------

def _find_passage_reranker_module(pipeline_config: dict) -> dict:
    for node_line in pipeline_config.get("node_lines", []):
        for node in node_line.get("nodes", []):
            if node.get("stage") == "passage_reranker":
                modules = node.get("modules", [])
                if modules:
                    return modules[0]
    return {}


@register_cost_model("passage_reranker_sim")
class PassageRerankerCostModel(GenZLLMSimBase):
    """Passage reranker: prefill only, encoder-only model."""

    # Default models per encoder-based reranker component
    # Verified against actual evaluator code defaults
    _DEFAULT_MODELS: dict[str, str] = {
        "tart": "tart_flan_t5_xl",                       # facebook/tart-full-flan-t5-xl (Flan-T5 encoder-only, ~341M)
        "monot5": "monot5-3b",                           # castorini/monot5-3b-msmarco-10k (T5-3B encoder-decoder)
        "flag_embedding_reranker": "bge_reranker_large", # BAAI/bge-reranker-large (XLM-R, ~560M)
        "openvino_reranker": "bge_reranker_large",       # same BGE weights, OpenVINO runtime
        "sentence_transformer_reranker": "ms_marco_minilm_l2",  # cross-encoder/ms-marco-MiniLM-L-2-v2 (~22M)
        "colbert_reranker": "colbertv2",                 # colbert-ir/colbertv2.0 (BERT-base, ~110M)
        # Remaining fall back to sbert_120m as approximation:
        # flashrank_reranker: TinyBERT ~4.4M
        # koreranker: ko-reranker ~278M
        # cohere/jina/voyageai/mixedbreadai: API-based, simulate as small encoder
        # time_reranker: unmapped in cost_model_mapping.json (pure-Python sort)
    }

    def _sweep_max_devices(self, search_space: dict) -> int:
        # Rerankers are small BERT cross-encoders / late-interaction models with
        # NO tensor-parallel implementation in the real runtime — multi-GPU is
        # DATA-PARALLEL (replicas), not TP. So emit ONLY single-chip GenZ rows;
        # RAGO's scale_up_performance_to_more_chips then extrapolates num_chips
        # 2,4,… as qps×N (data-parallel), matching the measured DP-replica path.
        return 1

    def _get_model(self, pipeline_config: dict) -> str | ModelConfig:
        module = _find_passage_reranker_module(pipeline_config)
        # User-specified model_name in config takes priority
        name = module.get("model")
        if name:
            if isinstance(name, list):
                name = name[0]
            return resolve_genz_model(name)
        # Fall back to component-specific default
        mod_type = module.get("component", "")
        default_key = self._DEFAULT_MODELS.get(mod_type, "sbert_120m")
        return resolve_genz_model(default_key)

    def _get_params(self, pipeline_config: dict) -> dict:
        from rag_stack.cost_model.token_stats import TokenStats
        module = _find_passage_reranker_module(pipeline_config)
        ts = TokenStats.resolve(pipeline_config)
        return {
            "input_tokens": ts.get_passage_reranker_input_tokens(),
            "output_tokens": 1,
            "bits": module.get("bits", "bf16"),
        }

    def _find_module(self, pipeline_config: dict) -> dict:
        return _find_passage_reranker_module(pipeline_config)

    def _get_model_name_for_df(self, pipeline_config: dict) -> str:
        module = _find_passage_reranker_module(pipeline_config)
        name = module.get("model")
        if name:
            if isinstance(name, list):
                name = name[0]
            return name
        # Fall back to the component's default model key (matches _get_model's
        # resolution path). This keeps filenames like
        # ``tart_flan_t5_xl__rtx_3090__f32.json`` aligned with the actual
        # ModelConfig used for prediction.
        mod_type = module.get("component", "")
        return self._DEFAULT_MODELS.get(mod_type, "sbert_120m")

    def _get_benchmark_runner(self, pipeline_config: dict):
        """Dispatch to per-component runner via the reranker factory."""
        from rag_stack.cost_model.llm_sim.calibration.reranker import get_runner
        module = _find_passage_reranker_module(pipeline_config)
        component = module.get("component", "")
        return get_runner(component)


# ---------------------------------------------------------------------------
# 4b. LLM-based Passage Reranker (prefill + decode) — rankgpt, upr, etc.
# ---------------------------------------------------------------------------

@register_cost_model("passage_reranker_llm_sim")
class PassageRerankerLLMCostModel(GenZLLMSimBase):
    """LLM-based passage reranker (rankgpt, upr, flag_embedding_llm_reranker):
    uses a generative LLM with prefill + decode, not an encoder model."""

    # Default models per LLM-based reranker component
    _DEFAULT_MODELS: dict[str, str] = {
        "flag_embedding_llm_reranker": "bge-reranker-v2-gemma",  # ~2B (gemma-2-2b)
        "rankgpt": "llama-2-7b",  # LLM-based, simulate as ~7B local deployment
        "upr": "t5-large",  # t5-large ~770M
    }

    def _get_model(self, pipeline_config: dict) -> str | ModelConfig:
        module = _find_passage_reranker_module(pipeline_config)
        # Try explicit model field first
        name = module.get("model")
        if name:
            if isinstance(name, list):
                name = name[0]
            return resolve_genz_model(name)
        # Fall back to component-specific default
        mod_type = module.get("component", "")
        default = self._DEFAULT_MODELS.get(mod_type, "llama-2-7b")
        return resolve_genz_model(default)

    def _get_params(self, pipeline_config: dict) -> dict:
        from rag_stack.cost_model.token_stats import TokenStats
        module = _find_passage_reranker_module(pipeline_config)
        ts = TokenStats.resolve(pipeline_config)
        top_k = _find_retrieval_top_k(pipeline_config)
        if isinstance(top_k, list):
            top_k = max(top_k)
        return {
            "input_tokens": ts.get_passage_reranker_input_tokens() * top_k,
            "output_tokens": 1,
            "bits": module.get("bits", "bf16"),
        }

    def _find_module(self, pipeline_config: dict) -> dict:
        return _find_passage_reranker_module(pipeline_config)

    def _get_model_name_for_df(self, pipeline_config: dict) -> str:
        module = _find_passage_reranker_module(pipeline_config)
        name = module.get("model")
        if name:
            if isinstance(name, list):
                name = name[0]
            return name
        # Calibration identity must name the same model architecture consumed
        # by _get_model(); using the component name here could report a profile
        # READY while production looked up a different record.
        mod_type = module.get("component", "")
        return self._DEFAULT_MODELS.get(mod_type, "llama-2-7b")


# ---------------------------------------------------------------------------
# 5. Passage Compressor (prefill + decode) — LLM-based (tree_summarize, refine)
# ---------------------------------------------------------------------------

def _find_passage_compressor_module(pipeline_config: dict) -> dict:
    for node_line in pipeline_config.get("node_lines", []):
        for node in node_line.get("nodes", []):
            if node.get("stage") == "passage_compressor":
                modules = node.get("modules", [])
                if modules:
                    return modules[0]
    return {}


@register_cost_model("passage_compressor_sim")
class PassageCompressorCostModel(GenZLLMSimBase):
    """Passage compressor (tree_summarize, refine): LLM prefill + decode."""

    def _get_model(self, pipeline_config: dict) -> str | ModelConfig:
        return _resolve_model_name(_find_passage_compressor_module(pipeline_config))

    def _get_params(self, pipeline_config: dict) -> dict:
        from rag_stack.cost_model.token_stats import TokenStats
        module = _find_passage_compressor_module(pipeline_config)
        ts = TokenStats.resolve(pipeline_config)
        # Passages reaching the compressor = chain min up to (excluding) it.
        top_k = _effective_context_top_k(
            pipeline_config, before_node_type="passage_compressor"
        )
        return {
            "input_tokens": ts.get_passage_compressor_input_tokens(top_k),
            "output_tokens": ts.get_passage_compressor_output_tokens(),
            "bits": module.get("bits", "bf16"),
        }

    def _find_module(self, pipeline_config: dict) -> dict:
        return _find_passage_compressor_module(pipeline_config)

    def _get_model_name_for_df(self, pipeline_config: dict) -> str:
        module = _find_passage_compressor_module(pipeline_config)
        return _resolve_model_name(module)


# ---------------------------------------------------------------------------
# 5b. Token Pruning Compressor (prefill only) — longllmlingua
# ---------------------------------------------------------------------------

@register_cost_model("passage_compressor_token_pruning_sim")
class PassageCompressorTokenPruningCostModel(GenZLLMSimBase):
    """Token pruning compressor (longllmlingua): uses a small LM (e.g. GPT-2/LLaMA-7B)
    in prefill-only mode to compute token importance scores, no decode step."""

    def _get_model(self, pipeline_config: dict) -> str | ModelConfig:
        # longllmlingua typically uses a small LM like llama-2-7b
        return resolve_genz_model("llama-2-7b")

    def _get_params(self, pipeline_config: dict) -> dict:
        from rag_stack.cost_model.token_stats import TokenStats
        ts = TokenStats.resolve(pipeline_config)
        # Passages reaching the compressor = chain min up to (excluding) it.
        top_k = _effective_context_top_k(
            pipeline_config, before_node_type="passage_compressor"
        )
        return {
            "input_tokens": ts.get_passage_compressor_input_tokens(top_k),
            "output_tokens": 1,  # no generation, just scoring tokens
            "bits": "bf16",
        }

    def _find_module(self, pipeline_config: dict) -> dict:
        return _find_passage_compressor_module(pipeline_config)

    def _get_model_name_for_df(self, pipeline_config: dict) -> str:
        return "longllmlingua"


# ---------------------------------------------------------------------------
# 6. Passage Filter (prefill only) — embedding-based similarity filtering
# ---------------------------------------------------------------------------

def _find_passage_filter_module(pipeline_config: dict) -> dict:
    for node_line in pipeline_config.get("node_lines", []):
        for node in node_line.get("nodes", []):
            if node.get("stage") == "passage_filter":
                modules = node.get("modules", [])
                if modules:
                    return modules[0]
    return {}


@register_cost_model("passage_filter_sim")
class PassageFilterCostModel(GenZLLMSimBase):
    """Passage filter (similarity-based): prefill only, embedding model."""

    def _get_model(self, pipeline_config: dict) -> str | ModelConfig:
        return resolve_genz_model("sbert_120m")

    def _get_params(self, pipeline_config: dict) -> dict:
        from rag_stack.cost_model.token_stats import TokenStats
        module = _find_passage_filter_module(pipeline_config)
        ts = TokenStats.resolve(pipeline_config)
        return {
            "input_tokens": ts.get_passage_filter_input_tokens(),
            "output_tokens": 1,
            "bits": module.get("bits", "bf16"),
        }

    def _find_module(self, pipeline_config: dict) -> dict:
        return _find_passage_filter_module(pipeline_config)

    def _get_model_name_for_df(self, pipeline_config: dict) -> str:
        return "filter"
