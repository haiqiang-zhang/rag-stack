"""Adapter: global types → faiss_ivf_sim internals → CostModelResult."""

from __future__ import annotations

import os

from rag_stack.calibration_paths import system_profile_path
from rag_stack.cost_model import register_cost_model
from rag_stack.cost_model.base import CostModelModule
from rag_stack.cost_model.calibration_mode import is_uncalibrated_spec_mode
from rag_stack.cost_model.types import CostModelResult, OperatorBreakdown
from rag_stack.types import SystemConfig

from .ivfpq_config import IVFPQConfig
from .systems import SystemConfig as FaissHardwareConfig
from .cost_model import IVFPQCostModel


def _find_vectordb_config(pipeline_config: dict, vectordb_name: str) -> dict:
    """Look up a vectordb entry by name from the pipeline config."""
    for vdb in pipeline_config.get("vectordb", []):
        if vdb.get("name") == vectordb_name:
            return vdb
    return {}


def _system_retrieval(pipeline_config: dict) -> dict:
    """The resolved ``system.retrieval`` runtime block ({} when absent)."""
    block = (pipeline_config.get("system") or {}).get("retrieval")
    return block if isinstance(block, dict) else {}


def _find_retrieval_module(pipeline_config: dict) -> tuple[dict, dict]:
    """Find the retrieval node and its active vectordb module.

    Returns:
        (node_dict, module_dict) for the first retrieval node found.
    """
    for node_line in pipeline_config.get("node_lines", []):
        for node in node_line.get("nodes", []):
            if node.get("stage") in ("semantic_retrieval", "hybrid_retrieval"):
                modules = node.get("modules", [])
                if modules:
                    return node, modules[0]
    return {}, {}


@register_cost_model("faiss_ivf")
class FaissIVFCostModel(CostModelModule):
    """Unified IVF cost model (IVF-PQ + IVF-Flat).

    Both index types run through the same analytical engine
    (:class:`IVFPQCostModel`); ``index_type`` (read from the vectordb config,
    default ``"pq"``) selects the PQ pipeline (LUT + PQ scan + optional refine)
    or the Flat pipeline (exact full-vector scan).
    """

    @classmethod
    def family(cls) -> str:
        return "retrieval"

    @classmethod
    def capabilities(cls) -> list[str]:
        # Profiles are served via a dedicated extension router (shape differs
        # from the LLM-sim CalibrationRecord); advertise the capability here
        # so the frontend renders the Profiles tab uniformly.
        return ["presets", "calibration_profiles", "benchmarking"]

    @classmethod
    def profiles_subdir(cls) -> str | None:
        return "faiss_ivf"

    # The run's corpus id (set by init) and a lazy {profile_key: profile} cache.
    # Class-level so the per-eval instances build_performance_stage_cost_models() creates all
    # share them. Empty corpus id → estimate() falls back to balanced cells (f=1).
    _corpus_id: str = ""
    _profile_cache: dict = {}

    def init(self, ctx) -> None:
        """Record the run's corpus id so estimate() can load the right per-(corpus,
        embedding) cell-imbalance profile lazily.

        The profiles themselves are BUILT + saved by the eval backend when it
        builds the index (the index is already in hand there). Here the cost model
        only CONSUMES them: on CM-init / system-migration it loads instantly (no
        index build); if a profile is missing it falls back to balanced cells
        (f=1) until the eval backend populates it. ``embedding_model`` can be a
        search dimension, so the actual profile key is resolved per config in
        :meth:`_cell_imbalance`, not fixed here.
        """
        type(self)._corpus_id = str(getattr(ctx, "corpus_key", "") or "")
        type(self)._profile_cache = {}

        # Eagerly BUILD-on-miss the per-(corpus, embedding) workload profiles
        # (IVF imbalance + HNSW predictor) from the controller-supplied vectors,
        # so the CM starts saturated instead of degrading to f=1 until an eval
        # builds them. Guarded + idempotent (cache-hit cheap), so it's safe even
        # though the HNSW adapter's init() makes the same call. No vector_provider
        # (e.g. measured runs, or the direct-assembly benchmark path) → no-op.
        if getattr(ctx, "vector_provider", None) is not None:
            try:
                from rag_stack.cost_model.workload_profile import ensure_workload_profiles
                ensure_workload_profiles(ctx)
            except Exception as exc:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning(
                    "[cm] workload-profile build skipped: %s", exc)

    def _cell_imbalance(self, embedding_model: str, nlist: int, nprobe: int,
                        corpus_id: str = "", n_vectors: int = 0) -> float:
        """f for this config's (corpus, embedding, nlist, nprobe); 1.0 if no profile.

        Keyed by the SHARED ``imbalance_key`` so it finds whatever the eval
        backend saved. Per-(corpus,embedding) profile is loaded once and cached.
        ``corpus_id`` (the pipeline_config dataset_name) is the fallback when the
        CM was inited without a ctx carrying ``corpus_key`` (e.g. the fixed-
        placement benchmark path drives the assembly directly, not the controller).
        """
        cid = self._corpus_id or corpus_id
        if not cid:
            return 1.0
        from .imbalance import profile_cell_imbalance
        return profile_cell_imbalance(
            cid,
            embedding_model or "",
            nlist,
            nprobe,
            n_vectors=n_vectors,
            profile_cache=type(self)._profile_cache,
        )

    def estimate(
        self, pipeline_config: dict, hardware: SystemConfig
    ) -> CostModelResult:
        ivf_cfg = self._build_config(pipeline_config)
        faiss_hw = self._resolve_hardware(pipeline_config, hardware)
        model = IVFPQCostModel(ivf_cfg, faiss_hw)
        return self._translate_result(model.estimate())

    # ------------------------------------------------------------------

    def _build_config(self, pipeline_config: dict) -> IVFPQConfig:
        """Build an IVFPQConfig (pq or flat) from the pipeline config.

        Index params (N, d, nlist, and for PQ: M/nbits) come from the vectordb
        section; search params (nprobe, top_k, batch_size) from the retrieval
        module. ``index_type`` (default "pq") selects the pipeline.
        """
        node, module = _find_retrieval_module(pipeline_config)
        vectordb_name = module.get("vectordb", "")
        vdb = _find_vectordb_config(pipeline_config, vectordb_name)

        index_type = vdb.get("index_type", "pq")
        if vdb.get("N") is None:
            # No silent 1,000,000-vector default: pricing an index whose size
            # is unknown was a mispricing bug. The corpus size reaches the CM
            # through WorkflowSchema.corpus_stats.vectordb_index_sizes
            # (materialized back onto vectordb[i]["N"] at the entry).
            raise ValueError(
                f"vectordb {vectordb_name or vdb.get('name') or '<unnamed>'!r} "
                "has no 'N' (index vector count). The FAISS IVF cost model "
                "requires the producer-stamped corpus size via "
                "WorkflowSchema.corpus_stats.vectordb_index_sizes."
            )
        N = int(vdb["N"])
        d = int(vdb.get("embedding_dim", 768))
        nlist = int(vdb.get("nlist", 1024))

        # Search params from module + node. nprobe default mirrors the runtime
        # default (10) and is clamped to nlist, like the store's query-time override.
        nprobe = int(module.get("nprobe") or vdb.get("nprobe") or 10)
        nprobe = min(nprobe, nlist)
        top_k = int(node.get("top_k", module.get("top_k", 10)))
        batch_size = int(module.get("batch_size", 1))
        # parallel_mode is a SYSTEM-level runtime knob (system.retrieval.
        # faiss_ivf_parallel_mode); the vectordb-level key is the legacy fallback.
        parallel_mode = int(
            _system_retrieval(pipeline_config).get(
                "faiss_ivf_parallel_mode", vdb.get("parallel_mode", 0),
            )
        )

        # Data-aware scan count: per-(corpus, embedding) cell-imbalance profile
        # (1.0 fallback if no profile yet → balanced cells, no regression). The
        # pipeline_config dataset_name is the corpus-id fallback for the direct-
        # assembly path (no controller ctx → self._corpus_id empty).
        cell_imbalance = self._cell_imbalance(
            str(vdb.get("embedding_model") or ""), nlist, nprobe,
            corpus_id=str((pipeline_config.get("dataset") or {}).get("dataset_name") or ""),
            n_vectors=N)

        if index_type == "flat":
            return IVFPQConfig(
                index_type="flat",
                d=d, N=N, nlist=nlist, nprobe=nprobe,
                k=top_k, batch_size=batch_size, parallel_mode=parallel_mode,
                flat_dtype_bytes=int(vdb.get("flat_dtype_bytes", 4)),
                cell_imbalance=cell_imbalance,
            )

        # PQ — the runtime builds the index EXACTLY as configured (no
        # auto-reduction), so the model prices the configured shape as-is.
        M = int(vdb.get("M", 96))
        nbits = int(vdb.get("nbits", 8))
        return IVFPQConfig(
            index_type="pq",
            d=d, N=N, nlist=nlist, nprobe=nprobe, M=M, nbits=nbits,
            k=top_k, batch_size=batch_size, parallel_mode=parallel_mode,
            refine_ratio=float(vdb.get("refine_ratio", 0.0)),
            refine_dtype_bytes=int(vdb.get("refine_dtype_bytes", 4)),
            use_precomputed_table=bool(vdb.get("use_precomputed_table", True)),
            # Unset → None → IVFPQConfig applies the FAISS per-kernel default
            # (FastScan False, std True). Explicit YAML value is honored.
            by_residual=(None if vdb.get("by_residual") is None
                         else bool(vdb.get("by_residual"))),
            cell_imbalance=cell_imbalance,
        )

    def _resolve_hardware(
        self, pipeline_config: dict, hardware: SystemConfig
    ) -> FaissHardwareConfig:
        """Resolve the faiss hardware config: prefer a calibrated coefficients
        JSON (shared physical + per-index efficiencies), else translate the
        global SystemConfig directly.

        Thread count mirrors the production store: the measured store runs at
        OMP=1 unless system.retrieval.faiss_num_threads raises it per trial, so
        the model uses the SAME resolved value (default 1). NOTE: calibration
        profiles were fit at active_threads=1; thread-scaling beyond 1 is
        extrapolated by the roofline, not calibrated.
        """
        _, module = _find_retrieval_module(pipeline_config)
        vectordb_name = module.get("vectordb", "")
        vdb = _find_vectordb_config(pipeline_config, vectordb_name)
        if vdb.get("coefficient_overrides") is not None:
            raise ValueError(
                "vectordb.coefficient_overrides is retired; calibrate and "
                "promote an exact CPU-keyed FAISS profile instead"
            )
        if vdb.get("calibrated_coefficients") is not None:
            raise ValueError(
                "vectordb.calibrated_coefficients is retired; the exact "
                "CPU-keyed profile is resolved from system.cpu.hardware_key"
            )
        sys_retrieval = _system_retrieval(pipeline_config)
        active_threads = max(1, int(sys_retrieval.get("faiss_num_threads", 1) or 1))
        cpu_key = (
            getattr(hardware, "cpu_key", "")
            or getattr(hardware, "cpu_preset", "")
        )

        # Prefer the COMMITTED calibrated profile for this CPU over the raw preset spec:
        # auto-resolve calibration_profiles/faiss_ivf/<cpu_key>.json when no explicit path.
        if not cpu_key:
            raise ValueError(
                "IVF pricing requires system.cpu.hardware_key "
                "(or a CPU preset)"
            )
        coefficients_path = str(system_profile_path("faiss_ivf", cpu_key))
        # ``lexists`` treats a broken symlink as present-but-invalid evidence;
        # spec mode must not disguise it as an absent profile.
        if not os.path.lexists(coefficients_path):
            if is_uncalibrated_spec_mode(pipeline_config, hardware):
                from .system_presets import SYSTEM_PRESETS
                from .systems import system_config_to_faiss_hardware

                if (
                    getattr(hardware, "cpu_preset", "")
                    and cpu_key in SYSTEM_PRESETS
                ):
                    return SYSTEM_PRESETS[cpu_key](
                        active_threads=active_threads,
                    )
                return system_config_to_faiss_hardware(
                    hardware,
                    active_threads=active_threads,
                )
            raise ValueError(
                f"missing IVF CPU profile for hardware {cpu_key!r}: "
                f"{coefficients_path}"
            )

        from .coefficients import load_coefficients, build_hardware_from_coefficients
        coefficients = load_coefficients(coefficients_path)
        faiss_hw = build_hardware_from_coefficients(
            coefficients,
            active_threads=active_threads,
            system_config=hardware,
        )
        return faiss_hw

    def _translate_result(self, breakdown) -> CostModelResult:
        operators = [
            OperatorBreakdown(
                name=op.name,
                compute_time_s=op.compute_time_s,
                memory_time_s=op.memory_time_s,
                time_s=op.time_s,
                bottleneck=op.bottleneck,
            )
            for op in breakdown.operators
        ]
        return CostModelResult(
            latency_s=breakdown.total_time_s,
            throughput_qps=breakdown.throughput_qps,
            operators=operators,
        )
