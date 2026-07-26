"""Adapter: pipeline config -> HNSW cost model -> CostModelResult."""

from __future__ import annotations

from pathlib import Path

from rag_stack.cost_model import register_cost_model
from rag_stack.cost_model.base import CostModelModule
from rag_stack.cost_model.calibration_mode import is_uncalibrated_spec_mode
from rag_stack.cost_model.types import CostModelResult, OperatorBreakdown
from rag_stack.types import SystemConfig

from .hnsw_config import HNSWConfig
from .graph_stats import estimate_graph_stats
from .workload_predictor import WorkloadPredictor
from .cost_model import HNSWCoefficients, HNSWCostModel
from .coefficients import (
    build_hardware_from_profile,
    load_profile_for_preset,
    profile_path_for_preset,
)


def _find_vectordb_config(pipeline_config: dict, vectordb_name: str) -> dict:
    for vdb in pipeline_config.get("vectordb", []):
        if vdb.get("name") == vectordb_name:
            return vdb
    return {}


def _find_retrieval_module(pipeline_config: dict) -> tuple[dict, dict]:
    for node_line in pipeline_config.get("node_lines", []):
        for node in node_line.get("nodes", []):
            if node.get("stage") in ("semantic_retrieval", "hybrid_retrieval"):
                modules = node.get("modules", [])
                if modules:
                    return node, modules[0]
    return {}, {}


@register_cost_model("faiss_hnsw")
class FaissHNSWCostModel(CostModelModule):
    """CostModelModule adapter for HNSW.

    Resolves the workload predictor PER (dataset, embedding) — the trained-dataset
    half of the split profile — at estimate() time, falling back to the shipped
    default for unknown datasets. The exact-CPU system profile supplies every
    non-raw HNSW hardware-model parameter plus the four standalone-fitted
    empirical coefficients; raw CPU facts come from the matching preset or
    inline CPU config.
    At estimate() time: extract config -> predict workload -> roofline -> result.
    """

    # Predictors cached by (dataset_name, embedding_model) across evaluations.
    _predictor_cache: dict = {}

    @classmethod
    def family(cls) -> str:
        return "retrieval"

    def __init__(self):
        pass

    def init(self, ctx) -> None:
        """One-time CM-init: eagerly BUILD-on-miss the per-(corpus, embedding)
        workload profiles (HNSW predictor + IVF imbalance) from the controller-
        supplied vectors, so the HNSW model starts with the real ndis/nhops
        predictor instead of the generic shipped default.

        Delegates to the SINGLE unified entry (it builds BOTH FAISS families).
        Guarded + idempotent (cache-hit cheap), so running it again here after
        the IVF adapter's init() is a no-op. No vector_provider → no-op.
        """
        if getattr(ctx, "vector_provider", None) is None:
            return
        try:
            from rag_stack.cost_model.workload_profile import ensure_workload_profiles
            ensure_workload_profiles(ctx)
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "[cm] workload-profile build skipped: %s", exc)

    def _resolve_predictor(self, pipeline_config: dict, vdb: dict):
        """Per-dataset trained predictor (calibration_profiles/faiss_hnsw/
        <dataset>__<embedding>.joblib), else the shipped default. Cached by key."""
        ds = ((pipeline_config.get("dataset") or {}).get("dataset_name")
              or vdb.get("dataset_name") or "")
        emb = vdb.get("embedding_model") or ""
        key = (ds, emb)
        cache = type(self)._predictor_cache
        if key not in cache:
            cache[key] = (WorkloadPredictor.load_for_dataset(ds, emb)
                          or WorkloadPredictor.load_default())
        return cache[key]

    def estimate(
        self, pipeline_config: dict, hardware: SystemConfig,
    ) -> CostModelResult:
        hnsw_cfg = self._build_hnsw_config(pipeline_config)
        graph_stats = estimate_graph_stats(
            hnsw_cfg.N, hnsw_cfg.d, hnsw_cfg.M, hnsw_cfg.ef_construction,
        )
        # The index-free predictor needs the corpus LID (degree/layer are unused).
        # Read the LID harvested from the corpus at ingest time (``lid`` on the
        # retrieval module or vectordb config — the cheap, data-only stat that
        # does NOT require building the index). Absent it, predict() falls back to
        # ambient dim (uniform-random assumption).
        node, module = _find_retrieval_module(pipeline_config)
        vdb = _find_vectordb_config(pipeline_config, module.get("vectordb", ""))
        if (
            vdb.get("workload_calibration") is not None
            or module.get("workload_calibration") is not None
        ):
            raise ValueError(
                "workload_calibration is retired; HNSW workload evidence must "
                "come from the exact corpus-and-embedding profile"
            )
        corpus_lid = module.get("lid", vdb.get("lid"))
        if corpus_lid is not None:
            graph_stats.lid = float(corpus_lid)

        faiss_hw, empirical_coefficients = self._resolve_hardware(
            pipeline_config, hardware,
        )
        predictor = self._resolve_predictor(pipeline_config, vdb)
        workload = predictor.predict(hnsw_cfg, graph_stats)

        model = HNSWCostModel(
            hnsw_cfg,
            faiss_hw,
            workload,
            coefficients=empirical_coefficients,
        )
        breakdown = model.estimate()
        return self._translate_result(breakdown)

    def _resolve_hardware(
        self,
        pipeline_config: dict,
        hardware: SystemConfig,
    ):
        """Resolve exact HNSW calibration or the explicit spec fallback.

        File presence is checked separately from loading so
        ``uncalibrated_spec`` applies only to a genuinely absent exact profile.
        A present directory, unreadable JSON, malformed schema, or identity
        mismatch still enters the strict loader and raises.
        """

        cpu_key = str(
            getattr(hardware, "cpu_key", "")
            or getattr(hardware, "cpu_preset", "")
            or ""
        )
        profile_path = Path(profile_path_for_preset(cpu_key))

        # Mirror the production store's thread count (default 1). Calibration
        # is single-threaded; larger values are roofline extrapolations.
        sys_retrieval = (pipeline_config.get("system") or {}).get("retrieval")
        if not isinstance(sys_retrieval, dict):
            sys_retrieval = {}
        active_threads = max(
            1, int(sys_retrieval.get("faiss_num_threads", 1) or 1),
        )

        if profile_path.exists() or profile_path.is_symlink():
            profile = load_profile_for_preset(cpu_key)
            return (
                build_hardware_from_profile(
                    profile,
                    active_threads=active_threads,
                    system_config=hardware,
                ),
                profile.empirical_coefficients,
            )

        if is_uncalibrated_spec_mode(pipeline_config, hardware):
            from .hardware import system_config_to_faiss_hardware
            from .system_presets import SYSTEM_PRESETS

            if (
                getattr(hardware, "cpu_preset", "")
                and cpu_key in SYSTEM_PRESETS
            ):
                faiss_hardware = SYSTEM_PRESETS[cpu_key](
                    active_threads=active_threads,
                )
            else:
                # These are the HNSW module's documented low-level analytical
                # defaults.  They are reachable in production only through the
                # explicit uncalibrated_spec mode.
                faiss_hardware = system_config_to_faiss_hardware(
                    hardware,
                    active_threads=active_threads,
                    simd_efficiency=0.55,
                    dram_random_access_latency_ns=100.0,
                    scattered_bw_ceiling=0.35,
                    numa_local_bytes=0,
                    numa_remote_bw_ceiling=0.06,
                )
            return faiss_hardware, HNSWCoefficients()

        # Preserve the strict loader's existing missing-profile diagnostic.
        profile = load_profile_for_preset(cpu_key)
        raise AssertionError(f"unreachable after loading HNSW profile {profile!r}")

    def _build_hnsw_config(self, pipeline_config: dict) -> HNSWConfig:
        node, module = _find_retrieval_module(pipeline_config)
        vectordb_name = module.get("vectordb", "")
        vdb = _find_vectordb_config(pipeline_config, vectordb_name)

        if vdb.get("N") is None:
            # No silent 1,000,000-vector default: pricing an index whose size
            # is unknown was a mispricing bug. The corpus size reaches the CM
            # through WorkflowSchema.corpus_stats.vectordb_index_sizes
            # (materialized back onto vectordb[i]["N"] at the entry).
            raise ValueError(
                f"vectordb {vectordb_name or vdb.get('name') or '<unnamed>'!r} "
                "has no 'N' (index vector count). The FAISS HNSW cost model "
                "requires the producer-stamped corpus size via "
                "WorkflowSchema.corpus_stats.vectordb_index_sizes."
            )
        N = int(vdb["N"])
        d = int(vdb.get("embedding_dim", 768))
        M = int(vdb.get("M", 16))
        ef_search = int(module.get("ef_search", vdb.get("ef_search", 64)))
        ef_construction = int(vdb.get("ef_construction", 200))
        top_k = int(node.get("top_k", module.get("top_k", 10)))
        batch_size = int(module.get("batch_size", 1))

        return HNSWConfig(
            N=N, d=d, M=M, ef_search=ef_search, k=top_k,
            batch_size=batch_size, ef_construction=ef_construction,
        )

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
