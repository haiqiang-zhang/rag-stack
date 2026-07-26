"""Cost model base interface.

Each cost model module implements the CostModelModule ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from rag_stack.types import SystemConfig
from rag_stack.cost_model.types import CostModelResult


@dataclass
class CmInitContext:
    """Context handed to :meth:`CostModelModule.init` once at controller startup.

    It lets a component build any DATA-side artifact it needs ONCE — e.g. the
    IVF cell-imbalance profile or the HNSW workload predictor — instead of per
    evaluation. The unified primitive both FAISS families need is corpus base
    vectors + a query-vector sample for a given embedding model; the controller
    supplies them DOWN via ``vector_provider`` (which goes through the production
    SRE encode path + global embedding cache) so ``cost_model`` never imports
    ``static_rag_evaluator``. From those vectors the cost model builds the
    indexes ITSELF with faiss and computes/caches the profile — eagerly, on a
    cache MISS, at startup (build-on-miss); a warm profile loads with no build,
    which is what the CM-init (saturated-$P$-from-the-start) and system-migration
    paths require.

    Fields are best-effort: a component that finds what it needs missing simply
    falls back to its default behaviour (e.g. balanced cells, ``f = 1``).
    """
    config: dict                                        # full run config
    corpus_key: str = ""                                # corpus (dataset) id → profile key
    query_sample: Any = None                            # representative queries (array)
    index_builder: Optional[Callable[..., Any]] = None  # (params)->faiss index, eval backend (legacy)
    search_space: dict = field(default_factory=dict)    # swept values: embedding_models, nlist, nprobe
    # embedding_model -> (base_vecs float32, query_vecs float32), or None if
    # unavailable. Drives the eager build-on-miss in
    # ``cost_model.workload_profile.ensure_workload_profiles``.
    vector_provider: Optional[Callable[[str], Any]] = None


class CostModelModule(ABC):
    """Abstract base for a cost model module.

    Each module (faiss_ivf_sim, llm_sim, etc.) implements ``estimate()``.
    The controller calls it with the resolved pipeline config dict and
    the global hardware config.
    """

    def init(self, ctx: "CmInitContext") -> None:
        """One-time setup at controller startup. Default: no-op.

        Components that depend on a cached data-side artifact (built once via the
        eval backend, then reused across evaluations and across hardware) prepare
        it here — checking for a profile, and only building it on a miss. Most
        components need nothing and inherit this no-op.
        """
        return None

    @abstractmethod
    def estimate(
        self, pipeline_config: dict, hardware: SystemConfig
    ) -> CostModelResult:
        """Estimate cost for a single resolved pipeline configuration."""
        ...

    def generate_sweep_df(
        self,
        pipeline_config: dict,
        hardware: SystemConfig,
        stage: str,
        max_batch_size: int,
    ) -> pd.DataFrame:
        """Generate a RAGO-compatible sweep DataFrame for this stage.

        Implementations should sweep ``get_power_of_two_list(max_batch_size)``
        so the resulting DataFrame's ``batch_size`` column matches what RAGO
        will ask for downstream (no interpolation needed).

        Default: NOT implemented — raises NotImplementedError. Cost models
        that need multi-device × parallelism × Pareto filtering (e.g.
        LLM/reranker) override this. Retrieval cost models (FAISS) don't
        use this path — assembly.py handles their sweep by calling
        ``estimate()`` directly per batch_size.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_sweep_df; "
            f"callers should use estimate() for per-batch costs."
        )

    def estimate_batch(
        self, configs: list[dict], hardware: SystemConfig
    ) -> list[CostModelResult]:
        """Estimate cost for multiple configs. Default loops over estimate()."""
        return [self.estimate(c, hardware) for c in configs]

    # ---------------------------------------------------------------------
    # Calibration interface (optional, opt-in per subclass)
    # ---------------------------------------------------------------------

    @classmethod
    def family(cls) -> str:
        """High-level grouping used by UI (sidebar sections, mapping page
        columns). Known values: ``"llm_sim"``, ``"retrieval"``. Extend with
        new categories as cost model types are added.
        """
        return "other"

    @classmethod
    def capabilities(cls) -> list[str]:
        """Capability flags advertised to the API layer / UI.

        Supported values:
          - ``calibration_profiles`` — persists per-(module,model,hw,bits) JSON
            records readable by the frontend Profiles tab.
          - ``calibration_runs`` — supports ``benchmark()`` / ``calibrate()``
            for on-demand real-hardware measurement.

        Default: no capabilities. Subclasses override to advertise support.
        """
        return []

    @classmethod
    def profiles_subdir(cls) -> str | None:
        """Filesystem subdir (relative to ``calibration_profiles/``) where
        this cost model's profiles live. Used by the API to list + serve
        JSON records. ``None`` means no on-disk profile store.
        """
        return None

    def benchmark(self, *args, **kwargs):
        """Measure real-hardware latency vs predicted; return a diagnostic
        report. Subclasses with ``calibration_runs`` must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support benchmarking."
        )

    def calibrate(self, *args, **kwargs):
        """Measure + fit (ceff, meff) and persist a JSON profile.
        Subclasses with ``calibration_runs`` must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support calibration."
        )