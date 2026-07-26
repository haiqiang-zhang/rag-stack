"""HNSW hardware calibrator — measure-once / fit-on-CM, 4-coeff roofline DE.

HNSW differs from IVF: its workload (ndis/nhops per query) is a graph-traversal
result with no analytic form, so the split profile is (exact CPU system profile,
per-system) + (workload predictor = GBR, per-dataset). This calibrator owns only
the system half. It fits four empirical graph-search coefficients and also saves
the five effective hardware-model values that were actually applied during the
fit; raw CPU facts remain in the preset/inline CPU spec. ``measure`` runs
``collect_hnsw_benchmarks`` once over the
6 cm_benchmark datasets (d separates the d-dependent memory term from the
d-independent compute term) and freezes (latency, ndis, nhops) per config to
calibration_data/hnsw/<system>.parquet. ``fit`` does the direct ``differential_
evolution`` over the 4 coeffs using the FROZEN measured ndis/nhops — CM-only, no
FAISS. Evidence is standalone component data, never RAG end-to-end ground truth.
(The workload GBR remains the separate ``calibrate hnsw-workload`` path.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from rag_stack.calibration_paths import system_profile_path

from ..calibration_base import Calibrator

_D = "datasets/cm_benchmark"
DATASETS = [   # (label, hdf5) — d = 100/384/1024 + 128/960/96 separates the terms
    ("glove",      f"{_D}/glove-100-angular.hdf5"),
    ("cifar",      f"{_D}/cifar10-dinov2small-384.hdf5"),
    ("dragonball", f"{_D}/dragonball-bgem3-1024.hdf5"),
    ("sift",       f"{_D}/sift1M.hdf5"),
    ("gist",       f"{_D}/gist1M.hdf5"),
    ("deep",       f"{_D}/deep1m.hdf5"),
]
M_VALUES = [8, 16, 32, 48]
EF_SEARCH = [16, 32, 64, 128, 256]
EF_CONSTRUCTION, BASE_N, NQ, K = 200, 100_000, 300, 10
# WIDE bounds (the deprecated Ax calibrator pinned 3/4 coeffs to Intel brackets).
BOUNDS = {
    "hops_per_upper_layer": (0.5, 8.0),
    "cache_random_access_latency_multiplier": (0.5, 30.0),
    "cycles_per_visited_probe": (1.0, 800.0),
    "cycles_per_heap_op": (1.0, 800.0),
}


class HNSWHardwareCalibrator(Calibrator):
    component = "hnsw"
    measured_col = "latency_ms"
    cm_col = "cm_ms"
    group_cols = ("dataset",)

    def __init__(
        self,
        system="epyc_9124",
        *,
        maxiter=50,
        base_n=BASE_N,
        n_query=NQ,
        hardware_model_parameters=None,
    ):
        super().__init__(system)
        self.maxiter = maxiter
        self.base_n = base_n
        self.n_query = n_query
        if hardware_model_parameters is None:
            self._explicit_hardware_model_parameters = None
        else:
            from .coefficients import HNSWHardwareModelParameters

            if isinstance(
                hardware_model_parameters, HNSWHardwareModelParameters,
            ):
                self._explicit_hardware_model_parameters = (
                    hardware_model_parameters
                )
            elif isinstance(hardware_model_parameters, dict):
                self._explicit_hardware_model_parameters = (
                    HNSWHardwareModelParameters(**hardware_model_parameters)
                )
            else:
                raise TypeError(
                    "hardware_model_parameters must be a mapping or "
                    "HNSWHardwareModelParameters"
                )

    # ── measure (real FAISS HNSW, once) — latency + ndis + nhops ────────────
    def _measure(self) -> pd.DataFrame:
        import h5py
        from .calibrator import collect_hnsw_benchmarks
        datasets, queries = {}, {}
        for name, path in DATASETS:
            with h5py.File(path, "r") as f:
                base = np.ascontiguousarray(f["train"][:self.base_n].astype(np.float32))
                q = np.ascontiguousarray(f["test"][:self.n_query].astype(np.float32))
            datasets[name], queries[name] = base, q
            print(f"{name}: base {base.shape} query {q.shape}", flush=True)
        pts = collect_hnsw_benchmarks(
            datasets, queries, M_values=M_VALUES, ef_search_values=EF_SEARCH,
            ef_construction=EF_CONSTRUCTION, batch_sizes=[1], k=K,
            n_repeat=20, warm_cache=True,
        )
        rows = [dict(dataset=p.dataset, N=p.N, d=p.d, M=p.M, ef_search=p.ef_search,
                     ef_construction=p.ef_construction, batch_size=p.batch_size, k=p.k,
                     ndis_per_query=p.ndis_per_query, nhops_per_query=p.nhops_per_query,
                     latency_ms=p.latency_ms)
                for p in pts if p.latency_ms > 0]
        return pd.DataFrame(rows).reset_index(drop=True)

    # ── fit (CM-only) ───────────────────────────────────────────────────────
    def _predict_ms(self, p, coef, hardware) -> float:
        from .cost_model import HNSWCostModel
        from .hnsw_config import HNSWConfig
        from .workload_predictor import WorkloadProfile
        cfg = HNSWConfig(N=int(p.N), d=int(p.d), M=int(p.M), ef_search=int(p.ef_search),
                         k=int(p.k), batch_size=int(p.batch_size),
                         ef_construction=int(p.ef_construction))
        wl = WorkloadProfile(ndis_per_query=float(p.ndis_per_query),
                             nhops_per_query=float(p.nhops_per_query))
        return HNSWCostModel(
            cfg, hardware, wl, coefficients=coef,
        ).estimate().latency_ms

    def _resolve_hardware_model_parameters(self):
        """Resolve explicit CPU-specific values without a custom-CPU default.

        Built-in HNSW presets carry CPU-specific calibration seeds. A custom
        CPU must either have an existing latest-schema profile (for repeatable
        refits) or receive all five values explicitly from the calibration CLI.
        The generic :class:`hardware.SystemConfig` defaults are never consulted.
        """

        from .coefficients import (
            HNSWHardwareModelParameters,
            HNSWProfileError,
            load_profile_for_preset,
            profile_path_for_preset,
        )
        from .system_presets import SYSTEM_PRESETS
        from rag_stack.cost_model.cpu_specs import CPU_SPECS

        if self._explicit_hardware_model_parameters is not None:
            return self._explicit_hardware_model_parameters
        if Path(profile_path_for_preset(self.system)).is_file():
            # An existing profile is authoritative: malformed/legacy content
            # must fail closed rather than being hidden by a preset seed.
            return load_profile_for_preset(
                self.system
            ).hardware_model_parameters
        if self.system in CPU_SPECS:
            if self.system not in SYSTEM_PRESETS:
                raise ValueError(
                    f"built-in CPU {self.system!r} has no HNSW preset"
                )
            return HNSWHardwareModelParameters.from_hardware(
                SYSTEM_PRESETS[self.system](active_threads=1)
            )
        raise HNSWProfileError(
            f"custom CPU {self.system!r} has no exact HNSW hardware-model "
            "parameters. Supply all five --hnsw-* fit options; generic "
            "algorithm defaults are forbidden."
        )

    def _hw(self, parameters):
        from .coefficients import HNSWHardwareModelParameters, HNSWProfileError
        from .system_presets import SYSTEM_PRESETS
        if self.system not in SYSTEM_PRESETS:
            from rag_stack.cost_model.faiss_system_registry import require_configured_faiss_preset
            require_configured_faiss_preset("hnsw", self.system)
        hardware = SYSTEM_PRESETS[self.system](
            active_threads=1,
            **parameters.to_hardware_kwargs(),
        )
        actual = HNSWHardwareModelParameters.from_hardware(hardware)
        if actual != parameters:
            raise HNSWProfileError(
                f"HNSW calibrator hardware builder changed requested values: "
                f"requested={parameters!r}, actual={actual!r}"
            )
        return hardware

    def _fit(self, frozen: pd.DataFrame):
        from .cost_model import HNSWCoefficients
        from .coefficients import (
            HNSWFitMetadata,
            HNSWSystemProfile,
        )

        names = list(BOUNDS)
        hardware_parameters = self._resolve_hardware_model_parameters()
        hardware = self._hw(hardware_parameters)

        def apes_for(x):
            coef = HNSWCoefficients(**dict(zip(names, x)))
            return np.array([
                abs(self._predict_ms(p, coef, hardware) - p.latency_ms)
                / p.latency_ms * 100
                for p in frozen.itertuples()
            ])

        res = differential_evolution(lambda x: float(np.mean(apes_for(x))),
                                     [BOUNDS[n] for n in names], seed=0,
                                     maxiter=self.maxiter, tol=1e-3, polish=True, disp=False)
        best = {name: float(value) for name, value in zip(names, res.x)}
        ape = apes_for(res.x)
        for n, v in best.items():
            edge = " (AT BOUND)" if abs(v - BOUNDS[n][0]) < 1e-3 or abs(v - BOUNDS[n][1]) < 1e-3 else ""
            print(f"  {n:42s} = {v:.3f}{edge}", flush=True)
        coefficients = HNSWCoefficients(**best)
        profile = HNSWSystemProfile(
            hardware_key=self.system,
            hardware_model_parameters=hardware_parameters,
            empirical_coefficients=coefficients,
            fit_metadata=HNSWFitMetadata(
                mape_latency_pct=float(np.mean(ape)),
                max_ape_latency_pct=float(np.max(ape)),
                num_trials=int(len(frozen)),
                dataset_configs=tuple(name for name, _ in DATASETS),
            ),
        )
        scored = frozen.copy()
        scored["cm_ms"] = [
            self._predict_ms(p, coefficients, hardware)
            for p in scored.itertuples()
        ]
        self._coef = coefficients
        return profile, scored

    def _save(self, profile) -> str:
        from .coefficients import save_profile
        return save_profile(
            profile, str(system_profile_path("faiss_hnsw", self.system))
        )
