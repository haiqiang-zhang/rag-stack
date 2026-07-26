"""IVF hardware calibrator — measure-once / fit-on-CM, grouped DE per scan kernel.

The proven structure (lifted from benchmarks/rag_cm_accuracy/ivf/calibrate_hardware.py):
IVF has ~20 efficiency coefficients but they are **physically separable into three
scan kernels** (Flat / PQ-std / PQ-fastscan), so we fit them GROUPED (3-6 params
each, well-conditioned) with ``differential_evolution`` — NOT a single 20-dim
global fit. Physical constants (DRAM random-access latency, bandwidth, the shared
coarse-quantizer efficiencies) are PINNED at measured/base values; degenerate
datasets are excluded from the fit but reported (transfer test). DE is used because
the CM's ``max(compute, memory)`` bottleneck-switching makes the coefficient space
non-smooth.

``measure`` builds the 6 cm_benchmark indices once and freezes per-config latency
(+ deterministic cell-imbalance) to calibration_data/ivf/<system>.parquet.
``fit`` re-evaluates the CM over the frozen rows — no FAISS.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from rag_stack.calibration_paths import system_profile_path

from ..calibration_base import Calibrator
from .faiss_benchmark import FaissBenchmarkRunner
from .ivfpq_config import IVFPQConfig
from .cost_model import estimate_search_cost
from .coefficients import (
    IVF_PROFILE_KIND,
    IVF_RUNTIME_COEFFICIENT_FIELDS,
    CalibratedCoefficients,
    save_coefficients,
    build_hardware_from_coefficients,
)
from .imbalance import imbalance_key, save_imbalance_profile

# (dataset registry key, embedding label, d) — the 4 classic ANN-benchmark sets
# spanning the d range (96/100/128/960). These are HARDWARE anchors; the cost
# model's per-dataset behavior is carried by the imbalance/d, not the embedding,
# so a small fixed set suffices to calibrate the hardware roofline + thread/batch.
DATASETS = [
    ("sift1m",     "raw",          128),
    ("gist1m",     "raw",          960),
    ("deep1m",     "raw",           96),
    ("glove",      "glove",        100),
]
BASE_N, NQ, K = 200_000, 500, 10
NPROBE = [1, 4, 16, 64, 128]
# No degenerate corpora in the 4-set anchor list → nothing excluded by default.
FIT_EXCLUDE_DEFAULT: set = set()

# Every non-raw-hardware runtime field is fitted from this frozen component grid.
# Shared operator parameters are separated from per-index scan parameters so the
# optimizer never performs an opaque high-dimensional global fit.
SHARED_SINGLETHREAD_GROUP = {
    # Shared coarse-quantizer/LUT/refinement SIMD utilization. The four d
    # anchors and three index kernels make it separable from scan work.
    "simd_efficiency":                  (0.05, 1.0),
}

# Per-index single-thread groups.
GROUPS = {
    "flat": {
        "flat_scan_efficiency":        (0.05, 1.0),
        "flat_fixed_overhead_us":      (0.0, 50.0),
        "flat_per_probe_overhead_us":  (0.0, 5.0),
    },
    "pq_fastscan": {
        "pq_fastscan_efficiency":            (0.1, 2.0),
        "pq_fastscan_per_probe_overhead_us": (0.0, 5.0),
        # d-aware per-call PQ LUT-setup floor (µs/dim) — SHARED by both PQ kernels
        # (fit here on the cleanest fastscan floor; pq_std inherits via the
        # accumulated base). ~0.02 µs/dim reproduces the 5µs@d128 → 24µs@d1000 floor.
        "pq_setup_per_dim_us":               (0.0, 0.05),
    },
    "pq_std": {
        "pq_standard_efficiency":            (0.01, 0.6),
        "pq_standard_per_probe_overhead_us": (0.0, 5.0),
        "pq_rap_standard_c1":                (0.0, 50.0),
        "pq_rap_standard_c2":                (0.0, 120.0),
        "pq_rap_standard_c3":                (0.0, 400.0),
    },
}

# AFFORDABLE batch + multithread grid. Batch-efficiency + thread-scaling are
# HARDWARE-level (shared across datasets), so only a few d-anchor datasets are
# measured (low/mid/high d) at anchor nprobe — NOT the full grid. Keeps the
# `calibrate measure ivf` campaign cheap (~minutes), all inside rag_stack.
MT_ANCHORS = {"sift1m", "gist1m"}                # d = 128 (low) / 960 (high) — span d for batch/MT
MT_FLAT_MAX_D = 256                              # skip FLAT batch/MT above this d (gist960 flat scan
#                                                  is pathologically slow; NUMA only needs the low-d
#                                                  flat anchor — it's a hardware, d-independent effect).
THREAD_SWEEP = [1, 4, 8, 16, 32]                 # multithread regime (@ MT_BATCH)
BATCH_SWEEP = [4, 16, 64]                         # batch regime (@ thread 1); 1 = singlethread (256 too slow)
FASTSCAN_BATCH_SWEEP = [4, 16, 64, 256]           # fastscan is cheap → reach higher batch to anchor the amortization curve
MT_NPROBE = [16, 128]                            # anchor nprobe for batch/MT sweeps
MT_BATCH = 64

# Regime fit GROUPS (fit AFTER the per-kernel single-thread groups). batch eff is
# the only thread-independent batch knob; thread scaling uses the two SHARED,
# thread-aware params (NUMA numa_local_bytes is intentionally NOT fit here — its
# scattered-path branch is a ceiling-replacement, not yet thread-aware).
REGIME_GROUPS = {
    "batch": {                                   # rows: regime=='batch' & group=='pq_fastscan'
        "pq_fastscan_batch_efficiency": (0.1, 2.0),
        # QBS compute-amortization floor: 1.0 = no compute amortization (loads
        # already amortized on the memory side); the old 0.65 over-predicted batch.
        "pq_fastscan_qbs_floor":        (0.5, 1.0),
        # Per-probe overhead batch-amortization exponent. sqrt-B (0.5) over-amortizes
        # the per-query overhead (CM batch=64 too fast ~1.6×); fits to ~0.1-0.2.
        "pq_fastscan_overhead_alpha":   (0.0, 0.5),
    },
}

# Batch + multi-thread rows jointly identify the shared BLAS path and the
# scattered/NUMA bandwidth curve. Batch rows pin BLAS without NUMA activation;
# the thread sweep then identifies the bandwidth ceiling and socket crossover.
SHARED_PARALLEL_GROUP = {
    "blas_efficiency":              (0.05, 1.0),
    "scattered_bw_ceiling":         (0.05, 1.0),
    "numa_local_bytes":             (1.0e6, 200.0e6),
}

_FASTSCAN_BW_GROUP = {"pq_fastscan_scan_bw_ceiling": (0.1, 1.0)}


def fitted_coefficient_fields() -> frozenset[str]:
    """Return the exact code-level fit coverage for the runtime profile."""

    groups = [
        SHARED_SINGLETHREAD_GROUP,
        *GROUPS.values(),
        _FASTSCAN_BW_GROUP,
        *REGIME_GROUPS.values(),
        SHARED_PARALLEL_GROUP,
    ]
    names = [name for group in groups for name in group]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"IVF coefficients appear in multiple fit groups: {duplicates}")
    return frozenset(names)


_FITTED_COEFFICIENT_FIELDS = fitted_coefficient_fields()
if _FITTED_COEFFICIENT_FIELDS != frozenset(IVF_RUNTIME_COEFFICIENT_FIELDS):
    raise RuntimeError(
        "IVF runtime/fit coefficient coverage mismatch: "
        f"unfitted={sorted(set(IVF_RUNTIME_COEFFICIENT_FIELDS) - _FITTED_COEFFICIENT_FIELDS)}, "
        f"non_runtime={sorted(_FITTED_COEFFICIENT_FIELDS - set(IVF_RUNTIME_COEFFICIENT_FIELDS))}"
    )


FROZEN_COLUMNS = (
    "dataset", "embedding", "d", "N", "nlist", "group", "index_type",
    "M", "nbits", "by_residual", "nprobe", "imbalance", "regime",
    "num_threads", "batch_size", "measured_ms",
)
def pq_m(d: int) -> int:
    """Largest even M ≤ 64 dividing d with dsub = d/M ≥ 4 (a realistic PQ config)."""
    best = 2
    for M in range(2, min(64, d) + 1, 2):
        if d % M == 0 and d // M >= 4:
            best = M
    return best


def nlist_for(n: int) -> int:
    return int(min(1024, max(64, 2 ** round(math.log2(max(2, math.sqrt(n)))))))


def index_configs(d: int, nlist: int):
    """(factory, group, M, nbits, by_residual) for Flat / PQ-std / PQ-fastscan."""
    m = pq_m(d)
    return [
        # Each kernel uses its FAISS DEFAULT by_residual so the CM's parameter set
        # matches what FAISS measured does: IndexIVFPQ → True, IndexIVFPQFastScan →
        # False ("faster"), IVFFlat → False. The CM supports by_residual=True for
        # FastScan too, but its DEFAULT (the only behavior we need) is False.
        (f"IVF{nlist},Flat",       "flat",        0, 0, False),
        (f"IVF{nlist},PQ{m}",      "pq_std",      m, 8, True),
        (f"IVF{nlist},PQ{m}x4fs",  "pq_fastscan", m, 4, False),
    ]


class IVFHardwareCalibrator(Calibrator):
    component = "ivf"
    measured_col = "measured_ms"
    cm_col = "cm_ms"
    group_cols = ("dataset", "group")

    def __init__(self, system="epyc_9124", *, maxiter=40, fit_exclude=None):
        super().__init__(system)
        self.maxiter = maxiter
        self.fit_exclude = FIT_EXCLUDE_DEFAULT if fit_exclude is None else set(fit_exclude)

    # ── measure (real FAISS, once) ──────────────────────────────────────────
    def _measure(self) -> pd.DataFrame:
        rows = []
        for ds_key, emb, d in DATASETS:
            r = FaissBenchmarkRunner(num_threads=1, dataset=ds_key)
            r.xb = np.ascontiguousarray(r.xb[:BASE_N]); r.nb = r.xb.shape[0]
            r.xt = np.ascontiguousarray(r.xt[:BASE_N])
            r.xq = np.ascontiguousarray(r.xq[:NQ]); r.nq = r.xq.shape[0]
            nlist = nlist_for(r.nb)
            nprobe = [p for p in NPROBE if p < nlist]  # exclude nprobe==nlist (full scan)
            np_anchor = [p for p in MT_NPROBE if p < nlist]
            print(f"\n=== {ds_key} ({emb}) N={r.nb} d={d} nlist={nlist} "
                  f"nprobe={nprobe} ===", flush=True)
            for factory, group, m, nbits, by_res in index_configs(d, nlist):
                imb = r.cell_imbalance_by_nprobe(factory, nprobe)
                save_imbalance_profile(imbalance_key(ds_key, emb), {
                    "d": d, "N": r.nb,
                    "nlist": {str(nlist): {str(p): float(imb.get(p, 1.0)) for p in nprobe}},
                })

                def emit(res, regime, nthreads, bs):
                    for br in res:
                        rows.append(dict(
                            dataset=ds_key, embedding=emb, d=d, N=r.nb, nlist=nlist,
                            group=group, index_type=("flat" if group == "flat" else "pq"),
                            M=m, nbits=nbits, by_residual=by_res, nprobe=br.nprobe,
                            imbalance=float(imb.get(br.nprobe, 1.0)), regime=regime,
                            num_threads=nthreads, batch_size=bs,
                            measured_ms=float(br.latency_ms)))

                # singlethread (the original calibration grid) — batch=1, thread=1
                r.num_threads = 1
                emit(r.benchmark_index(factory, nprobe, k=K, batch_size=1, parallel_mode=0),
                     "singlethread", 1, 1)
                # AFFORDABLE batch + multithread sweeps — only on the d-anchor datasets,
                # and skip FLAT above MT_FLAT_MAX_D (gist960 flat scan is too slow; NUMA
                # is a hardware effect captured by the low-d flat anchor).
                if (ds_key in MT_ANCHORS and np_anchor
                        and not (group == "flat" and d > MT_FLAT_MAX_D)):
                    # FastScan is cheap → sweep to 256 so the fit sees the high-batch
                    # over-amortization drift (the benchmark goes to 2048); flat/std
                    # stay ≤64 (gist960 high-batch scan is too slow to measure).
                    sweep = FASTSCAN_BATCH_SWEEP if group == "pq_fastscan" else BATCH_SWEEP
                    for bs in sweep:
                        if bs <= r.nq:
                            emit(r.benchmark_index(factory, np_anchor, k=K, batch_size=bs,
                                                   parallel_mode=0), "batch", 1, bs)
                    if MT_BATCH <= r.nq:
                        for th in THREAD_SWEEP:
                            r.num_threads = th
                            emit(r.benchmark_index(factory, np_anchor, k=K, batch_size=MT_BATCH,
                                                   parallel_mode=0), "multithread", th, MT_BATCH)
                    r.num_threads = 1
            del r
        df = pd.DataFrame(rows)
        return df[df.measured_ms > 0].reset_index(drop=True)

    # ── fit (CM-only) ───────────────────────────────────────────────────────
    def _base_coef(self):
        """Build a deterministic, profile-independent optimization seed.

        Every runtime coefficient is initialized to the midpoint of its own fit
        bound and is overwritten by a fit group. No preset, final profile,
        ``.base`` file, environment variable, or RAG result participates.
        """

        all_groups = [
            SHARED_SINGLETHREAD_GROUP,
            *GROUPS.values(),
            _FASTSCAN_BW_GROUP,
            *REGIME_GROUPS.values(),
            SHARED_PARALLEL_GROUP,
        ]
        values = {
            name: (float(lower) + float(upper)) / 2.0
            for group in all_groups
            for name, (lower, upper) in group.items()
        }
        values["numa_local_bytes"] = int(round(values["numa_local_bytes"]))
        return CalibratedCoefficients(
            **values,
            profile_kind=IVF_PROFILE_KIND,
            hardware_preset=self.system,
            mape_latency=0.0,
            max_ape_latency=0.0,
            num_trials=1,
            index_configs=[],
            batch_sizes=[],
            fitted_fields=list(IVF_RUNTIME_COEFFICIENT_FIELDS),
        )

    def _predict_ms(self, p, hw) -> float:
        # num_threads / batch_size default to 1 (back-compat with single-thread frozen data).
        bs = int(getattr(p, "batch_size", 1) or 1)
        cfg = IVFPQConfig(
            index_type=p.index_type, d=int(p.d), N=int(p.N), nlist=int(p.nlist),
            nprobe=int(p.nprobe), M=int(p.M), nbits=int(p.nbits), k=K, batch_size=bs,
            by_residual=bool(p.by_residual), parallel_mode=0,
            cell_imbalance=float(p.imbalance),
        )
        return estimate_search_cost(cfg, hw, warm_cache=True).latency_ms

    def _apes(self, coef, pts) -> np.ndarray:
        predicted = self._predictions(coef, pts)
        measured = pts["measured_ms"].to_numpy(dtype=float)
        return np.abs(predicted - measured) / measured * 100.0

    def _predictions(self, coef, pts) -> np.ndarray:
        # Build hardware ONCE per distinct num_threads (active_threads drives the
        # thread-scaling model); rows without the column default to 1 thread.
        hw_cache = {}
        def hw_for(nt):
            if nt not in hw_cache:
                hw_cache[nt] = build_hardware_from_coefficients(
                    coef, active_threads=int(nt), system_config=None)
            return hw_cache[nt]
        return np.array([
            self._predict_ms(
                p, hw_for(int(getattr(p, "num_threads", 1) or 1)),
            )
            for p in pts.itertuples()
        ])

    def _require_identifiable(self, base_coef, bounds: dict, pts) -> None:
        """Reject a nominal fit knob that the selected frozen rows cannot see."""

        if pts.empty:
            raise ValueError(
                f"frozen IVF grid has no rows for coefficient group {sorted(bounds)}"
            )
        for name, (lower, upper) in bounds.items():
            # Check the current coordinate plus both group-bound corners. This
            # catches conditional parameters such as remote NUMA bandwidth,
            # which is observable only when the fitted crossover is also low.
            contexts = [base_coef]
            for corner in (0, 1):
                values = {
                    other: limits[corner]
                    for other, limits in bounds.items()
                    if other != name
                }
                contexts.append(dataclasses.replace(base_coef, **values))
            max_relative_change = 0.0
            for context in contexts:
                low_coef = dataclasses.replace(context, **{name: lower})
                high_coef = dataclasses.replace(context, **{name: upper})
                a = self._predictions(low_coef, pts)
                b = self._predictions(high_coef, pts)
                scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)
                max_relative_change = max(
                    max_relative_change,
                    float(np.max(np.abs(a - b) / scale)),
                )
            if max_relative_change <= 1e-8:
                raise ValueError(
                    f"IVF coefficient {name!r} is not identifiable from its "
                    "frozen standalone rows; remove the term or expand the "
                    "standalone measurement schema"
                )

    def _fit_group(self, base_coef, bounds: dict, pts):
        """Fit the coefficients named in ``bounds`` against the pre-filtered ``pts``."""
        names = list(bounds)
        self._require_identifiable(base_coef, bounds, pts)

        def mape(x):
            coef = dataclasses.replace(base_coef, **dict(zip(names, x)))
            return float(np.mean(self._apes(coef, pts)))

        res = differential_evolution(mape, [bounds[n] for n in names], seed=0,
                                     maxiter=self.maxiter, tol=1e-3, polish=True, disp=False)
        return dict(zip(names, res.x)), float(res.fun)

    def _regime(self, df, regime):
        """Rows of a regime (back-compat: no `regime` column → all rows are singlethread)."""
        d = df[~df.dataset.isin(self.fit_exclude)]
        return d[d.regime == regime]

    def _validate_frozen(self, frozen: pd.DataFrame) -> None:
        actual = tuple(frozen.columns)
        if actual != FROZEN_COLUMNS:
            raise ValueError(
                "IVF frozen schema mismatch; exact ordered columns required: "
                f"expected={list(FROZEN_COLUMNS)}, actual={list(actual)}"
            )
        if frozen.empty:
            raise ValueError("IVF frozen component grid is empty")
        if set(frozen["group"]) != set(GROUPS):
            raise ValueError(
                "IVF frozen grid must contain flat, pq_fastscan, and pq_std rows"
            )
        if set(frozen["regime"]) != {"singlethread", "batch", "multithread"}:
            raise ValueError(
                "IVF frozen grid must contain singlethread, batch, and multithread regimes"
            )
        numeric_positive = (
            "d", "N", "nlist", "nprobe", "imbalance",
            "num_threads", "batch_size", "measured_ms",
        )
        for column in numeric_positive:
            values = pd.to_numeric(frozen[column], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(
                    f"IVF frozen column {column!r} must be finite and positive"
                )
        for column in ("M", "nbits"):
            values = pd.to_numeric(frozen[column], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(
                    f"IVF frozen column {column!r} must be finite and non-negative"
                )
        if not np.all(frozen.loc[frozen.regime == "singlethread", "num_threads"] == 1):
            raise ValueError("IVF singlethread evidence must use num_threads=1")

    def _fit(self, frozen: pd.DataFrame):
        self._validate_frozen(frozen)
        base = self._base_coef()
        fitted, cur = {}, base
        # Coordinate-descent passes keep the fit grouped/inspectable while
        # eliminating dependency on arbitrary initial values in another group.
        # Every group is deterministically re-fit on the same frozen rows.
        st = self._regime(frozen, "singlethread")
        for fit_pass in range(2):
            best, gm = self._fit_group(cur, SHARED_SINGLETHREAD_GROUP, st)
            fitted.update(best); cur = dataclasses.replace(cur, **best)
            print(f"  [shared-st p{fit_pass + 1}] MAPE={gm:5.2f}%  " +
                  "  ".join(f"{k}={v:.4f}" for k, v in best.items()), flush=True)

            for group, bounds in GROUPS.items():
                best, gm = self._fit_group(cur, bounds, st[st.group == group])
                fitted.update(best); cur = dataclasses.replace(cur, **best)
                print(f"  [{group:11s} p{fit_pass + 1}] MAPE={gm:5.2f}%  " +
                      "  ".join(f"{k}={v:.4f}" for k, v in best.items()), flush=True)

            st_fs = st[st.group == "pq_fastscan"]
            best, gm = self._fit_group(cur, _FASTSCAN_BW_GROUP, st_fs)
            fitted.update(best); cur = dataclasses.replace(cur, **best)
            print(f"  [fastscan-bw p{fit_pass + 1}] MAPE={gm:5.2f}%  " +
                  "  ".join(f"{k}={v:.4f}" for k, v in best.items()), flush=True)

            for rg, bounds in REGIME_GROUPS.items():
                pts = self._regime(frozen, rg)
                if rg == "batch":
                    pts = pts[pts.group == "pq_fastscan"]
                best, gm = self._fit_group(cur, bounds, pts)
                fitted.update(best); cur = dataclasses.replace(cur, **best)
                print(f"  [{rg:11s} p{fit_pass + 1}] MAPE={gm:5.2f}%  " +
                      "  ".join(f"{k}={v:.4f}" for k, v in best.items()), flush=True)

            parallel = pd.concat([
                self._regime(frozen, "batch"),
                self._regime(frozen, "multithread"),
            ], ignore_index=True)
            best, gm = self._fit_group(cur, SHARED_PARALLEL_GROUP, parallel)
            best["numa_local_bytes"] = int(round(best["numa_local_bytes"]))
            fitted.update(best); cur = dataclasses.replace(cur, **best)
            print(f"  [shared-par p{fit_pass + 1}] MAPE={gm:5.2f}%  " +
                  "  ".join(f"{k}={v:.4f}" for k, v in best.items()), flush=True)

        final = dataclasses.replace(
            cur,
            hardware_preset=self.system,
            num_trials=int(len(frozen[~frozen.dataset.isin(self.fit_exclude)])),
            index_configs=["IVF,Flat", "IVF,PQ", "IVF,PQx4fs"],
            batch_sizes=sorted({1, *BATCH_SWEEP, MT_BATCH}),
            fitted_fields=list(IVF_RUNTIME_COEFFICIENT_FIELDS),
        )
        fit_df = frozen[~frozen.dataset.isin(self.fit_exclude)]
        ape = self._apes(final, fit_df)            # thread/batch-aware
        final = dataclasses.replace(final, mape_latency=float(np.mean(ape)),
                                    max_ape_latency=float(np.max(ape)))

        scored = frozen.copy()
        hw_cache = {}
        def hw_for(nt):
            if nt not in hw_cache:
                hw_cache[nt] = build_hardware_from_coefficients(final, active_threads=int(nt))
            return hw_cache[nt]
        scored["cm_ms"] = [self._predict_ms(p, hw_for(int(getattr(p, "num_threads", 1) or 1)))
                           for p in scored.itertuples()]
        return final, scored

    def _save(self, profile) -> str:
        prof = str(system_profile_path("faiss_ivf", self.system))
        save_coefficients(profile, prof)
        return prof
