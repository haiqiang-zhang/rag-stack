"""Foundation for the unified RAG-CM calibration framework.

MEASURE-ONCE / FIT-ON-CM (the central architecture):
  - ``measure()`` is the ONLY step that touches real hardware. It runs the
    measurement grid (warm cache, single-thread, isolated) and FREEZES the ground
    truth to ``calibration_data/<component>/<system>.parquet``.
  - ``fit()`` runs PURELY against the frozen measurements by re-evaluating the
    cost model — fast (CM eval is ms), repeatable, no hardware. The agentic
    diagnosis loop lives entirely here: change the CM, re-fit, re-guard — never
    re-measure (measurement noise can't leak into the fit).

Re-``measure`` only when the hardware or a dataset/corpus changes.

Each component subclasses :class:`Calibrator` and implements ``_measure`` (real
hardware → rows to freeze), ``_fit`` (frozen → profile + scored df), ``_save``.
The base provides frozen parquet I/O and the guard/transfer accuracy report
(Spearman = the ranking metric the optimizer needs; MAPE = magnitude).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rag_stack.calibration_paths import frozen_measurement_path


def frozen_path(component: str, system: str) -> str:
    return str(frozen_measurement_path(component, system))


def save_frozen(df: pd.DataFrame, component: str, system: str) -> str:
    p = frozen_path(component, system)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def load_frozen(component: str, system: str) -> Optional[pd.DataFrame]:
    p = frozen_path(component, system)
    return pd.read_parquet(p) if os.path.exists(p) else None


def accuracy(measured, predicted) -> tuple[float, float, float]:
    """(Spearman rank-corr, MAPE %, median ratio cm/measured) over valid rows."""
    m = np.asarray(measured, dtype=float)
    c = np.asarray(predicted, dtype=float)
    ok = (m > 0) & np.isfinite(m) & np.isfinite(c)
    m, c = m[ok], c[ok]
    if len(m) == 0:
        return float("nan"), float("nan"), float("nan")
    sp = float(spearmanr(m, c).correlation) if len(m) > 2 else float("nan")
    mape = float(np.mean(np.abs(c / m - 1.0)) * 100.0)
    ratio_med = float(np.median(c / m))
    return sp, mape, ratio_med


class Calibrator:
    """measure-once (frozen parquet) + fit-on-CM, with a guard/transfer report.

    Subclass contract:
      - ``component``           : short name → calibration_data/<component>/...
      - ``default_system``      : default hardware key
      - ``measured_col``/``cm_col``/``group_cols`` : reporting columns
      - ``_measure() -> df``    : real-hardware rows to freeze
      - ``_fit(frozen) -> (profile, scored_df)`` : CM-only fit; ``scored_df`` has
            ``cm_col`` filled so the base can report. May set ``self.fit_exclude``.
      - ``_save(profile) -> path``
    """

    component: str = ""
    default_system: str = "epyc_9124"
    measured_col: str = "measured_ms"
    cm_col: str = "cm_ms"
    group_cols: tuple[str, ...] = ("dataset", "group")

    def __init__(self, system: Optional[str] = None):
        self.system = system or self.default_system
        self.fit_exclude: set[str] = set()

    # ── measure (once, real hardware) → frozen parquet ──────────────────────
    def measure(self, force: bool = False) -> pd.DataFrame:
        if not force:
            cached = load_frozen(self.component, self.system)
            if cached is not None:
                print(f"[{self.component}] reusing frozen {len(cached)} rows "
                      f"({frozen_path(self.component, self.system)})", flush=True)
                return cached
        import time
        t0 = time.time()
        df = self._measure()
        p = save_frozen(df, self.component, self.system)
        print(f"[{self.component}] measured + froze {len(df)} rows in "
              f"{time.time()-t0:.0f}s → {p}", flush=True)
        return df

    # ── fit (CM-only, repeatable) ───────────────────────────────────────────
    def fit(self, frozen: Optional[pd.DataFrame] = None):
        if frozen is None:
            frozen = load_frozen(self.component, self.system)
            if frozen is None:
                raise FileNotFoundError(
                    f"no frozen measurements for {self.component}/{self.system}; "
                    f"run `calibrate measure {self.component}` first")
        profile, scored = self._fit(frozen)
        self.report(scored)
        path = self._save(profile)
        print(f"[{self.component}] saved profile → {path}", flush=True)
        return profile, scored

    # ── reporting: overall + per-group (in-regime vs flagged transfer) ──────
    def report(self, scored: pd.DataFrame) -> None:
        inr = scored[~scored.get("dataset", pd.Series(index=scored.index)).isin(self.fit_exclude)] \
            if self.fit_exclude and "dataset" in scored else scored
        sp, mape, _ = accuracy(inr[self.measured_col], inr[self.cm_col])
        tag = f"  (in-regime; excl {self.fit_exclude})" if self.fit_exclude else ""
        print(f"\n[{self.component}] IN-REGIME n={len(inr)} Spearman={sp:.3f} "
              f"MAPE={mape:.1f}%{tag}", flush=True)
        if len(inr) != len(scored):
            spa, mpa, _ = accuracy(scored[self.measured_col], scored[self.cm_col])
            print(f"[{self.component}] ALL      n={len(scored)} Spearman={spa:.3f} "
                  f"MAPE={mpa:.1f}%", flush=True)
        for gc in self.group_cols:
            if gc not in scored.columns:
                continue
            print(f"  by {gc}:", flush=True)
            for g, s in scored.groupby(gc):
                a, b, r = accuracy(s[self.measured_col], s[self.cm_col])
                flag = "  <- transfer (excl from fit)" if g in self.fit_exclude else ""
                print(f"    {str(g):18s} n={len(s):4d} Sp={a:.3f} MAPE={b:5.1f}% "
                      f"ratio_med={r:.2f}{flag}", flush=True)

    # ── subclass hooks ──────────────────────────────────────────────────────
    def _measure(self) -> pd.DataFrame:
        raise NotImplementedError

    def _fit(self, frozen: pd.DataFrame):
        raise NotImplementedError

    def _save(self, profile) -> str:
        raise NotImplementedError
