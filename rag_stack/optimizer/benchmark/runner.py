"""BenchmarkRunner: drives the optimizer loop and records metrics."""

import csv
import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from rag_stack.optimizer.base import BaseOptimizer
from rag_stack.optimizer.benchmark.metrics import (
    hypervolume_absolute,
    hv_difference,
    inverted_generational_distance,
)
from rag_stack.optimizer.benchmark.synthetic_functions.base import SyntheticFunction


# Default backoff schedule for transient CUDA OOM (concurrent GPU users on
# shared machines come and go). Cumulative max wait ≈ 17 min.
_OOM_RETRY_DELAYS_S = (30.0, 60.0, 120.0, 240.0, 480.0)


def _is_cuda_oom(exc: BaseException) -> bool:
    """Match CUDA out-of-memory across PyTorch versions.

    Newer torch raises ``torch.cuda.OutOfMemoryError`` (a subclass of
    ``RuntimeError``); older versions raise plain ``RuntimeError`` whose
    message contains "out of memory". Both forms have "out of memory" in
    the stringified message, so a single string check covers them.
    """
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return "out of memory" in msg or "outofmemoryerror" in msg


def _with_oom_retry(fn: Callable, label: str, delays_s=_OOM_RETRY_DELAYS_S):
    """Run ``fn()`` with retry on CUDA OOM. Frees torch CUDA cache and
    sleeps with exponential backoff between attempts. Re-raises non-OOM
    exceptions immediately and re-raises the last OOM after all retries.
    """
    n_attempts = len(delays_s) + 1
    for attempt in range(n_attempts):
        try:
            return fn()
        except RuntimeError as e:
            if not _is_cuda_oom(e):
                raise
            if attempt >= len(delays_s):
                raise
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            delay = delays_s[attempt]
            print(
                f"  [{label}] CUDA OOM (attempt {attempt + 1}/{n_attempts}); "
                f"freed cache, sleeping {delay:.0f}s before retry. Cause: {e}",
                flush=True,
            )
            time.sleep(delay)


@dataclass
class Snapshot:
    """One row of benchmark output, recorded at each checkpoint."""
    iteration: int
    cumulative_gt_calls: int
    cumulative_cm_calls: int
    cumulative_optimizer_time_s: float
    hv: float            # absolute hypervolume of the optimizer's Pareto (higher=better)
    hv_diff: float       # legacy regret-vs-true-Pareto; nan now (no true Pareto for real-data fn)
    igd: float           # legacy; nan now
    get_pareto_time_s: float


@dataclass
class EvalRecord:
    """One evaluated configuration, recorded for the (obj_a, obj_b) plots.

    ``obj_values`` is keyed by ``objective_name`` exactly as defined in
    ``SyntheticFunction.objectives``. The runner records GROUND-TRUTH values
    for every objective on every eval — even cheap-only iters get full
    readings because the runner re-eval the suggested config through
    ``function.evaluate(...)``.

    ``config`` is the full nested config dict that was evaluated. Stored so
    pareto-cache rebuilds can re-evaluate it noise-free (see
    ``SyntheticRAGFunction._pool_noise_free_evals``).
    """
    phase: str           # "cheap_batch", "cheap_bo", "main_both", "main_cheap"
    iteration: int       # 0 for pre-loop phases
    cumulative_gt_calls: int
    cumulative_cm_calls: int
    obj_values: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Run an optimizer on a synthetic function and record metrics.

    Usage:
        func = BraninCurrinFunction()
        optimizer = SMACMOBOOptimizer(
            config_space=func.search_space,
            objectives=func.objectives,
        )
        runner = BenchmarkRunner(func, optimizer, n_iterations=80, checkpoint_interval=10)
        snapshots = runner.run()
        runner.save_csv(snapshots, "results.csv")
    """

    def __init__(
        self,
        function: SyntheticFunction,
        optimizer: BaseOptimizer,
        gt_budget: int = 80,
        checkpoint_interval: int = 10,
        cache_dir: Optional[Path] = None,
    ):
        self.function = function
        self.optimizer = optimizer
        self.gt_budget = gt_budget
        self.checkpoint_interval = checkpoint_interval
        # Per-benchmark subfolder where the frozen reference Pareto lives
        # (``<cache_dir>/_pareto_cache.npy``). When None, cache load/save
        # are silent no-ops — every call falls through to the function's
        # `compute_true_pareto`. The benchmark CLI always passes a
        # `cache_dir`; ad-hoc programmatic uses (e.g. unit tests) can omit
        # it to skip caching.
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Objectives with cost <= this threshold are "cheap" (like CM),
        # only expensive objectives count as GT calls
        costs = [obj["cost"] for obj in function.objectives]
        self._cheap_threshold = min(costs)
        self._cheap_obj_names = [
            obj["objective_name"] for obj in function.objectives
            if obj["cost"] <= self._cheap_threshold
        ]
        self._all_obj_names = [obj["objective_name"] for obj in function.objectives]

        # Precompute true Pareto front once
        self._true_pareto: Optional[np.ndarray] = None

        # Per-eval ground-truth (obj_a, obj_b, ...) record for distribution plots.
        self.eval_history: List[EvalRecord] = []

    @property
    def true_pareto(self) -> np.ndarray:
        """(n_pareto, n_obj) matrix in ``self._all_obj_names`` column order.

        Resolution order:
        1. **Frozen cache** at ``<cache_dir>/_pareto_cache.npy`` — built
           once via ``build_pareto_cache.py``. Microsecond load.
        2. **Fallback** to ``function.compute_true_pareto()`` (NSGA-II).
           Result is auto-saved to the cache file so subsequent runs are
           fast.

        ``compute_true_pareto`` returns ``Dict[obj_name, ndarray]``; we
        look up each name explicitly here so the matrix's column order
        is keyed by name. The cached array is already in column order
        (the build script writes it that way) — loads as-is.
        """
        if self._true_pareto is None:
            cached = self._load_cached_pareto()
            if cached is not None:
                self._true_pareto = cached
            else:
                pareto_dict = self.function.compute_true_pareto()
                missing = [n for n in self._all_obj_names if n not in pareto_dict]
                if missing:
                    raise ValueError(
                        f"compute_true_pareto missing keys for objectives: {missing}. "
                        f"Got {sorted(pareto_dict.keys())}, "
                        f"need {self._all_obj_names}."
                    )
                self._true_pareto = np.column_stack(
                    [pareto_dict[n] for n in self._all_obj_names]
                )
                # Auto-cache so subsequent runs hit the disk cache instead
                # of re-running NSGA-II (the new fallback path — tens of
                # seconds to minutes, depending on n_samples and surrogate
                # cost). Skipped silently when no cache_dir was given.
                self._save_cached_pareto(self._true_pareto)
        return self._true_pareto

    def _load_cached_pareto(self) -> Optional[np.ndarray]:
        """Load ``<cache_dir>/_pareto_cache.npy`` if it exists and matches
        the function's objective count. Returns ``None`` for any miss
        (no cache_dir, no file, shape mismatch) → caller falls through to
        recompute."""
        from rag_stack.optimizer.benchmark.build_pareto_cache import (
            load_cached_pareto,
        )

        if self.cache_dir is None:
            return None
        return load_cached_pareto(self.cache_dir, len(self._all_obj_names))

    def _save_cached_pareto(self, arr: np.ndarray) -> None:
        """Persist ``arr`` to ``<cache_dir>/_pareto_cache.npy``. Silent
        no-op when ``cache_dir`` is ``None`` or the directory can't be
        written."""
        from rag_stack.optimizer.benchmark.build_pareto_cache import (
            save_cached_pareto,
        )

        if self.cache_dir is None:
            return
        save_cached_pareto(self.cache_dir, arr)

    def _record_eval(
        self,
        composite_values: dict,
        config: dict,
        phase: str,
        iteration: int,
        cumulative_gt_calls: int,
        cumulative_cm_calls: int,
    ) -> None:
        """Append one full-obj ground-truth reading to eval_history.

        ``composite_values`` is the dict returned by ``function.evaluate``.
        Each obj's value is either a float or a dict with ``"combined"`` plus
        optional sub-metric keys; we record the canonical scalar.

        ``config`` is the nested config that was evaluated; persisted to the
        eval CSV as JSON so pareto-cache rebuilds can re-evaluate noise-free.
        """
        obj_values: Dict[str, float] = {}
        for name in self._all_obj_names:
            v = composite_values.get(name)
            if isinstance(v, dict):
                v = v.get("combined", float("nan"))
            obj_values[name] = float(v) if v is not None else float("nan")
        self.eval_history.append(EvalRecord(
            phase=phase,
            iteration=iteration,
            cumulative_gt_calls=cumulative_gt_calls,
            cumulative_cm_calls=cumulative_cm_calls,
            obj_values=obj_values,
            config=config,
        ))

    def _evaluate(self, config: dict) -> dict:
        """Evaluate one config. Returns ``{obj_name: value}`` where value
        is either a float or a dict with at least ``"combined"`` (see the
        contract on ``SyntheticFunction.evaluate``). Sub_metric-aware
        consumers (topo, smac_decoupled_*) read the dict; scalar-only
        consumers unwrap ``v["combined"]``."""
        return self.function.evaluate(config)

    def run(self) -> List[Snapshot]:
        """Run the full optimization loop, return snapshots at GT checkpoints.

        Single uniform loop: each iteration calls ``optimizer.suggest()``,
        evaluates only the requested ``eval_obj_names``, and stops when
        ``cumulative_gt_calls >= self.gt_budget``. The optimizer owns all
        phase scheduling internally (cheap-only init, both-init, main loop),
        signalling phase via the ``eval_obj_names`` it returns.

        - If ``eval_obj_names`` includes any expensive objective: GT call,
          phase tagged ``main_both`` (joint) or ``main_quality`` (expensive-only).
        - Else: cheap-only, phase tagged ``main_cheap``.
        """
        snapshots: List[Snapshot] = []
        cumulative_gt_calls = 0
        cumulative_cm_calls = 0
        cumulative_optimizer_time = 0.0
        iteration = 0
        # Per-objective call counts (for log display). Initialized from the
        # function's declared objectives so the printed order is stable
        # across optimizer types (some optimizers don't expose objective_names).
        obj_call_counts: Dict[str, int] = {
            name: 0 for name in self._all_obj_names
        }

        while cumulative_gt_calls < self.gt_budget:
            iteration += 1
            t0 = time.perf_counter()
            configs, eval_obj_names = _with_oom_retry(
                lambda: self.optimizer.suggest(n_suggestions=1),
                label=f"suggest@iter{iteration}",
            )
            suggest_time = time.perf_counter() - t0

            if configs is None:
                print(f"  [iter {iteration}] optimizer exhausted, stopping early")
                break

            for config in configs:
                composite_values = self._evaluate(config)
                obj_dict = {
                    name: [composite_values[name]]
                    for name in eval_obj_names
                    if name in composite_values
                }
                # Per-obj call counter: every evaluated obj this iter ticks +1.
                for name in eval_obj_names:
                    if name in obj_call_counts:
                        obj_call_counts[name] += 1
                has_expensive_obj = any(
                    name not in self._cheap_obj_names
                    for name in eval_obj_names
                )
                if has_expensive_obj:
                    cumulative_gt_calls += 1
                    phase = "main_both" if len(eval_obj_names) > 1 else "main_quality"
                else:
                    cumulative_cm_calls += 1
                    phase = "main_cheap"
                self._record_eval(
                    composite_values, config, phase, iteration,
                    cumulative_gt_calls=cumulative_gt_calls,
                    cumulative_cm_calls=cumulative_cm_calls,
                )

                self.optimizer.add_objectives([config], obj_dict)
                add_time = time.perf_counter() - t0
                cumulative_optimizer_time += suggest_time + add_time

            # Heartbeat for cheap-only iters: a snapshot fires only on GT,
            # so without this Phase A is silent for hundreds of iters.
            # Print every 50 cheap iters; non-blocking, keeps the log alive.
            if not has_expensive_obj and cumulative_cm_calls % 50 == 0:
                counts_str = "  ".join(
                    f"{n}={c:4d}" for n, c in obj_call_counts.items()
                )
                print(
                    f"  [iter {iteration:4d}] cheap-only progress  "
                    f"{counts_str}  "
                    f"opt_time={cumulative_optimizer_time:.2f}s",
                    flush=True,
                )

            # --- checkpoint at GT-call milestones ---
            # Snapshot every ``checkpoint_interval`` GT calls and at the
            # final GT call. Cheap-only iters don't trigger snapshots — they
            # don't change the GT-vs-true-Pareto comparison, so a snapshot
            # would be a no-op repeat.
            if has_expensive_obj and (
                cumulative_gt_calls % self.checkpoint_interval == 0
                or cumulative_gt_calls == self.gt_budget
            ):
                snapshot = self._take_snapshot(
                    cumulative_gt_calls, cumulative_gt_calls,
                    cumulative_cm_calls, cumulative_optimizer_time,
                )
                snapshots.append(snapshot)
                counts_str = "  ".join(
                    f"{n}={c:4d}" for n, c in obj_call_counts.items()
                )
                print(
                    f"  [iter {iteration:4d}] {counts_str}  "
                    f"opt_time={cumulative_optimizer_time:.2f}s  "
                    f"hv={snapshot.hv:.6f}  "
                    f"pareto_time={snapshot.get_pareto_time_s:.2f}s"
                )

        return snapshots

    def _take_snapshot(
        self,
        iteration: int,
        cumulative_gt_calls: int,
        cumulative_cm_calls: int,
        cumulative_optimizer_time: float,
    ) -> Snapshot:
        """Call get_pareto() and compute metrics against true Pareto front."""
        t0 = time.perf_counter()
        pareto_configs, pareto_pred_objs, _ = _with_oom_retry(
            lambda: self.optimizer.get_pareto(),
            label=f"get_pareto@gt{cumulative_gt_calls}",
        )
        get_pareto_time = time.perf_counter() - t0

        # Re-evaluate pareto configs with the **noise-free** function — this
        # is what makes the snapshot's hv_diff an inference-regret measure:
        # truth values at optimizer-recommended configs. Calling plain
        # ``evaluate`` here would re-noise the obs and silently turn the
        # snapshot into observation regret (winner's curse biases hv_diff
        # low under noise, breaking noise-vs-noiseless comparison). For
        # deterministic functions ``evaluate_noise_free`` defaults to
        # ``evaluate`` so this is a no-op.
        true_objs = []
        for config in pareto_configs:
            vals = self.function.evaluate_noise_free(config)
            row = []
            for name in self._all_obj_names:
                v = vals[name]
                row.append(v["combined"] if isinstance(v, dict) else v)
            true_objs.append(row)

        if true_objs:
            pred_array = np.array(true_objs)
        else:
            pred_array = np.empty((0, self.function.n_objectives))

        ref = self.function.ref_point
        # Absolute HV (higher=better) — the real-world metric. NO true Pareto is
        # computed (it is unknowable in practice, exactly as in rag_stack e2e), so
        # the expensive NSGA-II `compute_true_pareto` is skipped entirely. The
        # function's `HV_NORM = (s_perf, s_quality)` normalizes each objective to a
        # fair [0,1] scale before HV, so neither axis dominates by raw magnitude.
        hv = hypervolume_absolute(
            pred_array, ref, norm=getattr(self.function, "HV_NORM", None)
        )

        return Snapshot(
            iteration=iteration,
            cumulative_gt_calls=cumulative_gt_calls,
            cumulative_cm_calls=cumulative_cm_calls,
            cumulative_optimizer_time_s=round(cumulative_optimizer_time, 4),
            hv=round(hv, 6),
            hv_diff=float("nan"),
            igd=float("nan"),
            get_pareto_time_s=round(get_pareto_time, 4),
        )

    def save_eval_history(self, metrics_csv_path: str, seed: int) -> None:
        """Write the per-eval ground-truth trail next to the metrics CSV.

        Schema: ``seed,noise_sigma,phase,iteration,cumulative_gt_calls,
        cumulative_cm_calls,<obj_name_0>,<obj_name_1>,...,config_json`` —
        obj-name columns come from ``function.objectives`` order (no
        hardcoded ``quality``/``speed``). Re-running one seed replaces only
        that seed's rows.

        ``config_json`` is the full nested config that was evaluated,
        serialized as JSON. The pareto-cache rebuild (in
        ``SyntheticRAGFunction._pool_noise_free_evals``) re-evaluates each
        config noise-free and pools the results into the reference Pareto.
        Without this column, noisy obs would either contaminate the
        reference (via extreme-value statistics) or — under the prior
        ``noise_sigma==0`` filter — silently drop every production eval.
        """
        metrics_path = Path(metrics_csv_path)
        evals_path = metrics_path.with_name(
            f"{metrics_path.stem}__evals{metrics_path.suffix}"
        )
        evals_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclude obj_values (split across obj columns) and config (split into
        # config_json column) from the scalar passthrough.
        scalar_fields = [
            f.name for f in fields(EvalRecord)
            if f.name not in ("obj_values", "config")
        ]
        fieldnames = (
            ["seed", "noise_sigma"]
            + scalar_fields
            + list(self._all_obj_names)
            + ["config_json"]
        )
        noise_sigma = float(getattr(self.function, "_noise_sigma", 0.0))

        existing_rows: List[dict] = []
        if evals_path.exists():
            with open(evals_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row.get("seed", -1)) != seed:
                        # Legacy rows pre-dating the noise_sigma column are
                        # treated as noise=0 (consistent with how those runs
                        # were configured at the time).
                        row.setdefault("noise_sigma", "0.0")
                        # Legacy rows pre-dating config_json: empty string
                        # means "config unknown, skip during pareto rebuild".
                        row.setdefault("config_json", "")
                        existing_rows.append(row)

        new_rows = []
        for rec in self.eval_history:
            row = {"seed": seed, "noise_sigma": noise_sigma}
            for fn in scalar_fields:
                row[fn] = getattr(rec, fn)
            for name in self._all_obj_names:
                row[name] = rec.obj_values.get(name, float("nan"))
            row["config_json"] = json.dumps(rec.config, sort_keys=True)
            new_rows.append(row)

        with open(evals_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerows(new_rows)
        print(f"Eval history saved to {evals_path} (seed {seed}, {len(new_rows)} evals)")

    @staticmethod
    def save_csv(snapshots: List[Snapshot], path: str, seed: int) -> None:
        """Append this run's snapshots to a per-(optimizer, fn) CSV.

        Each row carries a ``seed`` column. If ``path`` already exists (from a
        prior seed's run), rows for *this* seed are replaced and rows for other
        seeds are preserved — so re-running one seed is idempotent and won't
        corrupt the other seeds' data.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        snapshot_fields = [f.name for f in fields(Snapshot)]
        fieldnames = ["seed"] + snapshot_fields

        # Preserve rows from OTHER seeds in the same file.
        existing_rows: List[dict] = []
        if path_obj.exists():
            with open(path_obj, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row.get("seed", -1)) != seed:
                        existing_rows.append(row)

        new_rows = [
            {"seed": seed, **{fn: getattr(s, fn) for fn in snapshot_fields}}
            for s in snapshots
        ]

        with open(path_obj, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerows(new_rows)
        print(f"Results saved to {path} (seed {seed})")
