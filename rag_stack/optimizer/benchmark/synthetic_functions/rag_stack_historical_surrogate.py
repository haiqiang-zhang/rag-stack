"""Surrogate benchmark over the rag-stack search space, backed by historical
evaluation data for quality and optionally performance.

Motivation: real ``rag_stack`` optimizer runs take ~5 hours for 50 evals
because each eval calls a remote LLM judge per sub-metric. Comparing two
optimizers head-to-head over multiple seeds at that cost is impractical.

This benchmark provides a fast, deterministic stand-in:
  * **Quality** — a multi-output GP surrogate trained on N historical
    (config, sub-metric) tuples pooled from finished optimizer-run CSVs.
    The surrogate returns posterior MEAN per sub-metric (no MC, no noise)
    so the function is deterministic. ``aggregate_quality_metrics`` then
    folds the sub-metrics into the scalar ``combined`` exactly the same
    way the live evaluator does, so optimizer code paths are unchanged.
  * **Performance** — by default the benchmark calls the real project cost
    model (``StaticAssembly``), which is deterministic and calibrated.
    For fast optimizer-method studies, ``performance_mode=historical_surrogate``
    trains a GBM on the historical ``performance`` column and avoids
    per-candidate cost-model assembly.

Apples-to-apples: with ``performance_mode=cost_model``, the live optimizer
trajectories on the same YAML show the same quality values (within
~LLM-judge noise) and the exact same performance values that this benchmark
produces. With ``historical_surrogate`` mode, the whole objective surface is
learned from past runs and is intended for fast optimizer-mechanism tests.

Use cases:
  * Race two optimizer variants (e.g. topo vs ax_qnehvi_mixedgp) across
    many seeds in seconds-per-iter instead of minutes-per-iter
  * Ablate a single optimizer hyperparameter without paying for fresh LLM
    judge calls
  * Verify a candidate fix (e.g. new GP kernel / noise prior) reproduces
    or beats the live trajectory before spending real eval budget on it

Wired into the benchmark CLI via the
``rag_stack_historical_surrogate`` registry key. See
:func:`rag_stack.optimizer.benchmark.cli` for usage.
"""
from __future__ import annotations

import ast
import csv
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from ax.core.parameter import ChoiceParameter
from ax.core.search_space import SearchSpace
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import SyntheticFunction, run_nsga2_pareto

logger = logging.getLogger("RAG-Stack")


# Default sub-metric set (matches the post-prune YAMLs of 2026-05-20).
# Override by passing ``sub_metrics=`` to the constructor.
DEFAULT_SUB_METRICS: Tuple[str, ...] = (
    "deepeval_faithfulness",
    "deepeval_answer_relevancy",
    "deepeval_context_precision",
    "deepeval_context_recall",
    "deepeval_contextual_relevancy",
    "retrieval_recall",
    "retrieval_ndcg",
)

# Repo root, resolved from this module's location (robust to CWD).
_REPO_ROOT = Path(__file__).resolve().parents[4]

# Default historical-eval pool for the quality surrogate: ALL FOUR runs under
# benchmarks/historical_surrogate/ (≈195 rows). They share ONE raw corpus
# (datasets/dragonball/raw_corpus_en.parquet), ONE llama judge, and ONE
# search space (chunk_size is a search-space dim captured in X) — differing only
# in which optimizer explored the configs, so pooling is a clean coverage gain
# with no corpus/judge confound. A config MAY override ``historical_eval_dirs``
# (e.g. a subset), but this is the canonical default. ax_correctness is FIRST so
# it provides the cost-model ``data/`` dir.
DEFAULT_HISTORICAL_EVAL_DIRS: List[str] = [
    str(_REPO_ROOT / "benchmarks" / "historical_surrogate" / name)
    for name in (
        "ax_correctness",
        "topo_stacked_correctness_v2",
        "topo_stacked_correctness_v3",
        "evolve_ax_aux_constraint",
    )
]


def _coerce_csv_scalar(v: str) -> Any:
    """Parse a scalar CSV cell back to a reasonable Python value."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if f.is_integer():
        return int(f)
    return f


def _decode_csv_row_to_nested(row: dict, vdb_names: Sequence[str]) -> Optional[dict]:
    """Reconstruct a nested config dict from a CSV row.

    Walks the row's namespaced columns (``pipeline__sequential``,
    ``corpus__chunker``, ``<vdb_name>__<key>``) back into the nested
    layout that ``RagSearchSpace.encode_flat_config`` expects. Returns
    ``None`` on parse failure so the loader can skip the row.
    """
    try:
        nested: Dict[str, Any] = {}
        if row.get("pipeline__sequential"):
            nested["pipeline"] = {
                "sequential": ast.literal_eval(row["pipeline__sequential"])
            }
        if row.get("corpus__chunker"):
            nested["corpus"] = {"chunker": ast.literal_eval(row["corpus__chunker"])}
        for vdb_name in vdb_names:
            prefix = f"{vdb_name}__"
            vdb: Dict[str, Any] = {}
            for col, v in row.items():
                if not col.startswith(prefix) or v in (None, ""):
                    continue
                key = col[len(prefix):]
                vdb[key] = _coerce_csv_scalar(v)
            if vdb:
                nested[vdb_name] = vdb
        return nested
    except Exception:
        return None


class RagStackHistoricalSurrogateFunction(SyntheticFunction):
    """SyntheticFunction over the rag-stack search space.

    Args:
        yaml_path: path to a rag-stack YAML config (defines search space +
            cost model hardware + combined_quality aggregation rule)
        historical_eval_dirs: list of ``tmp_outputs/rag_stack_projects/<proj>``
            directories whose ``all_evaluations.csv`` should be pooled to
            train the quality surrogate. All projects must share the same
            YAML schema and sub-metric set. The FIRST entry is also used
            as the data source for cost-model setup (its ``data/qa.parquet``
            + ``data/corpus.parquet`` are read once by the Controller-light
            constructor) — pick a dir where these files exist.
        sub_metrics: sub-metric names to surrogate. Defaults to the
            post-prune 7-metric set.
        vdb_name: vectordb key prefix in the YAML (e.g.
            ``faiss_ivf_index``). Pulled from yaml if absent.

    The fitted surrogate GPs and the training X/Y tensors are NOT saved
    to disk — fit time is ~1-2 s for N=140, K=7 sub-metrics, fast enough
    that on-construct fit is preferable to a stale on-disk cache.
    """

    name = "rag_stack_historical_surrogate"

    def __init__(
        self,
        yaml_path: str,
        historical_eval_dirs: Optional[List[str]] = None,
        sub_metrics: Tuple[str, ...] = DEFAULT_SUB_METRICS,
        vdb_name: Optional[str] = None,
        stochastic: bool = False,
        noise_scale: float = 1.0,
        rng_seed: Optional[int] = None,
        surrogate_difficulty: float = 0.0,
        gbm_n_estimators: int = 300,
        gbm_max_depth: int = 5,
        use_gpu: bool = False,
        performance_mode: str = "cost_model",
        invalid_objective_penalty: bool = False,
        invalid_performance: float = 0.0,
        invalid_quality: float = 0.0,
    ):
        # Default to the canonical 4-run pool (≈195 rows) when a caller/config
        # doesn't specify; a config MAY still override with its own list.
        if not historical_eval_dirs:
            historical_eval_dirs = DEFAULT_HISTORICAL_EVAL_DIRS
            logger.info(
                f"{self.name}: historical_eval_dirs not given — using default "
                f"4-run pool ({len(DEFAULT_HISTORICAL_EVAL_DIRS)} dirs)."
            )
        self.historical_eval_dirs = list(historical_eval_dirs)
        self.stochastic = bool(stochastic)
        self.noise_scale = float(noise_scale)
        self.surrogate_difficulty = max(0.0, min(1.0, float(surrogate_difficulty)))
        self.gbm_n_estimators = int(gbm_n_estimators)
        self.gbm_max_depth = int(gbm_max_depth)
        mode = str(performance_mode).strip().lower()
        if mode in {"surrogate", "historical", "historical_surrogate", "gbm"}:
            mode = "historical_surrogate"
        elif mode != "cost_model":
            raise ValueError(
                f"{self.name}: unknown performance_mode={performance_mode!r}; "
                "expected 'cost_model' or 'historical_surrogate'."
            )
        self.performance_mode = mode
        self.invalid_objective_penalty = bool(invalid_objective_penalty)
        self.invalid_performance = float(invalid_performance)
        self.invalid_quality = float(invalid_quality)
        # Fit the quality GPs on GPU when requested (and available). evaluate()
        # already matches the query tensor to each GP's device, so a cuda fit is
        # transparent to callers.
        self._device = torch.device(
            "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        )
        # Per-instance RNG so concurrent races on different seeds get
        # different noise streams but each race is reproducible.
        self._rng = torch.Generator()
        if rng_seed is not None:
            self._rng.manual_seed(int(rng_seed))
        # Cost-model qa/corpus are SHARED across all pooled runs (byte-identical),
        # so they live in ONE ``data/`` dir at the common parent
        # (benchmarks/historical_surrogate/data/), NOT under any one run dir — the
        # per-run dirs hold only all_evaluations.csv. (Re-chunking uses the YAML's
        # raw corpus; build scratch goes to tmp — see PerformanceContext below.)
        self._data_dir = self._resolve_data_dir()
        if not (self._data_dir / "qa.parquet").exists():
            raise ValueError(
                f"{self.name}: shared cost-model data not found at {self._data_dir} "
                f"(expected qa.parquet + corpus.parquet next to the pooled run dirs)."
            )
        self.sub_metrics = tuple(sub_metrics)

        # Build ONLY the cost-model context (search space + config + resolve +
        # StaticAssembly), not a full Controller. The full Controller would also
        # construct the optimizer and the quality evaluator (which loads
        # embedding/reranker/generation models) — neither is needed here:
        # quality comes from the GP surrogate, performance from StaticAssembly.
        # Controller composes the SAME PerformanceContext, so the cost-model
        # behavior here still matches the live run path exactly (same
        # resolve_config, same StaticAssembly, same hardware spec) — but without
        # the model-loading overhead.
        from rag_stack.performance_context import PerformanceContext
        from rag_stack.search_space import RagSearchSpace

        # Suppress noisy WARNINGs during context startup.
        old_level = logging.getLogger("RAG-Stack").level
        logging.getLogger("RAG-Stack").setLevel(logging.ERROR)
        try:
            qa_path = str(self._data_dir / "qa.parquet")
            corpus_path = str(self._data_dir / "corpus.parquet")
            # Inputs are READ from the frozen, reproducible shared data/ dir, but
            # the PerformanceContext's runtime SCRATCH (re-chunked corpus, faiss
            # index, controller artifacts) must NOT pollute benchmarks/ — route
            # project_dir to a stable tmp scratch dir (cacheable across builds).
            scratch_dir = os.path.join("tmp_outputs", "_surrogate_scratch", "historical_surrogate")
            os.makedirs(scratch_dir, exist_ok=True)
            self._ctx = PerformanceContext(
                config_path=yaml_path,
                project_dir=scratch_dir,
                qa_data_path=qa_path,
                corpus_data_path=corpus_path,
                # The surrogate never runs the RAG models (quality is the GP,
                # performance is the analytic cost model), so skip the HF
                # model-cache check that would otherwise snapshot_download the
                # configured embedding/reranker/generation models.
                check_model_cache=False,
            )
        finally:
            logging.getLogger("RAG-Stack").setLevel(old_level)

        self._ssops: RagSearchSpace = self._ctx._ssops
        self._combined_quality_spec = (
            (self._ctx.config.get("eval_backend_setting") or {}).get("combined_quality")
        )
        from rag_stack.search_space import algo_vectordb_blocks

        vdb_blocks = algo_vectordb_blocks(self._ctx.config)
        self._vdb_names = [
            str(b["name"]) for b in vdb_blocks
            if isinstance(b, dict) and b.get("name")
        ]
        if vdb_name:
            self._vdb_names = [vdb_name]

        # Train quality surrogate
        logger.info(
            f"{self.name}: loading historical eval data from "
            f"{len(self.historical_eval_dirs)} project dirs ..."
        )
        X, Y_sub, Y_perf = self._load_training_data()
        X = X.to(self._device)
        Y_sub = Y_sub.to(self._device)
        logger.info(
            f"{self.name}: training {len(self.sub_metrics)} quality GP(s) "
            f"on N={X.shape[0]} obs ({X.shape[1]}D input) on {self._device}."
        )
        self._cat_dims = self._detect_cat_dims()
        self._quality_gps = self._fit_sub_metric_gps(X, Y_sub)
        # GBM trained only when difficulty > 0. Captures local non-smoothness
        # (categorical phase transitions, top_k saturation, threshold effects)
        # that GP Matern-2.5 smooths over. Blended into evaluate() with weight
        # equal to surrogate_difficulty.
        if self.surrogate_difficulty > 0:
            self._quality_gbms = self._fit_sub_metric_gbms(X, Y_sub)
            logger.info(
                f"{self.name}: GBM layer fit "
                f"(difficulty={self.surrogate_difficulty:.2f}, "
                f"n_estimators={self.gbm_n_estimators}, "
                f"max_depth={self.gbm_max_depth})"
            )
        else:
            self._quality_gbms = None
        if self.performance_mode == "historical_surrogate":
            self._performance_gbm = self._fit_performance_gbm(X, Y_perf)
            logger.info(
                f"{self.name}: performance GBM surrogate fit "
                f"(n_estimators={self.gbm_n_estimators}, "
                f"max_depth={self.gbm_max_depth})"
            )
        else:
            self._performance_gbm = None
        logger.info(
            f"{self.name}: surrogate ready. Cat dims: {self._cat_dims}. "
            f"Sub-metrics: {list(self.sub_metrics)}. "
            f"performance_mode={self.performance_mode}."
        )

    # ------------------------------------------------------------------
    # Data loading + GP fitting
    # ------------------------------------------------------------------

    def _resolve_data_dir(self) -> Path:
        """Find qa/corpus data for historical-project pools.

        Older benchmark pools store one shared ``data/`` directory next to the
        per-run folders. Live ``tmp_outputs/rag_stack_projects/<project>``
        folders store their own ``data/`` subdirectory. Support both layouts so
        the same surrogate builder can train from checked-in benchmark data and
        fresh tmp-output projects.
        """
        candidates: List[Path] = []
        for d in self.historical_eval_dirs:
            p = Path(d).resolve()
            candidates.append(p / "data")
            candidates.append(p.parent / "data")
        seen: set[Path] = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            if (cand / "qa.parquet").exists() and (cand / "corpus.parquet").exists():
                return cand
        return candidates[0]

    def _load_training_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool eval CSVs from each historical dir, decode + re-encode each
        row through ``RagSearchSpace.encode_flat_config`` to get a flat
        normalized X. Returns ``(X, Y_sub, Y_perf)`` shaped (N, d), (N, K),
        and (N, 1)."""
        X_rows: List[torch.Tensor] = []
        Y_rows: List[List[float]] = []
        P_rows: List[float] = []
        # Row usability (07-07 outer-join fix): a row is usable when perf and
        # the OBJECTIVE metrics (the combined_quality inputs) are present —
        # diagnostic-only sub-metrics may be missing per row (recorded as NaN;
        # each metric's model trains on its own present subset). The old
        # strict-join dropped a row missing ANY listed metric, which zeroed
        # whole pools whenever the list and the data disagreed.
        objective_metrics = set(
            (self._combined_quality_spec or {}).get("metrics")
            or list(self.sub_metrics[:1])
        ) & set(self.sub_metrics)
        if not objective_metrics:
            objective_metrics = set(self.sub_metrics[:1])
        drop_counts: Dict[str, int] = {}
        nan_counts: Dict[str, int] = {sm: 0 for sm in self.sub_metrics}
        seen_quality_cols: set = set()
        for d in self.historical_eval_dirs:
            csv_path = Path(d) / "all_evaluations.csv"
            if not csv_path.exists():
                logger.warning(f"{self.name}: skip {d}: no all_evaluations.csv")
                continue
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                seen_quality_cols.update(
                    c[len("quality__"):] for c in rows[0]
                    if c.startswith("quality__")
                )
            kept = 0
            for r in rows:
                try:
                    perf = float(r.get("performance"))
                except (TypeError, ValueError):
                    drop_counts["performance"] = drop_counts.get("performance", 0) + 1
                    continue
                if not np.isfinite(perf):
                    drop_counts["performance"] = drop_counts.get("performance", 0) + 1
                    continue
                sub_vals: List[float] = []
                killer = None
                for sm in self.sub_metrics:
                    v = r.get(f"quality__{sm}")
                    val = math.nan
                    if v not in (None, "", "nan"):
                        try:
                            val = float(v)
                        except ValueError:
                            val = math.nan
                    if math.isnan(val):
                        if sm in objective_metrics:
                            killer = sm
                            break
                        nan_counts[sm] += 1
                    sub_vals.append(val)
                if killer is not None:
                    drop_counts[killer] = drop_counts.get(killer, 0) + 1
                    continue
                nested = _decode_csv_row_to_nested(r, self._vdb_names)
                if nested is None:
                    drop_counts["<decode>"] = drop_counts.get("<decode>", 0) + 1
                    continue
                try:
                    X_row = self._ssops.encode_flat_config(nested).reshape(-1)
                except Exception:
                    drop_counts["<encode>"] = drop_counts.get("<encode>", 0) + 1
                    continue
                X_rows.append(X_row)
                Y_rows.append(sub_vals)
                P_rows.append(perf)
                kept += 1
            logger.info(f"{self.name}:   {Path(d).name}: kept {kept}/{len(rows)}")
        missing_diag = {sm: n for sm, n in nan_counts.items() if n}
        if missing_diag:
            logger.info(
                f"{self.name}: diagnostic sub-metrics missing per row "
                f"(NaN-masked; each metric trains on its present subset): "
                f"{missing_diag}"
            )
        if not X_rows:
            raise ValueError(
                f"{self.name}: no usable training rows found in "
                f"{self.historical_eval_dirs}. Per-cause drop counts: "
                f"{drop_counts or '{}'}; objective metrics required per row: "
                f"{sorted(objective_metrics)}; quality__ columns present in "
                f"the data: {sorted(seen_quality_cols)}."
            )
        X = torch.stack(X_rows, dim=0)
        Y = torch.tensor(Y_rows, dtype=X.dtype)
        P = torch.tensor(P_rows, dtype=X.dtype).reshape(-1, 1)
        return X, Y, P

    def _detect_cat_dims(self) -> List[int]:
        """Indices of un-ordered ChoiceParameters in the search space.

        Matches ``MixedSingleTaskGP`` ``cat_dims`` semantics: only
        Choice-without-is_ordered contributes Hamming. Ordered Choice and
        Range params are treated as continuous.
        """
        cat_dims: List[int] = []
        for i, p in enumerate(self._ssops.ss.parameters.values()):
            if isinstance(p, ChoiceParameter) and not getattr(p, "is_ordered", False):
                cat_dims.append(i)
        return cat_dims

    def _fit_sub_metric_gbms(
        self, X: torch.Tensor, Y_sub: torch.Tensor,
    ) -> Dict[str, Any]:
        """One Gradient-Boosted regressor per sub-metric.

        Trees do NOT smooth across categoricals and capture non-linear
        interactions natively — exactly the "ruggedness" real RAG has
        that the GP regresses away. Blended with GP in ``evaluate`` to
        increase surrogate difficulty without breaking the data-fit
        contract.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        out: Dict[str, Any] = {}
        X_np = X.detach().cpu().numpy()
        Y_np = Y_sub.detach().cpu().numpy()
        for k, sm in enumerate(self.sub_metrics):
            y_col = Y_np[:, k]
            mask = np.isfinite(y_col)   # 07-07 outer-join: per-metric subset
            if int(mask.sum()) < 3 or sm not in self._quality_gps:
                continue  # mirrors the GP skip — metric absent from evaluate()
            gbm = GradientBoostingRegressor(
                n_estimators=self.gbm_n_estimators,
                max_depth=self.gbm_max_depth,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            gbm.fit(X_np[mask], y_col[mask])
            out[sm] = gbm
        return out

    def _fit_performance_gbm(self, X: torch.Tensor, Y_perf: torch.Tensor) -> Any:
        """One fast regressor for the historical performance column."""
        from sklearn.ensemble import GradientBoostingRegressor

        gbm = GradientBoostingRegressor(
            n_estimators=self.gbm_n_estimators,
            max_depth=self.gbm_max_depth,
            learning_rate=0.05,
            subsample=0.8,
            random_state=43,
        )
        gbm.fit(
            X.detach().cpu().numpy(),
            Y_perf.detach().cpu().numpy().reshape(-1),
        )
        return gbm

    def _predict_performance_surrogate(self, config: dict) -> float:
        if self._performance_gbm is None:
            raise RuntimeError("performance surrogate requested before fit")
        X = self._ssops.encode_flat_config(config).reshape(1, -1)
        pred = float(self._performance_gbm.predict(X.detach().cpu().numpy())[0])
        return max(0.0, pred)

    def _fit_sub_metric_gps(
        self, X: torch.Tensor, Y_sub: torch.Tensor,
    ) -> Dict[str, Any]:
        """One independent GP per sub-metric.

        Uses ``MixedSingleTaskGP`` when ``cat_dims`` is non-empty (the
        Wan et al. mixed kernel handles cat dims natively), else falls
        back to ``SingleTaskGP``. All hyperparameters use BoTorch
        defaults (inferred noise) — matches ``ax_qnehvi_mixedgp`` and the
        post-fix topo path.
        """
        out: Dict[str, Any] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k, sm in enumerate(self.sub_metrics):
                y_k = Y_sub[:, k:k + 1]
                # per-metric present subset (07-07 outer-join: diagnostic
                # metrics may be NaN on rows that lack them)
                mask = torch.isfinite(y_k.reshape(-1))
                if int(mask.sum()) < 3:
                    logger.warning(
                        f"{self.name}: sub-metric {sm!r} has only "
                        f"{int(mask.sum())} training rows — no surrogate; it "
                        "will be absent from evaluate()'s quality dict."
                    )
                    continue
                X_k, y_k = X[mask], y_k[mask]
                if self._cat_dims:
                    gp = MixedSingleTaskGP(
                        train_X=X_k, train_Y=y_k,
                        cat_dims=self._cat_dims,
                        outcome_transform=Standardize(m=1),
                    )
                else:
                    gp = SingleTaskGP(
                        train_X=X_k, train_Y=y_k,
                        outcome_transform=Standardize(m=1),
                    )
                gp = gp.to(X.device)  # follow the training-data device (cpu/cuda)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                gp.eval()
                out[sm] = gp
        return out

    # ------------------------------------------------------------------
    # SyntheticFunction interface
    # ------------------------------------------------------------------

    @property
    def search_space(self) -> SearchSpace:
        return self._ssops.ss

    # Per-objective normalizers for the benchmark's absolute HV (read by
    # BenchmarkRunner._take_snapshot). Each objective is mapped to
    # clip(value / s, 0, 1) before HV, so neither axis dominates by raw
    # magnitude — a FAIR [0,1]×[0,1] hypervolume. Order matches `objectives`:
    #   performance / 10000  (cost-model throughput, hardware range ~100-10000)
    #   quality     / 1.0    (combined correctness scalar, already ∈ [0, 1])
    HV_NORM = (10000.0, 1.0)

    @property
    def objectives(self) -> List[Dict]:
        """Same shape as ``DEFAULT_OBJECTIVES`` in :mod:`controller`.

        ``performance`` from the cost model can be any positive throughput
        value (range varies with the hardware in the YAML — typically
        100-10000 on the H100 setup). ``quality`` from the surrogate is
        the combined sub-metric scalar, ∈ [0, 1] under the standard mean
        / weighted-mean aggregators.

        ref_point=0 on both: standard maximization convention (every
        valid eval is above the ref) — matches the live ``DEFAULT_OBJECTIVES``.
        """
        return [
            {"objective_name": "performance", "cost": 1.0, "ref_point": 0.0},
            {"objective_name": "quality", "cost": 10.0, "ref_point": 0.0},
        ]

    @property
    def ssops(self):
        return self._ssops

    def evaluate(self, config: dict) -> Dict[str, Any]:
        """Return ``{performance, quality}`` for one nested config dict.

        Quality: GP surrogate posterior MEAN per sub-metric, aggregated
        via ``aggregate_quality_metrics`` so the combined scalar matches
        what the live ``StaticRAGEvaluatorQualityOnly`` would produce given
        the same sub-metric outputs.

        Performance: exact ``RAGCMAssembly.evaluate_static`` on the resolved
        pipeline_config in ``cost_model`` mode, or a historical GBM prediction
        in ``historical_surrogate`` mode.

        Quality output structure:
            ``{"combined": float, "<sub_metric_1>": float, ...}``
        Performance output: plain float.

        Optimizer consumers expecting topo-style composite quality
        already handle the dict shape (see ``Controller._evaluate_objectives``).
        """
        # Performance via cost model or historical surrogate.
        if self.performance_mode == "historical_surrogate":
            perf = self._predict_performance_surrogate(config)
        else:
            pipeline_config = self._ctx.resolve_config(config)
            try:
                result = self._ctx.assembly.evaluate_static(pipeline_config)
                perf = float(result.performance_score)
            except Exception as exc:  # noqa: BLE001
                if not self.invalid_objective_penalty:
                    raise
                logger.warning(
                    "%s: cost-model invalid config penalized (%s)",
                    self.name,
                    exc,
                )
                q = max(0.0, min(1.0, self.invalid_quality))
                return {
                    "performance": self.invalid_performance,
                    "quality": {"combined": q, **{sm: q for sm in self.sub_metrics}},
                }

        # Quality via surrogate.  Match BOTH dtype and device of the GP's
        # train_inputs — otherwise the cat(train, test) inside gpytorch's
        # _get_test_prior_mean_and_covariances raises
        # "Expected all tensors to be on the same device" when, for example,
        # a topo run elsewhere has driven the surrogate GPs onto cuda but
        # encode_flat_config keeps producing CPU tensors.
        _ref = next(iter(self._quality_gps.values())).train_inputs[0]
        X = self._ssops.encode_flat_config(config).reshape(1, -1).to(
            dtype=_ref.dtype, device=_ref.device,
        )
        X_np = X.detach().cpu().numpy() if self._quality_gbms is not None else None
        sub_means: Dict[str, float] = {}
        with torch.no_grad():
            for sm, gp in self._quality_gps.items():
                post = gp.posterior(X)
                mu = post.mean.squeeze()
                _gbm = (self._quality_gbms or {}).get(sm)
                if _gbm is not None:
                    gbm_pred = float(_gbm.predict(X_np)[0])
                    alpha = self.surrogate_difficulty
                    base = (1.0 - alpha) * float(mu.item()) + alpha * gbm_pred
                else:
                    base = float(mu.item())
                if self.stochastic:
                    # Inject GP-posterior-σ scaled noise so the surrogate
                    # function is non-deterministic — each evaluate at the
                    # same x returns a sample around μ. Without this, every
                    # MOBO algorithm converges to the same Pareto-best (no
                    # discriminative power between optimizers because the
                    # function is "too smooth"). Note: ε is drawn from the
                    # instance RNG, so the noise stream is reproducible if
                    # ``rng_seed`` is fixed at construction.
                    sig = post.variance.sqrt().squeeze()
                    # CPU generator → sample on CPU, move to mu's device.
                    # Avoids "cuda device generator" error when mu lives on GPU.
                    eps_cpu = torch.randn(
                        (), dtype=mu.dtype, generator=self._rng,
                    )
                    eps = eps_cpu.to(mu.device)
                    noise = float((self.noise_scale * sig * eps).item())
                    sub_means[sm] = base + noise
                else:
                    sub_means[sm] = base
        # Clamp to [0, 1] — surrogate posterior can mildly extrapolate
        # outside the training range; sub-metrics are constructed in [0, 1]
        # by definition so clipping preserves the live evaluator's contract.
        for sm in sub_means:
            sub_means[sm] = max(0.0, min(1.0, sub_means[sm]))

        from rag_stack.controller import aggregate_quality_metrics
        q_combined = float(
            aggregate_quality_metrics(sub_means, self._combined_quality_spec)
        )
        return {
            "performance": perf,
            "quality": {"combined": q_combined, **sub_means},
        }

    def compute_true_pareto(
        self,
        n_samples: int = 200_000,
        evals_dir: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Approximate the true Pareto via NSGA-II on the surrogate.

        Standard practice for complex hierarchical mixed search spaces
        (same as :class:`SyntheticRAGFunction`). Random sampling would
        under-cover the high-quality region. NSGA-II concentrates queries
        near the front and yields a tight reference at a fraction of the
        evals required for random sampling.

        ``evals_dir`` is honored for consistency with the runner — when
        passed, pooled optimizer eval CSVs are used as the reference
        instead of NSGA-II (typically denser near the actual Pareto on
        very high-D problems).
        """
        if evals_dir is not None:
            from .base import pool_evals_pareto
            obj_names = [obj["objective_name"] for obj in self.objectives]
            Y = pool_evals_pareto(evals_dir, obj_names)
            return {n: Y[:, i] for i, n in enumerate(obj_names)}

        # NSGA-II pop_size×n_gen ≈ n_samples ; capped at 4096/200 default.
        pop_size = min(200, max(50, n_samples // 200))
        Y = run_nsga2_pareto(self, n_evals=n_samples, pop_size=pop_size)
        obj_names = [obj["objective_name"] for obj in self.objectives]
        return {n: Y[:, i] for i, n in enumerate(obj_names)}
