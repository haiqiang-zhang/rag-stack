"""ML workload predictor for HNSW search.

Predicts hardware-agnostic workload profile (ndis, nhops) from
index configuration and graph statistics. The roofline model then
maps this to latency on specific hardware.

When no trained model is available, falls back to analytical estimates.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .hnsw_config import HNSWConfig
from .graph_stats import HNSWGraphStats, estimate_graph_stats

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parent / "models"
_DEFAULT_MODEL_PATH = _MODELS_DIR / "default.joblib"

# Feature names in the order expected by the ML model
# Index-free feature set: a-priori config + corpus LID only. The index-derived
# degree/layer stats were dropped — they help only when REAL (GT points, where we
# instead harvest the measured workload) and POISON the model when guessed at
# cost-model-only points (degree guess is 1.5-1.9x off). See HNSW_STATUS.
FEATURE_NAMES = [
    "log_N", "d", "M", "ef_search",
    "lid",  # Local Intrinsic Dimensionality (Amsaleg et al. 2015), from corpus
]


@dataclass
class WorkloadProfile:
    """Predicted workload for a single HNSW query."""

    ndis_per_query: float   # Distance computations per query
    nhops_per_query: float  # Node expansions (graph hops) per query


def _extract_features(
    config: HNSWConfig,
    graph_stats: HNSWGraphStats,
) -> np.ndarray:
    """Build the feature vector for the ML model."""
    # Fall back to ambient dim when LID is unknown (matches uniform-random case)
    lid = graph_stats.lid
    if not math.isfinite(lid):
        lid = float(config.d)
    return np.array([
        math.log(config.N) if config.N > 0 else 0.0,
        float(config.d),
        float(config.M),
        float(config.ef_search),
        lid,
    ], dtype=np.float64)


def _analytical_fallback(
    config: HNSWConfig,
    graph_stats: HNSWGraphStats,
) -> WorkloadProfile:
    """Analytical estimate when no trained model is available.

    Simple heuristic based on HNSW paper analysis:
    - ndis ~ ef_search * avg_degree * (1 - skip_rate)
    - nhops ~ ef_search (each candidate expansion = one hop)
    """
    avg_degree = graph_stats.actual_avg_degree
    skip_rate = 0.15  # Fraction of neighbors skipped by visited set
    ndis = config.ef_search * avg_degree * (1.0 - skip_rate)
    nhops = float(config.ef_search)
    return WorkloadProfile(ndis_per_query=ndis, nhops_per_query=nhops)


class WorkloadPredictor:
    """Predicts HNSW search workload from configuration + graph stats.

    Uses two GradientBoostingRegressor models (one for ndis, one for nhops).
    Falls back to analytical estimates when no trained model is loaded.
    """

    def __init__(self):
        self._ndis_model = None
        self._nhops_model = None

    @property
    def is_trained(self) -> bool:
        return self._ndis_model is not None and self._nhops_model is not None

    def predict(
        self,
        config: HNSWConfig,
        graph_stats: Optional[HNSWGraphStats] = None,
    ) -> WorkloadProfile:
        """Predict workload profile for a given configuration.

        Args:
            config: HNSW index/search configuration.
            graph_stats: Graph statistics. If None, uses analytical estimates.
        """
        if graph_stats is None:
            graph_stats = estimate_graph_stats(
                config.N, config.d, config.M, config.ef_construction,
            )

        if not self.is_trained:
            return _analytical_fallback(config, graph_stats)

        X = _extract_features(config, graph_stats).reshape(1, -1)
        ndis = float(self._ndis_model.predict(X)[0])
        nhops = float(self._nhops_model.predict(X)[0])

        # Clamp to reasonable minimums
        ndis = max(ndis, config.ef_search)
        nhops = max(nhops, 1.0)

        return WorkloadProfile(ndis_per_query=ndis, nhops_per_query=nhops)

    def train(self, data) -> dict:
        """Train from a DataFrame with columns matching data_collector.COLUMNS.

        Args:
            data: pandas DataFrame with training samples.

        Returns:
            Dict with training metrics (r2_ndis, r2_nhops).
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score

        feature_cols = ["N", "d", "M", "ef_search", "lid"]
        X = data[feature_cols].copy()
        X["N"] = np.log(X["N"].clip(lower=1))
        # Fill missing LID with ambient dimension (uniform-random assumption)
        X["lid"] = X["lid"].fillna(X["d"])
        X = X.rename(columns={"N": "log_N"}).values

        y_ndis = data["ndis_per_query"].values
        y_nhops = data["nhops_per_query"].values

        self._ndis_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=42,
        )
        self._nhops_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=42,
        )

        self._ndis_model.fit(X, y_ndis)
        self._nhops_model.fit(X, y_nhops)

        # Cross-validation scores
        cv_ndis = cross_val_score(
            GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                min_samples_leaf=5, random_state=42,
            ),
            X, y_ndis, cv=5, scoring="r2",
        )
        cv_nhops = cross_val_score(
            GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                min_samples_leaf=5, random_state=42,
            ),
            X, y_nhops, cv=5, scoring="r2",
        )

        metrics = {
            "r2_ndis": float(cv_ndis.mean()),
            "r2_nhops": float(cv_nhops.mean()),
            "n_samples": len(data),
        }
        logger.info(
            f"Trained on {len(data)} samples: "
            f"R2_ndis={metrics['r2_ndis']:.4f}, "
            f"R2_nhops={metrics['r2_nhops']:.4f}"
        )
        return metrics

    def save(self, path: Optional[str] = None) -> None:
        """Save trained models to disk."""
        import joblib

        if not self.is_trained:
            raise RuntimeError("No trained models to save")

        save_path = Path(path) if path else _DEFAULT_MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {"ndis_model": self._ndis_model, "nhops_model": self._nhops_model},
            save_path,
        )
        logger.info(f"Saved models to {save_path}")

    @classmethod
    def load(cls, path: str) -> WorkloadPredictor:
        """Load pre-trained models from a .joblib file."""
        import joblib

        predictor = cls()
        bundle = joblib.load(path)
        predictor._ndis_model = bundle["ndis_model"]
        predictor._nhops_model = bundle["nhops_model"]
        logger.info(f"Loaded models from {path}")
        return predictor

    @classmethod
    def load_default(cls) -> WorkloadPredictor:
        """Load the default shipped model, or return untrained (fallback mode)."""
        predictor = cls()
        if _DEFAULT_MODEL_PATH.exists():
            try:
                return cls.load(str(_DEFAULT_MODEL_PATH))
            except Exception as e:
                logger.warning(f"Failed to load default model: {e}, using fallback")
        return predictor

    @classmethod
    def load_for_dataset(cls, dataset_name: str, embedding_model: str):
        """Load the PER-DATASET trained predictor — the "trained dataset" half of
        the split profile: calibration_profiles/faiss_hnsw/<dataset>__<embedding>.joblib.
        Returns None if absent (caller falls back to :meth:`load_default`)."""
        from pathlib import Path as _P
        from .coefficients import dataset_predictor_key, dataset_predictor_path
        if not (dataset_name and embedding_model):
            return None
        candidates = [(dataset_name, embedding_model)]
        candidates.extend({
            ("sift1m", "raw"): [("sift1M", "raw")],
            ("gist1m", "raw"): [("gist1M", "raw")],
            ("glove", "glove"): [("glove-100-angular", "glove")],
        }.get((dataset_name, embedding_model), []))
        for ds, emb in candidates:
            path = dataset_predictor_path(dataset_predictor_key(ds, emb))
            if _P(path).exists():
                try:
                    return cls.load(path)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to load dataset predictor {path}: {e}")
        return None
