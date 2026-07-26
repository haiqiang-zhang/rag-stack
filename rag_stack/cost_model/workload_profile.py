"""Unified build-on-miss for the per-(corpus×embedding) FAISS workload profiles.

Both FAISS families carry a DATA-side artifact the cost model needs but cannot
derive from config alone:

* IVF — the cell-imbalance factor ``f(nprobe)`` (faiss_ivf_sim/imbalance.py),
  keyed ``<dataset>__<embedding>`` per nlist.
* HNSW — the ndis/nhops workload predictor (faiss_hnsw_sim), keyed the same way.

This module is the SINGLE unified entry that builds BOTH, eagerly at controller
startup, so the cost model starts with the real data-aware factors instead of
degrading (IVF ``f = 1``, HNSW generic default) until an eval happens to build
them. It is ``static_rag_evaluator``-free: the corpus base vectors + a query
sample arrive via ``ctx.vector_provider`` (built in the controller, which may
import SRE), and faiss-index construction happens HERE from those vectors.

Fully best-effort: every step is guarded and never raises into the run. A
failure leaves the missing profile to be built lazily by the eval backend (no
regression). It is also cheap-on-cache-hit and idempotent, so the IVF and HNSW
adapters can both call it at init() without double-paying.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Mirrors static_rag_evaluator._save_imbalance_profiles NPROBES.
_DEFAULT_NPROBE = [1, 4, 16, 64, 128, 256, 512, 1024]

# Process-level dedup so a second init() call (the other FAISS adapter) is a
# pure no-op — avoids re-invoking ``vector_provider`` (which re-encodes the
# query sample / re-loads the embedding model) once a key is handled.
_PROCESSED: set = set()


def workload_corpus_id(config: dict, data_manager: Any) -> str:
    """Return the explicit dataset identity used by workload profiles."""
    return str(
        getattr(data_manager, "dataset_name", None)
        or (config.get("dataset") or {}).get("dataset_name", "")
        or ""
    )


def swept_leaf_values(search_space: Any, leaf: str) -> list[Any]:
    """Return every categorical value whose search-space leaf is ``leaf``."""
    try:
        from rag_stack.search_space import SEP

        parameters = getattr(search_space, "parameters", None)
        if parameters is None:
            parameters = getattr(
                getattr(search_space, "ss", None), "parameters", {}
            )
        out: list[Any] = []
        for name, parameter in (parameters or {}).items():
            if name == leaf or name.endswith(SEP + leaf):
                values = getattr(parameter, "values", None)
                if values:
                    out.extend(list(values))
        return out
    except Exception:  # noqa: BLE001 - metadata introspection is best-effort
        return []


def build_workload_search_space(
    config: dict,
    search_space: Any,
    corpus_data: Any,
) -> dict | None:
    """Enumerate the exact FAISS workload axes reachable from one RAG config."""
    from rag_stack.search_space import algo_vectordb_blocks

    vectordbs = algo_vectordb_blocks(config)
    ivf = [
        value for value in vectordbs
        if isinstance(value, dict) and value.get("db_type") == "faiss_ivf"
    ]
    hnsw = [
        value for value in vectordbs
        if isinstance(value, dict) and value.get("db_type") == "faiss_hnsw"
    ]
    if not (ivf or hnsw):
        return None

    embeddings = {
        str(value) for value in swept_leaf_values(search_space, "embedding_model")
    }
    for vectordb in (*ivf, *hnsw):
        embedding = vectordb.get("embedding_model")
        if isinstance(embedding, str) and embedding:
            embeddings.add(embedding)
    if not embeddings:
        return None

    nprobe = sorted({
        int(value)
        for value in swept_leaf_values(search_space, "nprobe")
        if int(value) > 0
    })
    nlist: set[int] = set()
    for value in swept_leaf_values(search_space, "nlist"):
        try:
            nlist.add(int(value))
        except (TypeError, ValueError):
            continue
    for vectordb in ivf:
        value = vectordb.get("nlist")
        if isinstance(value, (int, float)):
            nlist.add(int(value))

    corpus_size = max(1, int(len(corpus_data))) if corpus_data is not None else 0
    factors = list(swept_leaf_values(search_space, "nlist_factor"))
    for vectordb in ivf:
        value = vectordb.get("nlist_factor")
        if isinstance(value, (int, float)):
            factors.append(value)
    for factor in factors:
        if corpus_size > 0:
            nlist.add(max(
                1,
                min(int(round(float(factor) * (corpus_size ** 0.5))), corpus_size),
            ))

    policies = {
        str(value)
        for vectordb in hnsw
        for value in [vectordb.get("hnsw_workload_profile")]
        if value is not None and str(value).strip()
    }
    if len(policies) > 1:
        raise ValueError(
            "all HNSW search-space blocks must use one workload-profile policy"
        )
    policy = next(iter(policies), None)
    return {
        "embedding_models": sorted(embeddings),
        "nlist": sorted(value for value in nlist if value > 0),
        "nprobe": nprobe,
        **({"hnsw_workload_profile": policy} if policy is not None else {}),
    }


def make_workload_vector_provider(
    *,
    corpus_data: Any,
    qa_data: Any,
    corpus_key: str,
):
    """Build the production embedding-cache-backed vector provider."""
    try:
        contents = corpus_data["contents"].astype(str).tolist()
    except Exception:  # noqa: BLE001 - preserve controller best-effort behavior
        contents = []
    try:
        queries = qa_data["query"].astype(str).tolist()[:512]
    except Exception:  # noqa: BLE001 - preserve controller best-effort behavior
        queries = []
    if not contents:
        return None

    def provider(embedding_model: str):
        try:
            import numpy as np
            from rag_stack_evaluator.static_rag_evaluator.embedding.base import EmbeddingModel
            import rag_stack_evaluator.static_rag_evaluator.embedding_cache as embedding_cache

            lazy = EmbeddingModel.load_from_str(embedding_model)
            embedding = lazy() if callable(lazy) else lazy
            model = getattr(embedding, "_model", None)
            normalize = bool(getattr(embedding, "normalize", True))

            def encode(texts):
                if model is not None and hasattr(model, "encode"):
                    return np.asarray(model.encode(
                        texts,
                        batch_size=256,
                        normalize_embeddings=normalize,
                        show_progress_bar=False,
                    ), dtype="float32")
                return np.asarray(
                    embedding.get_text_embedding_batch(texts), dtype="float32"
                )

            base = embedding_cache.get_or_encode(
                contents,
                lambda: encode(contents),
                dataset_name=corpus_key,
                embedding_id=str(embedding_model),
                normalize=normalize,
            )
            base = np.asarray(base, dtype="float32")
            query = encode(queries) if queries else base[: min(len(base), 2000)]
            return base, query
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[cm] vector_provider failed for %s: %s", embedding_model, exc
            )
            return None

    return provider


def build_workload_profiles(
    *,
    config_path: str | Path,
    project_dir: str | Path,
    qa_data_path: str | Path | None = None,
    corpus_data_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build workload profiles without constructing the optimizer Controller."""
    from rag_stack.cost_model.base import CmInitContext
    from rag_stack.performance_context import PerformanceContext

    context = PerformanceContext(
        config_path=str(config_path),
        project_dir=str(project_dir),
        qa_data_path=str(qa_data_path) if qa_data_path is not None else None,
        corpus_data_path=(
            str(corpus_data_path) if corpus_data_path is not None else None
        ),
        validate=True,
        check_model_cache=False,
    )
    corpus_key = workload_corpus_id(context.config, context._data)
    search_space = build_workload_search_space(
        context.config,
        context.search_space,
        context._data.corpus_data,
    ) or {}
    if not search_space:
        raise ValueError("full config has no reachable FAISS workload")
    vector_provider = make_workload_vector_provider(
        corpus_data=context._data.corpus_data,
        qa_data=context._data.qa_data,
        corpus_key=corpus_key,
    )
    if vector_provider is None:
        raise ValueError("could not construct the corpus vector provider")
    ensure_workload_profiles(CmInitContext(
        config=context.config,
        corpus_key=corpus_key,
        search_space=search_space,
        vector_provider=vector_provider,
    ))
    return {
        "corpus_key": corpus_key,
        "embedding_models": list(search_space["embedding_models"]),
        "project_dir": str(Path(project_dir).expanduser().resolve()),
    }


def _build_ivf_index(base, nlist):
    """Build a trained IVF-Flat index over ``base`` at ``nlist`` (L2, flat
    quantizer) — the standard faiss recipe used by faiss_ivf_sim.faiss_benchmark
    (``index_factory`` so the index OWNS its coarse quantizer, no dangling ref).
    """
    import faiss
    import numpy as np

    x = np.ascontiguousarray(np.asarray(base, dtype="float32"))
    d = int(x.shape[1])
    index = faiss.index_factory(d, f"IVF{int(nlist)},Flat", faiss.METRIC_L2)
    index.train(x)
    index.add(x)
    return index


def _ivf_imbalance_anchor_nlists(n_vectors: int) -> list[int]:
    """Sparse corpus calibration policy: build one canonical IVF anchor only."""
    from rag_stack.cost_model.faiss_ivf_sim.imbalance import imbalance_anchor_nlist

    return [imbalance_anchor_nlist(n_vectors)]


def ensure_workload_profiles(ctx) -> None:
    """Build (on miss) the IVF + HNSW workload profiles for every swept embedding.

    Reads ``ctx.vector_provider``, ``ctx.corpus_key`` (dataset id) and
    ``ctx.search_space`` (``embedding_models`` / ``nlist`` / ``nprobe``). Best
    effort: returns silently when anything required is missing or fails.
    """
    try:
        vector_provider = getattr(ctx, "vector_provider", None)
        if vector_provider is None:
            return
        corpus_key = str(getattr(ctx, "corpus_key", "") or "")
        ss = getattr(ctx, "search_space", None) or {}
        embeddings = [e for e in (ss.get("embedding_models") or []) if e]
        if not corpus_key or not embeddings:
            return
        nprobe_values = sorted({int(p) for p in (ss.get("nprobe") or _DEFAULT_NPROBE)
                                if int(p) > 0})
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cm] workload-profile init skipped: %s", exc)
        return

    for emb in dict.fromkeys(str(e) for e in embeddings):
        tag = (corpus_key, emb)
        if tag in _PROCESSED:
            continue
        try:
            _ensure_for_embedding(corpus_key, emb, vector_provider,
                                  nprobe_values)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[cm] workload profile %s/%s skipped: %s",
                           corpus_key, emb, exc)
        _PROCESSED.add(tag)


def _ensure_for_embedding(corpus_key, emb, vector_provider, nprobe_values) -> None:
    import numpy as np

    pair = vector_provider(emb)
    if not pair:
        logger.info("[cm] no vectors for embedding %s — skipping workload profile", emb)
        return
    base, query = pair
    base = np.asarray(base, dtype="float32")
    if base.ndim != 2 or len(base) == 0:
        return
    if query is None or len(np.asarray(query)) == 0:
        query = base[: min(len(base), 2000)]
    query = np.ascontiguousarray(np.asarray(query, dtype="float32"))
    n_base = int(len(base))

    # ── IVF cell-imbalance (one curve per nlist; build-on-miss, instant on hit) ──
    try:
        from rag_stack.cost_model.faiss_ivf_sim.imbalance import (
            imbalance_key, get_or_build_imbalance,
        )
        key = imbalance_key(corpus_key, emb)
        for nlist in _ivf_imbalance_anchor_nlists(n_base):
            nl = min(int(nlist), n_base)   # nlist > N is untrainable in faiss
            if nl < 1:
                continue
            nps = [p for p in nprobe_values if p <= nl]
            if not nps:
                continue
            get_or_build_imbalance(
                key, nl, nps,
                build_index=lambda nl=nl: _build_ivf_index(base, nl),
                queries=query,
            )
            logger.info("[cm] IVF imbalance ready: %s (nlist=%d)", key, nl)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cm] IVF imbalance build skipped (%s/%s): %s",
                       corpus_key, emb, exc)

    # ── HNSW workload predictor (per-dataset GBR; build-on-miss) ──
    try:
        from rag_stack.cost_model.faiss_hnsw_sim.train import get_or_build_predictor
        get_or_build_predictor(dataset=corpus_key, embedding=emb, base_vecs=base)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cm] HNSW predictor build skipped (%s/%s): %s",
                       corpus_key, emb, exc)
