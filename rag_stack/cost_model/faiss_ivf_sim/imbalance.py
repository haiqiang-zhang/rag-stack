"""Cell-imbalance: the data-aware IVF scan-count factor, from index geometry.

The number of vectors an IVF query scans is NOT the balanced
``nprobe·N/nlist``: real (clustered) embeddings put a query's nearest cells
above average size, so it scans more. The cell-imbalance factor ``f`` corrects
this::

    N_scan = f(nprobe) · (nprobe/nlist) · N .

``f`` is a pure DATA property of the (corpus, embedding, nlist) triple — read
from the built index's list sizes plus a query sample's coarse assignments,
NEVER from a latency measurement. It is the only per-dataset input the IVF cost
model needs; the host-side η coefficients stay global. ``f(1)`` is the
query~data expected cell size ``nlist·Σsᵢ²/N²``; it decays toward 1.0 as
``nprobe → nlist`` (a full scan reads every vector exactly once → ``f ≡ 1``).

This module is the SINGLE SOURCE OF TRUTH: the sim benchmark and (later) the
production cost-model path call these functions so they compute ``f``
identically from whatever IVF index is already built — no pre-stored,
nlist-keyed profile and no separate index sweep are needed, because the factor
rides on the index that retrieval/quality evaluation builds anyway.
"""
from __future__ import annotations

import json
import os

import faiss
import numpy as np

from rag_stack.calibration_paths import atomic_write_json, calibration_profile_root
from rag_stack.ivf_imbalance_profile_schema import (
    unwrap_ivf_imbalance_profile,
    wrap_ivf_imbalance_profile,
)

# Canonical cache for imbalance profiles: a DATA-side artifact (per corpus ×
# embedding), so it lives next to the host-side η profiles but in its own
# subfolder and is REUSED across hardware (system migration never rebuilds it).
IMBALANCE_DIR = str(calibration_profile_root() / "faiss_ivf" / "imbalance")


def _ivf_list_sizes(index):
    """Return ``(ivf, sizes, N)`` for a faiss index.

    The caller MUST keep a reference to ``index`` for the lifetime of the
    returned ``ivf`` handle: ``extract_index_ivf`` returns a view into the
    parent, and letting the parent be garbage-collected dangles it (SWIG).
    """
    ivf = faiss.extract_index_ivf(index)
    sizes = np.array([ivf.invlists.list_size(i) for i in range(ivf.nlist)],
                     dtype=np.float64)
    return ivf, sizes, float(sizes.sum())


def cell_imbalance_np1(index) -> float:
    """nprobe=1 imbalance constant ``nlist·Σ(sᵢ²)/N²`` (query~data factor).

    1.0 = perfectly balanced cells; >1 = clustered. Cheap: list sizes only, no
    query scan. This is the high-water mark of ``f``; prefer
    :func:`compute_cell_imbalance` for the per-nprobe curve.
    """
    ivf, sizes, N = _ivf_list_sizes(index)
    return float(ivf.nlist * np.sum(sizes ** 2) / N ** 2) if N > 0 else 1.0


def compute_cell_imbalance(
    index,
    queries,
    nprobe_values,
    n_query_sample: int = 2000,
) -> dict[int, float]:
    """Per-nprobe imbalance ``f(nprobe) = mean_scanned / (nprobe·N/nlist)``.

    Computed from real query→centroid assignments on a sample of ``queries``
    (the corpus's own queries, or any representative query set at deploy time).
    The constant :func:`cell_imbalance_np1` over-counts at high nprobe
    (e.g. gist960 Flat np1024: +62%); this returns the exact per-nprobe form,
    which equals 1.0 at ``nprobe = nlist`` by construction.

    Args:
        index: a trained faiss IVF index (the full index; kept alive by caller).
        queries: ``(nq, d)`` float32-able array of query vectors.
        nprobe_values: nprobe values to compute ``f`` for.
        n_query_sample: cap on queries used — a sample suffices and keeps it cheap.

    Returns:
        ``{nprobe: f}`` for every requested nprobe.
    """
    ivf, sizes, N = _ivf_list_sizes(index)
    nlist = ivf.nlist
    nps = [int(p) for p in nprobe_values]
    if N <= 0:
        return {p: 1.0 for p in nps}
    q = np.ascontiguousarray(np.asarray(queries)[:n_query_sample], dtype="float32")
    pmax = min(max(nps), nlist)
    # pmax nearest centroids per query; slice [:p] for each requested nprobe.
    _, assign = ivf.quantizer.search(q, pmax)
    out: dict[int, float] = {}
    for p in nps:
        pp = min(p, nlist)
        scanned = sizes[assign[:, :pp]].sum(axis=1).mean()
        balanced = pp * N / nlist
        out[p] = float(scanned / balanced) if balanced > 0 else 1.0
    return out


# ── Imbalance profiles (cached data-side artifact, for the CM-init path) ──────
#
# A profile is one JSON per (corpus × embedding) key, holding a sparse set of
# imbalance curves. Corpus calibration should stay cheap: one canonical nlist is
# enough for e2e CM validation, and nearby nlist values use the nearest cached
# curve instead of forcing a fresh IVF train.
#
#   {"d": 960, "N": 1000000,
#    "nlist": {"1024": {"1": 1.578, "128": 1.518, ...}, "4096": {...}}}
#
# The factor depends on (corpus, embedding, nlist) — NOT on hardware, code type,
# or nprobe-beyond-the-curve — so one profile is reused across hosts (migration)
# and across IVF-Flat / IVF-PQ / FastScan (they share the coarse quantizer).

import re as _re

DEFAULT_IMBALANCE_NLIST_FACTOR = 4.0


def imbalance_anchor_nlist(
    n_vectors: int,
    factor: float = DEFAULT_IMBALANCE_NLIST_FACTOR,
) -> int:
    """Canonical sparse-calibration IVF nlist for a corpus.

    Matches the e2e benchmark's default ``nlist_factor=4`` policy. The cost model
    may evaluate other nlist values, but those should reuse this anchor's
    imbalance curve rather than rebuilding one IVF index per nlist.
    """
    n = max(1, int(n_vectors or 1))
    return max(1, min(int(round(float(factor) * (n ** 0.5))), n))


def _sani(s) -> str:
    """Sanitize a name for a filename, keeping it readable (alnum + underscore)."""
    return _re.sub(r"[^A-Za-z0-9_]+", "-", str(s)).strip("-_") or "x"


def imbalance_key(dataset_name, embedding_model) -> str:
    """Shared imbalance-profile key ``<dataset_name>__<embedding>`` — call this on
    BOTH the load side (cost model / controller) and the save side (eval backend)
    so the cache filename always matches.

    ``dataset_name`` is the EXPLICIT, mandatory ``dataset.dataset_name`` from the
    run config (NOT derived from a file path — paths are unreliable: the
    in-project corpus copy has a fixed name, and absolute paths differ across
    machines). The key is (dataset × embedding) ONLY — deliberately NOT chunk
    settings and NOT nlist:

    * **No chunk settings.** RAG-Stack sweeps chunk method/size/overlap; keying
      by the chunk hash would force one index build (re-chunk + re-embed the
      whole corpus) per setting — impractical. The cell-imbalance is a coarse
      clustering property dominated by corpus semantics + embedding and is
      ~invariant to chunk granularity, so ONE profile is reused across chunks.
    * **Sparse nlist.** nlist lives INSIDE the profile, but validation builds
      only the canonical ``round(4·sqrt(N))`` anchor by default. Other nlist
      values reuse the nearest cached curve.
    """
    return f"{_sani(dataset_name)}__{_sani(embedding_model)}"


def imbalance_profile_path(key: str, directory: str = IMBALANCE_DIR) -> str:
    """Canonical path for the imbalance profile of corpus×embedding ``key``."""
    return os.path.join(directory, f"{key}.json")


def load_imbalance_profile(key: str, directory: str = IMBALANCE_DIR) -> dict | None:
    """Load a cached payload, or ``None`` if it does not exist yet.

    The disk document is the strict current envelope.  Callers retain
    the historical payload-only API.
    """
    path = imbalance_profile_path(key, directory)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as stream:
        document = json.load(stream)
    return unwrap_ivf_imbalance_profile(document, source=path)


def _the_curve(profile: dict) -> tuple[int, dict] | None:
    """THE single per-(corpus × embedding) imbalance curve.

    Policy: ONE curve per profile — the imbalance model is a property of the
    corpus content × embedding, deliberately chunk-/nlist-agnostic (rebuilding
    per chunk config or per nlist would defeat the cost model's purpose).
    Legacy multi-curve files (pre-collapse chunk-drift artifacts) resolve to
    the RICHEST curve (most nprobe points; tie → largest nlist basis).
    """
    curves = profile.get("nlist") or {}
    if not curves:
        return None
    nl = max(curves, key=lambda k: (len(curves[k]), int(k)))
    return int(nl), curves[nl]


def _curve_lookup(curve_nlist: int, curve: dict, nlist: int, nprobe: int) -> float:
    """Coverage-matched lookup: a request at a different ``nlist`` maps to the
    curve point with the SAME probed fraction (nprobe/nlist) — the first-order
    invariant of cell imbalance — then nearest available nprobe."""
    eff = float(nprobe) * float(curve_nlist) / max(1.0, float(nlist))
    nearest = min(curve, key=lambda p: abs(float(p) - eff))
    return float(curve[nearest])


def profile_cell_imbalance(
    dataset_name: str,
    embedding_model: str,
    nlist: int,
    nprobe: int,
    *,
    n_vectors: int = 0,
    profile_cache: dict | None = None,
    directory: str = IMBALANCE_DIR,
) -> float:
    """Read the IVF cell-imbalance factor from the managed calibration profile.

    Shared load-side policy for both production/e2e CM and the IVF
    microbenchmark: an EXACT (nlist, nprobe) measurement wins when present;
    everything else is served from THE single corpus×embedding curve via
    coverage-matched nprobe (see :func:`_the_curve`). ``n_vectors`` is
    accepted for call-site compatibility but no longer selects a curve — the
    single-curve policy is chunk-count-agnostic by design.
    """
    if not dataset_name:
        return 1.0
    key = imbalance_key(dataset_name, embedding_model or "")
    if profile_cache is not None:
        profile = profile_cache.get(key)
        if profile is None:
            profile = load_imbalance_profile(key, directory) or {}
            profile_cache[key] = profile
    else:
        profile = load_imbalance_profile(key, directory) or {}

    nlists = profile.get("nlist") or {}
    exact = nlists.get(str(int(nlist)), {}).get(str(int(nprobe)))
    if exact is not None:
        return float(exact)
    picked = _the_curve(profile)
    if picked is None:
        return 1.0
    curve_nlist, curve = picked
    return _curve_lookup(curve_nlist, curve, int(nlist), int(nprobe))


def save_imbalance_profile(key: str, profile: dict, directory: str = IMBALANCE_DIR) -> str:
    """Atomically write a payload in the strict current envelope."""
    path = imbalance_profile_path(key, directory)
    atomic_write_json(path, wrap_ivf_imbalance_profile(profile))
    return path


def get_or_build_imbalance(
    key: str,
    nlist: int,
    nprobe_values,
    *,
    build_index,
    queries,
    n_query_sample: int = 2000,
    directory: str = IMBALANCE_DIR,
) -> dict[int, float]:
    """Return ``{nprobe: f}`` for (``key``, ``nlist``) under the single-curve policy.

    This is the CM-init entry point. A non-empty profile is ALWAYS a hit
    (served from THE corpus×embedding curve — no index, no growth). Only an
    empty profile pays the one-time cost: ``build_index()`` (the slow step) →
    :func:`compute_cell_imbalance` → persist as THE curve.

    Args:
        key: corpus×embedding identifier (e.g. ``"dragonball_en__bge-small__"``).
        nlist: the IVF coarse-list count the imbalance is being requested for
            (only honored verbatim on the FIRST fill — afterwards THE single
            corpus×embedding curve serves all nlist values, coverage-matched).
        nprobe_values: nprobe values needed.
        build_index: zero-arg callable returning the trained faiss IVF index;
            called ONLY when the profile is empty (one build per
            corpus×embedding, ever).
        queries: query sample array, or a zero-arg callable returning one.
        n_query_sample: cap on queries used for the coarse-assignment sample.
    """
    profile = load_imbalance_profile(key, directory) or {}
    nl = str(int(nlist))
    need = [int(p) for p in nprobe_values]
    picked = _the_curve(profile)
    if picked is not None:
        # Single-curve policy: a non-empty profile NEVER grows another curve
        # (the model is corpus×embedding, chunk-/nlist-agnostic). Serve every
        # request from THE curve — exact where measured, coverage-matched
        # otherwise. No index is ever built here again.
        curve_nlist, curve = picked
        cached = profile.get("nlist", {}).get(nl, {})
        return {
            p: float(cached[str(p)]) if str(p) in cached
            else _curve_lookup(curve_nlist, curve, int(nlist), p)
            for p in need
        }

    index = build_index()                                 # first fill: the one-time slow step
    q = queries() if callable(queries) else queries
    f = compute_cell_imbalance(index, q, need, n_query_sample)
    ivf, _, N = _ivf_list_sizes(index)
    profile.setdefault("d", int(ivf.d))
    profile.setdefault("N", int(N))
    profile.setdefault("nlist", {}).setdefault(nl, {}).update(
        {str(p): v for p, v in f.items()})
    save_imbalance_profile(key, profile, directory)
    return f
