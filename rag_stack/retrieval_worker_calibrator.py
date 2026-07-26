"""Standalone calibration for CPU retrieval-worker operators.

The current profile measures the production multi-list RRF implementation and
fits its service demand per user row:

    service_s = per_call_s + per_item_s * (fanout * top_k)

This module lives at the top level of :mod:`rag_stack` so the installed CLI can
import it without depending on repository-only ``scripts/`` or triggering cost
model auto-discovery.  It consumes no RAG end-to-end throughput evidence.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any


K_GRID = (2, 3, 4, 6)
TOPK_GRID = (4, 8, 16, 32, 64)
OVERLAP_GRID = (0.0, 0.5)
REPS = 40
RRF_K = 60


def _make_lists(k: int, top_k: int, overlap: float, seed: int):
    """Build realistic ranked candidate lists for one isolated RRF call."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n_shared = int(round(top_k * overlap))
    shared = [
        f"doc_{item}_c{item % 7}"
        for item in rng.integers(0, 10_000_000, n_shared)
    ]
    ids, scores = [], []
    for _ in range(k):
        own = [
            f"doc_{item}_c{item % 7}"
            for item in rng.integers(0, 10_000_000, top_k - n_shared)
        ]
        ranked_ids = shared + own
        rng.shuffle(ranked_ids)
        ranked_scores = np.sort(
            rng.uniform(0.2, 0.95, len(ranked_ids))
        )[::-1].tolist()
        ids.append(ranked_ids)
        scores.append(ranked_scores)
    return tuple(ids), tuple(scores)


def _profile_document(
    *,
    cpu_key: str,
    config_path: Path,
    rows: list[dict[str, Any]],
    per_call_s: float,
    per_item_s: float,
    fit_mape: float,
) -> dict[str, Any]:
    """Build the strict current output for the production CM family."""
    return {
        "profile_kind": "retrieval_worker",
        "hardware_key": cpu_key,
        "components": {
            "rrf_fusion": {
                "operator": (
                    "rag_stack_evaluator.static_rag_evaluator.nodes."
                    "hybridretrieval.hybrid_rrf.rrf_pure"
                ),
                "service_unit": "seconds_per_user_row",
                "demand_axis": "input_items",
                "fit_form": "nonnegative_affine",
                "equation": (
                    "service_s = per_call_s + per_item_s * "
                    "(fanout * top_k)"
                ),
                "per_call_s": round(float(per_call_s), 8),
                "per_item_s": round(float(per_item_s), 10),
                "fit_mape": round(float(fit_mape), 4),
                "grid": rows,
            }
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag_stack calibrate retrieval-worker"
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "rag_stack YAML whose system block names the CPU "
            "(profile file = retrieval_worker/<cpu_key>.json)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="measure + fit, do not write the profile",
    )
    args = parser.parse_args(argv)

    import numpy as np

    from rag_stack.calibration_paths import atomic_write_json, system_profile_path
    from rag_stack.cost_model.llm_sim.calibration import hardware_from_config
    from rag_stack_evaluator.static_rag_evaluator.nodes.hybridretrieval.hybrid_rrf import (
        rrf_pure,
    )

    config_path = Path(args.config).expanduser().resolve()
    hardware, _ = hardware_from_config(str(config_path))
    cpu_key = str(
        getattr(hardware, "cpu_key", "")
        or getattr(hardware, "cpu_preset", "")
    ).strip()
    if not cpu_key:
        raise ValueError(
            f"{config_path}: system.cpu must declare hardware_key or a preset"
        )

    rows: list[dict[str, Any]] = []
    for k in K_GRID:
        for top_k in TOPK_GRID:
            for overlap in OVERLAP_GRID:
                ids, scores = _make_lists(
                    k, top_k, overlap, seed=k * 1000 + top_k
                )
                for _ in range(5):
                    rrf_pure(ids, scores, RRF_K, top_k)
                timings = []
                for _ in range(REPS):
                    started = time.perf_counter()
                    rrf_pure(ids, scores, RRF_K, top_k)
                    timings.append(time.perf_counter() - started)
                median_s = float(np.median(timings))
                rows.append({
                    "k": k,
                    "top_k": top_k,
                    "overlap": overlap,
                    "items": k * top_k,
                    "seconds": median_s,
                })
                print(
                    f"K={k} top_k={top_k:>3} overlap={overlap:.1f} "
                    f"items={k * top_k:>4} -> {median_s * 1000:7.3f} ms",
                    flush=True,
                )

    items = np.array([row["items"] for row in rows], dtype=float)
    seconds = np.array([row["seconds"] for row in rows], dtype=float)
    design = np.stack([np.ones_like(items), items], axis=1)
    try:
        from scipy.optimize import nnls

        coefficients, _ = nnls(design, seconds)
    except ImportError:
        coefficients, *_ = np.linalg.lstsq(design, seconds, rcond=None)
        coefficients = np.clip(coefficients, 0.0, None)
    per_call_s, per_item_s = map(float, coefficients)
    prediction = design @ coefficients
    absolute_percentage_error = np.abs(prediction - seconds) / seconds
    fit_mape = float(absolute_percentage_error.mean())
    print(
        f"\nfit: per_call_s={per_call_s * 1e3:.4f} ms  "
        f"per_item_s={per_item_s * 1e6:.3f} us/item  "
        f"MAPE={fit_mape:.3f}  "
        f"max APE={absolute_percentage_error.max():.3f}"
    )

    if args.dry_run:
        return 0

    profile = _profile_document(
        cpu_key=cpu_key,
        config_path=config_path,
        rows=rows,
        per_call_s=per_call_s,
        per_item_s=per_item_s,
        fit_mape=fit_mape,
    )
    profile_path = system_profile_path("retrieval_worker", cpu_key)
    atomic_write_json(profile_path, profile)
    print(f"wrote {profile_path} components.rrf_fusion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
