"""Cross-optimizer utilities — non-SearchSpace concerns only.

SearchSpace operational helpers (decode / clamp / snap / fixed_features /
``__`` convention parsing / ...) live in:

  * :mod:`rag_stack.search_space.ax_search_space_ops` — generic Ax adapter
    (any SearchSpace; no naming convention assumed)
  * :mod:`rag_stack.search_space.rag_search_space` — :class:`RagSearchSpace`
    (rag-stack ``__`` three-segment naming convention)

This module keeps only the items that are SearchSpace-independent: torch
device/dtype routing, RNG seeding, dict signatures / hashes, dominance
mask, hashable param tuple, CSV exporter.
"""

import csv
import hashlib
import json
import logging
import os
import random
from typing import Dict, List, Optional, Set

import numpy as np
import torch

logger = logging.getLogger("RAG-Stack")


# ---------------------------------------------------------------------------
# Torch device/dtype routing
# ---------------------------------------------------------------------------

def default_tkwargs(
    tkwargs: Optional[Dict] = None,
    *,
    prefer_gpu: bool = True,
) -> Dict:
    """Resolve a torch device + dtype dict (the BoTorch ``tkwargs`` convention).

    Single source of truth for "which device do I run on" across optimizers.
    Behavior:

      * If ``tkwargs`` is ``None`` (the common case): auto-detect.
        - GPU is selected when ``torch.cuda.is_available()`` AND ``prefer_gpu``.
        - Otherwise falls back to ``cpu``.
        - ``dtype`` defaults to ``torch.double`` (BoTorch convention).
      * If ``tkwargs`` is given: normalize it.
        - String ``"cuda"`` / ``"cpu"`` is converted to ``torch.device``.
        - Missing ``dtype`` is filled with ``torch.double``.
        - User-passed ``"cuda"`` is ALSO downgraded to ``"cpu"`` if the
          machine has no GPU (rather than letting torch raise downstream).

    The output is suitable for ``torch.tensor(..., **tkwargs)`` and the
    BoTorch tutorial idiom.
    """
    out: Dict = dict(tkwargs) if tkwargs else {}
    out.setdefault("dtype", torch.double)
    desired = out.get("device")
    cuda_ok = torch.cuda.is_available()
    if desired is None:
        out["device"] = torch.device("cuda" if (prefer_gpu and cuda_ok) else "cpu")
    else:
        if isinstance(desired, str):
            desired = torch.device(desired)
        # If user asked for cuda but CUDA isn't available, fall back to cpu
        # rather than raising a confusing error deep in torch internals.
        if desired.type == "cuda" and not cuda_ok:
            desired = torch.device("cpu")
        out["device"] = desired
    return out


# ---------------------------------------------------------------------------
# RNG seeding
# ---------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    """Seed Python / numpy / torch RNGs from a single integer seed.

    We do NOT enable ``torch.use_deterministic_algorithms`` or set a fixed
    cuBLAS workspace: CUDA's natural float-summation non-determinism is
    actually load-bearing here. With strict determinism a stuck plateau
    (GP overconfident in a wrong region; qLogNEHVI surface flat over the
    true Pareto improvement direction) is a permanent absorbing state.
    Empirically v5_csv-gen runs (un-patched) had ~22% stuck-at-iter-100
    rate vs ~50% under strict determinism — the float-order jitter occasionally
    perturbs the GP just enough to escape the plateau. Same-seed reproducibility
    is sacrificed for stuck-recovery probability.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config dict utilities
# ---------------------------------------------------------------------------

def config_signature(config_dict: dict) -> tuple:
    """Create a hashable signature from a config dict for dedup.

    Recursive so deeply nested RAG configs (e.g. ``pipeline.sequential.
    generator.{model, temperature, ...}``) flatten cleanly — the previous
    2-level flattening left dict values at level 2, producing tuples with
    unhashable dict elements (TypeError at insert time).  Lists are
    converted to tuples; non-hashable atoms (rare) fall back to ``repr``.
    """
    def _norm(v):
        if isinstance(v, dict):
            return tuple((k, _norm(v[k])) for k in sorted(v.keys()))
        if isinstance(v, (list, tuple)):
            return tuple(_norm(x) for x in v)
        try:
            hash(v)
            return v
        except TypeError:
            return repr(v)
    return tuple((k, _norm(config_dict[k])) for k in sorted(config_dict.keys()))


def config_hash(config_dict: dict) -> str:
    """Deterministic hex digest from a decoded config dict.

    Uses ``config_signature`` to produce a stable hashable representation,
    then returns a 16-char hex string suitable for filenames and dedup keys.
    """
    sig = config_signature(config_dict)
    return hashlib.sha256(
        json.dumps(sig, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def is_duplicate(new_config: dict, existing_configs: List[dict]) -> bool:
    """Check if new_config matches any existing config."""
    new_sig = config_signature(new_config)
    return any(config_signature(c) == new_sig for c in existing_configs)


def params_to_tuple(params_dict: dict, active_keys: Optional[Set[str]] = None) -> tuple:
    """Convert decoded params dict to hashable tuple for deduplication."""
    keys = sorted(active_keys) if active_keys else sorted(params_dict.keys())
    return tuple(params_dict.get(k) for k in keys)


# ---------------------------------------------------------------------------
# Dominance mask
# ---------------------------------------------------------------------------

def is_non_dominated_np(Y: np.ndarray) -> np.ndarray:
    """Return a boolean mask of non-dominated points (maximization)."""
    n = Y.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            if np.all(Y[j] >= Y[i]) and np.any(Y[j] > Y[i]):
                mask[i] = False
                break
    return mask


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def append_evaluation_to_csv(
    eval_entry: Dict,
    csv_path: str,
    obj_columns: List[str],
):
    """Upsert a single evaluation entry as a row in the CSV.

    eval_id MUST be unique in the CSV. If a row with the same eval_id
    already exists, this function REPLACES it (last-write-wins) rather
    than appending. This is defense-in-depth against upstream paths that
    might re-evaluate an eval_id (e.g. a resume whose counter rewound
    onto a CSV-occupied id, or a retry that fires the save twice). The
    controller's monotonic counter should normally prevent the collision
    in the first place; this is the fail-safe.

    Args:
        eval_entry: Dict with keys ``config``, ``objectives``, ``eval_type``,
            ``eval_id``.
        csv_path: Path to the CSV file (created if missing).
        obj_columns: Pre-built objective column names, already expanded.
            E.g. ["performance", "quality__combined", "quality__deepeval_faithfulness", ...]
    """
    cfg = eval_entry["config"]
    objs = eval_entry.get("objectives", {})
    eval_type = eval_entry.get("eval_type", "")
    eval_error = eval_entry.get("eval_error", "")
    eval_id = eval_entry.get("eval_id", "")

    # Build parameter columns from this config
    param_keys: list = []
    for stage, params in cfg.items():
        if isinstance(params, dict):
            for k in params:
                col = f"{stage}__{k}"
                if col not in param_keys:
                    param_keys.append(col)

    file_exists = os.path.exists(csv_path)

    # Establish header + load existing rows (if any) into memory. We rewrite
    # the whole file at the end so we can drop any prior row with this eval_id
    # and append the fresh one. For typical projects (≤ a few thousand evals)
    # the I/O cost is negligible.
    existing_rows: list = []
    if file_exists:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
            existing_rows = list(reader)
        extra_cols = [c for c in param_keys if c not in existing_header]
        # Backfill the eval_error column onto older CSVs that predate it.
        if "eval_error" not in existing_header and "eval_error" not in extra_cols:
            extra_cols = extra_cols + ["eval_error"]
        header = existing_header + extra_cols if extra_cols else existing_header
    else:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        header = param_keys + obj_columns + ["eval_type", "eval_error", "eval_id"]

    # Build the new row — flatten dict objectives into sub-columns
    row_dict: Dict[str, str] = {}
    for col in param_keys:
        parts = col.split("__", 1)
        val = cfg.get(parts[0], {}).get(parts[1], "") if len(parts) == 2 else ""
        row_dict[col] = str(val)
    for name, val in objs.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                col = f"{name}__{sub_key}"
                row_dict[col] = f"{sub_val:.6f}" if isinstance(sub_val, (int, float)) else str(sub_val)
        elif isinstance(val, (int, float)):
            row_dict[name] = f"{val:.6f}"
        else:
            row_dict[name] = str(val)
    row_dict["eval_type"] = str(eval_type)
    row_dict["eval_error"] = str(eval_error)
    row_dict["eval_id"] = str(eval_id)
    new_row = [row_dict.get(col, "") for col in header]

    # Drop any pre-existing row(s) with this eval_id.
    eval_id_str = str(eval_id)
    eval_id_col = header.index("eval_id") if "eval_id" in header else -1
    kept_rows: list = []
    replaced = 0
    for row in existing_rows:
        if eval_id_col >= 0 and eval_id_col < len(row) and row[eval_id_col] == eval_id_str:
            replaced += 1
            continue
        # Pad to current header width if a column was just added.
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        kept_rows.append(row)
    if replaced:
        # Loud warning so the upstream collision is visible in the log.
        import logging
        logging.getLogger("RAG-Stack").warning(
            f"append_evaluation_to_csv: replaced {replaced} existing row(s) "
            f"for eval_id={eval_id} (collision — should not happen under a "
            f"monotonic counter)."
        )

    # Atomic-ish rewrite: write to temp file then rename.
    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in kept_rows:
            writer.writerow(row)
        writer.writerow(new_row)
    os.replace(tmp_path, csv_path)
