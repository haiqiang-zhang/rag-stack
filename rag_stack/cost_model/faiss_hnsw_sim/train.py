"""CLI: collect training data and train HNSW workload predictor.

Usage:
    python -m rag_stack.cost_model.faiss_hnsw_sim.train [OPTIONS]

Examples:
    # Train on synthetic data only (no SIFT download needed)
    python -m rag_stack.cost_model.faiss_hnsw_sim.train --synthetic-only

    # Full sweep with custom output
    python -m rag_stack.cost_model.faiss_hnsw_sim.train \
        --output models/v2.joblib \
        --N 10000 50000 100000
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .data_collector import (
    collect_training_data,
    generate_synthetic_datasets,
    load_sift_learn,
)
from .workload_predictor import WorkloadPredictor

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT = Path(__file__).parent / "models" / "default.joblib"


@contextmanager
def _predictor_build_lock(path: str):
    """Serialize one dataset predictor build across shared-NFS clients."""
    lock_path = f"{path}.build.lock"
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _valid_predictor(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return False
    try:
        WorkloadPredictor.load(path)
        return True
    except Exception as exc:  # noqa: BLE001 - corrupted cache is a rebuild miss
        logger.warning("Ignoring incomplete HNSW predictor %s: %s", path, exc)
        return False


def _atomic_save_predictor(predictor: WorkloadPredictor, path: str) -> None:
    """Publish a joblib predictor with a same-directory atomic rename."""
    token = f"tmp.{os.getpid()}.{uuid.uuid4().hex}"
    tmp = f"{path}.{token}"
    try:
        predictor.save(tmp)
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def train_from_vectors(
    base_vecs,
    dataset_name: str,
    embedding: str,
    *,
    max_n: int = 200_000,
    M_values: list[int] | None = None,
    ef_search_values: list[int] | None = None,
    ef_construction_values: list[int] | None = None,
    n_queries: int = 1000,
    output: str | None = None,
) -> tuple[str, dict]:
    """Train the per-dataset HNSW workload predictor from in-memory base vectors.

    The shared core of BOTH the ``--dataset-hdf5`` CLI branch and the build-on-miss
    :func:`get_or_build_predictor`: subsample to ``max_n``, sweep HNSW configs to
    measure ndis/nhops (collect_training_data), fit the GBR, and save the predictor
    to the per-(dataset, embedding) profile path. Returns ``(saved_path, metrics)``.
    """
    from .coefficients import dataset_predictor_key, dataset_predictor_path

    base = np.asarray(base_vecs, dtype=np.float32)
    if len(base) > max_n:
        rng = np.random.default_rng(0)
        base = base[rng.choice(len(base), max_n, replace=False)]
    name = dataset_name or "dataset"
    logger.info("HNSW predictor: dataset '%s' base %s", name, base.shape)
    df = collect_training_data(
        datasets={name: base}, M_values=M_values,
        ef_search_values=ef_search_values,
        ef_construction_values=ef_construction_values, n_queries=n_queries,
    )
    predictor = WorkloadPredictor()
    metrics = predictor.train(df)
    key = dataset_predictor_key(name, embedding)
    out = output or dataset_predictor_path(key)
    _atomic_save_predictor(predictor, out)
    logger.info(
        "Per-dataset predictor [%s] saved to %s (R2_ndis=%.4f, R2_nhops=%.4f)",
        key, out, metrics["r2_ndis"], metrics["r2_nhops"],
    )
    return out, metrics


def get_or_build_predictor(
    dataset: str,
    embedding: str,
    base_vecs,
    *,
    max_n: int = 200_000,
) -> str:
    """Return the per-(dataset, embedding) HNSW predictor path, building on miss.

    Cache HIT (the joblib already exists) returns instantly — the CM-init /
    system-migration fast path, no HNSW build. On a MISS it pays the one-time
    cost via :func:`train_from_vectors`. The unified
    ``cost_model.workload_profile`` entry calls this with the controller-supplied
    corpus base vectors.
    """
    from .coefficients import dataset_predictor_key, dataset_predictor_path

    path = dataset_predictor_path(dataset_predictor_key(dataset, embedding))
    if _valid_predictor(path):
        logger.info("[cm] HNSW predictor cached: %s", path)
        return path
    with _predictor_build_lock(path):
        # Another optimizer process may have completed the profile while this
        # process waited. Validate it, rather than trusting mere existence.
        if _valid_predictor(path):
            logger.info("[cm] HNSW predictor cached after lock: %s", path)
            return path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        for orphan in Path(path).parent.glob(f"{Path(path).name}.tmp.*"):
            try:
                orphan.unlink()
            except FileNotFoundError:
                pass
        out, _ = train_from_vectors(base_vecs, dataset, embedding, max_n=max_n)
        if not _valid_predictor(out):
            raise RuntimeError(f"HNSW predictor build did not publish a valid file: {out}")
        return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train HNSW workload predictor from instrumented FAISS search.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(_DEFAULT_OUTPUT),
        help=f"Output path for trained model (default: {_DEFAULT_OUTPUT})",
    )
    # ── Per-dataset training (the "trained dataset" half of the split profile) ──
    parser.add_argument(
        "--dataset-hdf5", type=str, default="",
        help="Train on a REAL dataset (ann-benchmarks hdf5; key 'train' = base "
             "vectors). Saves the predictor as the PER-DATASET profile "
             "calibration_profiles/faiss_hnsw/<dataset>__<embedding>.joblib.",
    )
    parser.add_argument("--dataset-name", type=str, default="",
                        help="Dataset id for the per-dataset profile key.")
    parser.add_argument("--embedding", type=str, default="",
                        help="Embedding-model id for the per-dataset profile key.")
    parser.add_argument("--max-n", type=int, default=200_000,
                        help="Subsample the real base to ≤ this many vectors "
                             "(HNSW build/measure tractability).")
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Use only synthetic datasets (no external data needed).",
    )
    parser.add_argument(
        "--sift-learn-path",
        type=str,
        default="data/sift/sift_learn.fvecs",
        help="Path to SIFT learning set (disjoint from sift_base).",
    )
    parser.add_argument(
        "--N",
        type=int,
        nargs="+",
        default=[10_000, 50_000, 100_000],
        help="Dataset sizes for synthetic data.",
    )
    parser.add_argument(
        "--d",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 768],
        help="Vector dimensions for synthetic data.",
    )
    parser.add_argument(
        "--M",
        type=int,
        nargs="+",
        default=[8, 12, 16, 24, 32, 48, 64],
        help="HNSW M values to sweep.",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512],
        help="ef_search values to sweep.",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        nargs="+",
        default=[100, 200, 400],
        help="ef_construction values to sweep.",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=1000,
        help="Number of queries per configuration.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # ── Per-dataset path: train on ONE real dataset → keyed profile, then return ──
    if args.dataset_hdf5:
        import os
        import h5py
        with h5py.File(args.dataset_hdf5, "r") as f:
            base = np.asarray(f["train"][:], dtype=np.float32)
        name = args.dataset_name or os.path.basename(args.dataset_hdf5)
        # Honor an explicit --output override; else train_from_vectors uses the
        # canonical per-(dataset, embedding) profile path.
        output = args.output if args.output != str(_DEFAULT_OUTPUT) else None
        train_from_vectors(
            base, name, args.embedding,
            max_n=args.max_n, M_values=args.M,
            ef_search_values=args.ef_search,
            ef_construction_values=args.ef_construction,
            n_queries=args.n_queries, output=output,
        )
        return

    # Generate datasets
    logger.info("Generating synthetic datasets...")
    datasets = generate_synthetic_datasets(
        N_values=args.N, d_values=args.d,
    )
    logger.info(f"Generated {len(datasets)} synthetic datasets")

    if not args.synthetic_only:
        sift = load_sift_learn(args.sift_learn_path, max_N=max(args.N))
        if sift is not None:
            for N in args.N:
                if N <= len(sift):
                    datasets[f"sift_learn_N{N}_d{sift.shape[1]}"] = sift[:N]
            logger.info(f"Added SIFT learn subsets; total datasets: {len(datasets)}")

    # Collect training data
    logger.info("Collecting training data (this may take a while)...")
    df = collect_training_data(
        datasets=datasets,
        M_values=args.M,
        ef_search_values=args.ef_search,
        ef_construction_values=args.ef_construction,
        n_queries=args.n_queries,
    )
    logger.info(f"Collected {len(df)} training samples")

    # Train
    logger.info("Training workload predictor...")
    predictor = WorkloadPredictor()
    metrics = predictor.train(df)
    logger.info(
        f"Training complete: R2_ndis={metrics['r2_ndis']:.4f}, "
        f"R2_nhops={metrics['r2_nhops']:.4f}"
    )

    # Save
    predictor.save(args.output)
    logger.info(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
