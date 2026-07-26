"""Shared CLI infrastructure for calibrator and benchmark."""

import argparse
import multiprocessing as mp
from pathlib import Path

from .systems import SystemConfig
from .coefficients import load_coefficients, build_hardware_from_coefficients
from rag_stack.calibration_paths import system_profile_path
from rag_stack.cost_model.cm_benchmark_datasets import CM_BENCHMARK_DIR


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments shared by calibrate and benchmark subcommands."""
    parser.add_argument(
        "--index-configs", nargs="+",
        default=[
            "IVF1024,PQ32x4fs",
            "IVF1024,PQ64x4fs",
            "IVF1024,PQ16",
            "IVF1024,PQ32",
        ],
        help="FAISS index factory strings to benchmark",
    )
    parser.add_argument(
        "--nprobe-values", nargs="+", type=int,
        default=[1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128],
        help="nprobe values to test",
    )
    parser.add_argument("--k", type=int, default=1, help="Number of neighbors")
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 1000, 10000],
        help="Query batch sizes",
    )
    parser.add_argument(
        "--num-threads", nargs="+", type=int, default=[0, 1],
        help="FAISS thread counts to sweep (0 = all cores). Default [0, 1] = "
             "all-cores and single-thread (the calibrated regime).",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["sift1m", "deep1m", "gist1m"],
        help="Standard ANN datasets to sweep. Auto-downloaded: 'sift1m' (d=128, "
             "TEXMEX), 'gist1m' (d=960, TEXMEX), 'deep1m' (d=96, ann-benchmarks "
             "deep-image subset). Staged-only: 'bigann1m' (d=128), 'music100' (d=100).",
    )
    parser.add_argument(
        "--parallel-modes", nargs="+", type=int, default=[0],
        help="FAISS parallel_mode values; latest standalone calibration supports only 0",
    )
    parser.add_argument(
        "--hardware", default="epyc_9124",
        help="Exact CPU profile key (default: epyc_9124)",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to rag_stack YAML config file. Overrides --hardware.",
    )
    parser.add_argument(
        "--cold-cache", action="store_true",
        help="Use cold cache model (default: warm cache)",
    )
    parser.add_argument(
        "--data-dir", default=CM_BENCHMARK_DIR,
        help="Directory for the benchmark datasets (real fvecs cached here)",
    )
    parser.add_argument(
        "--download-url", default=None,
        help="Override the dataset tarball URL (default: derived per-dataset "
             "from the TEXMEX corpus)",
    )


def resolve_hardware_preset(args) -> str:
    """Resolve hardware preset name from --config or --hardware."""
    if args.config:
        from .calibrator import register_hardware_from_yaml
        preset = register_hardware_from_yaml(args.config)
        print(f"  Loaded hardware from YAML: {args.config} → preset '{preset}'")
        return preset
    return args.hardware


def resolve_num_threads(args) -> list[int]:
    """Resolve the --num-threads sweep to concrete thread counts (0 → all cores).

    Always returns a list (``--num-threads`` is nargs='+'). Single-int callers
    (e.g. the calibrator) take ``[0]``.
    """
    raw = args.num_threads if isinstance(args.num_threads, list) else [args.num_threads]
    return [t if t > 0 else mp.cpu_count() for t in raw]


def build_hardware(
    preset: str,
    num_threads: int,
    calibrated_path: str | None = None,
) -> SystemConfig:
    """Build hardware from an exact profile; raw preset guesses are forbidden."""
    profile = Path(calibrated_path) if calibrated_path else system_profile_path(
        "faiss_ivf", preset
    )
    if not profile.is_file():
        raise FileNotFoundError(
            f"missing exact IVF profile for hardware {preset!r}: {profile}"
        )
    coefficients = load_coefficients(str(profile))
    hw = build_hardware_from_coefficients(coefficients, active_threads=num_threads)
    print(f"  Using calibrated coefficients from {profile}")
    print(f"  Calibration MAPE: {coefficients.mape_latency:.2f}%")
    return hw


def print_config_header(args, num_threads, hardware_preset: str) -> None:
    """Print common configuration header. ``num_threads`` may be an int or list."""
    warm_cache = not args.cold_cache
    if sorted(set(args.parallel_modes)) != [0]:
        raise ValueError(
            "parallel_mode=1 is unsupported: the latest IVF profile is mode-0-only"
        )
    print(f"  Datasets:      {getattr(args, 'datasets', ['sift'])}")
    print(f"  Index configs: {args.index_configs}")
    print(f"  nprobe values: {args.nprobe_values}")
    print(f"  Batch sizes:   {args.batch_sizes}")
    print(f"  Hardware:      {hardware_preset}")
    print(f"  Num threads:   {num_threads}")
    print(f"  Parallel modes:{args.parallel_modes}")
    print(f"  Cache model:   {'warm' if warm_cache else 'cold'}")
