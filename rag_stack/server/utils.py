"""Shared utilities for the RAG-Stack API server."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("RAG-Stack")

# Single source of truth for the repository root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Process tree termination ────────────────────────────────────────────────

def kill_process_tree(pid: int, timeout: float = 5.0) -> dict:
    """SIGTERM the process and all descendants, then SIGKILL stragglers.

    Why we can't just SIGTERM the top pid: vLLM 0.16 spawns each worker as a
    separate multiprocessing subprocess (``VLLM::EngineCore``). When the
    parent rag_stack process gets SIGTERM, Python's atexit / __del__ hooks
    do not reliably fire before the process tears down, so the EngineCore
    subprocesses orphan to init and keep their GPU memory allocations alive
    indefinitely. We walk the descendant tree explicitly via psutil and reap
    everything bottom-up.

    Returns a small report dict for caller logging:
        ``{"sigterm_count": N, "sigkill_count": M, "missing": bool}``.
    """
    import os
    import signal

    report = {"sigterm_count": 0, "sigkill_count": 0, "missing": False}
    try:
        import psutil
    except ImportError:
        # Fallback: best-effort single-process kill.
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            report["missing"] = True
        return report

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        report["missing"] = True
        return report

    # Snapshot descendants BEFORE signalling — once children start dying,
    # `children()` calls race with reaping.
    try:
        descendants = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        descendants = []
    # Terminate leaves first so re-parenting doesn't lose them. ``descendants``
    # from psutil is already BFS-ordered; reverse for depth-first.
    targets = list(reversed(descendants)) + [parent]
    for proc in targets:
        try:
            proc.send_signal(signal.SIGTERM)
            report["sigterm_count"] += 1
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
        except psutil.AccessDenied:
            logger.warning(f"kill_process_tree: access denied SIGTERM pid={proc.pid}")
    # Wait briefly for graceful exit, then SIGKILL.
    _, alive = psutil.wait_procs(targets, timeout=timeout)
    for proc in alive:
        try:
            proc.kill()
            report["sigkill_count"] += 1
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
    logger.info(
        f"kill_process_tree pid={pid}: "
        f"sigterm={report['sigterm_count']} sigkill={report['sigkill_count']}"
    )
    return report

# ── Path resolution ─────────────────────────────────────────────────────────

def resolve_dir(configured: str, default: Path) -> Path:
    """Resolve a user-configured directory path.

    If *configured* is non-empty, treat it as absolute or relative to
    PROJECT_ROOT.  Falls back to *default* when the configured path is
    empty or does not exist on disk.
    """
    if configured:
        p = Path(configured)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_dir():
            return p
    return default


# ── Benchmark filename parsing ──────────────────────────────────────────────
# Format: ``{optimizer}__{function}.csv`` with seed carried as a column inside
# the CSV. The eval-distribution sibling ``…__evals.csv`` is explicitly
# rejected so the trajectory loader doesn't misparse it as ``fn="evals"``.

_FILENAME_RE = re.compile(r"^(.+)__(.+)$")
_EVAL_DIST_RE = re.compile(r"__evals$")


@dataclass
class RunInfo:
    """Metadata parsed from a benchmark CSV filename."""
    optimizer: str
    synthetic_function: str
    path: str


def parse_benchmark_filename(path: str) -> Optional[RunInfo]:
    """Extract optimizer, synthetic_function from a CSV filename.

    Expected pattern: ``{optimizer}__{function}.csv`` (seed lives inside the
    CSV as a column, written by ``BenchmarkRunner.save_csv``). Returns *None*
    for the eval-distribution sibling (``…__evals.csv``) and for filenames
    that do not match the pattern.
    """
    stem = Path(path).stem
    if _EVAL_DIST_RE.search(stem):
        return None
    m = _FILENAME_RE.match(stem)
    if not m:
        return None
    return RunInfo(
        optimizer=m.group(1),
        synthetic_function=m.group(2),
        path=path,
    )
