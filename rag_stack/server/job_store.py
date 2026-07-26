"""JSON-file persistence for background jobs with PID tracking.

Tracks **optimizer benchmark sweeps** and **cost-model calibration jobs**.
RAG optimization and quality-store runs are NOT tracked here — they
maintain their own ``summary.json`` inside the project / store directory,
launched via direct ``subprocess.Popen`` from their respective routers.

Jobs persist across server restarts via ``.rag_stack_jobs.json`` at the
repo root, with PID-liveness checks reconciling stale "running" status.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .utils import PROJECT_ROOT


def _csv_has_seed(csv_path: Path, seed: int) -> bool:
    """Return True iff ``csv_path`` exists and has at least one row with the
    given seed. CSVs now hold all seeds for one (optimizer, function) pair,
    so "job completed" means "this run's seed appears in the file", not
    "file exists"."""
    if not csv_path.is_file():
        return False
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row.get("seed", -1)) == seed:
                    return True
    except (OSError, ValueError):
        return False
    return False

_JOBS_FILE = PROJECT_ROOT / ".rag_stack_jobs.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Launch helper ───────────────────────────────────────────────────────────

def launch_job(
    cmd: list[str],
    config: dict,
    output_dir: str,
    job_type: str,
    log_name: str,
    project_name: str = "",
) -> dict:
    """Launch a detached subprocess and persist a job entry.

    Args:
        cmd: Command to execute (e.g. [sys.executable, "-m", ...]).
        config: Config dict to store with the job for display.
        output_dir: Absolute path to output directory (created if missing).
        job_type: "optimizer_benchmark" or "cost_model" (the only kinds
            tracked here — RAG-opt and quality-store have their own
            summary.json mechanism).
        log_name: Filename for stdout/stderr log inside output_dir.

    Returns:
        The created job dict.
    """
    abs_output = Path(output_dir)
    abs_output.mkdir(parents=True, exist_ok=True)

    log_path = abs_output / log_name
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    job = create_job(
        config=config,
        pid=proc.pid,
        output_dir=str(abs_output),
        job_type=job_type,
        project_name=project_name,
        log_path=str(log_path),
    )

    # Daemon thread to reap the child process, preventing zombies, and persist
    # the actual exit status for cost-model jobs whose success is not inferable
    # from benchmark CSVs.
    threading.Thread(
        target=_wait_and_finalize_job,
        args=(proc, log_fh, job["job_id"]),
        daemon=True,
    ).start()

    return job


# ── CRUD ────────────────────────────────────────────────────────────────────

def load_jobs(job_type: Optional[str] = None) -> list[dict]:
    """Load jobs, refreshing status of running ones.

    Args:
        job_type: If set, only return jobs of this type.
    """
    jobs = _read_jobs_raw()
    changed = False
    for job in jobs:
        if job["status"] == "running":
            if not _is_pid_alive(job.get("pid")):
                job["status"] = _infer_finished_status(job)
                job["finished_at"] = _now()
                if job["status"] == "failed":
                    job["error"] = _read_last_error(job)
                changed = True
    if changed:
        _write_jobs(jobs)
    if job_type:
        return [j for j in jobs if j.get("job_type") == job_type]
    return jobs


def get_job(job_id: str) -> Optional[dict]:
    for job in load_jobs():
        if job["job_id"] == job_id:
            return job
    return None


def kill_job(job_id: str) -> Optional[dict]:
    """Kill a running job and all descendant processes (e.g. vllm EngineCore)."""
    from .utils import kill_process_tree
    jobs = load_jobs()
    for job in jobs:
        if job["job_id"] == job_id:
            if job["status"] != "running":
                return job
            pid = job.get("pid")
            if pid:
                kill_process_tree(pid)
            job["status"] = "failed"
            job["finished_at"] = _now()
            job["error"] = "Killed by user"
            _write_jobs(jobs)
            return job
    return None


def create_job(
    config: dict,
    pid: int,
    output_dir: str,
    job_type: str = "",
    project_name: str = "",
    log_path: str = "",
) -> dict:
    """Create and persist a new job entry."""
    job = {
        "job_id": str(uuid.uuid4())[:8],
        "job_type": job_type,
        "config": config,
        "pid": pid,
        "status": "running",
        "started_at": _now(),
        "finished_at": None,
        "error": None,
        "output_dir": output_dir,
        "project_name": project_name,
        "log_path": log_path,
    }
    jobs = load_jobs()
    jobs.append(job)
    _write_jobs(jobs)
    return job


# ── Internal helpers ────────────────────────────────────────────────────────

def _write_jobs(jobs: list[dict]) -> None:
    with open(_JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)


def _read_jobs_raw() -> list[dict]:
    if not _JOBS_FILE.is_file():
        return []
    with open(_JOBS_FILE) as f:
        return json.load(f)


def _wait_and_finalize_job(proc: subprocess.Popen, log_fh, job_id: str) -> None:
    returncode = proc.wait()
    try:
        log_fh.close()
    except Exception:
        pass

    jobs = _read_jobs_raw()
    changed = False
    for job in jobs:
        if job.get("job_id") != job_id or job.get("status") != "running":
            continue
        job["status"] = "completed" if returncode == 0 else "failed"
        job["finished_at"] = _now()
        if returncode != 0:
            job["error"] = _read_last_error(job)
        changed = True
        break
    if changed:
        _write_jobs(jobs)


def _is_pid_alive(pid: Optional[int]) -> bool:
    """Check if a process is truly alive (not zombie, not dead)."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False

    # os.kill(pid, 0) succeeds for zombie processes too.
    # Try to reap it with waitpid (WNOHANG) — if it's our zombie child,
    # this cleans it up and confirms it's dead.
    try:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid != 0:
            return False  # was zombie, now reaped
    except ChildProcessError:
        pass  # not our child — fall through to /proc check

    # Check /proc on Linux for zombie state
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("State:"):
                    if "Z" in line:
                        return False  # zombie
                    break
    except (FileNotFoundError, PermissionError):
        pass

    return True


def _read_last_error(job: dict) -> str:
    """Try to read the last ERROR line from the job's log file.

    Only services jobs that go through ``launch_job`` (cost-model
    calibration + optimizer benchmark sweeps), which write logs to
    ``<output_dir>/_job_<id>.log``. RAG-optimization and quality-store
    runs don't reach this code — they handle their own logs and surface
    errors via their respective ``summary.json`` files instead.
    """
    output_dir = job.get("output_dir", "")
    if not output_dir:
        return "Process exited unexpectedly"
    out = Path(output_dir)
    explicit = job.get("log_path")
    if explicit:
        logs = [Path(explicit)]
    else:
        logs = []
    logs.extend(sorted(out.glob("_job*.log"), key=lambda p: p.stat().st_mtime, reverse=True))
    for log_path in logs:
        try:
            lines = log_path.read_text().strip().split("\n")
            for line in reversed(lines):
                if "ERROR" in line or "Error" in line or "Exception" in line:
                    return line.strip()[-200:]
            if lines:
                return lines[-1].strip()[-200:]
        except Exception:
            continue
    return "Process exited unexpectedly"


def _infer_finished_status(job: dict) -> str:
    """Infer whether a dead process completed or failed.

    Used only by ``optimizer_benchmark`` and cost-model jobs — RAG
    optimization and quality-store runs track their own state via
    ``summary.json`` inside the project / store directory, not via
    this module.
    """
    output_dir = job.get("output_dir", "")
    if not output_dir:
        return "failed"

    out = Path(output_dir)

    if job.get("job_type", "").startswith("cost_model_"):
        log_text = ""
        explicit = job.get("log_path")
        candidates = [Path(explicit)] if explicit else []
        candidates.extend(sorted(out.glob("_job*.log"), key=lambda p: p.stat().st_mtime, reverse=True))
        for log_path in candidates:
            try:
                log_text = log_path.read_text(errors="replace")[-4000:]
                break
            except Exception:
                continue
        failed_markers = ("Traceback", "Exception", "ERROR", "Error", "SystemExit")
        return "failed" if any(m in log_text for m in failed_markers) else "completed"

    # Optimizer benchmark: check for output CSVs
    config = job.get("config", {})
    opt_type = config.get("optimizer", {}).get("type", "")
    sf_config = config.get("synthetic_function", {})
    func_name = sf_config if isinstance(sf_config, str) else sf_config.get("type", "")
    seed = config.get("seed", 0)

    if not opt_type or not func_name:
        return "failed"

    if func_name == "all":
        for csv_path in out.glob(f"{opt_type}__*.csv"):
            if _csv_has_seed(csv_path, seed):
                return "completed"
        return "failed"

    csv_path = out / f"{opt_type}__{func_name}.csv"
    return "completed" if _csv_has_seed(csv_path, seed) else "failed"
