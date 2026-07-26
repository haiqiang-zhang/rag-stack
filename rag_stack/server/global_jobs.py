"""Global running-job discovery for the web UI.

This intentionally looks at OS processes and then collapses them to
RAG-Stack *main* processes. RAG optimization and calibration can fork many
children (vLLM, multiprocessing resource trackers, workers), so counting every
process whose environment happens to be ``rag-stack`` would over-report jobs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .job_store import _is_pid_alive, load_jobs
from .settings_store import load_settings
from .utils import PROJECT_ROOT, resolve_dir
from rag_stack.dataset_gen_summary import DatasetGenStatus, load_dataset_gen_summary
from rag_stack.project_summary import ProjectStatus, load_project_summary


_DEFAULT_PROJECT_DIR = PROJECT_ROOT / "tmp_outputs" / "rag_stack_projects"
_DEFAULT_QUALITY_STORE_DIR = PROJECT_ROOT / "quality_stores"
_DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets"
_PROC_DETAIL_DEFAULT_LIMIT = 160
_PROC_DETAIL_HARD_LIMIT = 500


def _clock_ticks_per_second() -> int:
    try:
        return int(os.sysconf("SC_CLK_TCK"))
    except (AttributeError, OSError, ValueError):
        return 100


def _project_dir() -> Path:
    return resolve_dir(load_settings().rag_optimization_project_dir, _DEFAULT_PROJECT_DIR)


def _quality_store_dir() -> Path:
    return resolve_dir(
        getattr(load_settings(), "quality_stores_dir", ""),
        _DEFAULT_QUALITY_STORE_DIR,
    )


def _dataset_dir() -> Path:
    return resolve_dir(load_settings().datasets_dir, _DEFAULT_DATASET_DIR)


def _read_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [p.decode(errors="replace") for p in raw.split(b"\0") if p]


def _read_status(pid: int) -> dict[str, str]:
    try:
        text = Path(f"/proc/{pid}/status").read_text(errors="replace")
    except OSError:
        return {}
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key] = value.strip()
    return fields


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _parse_kb_value(value: str | None) -> int:
    if not value:
        return 0
    parts = value.split()
    if not parts:
        return 0
    return _parse_int(parts[0]) * 1024


def _read_stat(pid: int) -> dict[str, Any] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    close = text.rfind(")")
    if close < 0:
        return None
    parts = text[close + 2 :].split()
    if len(parts) < 4:
        return None
    try:
        result = {
            "pid": pid,
            "state": parts[0],
            "ppid": int(parts[1]),
            "pgid": int(parts[2]),
            "sid": int(parts[3]),
        }
    except ValueError:
        return None
    optional_indexes = {
        "utime_ticks": 11,
        "stime_ticks": 12,
        "cutime_ticks": 13,
        "cstime_ticks": 14,
        "num_threads": 17,
        "starttime_ticks": 19,
    }
    for key, index in optional_indexes.items():
        if len(parts) > index:
            result[key] = _parse_int(parts[index])
    return result


def _iter_process_stats() -> list[dict[str, Any]]:
    procs: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        stat = _read_stat(pid)
        if not stat:
            continue
        procs.append(stat)
    return procs


def _iter_processes() -> list[dict[str, Any]]:
    procs: list[dict[str, Any]] = []
    for stat in _iter_process_stats():
        pid = int(stat["pid"])
        cmdline = _read_cmdline(pid)
        if not cmdline:
            continue
        procs.append({**stat, "cmdline": cmdline})
    return procs


def _module_after_dash_m(cmdline: list[str]) -> str:
    for i, part in enumerate(cmdline[:-1]):
        if part == "-m":
            return cmdline[i + 1]
    return ""


def _program_name(part: str) -> str:
    return Path(part).name


def _is_python_or_conda_command(cmdline: list[str]) -> bool:
    if not cmdline:
        return False
    program = _program_name(cmdline[0])
    if program.startswith("python") or program == "conda":
        return True
    # `conda run` itself is a Python script:
    #   /.../bin/python /.../bin/conda run -n rag-stack python ...
    return len(cmdline) > 1 and _program_name(cmdline[1]) == "conda"


def _is_repo_job_script(part: str) -> bool:
    if part.startswith("-") or not part.endswith(".py"):
        return False
    path = Path(part)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT)
    except (OSError, ValueError):
        return False
    if not rel.parts:
        return False
    return rel.parts[0] in {"benchmarks", "experiments", "scripts", "analysis"}


def _looks_like_rag_stack_main(cmdline: list[str]) -> bool:
    """True for top-level RAG-Stack commands, false for child workers.

    This deliberately excludes commands that merely run inside the ``rag-stack``
    conda env (for example vLLM API servers). Those are children/resources, not
    RAG-Stack jobs.
    """
    module = _module_after_dash_m(cmdline)
    if module == "rag_stack" or module.startswith("rag_stack."):
        return True

    for part in cmdline[1:]:
        normalized = part.replace("\\", "/")
        if "/rag_stack/" in normalized and normalized.endswith(".py"):
            return True
    if _is_python_or_conda_command(cmdline):
        return any(_is_repo_job_script(part) for part in cmdline[1:])
    return False


def _command_label(cmdline: list[str]) -> str:
    module = _module_after_dash_m(cmdline)
    if module == "rag_stack":
        try:
            idx = cmdline.index("-m")
            if len(cmdline) > idx + 2:
                return cmdline[idx + 2]
        except ValueError:
            pass
        return "rag_stack"
    if module == "rag_stack.cli":
        try:
            idx = cmdline.index("rag_stack.cli")
            if len(cmdline) > idx + 1:
                return cmdline[idx + 1]
        except ValueError:
            pass
        return "rag_stack.cli"
    if module.startswith("rag_stack."):
        return module.removeprefix("rag_stack.")
    for part in cmdline[1:]:
        if _is_repo_job_script(part):
            return str(Path(part))
        normalized = part.replace("\\", "/")
        if "/rag_stack/" in normalized:
            return Path(normalized).stem
    return Path(cmdline[0]).name


def _join_command(cmdline: list[str], max_chars: int = 1000) -> str:
    command = " ".join(cmdline)
    if len(command) <= max_chars:
        return command
    return command[: max_chars - 1] + "…"


def _main_rag_stack_processes() -> list[dict[str, Any]]:
    """Return matching process roots, not every matching descendant."""
    procs = _iter_processes()
    by_pid = {p["pid"]: p for p in procs}
    matching = {
        p["pid"]: p
        for p in procs
        if _looks_like_rag_stack_main(p["cmdline"]) and _is_pid_alive(p["pid"])
    }

    main: list[dict[str, Any]] = []
    for proc in matching.values():
        parent = by_pid.get(proc["ppid"])
        has_matching_ancestor = False
        seen: set[int] = set()
        while parent and parent["pid"] not in seen:
            seen.add(parent["pid"])
            if parent["pid"] in matching:
                has_matching_ancestor = True
                break
            parent = by_pid.get(parent["ppid"])
        if not has_matching_ancestor:
            main.append(proc)

    main.sort(key=lambda p: p["pid"])
    return main


def _running_rag_projects() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    root = _project_dir()
    if not root.is_dir():
        return result
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            summary = load_project_summary(str(path))
        except Exception:
            continue
        if not summary or summary.status != ProjectStatus.RUNNING or not summary.pid:
            continue
        if not _is_pid_alive(summary.pid):
            continue
        result[summary.pid] = {
            "kind": "rag_optimize",
            "name": path.name,
            "status": "running",
            "started_at": summary.timestamp,
            "path": str(path),
            "n_evaluations": summary.n_evaluations,
        }
    return result


def _running_quality_stores() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    root = _quality_store_dir()
    if not root.is_dir():
        return result
    for path in root.iterdir():
        manifest_path = path / "manifest.json"
        if not path.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue
        pid = manifest.get("pid")
        if manifest.get("status") != "running" or not pid or not _is_pid_alive(pid):
            continue
        result[int(pid)] = {
            "kind": "quality_store",
            "name": path.name,
            "status": "running",
            "started_at": manifest.get("created_at") or manifest.get("updated_at") or "",
            "path": str(path),
            "n_configs": manifest.get("n_configs", 0),
        }
    return result


def _running_dataset_jobs() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    root = _dataset_dir()
    if not root.is_dir():
        return result
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            summary = load_dataset_gen_summary(str(path))
        except Exception:
            continue
        if not summary or summary.status != DatasetGenStatus.RUNNING or not summary.pid:
            continue
        if not _is_pid_alive(summary.pid):
            continue
        result[summary.pid] = {
            "kind": "dataset_generation",
            "name": path.name,
            "status": "running",
            "started_at": summary.timestamp,
            "path": str(path),
            "strategy": summary.strategy,
        }
    return result


def _running_job_store_jobs() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for job in load_jobs():
        pid = job.get("pid")
        if job.get("status") != "running" or not pid or not _is_pid_alive(pid):
            continue
        result[int(pid)] = {
            "kind": job.get("job_type") or "job",
            "name": job.get("project_name") or job.get("job_id") or f"pid-{pid}",
            "status": "running",
            "started_at": job.get("started_at") or "",
            "path": job.get("output_dir") or "",
            "job_id": job.get("job_id"),
            "log_path": job.get("log_path"),
        }
    return result


def get_global_jobs_overview() -> dict[str, Any]:
    """Return running RAG-Stack jobs, counted by main process only."""
    metadata_by_pid: dict[int, dict[str, Any]] = {}
    for source in (
        _running_rag_projects(),
        _running_quality_stores(),
        _running_dataset_jobs(),
        _running_job_store_jobs(),
    ):
        metadata_by_pid.update(source)

    jobs: list[dict[str, Any]] = []
    for proc in _main_rag_stack_processes():
        pid = int(proc["pid"])
        meta = metadata_by_pid.get(pid, {})
        cmdline = proc["cmdline"]
        jobs.append({
            "pid": pid,
            "ppid": proc["ppid"],
            "pgid": proc["pgid"],
            "sid": proc["sid"],
            "kind": meta.get("kind") or _command_label(cmdline),
            "name": meta.get("name") or _command_label(cmdline),
            "status": meta.get("status") or "running",
            "started_at": meta.get("started_at") or "",
            "path": meta.get("path") or "",
            "job_id": meta.get("job_id"),
            "command": _join_command(cmdline, max_chars=4000),
            "argv": cmdline,
        })

    # Include metadata-backed jobs whose command line disappeared from the
    # scan window only if the PID is still alive. This keeps the UI useful on
    # non-/proc platforms or under transient read races.
    seen = {j["pid"] for j in jobs}
    for pid, meta in sorted(metadata_by_pid.items()):
        if pid in seen:
            continue
        if not _is_pid_alive(pid):
            continue
        jobs.append({
            "pid": pid,
            "ppid": None,
            "pgid": None,
            "sid": None,
            "kind": meta.get("kind") or "job",
            "name": meta.get("name") or f"pid-{pid}",
            "status": "running",
            "started_at": meta.get("started_at") or "",
            "path": meta.get("path") or "",
            "job_id": meta.get("job_id"),
            "command": "",
            "argv": [],
        })

    jobs.sort(key=lambda j: (j.get("started_at") or "", j["pid"]), reverse=True)
    return {
        "running_count": len(jobs),
        "jobs": jobs,
        "process_scan": {
            "strategy": "rag_stack_main_processes",
            "dedupe": "matched ancestors are collapsed; forked children are not counted",
        },
    }


def _descendant_pids(root_pid: int, max_processes: int) -> tuple[list[tuple[int, int]], bool]:
    stats = _iter_process_stats()
    by_ppid: dict[int, list[int]] = {}
    known_pids = {int(proc["pid"]) for proc in stats}
    if root_pid not in known_pids:
        return [], False
    for proc in stats:
        by_ppid.setdefault(int(proc["ppid"]), []).append(int(proc["pid"]))
    for children in by_ppid.values():
        children.sort(reverse=True)

    ordered: list[tuple[int, int]] = []
    seen: set[int] = set()
    stack: list[tuple[int, int]] = [(root_pid, 0)]
    truncated = False
    while stack:
        pid, depth = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if len(ordered) >= max_processes:
            truncated = True
            break
        ordered.append((pid, depth))
        for child_pid in by_ppid.get(pid, []):
            stack.append((child_pid, depth + 1))
    return ordered, truncated


def _process_snapshot(pid: int, depth: int) -> dict[str, Any] | None:
    stat = _read_stat(pid)
    if not stat:
        return None
    status = _read_status(pid)
    cmdline = _read_cmdline(pid)
    clock_ticks = _clock_ticks_per_second()
    cpu_ticks = int(stat.get("utime_ticks", 0)) + int(stat.get("stime_ticks", 0))
    state = status.get("State") or str(stat.get("state") or "")
    return {
        "pid": pid,
        "ppid": stat.get("ppid"),
        "pgid": stat.get("pgid"),
        "sid": stat.get("sid"),
        "depth": depth,
        "name": status.get("Name") or (Path(cmdline[0]).name if cmdline else f"pid-{pid}"),
        "state": state,
        "threads": _parse_int(status.get("Threads"), int(stat.get("num_threads", 0))),
        "rss_bytes": _parse_kb_value(status.get("VmRSS")),
        "cpu_time_seconds": cpu_ticks / clock_ticks if clock_ticks > 0 else 0,
        "command": _join_command(cmdline, max_chars=1000),
        "argv0": cmdline[0] if cmdline else "",
    }


def get_global_job_process_detail(
    pid: int,
    max_processes: int = _PROC_DETAIL_DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Return an on-demand, low-overhead process tree snapshot for one job.

    The overview endpoint intentionally avoids these reads. This detail path is
    called only after a user expands one process card, and it reads each relevant
    ``/proc`` file once. It reports cumulative CPU time instead of sampled CPU%
    to avoid sleeping or polling while benchmarks are running.
    """
    max_processes = max(1, min(int(max_processes), _PROC_DETAIL_HARD_LIMIT))
    pid = int(pid)
    scanned_at = datetime.now(timezone.utc).isoformat()
    ordered, truncated = _descendant_pids(pid, max_processes=max_processes)
    if not ordered:
        return {
            "pid": pid,
            "alive": False,
            "scanned_at": scanned_at,
            "summary": {
                "total_processes": 0,
                "descendant_processes": 0,
                "total_rss_bytes": 0,
                "total_threads": 0,
                "total_cpu_time_seconds": 0,
                "max_depth": 0,
            },
            "root": None,
            "processes": [],
            "collection": {
                "mode": "on_demand_procfs_snapshot",
                "max_processes": max_processes,
                "truncated": False,
                "cpu": "cumulative_user_plus_system_time_no_sampling",
            },
        }

    processes = [
        snapshot
        for proc_pid, depth in ordered
        if (snapshot := _process_snapshot(proc_pid, depth)) is not None
    ]
    total_rss_bytes = sum(int(proc.get("rss_bytes") or 0) for proc in processes)
    total_threads = sum(int(proc.get("threads") or 0) for proc in processes)
    total_cpu_time_seconds = sum(float(proc.get("cpu_time_seconds") or 0) for proc in processes)
    max_depth = max((int(proc.get("depth") or 0) for proc in processes), default=0)
    return {
        "pid": pid,
        "alive": bool(processes),
        "scanned_at": scanned_at,
        "summary": {
            "total_processes": len(processes),
            "descendant_processes": max(0, len(processes) - 1),
            "total_rss_bytes": total_rss_bytes,
            "total_threads": total_threads,
            "total_cpu_time_seconds": total_cpu_time_seconds,
            "max_depth": max_depth,
        },
        "root": processes[0] if processes else None,
        "processes": processes,
        "collection": {
            "mode": "on_demand_procfs_snapshot",
            "max_processes": max_processes,
            "truncated": truncated,
            "cpu": "cumulative_user_plus_system_time_no_sampling",
        },
    }
