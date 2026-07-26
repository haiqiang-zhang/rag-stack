"""Project summary datatype for RAG optimization project lifecycle management."""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ProjectStatus(str, Enum):
    CREATED = "created"            # folder created, config copied, not yet started
    RUNNING = "running"            # optimization in progress
    COMPLETED = "completed"        # optimization finished successfully
    FAILED = "failed"              # optimization errored out
    QUALITY_COMPLETE = "quality_complete"  # quality store sweep done, ready for hardware replay


CONFIG_FILENAME = "rag_stack_config.yaml"


class ProjectSummary(BaseModel):
    """Two-counter accounting (07-06): ``n_evaluations`` is the RAW attempt
    count (every eval id / folder, incl. penalty + invalid — folders are always
    preserved for audit); ``gt_count``/``cm_count`` count only VALID (success)
    evals and are what the ``n_iterations`` budget and the apps progress read.
    ``valid_eval_count`` is the explicit alias for gt_count + cm_count."""

    status: ProjectStatus
    timestamp: str = ""
    pid: Optional[int] = None
    total_cost: float = 0
    cm_count: int = 0
    gt_count: int = 0
    n_evaluations: int = 0
    n_pareto_points: int = 0
    n_invalid_configs: int = 0
    n_assembly_results: int = 0
    valid_eval_count: Optional[int] = None  # gt_count + cm_count (explicit)


def load_project_summary(project_dir: str) -> Optional[ProjectSummary]:
    """Load summary.json from a project directory. Returns None if missing."""
    path = os.path.join(project_dir, "summary.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return ProjectSummary(**data)


def save_project_summary(project_dir: str, summary: ProjectSummary) -> None:
    """Atomically write ``summary.json`` to a project directory.

    The API parent and optimize child can update lifecycle state only a few
    milliseconds apart.  A temp-file + fsync + replace commit prevents status
    readers (and the parent reaper) from observing a partially-written JSON
    document during that hand-off.
    """
    os.makedirs(project_dir, exist_ok=True)
    path = os.path.join(project_dir, "summary.json")
    fd, temp_path = tempfile.mkstemp(
        prefix=".summary.", suffix=".tmp", dir=project_dir,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(summary.model_dump(), f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def create_project_summary(status: ProjectStatus = ProjectStatus.CREATED) -> ProjectSummary:
    """Create a new ProjectSummary with current timestamp."""
    return ProjectSummary(
        status=status,
        timestamp=datetime.datetime.now().isoformat(),
    )


def pid_alive(pid: int) -> bool:
    """True if ``pid`` is an existing process (any owner) on this host."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # PID exists under another user; still "alive" for collision check.
        return True
    except OSError:
        return False
    return True
