"""Shared logging helpers for RAG-Stack entry points."""

from __future__ import annotations

import datetime
import logging
import os
import sys
from dataclasses import dataclass
from typing import TextIO


logger = logging.getLogger("RAG-Stack")

PROJECT_LOG_FILE_ENV = "RAG_STACK_PROJECT_LOG_FILE"
PROJECT_LOG_OWNED_BY_PARENT_ENV = "RAG_STACK_PROJECT_LOG_OWNED_BY_PARENT"
PROJECT_LOG_STDIO_REDIRECTED_ENV = "RAG_STACK_PROJECT_LOG_STDIO_REDIRECTED"


class Tee:
    """Write stdout to multiple streams while preserving the primary stream API."""

    def __init__(self, *streams: TextIO):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(self.streams and getattr(self.streams[0], "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.streams[0].fileno()


@dataclass
class ProjectLogState:
    path: str
    old_stdout: TextIO | None
    stdout_log: TextIO | None
    handler: logging.Handler


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


def attach_project_log(
    out_dir: str,
    command: str,
    *,
    log_file: str | None = None,
    latest_name: str | None = None,
    tee_stdout: bool = False,
) -> ProjectLogState:
    """Attach one persistent RAG-Stack log for a command.

    The log lives under ``<out_dir>/logs`` unless ``log_file`` is explicitly
    provided. ``RAG_STACK_PROJECT_LOG_FILE`` is exported so vLLM subprocess
    launchers can mirror their stdout/stderr into the same file.
    """
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_file or os.path.join(log_dir, f"{command}_{ts}.log")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    os.environ[PROJECT_LOG_FILE_ENV] = path
    stdout_log = open(path, "a", buffering=1, encoding="utf-8") if tee_stdout else None
    old_stdout = sys.stdout if tee_stdout else None
    if tee_stdout and stdout_log is not None:
        sys.stdout = Tee(sys.stdout, stdout_log)

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger()
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    root.addHandler(handler)

    latest = os.path.join(log_dir, f"{latest_name or command}_latest.log")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(path), latest)
    except OSError:
        pass

    logger.info("[logging] command=%r  argv=%s", command, " ".join(sys.argv))
    logger.info("[logging] persistent log -> %s", path)
    return ProjectLogState(path=path, old_stdout=old_stdout, stdout_log=stdout_log, handler=handler)


def detach_project_log(state: ProjectLogState | None) -> None:
    if state is None:
        return
    if state.old_stdout is not None:
        sys.stdout = state.old_stdout
    logging.getLogger().removeHandler(state.handler)
    state.handler.close()
    if state.stdout_log is not None:
        state.stdout_log.close()
