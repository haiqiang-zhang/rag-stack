"""Regression coverage for CLI descendant cleanup."""

from __future__ import annotations

from tests import REPO_ROOT

import atexit
import os
import signal
import subprocess
import sys
import textwrap

import pytest


REPO = str(REPO_ROOT)


def test_reaper_skips_resource_trackers_but_kills_worker(monkeypatch):
    """Tracker helpers need pipe-EOF shutdown; ordinary children are reaped."""
    import psutil

    from rag_stack.cli import _install_subproc_reaper

    class _Child:
        def __init__(self, command):
            self.command = command
            self.signals = []
            self.kills = 0

        def cmdline(self):
            return self.command.split()

        def send_signal(self, signum):
            self.signals.append(signum)

        def kill(self):
            self.kills += 1

    mp_tracker = _Child(
        "python -c from multiprocessing.resource_tracker import main;main(9)"
    )
    loky_tracker = _Child(
        "python -c from joblib.externals.loky.backend.resource_tracker "
        "import main;main(10)"
    )
    worker = _Child("python -c worker_main()")

    class _Parent:
        def children(self, recursive):
            assert recursive is True
            return [mp_tracker, worker, loky_tracker]

    callbacks = []
    monkeypatch.setattr(psutil, "Process", _Parent)

    def _wait_procs(processes, timeout):
        assert processes == [worker]
        assert timeout == 5.0
        return [], [worker]

    monkeypatch.setattr(psutil, "wait_procs", _wait_procs)
    monkeypatch.setattr(atexit, "register", callbacks.append)
    monkeypatch.setattr(signal, "signal", lambda *_args: None)

    _install_subproc_reaper()
    assert len(callbacks) == 1
    callbacks[0]()

    assert worker.signals == [signal.SIGTERM]
    assert worker.kills == 1
    assert mp_tracker.signals == []
    assert mp_tracker.kills == 0
    assert loky_tracker.signals == []
    assert loky_tracker.kills == 0


def test_reaper_allows_multiprocessing_tracker_clean_shutdown():
    """The exact loky-named SemLock lifecycle must exit without relaunching."""
    script = textwrap.dedent(
        """
        import multiprocessing as mp

        # Loky patches stdlib SemLock names to /loky-<pid>-..., matching the
        # persistent spawn ProcessPoolExecutor used by cost-model sweeps.
        import joblib.externals.loky.backend  # noqa: F401

        semaphores = [mp.get_context("spawn").Semaphore() for _ in range(5)]
        from rag_stack.cli import _install_subproc_reaper
        _install_subproc_reaper()
        assert all(sem._semlock.name.startswith("/loky-") for sem in semaphores)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "resource_tracker: process died unexpectedly" not in output
    assert "KeyError: '/loky-" not in output
    assert "SIGKILL: 1" not in output


def test_cmd_optimize_system_exit_marks_owned_summary_failed(
    monkeypatch, tmp_path,
):
    """SIGTERM's SystemExit path must not leave a dead project running."""
    from types import SimpleNamespace

    import rag_stack.cli as cli
    from rag_stack.project_summary import (
        ProjectStatus,
        create_project_summary,
        load_project_summary,
        save_project_summary,
    )

    project = tmp_path / "project"
    project.mkdir()
    (project / "rag_stack_config.yaml").write_text("dataset: {}\n")
    summary = create_project_summary(ProjectStatus.CREATED)
    save_project_summary(str(project), summary)

    class _Controller:
        def __init__(self, **_kwargs):
            pass

        def get_pareto(self, n_iterations=None):
            raise SystemExit(143)

    import rag_stack.controller as controller_module

    monkeypatch.setattr(controller_module, "Controller", _Controller)
    monkeypatch.setattr(cli, "_setup_logging", lambda: None)
    monkeypatch.setattr(cli, "_install_subproc_reaper", lambda: None)

    args = SimpleNamespace(
        project_dir=str(project),
        config=None,
        performance_source=None,
        mode="full",
        quality_store=None,
        from_project=None,
        qa_data=None,
        corpus_data=None,
        n_iterations=None,
        hardware_config=None,
        n_refine_iterations=0,
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_optimize(args)
    assert exc_info.value.code == 143

    final = load_project_summary(str(project))
    assert final is not None
    assert final.status is ProjectStatus.FAILED
    assert final.pid is None
