"""An isolated preview must not leave evaluator model processes resident."""

import subprocess
import sys
import textwrap

import psutil
import pytest


@pytest.mark.parametrize("fail,ignore_term", [(False, False), (True, True)])
def test_worker_exit_reaps_only_owned_children_on_success_and_failure(fail, ignore_term):
    child_code = (
        "import os,signal,time; "
        f"signal.signal(signal.SIGTERM, signal.SIG_IGN if {ignore_term!r} else signal.SIG_DFL); "
        "print(os.getpid(), flush=True); time.sleep(60)"
    )
    script = textwrap.dedent(f"""
        import subprocess, sys
        from rag_stack import cm_preview
        def run(*args):
            child = subprocess.Popen([sys.executable, '-c', {child_code!r}],
                                     stdout=subprocess.PIPE, text=True)
            print(child.stdout.readline().strip(), flush=True)
            if {fail!r}:
                raise RuntimeError('fixture worker failure')
            return 0
        cm_preview.run_worker = run
        sys.argv = ['cm-preview', '--job-dir', '.', '--run-id', 'fixture']
        cm_preview.main()
    """)
    sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        worker = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                text=True, timeout=15, start_new_session=True)
        pid = int(worker.stdout.strip())
        assert worker.returncode == (1 if fail else 0), worker.stderr
        assert not psutil.pid_exists(pid)
        assert sibling.poll() is None
    finally:
        sibling.terminate()
        sibling.wait(timeout=5)
