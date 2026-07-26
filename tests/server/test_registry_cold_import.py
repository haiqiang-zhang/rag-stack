"""Registry endpoints must share the evaluator's cold-import guard."""

from tests import REPO_ROOT

import builtins
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def test_dataflows_waits_for_evaluator_before_importing_search_space(monkeypatch):
    from rag_stack.server.routers import rag_project

    ready = False
    original_import = builtins.__import__

    def warm():
        nonlocal ready
        ready = True

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rag_stack.search_space.rag_search_space":
            assert ready, "dataflow registry bypassed the shared evaluator import guard"
            return SimpleNamespace(REACT_ACTIVE_STAGES={"generator", "semantic_retrieval"})
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(rag_project, "_warm_static_rag_evaluator", warm)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert rag_project.get_dataflow_registry() == {
        "branches": ["sequential", "react"],
        "react_active_stages": ["generator", "semantic_retrieval"],
        "react_required_knobs": ["max_iter"],
    }


def test_cold_concurrent_registry_requests_complete_without_loading_cuda():
    # A fresh interpreter is essential: running this after other evaluator
    # tests would turn the circular imports into harmless module-cache hits.
    script = """
from concurrent.futures import ThreadPoolExecutor
import sys
import threading
import torch

def forbidden(*args, **kwargs):
    raise AssertionError('Registry requests must not initialize CUDA')
torch.cuda._lazy_init = forbidden
torch.cuda.init = forbidden
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
if hasattr(torch._C, '_cuda_init'):
    torch._C._cuda_init = forbidden

from rag_stack.server.routers import rag_project
assert 'rag_stack_evaluator.static_rag_evaluator' not in sys.modules
barrier = threading.Barrier(4)
functions = [rag_project.get_dataflow_registry, rag_project.get_node_types,
             rag_project.get_vectordb_types_endpoint, rag_project.get_generator_backends]
def request(function):
    barrier.wait(timeout=10)
    return function()
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(request, functions))
assert results[0]['branches'] == ['sequential', 'react']
assert results[0]['react_active_stages']
assert results[1]['stages']
assert results[2]['vectordb_types']
assert results[3]['backends']
assert rag_project._SRE_WARMED
assert not torch.cuda.is_initialized()
print('cold registry requests passed; CUDA not initialized')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cold registry requests passed; CUDA not initialized" in result.stdout
