"""Host/evaluator boundaries that must survive an editable submodule install."""

from tests import REPO_ROOT

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


ROOT = REPO_ROOT


def test_installed_static_evaluator_is_discoverable_in_a_fresh_process():
    # Editable source imports can succeed while installed entry-point metadata
    # is stale. Exercise actual package discovery without synthetic metadata.
    code = """
from importlib import metadata
from rag_stack.evaluation.registry import available_backends, get_backend_spec

entries = metadata.distribution('rag-stack-evaluator').entry_points
assert any(
    entry.group == 'rag_stack.evaluators'
    and entry.name == 'static_gt'
    and entry.value == 'rag_stack_evaluator.backends:STATIC_GT'
    for entry in entries
), 'Reinstall the editable rag-stack-evaluator package to refresh its entry-point metadata'
assert 'static_gt' in available_backends()
spec = get_backend_spec('static_gt')
assert spec.name == 'static_gt'
assert spec.factory.__module__ == 'rag_stack_evaluator.backends'
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True,
        text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "first_import",
    [
        "from rag_stack.search_space import RagSearchSpace",
        "from rag_stack.dataset_manager import DatasetManager",
        "from rag_stack_evaluator.static_rag_evaluator import DatasetEvalManager",
    ],
)
def test_fresh_process_imports_use_the_installed_evaluator(first_import):
    # Subprocess launches must resolve the evaluator without a sys.path shim,
    # and neither direction of the host/evaluator dependency may form a cycle.
    code = first_import + "\n" + """
from pathlib import Path
import importlib.util
import rag_stack
import rag_stack_evaluator
from rag_stack.dataset_manager import DatasetManager
from rag_stack_evaluator.static_rag_evaluator import DatasetEvalManager
from rag_stack_evaluator.static_rag_evaluator.measured import MeasuredProvider
from rag_stack_evaluator.base import BaseEvaluator
from rag_stack_evaluator.static_rag_evaluator import StaticRAGEvaluatorQualityOnly
from rag_stack.cost_model.token_stats import TokenStats
root = Path(rag_stack.__file__).resolve().parents[1]
assert Path(rag_stack_evaluator.__file__).resolve().is_relative_to(root / 'RAG-Stack-Evaluator')
assert issubclass(DatasetManager, DatasetEvalManager)
assert TokenStats.__module__ == 'rag_stack.cost_model.token_stats'
assert importlib.util.find_spec('rag_stack.cost_model_v1') is None
assert importlib.util.find_spec('rag_stack.static_rag_evaluator') is None
assert importlib.util.find_spec('rag_stack.vllm_instrumentation') is None
assert importlib.util.find_spec('rag_stack.flashrag_quality_evaluator') is None
assert importlib.util.find_spec('rag_stack.base_evaluator') is None
assert issubclass(StaticRAGEvaluatorQualityOnly, BaseEvaluator)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True,
        text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_host_manager_publishes_corpus_to_evaluator_registry(tmp_path):
    from rag_stack.dataset_manager import DatasetManager
    from rag_stack_evaluator.static_rag_evaluator.dataset import (
        CorpusView,
        get_active_corpus,
    )

    # The host and retrieval nodes must share one registry/module identity.
    manager = object.__new__(DatasetManager)
    manager.project_dir = str(tmp_path / "project")
    manager._active_view_key = None
    manager.corpus_data = pd.DataFrame({"doc_id": ["old"], "contents": ["old"]})
    (Path(manager.project_dir) / "data").mkdir(parents=True)
    corpus_path = tmp_path / "corpus.parquet"
    expected = pd.DataFrame({"doc_id": ["one"], "contents": ["shared corpus"]})
    expected.to_parquet(corpus_path, index=False)
    active = manager.activate(CorpusView("fixture", str(corpus_path), None, 1))
    assert get_active_corpus(manager.project_dir) is active
    pd.testing.assert_frame_equal(active, expected)


def test_flashrag_dependency_is_owned_by_the_host():
    """A recursive checkout has one FlashRAG owner and no host implementation."""
    import configparser

    host_modules = configparser.ConfigParser()
    host_modules.read(ROOT / ".gitmodules")
    host_paths = {host_modules[section]["path"] for section in host_modules.sections()}
    assert "RAG-Stack-Evaluator" in host_paths
    assert "FlashRAG" in host_paths
    evaluator_modules = configparser.ConfigParser()
    evaluator_modules.read(ROOT / "RAG-Stack-Evaluator" / ".gitmodules")
    evaluator_paths = {
        evaluator_modules[section]["path"] for section in evaluator_modules.sections()
    }
    assert "FlashRAG" not in evaluator_paths
    assert not (ROOT / "rag_stack" / "flashrag_quality_evaluator").exists()
    assert not (ROOT / "rag_stack" / "base_evaluator.py").exists()
    assert not (ROOT / "RAG-Stack-Evaluator" / "rag_stack_evaluator" / "flashrag_quality_evaluator").exists()
    assert (ROOT / "FlashRAG" / "flashrag" / "integrations" / "rag_stack").is_dir()
