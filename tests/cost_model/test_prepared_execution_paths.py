"""Runtime path ownership and reuse across the dynamic CM phase boundaries."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import pickle

import pytest

from rag_stack.cost_model import workload
from rag_stack.cost_model.compiled import workload_from_paths


def _trace():
    return [[{
        "stage": "generator", "input_tokens": 8, "output_tokens": 2,
        "step_idx": 7, "call_id": "gen-0", "node_id": "generator",
        "input_profile": {"items": 3, "list_sizes": [1, 2]},
    }]]


def test_prepared_snapshot_owns_nested_input_and_reindexes_without_mutating_caller():
    raw = _trace()
    before = deepcopy(raw)
    prepared = workload.resolve_execution_paths(raw)
    assert raw == before
    assert prepared.paths[0][0]["step_idx"] == 0
    assert prepared.paths[0][0]["call_id"] == "gen-0"
    raw[0][0]["input_profile"]["list_sizes"][0] = 99
    raw[0].clear()
    assert prepared.paths[0][0]["input_profile"]["list_sizes"] == [1, 2]
    with pytest.raises(FrozenInstanceError):
        prepared.paths = []


def test_reuse_requires_the_exact_owned_snapshot_and_never_recopies(monkeypatch):
    prepared = workload.resolve_execution_paths(_trace())
    monkeypatch.setattr(workload, "compile_execution_paths",
                        lambda *a, **kw: pytest.fail("Prepared input must not be compiled again"))
    assert workload.resolve_execution_paths(prepared.paths, prepared) is prepared
    with pytest.raises(ValueError, match="must own"):
        workload.resolve_execution_paths(deepcopy(prepared.paths), prepared)
    with pytest.raises(TypeError, match="PreparedExecutionPaths"):
        workload.resolve_execution_paths(prepared.paths, True)


@pytest.mark.parametrize("raw", [
    [[{"input_tokens": 1}]],
    [[{"stage": "generator", "input_profile": {"items": 4, "list_sizes": [2, 3]}}]],
    [[{"stage": "generator", "node_id": "missing-call-id"}]],
])
def test_raw_preparation_keeps_stage_identity_and_io_validation(raw):
    with pytest.raises(ValueError):
        workload.resolve_execution_paths(raw)


def test_worker_transfer_preserves_path_identity_without_revalidating(monkeypatch):
    prepared = workload.resolve_execution_paths(_trace())
    payload = pickle.dumps((prepared.paths, prepared))
    monkeypatch.setattr(workload, "compile_execution_paths",
                        lambda *a, **kw: pytest.fail("Worker transfer must not compile"))
    paths, transferred = pickle.loads(payload)
    assert paths is transferred.paths
    assert workload.resolve_execution_paths(paths, transferred) is transferred
    assert paths == prepared.paths and paths is not prepared.paths


def test_fixed_adapter_reuses_prepared_paths_and_checks_question_id_alignment(monkeypatch):
    prepared = workload.resolve_execution_paths(_trace())
    monkeypatch.setattr(workload, "compile_execution_paths",
                        lambda *a, **kw: pytest.fail("Fixed adapter must reuse prepared paths"))
    fixed = workload_from_paths({}, prepared.paths, ["q0"], prepared_paths=prepared)
    assert fixed.prepared_paths is prepared
    assert fixed.execution_paths is prepared.paths
    assert fixed.source_question_ids == ["q0"]
    with pytest.raises(ValueError, match="align"):
        workload_from_paths({}, prepared.paths, [], prepared_paths=prepared)
