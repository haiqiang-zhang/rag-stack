"""Pure algorithm contracts, isolated from cost-model native auto-discovery."""

from tests import REPO_ROOT

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def api():
    # Import the real stdlib-only modules under an isolated package so these
    # CPU tests never trigger GenZ/FAISS discovery in cost_model.__init__.
    package_name = "_rag_cm_pure_algorithm_tests"
    source = REPO_ROOT / "rag_stack" / "cost_model"
    package = ModuleType(package_name)
    package.__path__ = [str(source)]
    sys.modules[package_name] = package
    modules = {}
    for name in ("algorithm", "rrf_algorithm"):
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{name}", source / f"{name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    yield modules["algorithm"], modules["rrf_algorithm"]
    for name in (*[f"{package_name}.{n}" for n in modules], package_name):
        sys.modules.pop(name, None)


def _workload(api, *, union=5):
    a, _ = api
    return a.AlgorithmWorkload(inputs={
        "dense": a.DataProfile(4, 64, {"unique_items": 4}),
        "lexical": a.DataProfile(3, 48, {"unique_items": 3}),
    }, statistics={"union_items": union})


def test_rrf_models_aligned_union_and_full_sort_before_topk(api):
    a, r = api
    model = r.RRFAlgorithmModel()
    profile = model.build({"top_k": 2}, _workload(api))
    phases = {phase.name: phase for phase in profile.phases}
    assert profile.metadata["input_items"] == 7
    assert profile.metadata["union_items"] == 5
    assert phases["sum"].work["rank_check"] == 10  # A 5-by-2 aligned table.
    assert phases["sum"].work["fp_divide"] == 7  # Only present scores.
    assert phases["sum"].work["fp_add"] == 14  # Offset and accumulation.
    assert profile.outputs["candidates"].items == 2
    all_candidates = model.build({"top_k": 100}, _workload(api))
    other = {phase.name: phase for phase in all_candidates.phases}
    assert phases["sort"] == other["sort"]  # It is not a bounded top-k heap.
    assert all_candidates.outputs["candidates"].items == 5
    assert a.ALGORITHM_MODEL_REGISTRY["rrf"] is r.RRFAlgorithmModel


def test_overlap_changes_table_work_but_not_number_of_input_scores(api):
    _, r = api
    model = r.RRFAlgorithmModel()
    overlapped = model.build({"top_k": 3}, _workload(api, union=4))
    disjoint = model.build({"top_k": 3}, _workload(api, union=7))
    lhs = {p.name: p for p in overlapped.phases}
    rhs = {p.name: p for p in disjoint.phases}
    assert lhs["sum"].work["fp_divide"] == rhs["sum"].work["fp_divide"]
    assert lhs["sum"].work["rank_check"] < rhs["sum"].work["rank_check"]
    assert lhs["align"].working_set_bytes < rhs["align"].working_set_bytes


def test_missing_observations_are_explicit_assumptions(api):
    a, r = api
    data = a.DataProfile(4)
    assert data.bytes is None
    profile = r.RRFAlgorithmModel().build({"top_k": 2}, a.AlgorithmWorkload(
        inputs={"a": data, "b": data},
    ))
    assert profile.metadata["union_items"] == 8
    assert "upper bound" in " ".join(profile.metadata["assumptions"])
    assert "packed" in " ".join(profile.metadata["assumptions"])
    assert profile.metadata["work_basis"] == "structural_estimate"


def test_within_list_deduplication_and_single_list_output_path(api):
    a, r = api
    profile = r.RRFAlgorithmModel().build({"top_k": 1}, a.AlgorithmWorkload(
        inputs={"one": a.DataProfile(5, properties={"unique_items": 3})},
        statistics={"union_items": 3},
    ))
    phases = {p.name: p for p in profile.phases}
    assert phases["align"].work["hash_insert"] == 5
    assert phases["sum"].work["fp_divide"] == 3
    assert phases["output"].work["copy_item"] == 2


def test_inconsistent_union_is_rejected(api):
    _, r = api
    for union in (2, 8):
        with pytest.raises(ValueError, match="inconsistent"):
            r.RRFAlgorithmModel().build({"top_k": 2}, _workload(api, union=union))


def test_generic_profile_checks_dependency_graph_without_system_state(api):
    a, _ = api
    phases = (a.PhaseWork("scan", {"comparison": 8}, 64),
              a.PhaseWork("select", {"comparison": 4}, 32))
    profile = a.OperatorWorkProfile(phases, (("scan", "select"),), {})
    assert all(not hasattr(phase, "latency_s") for phase in profile.phases)
    with pytest.raises(ValueError, match="acyclic"):
        a.OperatorWorkProfile(phases, (("scan", "select"), ("select", "scan")), {})
    with pytest.raises(ValueError, match="unknown"):
        a.OperatorWorkProfile(phases, (("scan", "missing"),), {})
    with pytest.raises(ValueError, match="non-negative"):
        a.PhaseWork("scan", {"comparison": -1}, 64)
