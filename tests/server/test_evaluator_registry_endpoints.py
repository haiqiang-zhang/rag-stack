"""The editor must describe the same installed evaluators as the CLI."""

import pytest
from fastapi import HTTPException

from rag_stack.server.routers import rag_project
from rag_stack.evaluation import registry as factory
from rag_stack.evaluation.registry import BackendSpec


@pytest.fixture
def external_backend(monkeypatch):
    monkeypatch.setattr(factory, "_BACKENDS", dict(factory._BACKENDS))
    monkeypatch.setattr(factory, "_entry_points", lambda: ())
    factory.register_backend(BackendSpec(
        name="example_eval", factory=lambda context: None,
        metrics=("external_quality",), pipeline_modes=("sequential", "custom_loop"),
    ))
    monkeypatch.setattr(
        rag_project, "_warm_static_rag_evaluator",
        lambda: pytest.fail("External metadata must not initialize the static evaluator"),
    )
    return "example_eval"


def test_editor_discovers_external_metrics_and_modes(external_backend):
    assert external_backend in rag_project.get_config_enums()["eval_backend"]
    assert rag_project.get_metrics(external_backend) == {
        "backend": external_backend, "requires_model": False, "metrics": ["external_quality"],
    }
    assert rag_project.get_pipeline_methods(external_backend) == {
        "methods": ["sequential", "custom_loop"],
    }


@pytest.mark.parametrize("endpoint", [rag_project.get_metrics, rag_project.get_pipeline_methods])
def test_unknown_backend_is_not_silently_replaced_by_static(endpoint):
    with pytest.raises(HTTPException) as caught:
        endpoint("unavailable_evaluator")
    assert caught.value.status_code == 400
    assert "Unknown global.eval_backend" in caught.value.detail


def test_external_scaffold_does_not_return_a_different_backend(external_backend):
    with pytest.raises(HTTPException) as caught:
        rag_project.scaffold_config(rag_project.ScaffoldRequest(eval_backend=external_backend))
    assert caught.value.status_code == 400
    assert "example supplied by its evaluator plugin" in caught.value.detail
