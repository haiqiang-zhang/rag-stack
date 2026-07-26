from __future__ import annotations

from tests import REPO_ROOT

from pathlib import Path


def test_shared_instrumentation_has_no_cost_model_dependency() -> None:
    package_root = (
        REPO_ROOT
        / "RAG-Stack-Evaluator"
        / "rag_stack_evaluator"
        / "vllm_instrumentation"
    )
    assert package_root.is_dir()
    offenders = [
        str(path.relative_to(package_root))
        for path in package_root.rglob("*.py")
        if "rag_stack.cost_model" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_production_launchers_use_shared_instrumentation_paths() -> None:
    repository_root = REPO_ROOT
    production_paths = (
        "rag_stack/cost_model/llm_sim/calibration/main_llm/"
        "offline_vllm_component.py",
        "RAG-Stack-Evaluator/rag_stack_evaluator/static_rag_evaluator/measured/vllm_subprocess.py",
        "RAG-Stack-Evaluator/rag_stack_evaluator/static_rag_evaluator/measured/vllm_pd_pair.py",
    )
    for relative_path in production_paths:
        source = (repository_root / relative_path).read_text(encoding="utf-8")
        assert "rag_stack_evaluator.vllm_instrumentation" in source
