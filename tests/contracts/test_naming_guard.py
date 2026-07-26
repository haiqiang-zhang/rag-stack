"""Naming guards — the taxonomy is the single vocabulary; drift fails loudly.

Source scans enforce active host module boundaries and taxonomy. Historical
result snapshots are excluded.
"""

from __future__ import annotations

from tests import REPO_ROOT

import re
from pathlib import Path

REPO = REPO_ROOT

# External monitor vocabulary is owned and tested by RAG-Stack-Evaluator.
# Only this guard names the retired host tokens.
_OLD_KEY_ALLOW = ("tests/contracts/test_naming_guard.py",)

_OLD_KEY = re.compile(r"\bnode_type\b|\bmodule_type\b")


def _py_and_yaml_files():
    for base in ("rag_stack", "configs", "tests", "benchmarks"):
        for p in (REPO / base).rglob("*"):
            if p.suffix not in (".py", ".yaml", ".yml"):
                continue
            if "__pycache__" in p.parts or "results" in p.parts:
                continue
            yield p


def test_no_old_keys_outside_adapters():
    offenders = []
    for p in _py_and_yaml_files():
        rel = p.relative_to(REPO).as_posix()
        if any(rel.startswith(a) or rel == a for a in _OLD_KEY_ALLOW):
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if _OLD_KEY.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "old node_type/module_type keys:\n" + "\n".join(offenders)


def test_registry_package_is_gone():
    assert not (REPO / "rag_stack" / "registry").exists()
    offenders = []
    for p in (REPO / "rag_stack").rglob("*.py"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if "rag_stack.registry" in line and "taxonomy" not in line:
                offenders.append(f"{p.relative_to(REPO)}:{i}: {line.strip()}")
    for p in (REPO / "tests").rglob("*.py"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"^\s*(from|import)\s+rag_stack\.registry\b", line):
                offenders.append(f"{p.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, "registry references remain:\n" + "\n".join(offenders)


def test_rago_adapter_contained_in_cost_model():
    """The RAGO name adapter is cost-model plumbing: nothing outside
    rag_stack/cost_model may import it (RAGO ids must not leak upward — the
    runtime backstop is assert_no_rago_names on every outbound result df)."""
    offenders = []
    for base in ("rag_stack",):
        for p in (REPO / base).rglob("*.py"):
            rel = p.relative_to(REPO).as_posix()
            if rel.startswith("rag_stack/cost_model"):
                continue
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if re.search(r"^\s*(from|import)\s+\S*rago_adapter", line):
                    offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "rago_adapter leaked out of cost_model:\n" + "\n".join(offenders)


def test_no_legacy_serving_values():
    """`serving: unified|disaggregated` is retired: PD classification is the
    derived pd_serving (collocated_pd|disagg_pd)."""
    pat = re.compile(r'''["']serving["']\s*[:=]''')
    offenders = []
    for p in (REPO / "rag_stack").rglob("*.py"):
        rel = p.relative_to(REPO).as_posix()
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pat.search(line.split("#", 1)[0]):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "legacy layout 'serving' field:\n" + "\n".join(offenders)


def test_cm_never_expands_semantic_retrieval():
    """The stage split is APPLIED only in rag_ir normalization. cost_model
    code must not call expand_stage — except dispatch.py, which consults the
    declared expansion as a fact table to VALIDATE that already-split input
    stages are legitimate for a component (it never rewrites input)."""
    offenders = []
    for p in (REPO / "rag_stack" / "cost_model").rglob("*.py"):
        rel = p.relative_to(REPO).as_posix()
        if rel == "rag_stack/cost_model/dispatch.py":
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"\bexpand_stage\s*\(", line.split("#", 1)[0]):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "expand_stage inside cost_model:\n" + "\n".join(offenders)


def test_server_entrypoints_import():
    """The uvicorn entrypoints must import — no test exercised
    server/cost_model_api, so a stale re-export import (the retired
    MODULE_TYPE_TO_COST_MODEL) only surfaced at server start."""
    from rag_stack.server.cost_model_api import app
    from rag_stack.server.routers.rag_project import router

    assert app.routes and router.routes
