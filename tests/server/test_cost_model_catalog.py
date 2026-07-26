"""Catalog follows production dispatch without pretending to compile a plan."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from rag_stack.cost_model import COST_MODEL_REGISTRY
from rag_stack.cost_model.compiled import CompilationBlocked, compile_workload
from rag_stack.cost_model.dispatch import cost_model_for
from rag_stack.rag_ir.normalize import dispatch_manifest
from rag_stack.pipeline_contract import retrieval_component_capabilities
from rag_stack.server.cost_model_catalog import build_cost_model_catalog, mapped_components
from rag_stack.taxonomy import TAXONOMY


def _rows(catalog):
    return [row for stage in catalog["stages"] for row in stage["bindings"]]


@pytest.mark.parametrize("db_type", ["faiss_ivf", "faiss_hnsw"])
def test_dense_catalog_matches_real_normalized_dispatch(db_type):
    config = {
        "vectordb": [{"name": "index", "db_type": db_type}],
        "node_lines": [{"nodes": [
            {"stage": "semantic_retrieval", "modules": [{"component": "vectordb", "vectordb": "index"}]},
            {"stage": "generator", "modules": [{"component": "vllm"}]},
        ]}],
    }
    catalog = build_cost_model_catalog()
    rows = _rows(catalog)
    manifest = dispatch_manifest(config)
    for entry in manifest:
        row, = [row for row in rows
                if (row["dispatch_stage"], row["component"], row["db_type"])
                == (entry.stage.value, entry.component, entry.db_type)]
        assert row["cost_model"] == cost_model_for(entry.stage, entry.component, entry.db_type)
        assert row["status"] == "priced"
    dense, = [stage for stage in catalog["stages"] if stage["stage"] == "semantic_retrieval"]
    assert {(row["dispatch_stage"], row["db_type"]) for row in dense["bindings"]} == {
        ("semantic_retrieval_encode", None),
        ("semantic_retrieval_vectorsearch", "faiss_ivf"),
        ("semantic_retrieval_vectorsearch", "faiss_hnsw"),
    }


@pytest.mark.parametrize("component,stage", [
    ("bm25", "lexical_retrieval"),
    ("hybrid_rrf", "hybrid_retrieval"),
    ("hybrid_cc", "hybrid_retrieval"),
])
def test_unsupported_components_match_compiler_rejection(component, stage):
    row, = [row for row in _rows(build_cost_model_catalog()) if row["component"] == component]
    assert row["status"] == "unsupported"
    assert row["cost_model"] is None
    assert row["performance_stages"] == []
    config = {"node_lines": [{"nodes": [{"stage": stage, "modules": [{"component": component}]}]}]}
    with pytest.raises(CompilationBlocked) as error:
        compile_workload(config, {})
    assert error.value.code == "unsupported_component"
    assert row["reason"] == retrieval_component_capabilities(component)["reason"]
    if component == "hybrid_cc":
        assert "measured" not in row["reason"]


def test_catalog_preserves_unpriced_components_and_observed_fusion_scope():
    catalog = build_cost_model_catalog()
    rows = _rows(catalog)
    assert {row["component"] for row in rows} == set(TAXONOMY.components)
    prompt, = [row for row in rows if row["component"] == "fstring"]
    assert prompt["status"] == "unpriced"
    assert "zero runtime cost" in prompt["reason"]
    fusion, = [stage for stage in catalog["stages"] if stage["stage"] == "retrieval_fusion"]
    assert fusion["user_selectable"] is False
    assert fusion["bindings"][0]["cost_model"] == "rrf"
    assert "rrf_pure" in fusion["bindings"][0]["reason"]
    assert fusion["bindings"][0]["resources"] == ["cpu_retrieval"]
    assert all(row["cost_model"] in COST_MODEL_REGISTRY
               for row in rows if row["status"] == "priced")
    assert catalog["scope"] == "dispatch_catalog"
    assert catalog["contract"]["compiled_plan_schema"] == "rag-cm.compiled-plan.v1"
    assert "readiness verdict" in catalog["contract"]["readiness"]
    assert "prediction" not in catalog and "ready" not in catalog


def test_catalog_http_contract_and_component_ownership():
    from rag_stack.server.cost_model_api import base_router
    from rag_stack.server.routers.rag_project import get_cost_model_mapping

    app = FastAPI()
    app.include_router(base_router, prefix="/api/cost-model")
    client = TestClient(app)
    response = client.get("/api/cost-model/catalog")
    assert response.status_code == 200, response.text
    assert response.json()["schema_version"] == "rag-stack.cost-model-catalog.v1"
    for name in ("retrieval_encode_sim", "faiss_ivf", "faiss_hnsw"):
        assert mapped_components(name) == ["vectordb"]
        info = client.get(f"/api/cost-model/{name}/info")
        assert info.status_code == 200, info.text
        assert info.json()["mapped_module_types"] == ["vectordb"]
        assert client.get(f"/api/cost-model/{name}/modules").json() == {"modules": ["vectordb"]}
    assert get_cost_model_mapping()["catalog_url"] == "/api/cost-model/catalog"
