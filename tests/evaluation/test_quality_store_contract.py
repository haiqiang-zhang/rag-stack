"""QualityStore cache invariants.

These tests stay at the file/contract boundary: no evaluator, model, CUDA, or
benchmark service is started. A record is one indivisible
``(canonical algo config, quality metrics, quality trace)`` observation.
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

import rag_stack.quality_store as quality_store_module
from rag_stack.quality_store import QualityStore
from rag_stack.rag_ir import make_quality_trace_envelope


@pytest.mark.parametrize("marker", ["summary.json", "config.freeze.json", "rag_stack_config.yaml"])
def test_project_directory_cannot_be_quality_store_import_destination(tmp_path, marker):
    target = tmp_path / "project"
    target.mkdir()
    marker_path = target / marker
    marker_path.write_text("retained project state")
    with pytest.raises(ValueError, match="destination is a project"):
        QualityStore.from_project(str(tmp_path / "source"), str(target))
    assert marker_path.read_text() == "retained project state"
    assert not (target / "manifest.json").exists()


@pytest.mark.parametrize("marker", ["summary.json", "config.freeze.json"])
def test_quality_sweep_cannot_recreate_config_in_a_frozen_project(tmp_path, marker):
    target = tmp_path / "project"
    target.mkdir()
    (target / marker).write_text("retained project state")
    # The guard runs before config loading, dataset setup, or evaluator creation.
    with pytest.raises(ValueError, match="destination is a project"):
        QualityStore(config_path=str(tmp_path / "other.yaml"), store_path=str(target))
    assert not (target / "rag_stack_config.yaml").exists()


def _trace(n_queries: int = 2, *, lossy: bool = False, token_offset: int = 0) -> dict:
    traces = []
    for idx in range(n_queries):
        traces.append([
            {
                "stage": "generator",
                "input_tokens": 20 + token_offset + idx,
                "output_tokens": 4 + idx,
                "step_idx": 0,
                "model_id": "fixture-generator",
            }
        ])
    provenance = {"lossy": True} if lossy else None
    return make_quality_trace_envelope(
        traces,
        question_ids=[str(idx) for idx in range(n_queries)],
        provenance=provenance,
    )


def _algo_config(*, chat: bool, temperature: float = 0.0, system: dict | None = None) -> dict:
    config = {
        "pipeline": {
            "sequential": {
                "generator": {
                    "component": "vllm",
                    "model": "fixture-generator",
                    "temperature": temperature,
                    "use_chat_template": chat,
                },
                "rag_dataflow": "sequential",
            }
        }
    }
    if system is not None:
        config["system"] = system
    return config


def _pipeline_config(*, chat: bool, temperature: float = 0.0) -> dict:
    return {
        "pipeline_runtime": {"mode": "sequential"},
        "node_lines": [
            {
                "node_line_name": "post_retrieve_node_line",
                "nodes": [
                    {
                        "stage": "generator",
                        "modules": [
                            {
                                "component": "vllm",
                                "model": "fixture-generator",
                                "temperature": temperature,
                                "use_chat_template": chat,
                            }
                        ],
                    }
                ],
            }
        ],
        # This is deliberately deployment-only and must not enter the record key.
        "system": {"gpu": "fixture-gpu"},
    }


def _bare_store(path: Path, *, qa_rows: int = 2) -> QualityStore:
    """Construct the lightweight append surface without initializing an evaluator."""
    store = object.__new__(QualityStore)
    store.store_path = str(path)
    store._evaluated = {}
    store._manifest = {}
    store._combined_quality_spec = {
        "mode": "mean",
        "metrics": ["answer_correctness"],
    }
    store.dataset_identity = {"qa_rows": qa_rows}
    return store


def _append(
    store: QualityStore,
    *,
    score: float,
    trace: dict,
    source_mtime_ns: int,
) -> tuple[str, bool]:
    return store.append_record(
        _algo_config(chat=True),
        _pipeline_config(chat=True),
        {"answer_correctness": score},
        trace,
        full=True,
        source={
            "kind": "test",
            "source_mtime_ns": source_mtime_ns,
        },
    )


def test_algo_key_ignores_system_but_distinguishes_chat_from_raw():
    raw_a = _algo_config(
        chat=False,
        system={"deployment_preset": "collocated", "batch_size_request": 8},
    )
    raw_b = _algo_config(
        chat=False,
        system={"deployment_preset": "disaggregated", "batch_size_request": 256},
    )
    chat = _algo_config(
        chat=True,
        system={"deployment_preset": "collocated", "batch_size_request": 8},
    )

    assert QualityStore.key_for_config(raw_a) == QualityStore.key_for_config(raw_b)
    assert QualityStore.key_for_config(raw_a) != QualityStore.key_for_config(chat)


def test_algo_key_normalizes_only_vllm_transport_aliases():
    local = _algo_config(chat=True, temperature=0.25)
    local["pipeline"]["sequential"]["query_expansion"] = {
        "component": "query_decompose",
        "generator_backend": "vllm",
        "model": "fixture-generator",
        "temperature": 0.3,
    }

    api = deepcopy(local)
    api_generator = api["pipeline"]["sequential"]["generator"]
    api_generator["component"] = "vllm_api"
    api_generator["uri"] = "http://vllm.invalid"
    api_expansion = api["pipeline"]["sequential"]["query_expansion"]
    api_expansion["generator_backend"] = "vllm_api"
    api_expansion["uri"] = "http://vllm.invalid"

    implicit_local = deepcopy(local)
    implicit_local["pipeline"]["sequential"]["query_expansion"].pop(
        "generator_backend"
    )

    expected = QualityStore.algorithm_config(implicit_local)
    assert QualityStore.algorithm_config(local) == expected
    assert QualityStore.algorithm_config(api) == expected
    assert QualityStore.key_for_config(local) == QualityStore.key_for_config(api)

    other_backend = deepcopy(implicit_local)
    other_backend["pipeline"]["sequential"]["query_expansion"][
        "generator_backend"
    ] = "openai_llm"
    assert QualityStore.algorithm_config(other_backend)["pipeline"]["sequential"][
        "query_expansion"
    ]["generator_backend"] == "openai_llm"
    assert QualityStore.key_for_config(other_backend) != QualityStore.key_for_config(
        implicit_local
    )

    raw_api = deepcopy(api)
    raw_api["pipeline"]["sequential"]["generator"]["use_chat_template"] = False
    assert QualityStore.key_for_config(raw_api) != QualityStore.key_for_config(api)


def test_algo_key_recursively_ignores_transport_but_keeps_semantics():
    first = _algo_config(chat=True, temperature=0.25)
    generator = first["pipeline"]["sequential"]["generator"]
    generator.update({
        "uri": "http://old-host:8000",
        "batch": 128,
        "request_timeout": 300,
        "project_dir": "/old/project",
        "gpu_memory_utilization": 0.9,
        "fallbacks": [{
            "uri": "http://old-fallback:8001",
            "batch": 64,
        }],
    })
    second = deepcopy(first)
    second_generator = second["pipeline"]["sequential"]["generator"]
    second_generator.update({
        "uri": "http://new-host:9000",
        "batch": 4,
        "request_timeout": 3,
        "project_dir": "/new/project",
        "gpu_memory_utilization": 0.5,
        "fallbacks": [{
            "uri": "http://new-fallback:9001",
            "batch": 2,
        }],
    })

    canonical = QualityStore.algorithm_config(first)
    assert canonical == QualityStore.algorithm_config(second)
    assert QualityStore.key_for_config(first) == QualityStore.key_for_config(second)

    def nested_keys(value) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(
                *(nested_keys(item) for item in value.values()),
            )
        if isinstance(value, (list, tuple)):
            return set().union(*(nested_keys(item) for item in value))
        return set()

    assert nested_keys(canonical).isdisjoint({
        "uri", "batch", "request_timeout", "project_dir",
        "gpu_memory_utilization",
    })
    semantic_change = deepcopy(second)
    semantic_change["pipeline"]["sequential"]["generator"]["temperature"] = 0.75
    assert QualityStore.key_for_config(first) != QualityStore.key_for_config(
        semantic_change,
    )


def test_same_algo_cannot_be_overwritten_by_different_judge_semantics(
    tmp_path: Path,
):
    store = _bare_store(tmp_path / "store")
    common = {
        "combined_quality": {
            "mode": "mean",
            "metrics": ["answer_correctness"],
        },
    }
    store.append_record(
        _algo_config(chat=True),
        _pipeline_config(chat=True),
        {"answer_correctness": 0.5},
        _trace(),
        full=True,
        source={"source_mtime_ns": 1},
        quality_spec={
            **common,
            "metrics": [{
                "metric_name": "answer_correctness",
                "model": "judge-a",
            }],
        },
    )

    with pytest.raises(ValueError, match="different quality/judge semantics"):
        store.append_record(
            _algo_config(chat=True),
            _pipeline_config(chat=True),
            {"answer_correctness": 0.75},
            _trace(token_offset=10),
            full=True,
            source={"source_mtime_ns": 2},
            quality_spec={
                **common,
                "metrics": [{
                    "metric_name": "answer_correctness",
                    "model": "judge-b",
                }],
            },
        )


def test_record_replacement_keeps_metrics_and_trace_as_one_atomic_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store_path = tmp_path / "store"
    store = _bare_store(store_path)
    first_trace = _trace(token_offset=0)
    second_trace = _trace(token_offset=100)

    record_hash, changed = _append(
        store, score=0.25, trace=first_trace, source_mtime_ns=1,
    )
    assert changed is True
    _, changed = _append(
        store, score=0.75, trace=second_trace, source_mtime_ns=2,
    )
    assert changed is True

    reloaded = QualityStore.load(str(store_path))
    [record] = reloaded.get_quality_results()
    assert record["config_hash"] == record_hash
    assert record["quality_metrics"] == {"answer_correctness": 0.75}
    assert reloaded.get_execution_dag(record_hash) == second_trace

    # Simulate a process dying halfway through a later replacement.  The old
    # commit marker must be invalidated before any artifact changes, so a
    # reader rejects the interrupted record instead of pairing new metrics
    # with the old trace.
    real_dump = quality_store_module._atomic_json_dump

    def fail_before_trace(path: str, payload):
        if Path(path).parent.name == "traces":
            raise RuntimeError("simulated interrupted record replacement")
        return real_dump(path, payload)

    monkeypatch.setattr(quality_store_module, "_atomic_json_dump", fail_before_trace)
    with pytest.raises(RuntimeError, match="simulated interrupted"):
        _append(store, score=0.99, trace=_trace(token_offset=200), source_mtime_ns=3)

    with pytest.raises(ValueError, match="incomplete|commit marker"):
        QualityStore.load(str(store_path))


def test_lossy_and_wrong_cardinality_traces_are_not_cacheable(tmp_path: Path):
    store = _bare_store(tmp_path / "store", qa_rows=2)

    with pytest.raises(ValueError, match="lossy"):
        _append(store, score=0.5, trace=_trace(lossy=True), source_mtime_ns=1)

    with pytest.raises(ValueError, match="cardinality"):
        _append(store, score=0.5, trace=_trace(n_queries=1), source_mtime_ns=1)


def test_marker_checksum_damage_is_rejected(tmp_path: Path):
    store_path = tmp_path / "store"
    store = _bare_store(store_path)
    record_hash, _ = _append(
        store, score=0.5, trace=_trace(), source_mtime_ns=1,
    )

    quality_path = store_path / "quality" / f"{record_hash}.json"
    quality_path.write_text(json.dumps({"answer_correctness": 0.125}))

    with pytest.raises(ValueError, match="checksum|sha256|integrity"):
        QualityStore.load(str(store_path))


def test_record_and_manifest_producers_write_current_shape(tmp_path: Path):
    store_path = tmp_path / "store"
    store = _bare_store(store_path)
    record_hash, _ = _append(
        store, score=0.5, trace=_trace(), source_mtime_ns=1,
    )

    marker = json.loads(
        (store_path / "records" / f"{record_hash}.json").read_text()
    )
    manifest = json.loads((store_path / "manifest.json").read_text())
    assert set(marker) >= {
        "algo_hash", "artifact_sha256", "full", "metric_names",
        "quality_spec", "source", "source_mtime_ns", "updated_at",
    }
    assert manifest["kind"] == "quality_store"
    assert manifest["key_kind"] == (
        "enriched_algo_config_without_system_and_transport"
    )


def test_current_store_rejects_transportful_record_identity(tmp_path: Path):
    store_path = tmp_path / "store"
    store = _bare_store(store_path)
    canonical_hash, _ = _append(
        store, score=0.5, trace=_trace(), source_mtime_ns=1,
    )

    config_path = store_path / "configs" / f"{canonical_hash}.json"
    config = json.loads(config_path.read_text())
    config["pipeline"]["sequential"]["generator"]["uri"] = (
        "http://transport-only.invalid"
    )
    transport_hash = quality_store_module._algo_record_key(config)
    assert transport_hash != canonical_hash
    config_path.write_text(json.dumps(config))

    for dirname in ("configs", "quality", "traces", "pipeline_configs", "records"):
        (store_path / dirname / f"{canonical_hash}.json").rename(
            store_path / dirname / f"{transport_hash}.json"
        )

    marker_path = store_path / "records" / f"{transport_hash}.json"
    marker = json.loads(marker_path.read_text())
    marker["algo_hash"] = transport_hash
    marker["artifact_sha256"]["config"] = quality_store_module._sha256_file(
        str(store_path / "configs" / f"{transport_hash}.json")
    )
    marker_path.write_text(json.dumps(marker))

    manifest_path = store_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config_hashes"] = [transport_hash]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="non-canonical algo config"):
        QualityStore.load(str(store_path))


def test_record_marker_loader_and_router_reject_unknown_field(tmp_path: Path):
    from rag_stack.server.routers.quality_store import _committed_hashes

    store_path = tmp_path / "store"
    store = _bare_store(store_path)
    record_hash, _ = _append(
        store, score=0.5, trace=_trace(), source_mtime_ns=1,
    )
    marker_path = store_path / "records" / f"{record_hash}.json"
    marker = json.loads(marker_path.read_text())
    marker["obsolete_header"] = 1
    marker_path.write_text(json.dumps(marker))
    manifest = json.loads((store_path / "manifest.json").read_text())

    with pytest.raises(
        ValueError,
        match="unsupported fields",
    ):
        QualityStore.load(str(store_path))
    assert _committed_hashes(store_path, manifest) == set()


def test_manifest_rejects_unknown_field(tmp_path: Path):
    from rag_stack.server.routers.quality_store import _committed_hashes

    store_path = tmp_path / "store"
    store = _bare_store(store_path)
    _append(store, score=0.5, trace=_trace(), source_mtime_ns=1)
    manifest_path = store_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["obsolete_header"] = 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        ValueError,
        match="unsupported fields",
    ):
        QualityStore.load(str(store_path))
    assert _committed_hashes(store_path, manifest) == set()


def test_manifest_with_unknown_or_wrong_key_kind_fails_closed(tmp_path: Path):
    store_path = tmp_path / "store"
    store = _bare_store(store_path)
    _append(store, score=0.5, trace=_trace(), source_mtime_ns=1)
    manifest_path = store_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["key_kind"] = "enriched_algo_config_without_system"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="key_kind"):
        QualityStore.load(str(store_path))


def _write_dataset(root: Path, *, answer: str = "answer") -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    qa_path = root / "qa.parquet"
    corpus_path = root / "corpus.parquet"
    pd.DataFrame(
        {
            "qid": ["q0", "q1"],
            "query": ["question zero", "question one"],
            "generation_gt": [answer, "second answer"],
        }
    ).to_parquet(qa_path, index=False)
    pd.DataFrame(
        {
            "doc_id": ["d0", "d1"],
            "contents": ["document zero", "document one"],
        }
    ).to_parquet(corpus_path, index=False)
    return qa_path, corpus_path


def _write_project(
    root: Path,
    *,
    qa_path: Path,
    corpus_path: Path,
    score: float,
    temperature: float,
    eval_type: str = "GT",
    eval_error: str = "",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "global": {"eval_backend": "static_gt", "rag_ir_mode": "dynamic"},
        "dataset": {
            "dataset_name": "fixture",
            "qa": str(qa_path),
            "corpus": str(corpus_path),
        },
        "eval_backend_setting": {
            "combined_quality": {
                "mode": "mean",
                "metrics": ["answer_correctness"],
            },
            "metrics": [{"metric_name": "answer_correctness", "model": "mock"}],
        },
        "algo_search_space": {
            "vectordb": [],
            "pipeline": {
                "node_lines": [
                    {
                        "node_line_name": "post_retrieve_node_line",
                        "nodes": [
                            {
                                "stage": "generator",
                                "modules": [
                                    {
                                        "component": "vllm",
                                        "model": "fixture-generator",
                                        "temperature": temperature,
                                        "use_chat_template": True,
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
        },
    }
    (root / "rag_stack_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    algo = _algo_config(chat=True, temperature=temperature)["pipeline"]["sequential"]
    with open(root / "all_evaluations.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["pipeline__sequential", "eval_type", "eval_error", "eval_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pipeline__sequential": repr(algo),
                "eval_type": eval_type,
                "eval_error": eval_error,
                "eval_id": "0",
            }
        )

    eval_dir = root / "evaluations" / "eval_0000"
    eval_dir.mkdir(parents=True)
    (eval_dir / "quality_metrics.json").write_text(
        json.dumps({"answer_correctness": score})
    )
    (eval_dir / "pipeline_config.json").write_text(
        json.dumps(_pipeline_config(chat=True, temperature=temperature))
    )
    (eval_dir / "execution_dag.json").write_text(json.dumps(_trace()))
    return root


def test_source_cm_oom_with_real_quality_and_trace_is_imported(
    tmp_path: Path,
):
    qa_path, corpus_path = _write_dataset(tmp_path / "data")
    project = _write_project(
        tmp_path / "source-cm-oom",
        qa_path=qa_path,
        corpus_path=corpus_path,
        score=0.625,
        temperature=0.3,
        eval_type="OOM",
        eval_error="CM: source H100 system has no feasible deployment",
    )

    store_path = tmp_path / "quality-store"
    QualityStore.from_project(str(project), str(store_path))
    store = QualityStore.load(str(store_path))

    [record] = store.get_quality_results()
    assert record["quality_metrics"] == {"answer_correctness": 0.625}
    assert record["record"]["source_eval_type"] == "OOM"
    assert record["record"]["source_eval_error"].startswith("CM:")
    assert record["record"]["source"]["eval_type"] == "OOM"
    assert set(store.get_execution_dag(record["config_hash"])) >= {"queries"}


def test_non_cm_oom_is_not_reinterpreted_as_reusable_quality(tmp_path: Path):
    qa_path, corpus_path = _write_dataset(tmp_path / "data")
    project = _write_project(
        tmp_path / "quality-harness-oom",
        qa_path=qa_path,
        corpus_path=corpus_path,
        score=0.625,
        temperature=0.3,
        eval_type="OOM",
        eval_error="CUDA out of memory in the quality harness",
    )

    with pytest.raises(RuntimeError, match="No valid GT evaluations"):
        QualityStore.from_project(str(project), str(tmp_path / "quality-store"))


def test_same_corpus_and_qa_append_but_different_dataset_is_rejected(tmp_path: Path):
    shared_qa, shared_corpus = _write_dataset(tmp_path / "shared-data")
    first = _write_project(
        tmp_path / "project-a",
        qa_path=shared_qa,
        corpus_path=shared_corpus,
        score=0.25,
        temperature=0.0,
    )
    second = _write_project(
        tmp_path / "project-b",
        qa_path=shared_qa,
        corpus_path=shared_corpus,
        score=0.75,
        temperature=0.7,
    )
    store_path = tmp_path / "quality-store"

    QualityStore.from_project(str(first), str(store_path), merge=False)
    QualityStore.from_project(str(second), str(store_path), merge=True)
    manifest_before = json.loads((store_path / "manifest.json").read_text())
    assert manifest_before["kind"] == "quality_store"
    assert len(manifest_before["config_hashes"]) == 2

    other_qa, other_corpus = _write_dataset(
        tmp_path / "different-data", answer="different answer",
    )
    different = _write_project(
        tmp_path / "project-c",
        qa_path=other_qa,
        corpus_path=other_corpus,
        score=0.9,
        temperature=1.0,
    )
    with pytest.raises(ValueError, match="corpus/QA|dataset"):
        QualityStore.from_project(str(different), str(store_path), merge=True)

    # A rejected append must leave the committed store index untouched.
    manifest_after = json.loads((store_path / "manifest.json").read_text())
    assert manifest_after["config_hashes"] == manifest_before["config_hashes"]
