"""Frozen CM input contract: quality-trace envelope and CorpusStats.

The contract: CM accepts exactly (workflow schema, quality trace). The
envelope has ONE semantic — one complete invocation DAG per dataset question —
with hard field whitelists at every layer, and the corpus statistics ride on
the schema instead of pipeline_config side channels."""

import pytest

from rag_stack.rag_ir import (
    CorpusStats,
    WorkflowSchema,
    corpus_stats_from_pipeline_config,
    is_trace_payload,
    make_quality_trace_envelope,
    normalize_quality_trace,
    validate_quality_trace_envelope,
)


def _gen_call(**over):
    call = {
        "stage": "generator",
        "input_tokens": 100,
        "output_tokens": 20,
        "step_idx": 0,
        "model_id": "m",
        "input_bytes": 400,
        "output_bytes": 80,
    }
    call.update(over)
    return call


def _trace():
    return [
        {"stage": "semantic_retrieval_encode", "input_tokens": 7,
         "output_tokens": 0, "step_idx": 0, "model_id": "mpnet"},
        _gen_call(step_idx=1),
    ]


# ---------------------------------------------------------------- factory --

def test_factory_produces_valid_trace_and_projects_diagnostics():
    dirty = [
        dict(_trace()[0], padded_pair_tokens=512, _pad_group=3),
        dict(_trace()[1], latency_s=1.2),
    ]
    env = make_quality_trace_envelope(
        [dirty], question_ids=["q0"], invocation_ids=["warmup-3"],
    )
    assert set(env) == {"queries"}
    assert "trace_scope" not in env
    calls = env["queries"][0]["calls"]
    # recorder-internal diagnostics never leave the factory
    assert all("padded_pair_tokens" not in c for c in calls)
    assert all("_pad_group" not in c for c in calls)
    assert all("latency_s" not in c for c in calls)
    assert env["queries"][0]["question_id"] == "q0"
    assert env["queries"][0]["invocation_id"] == "warmup-3"
    validate_quality_trace_envelope(env)  # idempotent


def test_factory_cardinality_mismatch_raises():
    with pytest.raises(ValueError, match="cardinality"):
        make_quality_trace_envelope([_trace()], question_ids=["a", "b"])


def test_source_question_lineage_is_distinct_from_unique_sample_identity():
    env = make_quality_trace_envelope(
        [[_gen_call(output_tokens=3)], [_gen_call(output_tokens=9)]],
        question_ids=["performance-sample-0", "performance-sample-1"],
        source_question_ids=["dataset-7", "dataset-7"],
        invocation_ids=["measured-18", "measured-41"],
    )
    validate_quality_trace_envelope(env)
    norm = normalize_quality_trace(env)
    assert norm.question_ids == (
        "performance-sample-0", "performance-sample-1",
    )
    assert norm.source_question_ids == ("dataset-7", "dataset-7")


def test_quality_question_id_is_default_stable_source_lineage():
    norm = normalize_quality_trace(_valid_env())
    assert norm.source_question_ids == norm.question_ids


def test_factory_provenance_rides_through():
    env = make_quality_trace_envelope(
        [_trace()], question_ids=["q0"],
        provenance={"lossy": True, "normalization": "legacy_shape_dedup_approx"},
    )
    norm = normalize_quality_trace(env)
    assert norm.lossy is True
    assert norm.normalization == "legacy_shape_dedup_approx"


# ------------------------------------------------------- validation matrix --

def _valid_env():
    return make_quality_trace_envelope(
        [_trace(), [_gen_call()]], question_ids=["q0", "q1"],
    )


def test_unknown_envelope_field_rejected():
    env = dict(_valid_env(), qps=3.0)
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_quality_trace_envelope(env)


def test_unknown_query_field_rejected():
    env = _valid_env()
    env["queries"][0] = dict(env["queries"][0], complete=True)
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_quality_trace_envelope(env)


def test_unknown_call_field_hard_rejected():
    env = _valid_env()
    env["queries"][0]["calls"][0] = dict(
        env["queries"][0]["calls"][0], padded_pair_tokens=512,
    )
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_quality_trace_envelope(env)


def test_missing_question_id_rejected():
    env = _valid_env()
    del env["queries"][0]["question_id"]
    with pytest.raises(ValueError, match="question_id"):
        validate_quality_trace_envelope(env)


def test_duplicate_question_id_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        make_quality_trace_envelope(
            [_trace(), [_gen_call()]], question_ids=["q0", "q0"],
        )


def test_incomplete_query_rejected():
    trace = [_trace()[0]]  # no terminal generator call
    with pytest.raises(ValueError, match="incomplete"):
        make_quality_trace_envelope([trace], question_ids=["q0"])


def test_missing_required_call_field_rejected():
    call = _gen_call()
    del call["step_idx"]
    with pytest.raises(ValueError, match="missing required"):
        make_quality_trace_envelope([[call]], question_ids=["q0"])


def test_bytes_stay_optional():
    call = _gen_call()
    del call["input_bytes"]
    del call["output_bytes"]
    env = make_quality_trace_envelope([[call]], question_ids=["q0"])
    kept = env["queries"][0]["calls"][0]
    assert "input_bytes" not in kept  # absent, NOT zero-filled
    validate_quality_trace_envelope(env)


# ------------------------------------------------------------- normalize --

def test_normalize_passthrough_and_identity():
    env = _valid_env()
    norm = normalize_quality_trace(env)
    assert norm.scope == "quality"
    assert norm.identity == "question_id"
    assert norm.kept_queries == 2
    assert norm.dropped_incomplete == 0
    assert norm.lossy is False
    assert norm.question_ids == ("q0", "q1")
    assert norm.invocation_ids == (None, None)
    assert [c["stage"] for c in norm.traces[0]] == [
        "semantic_retrieval_encode", "generator",
    ]


def test_normalize_rejects_legacy_forms():
    legacy = {
        "trace_scope": "dataset_quality_first_completion",
        "queries": [[_gen_call()]],
    }
    with pytest.raises(ValueError, match="unsupported fields"):
        normalize_quality_trace(legacy)
    with pytest.raises(ValueError, match="unsupported fields"):
        normalize_quality_trace(dict(legacy, trace_scope="measured_only"))
    with pytest.raises(ValueError, match="convert_legacy_traces.py"):
        normalize_quality_trace([[_gen_call()]])   # bare list
    with pytest.raises(TypeError, match="convert_legacy_traces.py"):
        normalize_quality_trace(None)


def test_is_trace_payload_recognizes_current_shape():
    assert is_trace_payload(_valid_env()) is True
    assert is_trace_payload({"shape": "aggregate"}) is False


# ------------------------------------------------ P1 producer persistence --
# The controller dumps ``__execution_dag__`` to execution_dag.json via
# safe_json_dump and the quality memo / resume path reloads it from disk,
# gating on is_trace_payload. An envelope must survive both unchanged.

def test_envelope_survives_safe_json_dump_unchanged(tmp_path):
    import json

    from rag_stack.security import safe_json_dump

    env = make_quality_trace_envelope(
        [_trace(), [_gen_call()]],
        question_ids=["0", "1"],
        invocation_ids=["warmup-3", "measured-11"],
    )
    path = tmp_path / "execution_dag.json"
    with open(path, "w") as f:
        safe_json_dump(env, f, indent=2, default=str)
    with open(path) as f:
        reloaded = json.load(f)
    assert reloaded == env  # no key stripped, no value rewritten


def test_envelope_round_trips_through_quality_memo_path(tmp_path):
    from rag_stack.quality_memo import QualityMemoEntry
    from rag_stack.security import safe_json_dump

    env = make_quality_trace_envelope(
        [_trace()], question_ids=["0"], invocation_ids=["final-2"],
    )
    dag_path = tmp_path / "execution_dag.json"
    with open(dag_path, "w") as f:
        safe_json_dump(env, f, indent=2, default=str)

    entry = QualityMemoEntry(
        quality_metrics={"exact_match": 1.0},
        execution_dag=None,
        full=True,
        execution_dag_path=str(dag_path),
    )
    loaded = entry.load_execution_dag()
    assert loaded == env
    # The controller's resume/memo gate must admit the envelope so a
    # dynamic-CM memo hit replays it instead of degrading.
    assert is_trace_payload(loaded) is True
    norm = normalize_quality_trace(loaded)
    assert norm.scope == "quality"
    assert norm.kept_queries == 1


# ------------------------------------------------- CorpusStats + schema --

_STATS = {
    "avg_chunk_tokens": 96,
    "prompt_template_tokens": 17,
    "avg_output_tokens": 22,
    "avg_query_tokens": 7,
    "bytes_per_token": 4.1,
    "chunk_token_quantiles": [[0.5, 96], [0.9, 180]],
}


def test_corpus_stats_lift_and_materialize_round_trip():
    pcfg = {
        "_token_stats": dict(_STATS),
        "vectordb": [{"name": "faiss_hnsw_index", "N": 8_841_823}],
    }
    stats = corpus_stats_from_pipeline_config(pcfg)
    assert stats is not None
    assert stats.vectordb_index_sizes == {"faiss_hnsw_index": 8_841_823}
    # bit-parity requirement: materializing back reproduces the dict exactly
    assert stats.token_stats_dict() == _STATS


def test_corpus_stats_absent_is_none_but_corrupt_raises():
    assert corpus_stats_from_pipeline_config({}) is None
    with pytest.raises(ValueError, match="missing required"):
        corpus_stats_from_pipeline_config(
            {"_token_stats": {"avg_chunk_tokens": 96}}
        )
    with pytest.raises(ValueError, match="unsupported"):
        corpus_stats_from_pipeline_config(
            {"_token_stats": dict(_STATS, qps=3.0)}
        )


def test_workflow_schema_json_round_trip():
    schema = WorkflowSchema(
        dataflow_type="sequential",
        stages=["semantic_retrieval_encode", "generator_prefill"],
        has_encode=True,
        corpus_stats=CorpusStats.from_dict(
            dict(_STATS, vectordb_index_sizes={"v": 10})
        ),
        runtime_stats={"shape": "aggregate", "num_retrievals": 1},
    )
    rebuilt = WorkflowSchema.from_dict(schema.to_dict())
    assert rebuilt == schema
    with pytest.raises(ValueError, match="unsupported"):
        WorkflowSchema.from_dict(dict(schema.to_dict(), extra=1))
