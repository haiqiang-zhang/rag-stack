"""Checkpoint CSV cell parsing — the resume-poisoning regression.

all_evaluations.csv stores whole config SECTIONS as python-repr dict strings
(``pipeline__sequential = "{'query_expansion': ...}"``). The checkpoint
loader must parse those back to dicts before ``decode_to_nested``, or every
dim under the blob silently falls back to defaults on re-encode and distinct
evals collapse onto a handful of archive keys — corrupting warm_start for
EVERY optimizer (observed live: 28 evals → 5 greedy archive keys).

Run: python -m pytest tests/evaluation/test_checkpoint_csv_parse.py -q
"""
import pytest

from rag_stack.controller import (
    _checkpoint_penalty_row_is_replayable,
    _parse_checkpoint_csv_cell as parse,
    _restore_checkpoint_selector_choices as restore_selectors,
)


def test_numeric_casts():
    assert parse("42") == 42 and isinstance(parse("42"), int)
    assert parse("0.4") == 0.4 and isinstance(parse("0.4"), float)
    assert parse("-3") == -3


def test_plain_strings_pass_through():
    assert parse("sequential") == "sequential"
    assert parse("Qwen/Qwen2.5-7B-Instruct") == "Qwen/Qwen2.5-7B-Instruct"


def test_dict_blob_parsed_back():
    blob = "{'query_expansion': {'optional': True, 'component': 'hyde', 'top_k': 4}}"
    out = parse(blob)
    assert isinstance(out, dict)
    assert out["query_expansion"]["component"] == "hyde"
    assert out["query_expansion"]["top_k"] == 4


def test_list_blob_parsed_back():
    out = parse("[{'name': 'idx', 'db_type': 'faiss_ivf'}]")
    assert isinstance(out, list) and out[0]["db_type"] == "faiss_ivf"


def test_malformed_blob_passes_through_unchanged():
    bad = "{'unterminated': "
    assert parse(bad) == bad


def test_braceless_strings_never_eval():
    # never literal_eval arbitrary strings — only {…}/[…] cells
    assert parse("None") == "None"
    assert parse("True") == "True"


def test_only_typed_deployment_invalid_is_replayable_as_a_penalty():
    assert _checkpoint_penalty_row_is_replayable({
        "eval_type": "INVALID",
        "eval_error": "deployment_preset: layout exceeds GPU budget",
    })
    assert not _checkpoint_penalty_row_is_replayable({
        "eval_type": "INVALID",
        "eval_error": "CM: scheduler crashed",
    })


def test_legacy_dict_selector_is_restored_from_resolved_runtime():
    decoded = {
        "pipeline": {
            "sequential": {
                "semantic_retrieval": {
                    "component": "vectordb",
                    "vectordb": {
                        "faiss_ivf_index": {"nprobe": [8, 16]},
                        "faiss_hnsw_index": {"ef_search": [32, 64]},
                    },
                    "ef_search": 32,
                },
            },
        },
        "faiss_hnsw_index": {
            "M": 48,
            "ef_construction": 400,
            "embedding_model": "bge-small",
        },
    }
    resolved = {
        "pipeline_runtime": {"mode": "sequential"},
        "node_lines": [
            {
                "nodes": [
                    {
                        "stage": "semantic_retrieval",
                        "modules": [
                            {
                                "component": "vectordb",
                                "vectordb": "faiss_hnsw_index",
                                "ef_search": 32,
                            },
                        ],
                    },
                ],
            },
        ],
    }

    restored = restore_selectors(
        decoded,
        resolved,
        {("pipeline", "sequential", "semantic_retrieval", "vectordb")},
    )
    stage = restored["pipeline"]["sequential"]["semantic_retrieval"]
    assert stage["vectordb"] == "faiss_hnsw_index"
    assert restored["faiss_hnsw_index"] == decoded["faiss_hnsw_index"]
    assert isinstance(decoded["pipeline"]["sequential"]["semantic_retrieval"]["vectordb"], dict)


def test_legacy_active_selector_fails_closed_without_runtime_choice():
    decoded = {
        "pipeline": {
            "sequential": {
                "semantic_retrieval": {
                    "component": "vectordb",
                    "vectordb": {"ivf": {}, "hnsw": {}},
                },
            },
        },
    }
    resolved = {
        "pipeline_runtime": {"mode": "sequential"},
        "node_lines": [],
    }

    with pytest.raises(ValueError, match="cannot be recovered"):
        restore_selectors(
            decoded,
            resolved,
            {("pipeline", "sequential", "semantic_retrieval", "vectordb")},
        )


def test_unrelated_dict_stage_setting_is_not_treated_as_selector():
    decoded = {
        "pipeline": {
            "sequential": {
                "generator": {
                    "component": "vllm",
                    "generation_kwargs": {"top_p": 0.9, "top_k": 20},
                },
            },
        },
    }
    resolved = {
        "pipeline_runtime": {"mode": "sequential"},
        "node_lines": [],
    }

    assert restore_selectors(decoded, resolved, set()) == decoded
