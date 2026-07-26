"""CPU-only structured trace contract checks using real public hooks."""

from copy import deepcopy
import json

import pytest

from rag_stack.evaluation import (
    capture_trace, summarize_items, summarize_lists, trace_batch, trace_query,
    trace_stage,
)
from rag_stack.evaluation import _recording
from rag_stack.rag_ir import make_quality_trace_envelope, normalize_quality_trace
from rag_stack.rag_ir.io_profile import validate_io_profile


def _generator():
    with trace_stage("generator", model_id="test-generator") as call:
        call.input(token_count=3)
        call.output(token_count=1)


def _fusion(**overrides):
    call = {
        "stage": "retrieval_fusion", "model_id": "rrf_pure",
        "input_tokens": 0, "output_tokens": 0, "step_idx": 0,
        "input_profile": summarize_lists([["a", "a", "b"], ["b", "c"]]),
        "output_profile": summarize_items(["a", "b"]),
    }
    call.update(overrides)
    return call


def _envelope(call):
    return make_quality_trace_envelope([[call, {
        "stage": "generator", "input_tokens": 1, "output_tokens": 1, "step_idx": 1,
    }]], question_ids=["q"])


def test_summaries_capture_duplicates_empty_lists_and_overlap_without_ids():
    summary = summarize_lists(iter([["a", "a", "b"], [], iter(["b", "c"])]))
    assert summary == {
        "items": 5, "list_sizes": [3, 0, 2],
        "list_unique_items": [2, 0, 2], "unique_items": 3,
    }
    assert summarize_items(iter(["a", "a", "b"])) == {"items": 3, "unique_items": 2}
    assert summarize_lists([]) == {
        "items": 0, "list_sizes": [], "list_unique_items": [], "unique_items": 0,
    }
    validate_io_profile(summary)
    assert json.loads(json.dumps(summary)) == summary


@pytest.mark.parametrize("profile", [
    {}, {"items": True}, {"items": -1}, {"items": 1.5},
    {"items": 0, "latency_s": 0}, {"items": 1, "ids": ["a"]},
    {"items": 1, "list_sizes": [2]}, {"items": 1, "list_sizes": (1,)},
    {"items": 1, "list_sizes": [True]},
    {"items": 1, "unique_items": 2}, {"items": 1, "unique_items": 0},
    {"items": 1, "list_unique_items": [1]},
    {"items": 2, "list_sizes": [2], "list_unique_items": [1, 1]},
    {"items": 2, "list_sizes": [2], "list_unique_items": [3]},
    {"items": 2, "list_sizes": [2], "list_unique_items": [0]},
    {"items": 6, "list_sizes": [3, 3], "list_unique_items": [2, 2], "unique_items": 1},
    {"items": 6, "list_sizes": [3, 3], "list_unique_items": [2, 2], "unique_items": 5},
])
def test_invalid_structured_profiles_fail_closed(profile):
    with pytest.raises((TypeError, ValueError)):
        validate_io_profile(profile)
    with pytest.raises((TypeError, ValueError)):
        _envelope(_fusion(input_profile=profile))


def test_hooks_capture_structured_only_calls_and_snapshot_profiles_immediately():
    source = summarize_lists([["a", "b"], ["b", "c"]])
    with capture_trace() as session, trace_query("q"):
        with trace_stage("retrieval_fusion", model_id="rrf_pure") as call:
            call.input(profile=source)
            source["list_sizes"][0] = 999
            call.output(profile=summarize_items(["b", "a"]))
        _generator()
    observed = session.finish(expected_question_ids=["q"])["queries"][0]["calls"][0]
    assert observed["input_profile"]["list_sizes"] == [2, 2]
    assert observed["input_tokens"] == observed["output_tokens"] == 0
    assert "input_bytes" not in observed and "output_bytes" not in observed


def test_batch_profiles_follow_question_ids_and_coexist_with_token_bytes():
    with capture_trace() as session, trace_batch(["q2", "q1"]):
        with trace_stage("retrieval_fusion", model_id="rrf_pure") as call:
            call.input(profiles=[summarize_lists([["a"]]), summarize_lists([["b"], ["c"]])])
            call.output(profiles=[summarize_items(["a"]), summarize_items(["c"])])
        with trace_stage("generator", model_id="test-generator") as call:
            call.input(token_counts=[3, 4], byte_counts=[5, 6],
                       profiles=[{"items": 1}, {"items": 2}])
            call.output(token_counts=[1, 1])
    queries = session.finish(expected_question_ids=["q1", "q2"])["queries"]
    assert [q["calls"][0]["input_profile"]["items"] for q in queries] == [2, 1]
    assert queries[0]["calls"][1]["input_tokens"] == 4
    assert queries[0]["calls"][1]["input_bytes"] == 6


@pytest.mark.parametrize("payload", [
    {"profile": {"items": 1}},
    {"profiles": [{"items": 1}]},
    {"profiles": [{"items": 1}, {"items": 2}], "profile": {"items": 1}},
])
def test_batch_profile_attribution_cannot_silently_truncate(payload):
    with pytest.raises(ValueError):
        with capture_trace(), trace_batch(["q1", "q2"]):
            with trace_stage("retrieval_fusion", model_id="rrf_pure") as call:
                call.input(**payload)


def test_profile_does_not_allow_unmeasured_text_tokens():
    with pytest.raises(ValueError, match="explicit tokenizer"):
        with capture_trace(), trace_query("q"):
            with trace_stage("generator", model_id="test-generator") as call:
                call.input(profile={"items": 1}, text="still needs actual token counts")


def test_invalid_profile_cannot_be_retried_to_publish_success():
    with pytest.raises(RuntimeError, match="successful input hook"):
        with capture_trace() as session, trace_query("q"):
            with trace_stage("retrieval_fusion", model_id="rrf_pure") as call:
                with pytest.raises(ValueError):
                    call.input(profile={"items": -1})
                call.input(profile={"items": 1})
                call.output(profile={"items": 1})
    with pytest.raises(RuntimeError, match="failed scope"):
        session.finish(expected_question_ids=["q"])


def test_rrf_cross_profiles_validate_union_and_unique_outputs():
    with pytest.raises(ValueError, match="union bound"):
        _envelope(_fusion(output_profile={"items": 4}))
    with pytest.raises(ValueError, match="unique items"):
        _envelope(_fusion(output_profile={"items": 2, "unique_items": 1}))
    with pytest.raises(ValueError, match="union bound"):
        _envelope(_fusion(
            input_profile={"items": 6, "list_sizes": [3, 3], "list_unique_items": [1, 1]},
            output_profile={"items": 3},
        ))
    # A generic concatenation operator may retain duplicate IDs.
    _envelope(_fusion(model_id="concat", output_profile={"items": 5, "unique_items": 3}))


def test_factory_and_normalization_deep_copy_nested_profiles():
    original = _fusion()
    env = _envelope(original)
    original["input_profile"]["list_sizes"][0] = 999
    assert env["queries"][0]["calls"][0]["input_profile"]["list_sizes"] == [3, 2]
    normalized = normalize_quality_trace(env)
    env["queries"][0]["calls"][0]["input_profile"]["list_sizes"][0] = 999
    assert normalized.traces[0][0]["input_profile"]["list_sizes"] == [3, 2]


def test_recorder_retains_profiles_through_deferred_readout_without_aliases(monkeypatch):
    monkeypatch.setattr(_recording, "count_tokens_batch", lambda texts, model: [7] * len(texts))
    original = _fusion()["input_profile"]
    recorder = _recording.TraceRecorder()
    recorder.record("q", "retrieval_fusion", input_tokens=0, input_profile=original,
                    output_profile={"items": 2}, pending_in_text="deferred")
    original["list_sizes"][0] = 999
    first = recorder.to_trace_v1()
    assert first[0][0]["input_tokens"] == 7
    assert first[0][0]["input_profile"]["list_sizes"] == [3, 2]
    first[0][0]["input_profile"]["list_sizes"][0] = 888
    assert recorder.to_trace_v1()[0][0]["input_profile"]["list_sizes"] == [3, 2]
    selected = recorder.trace_for_qids(["q"])
    selected[0][0]["input_profile"]["list_sizes"][0] = 777
    assert recorder.to_trace_v1()[0][0]["input_profile"]["list_sizes"] == [3, 2]


def test_recorder_batch_profiles_are_copied_and_validate_full_attribution():
    recorder = _recording.TraceRecorder()
    profiles = [summarize_lists([["a"]]), summarize_lists([["b"], ["c"]])]
    saved = deepcopy(profiles)
    recorder.record_batch("retrieval_fusion", qids=["q1", "q2"], input_tokens=[0, 0],
                          input_profiles=profiles, output_profiles=[{"items": 1}, {"items": 1}])
    profiles[0]["list_sizes"][0] = 999
    assert [q[0]["input_profile"] for q in recorder.to_trace_v1()] == saved
    with pytest.raises(ValueError, match="exactly one entry"):
        recorder.record_batch("retrieval_fusion", qids=["q1", "q2"], input_tokens=[0],
                              input_profiles=saved)
