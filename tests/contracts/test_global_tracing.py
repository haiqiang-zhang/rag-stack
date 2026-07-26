"""CPU contract checks for recorder-free component hooks and context isolation."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from rag_stack.evaluation import (
    TraceSession, capture_trace, get_trace_session, trace_batch, trace_query,
    trace_stage,
)


def generate(prompt_ids):
    """A component accepts only its own inputs, without recorder plumbing."""
    with trace_stage("generator", model_id="external/component") as call:
        call.input(token_ids=prompt_ids)
        answer_ids = [5, 7]
        call.output(token_ids=answer_ids, text="answer")
    return answer_ids


def test_global_hooks_require_explicit_activation():
    assert get_trace_session() is None
    for scope in (trace_query("q"), trace_batch(["q"]),
                  trace_stage("generator", model_id="model")):
        with pytest.raises(RuntimeError, match="no active trace session"):
            with scope:
                pass
    with capture_trace():
        with pytest.raises(RuntimeError, match="query.*batch"):
            generate([1])
    assert get_trace_session() is None


def test_evaluator_capture_connects_components_without_recorder_arguments():
    with capture_trace() as session:
        with trace_query("q"):
            assert generate([1, 2, 3]) == [5, 7]
    assert get_trace_session() is None
    trace = session.finish(expected_question_ids=["q"])
    assert trace["queries"][0]["calls"] == [{
        "stage": "generator", "input_tokens": 3, "output_tokens": 2,
        "output_bytes": 6, "step_idx": 0, "model_id": "external/component",
        "call_id": "call_0",
    }]


def test_global_batch_and_explicit_session_scopes_share_the_contract():
    session = TraceSession()
    with capture_trace(session):
        with trace_batch(["q1", "q2"]):
            with trace_stage("generator", model_id="batched") as call:
                call.input(token_counts=[4, 8])
                call.output(token_counts=[1, 2])
    with session.query("q3"):
        assert get_trace_session() is session
        generate([2, 3, 4])
    assert get_trace_session() is None
    trace = session.finish(expected_question_ids=["q1", "q2", "q3"])
    assert [q["calls"][0]["input_tokens"] for q in trace["queries"]] == [4, 8, 3]


def test_nested_captures_and_explicit_queries_restore_after_failure():
    with capture_trace() as outer:
        with trace_query("outer"):
            with capture_trace() as inner:
                assert get_trace_session() is inner
                with trace_query("inner"):
                    generate([1])
            assert get_trace_session() is outer
            failed = TraceSession()
            with pytest.raises(ValueError, match="component failed"):
                with failed.query("failed"):
                    assert get_trace_session() is failed
                    raise ValueError("component failed")
            assert get_trace_session() is outer
            generate([1, 2])
    assert get_trace_session() is None
    assert inner.finish(expected_question_ids=["inner"])["queries"][0]["question_id"] == "inner"
    assert outer.finish(expected_question_ids=["outer"])["queries"][0]["calls"][0]["input_tokens"] == 2
    with pytest.raises(RuntimeError, match="failed scope"):
        failed.finish(expected_question_ids=["failed"])


def test_capture_exception_outside_query_invalidates_completed_trace():
    with pytest.raises(ValueError, match="evaluation failed"):
        with capture_trace() as session:
            with trace_query("q"):
                generate([1])
            raise ValueError("evaluation failed")
    assert get_trace_session() is None
    with pytest.raises(RuntimeError, match="failed scope"):
        session.finish(expected_question_ids=["q"])


def test_async_workers_inherit_shared_capture_and_keep_queries_separate():
    async def run():
        ready = asyncio.Event()

        async def worker(qid, count):
            assert get_trace_session() is session
            with trace_query(qid):
                with trace_stage("generator", model_id="async") as call:
                    call.input(token_count=count)
                    if qid == "q2":
                        ready.set()
                    await ready.wait()
                    await asyncio.sleep(0)
                    call.output(token_count=count + 1)

        with capture_trace() as session:
            await asyncio.gather(worker("q1", 2), worker("q2", 4))
        return session.finish(expected_question_ids=["q1", "q2"])

    trace = asyncio.run(run())
    assert [q["calls"][0]["input_tokens"] for q in trace["queries"]] == [2, 4]
    assert [q["calls"][0]["output_tokens"] for q in trace["queries"]] == [3, 5]
    assert get_trace_session() is None


def test_concurrent_async_evaluator_captures_are_isolated():
    async def run():
        ready = asyncio.Event()

        async def evaluator(qid, count):
            with capture_trace() as session:
                with trace_query(qid):
                    if qid == "q2":
                        ready.set()
                    await ready.wait()
                    await asyncio.sleep(0)
                    assert get_trace_session() is session
                    generate(list(range(count)))
            return session.finish(expected_question_ids=[qid])

        return await asyncio.gather(evaluator("q1", 2), evaluator("q2", 4))

    traces = asyncio.run(run())
    assert [trace["queries"][0]["question_id"] for trace in traces] == ["q1", "q2"]
    assert [trace["queries"][0]["calls"][0]["input_tokens"] for trace in traces] == [2, 4]
    assert get_trace_session() is None


def test_thread_workers_activate_the_shared_session_explicitly():
    barrier = threading.Barrier(2)

    def worker(qid, session):
        assert get_trace_session() is None
        with capture_trace(session):
            with trace_query(qid):
                barrier.wait(timeout=5)
                generate([1, 2])
        assert get_trace_session() is None

    with capture_trace() as session:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [pool.submit(worker, qid, session) for qid in ["q1", "q2"]]
            for result in results:
                result.result()
        assert get_trace_session() is session
    assert len(session.finish(expected_question_ids=["q1", "q2"])["queries"]) == 2


def test_capture_cannot_reactivate_a_finished_session():
    with capture_trace() as session:
        with trace_query("q"):
            generate([1])
    session.finish(expected_question_ids=["q"])
    with pytest.raises(RuntimeError, match="finished"):
        with capture_trace(session):
            pass
    with pytest.raises(TypeError, match="TraceSession"):
        with capture_trace(object()):
            pass
    assert get_trace_session() is None
