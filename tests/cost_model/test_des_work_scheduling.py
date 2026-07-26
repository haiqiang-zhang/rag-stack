"""CPU-only scheduling checks for authoritative CM DES sweeps."""

from concurrent.futures import Future

import pytest

import rag_stack.cost_model.closed_loop_sim as closed_loop_sim


def _work(ordinal: int) -> closed_loop_sim._CandidateWork:
    topology = (f"layout-{ordinal % 17}",)
    request_batch = (8, 16, 32, 64)[ordinal % 4]
    decode_batch = (16, 32, 64, 128)[ordinal % 4]
    timeout = (0.002, 0.01, 0.05)[ordinal % 3]
    row = {
        "batch_size_request": request_batch,
        "batch_size_decode": decode_batch,
        "dynamic_batch_timeout_s": timeout,
    }
    signature = (
        topology,
        None,
        request_batch,
        decode_batch,
        timeout,
        1,
        None,
        (),
        (("batch_size_decode", decode_batch),),
        (),
    )
    return closed_loop_sim._CandidateWork(
        ordinal=ordinal,
        row=row,
        signature=signature,
    )


def _chunk_ordinals(chunks):
    return [[work.ordinal for work in chunk] for chunk in chunks]


def test_candidate_tasks_are_deterministic_complete_and_bounded():
    workers = 5
    works = [_work(ordinal) for ordinal in range(257)]

    first = closed_loop_sim._candidate_chunks(works, workers)
    second = closed_loop_sim._candidate_chunks(works, workers)

    assert workers < len(first)
    assert len(first) <= (
        workers * closed_loop_sim._DES_TASKS_PER_WORKER
    )
    assert all(first)
    assert _chunk_ordinals(first) == _chunk_ordinals(second)
    assert sorted(
        work.ordinal for chunk in first for work in chunk
    ) == list(range(len(works)))


class _ImmediatePool:
    def __init__(self, *, duplicate_first: bool = False):
        self.duplicate_first = duplicate_first
        self.submitted = []

    def submit(self, _function, task):
        chunk_id = task[0]
        candidate_rows = task[-1]
        self.submitted.append(tuple(work.ordinal for work in candidate_rows))
        results = [
            (work.ordinal, f"result-{work.ordinal}", None, "")
            for work in reversed(candidate_rows)
        ]
        if self.duplicate_first and chunk_id == 0:
            results.append(results[0])
        future = Future()
        future.set_result((chunk_id, results))
        return future


def test_parallel_candidate_tasks_use_worker_cap_and_restore_row_order(
    monkeypatch,
):
    works = [_work(ordinal) for ordinal in range(73)]
    workers = 3
    pool = _ImmediatePool()
    requested_process_counts = []

    def fake_pool(process_count):
        requested_process_counts.append(process_count)
        return pool

    monkeypatch.setattr(closed_loop_sim, "_des_process_pool", fake_pool)

    evaluated = closed_loop_sim._evaluate_candidates_in_processes(
        works,
        workers=workers,
        pipeline_config={},
        hardware=object(),
        traces=[],
        prefix_caching=True,
    )

    assert requested_process_counts == [workers]
    assert workers < len(pool.submitted)
    assert len(pool.submitted) <= (
        workers * closed_loop_sim._DES_TASKS_PER_WORKER
    )
    assert sorted(ordinal for task in pool.submitted for ordinal in task) == (
        list(range(len(works)))
    )
    assert [item[0] for item in evaluated] == list(range(len(works)))
    assert [item[1] for item in evaluated] == [
        f"result-{ordinal}" for ordinal in range(len(works))
    ]


def test_parallel_candidate_tasks_still_fail_closed_on_duplicate_result(
    monkeypatch,
):
    pool = _ImmediatePool(duplicate_first=True)
    monkeypatch.setattr(
        closed_loop_sim,
        "_des_process_pool",
        lambda _process_count: pool,
    )

    with pytest.raises(RuntimeError, match="duplicate DES signature ordinal"):
        closed_loop_sim._evaluate_candidates_in_processes(
            [_work(ordinal) for ordinal in range(20)],
            workers=2,
            pipeline_config={},
            hardware=object(),
            traces=[],
            prefix_caching=True,
        )


def test_des_serial_opt_out_is_unchanged(monkeypatch):
    monkeypatch.setenv("CM_DES_SERIAL", "1")
    monkeypatch.setenv("CM_DES_WORKERS", "30")

    assert closed_loop_sim._des_worker_count(2_007) == 1
