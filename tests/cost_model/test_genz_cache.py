from __future__ import annotations

from tests import REPO_ROOT

import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import sys
import time

import pytest

# Keep the repository's pinned GenZ ahead of any independently installed wheel.
# Several cost-model modules import GenZ while their package is initialized.
REPO = REPO_ROOT
if str(REPO / "genz_llm") not in sys.path:
    sys.path.insert(0, str(REPO / "genz_llm"))

from rag_stack.cost_model.llm_sim import genz_cache


@pytest.fixture(autouse=True)
def _isolated_cache_state(monkeypatch):
    genz_cache._reset_cache_state_for_tests()
    monkeypatch.setenv("CM_GENZ_CACHE", "0")
    yield
    genz_cache._reset_cache_state_for_tests()


def _fixed_result(**_kwargs):
    return {"Latency": 1.25, "Throughput": 7.5}


def _counted_result(**_kwargs):
    counter_path = os.environ["GENZ_CACHE_TEST_COUNTER"]
    descriptor = os.open(counter_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    finally:
        os.close(descriptor)
    time.sleep(0.25)
    return {"Latency": 2.5, "Throughput": 9.0}


def _pipeline_result(**_kwargs):
    return {
        "Latency": 2.0,
        "Throughput": 4.0,
        "SaturatedServiceLatency": 2.0,
        "TraversalLatency": 7.0,
    }


def _singleflight_worker(cache_path, counter_path, start, output):
    os.environ["CM_GENZ_CACHE"] = cache_path
    os.environ["GENZ_CACHE_TEST_COUNTER"] = counter_path
    genz_cache._reset_cache_state_for_tests()
    start.wait(10)
    try:
        value = genz_cache._call("prefill", _counted_result, {"tokens": 128})
        output.put(("ok", value))
    except BaseException as error:  # pragma: no cover - surfaced in parent
        output.put(("error", repr(error)))


def test_default_cache_is_node_local_and_namespaced(monkeypatch):
    monkeypatch.delenv("CM_GENZ_CACHE", raising=False)
    path = Path(genz_cache._disk_path())
    assert path.is_absolute()
    assert path.parts[:2] == ("/", "tmp")
    assert "rag-stack-genz-cache-" in str(path.parent.parent)
    assert path.suffix == ".sqlite"


def test_key_preserves_cost_inputs_and_ignores_only_diagnostics():
    base = genz_cache._key("prefill", {
        "model": "model-a",
        "input_tokens": 128,
        "debug": False,
    })
    assert base == genz_cache._key("prefill", {
        "debug": True,
        "input_tokens": 128,
        "model": "model-a",
    })
    assert base != genz_cache._key("prefill", {
        "model": "model-a",
        "input_tokens": 129,
        "debug": False,
    })
    assert base != genz_cache._key("decode", {
        "model": "model-a",
        "input_tokens": 128,
        "debug": False,
    })


def test_disabled_disk_keeps_exact_in_process_memo():
    calls = 0

    def real(**_kwargs):
        nonlocal calls
        calls += 1
        return {"Latency": 3.0, "Throughput": 4.0}

    first = genz_cache._call("decode", real, {"batch": 8})
    second = genz_cache._call("decode", real, {"batch": 8})
    assert first == second == {"Latency": 3.0, "Throughput": 4.0}
    assert calls == 1
    assert genz_cache.cache_stats() == {
        "memo_hit": 1,
        "disk_hit": 0,
        "miss": 1,
    }


def test_sqlite_hit_preserves_exact_scalar_values(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    monkeypatch.setenv("CM_GENZ_CACHE", str(cache_path))
    expected = genz_cache._call("prefill", _fixed_result, {"tokens": 31})
    genz_cache._reset_cache_state_for_tests()

    def must_not_run(**_kwargs):
        raise AssertionError("disk hit unexpectedly recomputed GenZ")

    actual = genz_cache._call("prefill", must_not_run, {"tokens": 31})
    assert actual == expected == {"Latency": 1.25, "Throughput": 7.5}
    assert genz_cache.cache_stats()["disk_hit"] == 1


def test_sqlite_hit_preserves_pipeline_service_and_traversal(
    monkeypatch, tmp_path,
):
    cache_path = tmp_path / "pipeline.sqlite"
    monkeypatch.setenv("CM_GENZ_CACHE", str(cache_path))
    expected = genz_cache._call("decode", _pipeline_result, {"batch": 8})
    genz_cache._reset_cache_state_for_tests()

    def must_not_run(**_kwargs):
        raise AssertionError("disk hit unexpectedly recomputed GenZ")

    actual = genz_cache._call("decode", must_not_run, {"batch": 8})
    assert actual == expected == _pipeline_result()
    assert genz_cache.scheduler_batch_traversal_latency_ms(
        actual,
        pipeline_parallel=4,
    ) == pytest.approx(7.0)


def test_existing_scalar_cache_schema_is_migrated_non_destructively(
    monkeypatch, tmp_path,
):
    cache_path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(cache_path) as connection:
        connection.execute(
            "CREATE TABLE genz "
            "(k TEXT PRIMARY KEY, latency REAL, throughput REAL)"
        )
        connection.execute(
            "INSERT INTO genz VALUES ('legacy-row', 1.0, 2.0)"
        )
        connection.commit()

    monkeypatch.setenv("CM_GENZ_CACHE", str(cache_path))
    assert genz_cache._call(
        "decode", _pipeline_result, {"batch": 16},
    ) == _pipeline_result()

    with sqlite3.connect(cache_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(genz)")
        }
        assert {
            "traversal_latency", "saturated_service_latency",
        }.issubset(columns)
        # Old derived entries are retained; the v5 cache key simply never
        # confuses their missing fields with the new output contract.
        assert connection.execute(
            "SELECT latency, throughput FROM genz WHERE k='legacy-row'"
        ).fetchone() == (1.0, 2.0)


def test_corrupt_cache_is_quarantined_not_deleted(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    corrupt_bytes = b"this is not a sqlite database\x00and must be preserved"
    cache_path.write_bytes(corrupt_bytes)
    monkeypatch.setenv("CM_GENZ_CACHE", str(cache_path))

    assert genz_cache._call("prefill", _fixed_result, {"tokens": 17}) == {
        "Latency": 1.25,
        "Throughput": 7.5,
    }
    quarantined = list(tmp_path.glob("cache.sqlite.corrupt.*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == corrupt_bytes
    with sqlite3.connect(cache_path) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert connection.execute("SELECT count(*) FROM genz").fetchone() == (1,)


def test_environment_path_change_reopens_cache(monkeypatch, tmp_path):
    first_path = tmp_path / "first.sqlite"
    second_path = tmp_path / "second.sqlite"
    monkeypatch.setenv("CM_GENZ_CACHE", str(first_path))
    genz_cache._call("prefill", _fixed_result, {"tokens": 1})
    genz_cache._MEMO.clear()
    monkeypatch.setenv("CM_GENZ_CACHE", str(second_path))
    genz_cache._call("prefill", _fixed_result, {"tokens": 1})
    assert first_path.is_file()
    assert second_path.is_file()


def test_remote_filesystem_is_rejected_before_sqlite_open(monkeypatch, tmp_path):
    cache_path = tmp_path / "remote.sqlite"
    monkeypatch.setenv("CM_GENZ_CACHE", str(cache_path))
    monkeypatch.setattr(genz_cache, "_filesystem_type", lambda _path: "nfs4")
    assert genz_cache._journal_mode("/remote/cache.sqlite") == "DELETE"
    assert genz_cache._get_conn() is None
    assert not cache_path.exists()

    genz_cache._reset_cache_state_for_tests()
    monkeypatch.setattr(genz_cache, "_filesystem_type", lambda _path: "ext4")
    assert genz_cache._journal_mode("/local/cache.sqlite") == "WAL"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX file locks")
def test_multiprocess_singleflight_computes_one_exact_key_once(tmp_path):
    cache_path = str(tmp_path / "shared.sqlite")
    counter_path = str(tmp_path / "calls.txt")
    context = mp.get_context("fork")
    start = context.Event()
    output = context.Queue()
    workers = [
        context.Process(
            target=_singleflight_worker,
            args=(cache_path, counter_path, start, output),
        )
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(20)
    assert [worker.exitcode for worker in workers] == [0, 0, 0, 0]
    results = [output.get(timeout=2) for _ in workers]
    assert results == [
        ("ok", {"Latency": 2.5, "Throughput": 9.0})
    ] * len(workers)
    assert len(Path(counter_path).read_text(encoding="ascii").splitlines()) == 1


def test_singleflight_does_not_swallow_real_function_oserror(monkeypatch, tmp_path):
    monkeypatch.setenv("CM_GENZ_CACHE", str(tmp_path / "cache.sqlite"))

    def fail(**_kwargs):
        raise OSError("GenZ failure")

    with pytest.raises(OSError, match="GenZ failure"):
        genz_cache._call("decode", fail, {"tokens": 3})
