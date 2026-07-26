"""quality_memo unit semantics: algo-hash keying, full/minimal gating, lazy trace.

The controller-loop behaviour (hit skips the judge, marker file written,
resume seeding) is exercised end-to-end by the RQ1 smoke run; these tests pin
the module's contracts, especially THE keying property — system dims never
change the key, algo dims always do.
"""

from __future__ import annotations

import json

from rag_stack.quality_memo import (
    MEMO_MARKER_FILENAME,
    QualityMemoEntry,
    quality_memo_key,
)


def _cfg(system=None, **algo):
    cfg = dict(algo)
    if system is not None:
        cfg["system"] = system
    return cfg


def test_key_ignores_system_dims():
    base = _cfg(
        pipeline={"sequential": {"generator": {"model": "qwen7b"}}},
        corpus={"chunk_size": 512},
    )
    with_preset = _cfg(
        pipeline={"sequential": {"generator": {"model": "qwen7b"}}},
        corpus={"chunk_size": 512},
        system={"deployment_preset": "disagg_2p2d", "batch_size_request": 256},
    )
    other_preset = _cfg(
        pipeline={"sequential": {"generator": {"model": "qwen7b"}}},
        corpus={"chunk_size": 512},
        system={"deployment_preset": "collocated_tp4", "batch_size_request": 16},
    )
    assert quality_memo_key(base) == quality_memo_key(with_preset)
    assert quality_memo_key(with_preset) == quality_memo_key(other_preset)


def test_key_changes_with_algo_dims():
    a = _cfg(corpus={"chunk_size": 512}, system={"deployment_preset": "x"})
    b = _cfg(corpus={"chunk_size": 1024}, system={"deployment_preset": "x"})
    assert quality_memo_key(a) != quality_memo_key(b)


def test_full_minimal_gating():
    full = QualityMemoEntry({"m": 1.0}, None, full=True)
    minimal = QualityMemoEntry({"m": 1.0}, None, full=False)
    assert full.serves(want_full=True) and full.serves(want_full=False)
    assert minimal.serves(want_full=False)
    assert not minimal.serves(want_full=True)


def test_lazy_dag_loads_once_from_path(tmp_path):
    dag = [{"stage": "generator", "tokens": 42}]
    p = tmp_path / "execution_dag.json"
    p.write_text(json.dumps(dag))
    entry = QualityMemoEntry({"m": 1.0}, None, full=True,
                             execution_dag_path=str(p))
    assert entry.load_execution_dag() == dag
    p.unlink()  # already cached in memory — second load must not re-read
    assert entry.load_execution_dag() == dag


def test_lazy_dag_missing_file_degrades_to_none(tmp_path):
    entry = QualityMemoEntry(
        {"m": 1.0}, None, full=True,
        execution_dag_path=str(tmp_path / "gone.json"),
    )
    assert entry.load_execution_dag() is None
    assert entry.execution_dag_path is None  # cleared, no repeated attempts


def test_marker_filename_stable():
    # persisted in eval dirs — renaming breaks resume accounting
    assert MEMO_MARKER_FILENAME == "quality_memo.json"
