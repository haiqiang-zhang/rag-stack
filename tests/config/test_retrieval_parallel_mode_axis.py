"""IVF parallel mode is pinned to the only standalone-calibrated value, 0.

Mode 1 is fail-closed in the sweep, config validator, and IVFPQConfig until a
dedicated standalone mode-1 measurement grid exists. It never becomes an
optimizer dimension.

Run: python -m pytest tests/config/test_retrieval_parallel_mode_axis.py -x
"""
from __future__ import annotations

from tests import REPO_ROOT

from pathlib import Path

import pytest
import yaml


from rag_stack.cost_model.sweep_inputs import (
    _generate_stage_sweep_df,
    _retrieval_parallel_modes,
)
from rag_stack.taxonomy import Stage

_CONFIG_PATH = (
    REPO_ROOT
    / "configs" / "rag_stack" / "paper_bench_agent_s43.yaml"
)


def _cfg(system: dict | None = None, db_type: str = "faiss_ivf") -> dict:
    return {
        "system": system or {},
        "vectordb": [{"name": "idx", "db_type": db_type}],
        "node_lines": [{
            "node_line_name": "r",
            "nodes": [{
                "stage": "semantic_retrieval",
                "modules": [{"component": "vectordb", "vectordb": "idx"}],
            }],
        }],
    }


# ---------------------------------------------------------------------------
# mode-grid resolution
# ---------------------------------------------------------------------------

def test_default_grid_contains_only_calibrated_mode_zero():
    assert _retrieval_parallel_modes(_cfg()) == [0]


def test_scalar_mode_zero_pin_wins():
    cfg = _cfg({"retrieval": {"faiss_ivf_parallel_mode": 0}})
    assert _retrieval_parallel_modes(cfg) == [0]


def test_scalar_mode_one_pin_fails_closed():
    cfg = _cfg({"retrieval": {"faiss_ivf_parallel_mode": 1}})
    with pytest.raises(ValueError, match="mode-0-only"):
        _retrieval_parallel_modes(cfg)


def test_cm_search_space_grid_overrides_default():
    cfg = _cfg({"cm_search_space": {"faiss_ivf_parallel_mode": [0]}})
    assert _retrieval_parallel_modes(cfg) == [0]


def test_cm_search_space_mode_one_fails_closed():
    cfg = _cfg({"cm_search_space": {"faiss_ivf_parallel_mode": [0, 1]}})
    with pytest.raises(ValueError, match="may contain only 0"):
        _retrieval_parallel_modes(cfg)


def test_list_under_retrieval_rejected():
    cfg = _cfg({"retrieval": {"faiss_ivf_parallel_mode": [0, 1]}})
    with pytest.raises(ValueError, match="SCALAR pin"):
        _retrieval_parallel_modes(cfg)


def test_hnsw_only_config_prices_single_mode():
    assert _retrieval_parallel_modes(_cfg(db_type="faiss_hnsw")) == [0]


# ---------------------------------------------------------------------------
# per-batch argmin inside the stage sweep
# ---------------------------------------------------------------------------

class _StubResult:
    def __init__(self, latency_s: float):
        self.latency_s = latency_s


class _StubRetrievalCM:
    """Records the calibrated mode used by each batch-size price."""

    def __init__(self):
        self.calls = []

    def estimate(self, config: dict, hardware) -> _StubResult:
        mode = int(config["system"]["retrieval"]["faiss_ivf_parallel_mode"])
        bs = None
        for nl in config["node_lines"]:
            for node in nl["nodes"]:
                bs = int(node["modules"][0]["batch_size"])
        self.calls.append((bs, mode))
        return _StubResult(0.0018 + 0.0004 * bs)    # batched: high intercept


def test_stage_sweep_prices_mode_zero_only():
    cm = _StubRetrievalCM()
    df = _generate_stage_sweep_df(
        cm, _cfg(), hardware=None,
        stage=Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH.value, max_batch_size=4,
    )
    rows = {int(r["batch_size"]): r for _, r in df.iterrows()}
    assert set(rows) == {1, 2, 4}
    assert {(b, m) for b, m in cm.calls} == {(b, 0) for b in (1, 2, 4)}
    assert int(rows[1]["faiss_ivf_parallel_mode"]) == 0
    assert abs(float(rows[1]["time_ms"]) - 2.2) < 1e-9
    assert int(rows[4]["faiss_ivf_parallel_mode"]) == 0
    assert abs(float(rows[4]["time_ms"]) - 3.4) < 1e-9


def test_stage_sweep_honors_pin():
    cm = _StubRetrievalCM()
    cfg = _cfg({"retrieval": {"faiss_ivf_parallel_mode": 0}})
    df = _generate_stage_sweep_df(
        cm, cfg, hardware=None,
        stage=Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH.value, max_batch_size=2,
    )
    assert {m for _, m in cm.calls} == {0}
    assert set(int(v) for v in df["faiss_ivf_parallel_mode"]) == {0}


# ---------------------------------------------------------------------------
# optimizer space + validator
# ---------------------------------------------------------------------------

def test_builder_emits_no_parallel_mode_dim():
    from rag_stack.search_space import RagSearchSpace, SEP

    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    ss = RagSearchSpace.from_yaml(cfg)
    assert f"system{SEP}faiss_ivf_parallel_mode" not in ss.param_names
    # even an explicit (illegal) list must not become a dim
    cfg.setdefault("system", {}).setdefault("retrieval", {})[
        "faiss_ivf_parallel_mode"] = [0, 1]
    ss2 = RagSearchSpace.from_yaml(cfg)
    assert f"system{SEP}faiss_ivf_parallel_mode" not in ss2.param_names


def test_validator_rejects_retrieval_list():
    from rag_stack.config_validator import ConfigValidator

    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("system", {}).setdefault("retrieval", {})[
        "faiss_ivf_parallel_mode"] = [0, 1]
    joined = "\n".join(ConfigValidator(cfg, check_model_cache=False).validate())
    assert "only the scalar value 0" in joined


# ---------------------------------------------------------------------------
# derived faiss_num_threads injection (regression: an absent system.retrieval
# block must NOT early-return — the derived value is decided in exactly one
# place, and the CM adapters / measured store default to 1 when it is unset,
# silently moving the operating point to single-thread)
# ---------------------------------------------------------------------------

def test_resolve_injects_derived_threads_without_retrieval_block():
    from types import SimpleNamespace

    from rag_stack.performance_context import PerformanceContext

    pc = PerformanceContext.__new__(PerformanceContext)
    pc.hardware = SimpleNamespace(cpu_preset="epyc_9124")  # 32 physical cores
    config = {"system": {"cm_search_space": {"max_batch_size_request": 256}}}
    pc._resolve_system_retrieval(config, {})
    assert config["system"]["retrieval"]["faiss_num_threads"] == 32

    # pinned block keeps the explicit value and the parallel-mode pin
    config2 = {"system": {
        "retrieval": {"faiss_num_threads": 4, "faiss_ivf_parallel_mode": 0},
        "cm_search_space": {"max_batch_size_request": 256},
    }}
    pc._resolve_system_retrieval(config2, {})
    assert config2["system"]["retrieval"]["faiss_num_threads"] == 4
    assert config2["system"]["retrieval"]["faiss_ivf_parallel_mode"] == 0


def test_inline_target_cpu_derives_threads_from_target_not_controller_host(
    monkeypatch,
):
    """Remote system transfer must honor the target's physical core count."""
    from types import SimpleNamespace

    from rag_stack.performance_context import PerformanceContext

    pc = PerformanceContext.__new__(PerformanceContext)
    pc.hardware = SimpleNamespace(cpu_preset=None, num_cores=128)
    monkeypatch.setattr("rag_stack.performance_context.os.cpu_count", lambda: 64)
    config = {"system": {
        "retrieval": {
            "faiss_num_threads": None,
            "faiss_ivf_parallel_mode": 0,
        },
        "cm_search_space": {"max_batch_size_request": 256},
    }}

    pc._resolve_system_retrieval(config, {})

    assert config["system"]["retrieval"]["faiss_num_threads"] == 128


def test_explicit_null_num_threads_means_derive():
    """`faiss_num_threads: null` in YAML = documented intent to DERIVE
    min(batch, cores) — validator accepts it, the builder emits no dim, and
    both resolve paths fall through to the derived value."""
    from types import SimpleNamespace

    from rag_stack.config_validator import ConfigValidator
    from rag_stack.performance_context import PerformanceContext
    from rag_stack.search_space import RagSearchSpace, SEP

    with open(_CONFIG_PATH) as f:
        base = yaml.safe_load(f)
    base["system"]["retrieval"]["faiss_num_threads"] = None

    errs = [e for e in ConfigValidator(base, check_model_cache=False).validate()
            if "faiss_num_threads" in e]
    assert not errs, errs

    ss = RagSearchSpace.from_yaml(base)
    assert f"system{SEP}faiss_num_threads" not in ss.param_names

    pc = PerformanceContext.__new__(PerformanceContext)
    pc.hardware = SimpleNamespace(cpu_preset="epyc_9124")  # 32 physical cores
    config = {"system": {
        "retrieval": {"faiss_num_threads": None, "faiss_ivf_parallel_mode": 0},
        "cm_search_space": {"max_batch_size_request": 256},
    }}
    pc._resolve_system_retrieval(config, {})
    assert config["system"]["retrieval"]["faiss_num_threads"] == 32
