"""Unit tests for the LLM cost model calibration framework.

Run directly (NOT pytest):

    conda run -n rag-stack python tests/calibration/test_llm_sim_calibration_framework.py

Checks:
1. Every registered LLM cost model exposes benchmark/calibrate hooks.
2. `_get_benchmark_runner` raises with a helpful message when no runner
   is registered for the component.
3. `detect_hardware_key` honors the YAML override and fails closed when unset.
4. `_normalize_hardware_key` produces stable keys for common GPUs.
5. `_inject_shape` deep-copies and correctly overrides (batch_size, seq_len).
6. The CLI argument parser accepts both subcommands without crashing.

No GPU required — we only exercise scaffolding.
"""

from __future__ import annotations

from tests import REPO_ROOT

import pytest

import os
import sys

PROJECT_ROOT = str(REPO_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from rag_stack.cost_model import COST_MODEL_REGISTRY
from rag_stack.cost_model.llm_sim.genz_adapter import (
    GenZLLMSimBase,
    PassageRerankerCostModel,
)
from rag_stack.cost_model.llm_sim.calibration import (
    BaseBenchmarkRunner,
    StaticRAGEvaluatorBenchmarkRunner,
    _inject_shape,
    _normalize_hardware_key,
    detect_hardware_key,
)
from rag_stack.cost_model.llm_sim.calibration.reranker import (
    RERANKER_RUNNERS,
    get_runner,
)
from rag_stack.cost_model.llm_sim.cli import build_parser
from rag_stack.types import SystemConfig


# ---------------------------------------------------------------------------
# 1. All LLM cost models expose benchmark/calibrate hooks
# ---------------------------------------------------------------------------

def test_all_llm_cost_models_have_hooks():
    llm_cost_models = [
        cls for cls in COST_MODEL_REGISTRY.values()
        if issubclass(cls, GenZLLMSimBase)
    ]
    assert llm_cost_models, "no GenZLLMSimBase subclasses in registry"
    for cls in llm_cost_models:
        assert hasattr(cls, "benchmark"), cls.__name__
        assert hasattr(cls, "calibrate"), cls.__name__
        assert hasattr(cls, "_get_benchmark_runner"), cls.__name__
        assert hasattr(cls, "_resolve_calibration"), cls.__name__
    print(f"  [1] OK — all {len(llm_cost_models)} GenZLLMSimBase subclasses "
          f"have hooks")


# ---------------------------------------------------------------------------
# 2. Unimplemented component raises a clear error
# ---------------------------------------------------------------------------

def test_unimplemented_runner_raises():
    # PassageRerankerCostModel has the factory wired in; 'flashrank_reranker'
    # isn't registered yet, so it should raise NotImplementedError mentioning it.
    cfg = {
        "node_lines": [{
            "node_line_name": "x",
            "nodes": [{
                "stage": "passage_reranker",
                "top_k": 4,
                "modules": [{"component": "flashrank_reranker"}],
            }],
        }],
    }
    try:
        PassageRerankerCostModel()._get_benchmark_runner(cfg)
    except NotImplementedError as e:
        assert "flashrank_reranker" in str(e), f"error msg must mention the component: {e}"
        print(f"  [2] OK — flashrank_reranker raises: {str(e).splitlines()[0]}")
        return
    raise AssertionError("expected NotImplementedError for unregistered module")


# ---------------------------------------------------------------------------
# 3. TART runner is registered and instantiates lazily
# ---------------------------------------------------------------------------

def test_tart_runner_registered():
    assert "tart" in RERANKER_RUNNERS, \
        f"tart not in RERANKER_RUNNERS={list(RERANKER_RUNNERS)}"
    runner = get_runner("tart")
    assert isinstance(runner, BaseBenchmarkRunner)
    # New convention: must inherit StaticRAGEvaluatorBenchmarkRunner
    assert isinstance(runner, StaticRAGEvaluatorBenchmarkRunner), (
        "reranker runners should subclass StaticRAGEvaluatorBenchmarkRunner "
        "to enforce the 'call into static_rag_evaluator' convention"
    )
    # Lazy module load — no HF weights pulled at construction
    assert runner._module is None, "production module should lazy-load"
    print(f"  [3] OK — tart runner: {type(runner).__name__} "
          f"(StaticRAGEvaluatorBenchmarkRunner, lazy)")


def test_tart_runner_uses_production_class():
    """Convention: module_class() must return the SAME class used in
    production (under static_rag_evaluator/nodes/...). Verifies the
    runner doesn't re-implement TART."""
    from rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker.tart.tart import (
        Tart,
    )
    runner = get_runner("tart")
    cls = runner.module_class()
    assert cls is Tart, (
        f"TartBenchmarkRunner.module_class() returned {cls!r}, "
        f"expected the production Tart class {Tart!r}"
    )
    # build_input must produce a DataFrame the production node can consume.
    # NB: ``retrieval_gt`` (chunk-UUID GT) is deprecated and no longer produced —
    # retrieval quality is now anchored on text (see utils/preprocess.py).
    df = runner.build_input(batch_size=2, seq_len=64)
    required_cols = {"qid", "query", "generation_gt",
                     "retrieved_contents", "retrieved_ids", "retrieve_scores"}
    assert required_cols.issubset(df.columns), (
        f"build_input missing required columns: "
        f"{required_cols - set(df.columns)}"
    )
    assert len(df["retrieved_contents"].iloc[0]) == 2, (
        "retrieved_contents must contain `batch_size` passages"
    )
    print(f"  [3b] OK — tart runner.module_class() == production Tart, "
          f"build_input produces production-shaped DataFrame")


# ---------------------------------------------------------------------------
# 4. Hardware key detection
# ---------------------------------------------------------------------------

def test_normalize_hardware_key():
    cases = {
        "NVIDIA GeForce RTX 3090": "rtx_3090",
        "NVIDIA H100 80GB HBM3": "h100_80gb_hbm3",
        "NVIDIA A100-SXM4-40GB": "a100_sxm4_40gb",
        "Tesla V100-SXM2-16GB": "tesla_v100_sxm2_16gb",
    }
    for raw, expected in cases.items():
        got = _normalize_hardware_key(raw)
        assert got == expected, f"{raw!r}: {got!r} != {expected!r}"
    print(f"  [4a] OK — normalized {len(cases)} GPU names")


def test_detect_hardware_key_yaml_override():
    hw = SystemConfig(num_cores=1, gpu={"hardware_key": "sgs_gpu02"})
    assert detect_hardware_key(hw) == "sgs_gpu02"
    print(f"  [4b] OK — YAML override honored")


def test_detect_hardware_key_never_probes_local_cuda():
    hw = SystemConfig(num_cores=1, gpu={"Flops": 1.0})
    with pytest.raises(ValueError, match="cannot be inferred from the local CUDA"):
        detect_hardware_key(hw)


# ---------------------------------------------------------------------------
# 5. _inject_shape correctly mutates a copy
# ---------------------------------------------------------------------------

def test_inject_shape():
    cfg = {
        "node_lines": [{
            "node_line_name": "x",
            "nodes": [{
                "stage": "passage_reranker",
                "modules": [{"component": "tart", "batch_size": 1}],
            }],
        }],
        "_token_stats": {"avg_chunk_tokens": 999},
    }
    out = _inject_shape(cfg, batch_size=32, seq_len=256)

    assert out is not cfg, "should return a deep copy"
    assert out["_token_stats"]["avg_chunk_tokens"] == 256
    mod = out["node_lines"][0]["nodes"][0]["modules"][0]
    assert mod["batch_size"] == 32

    # Original unchanged
    assert cfg["_token_stats"]["avg_chunk_tokens"] == 999
    assert cfg["node_lines"][0]["nodes"][0]["modules"][0]["batch_size"] == 1
    print(f"  [5] OK — _inject_shape deep-copies + overrides")


# ---------------------------------------------------------------------------
# 6. CLI parser accepts both subcommands
# ---------------------------------------------------------------------------

def test_cli_parser():
    parser = build_parser()
    ns = parser.parse_args([
        "benchmark", "--cost-model", "reranker_sim", "--module", "tart",
        "--config", "configs/rag_stack/sgs_config.yaml",
        "--batch-sizes", "1,2,4",
    ])
    assert ns.command == "benchmark"
    assert ns.cost_model == "reranker_sim"
    assert ns.module == "tart"
    assert ns.batch_sizes == [1, 2, 4]

    ns = parser.parse_args([
        "calibrate", "--cost-model", "reranker_sim", "--module", "tart",
        "--config", "configs/rag_stack/sgs_config.yaml",
        "--output-dir", "/tmp/llm_cal/",
    ])
    assert ns.command == "calibrate"
    assert ns.output_dir == "/tmp/llm_cal/"
    print(f"  [6] OK — CLI parser accepts benchmark + calibrate")


# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("LLM cost model calibration framework — unit tests")
    print("=" * 60)
    test_all_llm_cost_models_have_hooks()
    test_unimplemented_runner_raises()
    test_tart_runner_registered()
    test_tart_runner_uses_production_class()
    test_normalize_hardware_key()
    test_detect_hardware_key_yaml_override()
    test_inject_shape()
    test_cli_parser()
    print("=" * 60)
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
