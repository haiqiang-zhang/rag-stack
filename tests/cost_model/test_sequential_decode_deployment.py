"""Sequential score-vs-deployment consistency + max-batch population rule.

Postmortem (s45_eval_0047): the static/RAGO path priced a sequential row at
generator stage batch 128 while the saved deployment carried decode=2 —
CM scored one machine and archived another (measured 0.598 vs CM 5.70; CM's
own service times at the deployed concurrency reproduce 0.62-0.68). Fixes
under test:
  1. _expand_deployment_axes derives the deployment decode of sequential
     rows from the generator stage batches the pricing assumed.
  2. The measured harness may choose its own finite client population, while
     CM uses only a structural lower bound and lets DES find the saturated
     plateau.  The two policies must not be coupled through performance data.
"""

import pandas as pd

from rag_stack.cost_model.engines import (
    _closed_loop_load_concurrency,
    _expand_deployment_axes,
)
from rag_stack.cost_model.assembly import (
    _materialize_rago_decode_axis,
    _requested_decode_batch_grid,
)
from rag_stack_evaluator.static_rag_evaluator.measured.serving_runtime import (
    _measured_load_concurrency,
)


def _static_df():
    return pd.DataFrame({
        "batch_size_request": [2, 2, 64],
        "batch_size_generator_prefill": [2, 2, 32],
        "batch_size_generator_decode": [2, 128, 32],
        "qps": [0.6, 5.7, 30.0],
    })


def test_sequential_rows_derive_decode_from_stage_batches():
    df = _expand_deployment_axes(
        _static_df(), {}, agentic=False, decode_pinned=False, timeout_pinned=True,
    )
    assert list(df["batch_size_decode"]) == [2, 128, 64]
    # row 0: stage batches ≤ request → decode = request (previous behavior)
    # row 1: the s45_eval_0047 shape → deployment now matches the pricing
    # row 2: stage batches below request floor to request


def test_sequential_derivation_respects_decode_pin():
    df = _expand_deployment_axes(
        _static_df(), {}, agentic=False, decode_pinned=True, timeout_pinned=True,
    )
    assert "batch_size_decode" not in df.columns  # pin path untouched


def test_agentic_axis_expansion_unchanged():
    df = pd.DataFrame({"batch_size_request": [2], "qps": [1.0]})
    out = _expand_deployment_axes(
        df, {"max_batch_size_request": 8}, agentic=True,
        decode_pinned=False, timeout_pinned=True,
    )
    assert sorted(out["batch_size_decode"]) == [2, 4, 8]


def test_agentic_rago_decode_axis_is_materialized_without_second_cross_product():
    df = pd.DataFrame({
        "batch_size_request": [2, 2, 4],
        "batch_size_generator_decode": [2, 8, 8],
        "qps": [1.0, 2.0, 3.0],
    })
    out = _materialize_rago_decode_axis(
        df,
        {"system": {"cm_search_space": {"max_batch_size_request": 8}}},
    )

    assert len(out) == len(df)
    assert out["batch_size_decode"].tolist() == [2, 8, 8]
    assert out["qps"].tolist() == [1.0, 2.0, 3.0]


def test_agentic_rago_decode_axis_rejects_decode_below_request():
    df = pd.DataFrame({
        "batch_size_request": [8],
        "batch_size_generator_decode": [4],
        "qps": [1.0],
    })

    import pytest
    with pytest.raises(ValueError, match="decode batches below"):
        _materialize_rago_decode_axis(
            df, {"system": {}},
        )


def test_agentic_rago_decode_axis_honors_explicit_grid_and_pin():
    df = pd.DataFrame({
        "batch_size_request": [2, 2, 2],
        "batch_size_generator_decode": [2, 4, 8],
        "qps": [1.0, 2.0, 3.0],
    })
    gridded = _materialize_rago_decode_axis(
        df,
        {"system": {"cm_search_space": {"batch_size_decode": [8]}}},
    )
    assert gridded["batch_size_decode"].tolist() == [2, 8]
    assert gridded["qps"].tolist() == [1.0, 3.0]

    pinned = _materialize_rago_decode_axis(
        df, {"system": {"batch_size_decode": 4}},
    )
    assert pinned["batch_size_decode"].tolist() == [4]
    assert pinned["qps"].tolist() == [2.0]


def test_rago_decode_grid_is_cm_owned_not_pareto_owned():
    base = {"system": {"cm_search_space": {}}}
    assert _requested_decode_batch_grid(base, 8) == [1, 2, 4, 8]
    explicit = {"system": {"cm_search_space": {
        "batch_size_decode": [16],
    }}}
    assert _requested_decode_batch_grid(explicit, 8) == [1, 2, 4, 8, 16]
    pinned = {"system": {
        "batch_size_decode": 4,
        "cm_search_space": {},
    }}
    assert _requested_decode_batch_grid(pinned, 8) == [4, 8]


def test_rago_decode_axis_rejects_preexisting_public_mismatch():
    df = pd.DataFrame({
        "batch_size_request": [2],
        "batch_size_generator_decode": [8],
        "batch_size_decode": [4],
        "qps": [1.0],
    })

    import pytest
    with pytest.raises(ValueError, match="axes diverged"):
        _materialize_rago_decode_axis(df, {"system": {}})


def test_cm_analytical_population_is_structural_and_ignores_measured_load():
    assert _closed_loop_load_concurrency({}, 2, 128) == 128
    assert _closed_loop_load_concurrency({}, 64, 64) == 64
    assert _closed_loop_load_concurrency({}, 64, None) == 64
    assert _closed_loop_load_concurrency(
        {"measured_load_concurrency": 7}, 2, 128) == 128


def test_measured_population_remains_a_harness_only_policy():
    scfg = {"batching": {"request": 2, "decode": 128}}
    assert _measured_load_concurrency(scfg, 2) == 512
    assert _measured_load_concurrency(
        {"batching": {"request": 16, "decode": 256}}, 16) == 1024
    # decode ≤ request unchanged
    assert _measured_load_concurrency(
        {"batching": {"request": 64, "decode": 64}}, 64) == 256
    # CM deliberately does not mirror this finite benchmark policy.
    for req, dec in [(16, 32), (16, 64), (16, 256), (64, 256), (128, 128), (128, 256)]:
        scfg = {"batching": {"request": req, "decode": dec}}
        assert _measured_load_concurrency(scfg, req) >= max(req, dec)
        assert _closed_loop_load_concurrency(scfg, req, dec) == max(req, dec)
