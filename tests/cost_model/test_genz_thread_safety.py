"""GenZ thread-safety fixes (07-06 root-cause work).

Forensics on the 07-05 "tuple index out of range" concurrency failure pinned
two real hazards; these tests lock in their fixes:

1. ``get_inference_system`` mutated a CALLER-OWNED ``System`` instance in
   place (bits/ceff/meff/parallelism_hierarchy/...), so every System-passing
   caller shared mutable state across concurrent modeling calls. Fixed: the
   function configures and returns a shallow COPY.
2. ``_collocation_pool`` growth killed the LIVE process pool with
   ``cancel_futures=True``, destroying another thread's in-flight strategy
   futures. Fixed: retire-without-cancel + a swap lock.

Plus hygiene: mutable default args (``get_model_df(system=System(),
unit=Unit())``) now construct per call, and modeling entry points clean up
their ``save_layers`` temp CSVs (67k+ orphans had accumulated).
"""

from tests import REPO_ROOT
import inspect
import os
import sys
import time

REPO = str(REPO_ROOT)
if os.path.join(REPO, "genz_llm") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "genz_llm"))


def test_get_inference_system_does_not_mutate_caller():
    from GenZ.LLM_inference.utils import get_inference_system
    from GenZ.system import System

    caller = System()
    before = (caller.compute_efficiency, caller.memory_efficiency,
              caller.parallelism_hierarchy, caller.bits)
    out = get_inference_system(
        system_name=caller, bits="bf16", ceff=0.8, meff=0.9,
        parallelism_hierarchy="TP{2}_EP{1}_PP{1}",
    )
    # configured values land on the RETURNED object...
    assert out.compute_efficiency == 0.8
    assert out.memory_efficiency == 0.9
    assert out.parallelism_hierarchy == "TP{2}_EP{1}_PP{1}"
    # ...and the caller's object is untouched (the 07-05 race surface)
    assert out is not caller
    assert (caller.compute_efficiency, caller.memory_efficiency,
            caller.parallelism_hierarchy, caller.bits) == before


def test_get_model_df_defaults_are_per_call():
    """Mutable default args were shared across all omitting callers."""
    from GenZ.analyse_model import get_model_df, analysis_model

    assert inspect.signature(get_model_df).parameters["system"].default is None
    assert inspect.signature(get_model_df).parameters["unit"].default is None
    assert inspect.signature(analysis_model).parameters["unit"].default is None


def test_modeling_cleans_up_temp_layer_csv():
    """prefill/decode modeling must not leak layer CSVs into /tmp/genz."""
    from GenZ.LLM_inference import prefill_moddeling, decode_moddeling
    from GenZ.Models.get_language_model import DATA_PATH

    model_dir = os.path.join(DATA_PATH, "model")
    os.makedirs(model_dir, exist_ok=True)
    before = set(os.listdir(model_dir))
    prefill_moddeling(model="gpt-2", batch_size=1, input_tokens=64)
    decode_moddeling(model="gpt-2", batch_size=1, input_tokens=64, output_tokens=8)
    leaked = set(os.listdir(model_dir)) - before
    assert not leaked, f"layer CSVs leaked: {sorted(leaked)[:4]}"


def test_concurrent_pricing_matches_single_thread():
    """Two threads modeling concurrently with a SHARED System instance and
    different hierarchies/efficiencies must reproduce their single-thread
    numbers exactly (pre-fix, the in-place System writes corrupted this)."""
    import threading
    from GenZ.LLM_inference import prefill_moddeling
    from GenZ.system import System

    shared = System()
    cfg_a = dict(model="gpt-2", batch_size=1, input_tokens=128,
                 system_name=shared, ceff=0.5, meff=0.5,
                 parallelism_hierarchy="TP{1}_EP{1}_PP{1}")
    cfg_b = dict(model="gpt-2", batch_size=2, input_tokens=256,
                 system_name=shared, ceff=1.0, meff=1.0,
                 parallelism_hierarchy="TP{2}_EP{1}_PP{1}")
    ref_a = prefill_moddeling(**cfg_a)["Latency"]
    ref_b = prefill_moddeling(**cfg_b)["Latency"]

    results = {}
    def _run(tag, cfg, n=6):
        vals = [prefill_moddeling(**cfg)["Latency"] for _ in range(n)]
        results[tag] = vals
    ta = threading.Thread(target=_run, args=("a", cfg_a))
    tb = threading.Thread(target=_run, args=("b", cfg_b))
    ta.start(); tb.start(); ta.join(120); tb.join(120)
    assert results["a"] == [ref_a] * len(results["a"])
    assert results["b"] == [ref_b] * len(results["b"])


def test_colloc_pool_growth_does_not_cancel_inflight():
    """Growing the collocation pool must not cancel in-flight futures."""
    import rag_stack.cost_model.assembly as asm

    asm._collocation_pool_reset()
    try:
        pool_small = asm._collocation_pool(1)
        fut = pool_small.submit(_sleep_and_return, 3.0, 42)
        time.sleep(0.3)                      # let the worker pick it up
        pool_big = asm._collocation_pool(2)   # growth used to kill pool_small
        assert pool_big is not pool_small
        assert fut.result(timeout=60) == 42   # in-flight future survives
    finally:
        asm._collocation_pool_reset()


def _sleep_and_return(seconds, value):
    time.sleep(seconds)
    return value
