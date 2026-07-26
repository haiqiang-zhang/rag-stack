"""Lean latency prediction helper for the calibration framework.

Mirrors ``GenZLLMSimBase.estimate()`` but accepts a pre-built GenZ
``System`` object (for ``ceff``/``meff`` control) and returns only the
scalar latency — skips DataFrame assembly, operator breakdown, and
throughput computation.

Kept separate from ``genz_adapter.py`` so the calibration module can be
imported without pulling the entire sweep-generation machinery.
"""

from __future__ import annotations


def predict_latency(
    cost_model, pipeline_config: dict, hardware_dict: dict,
    compute_eff: float = 1.0, memory_eff: float = 1.0, bits: str = "bf16",
    per_call_overhead_s: float = 0.0,
) -> float:
    """Predict latency using a hardware dict + explicit (ceff, meff, bits).

    All four knobs are forwarded as function kwargs to GenZ's
    ``prefill_moddeling`` / ``decode_moddeling``. This avoids constructing
    a ``System`` object, which sidesteps GenZ's silent overwrite of
    pre-built System fields.

    ``bits`` must match the runtime dtype of the production module, e.g.
    ``"f32"`` for TART. GenZ's ``compute_multiplier`` / ``mem_multiplier``
    then correctly scale FLOPS / memory against the BF16-normalized peak.
    """
    from GenZ.LLM_inference import prefill_moddeling, decode_moddeling

    module = cost_model._find_module(pipeline_config)
    if not module:
        return 0.0

    model = cost_model._get_model(pipeline_config)
    params = cost_model._get_params(pipeline_config)
    batch_size = int(module.get("batch_size", 1))
    # Generator parallelism: tp1/pp1 unless the module declares it (so a multi-GPU
    # ceff/meff fit predicts at the SAME (tp,pp) it was measured at — GenZ + ICN_LL
    # model the all-reduce, ceff/meff captures the residual compute/memory efficiency).
    tp = int(module.get("tensor_parallel", 1) or 1)
    pp = int(module.get("pipeline_parallel", 1) or 1)
    # GenZ kinds derived from the single stage taxonomy (was _get_stages()).
    stages = cost_model._stage_kinds(pipeline_config)

    total_s = 0.0

    if "prefill" in stages:
        result = prefill_moddeling(
            model=model, batch_size=batch_size,
            input_tokens=params["input_tokens"],
            system_name=hardware_dict, bits=bits,
            ceff=compute_eff, meff=memory_eff,
            tensor_parallel=tp, pipeline_parallel=pp,
            model_offload=True,
        )
        total_s += result["Latency"] / 1000.0

    if "decode" in stages:
        # KV-capacity clamp (r13): price at the batch the KV pool can hold
        # and scale linearly back — vLLM reduces concurrency gracefully
        # instead of GenZ's catastrophic host-offload path.
        from rag_stack.cost_model.llm_sim.genz_adapter import (
            kv_capped_decode_batch,
        )
        _ctx = int(params["input_tokens"]) + int(params["output_tokens"])
        _b_eff = kv_capped_decode_batch(
            model, batch_size, _ctx, hardware_dict, bits, tp, pp,
        )
        result = decode_moddeling(
            Bb=1,  # vLLM serves beam=1; GenZ default Bb=4 quadrupled KV accounting (fake offload at 14B b256) and decode bytes — r13
            model=model, batch_size=_b_eff,
            input_tokens=params["input_tokens"],
            output_tokens=params["output_tokens"],
            system_name=hardware_dict, bits=bits,
            ceff=compute_eff, meff=memory_eff,
            tensor_parallel=tp, pipeline_parallel=pp,
            model_offload=True,
        )
        result = dict(result)
        result["Latency"] = result["Latency"] * (
            float(batch_size) / float(_b_eff)
        )
        # decode_moddeling returns PER-STEP latency; the full generation runs
        # ``output_tokens`` decode steps. Multiply so this matches the latency a
        # real generate() actually takes — and matches how the e2e RAGO sweep
        # composes it (``model.dec_steps = output_tokens``, StaticAssembly multiplies
        # per-step time_ms by dec_steps). Without this, calibration fits a
        # "prefill + 1 token" prediction against a full-generation measurement.
        total_s += (result["Latency"] / 1000.0) * int(params["output_tokens"])

    # Fixed per-call overhead: charged ONCE per model call (prefill + all
    # decode steps together) — the affine intercept the multiplicative
    # roofline lacks at small batch. 0.0 → bit-identical to the old model.
    total_s += max(0.0, float(per_call_overhead_s))

    return total_s
