"""Price observed RRF invocations independently of retrieval configuration.

Evaluator profiles identify input lists, cardinalities, and output size. The
algorithm model describes that exact invocation; the system adapter selects a
matching implementation profile instead of inferring work from query fanout.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from rag_stack.cost_model import register_cost_model
from rag_stack.cost_model.base import CostModelModule
from rag_stack.cost_model.types import CostModelResult


def observed_fusion_calls(pipeline_config: dict) -> tuple[dict, ...]:
    """The assembly-bound evaluator observations used for component tables."""
    calls = pipeline_config.get("_evaluator_fusion_calls")
    if not isinstance(calls, (list, tuple)) or not calls:
        raise ValueError("RRF component pricing requires observed evaluator fusion calls")
    if any(not isinstance(call, dict) or call.get("stage") != "retrieval_fusion"
           for call in calls):
        raise ValueError("_evaluator_fusion_calls must contain retrieval_fusion calls")
    return tuple(calls)


def build_fusion_work_profile(call: Mapping):
    """Build hardware-independent work from one evaluator-observed RRF call."""
    from rag_stack.rag_ir.io_profile import validate_io_profile
    from rag_stack.cost_model.algorithm import AlgorithmWorkload, DataProfile
    from rag_stack.cost_model.rrf_algorithm import RRFAlgorithmModel

    if call.get("stage") != "retrieval_fusion" or call.get("model_id") != "rrf_pure":
        raise ValueError("RRF pricing requires a retrieval_fusion call with model_id='rrf_pure'")
    incoming, outgoing = call.get("input_profile"), call.get("output_profile")
    if incoming is None or outgoing is None:
        raise ValueError("RRF pricing requires observed input_profile and output_profile")
    validate_io_profile(incoming, where="RRF input_profile")
    validate_io_profile(outgoing, where="RRF output_profile")
    sizes = incoming.get("list_sizes")
    if not sizes:
        raise ValueError("RRF input_profile requires non-empty observed list_sizes")
    uniques = incoming.get("list_unique_items")
    union = incoming.get("unique_items")
    bound = incoming["items"] if union is None else union
    if outgoing["items"] > bound:
        raise ValueError("RRF output items exceeds input union cardinality")
    if outgoing.get("unique_items", outgoing["items"]) != outgoing["items"]:
        raise ValueError("RRF output must contain unique candidate IDs")
    inputs = {
        f"candidates_{index}": DataProfile(
            items=size,
            properties={} if uniques is None else {"unique_items": uniques[index]},
        )
        for index, size in enumerate(sizes)
    }
    work = RRFAlgorithmModel().build(
        {"output_items": outgoing["items"]},
        AlgorithmWorkload(inputs=inputs,
                          statistics={} if union is None else {"union_items": union}),
    )
    if work.outputs["candidates"].items != outgoing["items"]:
        raise ValueError("RRF work profile disagrees with observed output cardinality")
    return work


def fusion_service_for_call(call: Mapping, pipeline_config: dict, hardware) -> float:
    """Price one observed invocation using the correct implementation path."""
    from rag_stack.cost_model.calibration_mode import is_uncalibrated_spec_mode
    from rag_stack.cost_model.retrieval_worker_profile import (
        RetrievalWorkerProfileError, _rrf_spec_model, resolve_rrf_fusion_profile,
    )

    work = build_fusion_work_profile(call)
    if work.metadata["input_lists"] > 1:
        return resolve_rrf_fusion_profile(hardware, pipeline_config).service_s(
            work.metadata["input_items"],
        )
    if not is_uncalibrated_spec_mode(pipeline_config, hardware):
        raise RetrievalWorkerProfileError(
            "single-list rrf_pure uses the Python fast path and has no matching "
            "standalone calibration; the multi-list pandas RRF fit cannot price it"
        )
    # No empirical Python coefficient is available for this branch. In the
    # explicitly exploratory mode use only logical traffic and integer work;
    # Python dispatch, allocation and floating-point division are unpriced.
    key = str(getattr(hardware, "cpu_key", "") or getattr(hardware, "cpu_preset", ""))
    system = _rrf_spec_model(hardware, key)
    return sum(max(
        phase.total_bytes / system.single_thread_bandwidth_bytes_s,
        sum(count for kind, count in phase.work.items() if not kind.startswith("fp_"))
        / system.single_core_int_ops_s,
    ) for phase in work.phases)


def fusion_service_per_visit(pipeline_config: dict, hardware) -> float:
    """Observed mean fusion demand per retrieval visit for coarse tables."""
    calls = observed_fusion_calls(pipeline_config)
    ratio = float(pipeline_config.get("_evaluator_fusion_calls_per_retrieval", 1.0))
    if not pd.notna(ratio) or ratio < 0 or ratio == float("inf"):
        raise ValueError("observed fusion calls per retrieval must be finite and nonnegative")
    return ratio * sum(fusion_service_for_call(call, pipeline_config, hardware) for call in calls) / len(calls)


@register_cost_model("rrf")
class RRFCostModel(CostModelModule):
    @classmethod
    def family(cls) -> str:
        return "retrieval"

    def build_work_profile(self, call):
        return build_fusion_work_profile(call)

    def estimate(self, pipeline_config, hardware) -> CostModelResult:
        batch = int((pipeline_config.get("system") or {}).get("batch_size_request", 1))
        if batch < 1:
            raise ValueError("RRF batch_size_request must be positive")
        seconds = batch * fusion_service_per_visit(pipeline_config, hardware)
        return CostModelResult(
            latency_s=seconds,
            throughput_qps=batch / seconds if seconds > 0 else float("inf"),
        )

    def generate_sweep_df(self, pipeline_config, hardware, stage, max_batch_size):
        if stage != "retrieval_fusion":
            raise ValueError(f"RRF cannot price stage {stage!r}")
        per_visit = fusion_service_per_visit(pipeline_config, hardware)
        rows = []
        batch = 1
        while batch <= max_batch_size:
            rows.append({
                "db.name": "rrf", "hardware.name": "system",
                "hardware.num_servers": 1, "batch_size": batch,
                "time_ms": 1000.0 * batch * per_visit,
            })
            batch *= 2
        return pd.DataFrame(rows)
