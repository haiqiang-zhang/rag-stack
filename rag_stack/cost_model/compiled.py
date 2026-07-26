"""The compiler-owned workload and bound execution plan used by fixed CM.

There is one physical lowering implementation: ``build_sim_inputs``. This
module owns its in-memory result and public description. Descriptions never
infer a path from configured stages or performance scores. Analysis and
simulation consume the bound object with its runtime pricing functions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
import json
from typing import Any

from rag_stack.cost_model.workload import PreparedExecutionPaths, resolve_execution_paths


SCHEMA_VERSION = "rag-cm.compiled-plan.v1"


class CompilationBlocked(ValueError):
    """A valid request for a preview which CM cannot currently compile."""

    def __init__(self, phase: str, code: str, message: str, **details):
        super().__init__(message)
        self.phase, self.code, self.details = phase, code, details

    def to_dict(self) -> dict:
        return {"status": "blocked", "phase": self.phase, "code": self.code,
                "message": str(self), "details": deepcopy(self.details)}


def _blocked(phase: str, exc: Exception) -> CompilationBlocked:
    code = "unsupported_component" if isinstance(exc, NotImplementedError) else (
        "invalid_rag_ir" if phase == "compile_workload" else "pricing_unavailable"
    )
    return CompilationBlocked(phase, code, str(exc), exception_type=type(exc).__name__)


@dataclass(frozen=True)
class CompiledWorkload:
    pipeline_config: dict
    prepared_paths: PreparedExecutionPaths
    source_question_ids: list
    summary: dict
    normalization: dict

    @property
    def execution_paths(self) -> list:
        return self.prepared_paths.paths


def compile_workload(resolved_pipeline: dict, rag_ir: dict) -> CompiledWorkload:
    """Validate canonical RAG-IR and freeze the cohort before deployment binding."""
    from rag_stack.cost_model.support import require_cm_pipeline_support
    from rag_stack.cost_model.workload import prepare_trace_workload
    from rag_stack.cost_model.sweep_inputs import _get_query_expansion_fanout
    from rag_stack.rag_ir.normalize import with_observed_trace_components

    try:
        require_cm_pipeline_support(resolved_pipeline)
        prepared = prepare_trace_workload(
            rag_ir, query_expansion_fanout=_get_query_expansion_fanout(resolved_pipeline),
        )
        config = with_observed_trace_components(
            deepcopy(resolved_pipeline), prepared.execution_paths,
        )
        return CompiledWorkload(
            config, prepared.prepared_paths, prepared.source_question_ids,
            prepared.summary, prepared.normalization.diagnostics(),
        )
    except CompilationBlocked:
        raise
    except Exception as exc:
        raise _blocked("compile_workload", exc) from exc


def workload_from_paths(config: dict, paths: list, question_ids=None, *,
                        prepared_paths: PreparedExecutionPaths | None = None) -> CompiledWorkload:
    """Adapt raw paths or reuse the assembly's owned execution snapshot."""
    from rag_stack.cost_model.support import require_cm_pipeline_support, require_cm_trace_support
    from rag_stack.cost_model.workload import summarize_trace

    require_cm_pipeline_support(config)
    prepared_paths = resolve_execution_paths(paths, prepared_paths)
    paths = prepared_paths.paths
    require_cm_trace_support(paths)
    ids = list(question_ids) if question_ids is not None else [None] * len(paths)
    if len(ids) != len(paths):
        raise ValueError("source question IDs must align with compiled workload queries")
    return CompiledWorkload(deepcopy(config), prepared_paths, ids, summarize_trace(paths), {})


@dataclass(frozen=True)
class CompiledCMPlan:
    workload: CompiledWorkload
    pipeline_config: dict
    hardware: Any = field(repr=False)
    candidate_row: Any = field(repr=False)
    sim_inputs: tuple = field(repr=False)
    latency: Any = field(repr=False)
    calibration: dict | None
    prefix_caching: bool
    _description: dict = field(repr=False)

    def describe(self) -> dict:
        return deepcopy(self._description)

    def analysis_paths(self) -> list:
        """Project logical calls from the actual lowered worker/engine graph."""
        paths = []
        for rounds in self.sim_inputs[2]:
            calls = []
            for item in rounds:
                for work in item.stage_items:
                    calls.extend(deepcopy(work.calls))
                if item.call is not None:
                    calls.extend(deepcopy(item.call.source_calls))
            paths.append(calls)
        return paths

    def validate_candidate(self, candidate_row) -> None:
        from rag_stack.cost_model.closed_loop_sim import simulation_signature
        if self.latency.validate_pricing_binding is not None:
            self.latency.validate_pricing_binding()
        if simulation_signature(candidate_row, self.pipeline_config) != simulation_signature(
            self.candidate_row, self.pipeline_config,
        ):
            raise CompilationBlocked("consume_plan", "candidate_changed",
                                     "The deployment no longer matches this compiled plan")


def _source(calls: tuple | list) -> dict:
    # compile_execution_paths normalizes step_idx to the original call-array
    # position. Producer step_idx values may contain gaps or duplicates.
    return {"source_call_ids": [call["call_id"] for call in calls if "call_id" in call],
            "source_call_indices": [int(call["step_idx"]) for call in calls],
            "source_node_ids": list(dict.fromkeys(call["node_id"] for call in calls if "node_id" in call))}


def _transfer(transfer) -> dict:
    return {"name": transfer.name, "resource": transfer.resource,
            "payload_bytes": transfer.payload_bytes,
            "bandwidth_bytes_s": transfer.bandwidth_bytes_s,
            "latency_s": transfer.latency_s}


def _describe_bound(workload, pipeline_config, sim_inputs, calibration, prefix_caching,
                    hardware, latency) -> dict:
    stages, engines, queries, cards, population, cores = sim_inputs
    workers = [{"id": f"worker:{i}", "name": stage.name,
                "devices": list(stage.devices), "cards": list(stage.cards),
                "batch_cap": stage.batch_cap, "timeout_s": stage.timeout_s,
                "host_width": stage.host_width,
                "service_binding": "batch_ops" if stage.batch_ops else "service_s"}
               for i, stage in enumerate(stages)]
    engine_rows = [{"id": f"engine:{i}", "name": engine.name, "role": engine.role,
                    "devices": list(engine.devices), "cards": list(engine.cards),
                    "depth": engine.depth, "budget": engine.budget,
                    "prefix_cache_capacity_blocks": engine.prefix_cache_capacity_blocks,
                    "prefix_cache_block_tokens": engine.prefix_cache_block_tokens,
                    "service_binding": {"prefill": bool(engine.prefill_batch_s),
                                        "decode": bool(engine.decode_step_s)}}
                   for i, engine in enumerate(engines)]
    query_rows = []
    for qi, rounds in enumerate(queries):
        operations = []

        def append(kind, name, calls, resource_ids, details, dependencies=None):
            op_id = f"query:{qi}/op:{len(operations)}"
            operations.append({"op_id": op_id, "kind": kind, "name": name,
                               **_source(calls), "resource_ids": resource_ids,
                               "depends_on": (dependencies if dependencies is not None else
                                              [operations[-1]["op_id"]] if operations else []),
                               "details": details})
            return op_id

        for ri, round_ in enumerate(rounds):
            for pos in range(len(round_.stage_idxs) + 1):
                source_calls = (round_.stage_items[pos].calls if pos < len(round_.stage_items)
                                else round_.call.source_calls if round_.call else ())
                for transfer in round_.boundary_transfers.get(pos, ()):
                    append("transfer", transfer.name, transfer.source_calls or source_calls,
                           [f"link:{transfer.resource}"], _transfer(transfer))
                if pos == len(round_.stage_idxs):
                    break
                idx = round_.stage_idxs[pos]
                stage, item = stages[idx], round_.stage_items[pos]
                details = {"round": ri, "input_tokens": item.input_tokens,
                           "output_tokens": item.output_tokens, "input_bytes": item.input_bytes,
                           "multiplicity": item.multiplicity,
                           "logical_stages": [call["stage"] for call in item.calls]}
                if stage.batch_ops is not None:
                    # The same callable supplies the DES's operator sequence.
                    # The sample exposes structure, not an end-to-end prediction.
                    details["operators"] = [
                        {"name": op.name, "resource": op.resource, "host_width": op.host_width,
                         "single_item_service_s": op.service_s}
                        for op in stage.batch_ops((item,))
                    ]
                append("worker", stage.name, item.calls, [f"worker:{idx}"], details)
            call = round_.call
            if call is None:
                continue
            pf = call.engine_pool or (call.engine,)
            dec = call.decode_engine_pool or ((call.decode_engine,) if call.decode_engine is not None else pf)
            pf_id = append("prefill", engines[pf[0]].name, call.source_calls,
                           [f"engine:{i}" for i in pf],
                           {"round": ri, "input_tokens": call.full_in, "effective_input_tokens": call.eff_in,
                            "admission_gate": call.prefill_admission_gate or call.admission_gate,
                            "admission_capacity": call.prefill_admission_capacity or call.admission_capacity})
            dependencies = [pf_id]
            if call.transfers:
                dependencies = []
                for (source, target), transfer in sorted(call.transfers.items()):
                    dependencies.append(append("transfer", transfer.name, call.source_calls,
                                               [f"link:{transfer.resource}"],
                                               {**_transfer(transfer), "source_engine": f"engine:{source}",
                                                "target_engine": f"engine:{target}", "selection": "replica_pair"},
                                               [pf_id]))
            append("decode", engines[dec[0]].name, call.source_calls,
                   [f"engine:{i}" for i in dec],
                   {"round": ri, "output_tokens": call.out,
                    "admission_gate": call.decode_admission_gate or call.admission_gate,
                    "admission_capacity": call.decode_admission_capacity or call.admission_capacity,
                    "dependency_policy": "selected_replica_pair" if call.transfers else "all"}, dependencies)
        query_rows.append({"question_id": workload.source_question_ids[qi], "operations": operations})
    return {"schema_version": SCHEMA_VERSION, "status": "compiled",
            "workload": {"summary": deepcopy(workload.summary),
                         "normalization": deepcopy(workload.normalization),
                         "source_question_ids": deepcopy(workload.source_question_ids)},
            "queries": query_rows,
            "resources": {"workers": workers, "engines": engine_rows,
                          "n_cards": cards, "cpu_cores": cores,
                          "initial_population": population},
            "deployment": deepcopy(pipeline_config.get("system") or {}),
            "calibration": deepcopy(calibration), "prefix_caching": prefix_caching,
            "pricing_binding": deepcopy(getattr(latency, "pricing_binding", {})),
            "hardware": asdict(hardware) if is_dataclass(hardware) else {},
            "prediction": None}


def bind_plan(workload: CompiledWorkload, *, hardware, candidate_row,
              prefix_caching: bool = True, latency=None, stage_batch_profiles=None) -> CompiledCMPlan:
    """Bind one real candidate through the production physical compiler."""
    from rag_stack.cost_model.closed_loop_sim import (
        build_sim_inputs, _pipeline_with_frozen_stage_batch_selection,
    )
    try:
        hardware = deepcopy(hardware)
        config = _pipeline_with_frozen_stage_batch_selection(workload.pipeline_config, hardware)
        capture = {}
        kwargs = {} if stage_batch_profiles is None else {"stage_batch_profiles": stage_batch_profiles}
        sim_inputs, reason = build_sim_inputs(
            config, hardware, workload.execution_paths, candidate_row.get("collocation_strategy", []),
            trace_source_question_ids=(workload.source_question_ids
                                       if all(q is not None for q in workload.source_question_ids) else None),
            batch_size_request=int(candidate_row["batch_size_request"]),
            batch_size_decode=int(candidate_row.get("batch_size_decode", candidate_row["batch_size_request"])),
            prefix_caching=prefix_caching, candidate_row=candidate_row,
            latency=latency, compiled_workload=workload, compilation_capture=capture, **kwargs,
        )
        if sim_inputs is None:
            raise CompilationBlocked("bind_plan", "unpriceable_deployment", reason or "CM rejected deployment")
        description = _describe_bound(workload, capture["pipeline_config"], sim_inputs,
                                      capture["calibration"], prefix_caching, hardware,
                                      capture["latency"])
        # Keep the in-memory description identical to a JSON round trip and
        # reject non-finite pricing/hardware facts before publishing a preview.
        description = json.loads(json.dumps(description, allow_nan=False))
        plan = CompiledCMPlan(workload, capture["pipeline_config"], hardware,
                              deepcopy(dict(candidate_row)), sim_inputs, capture["latency"],
                              capture["calibration"], prefix_caching, description)
        if plan.analysis_paths() != workload.execution_paths:
            raise CompilationBlocked("bind_plan", "unlowered_calls",
                                     "Physical compilation did not preserve every source invocation")
        return plan
    except CompilationBlocked:
        raise
    except Exception as exc:
        raise _blocked("bind_plan", exc) from exc


def describe(plan: CompiledCMPlan) -> dict:
    return plan.describe()


def simulate_plan(plan: CompiledCMPlan):
    """Execute the retained simulator graph, without rebuilding any route."""
    from rag_stack.cost_model.closed_loop_sim import _simulate_to_saturation
    plan.validate_candidate(plan.candidate_row)
    return _simulate_to_saturation(plan.sim_inputs).result
