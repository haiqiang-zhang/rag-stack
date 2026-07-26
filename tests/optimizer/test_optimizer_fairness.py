"""CPU-only fairness guards for component screening versus DES."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.cost_model import assembly as assembly_module
from rag_stack.cost_model import candidate_reprice
from rag_stack.cost_model import hybrid_screening
from rag_stack.cost_model.candidate_layout import (
    CANDIDATE_LAYOUT_COLUMN,
    candidate_system_layout_from_row,
)
from rag_stack.cost_model.llm_sim import genz_adapter
from rag_stack.rag_ir.workflow_schema import WorkflowSchema
from rag_stack.runtime_parallelism import (
    validate_candidate_frame_runtime_parallelism,
    validate_measured_runtime_parallelism,
)
from rag_stack.types import CMDeviceCapacityError, PerformanceSelection


def _sweep_row(
    batch: int,
    time_ms: float,
    *,
    tp: int,
    pp: int,
) -> dict:
    return {
        "model.name": "unit-model",
        "model.stage": "decode",
        "model.dec_steps": 32,
        "model.seq_len_inference_prefill": 128,
        "hardware.name": "unit-system",
        "hardware.num_chips": 2,
        "batch_size": batch,
        "time_ms": time_ms,
        "winning_tp": tp,
        "winning_pp": pp,
    }


def test_two_role_sweep_keeps_every_batch_after_per_batch_tp_pp_choice(
    monkeypatch,
) -> None:
    frame = pd.DataFrame([
        _sweep_row(1, 4.0, tp=1, pp=2),
        _sweep_row(1, 3.0, tp=2, pp=1),
        _sweep_row(2, 6.0, tp=1, pp=2),
        _sweep_row(2, 5.0, tp=2, pp=1),
        _sweep_row(4, 9.0, tp=1, pp=2),
        _sweep_row(4, 7.0, tp=2, pp=1),
    ])
    pareto_inputs: list[pd.DataFrame] = []

    def drop_larger_batches(winners: pd.DataFrame) -> pd.DataFrame:
        pareto_inputs.append(winners.copy())
        return winners[winners["batch_size"] == 1]

    monkeypatch.setattr(genz_adapter, "_get_pareto_df", drop_larger_batches)

    two_role = genz_adapter._reduce_parallelism_sweep(
        frame,
        preserve_batch_axis=True,
    )
    assert two_role["batch_size"].tolist() == [1, 2, 4]
    assert two_role["winning_tp"].tolist() == [2, 2, 2]
    assert pareto_inputs == []

    single_role = genz_adapter._reduce_parallelism_sweep(
        frame,
        preserve_batch_axis=False,
    )
    assert single_role["batch_size"].tolist() == [1]
    assert len(pareto_inputs) == 1
    assert pareto_inputs[0]["batch_size"].tolist() == [1, 2, 4]


def _layout(*, two_role_llm: bool) -> dict:
    if two_role_llm:
        stages = {
            "generator_prefill": {
                "engine": "generator",
                "role": "prefill",
                "resource": "gpu:0",
                "devices": ["cuda:0"],
                "tp": 1,
                "pp": 1,
                "replicas": 1,
            },
            "generator_decode": {
                "engine": "generator",
                "role": "decode",
                "resource": "gpu:0",
                "devices": ["cuda:0"],
                "tp": 1,
                "pp": 1,
                "replicas": 1,
            },
        }
        engines = {
            "generator": {
                "pd_serving": "collocated_pd",
                "devices": ["cuda:0"],
                "tp": 1,
                "pp": 1,
                "replicas": 1,
            },
        }
    else:
        stages = {
            "semantic_retrieval_encode": {
                "engine": "semantic_retrieval_encode",
                "role": "stage",
                "resource": "gpu:0",
                "devices": ["cuda:0"],
                "tp": 1,
                "pp": 1,
                "replicas": 1,
            },
        }
        engines = {}
    return {
        "device_layout_id": "unit-layout",
        "available_devices": ["cuda:0"],
        "resource_groups": [{
            "id": "gpu:0",
            "kind": "gpu",
            "devices": ["cuda:0"],
            "performance_stages": list(stages),
        }],
        "performance_stages": stages,
        "engines": engines,
    }


def _screen_frame(*, two_role_llm: bool) -> pd.DataFrame:
    layout = _layout(two_role_llm=two_role_llm)
    rows = []
    for position in range(16):
        row = {
            "qps_cap_upper_bound": 100.0 - position,
            "dynamic_analytical_qps": 100.0 - position,
            "batch_size_request": position + 1,
            "batch_size_decode": position + 1,
            "dynamic_batch_timeout_s": 0.002,
            "pipeline_mode": "dynamic",
            "candidate_system_layout": layout,
        }
        if two_role_llm:
            row.update({
                "qps_generator_prefill": 1000.0,
                "qps_generator_decode": 1000.0,
            })
        else:
            row["qps_semantic_retrieval_encode"] = 1000.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_candidate_layout_uses_the_single_current_contract() -> None:
    layout = _layout(two_role_llm=True)

    assert candidate_system_layout_from_row(
        {CANDIDATE_LAYOUT_COLUMN: layout}
    ) is layout


@pytest.mark.parametrize(
    "missing",
    ("available_devices", "resource_groups", "performance_stages", "engines"),
)
def test_candidate_layout_rejects_missing_required_fields(missing) -> None:
    layout = _layout(two_role_llm=True)
    del layout[missing]

    with pytest.raises(KeyError, match=missing):
        candidate_system_layout_from_row({CANDIDATE_LAYOUT_COLUMN: layout})


def test_two_role_candidates_use_trace_analytical_hybrid_ranking(
    monkeypatch,
) -> None:
    monkeypatch.setattr(hybrid_screening, "_MIN_ROWS_FOR_SCREENING", 1)
    monkeypatch.setenv("CM_DES_HYBRID_FORCE", "1")
    monkeypatch.delenv("CM_DES_EXACT_ALL", raising=False)

    llm_frame = _screen_frame(two_role_llm=True)
    result = hybrid_screening.screen_des_candidates(
        llm_frame,
        selection=PerformanceSelection(selection="max_throughput"),
        production=False,
    )
    assert 0 < len(result.selected_positions) < len(llm_frame)
    assert result.summary["mode"] == "hybrid_dynamic_engine_analytical_des"
    assert result.summary["score"] == (
        "dynamic_engine_trace_analytical_qps"
    )

    # A forward-only frame uses the same explicit DynamicEngine score contract.
    forward_frame = _screen_frame(two_role_llm=False)
    forward = hybrid_screening.screen_des_candidates(
        forward_frame,
        selection=PerformanceSelection(selection="max_throughput"),
        production=False,
    )
    assert 0 < len(forward.selected_positions) < len(forward_frame)
    assert forward.summary["mode"] == "hybrid_dynamic_engine_analytical_des"


def test_no_trace_static_fails_before_component_only_llm_sweep(
    monkeypatch,
) -> None:
    assembly = object.__new__(assembly_module.RAGCMAssembly)
    assembly.hardware = SimpleNamespace(performance={})
    schema = WorkflowSchema(
        dataflow_type="sequential",
        stages=["generator_prefill", "generator_decode"],
    )

    def sweep_is_too_late(*_args, **_kwargs):
        raise AssertionError("component-only sweep must not run")

    monkeypatch.setattr(
        assembly_module.RAGCMAssembly,
        "_run_sweep",
        sweep_is_too_late,
    )
    with pytest.raises(RuntimeError, match="no scheduler-cycle composition"):
        assembly_module.RAGCMAssembly.evaluate_static(
            assembly,
            {},
            workflow_schema=schema,
        )

    forward_schema = WorkflowSchema(
        dataflow_type="sequential",
        stages=["semantic_retrieval_encode"],
    )
    assert assembly_module._two_role_llm_engines_in_schema(forward_schema) == ()


def _exact_device_candidate_row(device: str) -> dict:
    layout = _layout(two_role_llm=True)
    layout["device_layout_id"] = f"unit-layout:{device}"
    layout["available_devices"] = [device]
    layout["resource_groups"][0]["devices"] = [device]
    for stage in ("generator_prefill", "generator_decode"):
        layout["performance_stages"][stage]["devices"] = [device]
    layout["engines"]["generator"]["devices"] = [device]
    return {
        "latency_s": 0.2,
        "latency_s_ttft": 0.1,
        "latency_s_tpot": 0.01,
        "qps": 40.0,
        "qps_per_chip": 40.0,
        "qps_bound": ["generator_prefill", "generator_decode"],
        "qps_generator_prefill": 80.0,
        "qps_generator_decode": 80.0,
        "batch_size_request": 8,
        "batch_size_generator_prefill": 8,
        "batch_size_generator_decode": 8,
        "num_chips_e2e": 1,
        CANDIDATE_LAYOUT_COLUMN: layout,
    }


class _ExactDeviceOutcomePricer:
    def __init__(self, failure: Exception | None = None):
        self.failure = failure

    def estimate_stage_for_devices(
        self,
        _pipeline_config,
        _hardware,
        *,
        stage,
        batch_size,
        devices,
        parallelism,
    ):
        if devices == ["cuda:0"] and self.failure is not None:
            raise self.failure
        tp, pp = parallelism
        return {
            "stage": stage,
            "devices": list(devices),
            "tp": tp,
            "pp": pp,
            "replicas": 1,
            "latency_s": 0.1,
            "throughput_qps": float(batch_size) / 0.1,
            "effective_icn_gbps": None,
            "effective_icn_ll_us": None,
        }


class _ForwardRerankerMustNotUseTpPricer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def estimate_stage_for_devices(self, *args, **kwargs):
        del args
        self.calls.append(dict(kwargs))
        raise AssertionError(
            "encoder-only reranker DP must retain RAGO scaling, not use TP/PP"
        )


class _ModelAwareConstantReplicaPricer:
    @staticmethod
    def parallelism_model(_pipeline_config):
        return "Qwen/Qwen2.5-14B-Instruct"

    def estimate_stage_for_devices(
        self,
        _pipeline_config,
        _hardware,
        *,
        stage,
        batch_size,
        devices,
        parallelism,
    ):
        tp, pp = parallelism
        assert len(devices) == tp * pp
        return {
            "stage": stage,
            "devices": list(devices),
            "tp": tp,
            "pp": pp,
            "replicas": 1,
            "latency_s": 0.1,
            "throughput_qps": float(batch_size) / 0.1,
            "effective_icn_gbps": 100.0,
            "effective_icn_ll_us": 1.0,
        }


def _four_gpu_llm_candidate(*, disaggregated: bool) -> dict:
    devices = [f"cuda:{index}" for index in range(4)]
    prefill_devices = devices[:2] if disaggregated else devices
    decode_devices = devices[2:] if disaggregated else devices
    stages = {
        "generator_prefill": {
            "kind": "gpu",
            "engine": "generator",
            "role": "prefill",
            "resource": "gpu:0",
            "devices": prefill_devices,
            "tp": len(prefill_devices),
            "pp": 1,
            "replicas": 1,
            "replica_devices": [prefill_devices],
        },
        "generator_decode": {
            "kind": "gpu",
            "engine": "generator",
            "role": "decode",
            "resource": "gpu:1" if disaggregated else "gpu:0",
            "devices": decode_devices,
            "tp": len(decode_devices),
            "pp": 1,
            "replicas": 1,
            "replica_devices": [decode_devices],
        },
    }
    groups = [{
        "id": "gpu:0",
        "kind": "gpu",
        "devices": prefill_devices,
        "performance_stages": ["generator_prefill"],
    }]
    if disaggregated:
        groups.append({
            "id": "gpu:1",
            "kind": "gpu",
            "devices": decode_devices,
            "performance_stages": ["generator_decode"],
        })
    else:
        groups[0]["performance_stages"].append("generator_decode")
    engine = {
        "pd_serving": "disagg_pd" if disaggregated else "collocated_pd",
        "devices": devices,
        "performance_stages": ["generator_prefill", "generator_decode"],
    }
    if disaggregated:
        engine["generator_prefill"] = dict(stages["generator_prefill"])
        engine["generator_decode"] = dict(stages["generator_decode"])
    else:
        engine.update({
            "tp": 4,
            "pp": 1,
            "replicas": 1,
            "replica_devices": [devices],
        })
    layout = {
        "source": "rago_physical_mapping",
        "device_layout_id": (
            "unit-disaggregated" if disaggregated else "unit-collocated"
        ),
        "available_devices": devices,
        "resource_groups": groups,
        "performance_stages": stages,
        "engines": {"generator": engine},
    }
    return {
        "latency_s": 0.2,
        "latency_s_ttft": 0.1,
        "latency_s_tpot": 0.01,
        "qps": 40.0,
        "qps_per_chip": 10.0,
        "qps_bound": ["generator_prefill", "generator_decode"],
        "qps_generator_prefill": 80.0,
        "qps_generator_decode": 80.0,
        "batch_size_request": 8,
        "batch_size_generator_prefill": 8,
        "batch_size_generator_decode": 8,
        "num_chips_e2e": 4,
        "num_chips_generator_prefill": len(prefill_devices),
        "num_chips_generator_decode": len(decode_devices),
        CANDIDATE_LAYOUT_COLUMN: layout,
    }


def test_sweep_expands_every_executable_collocated_exact_width_tp_pp() -> None:
    pricer = _ModelAwareConstantReplicaPricer()
    priced = candidate_reprice.reprice_candidate_genz_services(
        pd.DataFrame([_four_gpu_llm_candidate(disaggregated=False)]),
        pipeline_config={},
        hardware=None,
        stage_cost_models={
            "generator_prefill": pricer,
            "generator_decode": pricer,
        },
    )

    variants = set(zip(
        priced["candidate_tp_generator_prefill"],
        priced["candidate_pp_generator_prefill"],
        priced["candidate_replicas_generator_prefill"],
    ))
    assert variants == {
        (1, 4, 1), (2, 2, 1), (4, 1, 1),
    }
    assert priced["candidate_tp_generator_prefill"].tolist() == priced[
        "candidate_tp_generator_decode"
    ].tolist()
    assert priced["candidate_pp_generator_prefill"].tolist() == priced[
        "candidate_pp_generator_decode"
    ].tolist()
    assert set(priced["candidate_replicas_generator_prefill"]) == {1}
    assert set(priced["qps_generator_prefill"]) == {80.0}
    validate_candidate_frame_runtime_parallelism(
        priced, {}, source="unit CM sweep",
    )


def test_sweep_rejects_every_disagg_nixl_pp_variant() -> None:
    pricer = _ModelAwareConstantReplicaPricer()
    priced = candidate_reprice.reprice_candidate_genz_services(
        pd.DataFrame([_four_gpu_llm_candidate(disaggregated=True)]),
        pipeline_config={},
        hardware=None,
        stage_cost_models={
            "generator_prefill": pricer,
            "generator_decode": pricer,
        },
    )

    assert len(priced) == 1
    assert set(priced["candidate_pp_generator_prefill"]) == {1}
    assert set(priced["candidate_pp_generator_decode"]) == {1}
    assert set(priced["candidate_replicas_generator_prefill"]) == {1}
    assert set(priced["candidate_replicas_generator_decode"]) == {1}
    validate_candidate_frame_runtime_parallelism(
        priced, {}, source="unit CM sweep",
    )


def test_measured_runtime_rejects_hypothetical_llm_dp_layout() -> None:
    row = _four_gpu_llm_candidate(disaggregated=False)
    layout = row[CANDIDATE_LAYOUT_COLUMN]
    replicas = [[f"cuda:{index}"] for index in range(4)]
    for stage in ("generator_prefill", "generator_decode"):
        layout["performance_stages"][stage].update({
            "tp": 1,
            "pp": 1,
            "replicas": 4,
            "replica_devices": replicas,
        })
    layout["engines"]["generator"].update({
        "tp": 1,
        "pp": 1,
        "replicas": 4,
        "replica_devices": replicas,
    })

    with pytest.raises(NotImplementedError, match="requires one measured TP×PP"):
        validate_measured_runtime_parallelism(
            {}, layout=layout, source="unit hypothetical DP candidate",
        )


def test_multicard_forward_reranker_keeps_rago_dp_capacity() -> None:
    devices = [f"cuda:{index}" for index in range(4)]
    replica_devices = [[device] for device in devices]
    layout = {
        "source": "rago_physical_mapping",
        "device_layout_id": "forward-reranker-dp4",
        "available_devices": devices,
        "resource_groups": [{
            "id": "gpu:0",
            "kind": "gpu",
            "devices": devices,
            "performance_stages": ["passage_reranker"],
        }],
        "performance_stages": {
            "passage_reranker": {
                "kind": "gpu",
                "resource": "gpu:0",
                "devices": devices,
                "engine": "passage_reranker",
                # The bare PerformanceStage is a forward encoder even though
                # its heterogeneous user-stage parent also owns RankGPT P/D.
                "role": "prefill",
                "tp": 1,
                "pp": 1,
                "replicas": 4,
                "replica_devices": replica_devices,
            },
        },
        "engines": {
            "passage_reranker": {
                "pd_serving": "collocated_pd",
                "devices": devices,
                "num_chips": 4,
                "performance_stages": ["passage_reranker"],
                "tp": 1,
                "pp": 1,
                "replicas": 4,
                "replica_devices": replica_devices,
            },
        },
    }
    # RAGO scaled one measured single-card replica from 80 to 4 * 80 QPS.
    rows = pd.DataFrame([{
        "latency_s": 0.1,
        "qps": 320.0,
        "qps_per_chip": 80.0,
        "qps_bound": ["passage_reranker"],
        "qps_passage_reranker": 320.0,
        "batch_size_request": 8,
        "batch_size_passage_reranker": 8,
        "num_chips_e2e": 4,
        CANDIDATE_LAYOUT_COLUMN: layout,
    }])
    pricer = _ForwardRerankerMustNotUseTpPricer()

    priced = candidate_reprice.reprice_candidate_genz_services(
        rows,
        pipeline_config={
            "node_lines": [{
                "nodes": [{
                    "stage": "passage_reranker",
                    "modules": [{
                        "component": "sentence_transformer_reranker",
                    }],
                }],
            }],
        },
        hardware=None,
        stage_cost_models={"passage_reranker": pricer},
    )

    assert pricer.calls == []
    assert priced.loc[0, "qps_passage_reranker"] == pytest.approx(320.0)
    assert priced.loc[0, "qps"] == pytest.approx(320.0)
    assert priced.loc[0, "qps_per_chip"] == pytest.approx(80.0)
    assert not bool(priced.loc[0, "candidate_genz_repriced"])
    retained = candidate_system_layout_from_row(priced.loc[0])[
        "performance_stages"
    ]["passage_reranker"]
    assert (retained["tp"], retained["pp"], retained["replicas"]) == (1, 1, 4)
    assert retained["replica_devices"] == replica_devices


def test_exact_device_capacity_drops_only_the_infeasible_candidate() -> None:
    rows = pd.DataFrame(
        [
            _exact_device_candidate_row("cuda:0"),
            _exact_device_candidate_row("cuda:1"),
        ],
        index=["too_large", "fits"],
    )
    pricer = _ExactDeviceOutcomePricer(
        CMDeviceCapacityError("largest PP rank exceeds device memory")
    )

    priced = candidate_reprice.reprice_candidate_genz_services(
        rows,
        pipeline_config={},
        hardware=None,
        stage_cost_models={
            "generator_prefill": pricer,
            "generator_decode": pricer,
        },
    )

    assert priced.index.tolist() == ["fits"]
    assert bool(priced.loc["fits", "candidate_genz_repriced"])


def test_exact_device_candidate_boundary_does_not_hide_model_bugs() -> None:
    rows = pd.DataFrame(
        [
            _exact_device_candidate_row("cuda:0"),
            _exact_device_candidate_row("cuda:1"),
        ]
    )
    pricer = _ExactDeviceOutcomePricer(ValueError("tuple index out of range"))

    with pytest.raises(ValueError, match="tuple index out of range"):
        candidate_reprice.reprice_candidate_genz_services(
            rows,
            pipeline_config={},
            hardware=None,
            stage_cost_models={
                "generator_prefill": pricer,
                "generator_decode": pricer,
            },
        )


@pytest.mark.parametrize(
    "prefill,decode",
    [((1, 2), (1, 1)), ((1, 1), (1, 2))],
)
def test_candidate_pair_rejects_nixl_pipeline_parallelism(
    prefill, decode
) -> None:
    class ModelAwarePricer:
        @staticmethod
        def parallelism_model(_pipeline_config):
            return "Qwen/Qwen2.5-14B-Instruct"

    feasible, code, reason = candidate_reprice._check_disaggregated_pair(
        ModelAwarePricer(),
        {},
        prefill=prefill,
        decode=decode,
    )

    assert not feasible
    assert code == "pipeline_parallel_unsupported"
    assert "does not support pipeline parallelism" in reason
