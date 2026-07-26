"""Pydantic request/response models for the Cost Model API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


# ── Registry ─────────────────────────────────────────────────────────────────

class EndpointInfo(BaseModel):
    path: str
    method: str
    summary: str


class CostModelInfo(BaseModel):
    name: str
    class_name: str
    description: str
    family: str                     # "llm_sim" | "retrieval" | "other"
    mapped_module_types: list[str]
    capabilities: list[str]
    extensions: list[EndpointInfo]


class RegistryResponse(BaseModel):
    models: list[CostModelInfo]


# ── Estimate ─────────────────────────────────────────────────────────────────

class EstimateRequest(BaseModel):
    pipeline_config: dict
    hardware: dict


class OperatorBreakdownResponse(BaseModel):
    name: str
    compute_time_s: float
    memory_time_s: float
    time_s: float
    bottleneck: str


class EstimateResponse(BaseModel):
    scope: Literal["component_estimate"] = "component_estimate"
    latency_s: float
    throughput_qps: float
    operators: list[OperatorBreakdownResponse]


class EstimateBatchRequest(BaseModel):
    configs: list[dict]
    hardware: dict


# ── Sweep ────────────────────────────────────────────────────────────────────

class SweepRequest(BaseModel):
    pipeline_config: dict
    hardware: dict
    stage: str
    max_batch_size: int


# ── FAISS-specific ───────────────────────────────────────────────────────────

class HardwarePreset(BaseModel):
    name: str
    description: str


class CalibrationProfile(BaseModel):
    filename: str
    data: dict


class BenchmarkRow(BaseModel):
    index_name: str
    batch_size: int
    parallel_mode: int
    nprobe: int
    actual_latency_ms: float
    predicted_latency_ms: float
    latency_error_pct: float
    actual_qps: float
    predicted_qps: float
    qps_error_pct: float


# ── Settings ────────────────────────────────────────────────────────────────

class AppSettings(BaseModel):
    benchmark_results_dir: str = ""
    calibration_profiles_dir: str = ""
    optimizer_benchmark_dir: str = ""
    optimizer_configs_dir: str = ""
    rag_optimization_project_dir: str = ""
    rag_optimization_configs_dir: str = ""
    datasets_dir: str = ""
    dataset_generator_configs_dir: str = ""
    env_file_path: str = ""
