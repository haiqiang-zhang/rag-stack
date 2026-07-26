/**
 * Cost model estimation, registry, FAISS, LLM Sim, and GenZ API endpoints.
 */

import { apiFetch, apiPost, apiDelete } from "./base";
import type {
  CostModelInfo,
  CostModelCatalog,
  CalibrationProfile,
  BenchmarkRow,
  BenchmarkSummary,
  HardwarePreset,
  CalibrationProfileRecord,
  CalibrationJob,
  CalibrationRunRequest,
  CalibrationJobLog,
  UnifiedCalibrationProfileDetail,
  UnifiedCalibrationRunRequest,
  UnifiedCalibrationStatus,
} from "@/types/costModel";

// ── Registry ────────────────────────────────────────────────────────────────

export async function fetchRegistry(): Promise<CostModelInfo[]> {
  const data = await apiFetch<{ models: CostModelInfo[] }>(
    "/api/cost-model/"
  );
  return data.models;
}

export async function fetchCostModelCatalog(): Promise<CostModelCatalog> {
  const catalog = await apiFetch<CostModelCatalog>("/api/cost-model/catalog");
  if (catalog.schema_version !== "rag-stack.cost-model-catalog.v1" || catalog.scope !== "dispatch_catalog") {
    throw new Error("This cost-model catalog format is not supported. Refresh the app after updating the server.");
  }
  return catalog;
}

export async function fetchModelInfo(name: string): Promise<CostModelInfo> {
  return apiFetch<CostModelInfo>(`/api/cost-model/${name}/info`);
}

// ── FAISS IVF-PQ extensions ─────────────────────────────────────────────────

export async function fetchFaissPresets(): Promise<HardwarePreset[]> {
  const data = await apiFetch<{ presets: HardwarePreset[] }>(
    "/api/cost-model/faiss_ivf/presets"
  );
  return data.presets;
}

export async function fetchFaissProfiles(): Promise<CalibrationProfile[]> {
  const data = await apiFetch<{ profiles: CalibrationProfile[] }>(
    "/api/cost-model/faiss_ivf/profiles"
  );
  return data.profiles;
}

export async function fetchFaissProfile(
  filename: string
): Promise<CalibrationProfile> {
  return apiFetch<CalibrationProfile>(
    `/api/cost-model/faiss_ivf/profiles/${filename}`
  );
}

export async function fetchFaissBenchmarks(folder?: string): Promise<{
  rows: BenchmarkRow[];
  summary: BenchmarkSummary[];
}> {
  const qs = folder ? `?folder=${encodeURIComponent(folder)}` : "";
  return apiFetch<{ rows: BenchmarkRow[]; summary: BenchmarkSummary[] }>(
    `/api/cost-model/faiss_ivf/benchmarks${qs}`
  );
}

// ── Generic calibration endpoints (per cost model) ─────────────────────────

export class NotSupportedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NotSupportedError";
  }
}

async function apiFetchOrNotSupported<T>(path: string): Promise<T | null> {
  try {
    return await apiFetch<T>(path);
  } catch (err) {
    if (err instanceof Error && err.message.includes("501")) return null;
    throw err;
  }
}

export async function fetchCalibrationProfiles(
  costModel: string,
): Promise<CalibrationProfileRecord[] | null> {
  const data = await apiFetchOrNotSupported<{ profiles: CalibrationProfileRecord[] }>(
    `/api/cost-model/${encodeURIComponent(costModel)}/profiles`
  );
  return data ? data.profiles : null;
}

export async function deleteCalibrationProfile(
  costModel: string, filename: string,
): Promise<void> {
  await apiDelete(
    `/api/cost-model/${encodeURIComponent(costModel)}/profiles/${encodeURIComponent(filename)}`
  );
}

export async function fetchCostModelModules(costModel: string): Promise<string[]> {
  const data = await apiFetch<{ modules: string[] }>(
    `/api/cost-model/${encodeURIComponent(costModel)}/modules`
  );
  return data.modules;
}

export async function fetchCalibrationJobs(
  costModel: string,
): Promise<CalibrationJob[]> {
  const data = await apiFetch<{ jobs: CalibrationJob[] }>(
    `/api/cost-model/${encodeURIComponent(costModel)}/jobs`
  );
  return data.jobs;
}

export async function runCalibrationJob(
  costModel: string, req: CalibrationRunRequest,
): Promise<CalibrationJob> {
  return apiPost<CalibrationJob>(
    `/api/cost-model/${encodeURIComponent(costModel)}/run`, req,
  );
}

export async function fetchUnifiedCalibrationStatus(): Promise<UnifiedCalibrationStatus> {
  return apiFetch<UnifiedCalibrationStatus>("/api/cost-model/calibration/status");
}

export async function fetchUnifiedCalibrationProfileDetail(
  relativePath: string,
): Promise<UnifiedCalibrationProfileDetail> {
  return apiFetch<UnifiedCalibrationProfileDetail>(
    `/api/cost-model/calibration/profile?path=${encodeURIComponent(relativePath)}`
  );
}

export async function fetchUnifiedCalibrationJobs(): Promise<CalibrationJob[]> {
  const data = await apiFetch<{ jobs: CalibrationJob[] }>(
    "/api/cost-model/calibration/jobs"
  );
  return data.jobs;
}

export async function runUnifiedCalibrationJob(
  req: UnifiedCalibrationRunRequest,
): Promise<CalibrationJob> {
  return apiPost<CalibrationJob>("/api/cost-model/calibration/run", req);
}

export async function fetchUnifiedCalibrationJobLog(
  jobId: string,
  tail = 20000,
): Promise<CalibrationJobLog> {
  return apiFetch<CalibrationJobLog>(
    `/api/cost-model/calibration/jobs/${encodeURIComponent(jobId)}/log?tail=${tail}`
  );
}

export async function killJob(jobId: string): Promise<void> {
  await apiPost(`/api/jobs/${encodeURIComponent(jobId)}/kill`, {});
}

// ── GenZ presets ──────────────────────────────────────────────────────────

export async function fetchGpuPresets(): Promise<string[]> {
  const data = await apiFetch<{ presets: string[] }>("/api/genz/gpu-presets");
  return data.presets;
}

export async function fetchLlmModels(): Promise<string[]> {
  const data = await apiFetch<{ models: string[] }>("/api/genz/llm-models");
  return data.models;
}

// ── Cost model estimate ───────────────────────────────────────────────────

export async function postEstimate(
  modelName: string,
  pipelineConfig: Record<string, unknown>,
  hardware: Record<string, unknown>,
): Promise<{ latency_s: number; throughput_qps: number; operators: { name: string; compute_time_s: number; memory_time_s: number; time_s: number; bottleneck: string }[] }> {
  return apiPost(`/api/cost-model/${encodeURIComponent(modelName)}/estimate`, { pipeline_config: pipelineConfig, hardware });
}

// ── Benchmark folders ──────────────────────────────────────────────────────

export interface BenchmarkFolder {
  name: string;
  n_rows: number;
}

export async function fetchBenchmarkFolders(): Promise<{ root: string; folders: BenchmarkFolder[] }> {
  return apiFetch<{ root: string; folders: BenchmarkFolder[] }>(
    "/api/cost-model/faiss_ivf/benchmark-folders"
  );
}
