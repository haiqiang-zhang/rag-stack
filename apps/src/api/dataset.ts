/**
 * Dataset management API endpoints.
 */

import { apiFetch, apiPost, apiPut } from "./base";

// ── Types ────────────────────────────────────────────────────────────────

export interface GeneratorStrategy {
  name: string;
  module_path: string;
  class_name: string;
}

export interface DatasetParquetFile {
  filename: string;
  rows: number;
  size_bytes: number;
}

export interface DatasetInfo {
  name: string;
  parquet_files: DatasetParquetFile[];
  has_generation_summary: boolean;
  generation_status: string | null;
  generation_strategy: string | null;
}

export interface DatasetGenConfig {
  filename: string;
  strategy: string;
  content: Record<string, unknown>;
}

export interface DatasetGenJob {
  name: string;
  job_type: string;
  status: string;
  strategy: string;
  config_filename: string;
  pid: number | null;
  pid_alive: boolean;
  timestamp: string;
  elapsed_seconds: number;
  qa_rows: number;
  corpus_rows: number;
  error: string | null;
}

// ── Endpoints ────────────────────────────────────────────────────────────

export async function fetchDatasetGenRegistry(): Promise<GeneratorStrategy[]> {
  const data = await apiFetch<{ strategies: GeneratorStrategy[] }>("/api/dataset/registry");
  return data.strategies;
}

export async function fetchDatasetList(): Promise<DatasetInfo[]> {
  const data = await apiFetch<{ datasets: DatasetInfo[] }>("/api/dataset/list");
  return data.datasets;
}

export async function fetchDatasetPreview(
  name: string,
  file: string,
  limit: number = 20
): Promise<{ columns: string[]; rows: Record<string, unknown>[]; total_rows: number }> {
  return apiFetch(`/api/dataset/${encodeURIComponent(name)}/preview?file=${encodeURIComponent(file)}&limit=${limit}`);
}

export async function fetchDatasetGenConfigs(): Promise<DatasetGenConfig[]> {
  const data = await apiFetch<{ configs: DatasetGenConfig[] }>("/api/dataset/configs");
  return data.configs;
}

export async function saveDatasetGenConfig(filename: string, content: string): Promise<void> {
  await apiPut("/api/dataset/configs", { filename, content });
}

export async function launchDatasetGeneration(configFilename: string, outputName: string): Promise<Record<string, unknown>> {
  return apiPost("/api/dataset/generate", { config_filename: configFilename, output_name: outputName });
}

export async function fetchDatasetGenJobs(): Promise<DatasetGenJob[]> {
  const data = await apiFetch<{ jobs: DatasetGenJob[] }>("/api/dataset/jobs");
  return data.jobs;
}

export async function fetchDatasetGenJobStatus(name: string): Promise<DatasetGenJob> {
  return apiFetch<DatasetGenJob>(`/api/dataset/jobs/${encodeURIComponent(name)}/status`);
}

export async function killDatasetGenJob(name: string): Promise<void> {
  await apiPost(`/api/dataset/jobs/${encodeURIComponent(name)}/kill`, {});
}
