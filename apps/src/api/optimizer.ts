/**
 * Optimizer benchmark API endpoints.
 */

import { apiDelete, apiFetch, apiPost, apiPut } from "./base";
import type {
  OptimizerInfo,
  OptimizerBenchmarkRow,
  OptimizerBenchmarkSummary,
  EvalDistributionPayload,
  BenchmarkConfig,
  BenchmarkJob,
} from "@/types/optimizer";

export async function fetchOptimizerRegistry(): Promise<OptimizerInfo[]> {
  const data = await apiFetch<{ registry: OptimizerInfo[] }>("/api/optimizer/registry");
  return data.registry;
}

export async function fetchOptimizerBenchmarkResults(
  fn?: string
): Promise<{ functions: string[]; optimizers: string[]; rows: OptimizerBenchmarkRow[] }> {
  const qs = fn ? `?function=${encodeURIComponent(fn)}` : "";
  return apiFetch(`/api/optimizer/benchmark-results${qs}`);
}

export async function fetchOptimizerBenchmarkSummary(): Promise<OptimizerBenchmarkSummary[]> {
  const data = await apiFetch<{ summary: OptimizerBenchmarkSummary[] }>("/api/optimizer/benchmark-summary");
  return data.summary;
}

export async function deleteOptimizerBenchmarkResult(
  optimizer: string,
  syntheticFunction: string,
): Promise<{ deleted: string[] }> {
  const qs =
    `?optimizer=${encodeURIComponent(optimizer)}` +
    `&function=${encodeURIComponent(syntheticFunction)}`;
  return apiDelete(`/api/optimizer/benchmark-results${qs}`);
}

export async function fetchEvalDistribution(
  syntheticFunction: string,
): Promise<EvalDistributionPayload> {
  return apiFetch(
    `/api/optimizer/eval-distribution?function=${encodeURIComponent(syntheticFunction)}`,
  );
}

export async function fetchOptimizerConfigs(): Promise<BenchmarkConfig[]> {
  const data = await apiFetch<{ configs: BenchmarkConfig[] }>("/api/optimizer/configs");
  return data.configs;
}

export async function saveOptimizerConfig(filename: string, content: string): Promise<void> {
  await apiPut("/api/optimizer/configs", { filename, content });
}

export async function launchBenchmarkRun(configFilename: string): Promise<BenchmarkJob> {
  return apiPost("/api/optimizer/benchmark-run", { config_filename: configFilename });
}

export async function fetchBenchmarkJobs(): Promise<BenchmarkJob[]> {
  const data = await apiFetch<{ jobs: BenchmarkJob[] }>("/api/optimizer/jobs");
  return data.jobs;
}

export async function killJob(jobId: string): Promise<Record<string, unknown>> {
  return apiPost(`/api/jobs/${jobId}/kill`, {});
}
