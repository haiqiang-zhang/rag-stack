/**
 * Quality Store API endpoints.
 */

import { apiFetch, apiPost, apiDelete } from "./base";

// ── Types ────────────────────────────────────────────────────────────────

export interface QualityStoreInfo {
  name: string;
  n_configs: number;
  n_failed: number;
  strategy: string;
  n_grid: number | null;
  created_at: string;
  updated_at: string;
  status: string;
  store_path: string;
  dataset?: Record<string, unknown>;
}

export interface QualityStoreDetail extends QualityStoreInfo {
  token_stats?: Record<string, unknown>;
  config_hashes?: string[];
  failed_hashes?: string[];
}

export interface QualityStoreConfig {
  config_hash: string;
  config: Record<string, unknown>;
  quality_metrics: Record<string, number | null>;
  quality_combined: number;
}

export interface QualityStoreStatus {
  name: string;
  status: string;
  pid: number | null;
  pid_alive: boolean;
  n_configs: number;
  n_failed: number;
}

export interface CreateQualityStoreReq {
  name: string;
  config_filename: string;
  corpus_data: string;
  qa_data: string;
  n_grid?: number;
  n_configs?: number;
  strategy?: string;
}

// ── Endpoints ────────────────────────────────────────────────────────────

export async function fetchQualityStores(): Promise<QualityStoreInfo[]> {
  const data = await apiFetch<{ stores: QualityStoreInfo[] }>("/api/quality-store/list");
  return data.stores;
}

export async function fetchQualityStoreDetail(name: string): Promise<QualityStoreDetail> {
  return apiFetch<QualityStoreDetail>(`/api/quality-store/${encodeURIComponent(name)}`);
}

export async function fetchQualityStoreConfigs(name: string): Promise<QualityStoreConfig[]> {
  const data = await apiFetch<{ configs: QualityStoreConfig[] }>(
    `/api/quality-store/${encodeURIComponent(name)}/configs`,
  );
  return data.configs;
}

export async function fetchPipelineConfig(
  storeName: string,
  configHash: string,
): Promise<Record<string, unknown>> {
  return apiFetch<Record<string, unknown>>(
    `/api/quality-store/${encodeURIComponent(storeName)}/pipeline-config/${encodeURIComponent(configHash)}`,
  );
}

export async function fetchQualityStoreStatus(name: string): Promise<QualityStoreStatus> {
  return apiFetch<QualityStoreStatus>(`/api/quality-store/${encodeURIComponent(name)}/status`);
}

export async function createQualityStore(
  req: CreateQualityStoreReq,
): Promise<{ ok: boolean; name: string; pid: number }> {
  return apiPost("/api/quality-store/create", {
    name: req.name,
    config_filename: req.config_filename,
    corpus_data: req.corpus_data,
    qa_data: req.qa_data,
    n_grid: req.n_grid ?? 5,
    n_configs: req.n_configs ?? 0,
    strategy: req.strategy ?? "exhaustive",
  });
}

export async function killQualityStore(name: string): Promise<void> {
  await apiPost(`/api/quality-store/${encodeURIComponent(name)}/kill`, {});
}

export async function deleteQualityStore(name: string): Promise<void> {
  await apiDelete(`/api/quality-store/${encodeURIComponent(name)}`);
}
