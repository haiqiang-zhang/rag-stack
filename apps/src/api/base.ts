/**
 * Shared API helpers for the RAG-Stack backend.
 *
 * All requests include auth headers when a token is configured.
 */

import { getApiUrl, getAuthToken } from "@/lib/config";

function authHeaders(): Record<string, string> {
  const token = getAuthToken();
  const h: Record<string, string> = {};
  if (token) h["Authorization"] = `Bearer ${token}`;
  return h;
}

async function responseError(res: Response): Promise<Error> {
  const payload = await res.json().catch(() => null);
  const detail = payload?.detail;
  let message = res.statusText || "Request failed";
  if (typeof detail === "string" && detail) message = detail;
  else if (typeof detail?.message === "string") message = detail.message;
  else if (Array.isArray(detail)) {
    const errors = detail.filter((item) => typeof item?.msg === "string").map((item) => {
      const location = Array.isArray(item.loc) ? item.loc.join(".") : "";
      return location ? `${location}: ${item.msg}` : item.msg;
    });
    if (errors.length) message = errors.join("; ");
  }
  return new Error(`API error ${res.status}: ${message}`);
}

export async function apiFetch<T>(path: string): Promise<T> {
  const url = getApiUrl();
  const res = await fetch(`${url}${path}`, {
    cache: "no-store",
    headers: authHeaders(),
  });
  if (res.status === 401) {
    throw new AuthError("Unauthorized — check your token");
  }
  if (!res.ok) {
    throw await responseError(res);
  }
  return res.json() as Promise<T>;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const url = getApiUrl();
  const res = await fetch(`${url}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (res.status === 401) throw new AuthError("Unauthorized");
  if (!res.ok) throw await responseError(res);
  return res.json() as Promise<T>;
}

export async function apiPut<T>(path: string, body: unknown): Promise<T> {
  const url = getApiUrl();
  const res = await fetch(`${url}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (res.status === 401) throw new AuthError("Unauthorized");
  if (!res.ok) {
    throw await responseError(res);
  }
  return res.json() as Promise<T>;
}

export async function apiDelete<T>(path: string): Promise<T> {
  const url = getApiUrl();
  const res = await fetch(`${url}${path}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (res.status === 401) throw new AuthError("Unauthorized");
  if (!res.ok) throw await responseError(res);
  return res.json() as Promise<T>;
}

export class AuthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AuthError";
  }
}

// ── Global settings ────────────────────────────────────────────────────────

export interface AppSettings {
  benchmark_results_dir: string;
  calibration_profiles_dir: string;
  optimizer_benchmark_dir: string;
  optimizer_configs_dir: string;
  rag_optimization_project_dir: string;
  rag_optimization_configs_dir: string;
  datasets_dir: string;
  dataset_generator_configs_dir: string;
  env_file_path: string;
}

export async function fetchEnvKeys(): Promise<Record<string, string>> {
  const data = await apiFetch<{ keys: Record<string, string> }>("/api/settings/env-keys");
  return data.keys;
}

export async function updateEnvKeys(keys: Record<string, string>): Promise<Record<string, string>> {
  const data = await apiPut<{ keys: Record<string, string> }>("/api/settings/env-keys", { keys });
  return data.keys;
}

export async function fetchSettings(): Promise<AppSettings> {
  return apiFetch<AppSettings>("/api/settings/");
}

export async function updateSettings(settings: AppSettings): Promise<AppSettings> {
  return apiPut<AppSettings>("/api/settings/", settings);
}
