/** Project-scoped CM preview jobs. Saved path descriptions are for viewing only. */
import { getApiUrl, getAuthToken } from "@/lib/config";

export type PreviewStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "blocked"
  | "failed"
  | "cancelled"
  | "interrupted";
export type JsonRecord = Record<string, unknown>;
export type CompiledPathStatus = "available" | "unavailable" | null;

export interface CMPreviewRequest {
  eval_id?: number;
  parameters?: JsonRecord;
  algo_config?: JsonRecord;
  system_config?: JsonRecord;
  sample_count?: number;
  question_ids?: Array<string | number>;
  reuse_trace: boolean;
  skip_judge: boolean;
  qa_data?: string;
  corpus_data?: string;
  dataset_name?: string;
}

export interface CMPreviewBlocker {
  phase?: string;
  code?: string;
  message: string;
  details?: unknown;
}

export interface CMOperation {
  op_id: string;
  kind: "worker" | "prefill" | "decode" | "transfer";
  name: string;
  source_call_ids: string[];
  /** Zero-based positions in the original query's calls array, not step_idx. */
  source_call_indices: number[];
  depends_on: string[];
  resource_ids: string[];
  details: JsonRecord;
}

export interface CMQuery {
  question_id: string | number;
  operations: CMOperation[];
}
export interface CMResource extends JsonRecord {
  id: string;
  name: string;
  devices: unknown[];
}
export interface CMCompiledPlan extends JsonRecord {
  schema_version: "rag-cm.compiled-plan.v1";
  status: "compiled";
  workload: {
    summary: unknown;
    normalization: unknown;
    source_question_ids: Array<string | number>;
  };
  queries: CMQuery[];
  resources: {
    workers: CMResource[];
    engines: CMResource[];
    n_cards: number;
    cpu_cores: number;
  };
  deployment: JsonRecord;
  calibration: JsonRecord | null;
  pricing_binding: JsonRecord;
  hardware: JsonRecord;
  prediction: null;
}

export interface CMTraceCall extends JsonRecord {
  call_id?: string;
  stage?: string;
  node_id?: string;
  step_idx?: number;
}
export interface CMRagIR extends JsonRecord {
  queries?: Array<{
    question_id: string | number;
    source_question_id?: string | number;
    calls: CMTraceCall[];
  }>;
}

export interface CMParameter {
  name: string;
  type: "choice" | "range" | "fixed";
  value_type?: string;
  values?: unknown[];
  value?: unknown;
  bounds?: [number, number];
  dependents?: Record<string, string[]>;
}

export interface CMPreviewOptions {
  parameters: CMParameter[];
  dataset: {
    qa_available: boolean;
    corpus_available: boolean;
    dataset_name?: string | null;
  };
  selection_error: { code: string; message: string } | null;
  system_config_required: boolean;
}

export interface CMPreviewJob {
  schema_version: "rag-stack.cm-preview.v1";
  preview_id: string;
  project: string;
  status: PreviewStatus;
  phase: string;
  created_at?: string;
  updated_at?: string;
  source: { kind: string; eval_id?: number | null; reuse_trace: boolean };
  cohort: {
    requested_count: number;
    question_ids: Array<string | number>;
    count: number;
  };
  skip_judge: boolean;
  run_id?: string;
  log_tail?: string;
  error?: string | CMPreviewBlocker | null;
  compiled_plan?: CMCompiledPlan | null;
  compiled_path_status?: CompiledPathStatus;
  rag_ir?: CMRagIR | null;
  artifacts?: Array<{ name: string; size_bytes: number }>;
}

/** Downloads include display projections, never executable compiled objects. */
export function isCMPreviewArtifactDownloadable(name: string): boolean {
  return ["state.json", "request.json", "source_config.yaml", "algo_config.json",
    "system_config.json", "runtime_config.json", "rag_ir.json", "preview.log",
    "RUN.md", "source_trace.json", "compiled_description.json"].includes(name);
}

function basePath(project: string): string {
  return `/api/rag-project/projects/${encodeURIComponent(project)}/cm-previews`;
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const token = getAuthToken();
  const response = await fetch(`${getApiUrl()}${path}`, {
    method: body === undefined ? "GET" : "POST",
    cache: "no-store",
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      const value = data.detail ?? data.error ?? data.message;
      if (value !== undefined)
        detail = typeof value === "string" ? value : JSON.stringify(value);
    } catch {
      /* The HTTP status remains useful if a proxy returned plain text. */
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export function createCMPreview(
  project: string,
  body: CMPreviewRequest,
): Promise<CMPreviewJob> {
  return request(basePath(project), body);
}
export function listCMPreviews(
  project: string,
): Promise<{ previews: CMPreviewJob[] }> {
  return request(basePath(project));
}
export function fetchCMPreviewOptions(
  project: string,
): Promise<CMPreviewOptions> {
  return request(`${basePath(project)}/options`);
}
export function fetchCMPreview(
  project: string,
  id: string,
): Promise<CMPreviewJob> {
  return request(`${basePath(project)}/${encodeURIComponent(id)}`);
}
export function cancelCMPreview(
  project: string,
  id: string,
): Promise<CMPreviewJob> {
  return request(`${basePath(project)}/${encodeURIComponent(id)}/cancel`, {});
}
/** Fetch blobs with the configured bearer token, never a token in the URL. */
export async function downloadCMPreviewArtifact(
  project: string,
  id: string,
  name: string,
): Promise<void> {
  if (!isCMPreviewArtifactDownloadable(name)) {
    throw new Error("This preview artifact is not downloadable. Only the compiled path display description can be exported.");
  }
  const token = getAuthToken();
  const response = await fetch(
    `${getApiUrl()}${basePath(project)}/${encodeURIComponent(id)}/artifacts/${encodeURIComponent(name)}`,
    {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    },
  );
  if (!response.ok)
    throw new Error(
      `Download failed (${response.status}): ${response.statusText}`,
    );
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
