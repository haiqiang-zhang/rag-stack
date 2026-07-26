/**
 * RAG Project API endpoints.
 */

import { getApiUrl, getAuthToken } from "@/lib/config";
import { apiFetch, apiPost, apiPut, apiDelete } from "./base";

// ── Types ────────────────────────────────────────────────────────────────

export interface RagProjectConfig {
  filename: string;
  optimizer_type: string;
  n_iterations: number;
  eval_backend?: string;
  content: Record<string, unknown>;
  raw_content: string;
}

export interface DatasetFolder {
  name: string;
  parquet_files: string[];
}

export interface SystemYamlInfo {
  path: string;
  filename: string;
  n_refine_iterations: number;
  generator_backend: "inherit" | "vllm" | "vllm_api";
  summary: {
    cpu?: string;
    gpu?: string;
    performance_source?: string;
    available_gpus?: number;
  };
}

export interface ProjectStatus {
  status: string;
  pid: number | null;
  pid_alive: boolean;
  timestamp: string;
  n_evaluations: number;
  n_pareto_points: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────

function authHeaders(): Record<string, string> {
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// ── Endpoints ────────────────────────────────────────────────────────────

export interface TraceItemProfile {
  items: number;
  unique_items?: number;
  list_sizes?: number[];
  list_unique_items?: number[];
}

export interface ObservedTraceCall {
  stage: string;
  step_idx: number;
  model_id?: string | null;
  call_id?: string;
  node_id?: string;
  input_refs?: Record<string, string[]>;
  input_profile?: TraceItemProfile;
  output_profile?: TraceItemProfile;
  input_tokens: number;
  output_tokens: number;
}

export interface EvaluationTrace {
  available: boolean;
  source: string;
  reason?: string;
  trace: {
    queries: Array<{ question_id: string | number; calls: ObservedTraceCall[] }>;
    provenance?: Record<string, unknown>;
  } | null;
  workflow?: Record<string, unknown> | null;
}

export function fetchEvaluationTrace(project: string, evalId: number): Promise<EvaluationTrace> {
  return apiFetch(`/api/rag-project/evaluations/${evalId}/trace?project=${encodeURIComponent(project)}`);
}

export async function fetchRagProjectConfigs(): Promise<RagProjectConfig[]> {
  const data = await apiFetch<{ configs: RagProjectConfig[] }>("/api/rag-project/configs");
  return data.configs;
}

export async function createRagProject(projectName: string, configFilename: string): Promise<{ ok: boolean; project_name: string }> {
  return apiPost("/api/rag-project/create", { project_name: projectName, config_filename: configFilename });
}

export interface ProjectConfig {
  project_name: string;
  content: Record<string, unknown>;
  raw_content: string;
  editable: boolean;
  read_only_reason: string | null;
}

export function fetchProjectConfig(projectName: string): Promise<ProjectConfig> {
  return apiFetch(`/api/rag-project/config/${encodeURIComponent(projectName)}`);
}

export function saveProjectConfig(projectName: string, content: string): Promise<ProjectConfig> {
  return apiPut("/api/rag-project/config", { project_name: projectName, content });
}

// ── 7-section schema: validation + scaffolding (backend is source of truth) ──

export interface ConfigValidation {
  valid: boolean;
  errors: string[];
}

/** Validate a config (raw YAML) via the canonical ConfigValidator on the server. */
export async function validateConfig(content: string): Promise<ConfigValidation> {
  return apiPost("/api/rag-project/validate-config", { content });
}

// Search-space tree — what an Ax search space a config expands into.
export interface SearchSpaceNode {
  name: string;
  leaf: string;
  kind: "choice" | "range";
  param_type?: string;
  values?: string[];
  lower?: number;
  upper?: number;
  dependents?: Record<string, SearchSpaceNode[]>;
}
export interface SearchSpaceTree {
  tree?: SearchSpaceNode[];
  summary?: { n_params: number; n_range: number; is_hierarchical: boolean; discrete_size: number };
  error?: string;
}

/** Build the Ax search space for a config and return it as a visual tree. */
export async function fetchSearchSpaceTree(content: string): Promise<SearchSpaceTree> {
  return apiPost("/api/rag-project/search-space-tree", { content });
}

// Project HV comparison — normalized 2-objective hypervolume convergence.
export interface HvPoint { i: number; hv: number; }
export interface ParetoPoint { perf: number; qual: number; eval_id?: number | null; }
export interface EvalPoint { eval_id: number; perf: number | null; qual: number | null; valid: boolean; }
export interface ProjectHvSeries {
  label: string;
  missing?: boolean;
  n_init?: number;
  final_hv?: number;
  best_qual?: number | null;
  best_perf?: number | null;
  n_valid?: number;
  n_evals?: number;
  hv?: HvPoint[];
  pareto?: ParetoPoint[];
  points?: EvalPoint[];
}
export interface ProjectHvComparison {
  metric?: string;
  available_metrics: string[];
  bounds?: { perf: [number, number]; qual: [number, number] };
  projects?: ProjectHvSeries[];
  error?: string;
}

/** Compare optimization projects by normalized hypervolume convergence. */
export async function fetchProjectHvComparison(
  projects: string[],
  qualityMetric?: string,
): Promise<ProjectHvComparison> {
  return apiPost("/api/rag-project/compare-hv", { projects, quality_metric: qualityMetric });
}

export interface ScaffoldResult {
  content: string;                      // YAML text
  config: Record<string, unknown>;      // parsed object
}

/** Get a known-valid 7-section config template for a backend/optimizer combo. */
export async function scaffoldConfig(opts: {
  evalBackend?: "static_gt" | "flashrag";
  optimizerType?: string;
  performanceSource?: "cost_model" | "measured";
}): Promise<ScaffoldResult> {
  return apiPost("/api/rag-project/scaffold", {
    eval_backend: opts.evalBackend ?? "static_gt",
    optimizer_type: opts.optimizerType ?? "agent_qnehvi",
    performance_source: opts.performanceSource ?? "cost_model",
  });
}

export async function saveRagTemplateConfig(filename: string, content: string): Promise<void> {
  await apiPut(`/api/rag-project/configs/${encodeURIComponent(filename)}`, { content });
}

export async function deleteRagConfig(filename: string): Promise<void> {
  await apiDelete(`/api/rag-project/configs/${encodeURIComponent(filename)}`);
}

export async function duplicateRagConfig(filename: string, newFilename: string): Promise<{ filename: string }> {
  return apiPost(`/api/rag-project/configs/${encodeURIComponent(filename)}/duplicate`, { new_filename: newFilename });
}

export async function fetchDatasets(): Promise<DatasetFolder[]> {
  const data = await apiFetch<{ datasets: DatasetFolder[] }>("/api/rag-project/datasets");
  return data.datasets;
}

export async function fetchSystemYamls(): Promise<SystemYamlInfo[]> {
  const data = await apiFetch<{ system_yamls: SystemYamlInfo[] }>("/api/rag-project/system-yamls");
  return data.system_yamls;
}

export interface LaunchRagRunOpts {
  mode?: "full" | "hardware_replay";
  qualityStorePath?: string;
  /** Name of a previous project under the projects root whose GT evals
   *  will be synthesized into a QualityStore before hardware_replay runs.
   *  Mutually exclusive with `qualityStorePath`. */
  fromProject?: string;
  hardwareConfig?: Record<string, unknown>;
  nRefineIterations?: number;
  /** If true, wipe prior-run state (evaluations/, pareto_*, optimizer/,
   *  registry/, resources/, summary.json counters) before launching so the
   *  Controller starts fresh from Phase 1. Dataset, config, and logs are
   *  preserved. Default false → checkpoint-resume from the prior Pareto. */
  reset?: boolean;
}

export async function launchRagRun(
  projectName: string,
  /** Pass null/undefined when resuming a project that already has data/
   *  on disk — the backend will reuse the in-project copy. */
  corpusData: string | null,
  qaData: string | null,
  opts?: LaunchRagRunOpts,
): Promise<Record<string, unknown>> {
  return apiPost("/api/rag-project/run", {
    project_name: projectName,
    corpus_data: corpusData,
    qa_data: qaData,
    mode: opts?.mode ?? "full",
    quality_store_path: opts?.qualityStorePath ?? null,
    from_project: opts?.fromProject ?? null,
    hardware_config: opts?.hardwareConfig ?? null,
    n_refine_iterations: opts?.nRefineIterations ?? 0,
    reset: opts?.reset ?? false,
  });
}

export interface SaveProjectQualityStoreResult {
  ok: boolean;
  project_name: string;
  store_name: string;
  store_path: string;
  mode: "existing" | "new";
  n_before: number;
  n_after: number;
  n_imported: number;
  n_total: number;
  dataset: Record<string, unknown>;
  imported_hashes: string[];
}

export async function saveProjectQualityToStore(
  projectName: string,
  req: { mode: "existing" | "new"; storeName: string },
): Promise<SaveProjectQualityStoreResult> {
  return apiPost(
    `/api/rag-project/projects/${encodeURIComponent(projectName)}/save-quality-store`,
    { mode: req.mode, store_name: req.storeName },
  );
}

export async function launchSystemTransferProject(req: {
  projectName: string;
  sourceProject: string;
  systemYaml: string;
  nRefineIterations?: number;
}): Promise<Record<string, unknown>> {
  return apiPost("/api/rag-project/system-transfer", {
    project_name: req.projectName,
    source_project: req.sourceProject,
    system_yaml: req.systemYaml,
    ...(req.nRefineIterations === undefined
      ? {}
      : { n_refine_iterations: req.nRefineIterations }),
  });
}

export async function fetchProjectStatus(projectName: string): Promise<ProjectStatus> {
  return apiFetch<ProjectStatus>(`/api/rag-project/status/${encodeURIComponent(projectName)}`);
}

/** One parsed row of all_evaluations.csv, served by the backend.
 *  `config` is flat dot-path knobs — the server parses its own Python-repr
 *  blob cells (the pipeline / corpus columns) with ast.literal_eval; the UI
 *  never re-parses repr strings. */
export interface ProjectEvaluationRow {
  eval_id: number;
  eval_type: string;
  eval_error: string;
  performance: number;
  quality: Record<string, number>;
  gt_metrics: { gt_bleu: number; gt_meteor: number; gt_rouge: number } | null;
  config: Record<string, string | number | boolean>;
}

/** Parsed evaluations for a project ("" = single-project root layout). */
export async function fetchProjectEvaluations(
  project: string,
): Promise<ProjectEvaluationRow[]> {
  const data = await apiFetch<{ evaluations: ProjectEvaluationRow[] }>(
    `/api/rag-project/evaluations?project=${encodeURIComponent(project)}`,
  );
  return data.evaluations;
}

export async function killRagProject(projectName: string): Promise<void> {
  await apiPost(`/api/rag-project/kill/${encodeURIComponent(projectName)}`, {});
}

export async function deleteRagProject(projectName: string): Promise<void> {
  await apiDelete(`/api/rag-project/projects/${encodeURIComponent(projectName)}`);
}

export async function fetchCostModelMapping(): Promise<{ mapping: Record<string, string>; available_cost_models: string[]; read_only?: boolean }> {
  return apiFetch("/api/rag-project/registry/cost-model-mapping");
}

// The component -> cost-model bindings are code (cost_model/dispatch.py),
// validated against the taxonomy at import; the JSON-backed live-edit PUT is
// retired and the table is read-only.

export interface ClaudeCodeHealth {
  installed: boolean;
  working: boolean;
  error: string | null;
  model: string | null;
  response?: string;
}

export async function checkClaudeCodeHealth(model: string = "haiku"): Promise<ClaudeCodeHealth> {
  return apiFetch(`/api/rag-project/health/claude-code?model=${encodeURIComponent(model)}`);
}

export interface CodexCliHealth {
  installed: boolean;
  working: boolean;
  error: string | null;
  model: string | null;
  response?: string;
}

export async function checkCodexCliHealth(model: string = "gpt-5.4"): Promise<CodexCliHealth> {
  return apiFetch(`/api/rag-project/health/codex-cli?model=${encodeURIComponent(model)}`);
}
