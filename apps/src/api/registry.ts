/**
 * Pipeline registry API endpoints.
 */

import { apiFetch, apiPost } from "./base";

// ── Types ────────────────────────────────────────────────────────────────

export interface NodeTypeRegistry {
  [nodeType: string]: {
    modules: string[];
    aliases?: string[];
    component_capabilities?: Record<string, ComponentCapabilities>;
    component_parameters?: Record<string, Record<string, ComponentParameter>>;
  };
}

export interface ComponentCapabilities {
  execution_supported?: boolean | null;
  measured_supported?: boolean | null;
  cost_model_supported?: boolean | null;
  supported_dataflows?: string[];
  reason?: string;
  reasons?: string[];
}

export interface ComponentParameter {
  label?: string;
  default?: unknown;
  description?: string;
  type?: string;
}

export interface PipelinePlan {
  valid: boolean;
  errors: string[];
  workflow: {
    execution_policy: string;
    execution_order: string[];
    nodes: Array<{
      node_id: string;
      stage: string;
      components: string[];
      inputs: Record<string, string[]>;
      operations: string[];
    }>;
  } | null;
  capabilities: ComponentCapabilities;
}

export function fetchPipelinePlan(config: Record<string, unknown>): Promise<PipelinePlan> {
  return apiPost<PipelinePlan>("/api/rag-project/plan", { config });
}

// ── Endpoints ────────────────────────────────────────────────────────────

export async function fetchNodeTypeRegistry(): Promise<NodeTypeRegistry> {
  const data = await apiFetch<{ stages: NodeTypeRegistry }>("/api/rag-project/registry/node-types");
  return data.stages;
}

export async function fetchVectorDbTypes(): Promise<string[]> {
  const data = await apiFetch<{ vectordb_types: string[] }>("/api/rag-project/registry/vectordb-types");
  return data.vectordb_types;
}

// ── 7-section schema enums + per-backend vocab (server is source of truth) ──

export interface ConfigEnums {
  eval_backend: string[];
  performance_source: string[];
  rag_ir_mode: string[];
  combined_quality_mode: string[];
  performance_selection: string[];
}

export async function fetchConfigEnums(): Promise<ConfigEnums> {
  return apiFetch<ConfigEnums>("/api/rag-project/registry/enums");
}

/** `pipeline.rag_dataflow` vocabulary: the co-mode branches plus the react
 *  engine contract (which stages the loop runs; which sub-knobs it requires).
 *  Single source of truth is the backend search-space builder. */
export interface DataflowRegistry {
  branches: string[];
  react_active_stages: string[];
  react_required_knobs: string[];
}

export async function fetchDataflowRegistry(): Promise<DataflowRegistry> {
  return apiFetch<DataflowRegistry>("/api/rag-project/registry/dataflows");
}

export async function fetchMetricsForBackend(backend: string): Promise<string[]> {
  const data = await apiFetch<{ metrics: string[] }>(
    `/api/rag-project/registry/metrics?backend=${encodeURIComponent(backend)}`,
  );
  return data.metrics;
}

export async function fetchPipelineMethods(): Promise<string[]> {
  const data = await apiFetch<{ methods: string[] }>("/api/rag-project/registry/pipeline-methods");
  return data.methods;
}

export async function fetchGeneratorBackends(): Promise<string[]> {
  const data = await apiFetch<{ backends: string[] }>("/api/rag-project/registry/generator-backends");
  return data.backends;
}

/** Cost-model stage → stage mapping. Single source of truth lives in the
 *  backend (rag_stack.taxonomy); served here
 *  instead of a per-project JSON snapshot. */
export type {
  TaxonomyComponentDetail,
  TaxonomyStageInfo,
} from "@/types";
import type { TaxonomyStageInfo } from "@/types";

export async function fetchTaxonomyStages(): Promise<Record<string, TaxonomyStageInfo>> {
  const data = await apiFetch<{ stages: Record<string, TaxonomyStageInfo> }>(
    "/api/rag-project/taxonomy/stages",
  );
  return data.stages;
}
