/**
 * useConfigRegistries — fetch the 7-section schema's enumerable values from the
 * backend (the single source of truth) so the section forms never hardcode
 * valid values: the orthogonal-switch enums, the optimizer list, per-backend
 * metric vocab, FlashRAG pipeline methods, and generator backends.
 */
import { useEffect, useState } from "react";
import {
  fetchConfigEnums,
  fetchDataflowRegistry,
  fetchMetricsForBackend,
  fetchPipelineMethods,
  fetchGeneratorBackends,
  type ConfigEnums,
  type DataflowRegistry,
} from "@/api/registry";
import { fetchOptimizerRegistry } from "@/api/costModelClient";

export interface ConfigRegistries {
  enums: ConfigEnums;
  optimizers: { key: string; description?: string }[];
  metrics: { static_gt: string[]; flashrag: string[] };
  pipelineMethods: string[];
  generatorBackends: string[];
  dataflows: DataflowRegistry;
  loading: boolean;
}

const FALLBACK_ENUMS: ConfigEnums = {
  eval_backend: ["static_gt", "flashrag"],
  performance_source: ["cost_model", "measured"],
  rag_ir_mode: ["dynamic", "static"],
  combined_quality_mode: ["mean"],
  performance_selection: ["max_throughput"],
};

// Mirrors the backend /registry/dataflows payload (rag_search_space.py's
// REACT_ACTIVE_STAGES + the validator's branch set) for offline/API-down use.
const FALLBACK_DATAFLOWS: DataflowRegistry = {
  branches: ["sequential", "react"],
  react_active_stages: [
    "generator", "hybrid_retrieval", "passage_reranker", "semantic_retrieval",
  ],
  react_required_knobs: ["max_iter"],
};

export function useConfigRegistries(): ConfigRegistries {
  const [state, setState] = useState<ConfigRegistries>({
    enums: FALLBACK_ENUMS,
    optimizers: [],
    metrics: { static_gt: [], flashrag: [] },
    pipelineMethods: [],
    generatorBackends: [],
    dataflows: FALLBACK_DATAFLOWS,
    loading: true,
  });

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetchConfigEnums().catch(() => FALLBACK_ENUMS),
      fetchOptimizerRegistry().catch(() => []),
      fetchMetricsForBackend("static_gt").catch(() => []),
      fetchMetricsForBackend("flashrag").catch(() => []),
      fetchPipelineMethods().catch(() => []),
      fetchGeneratorBackends().catch(() => []),
      fetchDataflowRegistry().catch(() => FALLBACK_DATAFLOWS),
    ]).then(([enums, optimizers, mStatic, mFlash, methods, genBackends, dataflows]) => {
      if (cancelled) return;
      setState({
        enums: enums || FALLBACK_ENUMS,
        optimizers: (optimizers as { key: string; description?: string }[]) || [],
        metrics: { static_gt: mStatic, flashrag: mFlash },
        pipelineMethods: methods,
        generatorBackends: genBackends,
        dataflows: dataflows || FALLBACK_DATAFLOWS,
        loading: false,
      });
    });
    return () => { cancelled = true; };
  }, []);

  return state;
}
