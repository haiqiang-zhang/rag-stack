/** Data stored in a pipeline node (xyflow node.data) */
export interface PipelineNodeData {
  nodeType: string;
  nodeLineName: string;
  nodeIndex: number; // position within the node_line
  modules: ModuleConfig[];
  topK?: number | number[] | Record<string, unknown>;
  optional?: boolean;
  strategy?: {
    metrics?: string[];
    speed_threshold?: number;
    top_k?: number;
    retrieval_modules?: Record<string, unknown>[];
  };
  /** All other params from the YAML node (excluding stage, modules, strategy, optional) */
  extraParams: Record<string, unknown>;
}

export interface ModuleConfig {
  component: string;
  [key: string]: unknown; // all module params
}

/** Data stored in a vectordb node */
export interface VectorDbNodeData {
  name: string;
  dbType: string;
  params: Record<string, unknown>;
}

/** Data stored in a node-line label node */
export interface NodeLineLabelData {
  nodeLineName: string;
  nodeLineIndex: number;
}

/** Registry of node types and their available modules */
export type { NodeTypeRegistry } from "@/api/registry";

/** The full 6-section config, minus the pipeline node_lines the visual graph
 *  owns. Held verbatim during visual editing and written back on save. The
 *  visual editor only rewrites `algo_search_space.pipeline.node_lines`; every
 *  other key (including a flashrag `pipeline.methods`) round-trips untouched. */
export interface NonVisualConfig {
  global?: Record<string, unknown>;
  dataset?: Record<string, unknown>;
  system?: Record<string, unknown>;
  optimizer?: Record<string, unknown>;
  eval_backend_setting?: Record<string, unknown>;
  algo_search_space?: {
    /** Corpus chunker sweep spec (algo_search_space.corpus). */
    corpus?: CorpusConfig;
    /** Vector-index search space (algo_search_space.vectordb — the vectordb
     *  section is algo space; the backend resolver re-exposes the chosen,
     *  resolved blocks at runtime `config.vectordb`). */
    vectordb?: VectorDbEntry[];
    pipeline?: {
      node_lines?: NodeLineEntry[];
      methods?: Record<string, unknown>[];
      [k: string]: unknown;
    };
    [k: string]: unknown;
  };
  /** Tolerate any extra / internal keys (_token_stats, _pareto_objectives, …). */
  [k: string]: unknown;
}

/** Corpus / chunker config. Each chunker field is either a discrete value
 *  set (scalar = 1 value, list = multiple ChoiceParameter values) or a
 *  `{range: [lo, hi], n_grid?}` dict (RangeParameter). */
export interface CorpusConfig {
  raw_path?: string;
  chunker?: {
    component?: string | string[];
    chunk_size?: number | number[] | { range: [number, number]; n_grid?: number };
    chunk_overlap?: number | number[] | { range: [number, number]; n_grid?: number };
    [key: string]: unknown;
  };
}

/** Full parsed config structure */
export interface ParsedPipelineConfig {
  vectordb: VectorDbEntry[];
  nodeLines: NodeLineEntry[];
  nonVisual: NonVisualConfig;
}

export interface VectorDbEntry {
  name: string;
  db_type: string;
  [key: string]: unknown;
}

export interface NodeLineEntry {
  node_line_name: string;
  nodes: NodeEntry[];
}

export interface NodeEntry {
  stage: string;
  modules?: ModuleConfig[];
  strategy?: Record<string, unknown>;
  top_k?: number | number[] | Record<string, unknown>;
  [key: string]: unknown;
}
