import { SWEEP_STAGES } from "@/lib/stages";

/** Run summary from summary.json */
export interface Summary {
  status: "created" | "running" | "completed" | "failed";
  timestamp: string;
  pid?: number;
  cm_count: number;
  gt_count: number;
  n_evaluations: number;
  n_pareto_points: number;
  n_invalid_configs: number;
  n_assembly_results: number;
}

/** Outcome of one evaluation — mirrors backend `rag_stack.types.EvalType`.
 *  Success: GT (quality [+perf]) / CM (perf only). Failure buckets: DUPLICATE
 *  (already known-bad, not re-run), OOM (GPU out-of-memory), INVALID (any other
 *  failure — build/resolve error, over-budget layout, non-OOM launch fail, …).
 *  The precise message is in `eval_error`. */
export type EvalType = "GT" | "CM" | "DUPLICATE" | "OOM" | "INVALID";

/** GT/CM produced real objective values the optimizer observes. */
export function isEvalSuccess(t: string): boolean {
  return t === "GT" || t === "CM";
}
/** DUPLICATE/OOM/INVALID (and any unknown/legacy value) — no real scores. */
export function isEvalFailure(t: string): boolean {
  return !isEvalSuccess(t);
}

/** Single evaluation record from all_evaluations.csv (served parsed by the
 *  backend — GET /api/rag-project/evaluations). */
export interface Evaluation {
  eval_id: number;
  eval_type: EvalType;
  /** Failure reason for non-success rows (OOM/DUPLICATE/INVALID detail +
   *  message); empty for successful GT/CM rows. */
  eval_error?: string;
  performance: number;
  quality: number | null;
  /** All quality__* columns keyed by suffix, e.g. { combined: 0.95, deepeval_faithfulness: 0.91 } */
  quality_metrics: Record<string, number>;
  is_pareto: boolean;
  /** Flat dot-path search knobs, e.g.
   *  `pipeline.sequential.semantic_retrieval.top_k: 8` — the backend parses
   *  the CSV's Python-repr blob cells and flattens them. */
  config: Record<string, string | number | boolean>;
  gt_metrics: GtMetrics | null;
}

/** Ground truth metrics */
export interface GtMetrics {
  gt_bleu: number;
  gt_meteor: number;
  gt_rouge: number;
}

/** Single point from rago_sweep_NNNN.csv */
export interface RagoSweepPoint {
  latency_s: number;
  latency_s_ttft: number;
  latency_s_tpot: number;
  qps_per_chip: number;
  qps: number;
  qps_semantic_retrieval_encode?: number;
  qps_semantic_retrieval_vectorsearch?: number;
  qps_passage_reranker?: number;
  qps_passage_compressor?: number;
  qps_query_expansion_prefill?: number;
  qps_query_expansion_decode?: number;
  qps_generator_prefill?: number;
  qps_generator_decode?: number;
  batch_size_request: number;
  batch_size_semantic_retrieval_encode?: number;
  batch_size_semantic_retrieval_vectorsearch?: number;
  batch_size_passage_reranker?: number;
  batch_size_passage_compressor?: number;
  batch_size_query_expansion_prefill?: number;
  batch_size_query_expansion_decode?: number;
  batch_size_generator_prefill?: number;
  batch_size_generator_decode?: number;
  batch_size_decode?: number;
  /** Iterative-only: batch size for the in-iteration retrieve+prefill calls. */
  batch_size_iterative?: number;
  num_chips_e2e: number;
  num_chips_semantic_retrieval_encode?: number;
  num_chips_passage_reranker?: number;
  num_chips_passage_compressor?: number;
  num_chips_query_expansion_prefill?: number;
  num_chips_query_expansion_decode?: number;
  num_chips_generator_prefill?: number;
  num_chips_generator_decode?: number;
  num_retrieval_servers?: number;
  placement_policy: string;
  /** Sequential-only: collocation grouping like "[['retrieval'], ...]". */
  collocation_strategy?: string;
  scheduling_policy: string;
  /** Sentinel column added by rag_assembly to tag iterative vs sequential. */
  pipeline_mode?: string;
  /** Iterative-only: how many retrievals interleaved with decode. */
  num_retrievals?: number;
  [key: string]: string | number | undefined;
}

/** Assembly summary from evaluations/eval_NNNN/assembly.json */
export interface AssemblyResult {
  performance_score: number;
  best_hw_params: Record<string, string | number>;
  n_sweep_points: number;
  n_pareto_points: number;
  per_stage?: Record<
    string,
    {
      latency_s: number;
      throughput_qps: number;
      operators: Array<{
        name: string;
        compute_time_s: number;
        memory_time_s: number;
        time_s: number;
        bottleneck: string;
      }>;
    }
  >;
}

/** Single row from evaluations/eval_NNNN/sweep_dfs/<stage>.csv (normalized).
 *  num_chips: from `hardware.num_chips` (LLM/rerank stages) or
 *  `hardware.num_servers` (retrieval) — whichever the source CSV provides. */
export interface SweepRow {
  batch_size: number;
  num_chips: number;
  /** Model forward time in ms. For decode-family rows this is PER STEP —
   *  multiply by `dec_steps` for the batch's wall time. */
  time_ms: number;
  /** model.dec_steps: decode steps the batch runs (1 for prefill/encode;
   *  absent for retrieval sweeps). */
  dec_steps?: number;
  /** TP/PP split the cost model picked at this sweep point (LLM stages only;
   *  undefined for retrieval sweeps). */
  winning_tp?: number;
  winning_pp?: number;
}

/** One user stage of the central taxonomy
 *  (GET /api/rag-project/taxonomy/stages). */
export interface TaxonomyComponentDetail {
  name: string;
  /** false = pass-through (no CM pricing, no PerformanceStage). */
  priced: boolean;
  performance_stages: string[];
}
export interface TaxonomyStageInfo {
  components: string[];
  component_details?: TaxonomyComponentDetail[];
  performance_stages: string[];
}
export type TaxonomyStages = Record<string, TaxonomyStageInfo>;

/** A single enabled node in a pipeline_config.json, enriched with
 *  display params, the stages it owns, and any per-node max_batch_size override. */
export interface PipelineNode {
  node_line: string;
  stage: string;
  component: string;
  display: { label: string; value: string }[];
  stages: string[];
  override?: number | Record<string, number>;
}

/** Axis option for chart selectors */
export interface AxisOption {
  key: string;
  label: string;
  group: string;
}

/** RAGO sweep axis options grouped by category. Per-stage entries derive from
 *  the canonical vocabulary in lib/stages.ts (dataflow order). */
export const RAGO_AXIS_OPTIONS: AxisOption[] = [
  // Performance metrics
  { key: "latency_s", label: "Latency (s)", group: "Performance" },
  { key: "latency_s_ttft", label: "TTFT (s)", group: "Performance" },
  { key: "latency_s_tpot", label: "TPOT (s)", group: "Performance" },
  { key: "qps", label: "QPS", group: "Performance" },
  { key: "qps_per_chip", label: "QPS/Chip", group: "Performance" },
  // Per-stage QPS
  ...SWEEP_STAGES.map((s) => ({
    key: `qps_${s.name}`,
    label: `QPS ${s.label}`,
    group: "Per-stage QPS",
  })),
  // Hardware
  { key: "num_chips_e2e", label: "Chips (E2E)", group: "Hardware" },
  ...SWEEP_STAGES.map((s) => ({
    key: s.chipsColumn ?? `num_chips_${s.name}`,
    label: s.chipsLabel ?? `Chips ${s.label}`,
    group: "Hardware",
  })),
  // Batch sizes
  { key: "batch_size_request", label: "Batch Request", group: "Batch Size" },
  ...SWEEP_STAGES.map((s) => ({
    key: s.batchColumn ?? `batch_size_${s.name}`,
    label: `Batch ${s.label}`,
    group: "Batch Size",
  })),
  // Iterative-only
  { key: "batch_size_iterative", label: "Batch Iterative", group: "Iterative" },
  { key: "num_retrievals", label: "Num Retrievals", group: "Iterative" },
];

/** Color encoding options for RAGO sweep */
export const RAGO_COLOR_OPTIONS: AxisOption[] = [
  { key: "placement_policy", label: "Placement Policy", group: "Category" },
  {
    key: "collocation_strategy",
    label: "Collocation Strategy",
    group: "Category",
  },
  { key: "scheduling_policy", label: "Scheduling Policy", group: "Category" },
  { key: "pipeline_mode", label: "Pipeline Mode", group: "Category" },
];
