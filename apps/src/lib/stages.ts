/**
 * Canonical performance-stage vocabulary — the ONE frontend module that spells
 * out the cost model's PerformanceStage names and their display metadata.
 *
 * The authoritative stage↔component mapping lives in the backend taxonomy
 * (GET /api/rag-project/taxonomy/stages); what belongs here is everything the
 * UI must know *per stage name*: dataflow order, identity color, chart labels,
 * and the CSV column tokens that deviate from the generic patterns. When a
 * stage is renamed on the backend, update this list plus the optional column
 * typings on `RagoSweepPoint` — nothing else should hard-code stage names as
 * vocabulary (consumers may still branch on canonical names for domain logic).
 */

export interface PerfStageSpec {
  /** Full PerformanceStage name — the suffix of qps_* / batch_size_* /
   *  num_chips_* / tp_* / pp_* columns in rago_sweep.csv + best_hw_params. */
  name: string;
  /** Short human label for chart axis options ("Prefill", "Encode"). */
  label: string;
  /** Stable identity color. Reused across PipelineDiagram, StageBreakdownTable
   *  (row dot), TTFT waterfall segments, and StageScalingChart curves so the
   *  eye binds one hue to one stage everywhere. */
  color: string;
  /** Short token of the svc_<token>_s / qps_<token>_capacity columns in
   *  best_hw_params. Absent = the CM emits no service/capacity columns for
   *  this stage. */
  svcKey?: string;
  /** True when rago_sweep.csv carries per-stage qps_/batch_size_/chips columns
   *  for this stage (drives the chart axis option lists). */
  sweep?: boolean;
  /** Column overrides where the CSV deviates from the generic pattern. */
  chipsColumn?: string; // default `num_chips_${name}`
  chipsLabel?: string; // default `Chips ${label}`
  batchColumn?: string; // default `batch_size_${name}`
}

/** All performance stages in canonical left-to-right dataflow order. */
export const PERF_STAGES: PerfStageSpec[] = [
  { name: "query_expansion_prefill", label: "Query Prefill", color: "#06B6D4", sweep: true },
  { name: "query_expansion_decode", label: "Query Decode", color: "#F59E0B", sweep: true },
  { name: "semantic_retrieval_encode", label: "Encode", color: "#22D3EE", svcKey: "encode", sweep: true },
  {
    name: "semantic_retrieval_vectorsearch",
    label: "Retrieval",
    color: "#8B5CF6",
    svcKey: "retrieval",
    sweep: true,
    // Retrieval scales in whole servers, not accelerator chips.
    chipsColumn: "num_retrieval_servers",
    chipsLabel: "Retrieval Servers",
  },
  { name: "lexical_retrieval", label: "BM25", color: "#D97706" },
  { name: "retrieval_fusion", label: "Fusion", color: "#EA580C" },
  { name: "passage_reranker", label: "Reranker", color: "#0F766E", svcKey: "rerank", sweep: true },
  // Pass-through stage: priced=false in the taxonomy, so it never appears in
  // per-stage sweep columns — listed for its identity color + dataflow slot.
  { name: "passage_filter", label: "Filter", color: "#64748B" },
  { name: "passage_compressor", label: "Compressor", color: "#A855F7", svcKey: "compress", sweep: true },
  { name: "passage_compressor_prefill", label: "Compressor Prefill", color: "#A855F7", svcKey: "compress" },
  { name: "passage_compressor_decode", label: "Compressor Decode", color: "#F43F5E", svcKey: "compress" },
  { name: "generator_prefill", label: "Prefill", color: "#EC4899", svcKey: "prefill", sweep: true },
  {
    name: "generator_decode",
    label: "Decode",
    color: "#10B981",
    svcKey: "decode",
    sweep: true,
    // The sweep CSV's decode micro-batch column is plain `batch_size_decode`.
    batchColumn: "batch_size_decode",
  },
];

export const STAGE_BY_NAME: Record<string, PerfStageSpec> = Object.fromEntries(
  PERF_STAGES.map((s) => [s.name, s]),
);

/** Canonical dataflow order of stage names (table/waterfall row ordering). */
export const PERF_STAGE_ORDER = PERF_STAGES.map((s) => s.name);

/** Stages that carry per-stage sweep columns (chart axis candidates). */
export const SWEEP_STAGES = PERF_STAGES.filter((s) => s.sweep);

export const stageColor = (stage: string): string =>
  STAGE_BY_NAME[stage]?.color ?? "#94A3B8";

/** best_hw_params column holding this stage's chip/server count. */
export const stageChipsColumn = (stage: string): string =>
  STAGE_BY_NAME[stage]?.chipsColumn ?? `num_chips_${stage}`;
