import type { Evaluation } from "@/types";

/**
 * Search-space dimensions, derived from the flat dot-path `config` the backend
 * serves per evaluation (e.g. `pipeline.sequential.generator.model`).
 * This module owns the *display policy* — grouping, labels, what to hide —
 * while the backend owns parsing.
 */

export type KnobValue = string | number | boolean;

export interface KnobDim {
  /** Full dot-path key as served by the backend. */
  key: string;
  /** Display group — the pipeline stage for pipeline knobs, else the CSV
   *  column family (corpus, faiss_hnsw_index, …). */
  group: string;
  /** Short display label within the group. */
  label: string;
  /** True when every observed value is a number. */
  numeric: boolean;
  /** Distinct values across all evals (numeric asc / lexicographic). */
  distinct: KnobValue[];
  /** evalId → value, only for evals where the knob is present. */
  values: Map<number, KnobValue>;
}

/** Prompts are full templates — never a dimension worth plotting. */
function isHiddenKnob(key: string): boolean {
  return key === "prompt" || key.endsWith(".prompt");
}

/** Group + label policy for a dot-path key. */
export function splitKnobKey(key: string): { group: string; label: string } {
  const seg = key.split(".");
  if (seg[0] === "pipeline") {
    // pipeline.<branch>.<stage>.<param…> — the branch segment is the CSV
    // column identity, not information; the stage is the natural group.
    if (seg.length >= 4) return { group: seg[2], label: seg.slice(3).join(".") };
    if (seg.length === 3) return { group: "pipeline", label: seg[2] };
    return { group: "pipeline", label: seg.slice(1).join(".") || key };
  }
  return { group: seg[0], label: seg.slice(1).join(".") || key };
}

/** Stage-first ordering for knob groups (mirrors the pipeline reading order). */
const GROUP_ORDER = [
  "pipeline",
  "query_expansion",
  "semantic_retrieval",
  "passage_reranker",
  "passage_filter",
  "passage_compressor",
  "prompt_maker",
  "generator",
  "react",
  "corpus",
];

export function groupRank(group: string): number {
  const i = GROUP_ORDER.indexOf(group);
  return i >= 0 ? i : GROUP_ORDER.length;
}

function compareValues(a: KnobValue, b: KnobValue): number {
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b));
}

/**
 * Build the dimension list across a project's evaluations. Includes failure
 * rows (their configs are real points the optimizer proposed); consumers that
 * need outcome-conditioned subsets filter via `values` + their own getters.
 */
export function buildKnobDimensions(evaluations: Evaluation[]): KnobDim[] {
  const byKey = new Map<string, KnobDim>();
  for (const ev of evaluations) {
    for (const [key, raw] of Object.entries(ev.config)) {
      if (isHiddenKnob(key)) continue;
      let dim = byKey.get(key);
      if (!dim) {
        const { group, label } = splitKnobKey(key);
        dim = { key, group, label, numeric: true, distinct: [], values: new Map() };
        byKey.set(key, dim);
      }
      // Last row wins for a duplicated eval_id (CM → GT upgrade) — matches
      // the "latest row carries the freshest scalar" convention elsewhere.
      dim.values.set(ev.eval_id, raw);
      if (typeof raw !== "number") dim.numeric = false;
    }
  }
  for (const dim of byKey.values()) {
    const seen = new Set<KnobValue>();
    for (const v of dim.values.values()) seen.add(v);
    dim.distinct = [...seen].sort(compareValues);
  }
  return [...byKey.values()].sort(
    (a, b) =>
      groupRank(a.group) - groupRank(b.group) ||
      a.group.localeCompare(b.group) ||
      a.label.localeCompare(b.label),
  );
}

/** Dimensions the optimizer actually varied in this project. */
export function varyingDims(dims: KnobDim[]): KnobDim[] {
  return dims.filter((d) => d.distinct.length > 1);
}

/** Compact value formatting: strip HF org prefixes, shorten long strings. */
export function formatKnobValue(v: KnobValue | undefined): string {
  if (v == null) return "—";
  if (typeof v === "boolean") return v ? "true" : "false";
  const s = String(v);
  return s.includes("/") ? s.split("/").pop()! : s;
}
