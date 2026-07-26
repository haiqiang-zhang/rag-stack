import type { RagoSweepPoint } from "@/types";

export type SelectionPolicy =
  | "min_latency"
  | "min_ttft"
  | "min_tpot"
  | "max_throughput"
  | "weighted"
  | "constrained";

export type PerformanceMetricKey =
  | "latency_s"
  | "ttft_s"
  | "tpot_s"
  | "throughput_qps"
  | "throughput_qps_per_chip";

/**
 * Single source of truth for performance metrics, mirrors
 * `rag_stack.cost_model.selection.METRIC_SPEC` on the Python side.
 *
 * Adding a new metric only requires adding one entry here; all helpers and
 * UI options consume this map.
 */
interface MetricSpec {
  /** Column in RagoSweepPoint / rago_sweep CSV. */
  column: keyof RagoSweepPoint;
  /** True if higher values are better. */
  higherIsBetter: boolean;
  /** Human-readable label for UI. */
  label: string;
  /** Unit string for UI (suffix shown after numeric input). */
  unit: string;
}

export const METRIC_SPEC: Record<PerformanceMetricKey, MetricSpec> = {
  latency_s:               { column: "latency_s",      higherIsBetter: false, label: "Latency (E2E)", unit: "s" },
  ttft_s:                  { column: "latency_s_ttft", higherIsBetter: false, label: "TTFT",          unit: "s" },
  tpot_s:                  { column: "latency_s_tpot", higherIsBetter: false, label: "TPOT",          unit: "s" },
  throughput_qps:          { column: "qps",            higherIsBetter: true,  label: "Throughput (QPS)",     unit: "qps" },
  throughput_qps_per_chip: { column: "qps_per_chip",   higherIsBetter: true,  label: "Throughput (QPS/chip)", unit: "qps" },
};

/** Score returned for configurations with no SLO-feasible point. Mirrors
 *  Python `INFEASIBLE_SCORE`. Far below any feasible score so the cliff is
 *  visible to downstream MOBO / Pareto computation. */
export const INFEASIBLE_SCORE = -1e9;

// Convenience re-exports — keep the surface that callers already use.
export const METRIC_TO_COLUMN: Record<PerformanceMetricKey, keyof RagoSweepPoint> =
  Object.fromEntries(
    (Object.entries(METRIC_SPEC) as [PerformanceMetricKey, MetricSpec][])
      .map(([k, v]) => [k, v.column]),
  ) as Record<PerformanceMetricKey, keyof RagoSweepPoint>;

export const METRIC_LABEL: Record<PerformanceMetricKey, string> =
  Object.fromEntries(
    (Object.entries(METRIC_SPEC) as [PerformanceMetricKey, MetricSpec][])
      .map(([k, v]) => [k, v.label]),
  ) as Record<PerformanceMetricKey, string>;

export const METRIC_MAXIMIZE: Record<PerformanceMetricKey, boolean> =
  Object.fromEntries(
    (Object.entries(METRIC_SPEC) as [PerformanceMetricKey, MetricSpec][])
      .map(([k, v]) => [k, v.higherIsBetter]),
  ) as Record<PerformanceMetricKey, boolean>;

export const PERFORMANCE_METRIC_KEYS = Object.keys(METRIC_SPEC) as PerformanceMetricKey[];

export interface PerformancePolicy {
  selection: SelectionPolicy;
  /** For "weighted": weights keyed by PerformanceMetricKey. */
  weights?: Partial<Record<PerformanceMetricKey, number>>;
  /** For "constrained": threshold per metric (natural sense — latency ≤ L*, qps ≥ Q*). */
  constraint?: Partial<Record<PerformanceMetricKey, number>>;
  /** For "constrained": which metric to optimize within the feasible set. */
  optimize?: PerformanceMetricKey;
}

export interface SelectionResult {
  row: RagoSweepPoint | null;
  /** MOBO-style score: higher is better. Mirrors Python `apply_selection`.
   *  For `constrained` infeasible this is `INFEASIBLE_SCORE` — use it for
   *  Pareto / sorting (cliff is correct), but use `displayScore` for charts. */
  score: number;
  /** False when `constrained` had no feasible point. Undefined for other policies. */
  feasible?: boolean;
  /**
   * X-axis-friendly score for visualisation. Equals `score` for everything
   * except `constrained` infeasible, where it falls back to the *unconstrained*
   * best value of the `optimize` metric (still in higher-is-better convention,
   * so it lives on the same x scale as feasible points). This lets the chart
   * place an infeasible point at its "would-be" x position with a red marker
   * instead of pinning all infeasible at one location.
   */
  displayScore: number;
}

// ---------------------------------------------------------------------------
// Direction helpers — mirror Python `_to_max` / `_violation`.
// ---------------------------------------------------------------------------

/** Convert a raw value to higher-is-better convention. */
function toMax(value: number, metric: PerformanceMetricKey): number {
  return METRIC_SPEC[metric].higherIsBetter ? value : -value;
}

/** Constraint violation amount (>= 0). 0 when satisfied. Threshold uses
 *  natural sense: latency ≤ L*, qps ≥ Q*. */
function violation(value: number, threshold: number, metric: PerformanceMetricKey): number {
  return METRIC_SPEC[metric].higherIsBetter
    ? Math.max(threshold - value, 0)
    : Math.max(value - threshold, 0);
}

function getMetric(row: RagoSweepPoint, metric: PerformanceMetricKey): number {
  const v = row[METRIC_SPEC[metric].column];
  return typeof v === "number" ? v : Number(v);
}

function selectSingle(rows: RagoSweepPoint[], metric: PerformanceMetricKey): SelectionResult {
  const higher = METRIC_SPEC[metric].higherIsBetter;
  let bestIdx = -1;
  let bestVal = higher ? -Infinity : Infinity;
  for (let i = 0; i < rows.length; i++) {
    const v = getMetric(rows[i], metric);
    if (!Number.isFinite(v)) continue;
    if (higher ? v > bestVal : v < bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  if (bestIdx < 0) return { row: null, score: 0, displayScore: 0 };
  const score = toMax(bestVal, metric);
  return { row: rows[bestIdx], score, displayScore: score };
}

/**
 * Apply a selection policy to RAGO sweep points. Mirrors
 * `rag_stack.cost_model.selection.apply_selection` on the client side so users
 * can change the policy in the UI without re-running the sweep.
 *
 * Score is always higher-is-better (project convention), so callers can use
 * the same value for sorting / Pareto filters across all policies.
 */
export function applyPerformanceSelection(
  rows: RagoSweepPoint[],
  policy: PerformancePolicy,
): SelectionResult {
  if (rows.length === 0) return { row: null, score: 0, displayScore: 0 };

  const sel = policy.selection;

  if (sel === "min_latency") return selectSingle(rows, "latency_s");
  if (sel === "min_ttft")    return selectSingle(rows, "ttft_s");
  if (sel === "min_tpot")    return selectSingle(rows, "tpot_s");
  if (sel === "max_throughput") return selectSingle(rows, "throughput_qps");

  if (sel === "weighted") {
    const weights = policy.weights ?? {};
    const activeMetrics = (Object.keys(weights) as PerformanceMetricKey[])
      .filter((m) => (weights[m] ?? 0) > 0 && m in METRIC_SPEC);
    if (activeMetrics.length === 0) return { row: null, score: 0, displayScore: 0 };

    // Per-sweep min-max normalization on `_to_max(values)` so each metric
    // contributes in [0, 1] in the higher-is-better direction.
    const scores = new Array<number>(rows.length).fill(0);
    for (const m of activeMetrics) {
      const w = weights[m]!;
      const maxVals = rows.map((r) => toMax(getMetric(r, m), m));
      let lo = Infinity;
      let hi = -Infinity;
      for (const v of maxVals) {
        if (!Number.isFinite(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      const rng = hi - lo;
      for (let i = 0; i < rows.length; i++) {
        const norm = rng > 0 ? (maxVals[i] - lo) / rng : 0;
        scores[i] += w * norm;
      }
    }
    let bestIdx = 0;
    for (let i = 1; i < scores.length; i++) {
      if (scores[i] > scores[bestIdx]) bestIdx = i;
    }
    return { row: rows[bestIdx], score: scores[bestIdx], displayScore: scores[bestIdx] };
  }

  if (sel === "constrained") {
    const constraint = policy.constraint ?? {};
    const optimize = policy.optimize ?? "throughput_qps";

    let feasible = rows;
    for (const metric of Object.keys(constraint) as PerformanceMetricKey[]) {
      const threshold = constraint[metric];
      if (threshold == null || !(metric in METRIC_SPEC)) continue;
      feasible = feasible.filter((r) => {
        const v = getMetric(r, metric);
        return METRIC_SPEC[metric].higherIsBetter ? v >= threshold : v <= threshold;
      });
    }

    if (feasible.length === 0) {
      // Hard cliff (mirrors Python). Return the closest-to-feasible row for
      // diagnostics but mark the result infeasible so consumers can skip it.
      let bestIdx = 0;
      let bestViol = Infinity;
      for (let i = 0; i < rows.length; i++) {
        let v = 0;
        for (const m of Object.keys(constraint) as PerformanceMetricKey[]) {
          const thr = constraint[m];
          if (thr == null || !(m in METRIC_SPEC)) continue;
          v += violation(getMetric(rows[i], m), thr, m);
        }
        if (v < bestViol) {
          bestViol = v;
          bestIdx = i;
        }
      }
      // `displayScore` = best `optimize` value the sweep can achieve when
      // we drop the SLO. Same higher-is-better scale as feasible scores, so
      // chart can plot the infeasible point at its "would-be" x position.
      const unconstrained = selectSingle(rows, optimize);
      return {
        row: rows[bestIdx],
        score: INFEASIBLE_SCORE,
        feasible: false,
        displayScore: unconstrained.score,
      };
    }

    const result = selectSingle(feasible, optimize);
    return { ...result, feasible: true };
  }

  return { row: null, score: 0, displayScore: 0 };
}
