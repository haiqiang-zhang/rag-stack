import type { Evaluation } from "@/types";

export type FormulaType = "mean" | "weighted_mean" | "min" | "max" | "geometric_mean";

/**
 * Apply a combination formula to quality sub-metrics.
 * Excludes the "combined" key (it's the output, not an input).
 */
export function applyFormula(
  metrics: Record<string, number>,
  formula: FormulaType,
  weights: Record<string, number>,
): number {
  const entries = Object.entries(metrics).filter(
    ([k, v]) => k !== "combined" && v != null && !isNaN(v),
  );
  if (entries.length === 0) return 0;

  switch (formula) {
    case "weighted_mean": {
      const totalW = entries.reduce((s, [k]) => s + (weights[k] ?? 1), 0);
      return totalW > 0
        ? entries.reduce((s, [k, v]) => s + (weights[k] ?? 1) * v, 0) / totalW
        : 0;
    }
    case "min":
      return Math.min(...entries.map(([, v]) => v));
    case "max":
      return Math.max(...entries.map(([, v]) => v));
    case "geometric_mean": {
      const product = entries.reduce((acc, [, v]) => acc * Math.max(v, 1e-10), 1);
      return Math.pow(product, 1 / entries.length);
    }
    default: // "mean"
      return entries.reduce((s, [, v]) => s + v, 0) / entries.length;
  }
}

/**
 * Compute Pareto frontier from evaluations using a quality getter.
 * Performance: lower is better (negative latency). Quality: higher is better.
 * Returns non-dominated GT evaluations sorted by performance ascending.
 */
export function computePareto(
  evaluations: Evaluation[],
  getQuality: (ev: Evaluation) => number | null,
  getPerformance: (ev: Evaluation) => number = (ev) => ev.performance,
  isFeasible: (ev: Evaluation) => boolean = () => true,
): Evaluation[] {
  // Drop SLO-infeasible evals before Pareto: they have score = INFEASIBLE_SCORE
  // (-1e9), so they won't dominate anything on the perf axis, but they CAN be
  // non-dominated on the quality axis (highest quality with no perf rival),
  // which both lands a useless point on the frontier and stretches the chart's
  // x-axis to ~-1e9.
  const gt = evaluations
    .filter((ev) => ev.eval_type === "GT" && isFeasible(ev))
    .map((ev) => ({ ev, q: getQuality(ev), p: getPerformance(ev) }))
    .filter((x): x is { ev: Evaluation; q: number; p: number } => x.q != null);

  if (gt.length === 0) return [];

  // Non-dominated filter (both objectives: higher is better)
  const pareto = gt.filter(({ p: pa, q: qa }, i) =>
    !gt.some(
      ({ p: pb, q: qb }, j) =>
        i !== j &&
        pb >= pa &&
        qb >= qa &&
        (pb > pa || qb > qa),
    ),
  );

  return pareto.sort((a, b) => a.p - b.p).map((x) => x.ev);
}

/**
 * Generate a CSV string for the Pareto frontier, matching server format.
 */
export function exportParetoCSV(
  pareto: Evaluation[],
  getQuality: (ev: Evaluation) => number | null,
  getPerformance: (ev: Evaluation) => number = (ev) => ev.performance,
): string {
  if (pareto.length === 0) return "";

  // Collect all config keys across Pareto points
  const configKeys = new Set<string>();
  for (const ev of pareto) {
    for (const k of Object.keys(ev.config)) configKeys.add(k);
  }
  const sortedKeys = [...configKeys].sort();

  const header = [...sortedKeys, "performance", "quality", "eval_id"];
  const rows = pareto.map((ev) => {
    const vals: string[] = sortedKeys.map((k) => String(ev.config[k] ?? ""));
    vals.push(getPerformance(ev).toFixed(6));
    vals.push((getQuality(ev) ?? 0).toFixed(6));
    vals.push(String(ev.eval_id));
    return vals;
  });

  const escape = (s: string) =>
    s.includes(",") || s.includes('"') || s.includes("\n")
      ? `"${s.replace(/"/g, '""')}"`
      : s;

  return [
    header.map(escape).join(","),
    ...rows.map((r) => r.map(escape).join(",")),
  ].join("\n");
}
