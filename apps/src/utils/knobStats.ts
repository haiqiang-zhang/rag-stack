import type { Evaluation } from "@/types";
import type { KnobDim, KnobValue } from "@/utils/searchKnobs";

/**
 * Knob ⟷ objective association statistics for the Trade-off view.
 *
 * Numeric knobs → Spearman rank correlation (signed, −1…1): monotone
 * "turning this up moves the objective this way".
 * Categorical knobs → correlation ratio η (unsigned, 0…1): "how much of the
 * objective's variance the choice explains", plus the best-mean value label.
 */

const MIN_POINTS = 4;

function rank(xs: number[]): number[] {
  const idx = xs.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
  const ranks = new Array<number>(xs.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    const r = (i + j) / 2 + 1; // average rank for ties
    for (let k = i; k <= j; k++) ranks[idx[k][1]] = r;
    i = j + 1;
  }
  return ranks;
}

function pearson(xs: number[], ys: number[]): number | null {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  if (sxx <= 0 || syy <= 0) return null; // constant input — no signal
  return sxy / Math.sqrt(sxx * syy);
}

/** Spearman ρ with tie-averaged ranks. Null below MIN_POINTS or degenerate. */
export function spearman(xs: number[], ys: number[]): number | null {
  if (xs.length < MIN_POINTS) return null;
  return pearson(rank(xs), rank(ys));
}

/** Correlation ratio η = sqrt(SS_between / SS_total), 0…1. */
export function correlationRatio(categories: KnobValue[], ys: number[]): number | null {
  if (ys.length < MIN_POINTS) return null;
  const groups = new Map<string, number[]>();
  for (let i = 0; i < ys.length; i++) {
    const k = String(categories[i]);
    (groups.get(k) ?? groups.set(k, []).get(k)!).push(ys[i]);
  }
  if (groups.size < 2) return null;
  const mean = ys.reduce((a, b) => a + b, 0) / ys.length;
  let ssTotal = 0;
  for (const y of ys) ssTotal += (y - mean) ** 2;
  if (ssTotal <= 0) return null;
  let ssBetween = 0;
  for (const vals of groups.values()) {
    const gm = vals.reduce((a, b) => a + b, 0) / vals.length;
    ssBetween += vals.length * (gm - mean) ** 2;
  }
  return Math.sqrt(Math.max(0, Math.min(1, ssBetween / ssTotal)));
}

/** Value with the highest group mean (≥2 samples), for categorical tooltips. */
function bestCategory(categories: KnobValue[], ys: number[]): string | null {
  const groups = new Map<string, number[]>();
  for (let i = 0; i < ys.length; i++) {
    const k = String(categories[i]);
    (groups.get(k) ?? groups.set(k, []).get(k)!).push(ys[i]);
  }
  let best: string | null = null;
  let bestMean = -Infinity;
  for (const [k, vals] of groups) {
    if (vals.length < 2) continue;
    const m = vals.reduce((a, b) => a + b, 0) / vals.length;
    if (m > bestMean) {
      bestMean = m;
      best = k;
    }
  }
  return best;
}

export interface KnobImpact {
  dim: KnobDim;
  /** Signed Spearman ρ for numeric dims; null for categorical/insufficient. */
  qualityRho: number | null;
  perfRho: number | null;
  /** Unsigned η for categorical dims; null for numeric/insufficient. */
  qualityEta: number | null;
  perfEta: number | null;
  /** Categorical only: value with the best mean objective. */
  bestQualityValue: string | null;
  bestPerfValue: string | null;
  nQuality: number;
  nPerf: number;
  /** max |association| across both objectives — sort key. */
  strength: number;
}

/**
 * Associations for every varying dim.
 * Quality pairs come from GT evals (only they have quality); performance
 * pairs from all feasible successes (CM rows count — more statistical power).
 */
export function buildKnobImpacts(
  dims: KnobDim[],
  evaluations: Evaluation[],
  getQuality: (ev: Evaluation) => number | null,
  getPerformance: (ev: Evaluation) => number,
  isFeasible: (ev: Evaluation) => boolean,
): KnobImpact[] {
  // Deduplicate by eval_id (CM→GT upgrades append rows): last row wins.
  const byId = new Map<number, Evaluation>();
  for (const ev of evaluations) byId.set(ev.eval_id, ev);
  const evals = [...byId.values()];

  const out: KnobImpact[] = [];
  for (const dim of dims) {
    const qX: KnobValue[] = [];
    const qY: number[] = [];
    const pX: KnobValue[] = [];
    const pY: number[] = [];
    for (const ev of evals) {
      const v = dim.values.get(ev.eval_id);
      if (v == null) continue;
      if (ev.eval_type === "GT") {
        const q = getQuality(ev);
        if (q != null && Number.isFinite(q)) {
          qX.push(v);
          qY.push(q);
        }
      }
      if ((ev.eval_type === "GT" || ev.eval_type === "CM") && isFeasible(ev)) {
        const p = getPerformance(ev);
        if (Number.isFinite(p)) {
          pX.push(v);
          pY.push(p);
        }
      }
    }

    let qualityRho: number | null = null;
    let perfRho: number | null = null;
    let qualityEta: number | null = null;
    let perfEta: number | null = null;
    let bestQualityValue: string | null = null;
    let bestPerfValue: string | null = null;

    if (dim.numeric) {
      qualityRho = spearman(qX as number[], qY);
      perfRho = spearman(pX as number[], pY);
    } else {
      qualityEta = correlationRatio(qX, qY);
      perfEta = correlationRatio(pX, pY);
      if (qualityEta != null) bestQualityValue = bestCategory(qX, qY);
      if (perfEta != null) bestPerfValue = bestCategory(pX, pY);
    }

    const strength = Math.max(
      Math.abs(qualityRho ?? 0),
      Math.abs(perfRho ?? 0),
      qualityEta ?? 0,
      perfEta ?? 0,
    );
    out.push({
      dim,
      qualityRho,
      perfRho,
      qualityEta,
      perfEta,
      bestQualityValue,
      bestPerfValue,
      nQuality: qY.length,
      nPerf: pY.length,
      strength,
    });
  }
  return out.sort((a, b) => b.strength - a.strength);
}
