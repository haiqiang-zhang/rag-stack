/**
 * projectSeries — user-defined grouping of benchmark projects into chart series.
 *
 * A "series" is what gets one line + one color on the ProjectBenchmark charts.
 * It is either a single project, or a user-created group of projects
 * (typically the same experiment config run under different seeds) whose HV
 * curves are aggregated client-side into mean ± 1 std — the same convention
 * BenchmarkChartGrid uses on the synthetic-functions tab.
 *
 * Grouping is always explicit (the user merges chips in the SeriesTray),
 * never inferred from project-name conventions. Group definitions persist in
 * localStorage keyed by project name, so re-selecting the same runs in a
 * later session reassembles the same series. Aggregation happens on the
 * frontend: /compare-hv already normalizes every selected project on shared
 * bounds, so regrouping never needs a refetch.
 */
import { CHART_COLORS } from "@/lib/theme";
import type { HvPoint } from "@/api/ragProject";

export interface SeriesGroup {
  id: string;
  name: string;
  members: string[]; // project names (full definition, selection-independent)
}

export interface BenchSeries {
  /** Stable key: `group:<id>` or `proj:<name>`. */
  key: string;
  name: string;
  color: string;
  /** Members that are currently SELECTED (what actually gets drawn). */
  members: string[];
  /** Present when this series is backed by a user-created group. */
  group?: SeriesGroup;
}

// ── Persistence ──────────────────────────────────────────────────────────

const GROUPS_KEY = "ragBenchmark.seriesGroups.v1";
const SELECTION_KEY = "ragBenchmark.selection.v1";
const HIDDEN_KEY = "ragBenchmark.hiddenSeries.v1";

export function loadGroups(): SeriesGroup[] {
  try {
    const raw = localStorage.getItem(GROUPS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (g): g is SeriesGroup =>
        g && typeof g.id === "string" && typeof g.name === "string" && Array.isArray(g.members),
    );
  } catch {
    return [];
  }
}

export function saveGroups(groups: SeriesGroup[]): void {
  try {
    localStorage.setItem(GROUPS_KEY, JSON.stringify(groups));
  } catch {
    /* storage full / privacy mode — grouping just won't persist */
  }
}

export function loadSelection(): string[] {
  try {
    const raw = localStorage.getItem(SELECTION_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === "string") : [];
  } catch {
    return [];
  }
}

export function saveSelection(selection: string[]): void {
  try {
    localStorage.setItem(SELECTION_KEY, JSON.stringify(selection));
  } catch {
    /* ignore */
  }
}

/** Hidden series keys (`group:<id>` / `proj:<name>`). Hiding is visual-only —
 *  the series keeps its selection, grouping, color, and normalization bounds;
 *  it just isn't drawn. Persisted so a hide survives reloads (the whole point:
 *  no more deselect-everything-then-reselect round trips). */
export function loadHidden(): string[] {
  try {
    const raw = localStorage.getItem(HIDDEN_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === "string") : [];
  } catch {
    return [];
  }
}

export function saveHidden(hidden: string[]): void {
  try {
    localStorage.setItem(HIDDEN_KEY, JSON.stringify(hidden));
  } catch {
    /* ignore */
  }
}

// ── Series assembly ──────────────────────────────────────────────────────

/**
 * Build the ordered series list for the current selection: groups first (in
 * stored order, only those with ≥1 selected member), then the remaining
 * selected projects as single-run series. Colors follow tray order.
 */
export function buildSeriesList(selected: string[], groups: SeriesGroup[]): BenchSeries[] {
  const grouped = new Set(groups.flatMap((g) => g.members));
  const list: BenchSeries[] = [];
  for (const g of groups) {
    const members = selected.filter((n) => g.members.includes(n));
    if (members.length === 0) continue;
    list.push({ key: `group:${g.id}`, name: g.name, color: "", members, group: g });
  }
  for (const name of selected) {
    if (!grouped.has(name)) {
      list.push({ key: `proj:${name}`, name, color: "", members: [name] });
    }
  }
  return list.map((s, i) => ({ ...s, color: CHART_COLORS[i % CHART_COLORS.length] }));
}

/** Group id; crypto.randomUUID needs a secure context, and the app is often
 *  served over plain HTTP from a lab box — fall back to a random slug. */
export function newGroupId(): string {
  return typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `g_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Default name for a merged series: longest common prefix of the member
 * names with any dangling seed tail ("_s4", "-seed", trailing digits or
 * separators) stripped. Purely a suggestion — the user renames freely; the
 * grouping itself never depends on names.
 */
export function suggestGroupName(members: string[]): string {
  if (members.length === 0) return "series";
  let prefix = members[0];
  for (const m of members.slice(1)) {
    while (prefix && !m.startsWith(prefix)) prefix = prefix.slice(0, -1);
  }
  const cleaned = prefix
    .replace(/[-_./]?(?:s(?:eed)?)?\d*$/i, "")
    .replace(/[-_./]+$/, "");
  return cleaned || "series";
}

// ── HV aggregation (mean ± 1 std across runs) ────────────────────────────

export interface AggregatedHv {
  mean: [number, number][];
  lower: [number, number][];
  upper: [number, number][];
  /** Per-eval-index sample std, for "mean ± std" tooltips. */
  stdByX: Map<number, number>;
  /** Number of runs that contributed data. */
  n: number;
}

function mean(xs: number[]): number {
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function stddev(xs: number[], mu: number): number {
  if (xs.length < 2) return 0;
  let s = 0;
  for (const x of xs) {
    const d = x - mu;
    s += d * d;
  }
  return Math.sqrt(s / (xs.length - 1));
}

/**
 * Aggregate several runs' running-HV series onto the union eval grid.
 * Runs with fewer evals are carried forward at their final HV (running
 * best-so-far HV stays at its last value for any larger budget) — the same
 * step-sampling BenchmarkChartGrid uses across seeds. Lower band is clamped
 * to ≥ 0 (normalized HV is non-negative).
 */
export function aggregateHv(runs: HvPoint[][]): AggregatedHv {
  const withData = runs.filter((r) => r.length > 0);
  const gridSet = new Set<number>();
  for (const run of withData) for (const p of run) gridSet.add(p.i);
  const grid = Array.from(gridSet).sort((a, b) => a - b);

  const meanData: [number, number][] = [];
  const lower: [number, number][] = [];
  const upper: [number, number][] = [];
  const stdByX = new Map<number, number>();

  for (const gx of grid) {
    const ys: number[] = [];
    for (const run of withData) {
      if (gx < run[0].i) continue; // run hasn't produced its first eval yet
      let yv: number | null = null;
      for (const p of run) {
        if (p.i <= gx) yv = p.hv;
        else break;
      }
      if (yv !== null && Number.isFinite(yv)) ys.push(yv);
    }
    if (ys.length === 0) continue;
    const mu = mean(ys);
    const sd = stddev(ys, mu);
    meanData.push([gx, mu]);
    lower.push([gx, Math.max(0, mu - sd)]);
    upper.push([gx, mu + sd]);
    stdByX.set(gx, sd);
  }

  return { mean: meanData, lower, upper, stdByX, n: withData.length };
}

/** Mean ± std of the runs' final HV values, for the summary table. */
export function finalHvStats(finals: number[]): { mean: number; std: number } {
  const mu = finals.length ? mean(finals) : 0;
  return { mean: mu, std: stddev(finals, mu) };
}
