/**
 * Shared color/shape encoding for "color the objective-space map by a search
 * knob" — consumed by ParetoChart (series building) and by the external
 * legend the EvalNavigator renders under the compact map.
 */
import type { KnobDim } from "@/utils/searchKnobs";
import { formatKnobValue } from "@/utils/searchKnobs";

/** What drives the map's point coloring. */
export type MapColorBy = KnobDim | "progress" | null;

/** Categorical sub-palette — validated (dataviz six checks, light surface):
 *  lightness band, chroma floor, CVD ΔE ≥ 12, contrast ≥ 3:1. Deliberately
 *  NOT theme.CHART_COLORS (that one fails chroma/contrast). */
export const KNOB_CATEGORY_COLORS = [
  "#0D9488",
  "#2563EB",
  "#D97706",
  "#BE123C",
  "#8B5CF6",
  "#0891B2",
  "#DB2777",
  "#65A30D",
];

/** Secondary (shape) encoding so categories never rely on color alone. */
export const KNOB_CATEGORY_SYMBOLS = [
  "circle",
  "rect",
  "triangle",
  "diamond",
  "roundRect",
  "pin",
  "arrow",
  "circle",
];

/** Unicode twins of the ECharts symbols, for HTML legends. */
const SYMBOL_GLYPHS: Record<string, string> = {
  circle: "●",
  rect: "■",
  triangle: "▲",
  diamond: "◆",
  roundRect: "▮",
  pin: "⬟",
  arrow: "➤",
};

/** Sequential ramp (one hue, light→dark) for numeric knob values. */
export const NUMERIC_RAMP = ["#99F6E4", "#0D9488", "#134E4A"];
/** Progression ramp (early → late), mirrors the ParetoChart idiom. */
export const PROGRESS_RAMP = ["#9CA3AF", "#0F766E", "#D97706"];

export const MAX_CATEGORIES = 8;

export interface CategoryBucket {
  /** Raw category value ("Other" for the overflow bucket). */
  value: string;
  label: string;
  color: string;
  symbol: string;
  glyph: string;
}

/** Fixed-order category buckets for a categorical dim (≤ MAX, then "Other").
 *  The chart and the HTML legend both derive from this so they can't drift. */
export function categoryBuckets(dim: KnobDim): CategoryBucket[] {
  const cats = dim.distinct.map(String);
  const shown = cats.length > MAX_CATEGORIES ? cats.slice(0, MAX_CATEGORIES - 1) : cats;
  const names = cats.length > MAX_CATEGORIES ? [...shown, "Other"] : shown;
  return names.map((value, i) => {
    const symbol = KNOB_CATEGORY_SYMBOLS[i % KNOB_CATEGORY_SYMBOLS.length];
    return {
      value,
      label: value === "Other" ? "Other" : formatKnobValue(value),
      color: KNOB_CATEGORY_COLORS[i % KNOB_CATEGORY_COLORS.length],
      symbol,
      glyph: SYMBOL_GLYPHS[symbol] ?? "●",
    };
  });
}

/** Bucket lookup: raw value → bucket (folding overflow into "Other"). */
export function bucketOf(buckets: CategoryBucket[], raw: unknown): CategoryBucket | undefined {
  const s = String(raw);
  return buckets.find((b) => b.value === s) ?? buckets.find((b) => b.value === "Other");
}
