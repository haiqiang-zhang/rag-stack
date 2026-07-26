/**
 * Domain-specific palettes that shadcn's neutral tokens don't cover.
 *
 * Visual styling everywhere else goes through shadcn semantic classes
 * (`bg-background`, `text-foreground`, `bg-primary`, etc.). The constants
 * here are for places where we hand a raw color string to a non-CSS API:
 * ECharts options, inline SVG fills, dynamic per-stage tinting.
 */

/**
 * Multi-series chart palette (16 colors). ECharts series rotate through this.
 * Picked for legibility on light backgrounds and good ordinal contrast.
 * Independent of shadcn's `--chart-1..5` because we routinely plot >5 series.
 *
 * Layout: first 8 are the bright primary hues; the second 8 are darker /
 * shifted variants so adjacent indices don't collide visually. Optimizer
 * sweeps with 12+ runs (e.g. NAS-Bench-201 ablations) need this depth —
 * the old 8-color palette was wrapping and producing duplicate colors.
 */
export const CHART_COLORS = [
  // Tier 1 — bright primaries
  "#0F766E", // deep teal
  "#2563EB", // blue
  "#F59E0B", // amber
  "#EF4444", // red
  "#8B5CF6", // violet
  "#06B6D4", // cyan
  "#EC4899", // pink
  "#84CC16", // lime
  // Tier 2 — darker / shifted variants for ordinal continuation
  "#92400E", // chestnut brown
  "#1E3A8A", // navy
  "#C026D3", // fuchsia
  "#16A34A", // forest green
  "#BE123C", // rose
  "#0E7490", // dark cyan
  "#A16207", // golden olive
  "#475569", // slate gray
];

// Per-stage identity colors live with the rest of the performance-stage
// vocabulary in lib/stages.ts (single rename surface); re-exported here so
// color consumers keep one import home.
export { stageColor } from "./stages";
