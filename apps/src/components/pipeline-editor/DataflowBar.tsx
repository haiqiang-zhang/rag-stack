/**
 * DataflowBar — visual editor for `algo_search_space.pipeline.rag_dataflow`
 * (static_gt co-mode: sequential vs react swept as ONE flat knob over ONE
 * shared node_lines).
 *
 * Truthfulness contract:
 *   - both branches on   → the optimizer picks one per trial (a search dim);
 *   - one branch on      → the dataflow is fixed;
 *   - sequential only    → the key is REMOVED (absent = sequential, canonical);
 *   - react reachable    → `max_iter` (the loop's only round cap) is required,
 *     mirroring ConfigValidator.
 *
 * Hovering/focusing the react chip projects the react lens onto the canvas
 * (via DataflowLensContext): stages the loop never runs are ghosted with a
 * "stripped in react" tag. With react-only the lens is pinned — those stages
 * are stripped in EVERY trial. Branch sub-knobs other than max_iter round-trip
 * untouched (edit them in the YAML tab).
 */
import React, { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import type { DataflowRegistry } from "@/api/registry";
import type { NonVisualConfig } from "./types";

/** The react branch's visual identity — the loop. Used only here and by the
 *  canvas lens ring so the accent stays a single deliberate mark. */
export const REACT_LOOP_COLOR = "#4F46E5";

// ── Spec parsing / serialization ─────────────────────────────────────

type DataflowEntry = string | Record<string, Record<string, unknown> | null>;

interface ParsedDataflow {
  /** Branch name → sub-knob object ({} when bare). Insertion-ordered. */
  branches: Map<string, Record<string, unknown>>;
  /** True when the key is absent (implicit sequential-only). */
  implicit: boolean;
}

function parseDataflowSpec(spec: unknown): ParsedDataflow {
  const branches = new Map<string, Record<string, unknown>>();
  if (!Array.isArray(spec)) {
    branches.set("sequential", {});
    return { branches, implicit: true };
  }
  for (const entry of spec as DataflowEntry[]) {
    if (typeof entry === "string") {
      branches.set(entry, {});
    } else if (entry && typeof entry === "object") {
      const keys = Object.keys(entry);
      if (keys.length === 1) {
        branches.set(keys[0], { ...(entry[keys[0]] ?? {}) });
      }
    }
  }
  if (branches.size === 0) {
    branches.set("sequential", {});
    return { branches, implicit: true };
  }
  return { branches, implicit: false };
}

function serializeDataflowSpec(
  branches: Map<string, Record<string, unknown>>,
): DataflowEntry[] | undefined {
  // Canonical minimal form: sequential-only with no sub-knobs → absent key.
  if (
    branches.size === 1 &&
    branches.has("sequential") &&
    Object.keys(branches.get("sequential")!).length === 0
  ) {
    return undefined;
  }
  return [...branches.entries()].map(([name, knobs]) =>
    Object.keys(knobs).length > 0 ? { [name]: knobs } : name,
  );
}

function maxIterValues(knobs: Record<string, unknown>): number[] {
  const v = knobs.max_iter;
  if (Array.isArray(v)) return v.map(Number).filter((n) => Number.isFinite(n));
  if (typeof v === "number") return [v];
  return [];
}

// ── Loop glyph — one wire, three stops, a return arc ─────────────────

/** retrieval → (reranker) → generator with the loop wire back: what a react
 *  round IS. The dashed middle stop is the optional reranker. */
function LoopGlyph({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 26" className={className} fill="none" aria-hidden="true">
      <path d="M10 8 H54" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      {/* return arc: generator back to retrieval */}
      <path
        d="M54 10 C54 21 10 21 10 13"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
      {/* arrowhead re-entering retrieval */}
      <path d="M7.5 15 L10 11.5 L12.5 15" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="10" cy="8" r="2.4" fill="currentColor" />
      <circle cx="32" cy="8" r="2.4" fill="none" stroke="currentColor" strokeWidth="1.2" strokeDasharray="2 1.6" />
      <circle cx="54" cy="8" r="2.4" fill="currentColor" />
    </svg>
  );
}

// ── Component ────────────────────────────────────────────────────────

export interface DataflowBarProps {
  config: NonVisualConfig;
  patch: (fn: (c: any) => void) => void;
  registry: DataflowRegistry;
  /** Reports the effective lens (hover preview, or pinned for react-only). */
  onLensChange: (lens: "react" | null) => void;
}

export function DataflowBar({ config, patch, registry, onLensChange }: DataflowBarProps) {
  const spec = (config.algo_search_space?.pipeline as Record<string, unknown> | undefined)
    ?.rag_dataflow;
  const { branches } = parseDataflowSpec(spec);
  const seqOn = branches.has("sequential");
  const reactOn = branches.has("react");
  const reactKnobs = branches.get("react") ?? {};
  const iters = maxIterValues(reactKnobs);
  const extraReactKnobs = Object.entries(reactKnobs).filter(([k]) => k !== "max_iter");

  const [hoverLens, setHoverLens] = useState(false);
  const [draft, setDraft] = useState("");
  // Re-enabling react restores the sub-knobs it had before it was toggled off.
  const lastReactKnobs = useRef<Record<string, unknown>>({ max_iter: [5] });
  if (reactOn) lastReactKnobs.current = reactKnobs;

  const pinned = reactOn && !seqOn;
  const lens = pinned || hoverLens ? "react" : null;
  useEffect(() => {
    onLensChange(lens);
  }, [lens, onLensChange]);

  const writeBranches = (next: Map<string, Record<string, unknown>>) =>
    patch((c) => {
      const asp = { ...(c.algo_search_space ?? {}) };
      const pl = { ...(asp.pipeline ?? {}) };
      const serialized = serializeDataflowSpec(next);
      if (serialized === undefined) delete pl.rag_dataflow;
      else pl.rag_dataflow = serialized;
      asp.pipeline = pl;
      c.algo_search_space = asp;
    });

  const toggleBranch = (name: string) => {
    const next = new Map(branches);
    if (next.has(name)) {
      if (next.size === 1) return; // at least one branch must stay on
      next.delete(name);
    } else {
      next.set(name, name === "react" ? { ...lastReactKnobs.current } : {});
    }
    writeBranches(next);
  };

  const setIters = (values: number[]) => {
    const next = new Map(branches);
    next.set("react", { ...reactKnobs, max_iter: values });
    writeBranches(next);
  };

  const addIter = () => {
    const n = parseInt(draft, 10);
    setDraft("");
    if (!Number.isFinite(n) || n < 1 || iters.includes(n)) return;
    setIters([...iters, n].sort((a, b) => a - b));
  };

  const swept = seqOn && reactOn;
  const knownBranches = registry.branches.length > 0 ? registry.branches : ["sequential", "react"];

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-border bg-card px-3 py-2">
      <span className="text-[10px] font-semibold uppercase tracking-[1px] text-muted-foreground">
        Dataflow
      </span>

      {knownBranches.map((name) => {
        const on = branches.has(name);
        const isReact = name === "react";
        const lastOn = on && branches.size === 1;
        return (
          <button
            key={name}
            type="button"
            aria-pressed={on}
            onClick={() => toggleBranch(name)}
            onMouseEnter={isReact ? () => setHoverLens(true) : undefined}
            onMouseLeave={isReact ? () => setHoverLens(false) : undefined}
            onFocus={isReact ? () => setHoverLens(true) : undefined}
            onBlur={isReact ? () => setHoverLens(false) : undefined}
            title={
              lastOn
                ? "At least one dataflow must stay on"
                : on
                  ? `Remove ${name} from the search space`
                  : `Add ${name} to the search space`
            }
            className={cn(
              "flex items-center gap-1.5 rounded-md border px-2 py-1 font-mono text-[11px] font-medium outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring",
              on
                ? isReact
                  ? "border-transparent"
                  : "border-primary/40 bg-primary/10 text-primary"
                : "border-border bg-background text-muted-foreground hover:bg-muted",
              lastOn && "cursor-default",
            )}
            style={
              on && isReact
                ? {
                    borderColor: `${REACT_LOOP_COLOR}66`,
                    backgroundColor: `${REACT_LOOP_COLOR}14`,
                    color: REACT_LOOP_COLOR,
                  }
                : undefined
            }
          >
            <span
              className="size-1.5 rounded-full"
              style={{ backgroundColor: on ? "currentColor" : "transparent", boxShadow: on ? "none" : "inset 0 0 0 1px currentColor" }}
              aria-hidden="true"
            />
            {name}
            {isReact && on && <LoopGlyph className="h-4 w-10" />}
          </button>
        );
      })}

      {/* react sub-knob: max_iter (required — the loop's only round cap) */}
      {reactOn && (
        <div
          className="flex items-center gap-1.5"
          onMouseEnter={() => setHoverLens(true)}
          onMouseLeave={() => setHoverLens(false)}
        >
          <span className="font-mono text-[10px] text-muted-foreground">max_iter</span>
          {iters.map((n) => (
            <span
              key={n}
              className="flex items-center gap-0.5 rounded border border-border bg-background px-1.5 py-0.5 font-mono text-[11px] text-foreground"
            >
              {n}
              <button
                type="button"
                aria-label={`Remove max_iter ${n}`}
                title={iters.length === 1 ? "react needs at least one max_iter value" : `Remove ${n}`}
                disabled={iters.length === 1}
                onClick={() => setIters(iters.filter((x) => x !== n))}
                className="cursor-pointer opacity-60 hover:opacity-100 disabled:cursor-default disabled:opacity-25"
              >
                <X size={10} />
              </button>
            </span>
          ))}
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value.replace(/[^0-9]/g, ""))}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                addIter();
              }
            }}
            onBlur={addIter}
            placeholder="+"
            aria-label="Add a max_iter value"
            className="h-6 w-9 rounded border border-dashed border-border bg-transparent px-1 text-center font-mono text-[11px] text-foreground outline-none placeholder:text-muted-foreground/60 focus:border-solid focus:border-ring"
          />
          {iters.length === 0 && (
            <span className="text-[10px] font-medium text-destructive">
              react needs max_iter values
            </span>
          )}
          {extraReactKnobs.map(([k, v]) => (
            <span
              key={k}
              title={`react sub-knob ${k} — edit in the YAML tab`}
              className="rounded bg-muted/60 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
            >
              {k}: {JSON.stringify(v)}
            </span>
          ))}
        </div>
      )}

      <span className="ml-auto text-[10px] text-muted-foreground">
        {lens === "react"
          ? pinned
            ? "react only — dimmed stages are stripped from every trial"
            : "dimmed stages don't run under react"
          : swept
            ? "optimizer picks one per trial"
            : reactOn
              ? "react: retrieval → reranker (optional) → generator, looped"
              : "sequential: one straight pass"}
      </span>
    </div>
  );
}
