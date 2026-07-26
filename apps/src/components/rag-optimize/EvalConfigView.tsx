import React, { useEffect, useState } from "react";
import { ChevronDown, ChevronRight, ArrowDown, Loader2 } from "lucide-react";
import { stageColor } from "@/lib/theme";
import { ConfigViewer } from "@/components/ConfigViewer";
import { Card, CardContent } from "@/components/ui/card";
import { useProject } from "@/context/ProjectContext";
import { createDataClient } from "@/lib/config";
import { ObservedExecutionView } from "./ObservedExecutionView";

// Logical pipeline stages (from pipeline_config node_lines) → identity hue.
// Reuses the canonical per-performance-stage palette so a stage keeps one
// colour across the pipeline view, the stage table, and the charts.
const LOGICAL_HUE: Record<string, string> = {
  query_expansion: stageColor("query_expansion_prefill"),
  semantic_retrieval: stageColor("semantic_retrieval_vectorsearch"),
  lexical_retrieval: stageColor("lexical_retrieval"),
  hybrid_retrieval: stageColor("retrieval_fusion"),
  passage_reranker: stageColor("passage_reranker"),
  passage_compressor: stageColor("passage_compressor"),
  prompt_maker: "#64748b",
  passage_filter: "#64748b",
  generator: stageColor("generator_prefill"),
};
const stageHue = (s: string) => LOGICAL_HUE[s] ?? stageColor(s);

const STAGE_LABEL: Record<string, string> = {
  query_expansion: "Query expansion",
  semantic_retrieval: "Dense retrieval",
  lexical_retrieval: "BM25 retrieval",
  hybrid_retrieval: "Hybrid fusion",
  prompt_maker: "Prompt",
  passage_reranker: "Rerank",
  passage_compressor: "Compress",
  passage_filter: "Filter",
  generator: "Generator",
};
const stageLabel = (s: string) => STAGE_LABEL[s] ?? s.replace(/_/g, " ");

// Knobs worth colour-highlighting inside a stage card.
const HILITE = new Set([
  "model",
  "embedding_model",
  "vectordb",
  "top_k",
  "nprobe",
  "temperature",
  "rate",
  "bits",
]);
const HIDE_PARAM = new Set(["component", "prompt"]);

function shortVal(v: unknown): string {
  if (Array.isArray(v)) return v.map(shortVal).join(", ");
  const s = String(v);
  return s.includes("/") ? s.split("/").pop()! : s;
}
function baseName(p: unknown): string {
  const s = String(p ?? "");
  return s.split("/").pop() || s;
}

interface StageNode {
  stage: string;
  component: string;
  params: [string, unknown][];
  topK?: number | null;
}

function parsePipeline(cfg: Record<string, unknown> | null): StageNode[] {
  if (!cfg) return [];
  const lines = (cfg.node_lines as Array<Record<string, unknown>>) || [];
  const out: StageNode[] = [];
  for (const nl of lines) {
    for (const node of (nl.nodes as Array<Record<string, unknown>>) || []) {
      const modules = (node.modules as Array<Record<string, unknown>>) || [];
      for (const m of modules) {
        const params = Object.entries(m).filter(
          ([k, v]) => !HIDE_PARAM.has(k) && v != null && v !== "",
        ) as [string, unknown][];
        out.push({
          stage: node.stage as string,
          component: (m.component as string) || "—",
          params,
          topK: (node.top_k as number | null) ?? null,
        });
      }
    }
  }
  return out;
}

// ── Full-config tree: every remaining section, in a friendly order ────────
const SECTION_TITLE: Record<string, string> = {
  system: "System & hardware",
  global: "Run mode",
  dataset: "Dataset",
  pipeline_runtime: "Pipeline runtime",
  corpus_runtime: "Corpus",
  eval_backend_setting: "Evaluation backend",
  vectordb: "Vector DB",
  _token_stats: "Token statistics",
  _runtime: "Runtime",
  algo_search_space: "Search space (every tunable knob)",
};
const SECTION_ORDER = [
  "system",
  "global",
  "pipeline_runtime",
  "corpus_runtime",
  "eval_backend_setting",
  "vectordb",
  "_token_stats",
  "_runtime",
  "algo_search_space",
];
// Big / reference sections start collapsed; everything else is open so the
// knobs are visible without a click.
const DEFAULT_CLOSED = new Set(["algo_search_space", "_token_stats", "vectordb"]);
// node_lines is rendered as the visual pipeline; dataset is shown inline there.
const SKIP_SECTIONS = new Set(["node_lines", "dataset"]);

interface EvalConfigViewProps {
  /** Which eval to show. The component fetches its own pipeline_config unless
   *  `resolved` is passed in. */
  evalId?: number | null;
  /** Pre-fetched pipeline_config.json — pass to skip the internal fetch.
   *  Omit entirely to let the component load it from `evalId`. */
  resolved?: Record<string, unknown> | null;
}

/**
 * One reusable, self-contained view of an evaluation's resolved configuration.
 * Drop it in anywhere with just an `evalId`:
 *   <EvalConfigView evalId={id} />
 * It fetches the pipeline_config itself (or accepts a pre-fetched `resolved`).
 * The searched knobs are NOT rendered here — that's SearchPointCard's job.
 */
export function EvalConfigView({
  evalId,
  resolved: resolvedProp,
}: EvalConfigViewProps) {
  const { projectBaseUrl } = useProject();
  const controlled = resolvedProp !== undefined;
  const [fetched, setFetched] = useState<Record<string, unknown> | null>(null);
  const [fetching, setFetching] = useState(false);

  useEffect(() => {
    if (controlled) return;
    if (evalId == null || !projectBaseUrl) {
      setFetched(null);
      return;
    }
    let alive = true;
    setFetching(true);
    createDataClient(projectBaseUrl)
      .getPipelineConfig(evalId)
      .then((c) => alive && setFetched(c))
      .catch(() => alive && setFetched(null))
      .finally(() => alive && setFetching(false));
    return () => {
      alive = false;
    };
  }, [controlled, evalId, projectBaseUrl]);

  const resolved = controlled ? resolvedProp! : fetched;
  const loading = controlled ? false : fetching;
  const stages = parsePipeline(resolved);
  const dataset = (resolved?.dataset as Record<string, unknown>) || null;

  const sections = resolved
    ? SECTION_ORDER.filter((k) => resolved[k] != null)
        .concat(
          Object.keys(resolved).filter(
            (k) => !SECTION_ORDER.includes(k) && !SKIP_SECTIONS.has(k),
          ),
        )
        .filter((k) => !SKIP_SECTIONS.has(k))
    : [];

  return (
    <div className="flex flex-col gap-4">
      {/* ── Resolved pipeline ─────────────────────────────────────── */}
      <Card>
        <CardContent>
          <Header
            title="Resolved pipeline"
            hint={evalId != null ? `Eval #${evalId}` : undefined}
            count={stages.length ? `${stages.length} stages` : undefined}
          />

          {loading ? (
            <div className="flex items-center gap-2 py-8 text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              <span className="text-xs">Loading resolved config…</span>
            </div>
          ) : stages.length === 0 ? (
            <p className="py-6 text-center text-xs text-muted-foreground">
              No resolved pipeline for this eval.
            </p>
          ) : (
            <div className="mt-1 flex flex-col">
              {stages.map((s, i) => (
                <React.Fragment key={`${s.stage}-${i}`}>
                  <StageCard node={s} />
                  {i < stages.length - 1 && (
                    <div className="flex justify-center py-1">
                      <ArrowDown className="size-3.5 text-muted-foreground/50" />
                    </div>
                  )}
                </React.Fragment>
              ))}
            </div>
          )}

          {dataset && (
            <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-1.5 border-t border-border pt-3">
              <span className="text-[10px] font-semibold uppercase tracking-[0.6px] text-muted-foreground">
                Dataset
              </span>
              <Fact label="name" value={String(dataset.dataset_name ?? "—")} />
              {dataset.qa != null && (
                <Fact label="qa" value={baseName(dataset.qa)} />
              )}
              {dataset.corpus != null && (
                <Fact label="corpus" value={baseName(dataset.corpus)} />
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {evalId != null && <ObservedExecutionView evalId={evalId} />}

      {/* ── Full resolved config — every remaining section ────────── */}
      {sections.length > 0 && resolved && (
        <Card>
          <CardContent>
            <Header
              title="Full configuration"
              hint="everything this eval resolved to"
              count={`${sections.length} sections`}
            />
            <div className="flex flex-col gap-2">
              {sections.map((key) => (
                <Section
                  key={key}
                  name={key}
                  value={resolved[key]}
                  defaultOpen={!DEFAULT_CLOSED.has(key)}
                />
              ))}
            </div>
            <RawToggle
              config={resolved}
              title={`Resolved config (Eval #${evalId})`}
            />
          </CardContent>
        </Card>
      )}

    </div>
  );
}

function Header({
  title,
  hint,
  count,
}: {
  title: string;
  hint?: string;
  count?: string;
}) {
  return (
    <div className="mb-3 flex items-baseline justify-between gap-3">
      <div className="flex items-baseline gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[0.5px] text-foreground">
          {title}
        </span>
        {hint && (
          <span className="text-[11px] text-muted-foreground">{hint}</span>
        )}
      </div>
      {count && (
        <span className="font-mono text-[10px] text-muted-foreground">
          {count}
        </span>
      )}
    </div>
  );
}

function StageCard({ node }: { node: StageNode }) {
  const hue = stageHue(node.stage);
  const chips: [string, unknown, boolean][] = [];
  if (node.topK != null) chips.push(["top_k", node.topK, true]);
  for (const [k, v] of node.params) chips.push([k, v, HILITE.has(k)]);

  return (
    <div className="rounded-md border bg-muted/30 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span
          className="size-2.5 shrink-0 rounded-full"
          style={{ backgroundColor: hue }}
        />
        <span className="text-[13px] font-semibold" style={{ color: hue }}>
          {stageLabel(node.stage)}
        </span>
        <span className="rounded bg-background px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
          {node.component}
        </span>
      </div>
      {chips.length > 0 && (
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {chips.map(([k, v, hi]) => (
            <ParamChip key={k} k={k} v={v} highlight={hi} hue={hue} />
          ))}
        </div>
      )}
    </div>
  );
}

function ParamChip({
  k,
  v,
  highlight,
  hue,
}: {
  k: string;
  v: unknown;
  highlight?: boolean;
  hue: string;
}) {
  const full = String(v);
  return (
    <span
      title={`${k}: ${full}`}
      className="inline-flex max-w-full items-baseline gap-1.5 rounded border bg-background px-2 py-0.5 font-mono text-[11px]"
    >
      <span className="text-muted-foreground">{k}</span>
      <span
        className="truncate font-medium"
        style={{ color: highlight ? hue : "var(--foreground)" }}
      >
        {shortVal(v)}
      </span>
    </span>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-baseline gap-1.5 font-mono text-[11px]">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium text-foreground">{value}</span>
    </span>
  );
}

// ── Generic recursive config tree ─────────────────────────────────────────
function isLeaf(v: unknown): boolean {
  if (v == null || typeof v !== "object") return true;
  if (Array.isArray(v))
    return v.every((x) => x == null || typeof x !== "object");
  return false;
}
function fmtLeaf(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (Array.isArray(v)) return v.map((x) => String(x)).join(", ");
  return String(v);
}

function LeafRow({ label, value }: { label: string; value: unknown }) {
  const full = fmtLeaf(value);
  return (
    <div className="flex items-baseline justify-between gap-3 py-[3px]">
      <span className="shrink-0 font-mono text-[11px] text-muted-foreground">
        {label}
      </span>
      <span
        title={full}
        className="max-w-[62%] truncate text-right font-mono text-[11px] font-medium text-foreground"
      >
        {full || "—"}
      </span>
    </div>
  );
}

function TreeNode({ label, value }: { label: string; value: unknown }) {
  if (isLeaf(value)) return <LeafRow label={label} value={value} />;
  const entries: [string, unknown][] = Array.isArray(value)
    ? value.map((v, i) => [`#${i}`, v] as [string, unknown])
    : Object.entries(value as Record<string, unknown>);
  return (
    <div className="mt-1.5 first:mt-0">
      <div className="mb-1 font-mono text-[10px] font-semibold uppercase tracking-[0.4px] text-muted-foreground/80">
        {label.replace(/_/g, " ")}
      </div>
      <div className="ml-1 flex flex-col border-l border-border pl-3">
        {entries.map(([k, v]) => (
          <TreeNode key={k} label={k} value={v} />
        ))}
      </div>
    </div>
  );
}

function Section({
  name,
  value,
  defaultOpen,
}: {
  name: string;
  value: unknown;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const count = isLeaf(value)
    ? null
    : Array.isArray(value)
      ? value.length
      : Object.keys(value as object).length;
  const entries: [string, unknown][] = isLeaf(value)
    ? []
    : Array.isArray(value)
      ? value.map((v, i) => [`#${i}`, v] as [string, unknown])
      : Object.entries(value as Record<string, unknown>);

  return (
    <div className="overflow-hidden rounded-md border bg-muted/20">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 hover:bg-muted/40"
      >
        <span className="flex items-center gap-1.5">
          {open ? (
            <ChevronDown className="size-3.5 text-muted-foreground" />
          ) : (
            <ChevronRight className="size-3.5 text-muted-foreground" />
          )}
          <span className="text-[11px] font-semibold text-foreground">
            {SECTION_TITLE[name] ?? name.replace(/_/g, " ")}
          </span>
        </span>
        {count != null && (
          <span className="font-mono text-[10px] text-muted-foreground">
            {count} {count === 1 ? "key" : "keys"}
          </span>
        )}
      </button>
      {open && (
        <div className="border-t border-border px-3 py-2">
          {isLeaf(value) ? (
            <LeafRow label={name} value={value} />
          ) : (
            entries.map(([k, v]) => <TreeNode key={k} label={k} value={v} />)
          )}
        </div>
      )}
    </div>
  );
}

function RawToggle({
  config,
  title,
}: {
  config: Record<string, unknown>;
  title: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-3 border-t border-border pt-2">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 py-1 text-muted-foreground hover:text-foreground"
      >
        {open ? (
          <ChevronDown className="size-3.5" />
        ) : (
          <ChevronRight className="size-3.5" />
        )}
        <span className="font-mono text-[11px]">
          {open ? "Hide" : "Show"} raw YAML
        </span>
      </button>
      {open && (
        <div className="mt-1">
          <ConfigViewer config={config} title={title} />
        </div>
      )}
    </div>
  );
}
