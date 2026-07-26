/**
 * SearchSpacePanel — visualizes the Ax search space a config expands into.
 * Calls the backend (RagSearchSpace.build_ax_searchspace — the SAME builder
 * every optimizer consumes), so what you see is exactly what a run will search.
 * Hierarchical choice params (e.g. flashrag pipeline__mode) nest their per-value
 * subtrees; range params show their bounds. A summary line gives the discrete
 * size + continuous-dim count.
 */
import React, { useEffect, useState } from "react";
import { Loader2, ChevronRight, ChevronDown, Boxes, CircleAlert, Sigma } from "lucide-react";
import { fetchSearchSpaceTree, type SearchSpaceNode, type SearchSpaceTree } from "@/api/costModelClient";

function fmtSize(n: number): string {
  if (!Number.isFinite(n)) return "∞";
  if (n < 1_000_000) return n.toLocaleString();
  return n.toExponential(1).replace("e+", "e");
}

/** A readable leaf label from the full Ax param name. The last segment is the
 *  param "role"; for generic roles (enabled / component) that alone is
 *  ambiguous across stages, so we qualify it with the owning stage
 *  (e.g. `passage_reranker · enabled`). The full name is shown on hover. */
function displayLeaf(name: string): string {
  const parts = name.split("__");
  const last = parts[parts.length - 1];
  if ((last === "enabled" || last === "component") && parts.length >= 2) {
    return `${parts[parts.length - 2]} · ${last}`;
  }
  return last;
}

function ValueChips({ values }: { values: string[] }) {
  return (
    <span className="ml-2 inline-flex flex-wrap gap-1 align-middle">
      {values.map((v, i) => (
        <span key={i} className="rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-[10px] text-foreground/80">
          {v.length > 28 ? v.slice(0, 27) + "…" : v}
        </span>
      ))}
    </span>
  );
}

function SsNode({ node, depth }: { node: SearchSpaceNode; depth: number }) {
  const hasDeps = node.dependents && Object.keys(node.dependents).length > 0;
  const [open, setOpen] = useState(true);

  return (
    <div className="leading-tight">
      <div className="flex items-start gap-1 py-1">
        {hasDeps ? (
          <button type="button" onClick={() => setOpen((o) => !o)} className="mt-0.5 text-muted-foreground hover:text-foreground">
            {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          </button>
        ) : (
          <span className="w-[13px]" />
        )}
        <div className="min-w-0 flex-1">
          <span className="font-mono text-[12px] font-medium text-foreground" title={node.name}>
            {displayLeaf(node.name)}
          </span>
          {node.kind === "range" ? (
            <span className="ml-2 rounded bg-primary/10 px-1.5 py-0.5 font-mono text-[10px] text-primary">
              range [{node.lower} … {node.upper}] {node.param_type}
            </span>
          ) : (
            <>
              <span className="ml-1.5 text-[10px] text-muted-foreground">
                choice·{node.values?.length ?? 0}
              </span>
              {!hasDeps && node.values && <ValueChips values={node.values} />}
            </>
          )}
        </div>
      </div>

      {hasDeps && open && (
        <div className="ml-[6px] border-l border-border pl-3">
          {Object.entries(node.dependents!).map(([value, children]) => (
            <div key={value} className="py-0.5">
              <div className="flex items-center gap-1.5">
                <span className="text-muted-foreground/50">→</span>
                <span className="rounded border border-primary/30 bg-primary/5 px-1.5 py-0.5 font-mono text-[10px] font-medium text-primary" title={node.name}>
                  {displayLeaf(node.name)} = {value}
                </span>
                {children.length === 0 && <span className="text-[10px] italic text-muted-foreground/60">(no extra params)</span>}
              </div>
              {children.length > 0 && (
                <div className="ml-2 border-l border-border/60 pl-3">
                  {children.map((c) => <SsNode key={c.name} node={c} depth={depth + 1} />)}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function SearchSpacePanel({ yamlText }: { yamlText: string }) {
  const [data, setData] = useState<SearchSpaceTree | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const t = setTimeout(() => {
      fetchSearchSpaceTree(yamlText)
        .then((d) => { if (!cancelled) { setData(d); setLoading(false); } })
        .catch((e) => { if (!cancelled) { setData({ error: String(e) }); setLoading(false); } });
    }, 500);
    return () => { cancelled = true; clearTimeout(t); };
  }, [yamlText]);

  const summary = data?.summary;
  const tree = data?.tree ?? [];

  return (
    <div className="flex h-full flex-col">
      {/* Summary bar */}
      <div className="flex items-center gap-3 border-b border-border bg-card px-4 py-2">
        <div className="flex items-center gap-1.5">
          <Boxes size={15} className="text-primary" />
          <span className="text-[12px] font-semibold text-foreground">Search space</span>
        </div>
        {loading && <Loader2 size={13} className="animate-spin text-muted-foreground" />}
        {summary && (
          <>
            <span className="text-[11px] text-muted-foreground">{summary.n_params} params</span>
            <span className="flex items-center gap-1 text-[11px] text-muted-foreground">
              <Sigma size={11} /> ≈ {fmtSize(summary.discrete_size)} configs
            </span>
            {summary.n_range > 0 && (
              <span className="text-[11px] text-muted-foreground">+{summary.n_range} continuous</span>
            )}
          </>
        )}
      </div>

      {/* Body */}
      <div className="flex-1 overflow-auto p-4">
        {data?.error ? (
          <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2">
            <CircleAlert size={14} className="mt-0.5 shrink-0 text-amber-600 dark:text-amber-400" />
            <span className="text-[11px] text-amber-700 dark:text-amber-300">{data.error}</span>
          </div>
        ) : !loading && tree.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
            <Boxes size={32} className="text-muted-foreground/40" />
            <span className="text-sm text-muted-foreground">No search dimensions</span>
            <span className="max-w-sm text-[11px] text-muted-foreground/70">
              Nothing list-valued to sweep (or the config doesn't validate). A list like
              <span className="mx-1 font-mono">top_k: [4, 8, 16]</span> becomes a dimension; a scalar is fixed.
            </span>
          </div>
        ) : (
          <div className="space-y-0.5">
            {tree.map((n) => <SsNode key={n.name} node={n} depth={0} />)}
          </div>
        )}
      </div>
    </div>
  );
}
