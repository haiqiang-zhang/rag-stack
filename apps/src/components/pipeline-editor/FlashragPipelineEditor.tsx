/**
 * FlashragPipelineEditor — edits algo_search_space.pipeline.methods[] for the
 * flashrag backend. A method strip selects one method (mode); the selected
 * method's node_lines are edited in the SAME visual graph used for static_gt
 * (via the shared nodeLinesToFlow / flowToNodeLines helpers), and its
 * method-specific knobs (max_iter, threshold, …) are edited in a side panel.
 *
 * The optimizer's pipeline__mode ChoiceParameter picks one method per trial;
 * here we author the candidate set the sweep chooses from.
 */
import React, { useMemo, useState } from "react";
import { Plus, Trash2, X, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { VisualPipelineEditor } from "./VisualPipelineEditor";
import { nodeLinesToFlow } from "./hooks/useYamlToFlow";
import { flowToNodeLines } from "./hooks/useFlowToYaml";
import type { NodeTypeRegistry } from "./types";
import type { Node, Edge } from "@xyflow/react";

const RESERVED = new Set(["mode", "node_lines"]);

interface MethodBlock { mode?: string; node_lines?: any[]; [k: string]: unknown; }

export interface FlashragPipelineEditorProps {
  config: any;
  patch: (fn: (c: any) => void) => void;
  registry: NodeTypeRegistry;
  vectordbNames: string[];
  pipelineMethods: string[];
}

export function FlashragPipelineEditor({
  config, patch, registry, vectordbNames, pipelineMethods,
}: FlashragPipelineEditorProps) {
  const methods: MethodBlock[] = config?.algo_search_space?.pipeline?.methods ?? [];
  const [active, setActive] = useState(0);
  const idx = Math.min(active, Math.max(0, methods.length - 1));
  const method = methods[idx];

  // Build the graph for the active method's node_lines. Keyed re-mount on
  // method switch keeps each method's graph independent.
  const initial = useMemo(
    () => nodeLinesToFlow(method?.node_lines ?? []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [idx, methods.length],
  );

  const ensurePath = (c: any) => {
    c.algo_search_space ??= {};
    c.algo_search_space.pipeline ??= {};
    c.algo_search_space.pipeline.methods ??= [];
    return c.algo_search_space.pipeline.methods as MethodBlock[];
  };

  const onFlowChange = (nodes: Node[], edges: Edge[]) =>
    patch((c) => {
      const ms = ensurePath(c);
      if (ms[idx]) ms[idx] = { ...ms[idx], node_lines: flowToNodeLines(nodes, edges) };
    });

  const addMethod = (mode: string) =>
    patch((c) => {
      const ms = ensurePath(c);
      ms.push({
        mode,
        node_lines: [
          { node_line_name: "retrieve_node_line", nodes: [
            { stage: "semantic_retrieval", top_k: [4, 8], modules: [{ component: "vectordb", vectordb: vectordbNames[0] ?? "faiss_hnsw_index" }] } ] },
          { node_line_name: "generate_node_line", nodes: [
            { stage: "generator", modules: [{ component: "vllm", model: "Qwen/Qwen2.5-7B-Instruct", temperature: 0.5 }] } ] },
        ],
      });
    });

  const removeMethod = (i: number) =>
    patch((c) => {
      const ms = ensurePath(c);
      ms.splice(i, 1);
    });

  const setKnob = (key: string, raw: string) =>
    patch((c) => {
      const ms = ensurePath(c);
      let value: unknown = raw;
      try { const p = JSON.parse(raw); if (typeof p !== "string") value = p; } catch { /* keep string */ }
      ms[idx] = { ...ms[idx], [key]: value };
    });

  const removeKnob = (key: string) =>
    patch((c) => {
      const ms = ensurePath(c);
      const m = { ...ms[idx] };
      delete m[key];
      ms[idx] = m;
    });

  const addKnob = () =>
    patch((c) => {
      const ms = ensurePath(c);
      ms[idx] = { ...ms[idx], [`knob_${Object.keys(ms[idx]).length}`]: [] };
    });

  const knobs = method ? Object.entries(method).filter(([k]) => !RESERVED.has(k)) : [];
  const unusedModes = pipelineMethods.filter((m) => !methods.some((x) => x.mode === m));

  return (
    <div className="flex h-full flex-col">
      {/* Method strip */}
      <div className="flex items-center gap-1.5 border-b border-border bg-card px-3 py-2">
        <span className="mr-1 text-[10px] font-semibold uppercase tracking-[1px] text-muted-foreground">Methods</span>
        {methods.map((m, i) => {
          const on = i === idx;
          return (
            <div
              key={i}
              className={cn(
                "flex items-center gap-1 rounded-md border px-2 py-1 text-[11px] transition-colors",
                on ? "border-primary/40 bg-primary/10 text-primary" : "border-border bg-background text-muted-foreground hover:bg-muted",
              )}
            >
              <button type="button" onClick={() => setActive(i)} className="cursor-pointer font-medium">
                {m.mode || `method_${i}`}
              </button>
              <button type="button" onClick={() => removeMethod(i)} className="cursor-pointer opacity-60 hover:opacity-100" aria-label="Remove method">
                <X size={11} />
              </button>
            </div>
          );
        })}
        {methods.length === 0 && (
          <span className="text-[11px] italic text-muted-foreground">No methods — add one →</span>
        )}
        <Select value="" onValueChange={addMethod}>
          <SelectTrigger className="ml-1 h-7 w-[150px]">
            <span className="flex items-center gap-1 text-[11px]"><Plus size={11} className="text-primary" /> Add method</span>
          </SelectTrigger>
          <SelectContent>
            {unusedModes.map((m) => <SelectItem key={m} value={m}>{m}</SelectItem>)}
            {unusedModes.length === 0 && <div className="px-2 py-1.5 text-[11px] text-muted-foreground">all methods added</div>}
          </SelectContent>
        </Select>
      </div>

      {/* Active method: graph + knobs */}
      {method ? (
        <div className="flex flex-1 min-h-0">
          <div className="relative flex-1 min-w-0">
            <div className="absolute inset-0">
              <VisualPipelineEditor
                key={idx}
                initialNodes={initial.nodes}
                initialEdges={initial.edges}
                registry={registry}
                vectordbNames={vectordbNames}
                onFlowChange={onFlowChange}
              />
            </div>
          </div>

          {/* Method knobs */}
          <div className="w-[280px] shrink-0 overflow-auto border-l border-border bg-card p-4">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-[1px] text-muted-foreground">
                {method.mode} knobs ({knobs.length})
              </span>
              <Button type="button" variant="outline" size="sm" onClick={addKnob}>
                <Plus size={10} className="text-primary" /> Add
              </Button>
            </div>
            {knobs.length === 0 && (
              <p className="text-[11px] italic text-muted-foreground">
                No method knobs. e.g. ircot → max_iter, flare → threshold / look_ahead_steps.
                Use a JSON list to sweep: [3, 5].
              </p>
            )}
            <div className="space-y-2">
              {knobs.map(([key, value]) => (
                <div key={key} className="flex items-center gap-1.5">
                  <input
                    className="w-[42%] rounded-md border border-border bg-muted px-2 py-1.5 text-[11px] font-medium text-muted-foreground outline-none"
                    value={key}
                    onChange={(e) => {
                      const nk = e.target.value;
                      if (nk === key) return;
                      patch((c) => {
                        const ms = ensurePath(c);
                        const m = { ...ms[idx] } as MethodBlock;
                        const v = m[key]; delete m[key]; m[nk] = v;
                        ms[idx] = m;
                      });
                    }}
                  />
                  <input
                    className="w-[58%] rounded-md border border-border bg-background px-2 py-1.5 font-mono text-[11px] text-foreground outline-none"
                    value={typeof value === "object" ? JSON.stringify(value) : String(value ?? "")}
                    onChange={(e) => setKnob(key, e.target.value)}
                    placeholder="[3, 5]"
                  />
                  <button type="button" onClick={() => removeKnob(key)} className="text-destructive/70 hover:text-destructive" aria-label="Remove knob">
                    <Trash2 size={11} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="flex flex-1 flex-col items-center justify-center gap-2.5">
          <Layers size={36} className="text-muted-foreground/40" />
          <span className="text-sm text-muted-foreground">Add a method to start (sequential, ircot, flare, …)</span>
        </div>
      )}
    </div>
  );
}
