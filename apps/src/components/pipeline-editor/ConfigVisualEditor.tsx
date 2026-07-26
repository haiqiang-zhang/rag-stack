/**
 * ConfigVisualEditor — Reusable single-config editor with 3 tabs:
 *   Pipeline (xyflow DAG), VectorDB (list editor), YAML (text)
 *
 * Used by:
 *   - PipelineEditorContainer (template configs)
 *   - RAG Optimize project Run tab (project config)
 */
import React, { useState, useCallback, useMemo, useRef, useEffect } from "react";
import {
  GitBranch, Code, Database, Layers, Save, Plus, Trash2, X, Loader2,
  Settings2, Cpu, Target, CircleCheck, CircleAlert, Boxes,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { IconCircle } from "@/components/ui/IconCircle";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SidePanel, SidePanelHeader } from "@/components/ui/SidePanel";
import { SideItem } from "@/components/ui/SideItem";
import { Badge, BadgeText } from "@/components/ui/LegacyBadge";
import { cn } from "@/lib/utils";
import yaml from "js-yaml";

import { Separator } from "@/components/ui/separator";
import { ConfigEditor } from "@/components/ConfigEditor";
import { VisualPipelineEditor } from "./VisualPipelineEditor";
import { PipelinePlanView } from "./PipelinePlanView";
import { DataflowBar } from "./DataflowBar";
import { DataflowLensContext } from "./DataflowLensContext";
import { FlashragPipelineEditor } from "./FlashragPipelineEditor";
import { SearchSpacePanel } from "./SearchSpacePanel";
import { CorpusTab } from "./CorpusTab";
import { SetupTab } from "./panels/SetupTab";
import { SystemTab } from "./panels/SystemTab";
import { QualityTab } from "./panels/QualityTab";
import { useRegistry } from "./hooks/useRegistry";
import { useConfigRegistries } from "./hooks/useConfigRegistries";
import { yamlObjToFlow } from "./hooks/useYamlToFlow";
import { flowToConfig } from "./hooks/useFlowToYaml";
import { validateConfig } from "@/api/costModelClient";
import type { Node, Edge } from "@xyflow/react";
import type { NonVisualConfig, CorpusConfig } from "./types";

type EditorMode = "setup" | "pipeline" | "corpus" | "vectordb" | "system" | "quality" | "searchspace" | "yaml";

export interface ConfigEditorState {
  dirty: boolean;
  saving: boolean;
  valid: boolean | null;
}

export interface ConfigVisualEditorProps {
  /** Raw YAML text (with comments preserved) */
  initialYaml: string;
  /** Parsed config object */
  initialContent: Record<string, unknown>;
  /** Called when user clicks Save. Receives the YAML text to persist. */
  onSave: (yamlText: string) => Promise<void>;
  /** Optional extra toolbar content (e.g. "Create Project" button) */
  toolbarExtra?: React.ReactNode;
  /** Allows a project to launch only the saved, validated configuration. */
  onStateChange?: (state: ConfigEditorState) => void;
  /** Projects select QA/corpus in the launch controls instead of editing paths here. */
  datasetInputsExternal?: boolean;
}

export function ConfigVisualEditor({
  initialYaml,
  initialContent,
  onSave,
  toolbarExtra,
  onStateChange,
  datasetInputsExternal = false,
}: ConfigVisualEditorProps) {
  const { nodeTypes: registry, vectorDbTypes, loading: registryLoading, error: registryError } = useRegistry();
  const configRegistries = useConfigRegistries();

  const [editorMode, setEditorMode] = useState<EditorMode>("setup");
  const [validation, setValidation] = useState<{ valid: boolean; errors: string[] } | null>(null);
  const [yamlText, setYamlText] = useState(initialYaml);
  const [isDirty, setIsDirty] = useState(false);
  const [saveOk, setSaveOk] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    onStateChange?.({ dirty: isDirty, saving, valid: validation?.valid ?? null });
  }, [isDirty, saving, validation, onStateChange]);

  // Flow state
  const [flowNodes, setFlowNodes] = useState<Node[]>(() => {
    try { return yamlObjToFlow(initialContent).nodes; } catch { return []; }
  });
  const [flowEdges, setFlowEdges] = useState<Edge[]>(() => {
    try { return yamlObjToFlow(initialContent).edges; } catch { return []; }
  });
  const nonVisualRef = useRef<NonVisualConfig>({});

  // Dirty check baselines
  const savedYamlRef = useRef<string>(initialYaml);
  const savedRawYamlRef = useRef(initialYaml);

  // Initialize refs properly (lazy init)
  const didInit = useRef(false);
  if (!didInit.current) {
    didInit.current = true;
    try {
      const { nodes, edges, nonVisual } = yamlObjToFlow(initialContent);
      // Already set via useState initializer, but set refs
      nonVisualRef.current = nonVisual;
      savedYamlRef.current = yaml.dump(flowToConfig(nodes, nonVisual, edges), { lineWidth: 120, noRefs: true });
    } catch {
      nonVisualRef.current = {};
      savedYamlRef.current = initialYaml;
    }
  }

  // Sync yamlText with the visual flow ONLY if the dumped config is
  // semantically different from the canonical baseline (the dump of the
  // originally-parsed config). React-flow fires onNodesChange for non-
  // semantic events like dimension measurement on mount; if we
  // unconditionally overwrote yamlText we'd stomp the raw initial YAML
  // (which still carries the user's comments) with a comment-stripped
  // dump. js-yaml has no comment support, so the only way to keep
  // comments visible in the YAML tab is to leave yamlText untouched
  // when no real edit has occurred.
  const syncYamlFromFlow = useCallback(
    (nodes: Node[], edges: Edge[]) => {
      try {
        const config = flowToConfig(nodes, nonVisualRef.current, edges);
        const currentYaml = yaml.dump(config, { lineWidth: 120, noRefs: true });
        const baseline = savedYamlRef.current;
        if (currentYaml === baseline) {
          // No semantic change vs. the originally-loaded config — keep
          // whatever yamlText currently shows (likely the raw text with
          // comments). Just clear dirty.
          setIsDirty(false);
          return;
        }
        // Real edit. Adopt the dumped form (comments are lost from this
        // point — fundamental js-yaml limitation; preserving requires
        // switching to eemeli/yaml's Document API).
        setYamlText(currentYaml);
        setIsDirty(true);
        setSaveOk(false);
      } catch {
        setIsDirty(true);
      }
    },
    [],
  );

  // Flow changes → sync yamlText. Explicit order indices define execution;
  // moving nodes or groups on the canvas leaves the config unchanged.
  const handleFlowChange = useCallback(
    (nodes: Node[], edges: Edge[]) => {
      setFlowNodes(nodes);
      setFlowEdges(edges);
      syncYamlFromFlow(nodes, edges);
    },
    [syncYamlFromFlow],
  );

  // VectorDB changes → sync yamlText. The vectordb section is ALGO search
  // space: it lives at algo_search_space.vectordb (same home as corpus).
  const getVectordbList = useCallback((): any[] => {
    return (nonVisualRef.current.algo_search_space as any)?.vectordb || [];
  }, []);

  const setVectordbList = useCallback(
    (vdbs: any[]) => {
      const asp = (nonVisualRef.current.algo_search_space ??= {});
      (asp as any).vectordb = vdbs;
      syncYamlFromFlow(flowNodes, flowEdges);
    },
    [flowNodes, flowEdges, syncYamlFromFlow],
  );

  // Corpus block changes → sync yamlText. 7-section: the chunker sweep lives
  // at algo_search_space.corpus (not a top-level `corpus` key).
  const getCorpus = useCallback((): CorpusConfig | undefined => {
    return nonVisualRef.current.algo_search_space?.corpus;
  }, []);

  const setCorpus = useCallback(
    (next: CorpusConfig | undefined) => {
      const asp = (nonVisualRef.current.algo_search_space ??= {});
      if (next === undefined) delete asp.corpus;
      else asp.corpus = next;
      syncYamlFromFlow(flowNodes, flowEdges);
    },
    [flowNodes, flowEdges, syncYamlFromFlow],
  );

  // Generic section patch for the Setup/System/Quality forms: mutate the live
  // config base (nonVisualRef) and re-sync the YAML (which also re-renders, so
  // the form reads the fresh value back).
  const patchConfig = useCallback(
    (fn: (c: any) => void) => {
      fn(nonVisualRef.current);
      syncYamlFromFlow(flowNodes, flowEdges);
    },
    [flowNodes, flowEdges, syncYamlFromFlow],
  );

  // Live validation — debounce-call the canonical server-side ConfigValidator.
  useEffect(() => {
    let cancelled = false;
    setValidation(null);
    const t = setTimeout(() => {
      validateConfig(yamlText)
        .then((v) => { if (!cancelled) setValidation(v); })
        .catch(() => { if (!cancelled) setValidation(null); });
    }, 600);
    return () => { cancelled = true; clearTimeout(t); };
  }, [yamlText]);

  // Mode switch
  const switchMode = useCallback(
    (mode: EditorMode) => {
      if (mode === editorMode) return;
      if (editorMode === "yaml" && mode !== "yaml") {
        try {
          const config = yaml.load(yamlText) as Record<string, unknown>;
          const { nodes, edges, nonVisual } = yamlObjToFlow(config);
          setFlowNodes(nodes);
          setFlowEdges(edges);
          nonVisualRef.current = nonVisual;
        } catch {
          setError("Failed to parse YAML");
          return;
        }
      }
      setEditorMode(mode);
    },
    [editorMode, yamlText]
  );

  // Save
  const handleSave = useCallback(async () => {
    setSaving(true);
    setError(null);
    try {
      yaml.load(yamlText); // validate
      await onSave(yamlText);
      savedYamlRef.current = yamlText;
      savedRawYamlRef.current = yamlText;
      setIsDirty(false);
      setSaveOk(true);
      setTimeout(() => setSaveOk(false), 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }, [yamlText, onSave]);

  const vectordbNames = getVectordbList().map((v: any) => v.name).filter(Boolean);

  // rag_dataflow lens — the DataflowBar reports hover/pinned react preview;
  // PipelineNodeComponent reads it through context to ghost stripped stages.
  const [dataflowLens, setDataflowLens] = useState<"react" | null>(null);
  const dataflowLensValue = useMemo(
    () => ({
      lens: dataflowLens,
      reactActiveStages: new Set(configRegistries.dataflows.react_active_stages),
    }),
    [dataflowLens, configRegistries.dataflows.react_active_stages],
  );

  if (registryLoading) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Loader2 className="size-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col min-h-0">
      {/* Toolbar */}
      <div className="flex min-h-[40px] items-center justify-between border-b border-border bg-card px-3.5 py-1.5">
        <div className="flex flex-1 items-center gap-1.5">
          <div className="flex gap-0.5 rounded-lg bg-muted p-0.5">
            {([
              { key: "setup" as const, label: "Setup", Icon: Settings2 },
              { key: "pipeline" as const, label: "Pipeline", Icon: GitBranch },
              { key: "corpus" as const, label: "Corpus", Icon: Layers },
              { key: "vectordb" as const, label: "VectorDB", Icon: Database },
              { key: "system" as const, label: "System", Icon: Cpu },
              { key: "quality" as const, label: "Quality", Icon: Target },
              { key: "searchspace" as const, label: "Search Space", Icon: Boxes },
              { key: "yaml" as const, label: "YAML", Icon: Code },
            ]).map(({ key, label, Icon }) => {
              const active = editorMode === key;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => switchMode(key)}
                  className={cn(
                    "flex cursor-pointer select-none items-center gap-1.5 rounded-md px-2.5 py-1.5 transition-colors",
                    active ? "bg-primary/10" : "bg-transparent hover:bg-muted",
                  )}
                >
                  <Icon size={11} className={active ? "text-primary" : "text-muted-foreground"} />
                  <span
                    className={cn(
                      "text-[11px]",
                      active ? "font-semibold text-primary" : "font-normal text-muted-foreground",
                    )}
                  >
                    {label}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          {validation && (
            validation.valid ? (
              <span className="flex items-center gap-1 text-[11px] text-emerald-700 dark:text-emerald-300">
                <CircleCheck size={11} /> valid
              </span>
            ) : (
              <span className="flex items-center gap-1 text-[11px] text-amber-700 dark:text-amber-400">
                <CircleAlert size={11} /> {validation.errors.length} issue{validation.errors.length === 1 ? "" : "s"}
              </span>
            )
          )}
          {saveOk && (
            <span className="text-[11px] text-emerald-700 dark:text-emerald-300">Saved</span>
          )}
          {isDirty && <span className="text-[11px] text-primary">Unsaved</span>}
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={handleSave}
            disabled={!isDirty || saving}
          >
            {saving ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <>
                <Save size={12} className="text-primary" />
                Save
              </>
            )}
          </Button>
          {toolbarExtra}
        </div>
      </div>

      {/* Error */}
      {registryError && <div role="alert" className="border-b border-border bg-amber-500/10 px-3.5 py-2 text-xs text-amber-700">
        Component registry unavailable: {registryError}. Existing configuration is preserved; available components and capabilities are unknown.
      </div>}
      {error && (
        <div className="flex items-center justify-between bg-destructive/10 px-3.5 py-1.5">
          <span className="text-[11px] text-destructive">{error}</span>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={() => setError(null)}
            className="size-5 hover:bg-destructive/15"
          >
            <X size={12} className="text-destructive" />
          </Button>
        </div>
      )}

      {/* Schema validation banner (server-side ConfigValidator) */}
      {validation && !validation.valid && editorMode !== "yaml" && (
        <div className="border-b border-amber-500/20 bg-amber-500/10 px-3.5 py-1.5">
          <ul className="max-h-24 list-disc overflow-auto pl-5">
            {validation.errors.slice(0, 8).map((e, i) => (
              <li key={i} className="text-[10px] leading-snug text-amber-700/90 dark:text-amber-300/80">{e}</li>
            ))}
            {validation.errors.length > 8 && (
              <li className="text-[10px] text-amber-700/70">+{validation.errors.length - 8} more…</li>
            )}
          </ul>
        </div>
      )}

      {/* Editor content — `relative` provides a positioning context so each
          tab's child can use `absolute inset-0` to pin to this box's
          geometry. Bypasses the flex-chain height delivery (which proved
          unreliable for CorpusTab / VectorDbTab even with min-h-0 in place). */}
      <div className="relative flex flex-1 flex-col min-h-0">
        {editorMode === "setup" ? (
          <div className="absolute inset-0 overflow-auto">
            <SetupTab config={nonVisualRef.current} patch={patchConfig} registries={configRegistries} datasetInputsExternal={datasetInputsExternal} />
          </div>
        ) : editorMode === "system" ? (
          <div className="absolute inset-0 overflow-auto">
            <SystemTab config={nonVisualRef.current} patch={patchConfig} registries={configRegistries} />
          </div>
        ) : editorMode === "quality" ? (
          <div className="absolute inset-0 overflow-auto">
            <QualityTab config={nonVisualRef.current} patch={patchConfig} registries={configRegistries} />
          </div>
        ) : editorMode === "searchspace" ? (
          <div className="absolute inset-0">
            <SearchSpacePanel yamlText={yamlText} />
          </div>
        ) : editorMode === "pipeline" ? (
          // flashrag → methods[] editor (per-method graph + knobs); static_gt →
          // the single node_lines graph. Both reuse the same VisualPipelineEditor.
          (nonVisualRef.current.global as any)?.eval_backend === "flashrag" ? (
            <div className="absolute inset-0">
              <FlashragPipelineEditor
                config={nonVisualRef.current}
                patch={patchConfig}
                registry={registry}
                vectordbNames={vectordbNames}
                pipelineMethods={configRegistries.pipelineMethods}
              />
            </div>
          ) : (
            // VisualPipelineEditor uses `width: 100%; height: 100%` inline —
            // the inner `absolute inset-0` gives those percentages a definite
            // reference below the dataflow bar.
            <div className="absolute inset-0 flex flex-col">
              <DataflowBar
                config={nonVisualRef.current}
                patch={patchConfig}
                registry={configRegistries.dataflows}
                onLensChange={setDataflowLens}
              />
              <PipelinePlanView yamlText={yamlText} />
              <div className="relative min-h-0 flex-1">
                <div className="absolute inset-0">
                  <DataflowLensContext.Provider value={dataflowLensValue}>
                    <VisualPipelineEditor
                      initialNodes={flowNodes}
                      initialEdges={flowEdges}
                      registry={registry}
                      vectordbNames={vectordbNames}
                      onFlowChange={handleFlowChange}
                    />
                  </DataflowLensContext.Provider>
                </div>
              </div>
            </div>
          )
        ) : editorMode === "corpus" ? (
          <div className="absolute inset-0 overflow-auto">
            <CorpusTab corpus={getCorpus()} onChange={setCorpus} />
          </div>
        ) : editorMode === "vectordb" ? (
          <div className="absolute inset-0 flex">
            <VectorDbTab
              vectordbList={getVectordbList()}
              onChange={setVectordbList}
              supportedTypes={vectorDbTypes}
            />
          </div>
        ) : (
          <div className="absolute inset-0">
            <ConfigEditor
              value={yamlText}
              onChangeText={(t) => {
                setYamlText(t);
                setIsDirty(t !== savedRawYamlRef.current);
                setSaveOk(false);
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

// ── VectorDB Tab (shared) ─────────────────────────────────────────────

interface VectorDbTabProps {
  vectordbList: any[];
  onChange: (list: any[]) => void;
  supportedTypes: string[];
}

function VectorDbTab({ vectordbList, onChange, supportedTypes }: VectorDbTabProps) {
  const [selectedIdx, setSelectedIdx] = useState<number | null>(
    vectordbList.length > 0 ? 0 : null
  );

  const handleAdd = () => {
    const newVdb = { name: `new_vectordb_${vectordbList.length}`, db_type: supportedTypes[0] || "chroma" };
    const updated = [...vectordbList, newVdb];
    onChange(updated);
    setSelectedIdx(updated.length - 1);
  };

  const handleDelete = (idx: number) => {
    const updated = vectordbList.filter((_, i) => i !== idx);
    onChange(updated);
    setSelectedIdx(updated.length > 0 ? Math.min(idx, updated.length - 1) : null);
  };

  const handleUpdate = (idx: number, vdb: any) => {
    onChange(vectordbList.map((v, i) => (i === idx ? vdb : v)));
  };

  const handleFieldChange = (idx: number, key: string, raw: string) => {
    const vdb = { ...vectordbList[idx] };
    let value: unknown = raw;
    try {
      const parsed = JSON.parse(raw);
      if (typeof parsed !== "string") value = parsed;
    } catch { /* keep string */ }
    vdb[key] = value;
    handleUpdate(idx, vdb);
  };

  const handleAddParam = (idx: number) => {
    const vdb = { ...vectordbList[idx], [`param_${Object.keys(vectordbList[idx]).length}`]: "" };
    handleUpdate(idx, vdb);
  };

  const handleRemoveParam = (idx: number, key: string) => {
    const vdb = { ...vectordbList[idx] };
    delete vdb[key];
    handleUpdate(idx, vdb);
  };

  const selected = selectedIdx !== null ? vectordbList[selectedIdx] : null;
  const reservedKeys = new Set(["name", "db_type"]);
  const params = selected ? Object.entries(selected).filter(([k]) => !reservedKeys.has(k)) : [];

  return (
    // `min-h-0` on the row so the two columns inside (SidePanel + detail)
    // can each engage their own `overflow-auto`. Without it, this flex child
    // gets sized to content and neither child sees a bounded height.
    <div className="flex flex-1 min-h-0">
      {/* Left sidebar */}
      <SidePanel>
        <SidePanelHeader>
          <span className="text-[10px] font-semibold uppercase tracking-[1px] text-muted-foreground">
            VECTORDB ({vectordbList.length})
          </span>
          <Button type="button" variant="ghost" size="icon" onClick={handleAdd} className="size-6">
            <Plus size={12} className="text-primary" />
          </Button>
        </SidePanelHeader>
        <div className="flex-1 overflow-auto">
          {vectordbList.map((vdb, i) => {
            const isActive = selectedIdx === i;
            return (
              <SideItem
                key={`${vdb.name}-${i}`}
                active={isActive}
                onClick={() => setSelectedIdx(i)}
                className="gap-2.5 py-2.5"
              >
                <IconCircle size="sm" variant={isActive ? "accentSolid" : "muted"}>
                  <Database size={13} className={isActive ? "text-primary-foreground" : "text-muted-foreground"} />
                </IconCircle>
                <div className="flex flex-1 flex-col">
                  <span
                    className={cn(
                      "truncate text-xs",
                      isActive ? "font-semibold text-primary" : "font-normal text-foreground",
                    )}
                  >
                    {vdb.name || `vectordb_${i}`}
                  </span>
                  <span className="text-[10px] text-muted-foreground">{vdb.db_type || "—"}</span>
                </div>
              </SideItem>
            );
          })}
          {vectordbList.length === 0 && (
            <div className="flex flex-col items-center gap-2.5 pt-[60px]">
              <Database size={28} className="text-muted-foreground/60" />
              <span className="text-[13px] text-muted-foreground">No vector databases</span>
              <span className="text-[11px] text-muted-foreground/70">Click + to add one</span>
            </div>
          )}
        </div>
      </SidePanel>

      {/* Right: detail */}
      {selected ? (
        <div className="flex-1 min-h-0 overflow-auto p-7 pb-12">
          <div className="mb-6 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <IconCircle size="md" variant="accent">
                <Database size={16} className="text-primary" />
              </IconCircle>
              <div className="flex flex-col">
                <span className="text-lg font-semibold tracking-tight text-foreground">
                  {selected.name || "Untitled"}
                </span>
                <Badge variant="accent" className="mt-1 self-start">
                  <BadgeText variant="accent">{selected.db_type}</BadgeText>
                </Badge>
              </div>
            </div>
            <Button type="button" variant="ghost" size="sm" onClick={() => handleDelete(selectedIdx!)}>
              <Trash2 size={12} className="text-destructive" />
              <span className="select-none text-[11px] font-medium text-destructive">Delete</span>
            </Button>
          </div>

          <section className="space-y-3">
            <span className="block text-[10px] font-bold uppercase tracking-[1px] text-muted-foreground">
              IDENTITY
            </span>
            <div className="flex items-center gap-3">
              <span className="w-[60px] text-[11px] font-semibold tracking-[0.3px] text-muted-foreground">name</span>
              <input
                className="flex-1 rounded-lg border border-border bg-background px-3 py-2 text-[13px] text-foreground outline-none"
                value={selected.name || ""}
                onChange={(e) => handleFieldChange(selectedIdx!, "name", e.target.value)}
                placeholder="my_vectordb"
              />
            </div>
            <div className="flex items-center gap-3">
              <span className="w-[60px] text-[11px] font-semibold tracking-[0.3px] text-muted-foreground">db_type</span>
              <div className="flex-1">
                <Select
                  value={selected.db_type || ""}
                  onValueChange={(v) => handleFieldChange(selectedIdx!, "db_type", v)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {supportedTypes.map((t) => (
                      <SelectItem key={t} value={t}>
                        {t}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </section>

          <Separator className="my-6" />

          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-[1px] text-muted-foreground">
                PARAMETERS ({params.length})
              </span>
              <Button type="button" variant="outline" size="sm" onClick={() => handleAddParam(selectedIdx!)}>
                <Plus size={10} className="text-primary" />
                Add
              </Button>
            </div>
            {params.length === 0 && (
              <span className="text-xs italic text-muted-foreground">No additional parameters</span>
            )}
            {params.map(([key, value]) => (
              <div key={key} className="flex items-center gap-2">
                <div className="flex-[0.35]">
                  <input
                    className="w-full rounded-lg border border-border bg-muted px-2.5 py-2 text-xs font-medium text-muted-foreground"
                    value={key}
                    onChange={(e) => {
                      const newKey = e.target.value;
                      if (newKey === key) return;
                      const vdb = { ...selected };
                      const val = vdb[key];
                      delete vdb[key];
                      vdb[newKey] = val;
                      handleUpdate(selectedIdx!, vdb);
                    }}
                  />
                </div>
                <div className="flex-[0.55]">
                  <input
                    className="w-full rounded-lg border border-border bg-background px-2.5 py-2 font-mono text-xs text-foreground"
                    value={typeof value === "object" ? JSON.stringify(value) : String(value ?? "")}
                    onChange={(e) => handleFieldChange(selectedIdx!, key, e.target.value)}
                    placeholder="value"
                  />
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={() => handleRemoveParam(selectedIdx!, key)}
                  className="size-[22px] hover:bg-destructive/15"
                >
                  <Trash2 size={11} className="text-destructive" />
                </Button>
              </div>
            ))}
          </section>
        </div>
      ) : (
        <div className="flex flex-1 flex-col items-center justify-center gap-2.5">
          <div className="opacity-30"><Database size={40} className="text-muted-foreground" /></div>
          <span className="text-sm font-medium text-muted-foreground">
            {vectordbList.length === 0 ? "Add a vector database to get started" : "Select a vector database"}
          </span>
          {vectordbList.length === 0 && (
            <Button type="button" onClick={handleAdd}>
              <Plus size={12} color="#fff" />
              Add VectorDB
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
