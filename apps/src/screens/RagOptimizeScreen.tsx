/**
 * RagOptimizeScreen — top-level page for the RAG Optimize / home route.
 *
 * Web port of apps/app/(tabs)/index.tsx, redesigned around shadcn primitives:
 *   - shadcn <Tabs> for sub-tabs (Dashboard / History / Pareto Configs)
 *   - shadcn <DropdownMenu> for the project-list sort picker
 *   - <PageHeaderTitle> / <PageHeaderActions> matching every other route
 *
 * Layout:
 *   - No project selected → PageHeader + full-page <ProjectList>
 *   - Project selected → <Tabs> wraps PageHeader + split view
 *     (sidebar with project list  ·  main tab content  ·  optional analysis panel)
 *
 * `ProjectProvider` and `SelectionProvider` already wrap the app in
 * `__root.tsx`, so this screen only adds the analysis providers
 * (`PostAnalysisProvider`, `PerformancePolicyProvider`) and the bridge
 * that pumps the sweep cache into the policy context.
 */
import React, { useState, useEffect, useCallback } from "react";
import { useNavigate, useSearch } from "@tanstack/react-router";
import {
  ArrowLeft,
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  ArrowRightLeft,
  BarChart3,
  Database,
  FolderOpen,
  Loader2,
  Plus,
  RefreshCw,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  PageHeader,
  PageHeaderTitle,
  PageHeaderActions,
} from "@/components/ui/PageHeader";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { StatusPill } from "@/components/ui/StatusPill";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { useProject, type ProjectSortKey } from "@/context/ProjectContext";
import { useSelection } from "@/context/SelectionContext";
import { PostAnalysisProvider } from "@/context/PostAnalysisContext";
import { PerformancePolicyProvider } from "@/context/PerformancePolicyContext";
import { SweepCacheBridge } from "@/context/SweepCacheBridge";
import { EvalDataProvider } from "@/context/EvalDataContext";

import { ProjectList } from "@/components/rag-optimize/ProjectList";
import { ProjectSetupTab } from "@/components/rag-optimize/ProjectSetupTab";
import { projectView, type ProjectTab as SubTab } from "@/components/rag-optimize/projectView";
import { ProjectConfigContent } from "@/components/rag-optimize/ProjectConfigContent";
import { EvalNavigator } from "@/components/rag-optimize/EvalNavigator";
import { OverviewTab } from "@/components/rag-optimize/OverviewTab";
import { TradeoffTab } from "@/components/rag-optimize/TradeoffTab";
import { PerformanceTab } from "@/components/rag-optimize/PerformanceTab";
import { QualityTab } from "@/components/rag-optimize/QualityTab";
import { ConfigTab } from "@/components/rag-optimize/ConfigTab";
import { HistoryTab } from "@/components/rag-optimize/HistoryTab";
import { CMPreviewTab } from "@/components/rag-optimize/CMPreviewTab";
import { PostAnalysisPanel } from "@/components/rag-optimize/PostAnalysisPanel";
import { SystemYamlPicker } from "@/components/rag-optimize/SystemYamlPicker";

import {
  deleteRagProject,
  fetchProjectStatus,
  fetchSystemYamls,
  launchSystemTransferProject,
  saveProjectQualityToStore,
  type ProjectStatus,
  type SaveProjectQualityStoreResult,
  type SystemYamlInfo,
} from "@/api/ragProject";
import {
  fetchQualityStores,
  type QualityStoreInfo,
} from "@/api/qualityStore";
import {
  isValidOptionalRefineIterations,
  parseOptionalRefineIterations,
} from "@/lib/systemTransfer";
import { cn } from "@/lib/utils";

const SORT_OPTIONS: { key: ProjectSortKey; label: string }[] = [
  { key: "time", label: "Time" },
  { key: "name", label: "Name" },
  { key: "status", label: "Status" },
  { key: "evaluations", label: "Evaluations" },
];

export default function RagOptimizeScreen() {
  const { currentProject } = useProject();
  // Every request, editor draft and dataset selection belongs to one project.
  return <ProjectScreen key={currentProject ?? "__project_list__"} />;
}

function ProjectScreen() {
  const {
    projects,
    currentProject,
    setCurrentProject,
    refreshProjects,
    sortKey,
    sortDirection,
    setSortKey,
    setSortDirection,
  } = useProject();

  const [projStatus, setProjStatus] = useState<ProjectStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [launchDisabledReason, setLaunchDisabledReason] = useState<string | null>("Loading project configuration…");
  const currentProjectInfo = projects.find((p) => p.name === currentProject);
  const evaluationCount = projStatus?.n_evaluations ?? currentProjectInfo?.n_evaluations;
  const projectStatus = projStatus?.status ?? currentProjectInfo?.status;
  const isRunning = projectStatus === "running";

  const navigate = useNavigate({ from: "/" });
  const search = useSearch({ from: "/" });
  const { tab: subTab, tabs: subTabs, hasEvaluations, loading: projectLoading } = projectView(evaluationCount, search.tab);
  const setSubTab = useCallback(
    (next: SubTab) =>
      navigate({
        search: (prev) => ({ ...prev, tab: next }),
        replace: true,
      }),
    [navigate],
  );

  const [showNewProject, setShowNewProject] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [showSaveQualityStore, setShowSaveQualityStore] = useState(false);
  const [showSystemTransfer, setShowSystemTransfer] = useState(false);
  const hasSelection = currentProject != null;

  // Selection lives at app root, so a stale eval id from the previous
  // project leaks across — useRagoSweep then 404s on the new project's
  // baseUrl and the chart sticks at "No RAGO sweep data". Reset on
  // every project switch.
  const { setSelectedEvalId } = useSelection();
  useEffect(() => {
    setSelectedEvalId(null);
  }, [currentProject, setSelectedEvalId]);

  // Deep-link: /?project=X&tab=performance&eval=N (Benchmark → eval point).
  // Applies the one-shot param to SelectionContext, then strips it so later
  // manual selection isn't pinned to a stale URL. Declared AFTER the reset
  // effect — same-commit ordering guarantees the reset runs first.
  const evalParam = search.eval;
  useEffect(() => {
    if (evalParam == null) return;
    setSelectedEvalId(evalParam);
    navigate({
      search: (prev) => ({ ...prev, eval: undefined }),
      replace: true,
    });
  }, [evalParam, setSelectedEvalId, navigate]);

  // Trade-off knob coloring: picked in the TradeoffTab impact table, painted
  // on the sidebar map (the map is the ONE objective-space chart — the tab
  // adds function to it instead of duplicating it). Scoped to the Trade-off
  // tab and reset per project (knob keys are project-specific).
  const [tradeoffColorKey, setTradeoffColorKey] = useState<string>("none");
  useEffect(() => {
    setTradeoffColorKey("none");
  }, [currentProject]);

  // Lazy-activation set: a tab's heavy content only mounts after the
  // user actually visits it. Once mounted, `forceMount` keeps it alive
  // so subsequent tab switches don't re-fetch. Resets when the project
  // changes so opening another project doesn't carry over fetch work.
  const [activatedTabs, setActivatedTabs] = useState<Set<SubTab>>(
    () => new Set([subTab]),
  );
  useEffect(() => {
    setActivatedTabs(new Set([subTab]));
    // Intentionally only triggered by project change — see comment above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentProject]);
  useEffect(() => {
    setActivatedTabs((prev) => {
      if (prev.has(subTab)) return prev;
      const next = new Set(prev);
      next.add(subTab);
      return next;
    });
  }, [subTab]);
  useEffect(() => {
    if (currentProject == null) return;
    let active = true;
    const loadStatus = async () => {
      try {
        const next = await fetchProjectStatus(currentProject);
        if (active) { setProjStatus(next); setStatusError(null); }
      } catch (err) {
        if (active) setStatusError(err instanceof Error ? err.message : String(err));
      }
    };
    loadStatus();
    const interval = setInterval(loadStatus, 5000);
    return () => { active = false; clearInterval(interval); };
  }, [currentProject]);

  // ── No-selection view (full-page project list) ─────────────────────
  if (!hasSelection) {
    return (
      <div className="flex flex-1 flex-col bg-background">
        <PageHeader>
          <PageHeaderTitle icon={BarChart3} title="RAG Optimization" />
          <PageHeaderActions>
            <Button
              type="button"
              size="sm"
              onClick={() => setShowNewProject(!showNewProject)}
            >
              <Plus className="size-4" />
              New Project
            </Button>
          </PageHeaderActions>
        </PageHeader>
        <ProjectList
          showNewProject={showNewProject}
          setShowNewProject={setShowNewProject}
        />
      </div>
    );
  }

  // ── Selection view (split layout with tabs) ────────────────────────
  // `h-svh + overflow-hidden` locks JUST this screen at viewport height
  // (other routes — settings, dataset-generator, cost-models — keep their
  // default document-scroll behavior). Combined with the min-h-0 chain
  // inside, this is what lets CodeMirror, Corpus, VectorDB tabs all do
  // their internal scrolling without bleeding into page-level scroll.
  return (
    <Tabs
      value={subTab}
      onValueChange={(v) => setSubTab(v as SubTab)}
      className="flex h-svh flex-col gap-0 overflow-hidden bg-background min-w-0"
    >
      <PageHeader>
        <PageHeaderTitle
          icon={BarChart3}
          title="RAG Optimization"
          prefix={
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => setCurrentProject(null as any)}
              className="size-8"
              aria-label="Back to project list"
            >
              <ArrowLeft className="size-4 text-muted-foreground" />
            </Button>
          }
          badge={
            <>
              <div className="h-4 w-px bg-border" />
              <ProjectHeaderSelect
                projects={projects}
                currentProject={currentProject}
                setCurrentProject={setCurrentProject}
              />
            </>
          }
        />
        <div className="flex items-center gap-2">
          <TabsList>
            {subTabs.map((t) => (
              <TabsTrigger key={t.key} value={t.key}>
                {t.label}
              </TabsTrigger>
            ))}
          </TabsList>
          <PageHeaderActions>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => setShowAnalysis((v) => !v)}
              disabled={!hasEvaluations}
              aria-label="Toggle analysis panel"
              className={cn(
                "size-8",
                showAnalysis && "bg-primary/10 text-primary",
              )}
            >
              <SlidersHorizontal
                className={cn(
                  "size-4",
                  showAnalysis ? "text-primary" : "text-muted-foreground",
                )}
              />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => setShowSaveQualityStore(true)}
              disabled={isRunning || !hasEvaluations}
              aria-label="Save quality to store"
              className="size-8"
            >
              <Database className="size-4 text-muted-foreground" />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={refreshProjects}
              aria-label="Reload"
              className="size-8"
            >
              <RefreshCw className="size-4 text-muted-foreground" />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => setShowDeleteModal(true)}
              aria-label="Delete project"
              className="size-8 hover:bg-destructive/10"
            >
              <Trash2 className="size-4 text-destructive" />
            </Button>
          </PageHeaderActions>
        </div>
      </PageHeader>

      {/* Project-level run-bar — persists across all tabs, outside the
          eval-scoped analysis providers (it only needs ProjectContext). */}
      <ProjectConfigContent
        onSystemTransfer={() => setShowSystemTransfer(true)}
        launchDisabledReason={!hasEvaluations ? launchDisabledReason : null}
        onLaunched={setProjStatus}
      />

      {/* Analysis providers wrap BOTH the eval navigator sidebar AND the tabs
          so the sidebar Pareto + every tab share one policy / quality / sweep
          cache. SweepCacheBridge mounts unconditionally now — the sidebar
          Pareto is always visible and needs the bulk sweeps. EvalDataProvider
          fetches the eval list + client Pareto exactly once. */}
      <PostAnalysisProvider>
        <PerformancePolicyProvider>
          <SweepCacheBridge />
          <EvalDataProvider>
            <div className="flex flex-1 min-w-0 overflow-hidden">
              {/* ── Eval navigator sidebar (Pareto map + eval index).
                    On the Trade-off tab the map turns multi-functional:
                    the impact table's active knob recolors its points. ── */}
              {hasEvaluations && <EvalNavigator
                colorKey={subTab === "tradeoff" ? tradeoffColorKey : "none"}
              />}

              {/* ── Main tab content ─────────────────────────────── */}
              <div className="flex flex-1 min-w-0 min-h-0">
                <div className="flex flex-1 min-w-0 min-h-0 flex-col">
                  {/* forceMount + data-[state=inactive]:hidden + activatedTabs
                      keep each tab's state (selected sweep point, in-flight
                      fetches) alive across tab switches. */}
                  <TabsContent
                    forceMount
                    value="start"
                    className="data-[state=inactive]:hidden flex flex-1 min-w-0 min-h-0 flex-col m-0"
                  >
                    {!hasEvaluations && (projectLoading ? (
                      <div role={statusError ? "alert" : "status"} className="flex flex-1 items-center justify-center gap-2 text-sm text-muted-foreground">
                        {statusError ? statusError : <><Loader2 className="size-4 animate-spin" /> Loading project…</>}
                      </div>
                    ) : (
                      <ProjectSetupTab
                        project={currentProject!}
                        status={projectStatus}
                        onLaunchReadiness={setLaunchDisabledReason}
                      />
                    ))}
                  </TabsContent>
                  <TabsContent
                    forceMount
                    value="overview"
                    className="data-[state=inactive]:hidden flex flex-1 min-w-0 min-h-0 flex-col m-0"
                  >
                    {hasEvaluations && activatedTabs.has("overview") && <OverviewTab />}
                  </TabsContent>
                  <TabsContent
                    forceMount
                    value="tradeoff"
                    className="data-[state=inactive]:hidden flex flex-1 min-w-0 min-h-0 flex-col m-0"
                  >
                    {hasEvaluations && activatedTabs.has("tradeoff") && (
                      <TradeoffTab
                        colorKey={tradeoffColorKey}
                        onColorKeyChange={setTradeoffColorKey}
                      />
                    )}
                  </TabsContent>
                  <TabsContent
                    forceMount
                    value="performance"
                    className="data-[state=inactive]:hidden flex flex-1 min-w-0 min-h-0 flex-col m-0"
                  >
                    {hasEvaluations && activatedTabs.has("performance") && <PerformanceTab />}
                  </TabsContent>
                  <TabsContent
                    forceMount
                    value="quality"
                    className="data-[state=inactive]:hidden flex flex-1 min-w-0 min-h-0 flex-col m-0"
                  >
                    {hasEvaluations && activatedTabs.has("quality") && <QualityTab />}
                  </TabsContent>
                  <TabsContent
                    forceMount
                    value="config"
                    className="data-[state=inactive]:hidden flex flex-1 min-w-0 min-h-0 flex-col m-0"
                  >
                    {hasEvaluations && activatedTabs.has("config") && <ConfigTab />}
                  </TabsContent>
                  <TabsContent
                    forceMount
                    value="cm-preview"
                    className="data-[state=inactive]:hidden flex flex-1 min-w-0 min-h-0 flex-col m-0"
                  >
                    {activatedTabs.has("cm-preview") && <CMPreviewTab />}
                  </TabsContent>
                  <TabsContent
                    forceMount
                    value="history"
                    className="data-[state=inactive]:hidden flex flex-1 min-w-0 min-h-0 flex-col m-0"
                  >
                    {activatedTabs.has("history") && <HistoryTab />}
                  </TabsContent>
                </div>
                {showAnalysis && hasEvaluations && (
                  <PostAnalysisPanel onClose={() => setShowAnalysis(false)} />
                )}
              </div>
            </div>
          </EvalDataProvider>
        </PerformancePolicyProvider>
      </PostAnalysisProvider>

      <SaveQualityStoreDialog
        open={showSaveQualityStore}
        onOpenChange={setShowSaveQualityStore}
        projectName={currentProject || ""}
        onSaved={refreshProjects}
      />

      <SystemTransferDialog
        open={showSystemTransfer}
        onOpenChange={setShowSystemTransfer}
        sourceProject={currentProject || ""}
        onLaunched={async (projectName) => {
          await refreshProjects();
          setCurrentProject(projectName);
        }}
      />

      {/* Delete project modal */}
      <ConfirmDialog
        open={showDeleteModal}
        onOpenChange={setShowDeleteModal}
        title="Delete project"
        description={`Are you sure you want to delete "${currentProject}"? All evaluations, configs, and results will be permanently removed.`}
        confirmLabel="Delete"
        loading={deleting}
        destructive
        onConfirm={async () => {
          if (!currentProject) return;
          try {
            setDeleting(true);
            await deleteRagProject(currentProject);
            setShowDeleteModal(false);
            setCurrentProject(null as any);
            refreshProjects();
          } catch (_e) {
            setShowDeleteModal(false);
          } finally {
            setDeleting(false);
          }
        }}
      />
    </Tabs>
  );
}

// Root project (name === "") can't be a Radix Select value (empty string is
// reserved), so map it to a sentinel and translate back on change.
const ROOT_SENTINEL = "__root__";

function ProjectHeaderSelect({
  projects,
  currentProject,
  setCurrentProject,
}: {
  projects: { name: string; status?: string }[];
  currentProject: string | null;
  setCurrentProject: (name: string) => void;
}) {
  const value =
    currentProject == null
      ? ""
      : currentProject === ""
        ? ROOT_SENTINEL
        : currentProject;
  return (
    <Select
      value={value}
      onValueChange={(v) => setCurrentProject(v === ROOT_SENTINEL ? "" : v)}
    >
      <SelectTrigger size="sm" className="h-8 max-w-[260px] gap-1.5 font-medium">
        <FolderOpen className="size-3.5 shrink-0 text-muted-foreground" />
        <SelectValue placeholder="Select project" />
      </SelectTrigger>
      <SelectContent>
        {projects.map((p) => (
          <SelectItem
            key={p.name}
            value={p.name || ROOT_SENTINEL}
            textValue={p.name || "root"}
          >
            <span className="flex items-center gap-2">
              <StatusPill status={p.status ?? "created"} showLabel={false} />
              {p.name || "root"}
            </span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function datasetText(dataset?: Record<string, unknown>): string {
  if (!dataset) return "dataset unknown";
  const name = typeof dataset.dataset_name === "string" ? dataset.dataset_name : "";
  const qa = typeof dataset.qa === "string" ? dataset.qa : "";
  const corpus = typeof dataset.corpus === "string" ? dataset.corpus : "";
  if (name) return name;
  if (qa || corpus) return [qa, corpus].filter(Boolean).join(" · ");
  return "dataset unknown";
}

function SaveQualityStoreDialog({
  open,
  onOpenChange,
  projectName,
  onSaved,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectName: string;
  onSaved: () => Promise<void>;
}) {
  const [stores, setStores] = useState<QualityStoreInfo[]>([]);
  const [mode, setMode] = useState<"existing" | "new">("existing");
  const [selectedStore, setSelectedStore] = useState("");
  const [newStoreName, setNewStoreName] = useState("");
  const [loadingStores, setLoadingStores] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SaveProjectQualityStoreResult | null>(null);

  const loadStores = useCallback(async () => {
    setLoadingStores(true);
    try {
      const nextStores = await fetchQualityStores();
      setStores(nextStores);
      setSelectedStore((prev) => prev || nextStores[0]?.name || "");
      if (nextStores.length === 0) setMode("new");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingStores(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    setError(null);
    setResult(null);
    loadStores();
  }, [loadStores, open]);

  const targetStoreName = mode === "existing" ? selectedStore : newStoreName.trim();
  const canSave = !!projectName && !!targetStoreName && !saving;

  const handleSave = useCallback(async () => {
    if (!canSave) return;
    setSaving(true);
    setError(null);
    setResult(null);
    try {
      const saved = await saveProjectQualityToStore(projectName, {
        mode,
        storeName: targetStoreName,
      });
      setResult(saved);
      await Promise.all([loadStores(), onSaved()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }, [canSave, loadStores, mode, onSaved, projectName, targetStoreName]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Save to quality store</DialogTitle>
          <DialogDescription>
            Copy quality evaluations from {projectName} into a reusable quality store.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4">
          <div className="grid gap-2">
            <Label htmlFor="quality-store-mode">Target</Label>
            <Select value={mode} onValueChange={(v) => setMode(v as "existing" | "new")}>
              <SelectTrigger id="quality-store-mode">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="existing" disabled={stores.length === 0}>
                  Existing quality store
                </SelectItem>
                <SelectItem value="new">New quality store</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {mode === "existing" ? (
            <div className="grid gap-2">
              <Label htmlFor="quality-store-existing">Quality store</Label>
              <Select
                value={selectedStore}
                onValueChange={setSelectedStore}
                disabled={loadingStores || stores.length === 0}
              >
                <SelectTrigger id="quality-store-existing">
                  <SelectValue placeholder={loadingStores ? "Loading..." : "Select store"} />
                </SelectTrigger>
                <SelectContent>
                  {stores.map((store) => (
                    <SelectItem key={store.name} value={store.name}>
                      {store.name} · {store.n_configs} configs
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedStore && (
                <span className="text-xs text-muted-foreground">
                  {datasetText(stores.find((s) => s.name === selectedStore)?.dataset)}
                </span>
              )}
            </div>
          ) : (
            <div className="grid gap-2">
              <Label htmlFor="quality-store-new">Quality store name</Label>
              <Input
                id="quality-store-new"
                value={newStoreName}
                onChange={(e) => setNewStoreName(e.target.value)}
                placeholder={`${projectName}_quality`}
              />
            </div>
          )}

          {error && (
            <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive">
              {error}
            </div>
          )}

          {result && (
            <div className="rounded-md border bg-muted/40 px-3 py-2 text-sm text-muted-foreground">
              Imported {result.n_imported} configs into {result.store_name}. Total configs: {result.n_total}.
            </div>
          )}
        </div>

        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={saving}>
            Close
          </Button>
          <Button type="button" onClick={handleSave} disabled={!canSave}>
            {saving ? <Loader2 className="size-4 animate-spin" /> : <Database className="size-4" />}
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function SystemTransferDialog({
  open,
  onOpenChange,
  sourceProject,
  onLaunched,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sourceProject: string;
  onLaunched: (projectName: string) => Promise<void>;
}) {
  const [systemYamls, setSystemYamls] = useState<SystemYamlInfo[]>([]);
  const [projectName, setProjectName] = useState("");
  const [selectedSystemYaml, setSelectedSystemYaml] = useState("");
  const [refineIterations, setRefineIterations] = useState("");
  const [loadingYamls, setLoadingYamls] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSystemYamls = useCallback(async () => {
    setLoadingYamls(true);
    try {
      const items = await fetchSystemYamls();
      setSystemYamls(items);
      setSelectedSystemYaml((prev) => prev || items[0]?.path || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingYamls(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    setProjectName(sourceProject ? `${sourceProject}_system_transfer` : "");
    setSelectedSystemYaml("");
    setRefineIterations("");
    setError(null);
    loadSystemYamls();
  }, [loadSystemYamls, open, sourceProject]);

  const selectedSystemYamlInfo = systemYamls.find(
    (s) => s.path === selectedSystemYaml,
  );
  const yamlRefineDefault = selectedSystemYamlInfo?.n_refine_iterations ?? 5;
  const refineOverride = parseOptionalRefineIterations(refineIterations);
  const refineIterationsValid = isValidOptionalRefineIterations(refineIterations);
  const effectiveRefineIterations = refineOverride ?? yamlRefineDefault;

  const canLaunch =
    !!sourceProject &&
    !!projectName.trim() &&
    !!selectedSystemYaml &&
    refineIterationsValid &&
    !launching;

  const handleLaunch = useCallback(async () => {
    if (!canLaunch) return;
    const nextProject = projectName.trim();
    setLaunching(true);
    setError(null);
    try {
      await launchSystemTransferProject({
        projectName: nextProject,
        sourceProject,
        systemYaml: selectedSystemYaml,
        nRefineIterations: refineOverride,
      });
      onOpenChange(false);
      await onLaunched(nextProject);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLaunching(false);
    }
  }, [
    canLaunch,
    onLaunched,
    onOpenChange,
    projectName,
    refineOverride,
    selectedSystemYaml,
    sourceProject,
  ]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[calc(100vw-2rem)] max-w-[calc(100vw-2rem)] overflow-hidden sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>System transfer</DialogTitle>
          <DialogDescription>
            Create a new project from {sourceProject} with a different system block.
          </DialogDescription>
        </DialogHeader>

        <div className="grid min-w-0 gap-5 overflow-hidden lg:grid-cols-[minmax(0,1fr)_16rem]">
          <div className="grid min-w-0 gap-4 overflow-hidden">
            <div className="grid gap-2">
              <Label htmlFor="system-transfer-name">New project name</Label>
              <Input
                id="system-transfer-name"
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                className="font-mono text-sm"
              />
            </div>

            <div className="grid min-w-0 gap-2 overflow-hidden">
              <Label>System YAML</Label>
              <SystemYamlPicker
                items={systemYamls}
                value={selectedSystemYaml}
                onValueChange={setSelectedSystemYaml}
                loading={loadingYamls}
                disabled={launching}
              />
            </div>
          </div>

          <div className="grid min-w-0 content-start gap-4 overflow-hidden">
            <div className="min-w-0 overflow-hidden rounded-md border bg-muted/30 p-3">
              <div className="mb-3 flex items-center gap-2 text-sm font-medium">
                <ArrowRightLeft className="size-4 text-primary" />
                Transfer
              </div>
              <dl className="space-y-3 text-xs">
                <div className="min-w-0">
                  <dt className="text-muted-foreground">Source project</dt>
                  <dd className="mt-0.5 truncate font-mono text-foreground" title={sourceProject}>
                    {sourceProject || "none"}
                  </dd>
                </div>
                <div className="min-w-0">
                  <dt className="text-muted-foreground">System</dt>
                  <dd className="mt-0.5 truncate font-mono text-foreground" title={selectedSystemYaml}>
                    {selectedSystemYamlInfo?.filename || "none"}
                  </dd>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  <Badge variant="secondary" className="rounded-md">
                    {refineIterationsValid
                      ? refineOverride === undefined
                        ? `YAML default · ${effectiveRefineIterations}`
                        : `Override · ${refineOverride}`
                      : "Invalid override"}
                  </Badge>
                  {selectedSystemYamlInfo?.summary.performance_source && (
                    <Badge variant="outline" className="rounded-md">
                      {selectedSystemYamlInfo.summary.performance_source}
                    </Badge>
                  )}
                </div>
              </dl>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="system-transfer-refine">Refine iterations override</Label>
              <Input
                id="system-transfer-refine"
                type="number"
                min="0"
                value={refineIterations}
                onChange={(e) => setRefineIterations(e.target.value)}
                placeholder={`Use YAML default (${yamlRefineDefault})`}
                aria-describedby="system-transfer-refine-help"
                aria-invalid={!refineIterationsValid}
                className="font-mono text-sm"
              />
              <p id="system-transfer-refine-help" className="text-xs text-muted-foreground">
                {!refineIterationsValid
                  ? "Enter a non-negative whole number, or leave this field blank."
                  : refineOverride === undefined
                  ? `Leave blank to run the ${yamlRefineDefault} iterations declared by this YAML.`
                  : `This run will override the YAML default with ${refineOverride} iterations.`}
              </p>
            </div>
          </div>
        </div>

        {error && (
          <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive">
            {error}
          </div>
        )}

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={launching}
          >
            Cancel
          </Button>
          <Button type="button" onClick={handleLaunch} disabled={!canLaunch}>
            {launching ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <ArrowRightLeft className="size-4" />
            )}
            Transfer
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ── Project sort menu ────────────────────────────────────────────────

interface ProjectSortMenuProps {
  sortKey: ProjectSortKey;
  setSortKey: (k: ProjectSortKey) => void;
  sortDirection: "asc" | "desc";
  setSortDirection: (d: "asc" | "desc") => void;
}

function ProjectSortMenu({
  sortKey,
  setSortKey,
  sortDirection,
  setSortDirection,
}: ProjectSortMenuProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-6 gap-1 px-1.5 text-[10px] font-medium text-muted-foreground"
          aria-label="Sort projects"
        >
          <ArrowUpDown className="size-3" />
          <span className="capitalize">{sortKey}</span>
          {sortDirection === "asc" ? (
            <ArrowUp className="size-3" />
          ) : (
            <ArrowDown className="size-3" />
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        <DropdownMenuLabel>Sort by</DropdownMenuLabel>
        <DropdownMenuRadioGroup
          value={sortKey}
          onValueChange={(v) => setSortKey(v as ProjectSortKey)}
        >
          {SORT_OPTIONS.map((o) => (
            <DropdownMenuRadioItem key={o.key} value={o.key}>
              {o.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
        <DropdownMenuSeparator />
        <DropdownMenuLabel>Direction</DropdownMenuLabel>
        <DropdownMenuRadioGroup
          value={sortDirection}
          onValueChange={(v) => setSortDirection(v as "asc" | "desc")}
        >
          <DropdownMenuRadioItem value="asc">
            <ArrowUp className="size-3" />
            Ascending
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="desc">
            <ArrowDown className="size-3" />
            Descending
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
