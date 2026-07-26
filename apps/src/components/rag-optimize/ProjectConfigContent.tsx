import React, { useId, useState, useCallback, useEffect } from "react";
import {
  Play, Square, RotateCcw, RefreshCw, ArrowRightLeft, ChevronDown, ChevronUp, Loader2, Database, FileText, LockKeyhole,
  type LucideIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { StatusPill } from "@/components/ui/StatusPill";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

import { useProject } from "@/context/ProjectContext";
import {
  fetchDatasets, launchRagRun, fetchProjectStatus, killRagProject,
  type DatasetFolder, type ProjectStatus as ProjectStatusType,
  type LaunchRagRunOpts,
} from "@/api/ragProject";
import { fetchQualityStores, type QualityStoreInfo } from "@/api/qualityStore";

interface ProjectConfigContentProps {
  onSystemTransfer?: () => void;
  launchDisabledReason?: string | null;
  onLaunched?: (status: ProjectStatusType) => void;
}

export function ProjectConfigContent({
  onSystemTransfer,
  launchDisabledReason,
  onLaunched,
}: ProjectConfigContentProps) {
  const { currentProject, refreshProjects, projects } = useProject();

  const [datasets, setDatasets] = useState<DatasetFolder[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [corpusFile, setCorpusFile] = useState<string | null>(null);
  const [qaFile, setQaFile] = useState<string | null>(null);

  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [projStatus, setProjStatus] = useState<ProjectStatusType | null>(null);
  const [showConfirm, setShowConfirm] = useState(false);
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const [stopping, setStopping] = useState(false);
  // Reset & Run: destructive — wipes evaluations/, optimizer/, pareto_*,
  // registry/, resources/, summary.json counters; then launches a fresh
  // Phase-1A run on the same dataset and config. Guarded by ConfirmDialog.
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [resetting, setResetting] = useState(false);

  const [runMode, setRunMode] = useState<"full" | "hardware_replay">("full");
  const [qualityStores, setQualityStores] = useState<QualityStoreInfo[]>([]);
  const [selectedStore, setSelectedStore] = useState<string | null>(null);
  const [nRefineIterations, setNRefineIterations] = useState("0");
  const [showAdvanced, setShowAdvanced] = useState(false);

  const loadStatus = useCallback(async () => {
    if (!currentProject) return;
    try { setProjStatus(await fetchProjectStatus(currentProject)); }
    catch { setProjStatus(null); }
  }, [currentProject]);

  useEffect(() => {
    fetchDatasets().then(setDatasets).catch(() => {});
    fetchQualityStores().then((s) => setQualityStores(s.filter((st) => st.status === "completed"))).catch(() => {});
    loadStatus();
  }, [loadStatus]);

  // One-shot consume migration intent handed off from the New Project
  // dialog (sessionStorage key set by ProjectList.handleCreate when the user
  // chose Migrate). Pre-selects the Migrate toggle + source so the user lands
  // in the project view ready to launch.
  useEffect(() => {
    if (!currentProject) return;
    const key = `rag_migrate_intent_${currentProject}`;
    const raw = sessionStorage.getItem(key);
    if (!raw) return;
    try {
      const intent = JSON.parse(raw) as {
        source?: string;
        value: string;
        refineIters: number;
      };
      if (intent.source && intent.source !== "store") return;
      setRunMode("hardware_replay");
      setSelectedStore(intent.value);
      setNRefineIterations(String(intent.refineIters ?? 0));
      setShowAdvanced(true);
    } catch {
      /* malformed intent — drop silently */
    } finally {
      sessionStorage.removeItem(key);
    }
  }, [currentProject]);

  useEffect(() => {
    if (projStatus?.status !== "running") return;
    const interval = setInterval(loadStatus, 5000);
    return () => clearInterval(interval);
  }, [projStatus?.status, loadStatus]);

  const doLaunch = async () => {
    setShowConfirm(false);
    if (!currentProject || launching || launchDisabledReason) return;
    const status = projects.find((p) => p.name === currentProject)?.status;
    const isResume = status === "completed" || status === "failed";
    // First run: dataset is required. Resume: server reuses data/ from
    // the original run (a different dataset would invalidate the prior
    // quality scores), so don't send any dataset selection.
    if (!isResume && (!corpusFile || !qaFile || !selectedDataset)) return;
    setLaunching(true);
    setLaunchError(null);
    let succeeded = false;
    try {
      let opts: LaunchRagRunOpts | undefined;
      if (runMode === "hardware_replay") {
        if (selectedStore) {
          opts = {
            mode: "hardware_replay",
            qualityStorePath: qualityStores.find((s) => s.name === selectedStore)?.store_path,
            nRefineIterations: parseInt(nRefineIterations) || 0,
          };
        }
      }
      const corpusArg = isResume ? null : `${selectedDataset}/${corpusFile}`;
      const qaArg = isResume ? null : `${selectedDataset}/${qaFile}`;
      const resp = await launchRagRun(currentProject, corpusArg, qaArg, opts);
      succeeded = true;
      // Optimistically flip projStatus to running so the run-bar updates
      // immediately. Without this, there's a window between launchRagRun
      // resolving and summary.json being written by the subprocess where
      // the bar still shows "Start", and a quick second click would spawn
      // another run.
      const launchedStatus = {
        status: "running",
        pid: (resp as { pid?: number })?.pid ?? null,
        pid_alive: true,
        timestamp: new Date().toISOString(),
        n_evaluations: projStatus?.n_evaluations ?? projects.find((p) => p.name === currentProject)?.n_evaluations ?? 0,
        n_pareto_points: projStatus?.n_pareto_points ?? 0,
      };
      setProjStatus(launchedStatus);
      onLaunched?.(launchedStatus);
      refreshProjects();
      // Aggressive poll for ~10s so real PID/eval-count populate as run.py
      // boots and writes summary.json. After that the 5s slow poll takes
      // over via the existing useEffect.
      let polls = 0;
      const fastPoll = setInterval(() => {
        polls++;
        loadStatus();
        if (polls >= 20) clearInterval(fastPoll);
      }, 500);
    } catch (err) {
      setLaunchError(err instanceof Error ? err.message : String(err));
    } finally {
      // On success keep the spinner pinned briefly so the optimistic
      // running state has a chance to render before the button branch
      // re-evaluates; on failure unlock immediately so user can retry.
      if (succeeded) setTimeout(() => setLaunching(false), 800);
      else setLaunching(false);
    }
  };

  const handleLaunch = () => {
    const status = projects.find((p) => p.name === currentProject)?.status;
    // created / completed / failed all proceed without an extra confirm
    if (status === "created" || status === "completed" || status === "failed" || !status) {
      doLaunch();
    } else {
      setShowConfirm(true);
    }
  };

  const projectStatus = projects.find((p) => p.name === currentProject)?.status;
  const selectedDs = datasets.find((d) => d.name === selectedDataset);
  const replayReady = !!selectedStore;
  const isCompleted = projectStatus === "completed";
  const isFailed = projectStatus === "failed";
  const isResumable = isCompleted || isFailed;
  const isRunning = projStatus?.status === "running";
  // Migrate is only legal for raw (never-run) projects. Once a project has
  // any actual run history (completed/failed/running), migrating into it
  // would overwrite that data — hide the toggle so users can't pick it.
  const isRaw = !projectStatus || projectStatus === "created";
  const datasetReady = isResumable || (!!corpusFile && !!qaFile);
  const canLaunch =
    datasetReady &&
    !launchDisabledReason &&
    !launching &&
    (runMode === "full" || replayReady);

  // Auto-correct: if a non-raw project somehow ended up in migrate mode
  // (e.g. user toggled then resumed), snap back to optimize.
  useEffect(() => {
    if (!isRaw && runMode === "hardware_replay") {
      setRunMode("full");
      setShowAdvanced(false);
    }
  }, [isRaw, runMode]);

  return (
    <section aria-label="Evaluation run controls" className="shrink-0 border-b border-border bg-muted/20">
      <div className="flex flex-wrap items-end gap-x-5 gap-y-3 px-6 py-3">
        {isRunning ? (
          <>
            <div className="flex min-h-9 flex-wrap items-center gap-3">
              <StatusPill status="running" />
              <span className="text-sm font-medium tabular-nums">
                {projStatus?.n_evaluations ?? 0} evaluations
              </span>
              {projStatus?.pid && <span className="text-xs text-muted-foreground">PID {projStatus.pid}</span>}
            </div>
            <Button
              type="button"
              variant="outline"
              onClick={() => setShowStopConfirm(true)}
              className="ml-auto h-9 rounded-lg border-destructive/20 text-destructive shadow-none hover:bg-destructive/5 hover:text-destructive"
            >
              <Square className="size-3 fill-current" /> Stop run
            </Button>
          </>
        ) : (
          <>
            {isRaw && (
              <div className="flex shrink-0 flex-col gap-1.5">
                <span id="run-mode-label" className="text-[11px] font-medium text-muted-foreground">Run mode</span>
                <ToggleGroup
                  type="single"
                  value={runMode}
                  onValueChange={(v) => {
                    if (!v) return;
                    setRunMode(v as "full" | "hardware_replay");
                    setShowAdvanced(v === "hardware_replay");
                  }}
                  aria-labelledby="run-mode-label"
                  size="sm"
                  spacing={1}
                  className="h-9 rounded-lg bg-muted p-1"
                >
                  <ToggleGroupItem value="full" className="h-7 gap-1.5 rounded-md px-3 text-xs text-muted-foreground data-[state=on]:bg-background data-[state=on]:text-foreground data-[state=on]:shadow-sm">
                    <Play className="size-3" /> Evaluate
                  </ToggleGroupItem>
                  <ToggleGroupItem value="hardware_replay" className="h-7 gap-1.5 rounded-md px-3 text-xs text-muted-foreground data-[state=on]:bg-background data-[state=on]:text-foreground data-[state=on]:shadow-sm">
                    <ArrowRightLeft className="size-3.5" /> Migrate
                  </ToggleGroupItem>
                </ToggleGroup>
              </div>
            )}

            {isResumable ? (
              <div className="flex min-h-9 flex-wrap items-center gap-3">
                <StatusPill status={isFailed ? "failed" : "completed"} />
                <span className="text-xs tabular-nums text-muted-foreground">{projStatus?.n_evaluations ?? 0} evaluations</span>
                <span className="inline-flex items-center gap-1.5 rounded-md bg-muted px-2 py-1 text-xs text-muted-foreground" title="Resuming uses the original dataset saved with this project.">
                  <LockKeyhole className="size-3" /> Original dataset
                </span>
              </div>
            ) : (
              <div className="flex min-w-0 flex-[1_1_28rem] flex-wrap gap-3">
                <RunSelect
                  label="Dataset" icon={Database}
                  options={datasets.map((ds) => ({ key: ds.name, label: ds.name }))}
                  value={selectedDataset ?? ""}
                  onChange={(value) => { setSelectedDataset(value); setCorpusFile(null); setQaFile(null); }}
                  placeholder="Select dataset"
                />
                <RunSelect
                  label="Corpus" icon={FileText}
                  options={(selectedDs?.parquet_files ?? []).map((file) => ({ key: file, label: file }))}
                  value={corpusFile ?? ""} onChange={setCorpusFile}
                  placeholder="Select corpus" disabled={!selectedDs}
                />
                <RunSelect
                  label="QA" icon={FileText}
                  options={(selectedDs?.parquet_files ?? []).map((file) => ({ key: file, label: file }))}
                  value={qaFile ?? ""} onChange={setQaFile}
                  placeholder="Select QA" disabled={!selectedDs}
                />
              </div>
            )}

            <div className="ml-auto flex max-w-full flex-wrap items-center justify-end gap-2">
              {runMode === "hardware_replay" && (
                <Button
                  type="button" variant="ghost" size="sm"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  aria-expanded={showAdvanced} aria-controls="migration-options"
                  className="h-9 rounded-lg text-xs text-muted-foreground"
                >
                  Options {showAdvanced ? <ChevronUp className="size-3.5" /> : <ChevronDown className="size-3.5" />}
                </Button>
              )}
              {isResumable && runMode === "full" && onSystemTransfer && (
                <Button type="button" variant="outline" onClick={onSystemTransfer} disabled={launching || resetting} className="h-9 rounded-lg shadow-none">
                  <ArrowRightLeft className="size-3.5" /> System transfer
                </Button>
              )}
              {isResumable && runMode === "full" && (
                <Button
                  type="button" variant="outline"
                  onClick={() => setShowResetConfirm(true)} disabled={!canLaunch || launching || resetting}
                  title="Wipes prior evaluations and runs fresh on the same dataset and configuration"
                  className="h-9 rounded-lg border-destructive/20 text-destructive shadow-none hover:bg-destructive/5 hover:text-destructive"
                >
                  {resetting ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
                  {resetting ? "Resetting…" : "Reset & Run"}
                </Button>
              )}
              <Button
                type="button" onClick={handleLaunch} disabled={!canLaunch || launching}
                title={launchDisabledReason || undefined}
                className="h-9 min-w-32 rounded-lg px-4 text-xs font-semibold shadow-sm disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100 disabled:shadow-none"
              >
                {launching ? <Loader2 className="size-3.5 animate-spin" /> : runMode === "hardware_replay" ? <ArrowRightLeft className="size-3.5" /> : isResumable ? <RotateCcw className="size-3.5" /> : <Play className="size-3 fill-current" />}
                {launching ? "Starting…" : runMode === "hardware_replay" ? "Start migration" : isResumable ? "Resume run" : "Start evaluation"}
              </Button>
            </div>
          </>
        )}
      </div>

      {!isRunning && launchDisabledReason && (
        <p role="status" className="px-6 pb-3 text-xs text-muted-foreground">{launchDisabledReason}</p>
      )}
      {launchError && <p role="alert" className="px-6 pb-3 text-sm text-destructive">{launchError}</p>}

      {!isRunning && runMode === "hardware_replay" && showAdvanced && (
        <div id="migration-options" className="flex flex-wrap items-end gap-4 border-t border-border/60 px-6 py-3">
          <RunSelect
            label="Quality store" icon={Database}
            options={qualityStores.map((store) => ({ key: store.name, label: `${store.name} (${store.n_configs} configs)` }))}
            value={selectedStore ?? ""} onChange={setSelectedStore}
            placeholder="Select quality store"
          />
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="refine-iterations" className="text-[11px] font-medium text-muted-foreground">Refine iterations</Label>
            <Input id="refine-iterations" type="number" value={nRefineIterations}
              onChange={(event) => setNRefineIterations(event.target.value)}
              className="h-9 w-24 rounded-lg bg-background text-center font-mono text-xs shadow-none" />
          </div>
          <p className="pb-2 text-xs text-muted-foreground">0 = reuse quality without refinement</p>
        </div>
      )}

      {/* ── Dialogs ────────────────────────────────────────────── */}
      <ConfirmDialog
        open={showConfirm}
        onOpenChange={setShowConfirm}
        title="Project already run"
        description={`This project has status "${projectStatus}". Running again may overwrite existing results. Are you sure?`}
        confirmLabel="Continue"
        onConfirm={doLaunch}
      />
      <ConfirmDialog
        open={showStopConfirm}
        onOpenChange={setShowStopConfirm}
        title="Stop optimization"
        description={`Are you sure you want to stop "${currentProject}"? The process will be terminated and current progress will be saved.`}
        confirmLabel="Stop"
        loading={stopping}
        destructive
        onConfirm={async () => {
          if (!currentProject) return;
          try {
            setStopping(true);
            await killRagProject(currentProject);
            setShowStopConfirm(false);
            loadStatus(); refreshProjects();
          } catch { setShowStopConfirm(false); }
          finally { setStopping(false); }
        }}
      />
      <ConfirmDialog
        open={showResetConfirm}
        onOpenChange={setShowResetConfirm}
        title="Reset and run from scratch?"
        description={`Delete all evaluations for "${currentProject}" and run optimization again on the same dataset and config. Logs are preserved.`}
        confirmLabel="Reset & Run"
        loading={resetting}
        destructive
        onConfirm={async () => {
          if (!currentProject) return;
          try {
            setResetting(true);
            const resp = await launchRagRun(currentProject, null, null, {
              mode: "full",
              reset: true,
            });
            // Match the optimistic update in doLaunch so the bar flips to
            // "running" immediately, otherwise a quick second click could
            // race the subprocess writing summary.json.
            setProjStatus({
              status: "running",
              pid: (resp as { pid?: number })?.pid ?? null,
              pid_alive: true,
              timestamp: new Date().toISOString(),
              n_evaluations: 0,
              n_pareto_points: 0,
            } as ProjectStatusType);
            setShowResetConfirm(false);
            loadStatus(); refreshProjects();
          } catch { setShowResetConfirm(false); }
          finally { setResetting(false); }
        }}
      />
    </section>
  );
}

/** One consistent, labeled control for the run's data inputs. */
function RunSelect({ label, icon: Icon, options, value, onChange, placeholder, disabled = false }: {
  label: string;
  icon: LucideIcon;
  options: { key: string; label: string }[];
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="flex min-w-0 basis-36 grow flex-col gap-1.5 sm:max-w-64">
      <Label htmlFor={id} className="text-[11px] font-medium text-muted-foreground">{label}</Label>
      <Select value={value} onValueChange={onChange} disabled={disabled}>
        <SelectTrigger id={id} title={value || undefined} className="w-full min-w-0 gap-2 rounded-lg bg-background text-xs font-normal shadow-none hover:border-foreground/25 disabled:bg-muted/50 [&_[data-slot=select-value]]:block [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:flex-1 [&_[data-slot=select-value]]:truncate [&_[data-slot=select-value]]:text-left">
          <Icon className="size-3.5 text-muted-foreground" />
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent position="popper" align="start" className="max-w-[min(28rem,calc(100vw-2rem))]">
          {options.map((option) => <SelectItem key={option.key} value={option.key} title={option.label} className="text-xs [&>span:last-child]:block [&>span:last-child]:truncate">{option.label}</SelectItem>)}
        </SelectContent>
      </Select>
    </div>
  );
}
