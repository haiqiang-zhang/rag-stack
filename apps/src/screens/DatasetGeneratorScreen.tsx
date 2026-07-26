/**
 * DatasetGeneratorScreen — top-level page for the Dataset route.
 *
 * Two sub-tabs:
 *   - Generate: side panel of YAML configs + ConfigEditor + Launch + jobs list
 *   - Datasets: side panel of generated dataset folders + parquet preview table
 *
 * Layout follows the typography-driven, no-card-chrome pattern used in
 * Settings / Quality Stores / Optimizer. Sections lead with `<h2>` + optional
 * description + `<Separator className="my-6">`; rows sit on the page background
 * directly with no surface boxes around them.
 */
import React, { useState, useEffect, useCallback, useMemo } from "react";
import yaml from "js-yaml";
import {
  Loader2,
  Play,
  RefreshCw,
  Database,
  Square,
  FileText,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { PageHeader, PageHeaderTitle } from "@/components/ui/PageHeader";
import { SidePanel, SidePanelHeader } from "@/components/ui/SidePanel";
import { SideItem, SideItemDot } from "@/components/ui/SideItem";
import { Badge } from "@/components/ui/badge";
import { StatusPill } from "@/components/ui/StatusPill";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { PillTabs } from "@/components/common/PillTabs";
import { cn } from "@/lib/utils";

import {
  fetchDatasetList, fetchDatasetPreview,
  fetchDatasetGenConfigs, saveDatasetGenConfig,
  launchDatasetGeneration, fetchDatasetGenJobs, killDatasetGenJob,
  type DatasetInfo, type DatasetGenConfig, type DatasetGenJob,
} from "@/api/costModelClient";
import { ConfigEditor } from "@/components/ConfigEditor";

type SubTab = "generate" | "datasets";

/* -- Format helpers ------------------------------------------------- */

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s.toFixed(0)}s`;
}

/* -- Section header (matches Settings / Optimizer pattern) ---------- */

function SectionHeader({
  title,
  description,
  action,
}: {
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <>
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
          {description && (
            <p className="text-sm text-muted-foreground">{description}</p>
          )}
        </div>
        {action && <div className="shrink-0">{action}</div>}
      </div>
      <Separator className="my-6" />
    </>
  );
}

/* -- Data preview table (shadcn <Table>) ---------------------------- */

function DataPreviewTable({
  columns,
  rows,
}: {
  columns: string[];
  rows: Record<string, unknown>[];
}) {
  return (
    <div className="max-h-[420px] overflow-auto rounded-lg border border-border">
      <Table>
        <TableHeader className="sticky top-0 bg-muted/80 backdrop-blur">
          <TableRow className="hover:bg-transparent">
            {columns.map((col) => (
              <TableHead
                key={col}
                className="h-9 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"
              >
                {col}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row, i) => (
            <TableRow key={i}>
              {columns.map((col) => (
                <TableCell
                  key={col}
                  className="max-w-[320px] truncate text-xs text-foreground"
                  title={String(row[col] ?? "")}
                >
                  {String(row[col] ?? "")}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

/* -- Main Screen ---------------------------------------------------- */

export default function DatasetGeneratorScreen() {
  const [subTab, setSubTab] = useState<SubTab>("generate");

  // Generate tab state
  const [configs, setConfigs] = useState<DatasetGenConfig[]>([]);
  const [jobs, setJobs] = useState<DatasetGenJob[]>([]);
  const [selectedConfig, setSelectedConfig] = useState<string | null>(null);
  const [editYaml, setEditYaml] = useState("");
  const [editDirty, setEditDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveOk, setSaveOk] = useState(false);
  const [outputName, setOutputName] = useState("");
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);

  // Datasets tab state
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [preview, setPreview] = useState<{ columns: string[]; rows: Record<string, unknown>[]; total_rows: number } | null>(null);
  const [previewFile, setPreviewFile] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  // Load configs
  const loadConfigs = useCallback(async () => {
    try { setConfigs(await fetchDatasetGenConfigs()); } catch { setConfigs([]); }
  }, []);

  // Load jobs
  const loadJobs = useCallback(async () => {
    try { setJobs(await fetchDatasetGenJobs()); } catch { setJobs([]); }
  }, []);

  // Load datasets
  const loadDatasets = useCallback(async () => {
    try {
      setDatasetsLoading(true);
      setDatasets(await fetchDatasetList());
    } catch { setDatasets([]); }
    finally { setDatasetsLoading(false); }
  }, []);

  // Initial load
  useEffect(() => { loadConfigs(); loadJobs(); }, [loadConfigs, loadJobs]);

  // Poll jobs every 5s when on Generate tab
  useEffect(() => {
    if (subTab !== "generate") return;
    const interval = setInterval(loadJobs, 5000);
    return () => clearInterval(interval);
  }, [subTab, loadJobs]);

  // Load datasets when switching to Datasets tab
  useEffect(() => {
    if (subTab === "datasets") loadDatasets();
  }, [subTab, loadDatasets]);

  // Sync editYaml when selectedConfig changes
  useEffect(() => {
    const cfg = configs.find((c) => c.filename === selectedConfig);
    if (cfg) {
      setEditYaml(yaml.dump(cfg.content, { indent: 2, lineWidth: 120 }));
      setEditDirty(false);
      setSaveOk(false);
      const stem = cfg.filename.replace(/\.yaml$/, "");
      setOutputName(`${stem}_${Date.now()}`);
      setLaunchError(null);
    }
  }, [selectedConfig, configs]);

  const handleSaveConfig = async () => {
    if (!selectedConfig) return;
    try {
      setSaving(true);
      setSaveOk(false);
      await saveDatasetGenConfig(selectedConfig, editYaml);
      setSaveOk(true);
      setEditDirty(false);
      loadConfigs();
      setTimeout(() => setSaveOk(false), 2000);
    } catch { /* ignore */ } finally {
      setSaving(false);
    }
  };

  const handleLaunch = async () => {
    if (!selectedConfig || !outputName.trim()) return;
    try {
      setLaunching(true);
      setLaunchError(null);
      await launchDatasetGeneration(selectedConfig, outputName.trim());
      loadJobs();
    } catch (e) {
      setLaunchError(e instanceof Error ? e.message : String(e));
    } finally {
      setLaunching(false);
    }
  };

  const handleKillJob = async (name: string) => {
    try {
      await killDatasetGenJob(name);
      loadJobs();
    } catch { /* ignore */ }
  };

  const handlePreview = async (datasetName: string, file: string) => {
    try {
      setPreviewLoading(true);
      setPreviewFile(file);
      const data = await fetchDatasetPreview(datasetName, file);
      setPreview(data);
    } catch { setPreview(null); }
    finally { setPreviewLoading(false); }
  };

  const selectedDatasetInfo = useMemo(
    () => datasets.find((d) => d.name === selectedDataset),
    [datasets, selectedDataset],
  );

  const selectedConfigData = useMemo(
    () => configs.find((c) => c.filename === selectedConfig),
    [configs, selectedConfig],
  );

  return (
    <div className="flex flex-1 flex-col bg-background">
      <PageHeader>
        <PageHeaderTitle icon={Database} title="Dataset" />
        <PillTabs
          tabs={[
            { key: "generate", label: "Generate" },
            { key: "datasets", label: "Datasets" },
          ]}
          active={subTab}
          onSelect={setSubTab}
          actionKey="generate"
        />
      </PageHeader>

      {/* -- Generate tab -- */}
      {subTab === "generate" && (
        <div className="flex flex-1 overflow-hidden">
          {/* Left: config list */}
          <SidePanel>
            <SidePanelHeader>
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Configs · {configs.length}
              </span>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={loadConfigs}
                aria-label="Reload configs"
                className="size-6"
              >
                <RefreshCw className="size-3 text-muted-foreground" />
              </Button>
            </SidePanelHeader>
            <div className="flex-1 overflow-auto pt-1 pb-2">
              {configs.map((c) => {
                const isActive = selectedConfig === c.filename;
                return (
                  <SideItem
                    key={c.filename}
                    active={isActive}
                    onClick={() => setSelectedConfig(c.filename)}
                  >
                    <SideItemDot active={isActive} />
                    <div className="flex min-w-0 flex-1 flex-col">
                      <span
                        className={cn(
                          "truncate text-xs",
                          isActive
                            ? "font-semibold text-foreground"
                            : "font-medium",
                        )}
                      >
                        {c.filename}
                      </span>
                      <span className="truncate text-[10px] text-muted-foreground/70">
                        {c.strategy}
                      </span>
                    </div>
                  </SideItem>
                );
              })}
              {configs.length === 0 && (
                <p className="px-4 py-3 text-xs text-muted-foreground">
                  No configs found. Configure the directory in Settings.
                </p>
              )}
            </div>
          </SidePanel>

          {/* Right: detail */}
          <div className="flex-1 overflow-auto">
            <div className="mx-auto max-w-[1100px] px-8 pt-8 pb-20 space-y-10">
              {!selectedConfigData ? (
                <div className="flex h-[40vh] items-center justify-center">
                  <p className="text-sm text-muted-foreground">
                    Select a config on the left to view and launch
                  </p>
                </div>
              ) : (
                <section>
                  <SectionHeader
                    title={selectedConfigData.filename}
                    description="Edit the YAML and launch a generation job — the output name below names the destination folder."
                    action={
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary">
                          {selectedConfigData.strategy}
                        </Badge>
                        {saveOk && (
                          <span className="text-xs text-emerald-600">
                            Saved
                          </span>
                        )}
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={handleSaveConfig}
                          disabled={!editDirty || saving}
                        >
                          {saving ? (
                            <Loader2 className="size-4 animate-spin" />
                          ) : (
                            "Save"
                          )}
                        </Button>
                      </div>
                    }
                  />

                  {/* Run prompt — notebook-style command line that names the
                      output folder and launches the job. Sits inline above the
                      editor so Launch is always reachable without scrolling. */}
                  <div className="group flex items-stretch overflow-hidden rounded-md border border-border focus-within:border-ring focus-within:ring-2 focus-within:ring-ring/40">
                    <span className="flex shrink-0 items-center gap-1.5 border-r border-border bg-muted/60 px-3 font-mono text-xs font-medium text-muted-foreground">
                      <Play className="size-3 text-primary" />
                      generate&nbsp;→
                    </span>
                    <Input
                      id="output-name"
                      type="text"
                      value={outputName}
                      onChange={(e) => setOutputName(e.target.value)}
                      placeholder="output_folder_name"
                      className="flex-1 rounded-none border-0 bg-transparent px-3 font-mono text-[13px] shadow-none focus-visible:ring-0"
                    />
                    <Button
                      type="button"
                      onClick={handleLaunch}
                      disabled={launching || !outputName.trim()}
                      className="rounded-none border-0"
                    >
                      {launching ? (
                        <Loader2 className="size-4 animate-spin" />
                      ) : (
                        <>
                          <Play className="size-[13px]" />
                          Launch
                        </>
                      )}
                    </Button>
                  </div>
                  {launchError && (
                    <p className="mt-2 line-clamp-2 text-xs text-destructive">
                      {launchError}
                    </p>
                  )}

                  {/* Fixed height — ConfigEditor is now CodeMirror-backed
                      and needs a definite height to enable its internal
                      scroller. 480px is a comfortable section size; the
                      editor scrolls internally for long YAML. */}
                  <div className="mt-4 h-[480px] overflow-hidden rounded-md border border-border">
                    <ConfigEditor
                      value={editYaml}
                      onChangeText={(t) => {
                        setEditYaml(t);
                        setEditDirty(true);
                        setSaveOk(false);
                      }}
                    />
                  </div>
                </section>
              )}

              {/* Jobs section */}
              <section>
                <SectionHeader
                  title="Jobs"
                  description="Recent generation runs. Polls every 5 s while this tab is open."
                  action={
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      onClick={loadJobs}
                      aria-label="Reload jobs"
                      className="size-8"
                    >
                      <RefreshCw className="size-4 text-muted-foreground" />
                    </Button>
                  }
                />
                {jobs.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    No generation jobs yet.
                  </p>
                ) : (
                  <ul className="divide-y divide-border">
                    {jobs.map((job) => (
                      <li
                        key={job.name}
                        className="flex items-center gap-3 py-3"
                      >
                        <StatusPill status={job.status} showLabel={false} />
                        <div className="flex min-w-0 flex-1 flex-col">
                          <div className="flex items-center gap-2">
                            <span className="truncate text-sm font-medium text-foreground">
                              {job.name}
                            </span>
                            <Badge variant="secondary" className="text-[10px]">
                              {job.strategy}
                            </Badge>
                          </div>
                          <span className="truncate text-xs text-muted-foreground">
                            {job.pid ? `PID ${job.pid} · ` : ""}
                            {job.elapsed_seconds > 0
                              ? `${formatElapsed(job.elapsed_seconds)} · `
                              : ""}
                            {job.qa_rows > 0
                              ? `${job.qa_rows} QA · ${job.corpus_rows} corpus`
                              : job.timestamp
                                  ?.slice(0, 19)
                                  .replace("T", " ")}
                          </span>
                          {job.status === "failed" && job.error && (
                            <span className="line-clamp-2 text-xs text-destructive">
                              {job.error}
                            </span>
                          )}
                        </div>
                        <StatusPill status={job.status} />
                        {job.status === "running" && (
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            onClick={() => handleKillJob(job.name)}
                            aria-label="Kill job"
                            className="size-8 hover:bg-destructive/10"
                          >
                            <Square className="size-3 fill-destructive text-destructive" />
                          </Button>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </div>
          </div>
        </div>
      )}

      {/* -- Datasets tab -- */}
      {subTab === "datasets" && (
        <div className="flex flex-1 overflow-hidden">
          {/* Left: dataset folder list */}
          <SidePanel>
            <SidePanelHeader>
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Datasets · {datasets.length}
              </span>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={loadDatasets}
                aria-label="Reload datasets"
                className="size-6"
              >
                <RefreshCw className="size-3 text-muted-foreground" />
              </Button>
            </SidePanelHeader>
            <div className="flex-1 overflow-auto pt-1 pb-2">
              {datasetsLoading ? (
                <div className="flex items-center justify-center p-5">
                  <Loader2 className="size-4 animate-spin text-muted-foreground" />
                </div>
              ) : datasets.length === 0 ? (
                <p className="px-4 py-3 text-xs text-muted-foreground">
                  No datasets found.
                </p>
              ) : (
                datasets.map((d) => {
                  const isActive = selectedDataset === d.name;
                  return (
                    <SideItem
                      key={d.name}
                      active={isActive}
                      onClick={() => {
                        setSelectedDataset(d.name);
                        setPreview(null);
                        setPreviewFile(null);
                      }}
                    >
                      <SideItemDot active={isActive} />
                      <div className="flex min-w-0 flex-1 flex-col">
                        <span
                          className={cn(
                            "truncate text-xs",
                            isActive
                              ? "font-semibold text-foreground"
                              : "font-medium",
                          )}
                        >
                          {d.name}
                        </span>
                        <span className="truncate text-[10px] text-muted-foreground/70">
                          {d.parquet_files.length} file
                          {d.parquet_files.length !== 1 ? "s" : ""}
                          {d.generation_strategy
                            ? ` · ${d.generation_strategy}`
                            : ""}
                        </span>
                      </div>
                      {d.generation_status && (
                        <StatusPill
                          status={d.generation_status}
                          showLabel={false}
                        />
                      )}
                    </SideItem>
                  );
                })
              )}
            </div>
          </SidePanel>

          {/* Right: dataset detail */}
          <div className="flex-1 overflow-auto">
            <div className="mx-auto max-w-[1100px] px-8 pt-8 pb-20 space-y-10">
              {!selectedDatasetInfo ? (
                <div className="flex h-[40vh] items-center justify-center">
                  <p className="text-sm text-muted-foreground">
                    Select a dataset on the left to view details
                  </p>
                </div>
              ) : (
                <>
                  <section>
                    <SectionHeader
                      title={selectedDatasetInfo.name}
                      description={
                        selectedDatasetInfo.generation_strategy
                          ? `Generated with strategy ${selectedDatasetInfo.generation_strategy}.`
                          : undefined
                      }
                      action={
                        <div className="flex items-center gap-2">
                          {selectedDatasetInfo.generation_strategy && (
                            <Badge variant="secondary">
                              {selectedDatasetInfo.generation_strategy}
                            </Badge>
                          )}
                          {selectedDatasetInfo.generation_status && (
                            <StatusPill
                              status={selectedDatasetInfo.generation_status}
                            />
                          )}
                        </div>
                      }
                    />
                    <h3 className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                      Parquet Files
                    </h3>
                    <ul className="divide-y divide-border">
                      {selectedDatasetInfo.parquet_files.map((f) => {
                        const active = previewFile === f.filename;
                        return (
                          <li key={f.filename}>
                            <button
                              type="button"
                              onClick={() =>
                                handlePreview(
                                  selectedDatasetInfo.name,
                                  f.filename,
                                )
                              }
                              className={cn(
                                "flex w-full items-center gap-3 py-2.5 text-left outline-none transition-colors",
                                "hover:text-foreground focus-visible:bg-accent/50",
                                active
                                  ? "text-foreground"
                                  : "text-muted-foreground",
                              )}
                            >
                              <FileText
                                className={cn(
                                  "size-4 shrink-0",
                                  active
                                    ? "text-primary"
                                    : "text-muted-foreground",
                                )}
                              />
                              <span className="flex-1 truncate text-sm font-medium">
                                {f.filename}
                              </span>
                              <span className="tabular-nums text-xs text-muted-foreground">
                                {f.rows.toLocaleString()} rows
                              </span>
                              <span className="tabular-nums text-xs text-muted-foreground/70">
                                {formatBytes(f.size_bytes)}
                              </span>
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  </section>

                  {previewLoading && (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 className="size-4 animate-spin text-muted-foreground" />
                    </div>
                  )}
                  {preview && previewFile && (
                    <section>
                      <SectionHeader
                        title="Preview"
                        description={`${previewFile} · ${preview.total_rows.toLocaleString()} total rows · showing first ${preview.rows.length}`}
                        action={
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() => {
                              setPreview(null);
                              setPreviewFile(null);
                            }}
                          >
                            Close
                          </Button>
                        }
                      />
                      <DataPreviewTable
                        columns={preview.columns}
                        rows={preview.rows}
                      />
                    </section>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
