/**
 * QualityStoresScreen — top-level page for the Quality Stores route.
 *
 * Web port of apps/app/(tabs)/quality-stores.tsx, redesigned to drop the
 * heavy Card chrome — uses shadcn typography + Separator pattern (same
 * professional style as Settings).
 */
import React, { useState, useEffect, useCallback } from "react";
import {
  Loader2,
  Archive,
  Play,
  RefreshCw,
  Trash2,
  Square,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { PageHeader, PageHeaderTitle, PageHeaderActions } from "@/components/ui/PageHeader";
import { SidePanel, SidePanelHeader } from "@/components/ui/SidePanel";
import { SideItem, SideItemDot } from "@/components/ui/SideItem";
import { Badge } from "@/components/ui/badge";
import { StatusPill } from "@/components/ui/StatusPill";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Dropdown } from "@/components/selectors/Dropdown";
import { cn } from "@/lib/utils";

import {
  fetchQualityStores,
  fetchQualityStoreDetail,
  fetchQualityStoreConfigs,
  fetchPipelineConfig,
  createQualityStore,
  deleteQualityStore,
  killQualityStore,
  type QualityStoreInfo,
  type QualityStoreConfig,
  type QualityStoreDetail,
} from "@/api/qualityStore";

import {
  fetchRagProjectConfigs,
  fetchDatasets,
  type RagProjectConfig,
  type DatasetFolder,
} from "@/api/ragProject";

/** Shorten "deepeval_context_precision" → "ctx_prec" etc. */
function shortMetricName(name: string): string {
  return name
    .replace("deepeval_", "")
    .replace("contextual_relevancy", "ctx_rel")
    .replace("context_precision", "ctx_prec")
    .replace("context_recall", "ctx_rec")
    .replace("answer_relevancy", "ans_rel")
    .replace("faithfulness", "faith");
}

export default function QualityStoresScreen() {
  const [stores, setStores] = useState<QualityStoreInfo[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<QualityStoreDetail | null>(null);
  const [configs, setConfigs] = useState<QualityStoreConfig[]>([]);
  const [loading, setLoading] = useState(false);

  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [ragConfigs, setRagConfigs] = useState<RagProjectConfig[]>([]);
  const [selectedConfig, setSelectedConfig] = useState("");
  const [datasets, setDatasets] = useState<DatasetFolder[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [corpusFile, setCorpusFile] = useState<string | null>(null);
  const [qaFile, setQaFile] = useState<string | null>(null);
  const [nGrid, setNGrid] = useState("5");
  const [nConfigs, setNConfigs] = useState("50");
  const [strategy, setStrategy] = useState<"exhaustive" | "sobol">("exhaustive");
  const [creating, setCreating] = useState(false);

  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  const [expandedHash, setExpandedHash] = useState<string | null>(null);
  const [pipelineConfig, setPipelineConfig] = useState<Record<string, unknown> | null>(null);
  const [pcLoading, setPcLoading] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const loadStores = useCallback(async () => {
    try { setLoading(true); setStores(await fetchQualityStores()); }
    catch { setStores([]); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { loadStores(); }, [loadStores]);

  useEffect(() => {
    if (!stores.some((s) => s.status === "running")) return;
    const interval = setInterval(loadStores, 5000);
    return () => clearInterval(interval);
  }, [stores, loadStores]);

  useEffect(() => {
    if (!selected) { setDetail(null); setConfigs([]); setExpandedHash(null); setPipelineConfig(null); return; }
    setExpandedHash(null); setPipelineConfig(null);
    (async () => {
      try {
        const [d, c] = await Promise.all([
          fetchQualityStoreDetail(selected),
          fetchQualityStoreConfigs(selected),
        ]);
        setDetail(d);
        setConfigs(c);
      } catch { setDetail(null); setConfigs([]); }
    })();
  }, [selected]);

  useEffect(() => {
    if (!showCreate) return;
    fetchRagProjectConfigs().then(setRagConfigs).catch(() => {});
    fetchDatasets().then(setDatasets).catch(() => {});
  }, [showCreate]);

  const handleCreate = async () => {
    if (!newName.trim() || !selectedConfig || !corpusFile || !qaFile || !selectedDataset) return;
    try {
      setCreating(true);
      await createQualityStore({
        name: newName.trim(),
        config_filename: selectedConfig,
        corpus_data: `${selectedDataset}/${corpusFile}`,
        qa_data: `${selectedDataset}/${qaFile}`,
        n_grid: parseInt(nGrid) || 5,
        n_configs: parseInt(nConfigs) || 50,
        strategy,
      });
      setShowCreate(false);
      setNewName("");
      setSelectedConfig("");
      setSelectedDataset(null);
      setCorpusFile(null);
      setQaFile(null);
      loadStores();
    } catch { /* ignore */ }
    finally { setCreating(false); }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      setDeleting(true);
      await deleteQualityStore(deleteTarget);
      if (selected === deleteTarget) setSelected(null);
      setDeleteTarget(null);
      loadStores();
    } catch { setDeleteTarget(null); }
    finally { setDeleting(false); }
  };

  const selectedDs = datasets.find((d) => d.name === selectedDataset);

  const togglePipelineConfig = async (hash: string) => {
    if (expandedHash === hash) {
      setExpandedHash(null);
      setPipelineConfig(null);
      return;
    }
    if (!selected) return;
    try {
      setPcLoading(true);
      setExpandedHash(hash);
      const pc = await fetchPipelineConfig(selected, hash);
      setPipelineConfig(pc);
    } catch { setPipelineConfig(null); }
    finally { setPcLoading(false); }
  };

  return (
    <div className="flex flex-1 min-w-0 flex-col bg-background">
      {/* Header */}
      <PageHeader>
        <PageHeaderTitle icon={Archive} title="Quality Stores" />
        <PageHeaderActions>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={loadStores}
            aria-label="Reload"
            className="size-8"
          >
            <RefreshCw className="size-4 text-muted-foreground" />
          </Button>
          <Button type="button" size="sm" onClick={() => setShowCreate(!showCreate)}>
            {showCreate ? "Cancel" : "New Store"}
          </Button>
        </PageHeaderActions>
      </PageHeader>

      <div className="flex flex-1 min-w-0 overflow-hidden">
        {/* ── Left: store list ──────────────────────────────── */}
        <SidePanel className="w-[220px]">
          <SidePanelHeader>
            <span className="text-[10px] font-semibold tracking-[1px] text-muted-foreground">
              STORES ({stores.length})
            </span>
          </SidePanelHeader>
          <div className="flex-1 overflow-auto">
            {loading && stores.length === 0 ? (
              <div className="flex items-center justify-center p-5">
                <Loader2 className="size-4 animate-spin text-muted-foreground" />
              </div>
            ) : stores.length === 0 ? (
              <div className="flex items-center justify-center p-7">
                <span className="text-center text-xs text-muted-foreground/60">No quality stores yet.</span>
              </div>
            ) : (
              stores.map((s) => {
                const isActive = selected === s.name;
                return (
                  <SideItem key={s.name} active={isActive} onClick={() => { setSelected(s.name); setShowCreate(false); }}>
                    <SideItemDot active={isActive} />
                    <div className="flex min-w-0 flex-1 flex-col gap-[3px]">
                      <span className={cn("truncate text-xs", isActive ? "font-semibold text-foreground" : "font-normal text-muted-foreground")}>
                        {s.name}
                      </span>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] text-muted-foreground">{s.n_configs} configs</span>
                        <span className="text-[10px] text-muted-foreground/60">{s.strategy}</span>
                      </div>
                    </div>
                    <StatusPill status={s.status} showLabel={false} />
                  </SideItem>
                );
              })
            )}
          </div>
        </SidePanel>

        {/* ── Right: detail, create form, or empty state ───── */}
        <div className="flex-1 min-w-0 overflow-auto">
          {showCreate ? (
            <CreateForm
              newName={newName}
              setNewName={setNewName}
              ragConfigs={ragConfigs}
              selectedConfig={selectedConfig}
              setSelectedConfig={setSelectedConfig}
              datasets={datasets}
              selectedDataset={selectedDataset}
              setSelectedDataset={setSelectedDataset}
              selectedDs={selectedDs}
              corpusFile={corpusFile}
              setCorpusFile={setCorpusFile}
              qaFile={qaFile}
              setQaFile={setQaFile}
              strategy={strategy}
              setStrategy={setStrategy}
              nGrid={nGrid}
              setNGrid={setNGrid}
              nConfigs={nConfigs}
              setNConfigs={setNConfigs}
              creating={creating}
              onSubmit={handleCreate}
            />
          ) : !selected || !detail ? (
            <div className="flex h-full items-center justify-center p-10">
              <div className="flex flex-col items-center gap-3 text-center">
                <Archive className="size-10 text-muted-foreground/40" />
                <p className="text-sm text-muted-foreground">
                  Select a quality store to view details
                </p>
              </div>
            </div>
          ) : (
            <DetailView
              detail={detail}
              configs={configs}
              expandedHash={expandedHash}
              pipelineConfig={pipelineConfig}
              pcLoading={pcLoading}
              onTogglePipeline={togglePipelineConfig}
              onKill={() =>
                killQualityStore(detail.name).then(loadStores).catch(() => {})
              }
              onDelete={() => setDeleteTarget(detail.name)}
            />
          )}
        </div>
      </div>

      <ConfirmDialog
        open={!!deleteTarget}
        onOpenChange={() => setDeleteTarget(null)}
        title="Delete Quality Store"
        description={`Delete "${deleteTarget}"? All evaluated quality data will be removed permanently.`}
        confirmLabel="Delete"
        onConfirm={handleDelete}
        loading={deleting}
        destructive
      />
    </div>
  );
}

// ── Create form ───────────────────────────────────────────────────────

interface CreateFormProps {
  newName: string;
  setNewName: (v: string) => void;
  ragConfigs: RagProjectConfig[];
  selectedConfig: string;
  setSelectedConfig: (v: string) => void;
  datasets: DatasetFolder[];
  selectedDataset: string | null;
  setSelectedDataset: (v: string | null) => void;
  selectedDs: DatasetFolder | undefined;
  corpusFile: string | null;
  setCorpusFile: (v: string | null) => void;
  qaFile: string | null;
  setQaFile: (v: string | null) => void;
  strategy: "exhaustive" | "sobol";
  setStrategy: (v: "exhaustive" | "sobol") => void;
  nGrid: string;
  setNGrid: (v: string) => void;
  nConfigs: string;
  setNConfigs: (v: string) => void;
  creating: boolean;
  onSubmit: () => void;
}

function CreateForm(p: CreateFormProps) {
  const ready =
    p.newName.trim() && p.selectedConfig && p.corpusFile && p.qaFile && !p.creating;
  return (
    <div className="mx-auto max-w-[640px] px-8 pt-10 pb-20">
      <div className="mb-6 space-y-1">
        <h2 className="text-xl font-semibold tracking-tight">
          Create Quality Store
        </h2>
        <p className="text-sm text-muted-foreground">
          Build a quality store by sweeping pipeline configs against a dataset.
        </p>
      </div>
      <Separator className="my-6" />

      <div className="space-y-6">
        <div className="space-y-2">
          <Label>Store Name</Label>
          <Input
            type="text"
            value={p.newName}
            onChange={(e) => p.setNewName(e.target.value)}
            placeholder="e.g. hotpotqa_base"
            className="font-mono text-sm"
          />
        </div>

        <div className="space-y-2">
          <Label>Config Template</Label>
          <div className="relative z-50">
            <Dropdown
              options={p.ragConfigs.map((c) => ({ key: c.filename, label: c.filename }))}
              selectedKey={p.selectedConfig}
              onSelect={p.setSelectedConfig}
              placeholder="Select config..."
            />
          </div>
        </div>

        <div className="space-y-2">
          <Label>Dataset</Label>
          <div className="relative z-40 flex flex-wrap items-center gap-2.5">
            <div className="flex min-w-[160px] flex-1 flex-col">
              <Dropdown
                options={p.datasets.map((d) => ({ key: d.name, label: d.name }))}
                selectedKey={p.selectedDataset ?? ""}
                onSelect={(k) => {
                  p.setSelectedDataset(k);
                  p.setCorpusFile(null);
                  p.setQaFile(null);
                }}
                placeholder="Dataset..."
              />
            </div>
            {p.selectedDs && (
              <>
                <div className="relative z-30 flex min-w-[140px] flex-1 flex-col">
                  <Dropdown
                    options={p.selectedDs.parquet_files.map((f) => ({ key: f, label: f }))}
                    selectedKey={p.corpusFile ?? ""}
                    onSelect={p.setCorpusFile}
                    placeholder="Corpus..."
                  />
                </div>
                <div className="relative z-30 flex min-w-[140px] flex-1 flex-col">
                  <Dropdown
                    options={p.selectedDs.parquet_files.map((f) => ({ key: f, label: f }))}
                    selectedKey={p.qaFile ?? ""}
                    onSelect={p.setQaFile}
                    placeholder="QA..."
                  />
                </div>
              </>
            )}
          </div>
        </div>

        <div className="space-y-2">
          <Label>Strategy</Label>
          <div className="flex flex-wrap items-end gap-6">
            <div
              role="radiogroup"
              className="inline-flex rounded-md border border-border p-1"
            >
              {(["exhaustive", "sobol"] as const).map((v) => (
                <button
                  key={v}
                  type="button"
                  role="radio"
                  aria-checked={p.strategy === v}
                  onClick={() => p.setStrategy(v)}
                  className={cn(
                    "rounded-sm px-3 py-1 text-xs font-medium transition-colors",
                    p.strategy === v
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {v === "exhaustive" ? "Exhaustive" : "Sobol"}
                </button>
              ))}
            </div>
            {p.strategy === "exhaustive" ? (
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">
                  Grid Points
                </Label>
                <Input
                  type="number"
                  value={p.nGrid}
                  onChange={(e) => p.setNGrid(e.target.value)}
                  className="w-24 text-center font-mono text-sm"
                />
                <p className="text-[11px] text-muted-foreground">per continuous param</p>
              </div>
            ) : (
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">
                  Num Configs
                </Label>
                <Input
                  type="number"
                  value={p.nConfigs}
                  onChange={(e) => p.setNConfigs(e.target.value)}
                  className="w-24 text-center font-mono text-sm"
                />
                <p className="text-[11px] text-muted-foreground">quasi-random samples</p>
              </div>
            )}
          </div>
        </div>

        <div className="flex justify-end pt-2">
          <Button
            type="button"
            onClick={p.onSubmit}
            disabled={!ready}
          >
            {p.creating ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <>
                <Play className="size-[13px] fill-current" />
                Create &amp; Run Sweep
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}

// ── Detail view ───────────────────────────────────────────────────────

interface DetailViewProps {
  detail: QualityStoreDetail;
  configs: QualityStoreConfig[];
  expandedHash: string | null;
  pipelineConfig: Record<string, unknown> | null;
  pcLoading: boolean;
  onTogglePipeline: (hash: string) => void;
  onKill: () => void;
  onDelete: () => void;
}

function DetailView({
  detail,
  configs,
  expandedHash,
  pipelineConfig,
  pcLoading,
  onTogglePipeline,
  onKill,
  onDelete,
}: DetailViewProps) {
  return (
    <div className="mx-auto max-w-[1200px] px-8 pt-10 pb-20">
      {/* Title + status + actions */}
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2.5">
            <h2 className="text-xl font-semibold tracking-tight">{detail.name}</h2>
            <StatusPill status={detail.status ?? "unknown"} />
          </div>
          <p className="font-mono text-xs text-muted-foreground/80 truncate max-w-[600px]">
            {detail.store_path}
          </p>
        </div>
        <div className="flex shrink-0 gap-1.5">
          {detail.status === "running" && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={onKill}
              className="size-6 hover:bg-destructive/10"
            >
              <Square className="size-2.5 fill-destructive text-destructive" />
            </Button>
          )}
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={onDelete}
            className="size-6 hover:bg-destructive/10"
          >
            <Trash2 className="size-[13px] text-muted-foreground" />
          </Button>
        </div>
      </div>

      <Separator className="my-6" />

      {/* Stats — flat key-value, no card chrome */}
      <dl className="grid grid-cols-3 gap-x-8 gap-y-1 text-sm">
        <div>
          <dt className="text-xs uppercase tracking-wider text-muted-foreground">
            Configs
          </dt>
          <dd className="mt-1 text-2xl font-semibold tabular-nums">
            {detail.n_configs}
          </dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wider text-muted-foreground">
            Failed
          </dt>
          <dd
            className={cn(
              "mt-1 text-2xl font-semibold tabular-nums",
              detail.n_failed ? "text-destructive" : "text-foreground",
            )}
          >
            {detail.n_failed ?? 0}
          </dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wider text-muted-foreground">
            Strategy
          </dt>
          <dd className="mt-2 flex items-center gap-2">
            <Badge variant="secondary" className="capitalize">
              {detail.strategy}
            </Badge>
            {detail.n_grid != null && (
              <span className="font-mono text-xs text-muted-foreground">
                n_grid={detail.n_grid}
              </span>
            )}
          </dd>
        </div>
      </dl>

      <Separator className="my-6" />

      {/* Configs table */}
      <div className="space-y-3">
        <div className="flex items-baseline justify-between">
          <h3 className="text-sm font-semibold">Evaluated Configs</h3>
          <span className="text-xs text-muted-foreground">
            {configs.length} total
          </span>
        </div>
        {configs.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No configs evaluated yet.
          </p>
        ) : (
          <ConfigsTable
            configs={configs}
            expandedHash={expandedHash}
            pipelineConfig={pipelineConfig}
            pcLoading={pcLoading}
            onToggle={onTogglePipeline}
          />
        )}
      </div>
    </div>
  );
}

// ── Configs table (raw HTML table for column-alignment fidelity) ───────

interface ConfigsTableProps {
  configs: QualityStoreConfig[];
  expandedHash: string | null;
  pipelineConfig: Record<string, unknown> | null;
  pcLoading: boolean;
  onToggle: (hash: string) => void;
}

function ConfigsTable({
  configs,
  expandedHash,
  pipelineConfig,
  pcLoading,
  onToggle,
}: ConfigsTableProps) {
  const sorted = [...configs].sort(
    (a, b) => b.quality_combined - a.quality_combined,
  );
  const metricKeys = configs[0] ? Object.keys(configs[0].quality_metrics) : [];
  const nCols = 3 + metricKeys.length;

  return (
    <div className="max-h-[calc(100vh-340px)] overflow-auto rounded-md border border-border">
      <table className="w-full border-collapse text-xs">
        <thead className="sticky top-0 z-[1] bg-muted/50 backdrop-blur">
          <tr>
            <th className="border-b border-border px-3 py-2 text-left text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              #
            </th>
            <th className="border-b border-border px-3 py-2 text-left text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Hash
            </th>
            <th className="border-b border-border px-3 py-2 text-right text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Quality
            </th>
            {metricKeys.map((m) => (
              <th
                key={m}
                className="border-b border-border px-3 py-2 text-right text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"
              >
                {shortMetricName(m)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((c, i) => {
            const isExpanded = expandedHash === c.config_hash;
            return (
              <React.Fragment key={c.config_hash}>
                <tr
                  onClick={() => onToggle(c.config_hash)}
                  className={cn(
                    "cursor-pointer border-b border-border transition-colors hover:bg-muted/40",
                    isExpanded && "bg-primary/5",
                    !isExpanded && i % 2 === 1 && "bg-muted/20",
                  )}
                >
                  <td className="px-3 py-1.5 text-muted-foreground tabular-nums">
                    {i + 1}
                  </td>
                  <td className="px-3 py-1.5 font-mono">
                    {c.config_hash.slice(0, 8)}
                    <span
                      className={cn(
                        "ml-1.5 text-[9px]",
                        isExpanded ? "text-primary" : "text-muted-foreground/60",
                      )}
                    >
                      {isExpanded ? "▲" : "▼"}
                    </span>
                  </td>
                  <td className="px-3 py-1.5 text-right font-semibold text-primary tabular-nums">
                    {c.quality_combined.toFixed(4)}
                  </td>
                  {metricKeys.map((m) => {
                    const v = c.quality_metrics[m];
                    return (
                      <td
                        key={m}
                        className="px-3 py-1.5 text-right tabular-nums"
                      >
                        {v != null && !Number.isNaN(v as number)
                          ? (v as number).toFixed(4)
                          : "—"}
                      </td>
                    );
                  })}
                </tr>
                {isExpanded && (
                  <tr>
                    <td
                      colSpan={nCols}
                      className="border-b border-border bg-muted/40 p-0"
                    >
                      <div className="max-h-[400px] overflow-auto px-4 py-3">
                        {pcLoading ? (
                          <span className="text-xs text-muted-foreground">
                            Loading pipeline config...
                          </span>
                        ) : pipelineConfig ? (
                          <pre className="m-0 whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-foreground">
                            {JSON.stringify(pipelineConfig, null, 2)}
                          </pre>
                        ) : (
                          <span className="text-xs text-muted-foreground">
                            Pipeline config not available
                          </span>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
