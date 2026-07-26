/**
 * ProjectList — full-page project picker shown when no project is selected.
 * Redesigned around shadcn primitives: <Card> for the new-project form,
 * <Select> for the config template picker, <DropdownMenu> for sort, and a
 * `divide-y border rounded-md` row list (no per-row shadow — modern shadcn
 * data-table aesthetic).
 */
import React, { useState, useEffect } from "react";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  ChevronRight,
  FolderOpen,
  Loader2,
  RefreshCw,
  Trash2,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { StatusPill } from "@/components/ui/StatusPill";
import { cn } from "@/lib/utils";
import {
  isValidOptionalRefineIterations,
  parseOptionalRefineIterations,
} from "@/lib/systemTransfer";

import { useProject, type ProjectSortKey } from "@/context/ProjectContext";
import {
  fetchRagProjectConfigs,
  createRagProject,
  deleteRagProject,
  fetchSystemYamls,
  launchSystemTransferProject,
  type RagProjectConfig,
  type SystemYamlInfo,
} from "@/api/ragProject";
import { fetchQualityStores, type QualityStoreInfo } from "@/api/qualityStore";
import {
  ToggleGroup,
  ToggleGroupItem,
} from "@/components/ui/toggle-group";
import { SystemYamlPicker } from "@/components/rag-optimize/SystemYamlPicker";

interface ProjectListProps {
  showNewProject: boolean;
  setShowNewProject: (v: boolean) => void;
}

const SORT_OPTIONS: { key: ProjectSortKey; label: string }[] = [
  { key: "time", label: "Time" },
  { key: "name", label: "Name" },
  { key: "status", label: "Status" },
  { key: "evaluations", label: "Evaluations" },
];

export function ProjectList({
  showNewProject,
  setShowNewProject,
}: ProjectListProps) {
  const {
    projects,
    setCurrentProject,
    refreshProjects,
    sortKey,
    sortDirection,
    setSortKey,
    setSortDirection,
  } = useProject();
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const [newName, setNewName] = useState("");
  const [configs, setConfigs] = useState<RagProjectConfig[]>([]);
  const [selectedConfig, setSelectedConfig] = useState("");
  const [creating, setCreating] = useState(false);

  // Project kind: raw (start from scratch), migrate from an explicit
  // QualityStore, or transfer a prior project to a different system.
  const [projectKind, setProjectKind] = useState<
    "raw" | "migrate_store" | "system_transfer"
  >("raw");
  const [migrateSource, setMigrateSource] = useState("");
  const [migrateRefineIters, setMigrateRefineIters] = useState("0");
  const [qualityStores, setQualityStores] = useState<QualityStoreInfo[]>([]);
  const [systemYamls, setSystemYamls] = useState<SystemYamlInfo[]>([]);
  const [selectedSystemYaml, setSelectedSystemYaml] = useState("");

  // Existing projects eligible as a migration source (any prior run that
  // could plausibly hold quality data).
  const sourceProjects = projects.filter(
    (p) => p.status === "completed" || p.status === "failed",
  );

  useEffect(() => {
    if (showNewProject && configs.length === 0) {
      fetchRagProjectConfigs().then(setConfigs).catch(() => {});
    }
    if (showNewProject && qualityStores.length === 0) {
      fetchQualityStores()
        .then((s) => setQualityStores(s.filter((st) => st.status === "completed")))
        .catch(() => {});
    }
    if (showNewProject && systemYamls.length === 0) {
      fetchSystemYamls().then(setSystemYamls).catch(() => {});
    }
  }, [showNewProject, configs.length, qualityStores.length, systemYamls.length]);

  // Reset migration source whenever the kind switches, so a stale selection
  // from a prior open doesn't apply to the new kind.
  useEffect(() => {
    setMigrateSource("");
    setSelectedSystemYaml("");
    setMigrateRefineIters(projectKind === "system_transfer" ? "" : "0");
  }, [projectKind]);

  const selectedSystemYamlInfo = systemYamls.find(
    (system) => system.path === selectedSystemYaml,
  );
  const yamlRefineDefault = selectedSystemYamlInfo?.n_refine_iterations ?? 5;
  const transferRefineOverride = parseOptionalRefineIterations(migrateRefineIters);
  const transferRefineValid = isValidOptionalRefineIterations(migrateRefineIters);

  const migrateReady =
    projectKind === "raw" ||
    (projectKind === "system_transfer"
      ? !!migrateSource && !!selectedSystemYaml && transferRefineValid
      : !!migrateSource);

  const handleCreate = async () => {
    if (!newName.trim() || !migrateReady) return;
    if (projectKind !== "system_transfer" && !selectedConfig) return;
    try {
      setCreating(true);
      if (projectKind === "system_transfer") {
        await launchSystemTransferProject({
          projectName: newName.trim(),
          sourceProject: migrateSource,
          systemYaml: selectedSystemYaml,
          nRefineIterations: transferRefineOverride,
        });
      } else {
        await createRagProject(newName.trim(), selectedConfig);
      }
      // Hand off migration intent to the project view via sessionStorage —
      // ProjectConfigContent picks it up on first mount and pre-selects the
      // Migrate toggle + source. One-shot: cleared after consumption.
      if (projectKind !== "raw" && projectKind !== "system_transfer") {
        const intent = {
          source: "store",
          value: migrateSource,
          refineIters: parseInt(migrateRefineIters) || 0,
        };
        sessionStorage.setItem(
          `rag_migrate_intent_${newName.trim()}`,
          JSON.stringify(intent),
        );
      }
      setShowNewProject(false);
      setNewName("");
      setSelectedConfig("");
      setProjectKind("raw");
      setMigrateSource("");
      setMigrateRefineIters("0");
      setSelectedSystemYaml("");
      await refreshProjects();
      setCurrentProject(newName.trim());
    } catch {
      /* ignore */
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      setDeleting(true);
      await deleteRagProject(deleteTarget);
      setDeleteTarget(null);
      refreshProjects();
    } catch {
      setDeleteTarget(null);
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="flex-1 min-w-0 overflow-auto">
      <div className="mx-auto max-w-[760px] px-8 pt-8 pb-20">
        {/* Section header */}
        <div className="mb-4 flex items-end justify-between gap-4">
          <div className="space-y-1">
            <h2 className="text-xl font-semibold tracking-tight">Projects</h2>
            <p className="text-sm text-muted-foreground">
              {projects.length} project{projects.length === 1 ? "" : "s"} ·
              click to open or sort by{" "}
              <span className="font-medium text-foreground capitalize">
                {sortKey}
              </span>
            </p>
          </div>
          <div className="flex items-center gap-1.5">
            <SortMenu
              sortKey={sortKey}
              setSortKey={setSortKey}
              sortDirection={sortDirection}
              setSortDirection={setSortDirection}
            />
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
          </div>
        </div>
        <Separator className="mb-4" />

        {/* Empty state */}
        {projects.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <div className="flex size-14 items-center justify-center rounded-xl bg-muted">
              <FolderOpen className="size-6 text-muted-foreground/60" />
            </div>
            <p className="mt-4 text-sm font-semibold text-foreground">
              No projects yet
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              Click <span className="font-medium">New Project</span> in the
              header to start an optimization run.
            </p>
          </div>
        ) : (
          <ul className="divide-y divide-border rounded-md border border-border bg-card">
            {projects.map((p) => (
              <li
                key={p.name}
                onClick={() => setCurrentProject(p.name)}
                className={cn(
                  "group flex cursor-pointer items-center gap-3 px-4 py-3 transition-colors",
                  "hover:bg-muted/50",
                )}
              >
                <FolderOpen className="size-4 shrink-0 text-muted-foreground" />
                <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <span className="truncate text-sm font-medium text-foreground">
                    {p.name || "root"}
                  </span>
                  <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                    {p.timestamp && (
                      <span className="font-mono">
                        {p.timestamp.slice(0, 16).replace("T", " ")}
                      </span>
                    )}
                    <span className="tabular-nums">
                      {p.n_evaluations ?? 0} evals
                    </span>
                  </div>
                </div>
                <StatusPill status={p.status ?? "created"} showLabel={false} />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={(e) => {
                    e.stopPropagation();
                    setDeleteTarget(p.name);
                  }}
                  aria-label={`Delete ${p.name}`}
                  className="size-7 opacity-0 group-hover:opacity-100 hover:bg-destructive/10"
                >
                  <Trash2 className="size-3.5 text-destructive" />
                </Button>
                <ChevronRight className="size-4 text-muted-foreground/40 group-hover:text-muted-foreground" />
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* New project dialog — shadcn <Dialog>, replaces the old inline Card */}
      <Dialog
        open={showNewProject}
        onOpenChange={(open) => {
          if (creating) return;
          setShowNewProject(open);
          if (!open) {
            setNewName("");
            setSelectedConfig("");
            setProjectKind("raw");
            setMigrateSource("");
            setMigrateRefineIters("0");
            setSelectedSystemYaml("");
          }
        }}
      >
        <DialogContent
          className={cn(
            "max-h-[calc(100vh-2rem)] w-[calc(100vw-2rem)] max-w-[calc(100vw-2rem)] overflow-hidden sm:max-w-lg",
            projectKind === "system_transfer" && "sm:max-w-3xl",
          )}
        >
          <DialogHeader>
            <DialogTitle>Create new project</DialogTitle>
            <DialogDescription>
              {projectKind === "raw"
                ? "Start a fresh RAG optimization run from a config template."
                : projectKind === "system_transfer"
                  ? "Copy a prior project, replace only its system block, then replay and refine on the new system."
                  : "Migrate quality data into a new project on different hardware."}
            </DialogDescription>
          </DialogHeader>
          <div
            className={cn(
              "min-w-0 max-h-[calc(100vh-13rem)] overflow-y-auto overflow-x-hidden pr-1",
              projectKind === "system_transfer"
                ? "grid gap-4 lg:grid-cols-[minmax(16rem,20rem)_minmax(0,1fr)] lg:items-start"
                : "space-y-4",
            )}
          >
            {/* Kind selector */}
            <div className={cn("space-y-2", projectKind === "system_transfer" && "lg:col-span-2")}>
              <Label>Project kind</Label>
              <ToggleGroup
                type="single"
                value={projectKind}
                onValueChange={(v) => {
                  if (!v) return;
                  setProjectKind(v as typeof projectKind);
                }}
                size="sm"
                variant="outline"
                className="flex-wrap justify-start"
              >
                <ToggleGroupItem value="raw">Raw</ToggleGroupItem>
                <ToggleGroupItem value="migrate_store">
                  Migrate from quality store
                </ToggleGroupItem>
                <ToggleGroupItem value="system_transfer">
                  System transfer
                </ToggleGroupItem>
              </ToggleGroup>
            </div>

            <div className={cn("space-y-2", projectKind === "system_transfer" && "lg:col-start-1")}>
              <Label htmlFor="project-name">Project name</Label>
              <Input
                id="project-name"
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="my_first_project"
                className="font-mono text-sm"
                autoFocus
                onKeyDown={(e) => {
                  if (
                    e.key === "Enter" &&
                    newName.trim() &&
                    migrateReady &&
                    (projectKind === "system_transfer" || selectedConfig) &&
                    !creating
                  ) {
                    handleCreate();
                  }
                }}
              />
            </div>

            {projectKind !== "system_transfer" && (
            <div className="space-y-2">
              <Label htmlFor="project-config">Config template</Label>
              <Select
                value={selectedConfig}
                onValueChange={setSelectedConfig}
              >
                <SelectTrigger id="project-config">
                  <SelectValue placeholder="Select a config…" />
                </SelectTrigger>
                <SelectContent>
                  {configs.map((c) => (
                    <SelectItem key={c.filename} value={c.filename}>
                      <span className="font-mono text-xs">{c.filename}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {projectKind !== "raw" && (
                <p className="text-xs text-muted-foreground">
                  Pick a config whose hardware spec is the migration target.
                </p>
              )}
            </div>
            )}

            {/* Source project — only for system transfer */}
            {projectKind === "system_transfer" && (
              <div className="relative z-20 min-w-0 space-y-2 lg:col-start-1">
                <Label htmlFor="migrate-source-project">Source project</Label>
                <Select
                  value={migrateSource}
                  onValueChange={setMigrateSource}
                >
                  <SelectTrigger id="migrate-source-project" className="w-full min-w-0">
                    <SelectValue
                      className="min-w-0 truncate"
                      placeholder={
                        sourceProjects.length
                          ? "Select prior project…"
                          : "No prior projects available"
                      }
                    />
                  </SelectTrigger>
                  <SelectContent className="z-[70] max-w-[calc(100vw-2rem)]">
                    {sourceProjects.map((p) => (
                      <SelectItem key={p.name} value={p.name}>
                        <span className="block max-w-[22rem] truncate font-mono text-xs">
                          {p.name} · {p.status} · {p.n_evaluations ?? 0} evals
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            {projectKind === "system_transfer" && (
              <div className="min-w-0 space-y-2 overflow-hidden lg:col-start-2 lg:row-span-3 lg:row-start-2">
                <Label htmlFor="system-yaml">System YAML</Label>
                <SystemYamlPicker
                  items={systemYamls}
                  value={selectedSystemYaml}
                  onValueChange={setSelectedSystemYaml}
                />
                <p className="text-xs text-muted-foreground">
                  Only the YAML's system block is used; pipeline, quality, dataset, and search space come from the source project.
                </p>
              </div>
            )}

            {projectKind === "migrate_store" && (
              <div className="space-y-2">
                <Label htmlFor="migrate-source-store">Source quality store</Label>
                <Select
                  value={migrateSource}
                  onValueChange={setMigrateSource}
                >
                  <SelectTrigger id="migrate-source-store">
                    <SelectValue
                      placeholder={
                        qualityStores.length
                          ? "Select quality store…"
                          : "No quality stores available"
                      }
                    />
                  </SelectTrigger>
                  <SelectContent>
                    {qualityStores.map((s) => (
                      <SelectItem key={s.name} value={s.name}>
                        <span className="font-mono text-xs">
                          {s.name} ({s.n_configs} configs)
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            {projectKind !== "raw" && (
              <div className={cn("space-y-2", projectKind === "system_transfer" && "lg:col-start-1")}>
                <Label htmlFor="refine-iters">
                  {projectKind === "system_transfer"
                    ? "Refine iterations override"
                    : "Refine iterations"}
                </Label>
                <Input
                  id="refine-iters"
                  type="number"
                  min="0"
                  value={migrateRefineIters}
                  onChange={(e) => setMigrateRefineIters(e.target.value)}
                  placeholder={
                    projectKind === "system_transfer"
                      ? `Use YAML default (${yamlRefineDefault})`
                      : undefined
                  }
                  aria-describedby="refine-iters-help"
                  aria-invalid={
                    projectKind === "system_transfer" && !transferRefineValid
                  }
                  className="font-mono text-sm"
                />
                <p id="refine-iters-help" className="text-xs text-muted-foreground">
                  {projectKind === "system_transfer"
                    ? !transferRefineValid
                      ? "Enter a non-negative whole number, or leave this field blank."
                      : transferRefineOverride === undefined
                      ? `Leave blank to use the YAML default: ${yamlRefineDefault} refinement iterations.`
                      : `Override the YAML default with ${transferRefineOverride} refinement iterations.`
                    : "0 = direct Pareto only. Higher = additional MOBO steps after the initial migration."}
                </p>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => setShowNewProject(false)}
              disabled={creating}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={handleCreate}
              disabled={
                creating ||
                !newName.trim() ||
                (projectKind !== "system_transfer" && !selectedConfig) ||
                !migrateReady
              }
            >
              {creating ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                "Create"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        open={deleteTarget != null}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
        title="Delete project"
        description={`Are you sure you want to delete "${deleteTarget}"? All evaluations, configs, and results will be permanently removed.`}
        confirmLabel="Delete"
        loading={deleting}
        destructive
        onConfirm={handleDelete}
      />
    </div>
  );
}

// ── Sort menu ─────────────────────────────────────────────────────────

interface SortMenuProps {
  sortKey: ProjectSortKey;
  setSortKey: (k: ProjectSortKey) => void;
  sortDirection: "asc" | "desc";
  setSortDirection: (d: "asc" | "desc") => void;
}

function SortMenu({
  sortKey,
  setSortKey,
  sortDirection,
  setSortDirection,
}: SortMenuProps) {
  const currentLabel =
    SORT_OPTIONS.find((o) => o.key === sortKey)?.label ?? "Time";
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button type="button" variant="outline" size="sm" className="gap-1.5">
          <ArrowUpDown className="size-3.5" />
          {currentLabel}
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
