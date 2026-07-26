/**
 * PipelineEditorContainer — Optimization Configs page.
 * Sidebar with config file list + CRUD, main area uses ConfigVisualEditor.
 */
import React, { useState, useCallback, useEffect } from "react";
import {
  GitBranch, FileText, Plus, Copy, Trash2,
  RefreshCw, X, Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { PageHeader, PageHeaderTitle } from "@/components/ui/PageHeader";
import { SidePanel, SidePanelHeader } from "@/components/ui/SidePanel";
import { SideItem } from "@/components/ui/SideItem";
import { Badge } from "@/components/ui/badge";
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
import { useNavigate } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import yaml from "js-yaml";

import { useProject } from "@/context/ProjectContext";
import { ConfigVisualEditor } from "./ConfigVisualEditor";

import {
  fetchRagProjectConfigs,
  saveRagTemplateConfig,
  deleteRagConfig,
  duplicateRagConfig,
  createRagProject,
  scaffoldConfig,
  type RagProjectConfig,
} from "@/api/costModelClient";


export function PipelineEditorContainer() {
  const { setCurrentProject, refreshProjects } = useProject();
  const navigate = useNavigate();

  // Config list
  const [configs, setConfigs] = useState<RagProjectConfig[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [configLoading, setConfigLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // CRUD state
  const [showNewInput, setShowNewInput] = useState(false);
  const [newFilename, setNewFilename] = useState("");
  const [dupFilename, setDupFilename] = useState("");
  const [showDup, setShowDup] = useState(false);

  // Create project state
  const [showCreateProject, setShowCreateProject] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [creatingProject, setCreatingProject] = useState(false);

  // Select a config
  const selectConfig = useCallback((filename: string) => {
    setSelectedFile(filename);
    setShowCreateProject(false);
  }, []);

  // Load configs
  const loadConfigs = useCallback(async (autoSelect?: string) => {
    try {
      setConfigLoading(true);
      const data = await fetchRagProjectConfigs();
      setConfigs(data);
      const target = autoSelect || (data.length > 0 ? data[0].filename : null);
      if (target) selectConfig(target);
    } catch {
      setConfigs([]);
    } finally {
      setConfigLoading(false);
    }
  }, [selectConfig]);

  useEffect(() => { loadConfigs(); }, []);

  // Save handler for ConfigVisualEditor
  const handleSave = useCallback(async (yamlText: string) => {
    if (!selectedFile) throw new Error("No config selected");
    await saveRagTemplateConfig(selectedFile, yamlText);
    await loadConfigs(selectedFile);
  }, [selectedFile, loadConfigs]);

  // Create project
  const handleCreateProject = useCallback(async () => {
    if (!selectedFile || !projectName.trim()) return;
    setCreatingProject(true);
    setError(null);
    try {
      await createRagProject(projectName.trim(), selectedFile);
      await refreshProjects();
      setCurrentProject(projectName.trim());
      setShowCreateProject(false);
      setProjectName("");
      navigate({ to: "/" });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setCreatingProject(false);
    }
  }, [selectedFile, projectName, refreshProjects, setCurrentProject, navigate]);

  // Delete
  const handleDelete = useCallback(async () => {
    if (!selectedFile) return;
    try {
      await deleteRagConfig(selectedFile);
      setSelectedFile(null);
      await loadConfigs();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [selectedFile, loadConfigs]);

  // Duplicate
  const handleDuplicate = useCallback(async () => {
    if (!selectedFile || !dupFilename.trim()) return;
    const fn = dupFilename.trim().endsWith(".yaml") ? dupFilename.trim() : `${dupFilename.trim()}.yaml`;
    try {
      await duplicateRagConfig(selectedFile, fn);
      setShowDup(false);
      setDupFilename("");
      await loadConfigs(fn);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [selectedFile, dupFilename, loadConfigs]);

  // Create new — scaffold a valid 7-section template from the backend (the
  // server is the schema source of truth; no hardcoded template here).
  const handleCreate = useCallback(async () => {
    const fn = newFilename.trim().endsWith(".yaml") ? newFilename.trim() : `${newFilename.trim()}.yaml`;
    if (!fn || fn === ".yaml") return;
    try {
      const { content } = await scaffoldConfig({ evalBackend: "static_gt" });
      await saveRagTemplateConfig(fn, content);
      setShowNewInput(false);
      setNewFilename("");
      await loadConfigs(fn);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [newFilename, loadConfigs]);

  if (configLoading) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3">
        <Loader2 className="size-6 animate-spin text-muted-foreground" />
        <span className="text-[13px] text-muted-foreground">Loading...</span>
      </div>
    );
  }

  const selectedCfg = configs.find((c) => c.filename === selectedFile);

  // Create Project button for toolbar
  const toolbarExtra = selectedFile ? (
    <Button type="button" size="sm" onClick={() => setShowCreateProject(true)}>
      <Plus className="size-3" />
      Create Project
    </Button>
  ) : null;

  return (
    // `h-svh + overflow-hidden` locks THIS screen at viewport height (matching
    // RagOptimizeScreen) so the page itself never scrolls — only the inner
    // config list and the editor's tab body scroll independently.
    <div className="flex h-svh flex-col overflow-hidden bg-background">
      {/* Header */}
      <PageHeader>
        <PageHeaderTitle
          icon={GitBranch}
          title="Optimization Configs"
          badge={
            <Badge variant="secondary" className="text-[11px]">
              {configs.length}
            </Badge>
          }
        />
      </PageHeader>

      {/* Error */}
      {error && (
        <div className="flex items-center justify-between bg-destructive/10 px-4 py-1.5">
          <span className="text-[11px] text-destructive">{error}</span>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={() => setError(null)}
            className="size-5 hover:bg-destructive/15"
          >
            <X className="size-3 text-destructive" />
          </Button>
        </div>
      )}

      <div className="flex flex-1 min-h-0">
        {/* Sidebar */}
        <SidePanel className="w-[240px]">
          <SidePanelHeader>
            <span className="text-[10px] font-semibold uppercase tracking-[1px] text-muted-foreground">CONFIGS</span>
            <div className="flex gap-1">
              <Button type="button" variant="ghost" size="icon" onClick={() => setShowNewInput(!showNewInput)} className="size-6" aria-label="New config">
                <Plus className="size-3 text-muted-foreground" />
              </Button>
              {selectedFile && (
                <>
                  <Button type="button" variant="ghost" size="icon" onClick={() => setShowDup(!showDup)} className="size-6" aria-label="Duplicate">
                    <Copy className="size-3 text-muted-foreground" />
                  </Button>
                  <Button type="button" variant="ghost" size="icon" onClick={handleDelete} className="size-6 hover:bg-destructive/15" aria-label="Delete">
                    <Trash2 className="size-3 text-destructive" />
                  </Button>
                </>
              )}
              <Button type="button" variant="ghost" size="icon" onClick={() => loadConfigs(selectedFile || undefined)} className="size-6" aria-label="Reload">
                <RefreshCw className="size-3 text-muted-foreground" />
              </Button>
            </div>
          </SidePanelHeader>

          {showNewInput && (
            <div className="px-3 pb-2">
              <input
                className="w-full rounded-md border border-border bg-card px-2.5 py-1.5 text-xs text-foreground outline-none"
                value={newFilename}
                onChange={(e) => setNewFilename(e.target.value)}
                placeholder="new_config.yaml"
                onKeyDown={(e) => { if (e.key === "Enter") handleCreate(); }}
              />
            </div>
          )}
          {showDup && (
            <div className="px-3 pb-2">
              <input
                className="w-full rounded-md border border-border bg-card px-2.5 py-1.5 text-xs text-foreground outline-none"
                value={dupFilename}
                onChange={(e) => setDupFilename(e.target.value)}
                placeholder="copy_name.yaml"
                onKeyDown={(e) => { if (e.key === "Enter") handleDuplicate(); }}
              />
            </div>
          )}

          <div className="flex-1 overflow-auto">
          {configs.map((c) => {
            const isActive = selectedFile === c.filename;
            return (
              <SideItem
                key={c.filename}
                active={isActive}
                onClick={() => selectConfig(c.filename)}
              >
                <FileText
                  className={cn(
                    "size-3",
                    isActive ? "text-foreground" : "text-muted-foreground/60",
                  )}
                />
                <div className="flex flex-1 flex-col">
                  <span
                    className={cn(
                      "truncate text-xs",
                      isActive ? "font-semibold text-primary" : "font-normal text-foreground",
                    )}
                  >
                    {c.filename}
                  </span>
                  <span className="text-[10px] text-muted-foreground">{c.optimizer_type || "—"} · {c.n_iterations} iter</span>
                </div>
              </SideItem>
            );
          })}
          </div>
        </SidePanel>

        {/* Main editor */}
        <div className="flex flex-1 flex-col min-h-0 min-w-0">
          {selectedCfg ? (
            <ConfigVisualEditor
              key={selectedFile}
              initialYaml={selectedCfg.raw_content || yaml.dump(selectedCfg.content, { lineWidth: 120, noRefs: true })}
              initialContent={selectedCfg.content as Record<string, unknown>}
              onSave={handleSave}
              toolbarExtra={toolbarExtra}
            />
          ) : (
            <div className="flex flex-1 flex-col items-center justify-center">
              <FileText className="size-8 text-muted-foreground/40" />
              <span className="mt-3 text-sm text-muted-foreground">Select a config to edit</span>
            </div>
          )}
        </div>
      </div>

      {/* Create project dialog — shadcn <Dialog>, replaces the old inline banner */}
      <Dialog
        open={showCreateProject}
        onOpenChange={(open) => {
          if (creatingProject) return;
          setShowCreateProject(open);
          if (!open) setProjectName("");
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Create project</DialogTitle>
            <DialogDescription>
              Spin up a new RAG project from{" "}
              <span className="font-mono text-xs">{selectedFile}</span>.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <Label htmlFor="pe-project-name">Project name</Label>
            <Input
              id="pe-project-name"
              type="text"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              placeholder="my_project"
              className="font-mono text-sm"
              autoFocus
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  projectName.trim() &&
                  !creatingProject
                ) {
                  handleCreateProject();
                }
              }}
            />
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => setShowCreateProject(false)}
              disabled={creatingProject}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={handleCreateProject}
              disabled={creatingProject || !projectName.trim()}
            >
              {creatingProject ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <>
                  <Plus className="size-3.5" />
                  Create &amp; Go
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
