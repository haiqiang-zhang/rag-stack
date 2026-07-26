import { useCallback, useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConfigEditor } from "@/components/ConfigEditor";
import { ConfigVisualEditor, type ConfigEditorState } from "@/components/pipeline-editor/ConfigVisualEditor";
import { fetchProjectConfig, saveProjectConfig, type ProjectConfig } from "@/api/ragProject";

interface ProjectSetupTabProps {
  project: string;
  status?: string;
  onLaunchReadiness: (reason: string | null) => void;
}

/** Edits the project's own draft; the server freezes it on the first launch. */
export function ProjectSetupTab({ project, status, onLaunchReadiness }: ProjectSetupTabProps) {
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  const [editorState, setEditorState] = useState<ConfigEditorState>({ dirty: false, saving: false, valid: null });
  const editable = config?.editable && status === "created";

  useEffect(() => {
    let active = true;
    setConfig(null);
    setError(null);
    fetchProjectConfig(project).then((next) => {
      if (active) setConfig(next);
    }).catch((err) => {
      if (active) setError(err instanceof Error ? err.message : String(err));
    });
    return () => { active = false; };
  }, [project, revision, status]);

  useEffect(() => {
    let reason: string | null = null;
    if (!config) reason = error ? "Load the project configuration before starting." : "Loading project configuration…";
    else if (editable) {
      if (editorState.saving) reason = "Saving configuration…";
      else if (editorState.dirty) reason = "Save your configuration changes before starting.";
      else if (editorState.valid === null) reason = "Waiting for configuration validation…";
      else if (!editorState.valid) reason = "Fix the configuration issues before starting.";
    }
    onLaunchReadiness(reason);
  }, [config, editable, editorState, error, onLaunchReadiness]);

  const save = useCallback(async (yamlText: string) => {
    const saved = await saveProjectConfig(project, yamlText);
    setConfig(saved);
  }, [project]);

  return (
    <div className="flex flex-1 min-w-0 min-h-0 flex-col">
      <div className="shrink-0 border-b border-border px-6 py-4">
        <h2 className="text-base font-semibold">
          {status === "running" ? "Waiting for the first evaluation" : status === "created" ? "Start your first evaluation" : "No evaluations yet"}
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {status === "running"
            ? "The run is in progress. Results will appear here as evaluations finish."
            : status === "created"
              ? "Review and save your configuration, select a dataset above, then start the evaluation."
              : "The run has not produced an evaluation yet. Use the run controls above to resume."}
        </p>
      </div>
      {error ? (
        <div role="alert" className="flex items-center gap-3 p-6 text-sm text-destructive">
          <span>{error}</span>
          <Button variant="outline" onClick={() => setRevision((value) => value + 1)}>Retry</Button>
        </div>
      ) : !config ? (
        <div role="status" className="flex flex-1 items-center justify-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" /> Loading project configuration…
        </div>
      ) : editable ? (
        <ConfigVisualEditor
          initialYaml={config.raw_content}
          initialContent={config.content}
          onSave={save}
          onStateChange={setEditorState}
          datasetInputsExternal
        />
      ) : (
        <div className="flex flex-1 min-h-0 flex-col">
          <p className="px-6 py-2 text-xs text-muted-foreground">
            {config.read_only_reason || "This run uses the saved configuration shown below."}
          </p>
          <div className="flex-1 min-h-0">
            <ConfigEditor value={config.raw_content} onChangeText={() => {}} readOnly />
          </div>
        </div>
      )}
    </div>
  );
}
