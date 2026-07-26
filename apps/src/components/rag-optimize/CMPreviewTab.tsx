import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { Download, Loader2, Play, RefreshCw, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useProject } from "@/context/ProjectContext";
import { useEvalData } from "@/context/EvalDataContext";
import { useSelection } from "@/context/SelectionContext";
import {
  cancelCMPreview,
  createCMPreview,
  downloadCMPreviewArtifact,
  fetchCMPreview,
  fetchCMPreviewOptions,
  listCMPreviews,
  isCMPreviewArtifactDownloadable,
  type CMParameter,
  type CMPreviewJob,
  type CMPreviewOptions,
} from "@/api/cmPreview";
import {
  activeParameters,
  buildPreviewRequest,
  isPreviewActive,
  parseObject,
} from "./cmPreviewModel";
import { CMCompiledPlanView, JsonDetails } from "./CMCompiledPlanView";

const fieldClass =
  "w-full rounded-md border border-input bg-background px-3 py-2 text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring";
const messageOf = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

function PointControls({
  parameters,
  value,
  onChange,
}: {
  parameters: CMParameter[];
  value: string;
  onChange: (value: string) => void;
}) {
  let values: Record<string, unknown>;
  try {
    values = parseObject(value, "Point parameters");
  } catch {
    return (
      <p className="text-xs text-muted-foreground">
        Fix the JSON below to use the parameter controls.
      </p>
    );
  }
  const active = activeParameters(parameters, values);
  function setValue(name: string, next: unknown) {
    const selected = { ...values, [name]: next };
    if (next === undefined) delete selected[name];
    const names = new Set(
      activeParameters(parameters, selected).map((p) => p.name),
    );
    onChange(
      JSON.stringify(
        Object.fromEntries(
          Object.entries(selected).filter(([key]) => names.has(key)),
        ),
        null,
        2,
      ),
    );
  }
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {active.map((parameter) => (
        <label key={parameter.name} className="min-w-0 space-y-1 text-xs">
          <span className="block break-all font-mono text-[11px]">
            {parameter.name}
          </span>
          {parameter.type === "fixed" ? (
            <div className="rounded-md border bg-muted/30 px-3 py-2 font-mono">
              {JSON.stringify(parameter.value ?? parameter.values?.[0])}{" "}
              <span className="text-muted-foreground">(fixed)</span>
            </div>
          ) : parameter.type === "choice" ? (
            <select
              className={fieldClass}
              value={
                values[parameter.name] === undefined
                  ? ""
                  : JSON.stringify(values[parameter.name])
              }
              onChange={(event) =>
                setValue(
                  parameter.name,
                  event.target.value
                    ? JSON.parse(event.target.value)
                    : undefined,
                )
              }
            >
              <option value="">Select a value…</option>
              {(parameter.values ?? []).map((item, index) => (
                <option key={index} value={JSON.stringify(item)}>
                  {JSON.stringify(item)}
                </option>
              ))}
            </select>
          ) : (
            <Input
              type="number"
              className="h-9 text-xs"
              value={
                values[parameter.name] === undefined
                  ? ""
                  : String(values[parameter.name])
              }
              min={parameter.bounds?.[0]}
              max={parameter.bounds?.[1]}
              step={parameter.value_type === "int" ? 1 : "any"}
              onChange={(event) =>
                setValue(
                  parameter.name,
                  event.target.value === ""
                    ? undefined
                    : Number(event.target.value),
                )
              }
            />
          )}
          {parameter.bounds && (
            <span className="block text-[10px] text-muted-foreground">
              Range {parameter.bounds[0]}–{parameter.bounds[1]} (
              {parameter.value_type ?? "number"})
            </span>
          )}
        </label>
      ))}
    </div>
  );
}

export function CMPreviewTab() {
  const { currentProject } = useProject();
  return currentProject === null ? null : (
    <ProjectCMPreview key={currentProject} project={currentProject} />
  );
}

function ProjectCMPreview({ project }: { project: string }) {
  const { evaluations, loading: evaluationsLoading } = useEvalData();
  const { selectedEvalId, setSelectedEvalId } = useSelection();
  const search = useSearch({ from: "/" });
  const navigate = useNavigate({ from: "/" });
  const selectedId = search.preview ?? null;
  const openPreview = useCallback(
    (id: string) =>
      navigate({
        search: (previous) => ({ ...previous, tab: "cm-preview", preview: id }),
      }),
    [navigate],
  );
  const [options, setOptions] = useState<CMPreviewOptions | null>(null);
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<CMPreviewJob[]>([]);
  const [job, setJob] = useState<CMPreviewJob | null>(null);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [source, setSource] = useState<"evaluation" | "parameters">(
    "evaluation",
  );
  const [parameters, setParameters] = useState("{}");
  const [queryMode, setQueryMode] = useState<"sample" | "ids">("sample");
  const [sampleCount, setSampleCount] = useState("8");
  const [questionIds, setQuestionIds] = useState("");
  const [reuseTrace, setReuseTrace] = useState(false);
  const [skipJudge, setSkipJudge] = useState(true);
  const [qaData, setQaData] = useState("");
  const [corpusData, setCorpusData] = useState("");
  const [datasetName, setDatasetName] = useState("");
  const [systemConfig, setSystemConfig] = useState("");
  const mounted = useRef(true);
  const currentPreview = useRef(selectedId);
  currentPreview.current = selectedId;
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (!evaluationsLoading && evaluations.length === 0)
      setSource("parameters");
  }, [evaluationsLoading, evaluations.length]);

  useEffect(() => {
    let alive = true;
    fetchCMPreviewOptions(project)
      .then((result) => {
        if (!alive) return;
        setOptions(result);
        setOptionsError(null);
      })
      .catch((error: unknown) => {
        if (alive) setOptionsError(messageOf(error));
      });
    return () => {
      alive = false;
    };
  }, [project, refresh]);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const response = await listCMPreviews(project);
        if (alive) {
          setJobs(response.previews);
          setHistoryError(null);
        }
      } catch (error) {
        if (alive) setHistoryError(messageOf(error));
      } finally {
        if (alive) {
          setHistoryLoading(false);
          timer = setTimeout(load, 5000);
        }
      }
    }
    void load();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [project, refresh]);

  useEffect(() => {
    if (!selectedId && jobs.length > 0 && !submitting)
      void openPreview(jobs[0].preview_id);
  }, [selectedId, jobs, submitting, openPreview]);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    setJob(null);
    setDetailError(null);
    setDetailLoading(Boolean(selectedId));
    if (!selectedId) return;
    async function load() {
      try {
        const result = await fetchCMPreview(project, selectedId!);
        if (!alive) return;
        setJob(result);
        setDetailError(null);
        if (isPreviewActive(result.status)) timer = setTimeout(load, 2000);
      } catch (error) {
        if (alive) {
          setDetailError(messageOf(error));
          timer = setTimeout(load, 5000);
        }
      } finally {
        if (alive) setDetailLoading(false);
      }
    }
    void load();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [project, selectedId, refresh]);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setActionError(null);
    try {
      const body = buildPreviewRequest({
        source,
        evalId: selectedEvalId,
        parameters,
        queryMode,
        sampleCount,
        questionIds,
        reuseTrace,
        skipJudge,
        qaData,
        corpusData,
        datasetName,
        systemConfig,
        systemRequired: options?.system_config_required ?? false,
        parameterSchema: options?.parameters,
      });
      setSubmitting(true);
      const created = await createCMPreview(project, body);
      if (mounted.current) {
        setJob(created);
        await openPreview(created.preview_id);
        setRefresh((value) => value + 1);
      }
    } catch (error) {
      if (mounted.current) setActionError(messageOf(error));
    } finally {
      if (mounted.current) setSubmitting(false);
    }
  }

  async function cancel() {
    if (!job) return;
    setCancelling(true);
    setActionError(null);
    try {
      const result = await cancelCMPreview(project, job.preview_id);
      if (mounted.current && currentPreview.current === result.preview_id) {
        setJob(result);
        setRefresh((value) => value + 1);
      }
    } catch (error) {
      if (mounted.current) setActionError(messageOf(error));
    } finally {
      if (mounted.current) setCancelling(false);
    }
  }

  async function download(name: string) {
    if (!job) return;
    setDownloading(name);
    setActionError(null);
    try {
      await downloadCMPreviewArtifact(project, job.preview_id, name);
    } catch (error) {
      if (mounted.current) setActionError(messageOf(error));
    } finally {
      if (mounted.current) setDownloading(null);
    }
  }

  const error = job?.error;
  return (
    <div className="flex-1 min-w-0 overflow-auto p-4 pb-12 md:p-6">
      <div className="mx-auto max-w-7xl space-y-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold">CM Preview</h2>
            <p className="mt-1 max-w-3xl text-xs text-muted-foreground">
              Run a small real query cohort, inspect its RAG-IR, and compile the
              execution path with RAG-CM. Preview results, observed RAG-IR and
              job history are saved for viewing and can be reopened.
            </p>
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setRefresh((value) => value + 1)}
          >
            <RefreshCw />
            Refresh
          </Button>
        </div>

        <form
          onSubmit={submit}
          className="space-y-4 rounded-lg border bg-card p-4"
        >
          <h3 className="text-sm font-semibold">Create preview</h3>
          <div className="grid gap-4 lg:grid-cols-2">
            <label className="space-y-1.5 text-xs font-medium">
              Pipeline source
              <select
                className={fieldClass}
                value={source}
                onChange={(event) => {
                  setSource(event.target.value as typeof source);
                  setReuseTrace(false);
                }}
              >
                <option value="evaluation">
                  Existing evaluation · resolved pipeline
                </option>
                <option value="parameters">
                  Current project · choose a concrete point
                </option>
              </select>
            </label>
            {source === "evaluation" && (
              <label className="space-y-1.5 text-xs font-medium">
                Evaluation
                <select
                  className={fieldClass}
                  value={selectedEvalId ?? ""}
                  onChange={(event) =>
                    setSelectedEvalId(
                      event.target.value === ""
                        ? null
                        : Number(event.target.value),
                    )
                  }
                >
                  <option value="">Select an evaluation…</option>
                  {[
                    ...new Map(
                      evaluations.map((entry) => [entry.eval_id, entry]),
                    ).values(),
                  ].map((entry) => (
                    <option key={entry.eval_id} value={entry.eval_id}>
                      #{entry.eval_id} · {entry.eval_type}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </div>
          {source === "evaluation" ? (
            <p className="text-xs text-muted-foreground">
              Uses the selected evaluation's saved algorithm and deployment. A
              fresh trace is generated unless reuse is explicitly enabled.
            </p>
          ) : (
            <div className="space-y-3">
              <p className="text-xs text-muted-foreground">
                Choose every active parameter in the project's formal search
                space. The server resolves and validates this point before
                evaluation.
              </p>
              {optionsError && (
                <p role="alert" className="text-xs text-destructive">
                  Could not load point options: {optionsError}
                </p>
              )}
              {!options && !optionsError && (
                <p className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Loader2 className="size-3 animate-spin" />
                  Loading project parameters…
                </p>
              )}
              {options?.selection_error && (
                <p role="alert" className="text-xs text-destructive">
                  {options.selection_error.code}:{" "}
                  {options.selection_error.message}
                </p>
              )}
              {options && !options.selection_error && (
                <PointControls
                  parameters={options.parameters}
                  value={parameters}
                  onChange={setParameters}
                />
              )}
              {options?.parameters.length === 0 && !options.selection_error && (
                <p className="text-xs text-muted-foreground">
                  No variable parameters. The server will resolve the project's
                  fixed configuration.
                </p>
              )}
              <details className="rounded-md border p-3">
                <summary className="cursor-pointer text-xs font-medium">
                  Point parameters JSON
                </summary>
                <textarea
                  aria-label="Point parameters JSON"
                  rows={5}
                  value={parameters}
                  onChange={(event) => setParameters(event.target.value)}
                  className={`${fieldClass} mt-2 font-mono`}
                  spellCheck={false}
                />
              </details>
            </div>
          )}
          <div className="grid gap-4 md:grid-cols-2">
            <label className="space-y-1.5 text-xs font-medium">
              Query selection
              <select
                className={fieldClass}
                value={queryMode}
                onChange={(event) =>
                  setQueryMode(event.target.value as typeof queryMode)
                }
              >
                <option value="sample">Sample from project QA</option>
                <option value="ids">Explicit question IDs</option>
              </select>
            </label>
            {queryMode === "sample" ? (
              <label className="space-y-1.5 text-xs font-medium">
                Sample count
                <Input
                  type="number"
                  min={1}
                  max={128}
                  value={sampleCount}
                  onChange={(event) => setSampleCount(event.target.value)}
                />
              </label>
            ) : (
              <label className="space-y-1.5 text-xs font-medium">
                Question IDs (JSON array)
                <Input
                  value={questionIds}
                  onChange={(event) => setQuestionIds(event.target.value)}
                  placeholder={'["q1", "q2"]'}
                  className="font-mono text-xs"
                />
              </label>
            )}
          </div>
          <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={skipJudge}
                onChange={(event) => setSkipJudge(event.target.checked)}
              />
              Skip quality judge (keep real pipeline execution)
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={reuseTrace}
                disabled={source !== "evaluation"}
                onChange={(event) => setReuseTrace(event.target.checked)}
              />
              Reuse the selected evaluation's existing trace
            </label>
          </div>
          {reuseTrace && (
            <p className="rounded-md border border-amber-500/30 bg-amber-500/5 p-2 text-xs">
              Reuse is explicit: the server uses recorded calls for the selected
              cohort and reports unavailable or incompatible traces. It does not
              run new queries.
            </p>
          )}
          <details
            className="rounded-md border p-3"
            open={
              source === "parameters" && options?.system_config_required
                ? true
                : undefined
            }
          >
            <summary className="cursor-pointer text-xs font-medium">
              Data arguments and deployment
              {source === "parameters" && options?.system_config_required
                ? " · system config required"
                : " · optional overrides"}
            </summary>
            <p className="my-3 text-xs text-muted-foreground">
              QA and corpus are runtime inputs, separate from algorithm/system
              configs. Empty paths use the project's recorded data.
              {options &&
                ` Project QA: ${options.dataset.qa_available ? "available" : "missing"}; corpus: ${options.dataset.corpus_available ? "available" : "missing"}.`}
            </p>
            <div className="grid gap-3 md:grid-cols-2">
              <label className="space-y-1 text-xs">
                QA path
                <Input
                  value={qaData}
                  onChange={(event) => setQaData(event.target.value)}
                  placeholder="Use project QA"
                  className="font-mono text-xs"
                />
              </label>
              <label className="space-y-1 text-xs">
                Corpus path
                <Input
                  value={corpusData}
                  onChange={(event) => setCorpusData(event.target.value)}
                  placeholder="Use project corpus"
                  className="font-mono text-xs"
                />
              </label>
              <label className="space-y-1 text-xs">
                Corpus identity for CM profiles
                <Input
                  value={datasetName}
                  onChange={(event) => setDatasetName(event.target.value)}
                  placeholder={
                    options?.dataset.dataset_name ?? "e.g. dragonball_en"
                  }
                  className="font-mono text-xs"
                />
              </label>
            </div>
            <label className="mt-3 block space-y-1 text-xs">
              Concrete system config (JSON)
              <textarea
                rows={5}
                value={systemConfig}
                onChange={(event) => setSystemConfig(event.target.value)}
                placeholder="Leave empty to use the selected deployment"
                className={`${fieldClass} font-mono`}
                spellCheck={false}
              />
            </label>
          </details>
          {actionError && (
            <p
              role="alert"
              className="rounded-md border border-destructive/40 bg-destructive/5 p-3 text-xs text-destructive"
            >
              {actionError}
            </p>
          )}
          <div className="flex flex-wrap items-center gap-3">
            <Button
              type="submit"
              size="sm"
              disabled={
                submitting ||
                (source === "parameters" &&
                  (!options || Boolean(options.selection_error)))
              }
            >
              {submitting ? <Loader2 className="animate-spin" /> : <Play />}
              {submitting ? "Starting…" : "Create CM preview"}
            </Button>
            <span className="text-xs text-muted-foreground">
              {reuseTrace
                ? "Recorded trace → CM compilation"
                : "Real evaluation → RAG-IR → CM compilation"}
            </span>
          </div>
        </form>

        <section className="space-y-3 rounded-lg border bg-card p-4">
          <h3 className="text-sm font-semibold">Preview history</h3>
          {historyLoading && (
            <p className="text-xs text-muted-foreground">
              Loading preview history…
            </p>
          )}
          {historyError && (
            <p role="alert" className="text-xs text-destructive">
              Could not load previews: {historyError}
            </p>
          )}
          {!historyLoading && !historyError && jobs.length === 0 && (
            <p className="text-xs text-muted-foreground">
              No previews yet. Create one using a concrete configuration above.
            </p>
          )}
          {jobs.length > 0 && (
            <label className="block text-xs">
              Open preview
              <select
                className={`${fieldClass} mt-1.5`}
                value={selectedId ?? ""}
                onChange={(event) => void openPreview(event.target.value)}
              >
                <option value="" disabled>
                  Select a preview…
                </option>
                {!jobs.some((entry) => entry.preview_id === selectedId) &&
                  selectedId && (
                    <option value={selectedId}>{selectedId}</option>
                  )}
                {jobs.map((entry) => (
                  <option key={entry.preview_id} value={entry.preview_id}>
                    {entry.created_at ?? entry.preview_id} ·{" "}
                    {entry.source?.eval_id == null
                      ? "Project point"
                      : `Eval #${entry.source.eval_id}`}{" "}
                    · {entry.status}
                  </option>
                ))}
              </select>
            </label>
          )}
        </section>

        {detailLoading && (
          <p className="flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Loading preview…
          </p>
        )}
        {detailError && (
          <p
            role="alert"
            className="rounded-md border border-destructive/40 p-3 text-xs text-destructive"
          >
            Could not load this preview: {detailError}
          </p>
        )}
        {job && (
          <>
            <section className="space-y-3 rounded-lg border bg-card p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  {isPreviewActive(job.status) && (
                    <Loader2 className="size-4 animate-spin" />
                  )}
                  <h3 className="text-sm font-semibold">
                    {job.status} · {job.phase}
                  </h3>
                </div>
                {isPreviewActive(job.status) && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={cancel}
                    disabled={cancelling}
                  >
                    {cancelling ? (
                      <Loader2 className="animate-spin" />
                    ) : (
                      <Square />
                    )}
                    {cancelling ? "Cancelling…" : "Cancel preview"}
                  </Button>
                )}
              </div>
              <p className="break-all font-mono text-[11px] text-muted-foreground">
                {job.preview_id}
              </p>
              <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs">
                <span>
                  Source:{" "}
                  {job.source.eval_id == null
                    ? job.source.kind
                    : `evaluation #${job.source.eval_id}`}
                </span>
                <span>
                  {job.source.reuse_trace ? "Reused trace" : "Fresh evaluation"}
                </span>
                <span>
                  {job.skip_judge
                    ? "Quality judge skipped"
                    : "Quality judge enabled"}
                </span>
                <span>
                  {job.cohort.count} selected / {job.cohort.requested_count}{" "}
                  requested queries
                </span>
              </div>
              <p className="break-all text-xs">
                <span className="text-muted-foreground">
                  Actual question IDs:{" "}
                </span>
                <code>{JSON.stringify(job.cohort.question_ids)}</code>
              </p>
              {error && (
                <div
                  role="alert"
                  className="space-y-2 rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-xs"
                >
                  <p className="font-semibold">
                    {typeof error === "string"
                      ? error
                      : `${error.code ?? job.status}: ${error.message}`}
                  </p>
                  {typeof error !== "string" && error.phase && (
                    <p>Phase: {error.phase}</p>
                  )}
                  {typeof error !== "string" && error.details != null && (
                    <JsonDetails
                      title="Blocker details"
                      value={error.details}
                    />
                  )}
                </div>
              )}
              <details
                open={
                  isPreviewActive(job.status) ||
                  job.status === "failed" ||
                  job.status === "blocked"
                }
                className="rounded-md border px-3 py-2"
              >
                <summary className="cursor-pointer text-xs font-medium">
                  Job log
                </summary>
                <pre
                  aria-label="Preview job log"
                  className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-all font-mono text-[11px] text-muted-foreground"
                >
                  {job.log_tail || "No log output yet."}
                </pre>
              </details>
              <div className="flex flex-wrap gap-2">
                {(job.artifacts ?? []).filter((artifact) => isCMPreviewArtifactDownloadable(artifact.name)).map((artifact) => (
                  <Button
                    key={artifact.name}
                    size="xs"
                    variant="outline"
                    onClick={() => void download(artifact.name)}
                    disabled={downloading !== null}
                  >
                    {downloading === artifact.name ? (
                      <Loader2 className="animate-spin" />
                    ) : (
                      <Download />
                    )}
                    {artifact.name}
                  </Button>
                ))}
              </div>
            </section>
            <CMCompiledPlanView
              key={job.preview_id}
              plan={job.compiled_plan ?? null}
              ragIR={job.rag_ir ?? null}
              compiledPathStatus={job.compiled_path_status ?? (job.status === "succeeded" && !job.compiled_plan ? "unavailable" : null)}
            />
          </>
        )}
      </div>
    </div>
  );
}
