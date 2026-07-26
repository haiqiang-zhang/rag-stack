/**
 * SimulatePanel — pipeline cost-estimate playground for a single cost model.
 *
 * Redesigned around shadcn primitives — single-column max-width-centered
 * page (no fixed left rail), shadcn Card/CardHeader/CardContent slots,
 * shadcn Select for dropdowns, shadcn Button for the run action, and
 * Tailwind classes throughout. Drops nested-card chrome and the legacy
 * 300px-rail-vs-narrow-stack ResizeObserver branching.
 *
 * The four `formKey` branches (faiss_ivf · llm_sim · query_expansion_sim ·
 * passage_reranker_sim) and all config-builder logic are preserved verbatim.
 */
import React, { useState, useEffect, useId } from "react";
import {
  Play,
  Zap,
  Clock,
  Activity,
  Loader2,
  TriangleAlert,
} from "lucide-react";

import { postEstimate, fetchGpuPresets, fetchLlmModels, fetchFaissPresets } from "@/api/costModel";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { hasComponentEstimateForm } from "./estimateForms";
import type {
  CostModelInfo,
  EstimateResponse,
  OperatorBreakdown,
} from "@/types/costModel";

// ── Config builders (unchanged) ─────────────────────────────────────

function buildFaissConfig(f: Record<string, number>) {
  return {
    vectordb: [
      {
        name: "faiss_ivf_sim",
        db_type: "faiss_ivf",
        N: f.N,
        d: f.d,
        nlist: f.nlist,
        M: f.M,
        nbits: f.nbits,
        parallel_mode: f.parallel_mode,
      },
    ],
    node_lines: [
      {
        node_line_name: "retrieve",
        nodes: [
          {
            stage: "semantic_retrieval",
            top_k: f.top_k,
            modules: [
              {
                component: "vectordb",
                vectordb: "faiss_ivf_sim",
                nprobe: f.nprobe,
                batch_size: f.batch_size,
              },
            ],
          },
        ],
      },
    ],
  };
}

function fullTokenStats(f: Record<string, number>) {
  return {
    avg_chunk_tokens: f.avg_chunk_tokens,
    avg_query_tokens: f.avg_query_tokens,
    avg_output_tokens: f.avg_output_tokens,
    prompt_template_tokens: f.prompt_template_tokens,
  };
}

function buildLlmConfig(f: Record<string, any>) {
  return {
    node_lines: [
      // Retrieval node is required by llm_sim's adapter to compute input
      // tokens (top_k × avg_chunk_tokens). Module shape doesn't matter for
      // the LLM cost — only top_k flows through.
      {
        node_line_name: "retrieve",
        nodes: [
          {
            stage: "semantic_retrieval",
            top_k: f.top_k,
            modules: [{ component: "vectordb" }],
          },
        ],
      },
      {
        node_line_name: "generate",
        nodes: [
          {
            stage: "generator",
            top_k: f.top_k,
            modules: [
              {
                component: f.component,
                model: f.llm_model,
                batch_size: f.batch_size,
              },
            ],
          },
        ],
      },
    ],
    _token_stats: fullTokenStats(f),
  };
}

export function buildQueryExpansionConfig(f: Record<string, any>) {
  return {
    node_lines: [
      {
        node_line_name: "query_expansion",
        nodes: [
          {
            stage: "query_expansion",
            modules: [
              {
                component: f.component,
                model: f.llm_model,
                batch_size: f.batch_size,
              },
            ],
          },
        ],
      },
    ],
    _token_stats: fullTokenStats(f),
  };
}

function buildPassageCompressorConfig(f: Record<string, any>) {
  return {
    node_lines: [
      // Retrieval node — required by passage_compressor_sim's _get_params
      // to look up top_k via _find_retrieval_top_k().
      {
        node_line_name: "retrieve",
        nodes: [
          {
            stage: "semantic_retrieval",
            top_k: f.top_k,
            modules: [{ component: "vectordb" }],
          },
        ],
      },
      // Compressor node — what the cost model actually keys on.
      {
        node_line_name: "post_retrieve",
        nodes: [
          {
            stage: "passage_compressor",
            modules: [
              {
                component: f.component,
                model: f.llm_model,
                batch_size: f.batch_size,
              },
            ],
          },
        ],
      },
    ],
    _token_stats: fullTokenStats(f),
  };
}

function buildPassageRerankerConfig(f: Record<string, any>) {
  return {
    node_lines: [
      {
        node_line_name: "retrieve",
        nodes: [
          {
            stage: "semantic_retrieval",
            top_k: f.top_k,
            modules: [{ component: "vectordb" }],
          },
        ],
      },
      {
        node_line_name: "post_retrieve",
        nodes: [
          {
            stage: "passage_reranker",
            top_k: f.top_k,
            modules: [
              {
                component: "sentence_transformer_reranker",
                model: f.model_name,
                batch_size: f.batch_size,
              },
            ],
          },
        ],
      },
    ],
    _token_stats: fullTokenStats(f),
  };
}

// ── Format helpers ──────────────────────────────────────────────────

function fmtLatency(s: number): string {
  if (s < 0.001) return `${(s * 1e6).toFixed(1)} µs`;
  if (s < 1) return `${(s * 1000).toFixed(2)} ms`;
  return `${s.toFixed(3)} s`;
}

function fmtQPS(q: number): string {
  if (q >= 1e6) return `${(q / 1e6).toFixed(1)}M`;
  if (q >= 1e3) return `${(q / 1e3).toFixed(1)}K`;
  return q.toFixed(1);
}

function fmtTime(t: number): string {
  if (t < 0.001) return `${(t * 1e6).toFixed(1)} µs`;
  if (t < 1) return `${(t * 1000).toFixed(3)} ms`;
  return `${t.toFixed(4)} s`;
}

// ── Reusable form helpers ───────────────────────────────────────────

function ParamField({
  label,
  value,
  onChange,
  hint,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  hint?: string;
}) {
  const id = useId();
  return (
    <div className="space-y-1.5 min-w-0">
      <Label htmlFor={id} className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </Label>
      <Input
        id={id}
        type="text"
        inputMode="numeric"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="font-mono text-sm"
      />
      {hint && (
        <p className="text-[10px] text-muted-foreground">{hint}</p>
      )}
    </div>
  );
}

function FormSelect<T extends string>({
  label,
  value,
  onChange,
  options,
  placeholder,
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: Array<{ key: T; label: string }>;
  placeholder?: string;
}) {
  const id = useId();
  return (
    <div className="space-y-1.5 min-w-0">
      <Label htmlFor={id} className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </Label>
      <Select value={value} onValueChange={(v) => onChange(v as T)}>
        <SelectTrigger id={id} aria-label={label}>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {options.map((o) => (
            <SelectItem key={o.key} value={o.key}>
              {o.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

function FormSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold">{title}</h3>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-3">
        {children}
      </div>
    </div>
  );
}

// ── Operator breakdown table ────────────────────────────────────────

function OperatorTable({
  operators,
  totalLatency,
}: {
  operators: OperatorBreakdown[];
  totalLatency: number;
}) {
  return (
    <div className="overflow-auto rounded-md border border-border">
      <table className="w-full border-collapse text-xs">
        <thead className="sticky top-0 z-[1] bg-muted/50 backdrop-blur">
          <tr>
            <th className="border-b border-border px-3 py-2 text-left text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Operator
            </th>
            <th className="border-b border-border px-3 py-2 text-right text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Compute
            </th>
            <th className="border-b border-border px-3 py-2 text-right text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Memory
            </th>
            <th className="border-b border-border px-3 py-2 text-right text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Total
            </th>
            <th className="border-b border-border px-3 py-2 text-center text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Bottleneck
            </th>
          </tr>
        </thead>
        <tbody>
          {operators.map((op, i) => {
            const pct = totalLatency > 0 ? (op.time_s / totalLatency) * 100 : 0;
            const isCompute = op.bottleneck === "compute";
            return (
              <tr
                key={op.name}
                className={cn(
                  "border-b border-border last:border-0",
                  i % 2 === 1 && "bg-muted/20",
                )}
              >
                <td className="px-3 py-2 align-top">
                  <div className="text-sm font-medium text-foreground">
                    {op.name}
                  </div>
                  <div
                    className="mt-1 h-[2px] rounded-full bg-primary"
                    style={{ width: `${Math.min(pct, 100)}%` }}
                  />
                </td>
                <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-[11px] text-muted-foreground">
                  {fmtTime(op.compute_time_s)}
                </td>
                <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-[11px] text-muted-foreground">
                  {fmtTime(op.memory_time_s)}
                </td>
                <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-[11px] font-semibold text-foreground">
                  {fmtTime(op.time_s)}
                </td>
                <td className="px-3 py-2 text-center">
                  <span
                    className={cn(
                      "inline-block rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider",
                      isCompute
                        ? "bg-primary/10 text-primary"
                        : "bg-amber-500/10 text-amber-600",
                    )}
                  >
                    {op.bottleneck}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Main panel ──────────────────────────────────────────────────────

interface Props {
  model: CostModelInfo;
}

export function SimulatePanel({ model }: Props) {
  if (!hasComponentEstimateForm(model.name)) return (
    <div className="mx-auto max-w-xl p-8 text-center">
      <h2 className="text-base font-semibold">No component estimate form</h2>
      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{model.name} has no dedicated form here. Use your project’s CM Preview to compile its real RAG-IR and inspect the selected cost model and calibration.</p>
    </div>
  );
  return <ComponentEstimateForm key={model.name} model={model} />;
}

function ComponentEstimateForm({ model }: Props) {
  const formKey = model.name;

  const [cpuPresets, setCpuPresets] = useState<{ name: string; description: string }[]>([]);
  const [presetError, setPresetError] = useState<string | null>(null);
  const [presetAttempt, setPresetAttempt] = useState(0);
  const [calibrationMode, setCalibrationMode] = useState("strict");
  const [gpuPresets, setGpuPresets] = useState<string[]>([]);
  const [llmModels, setLlmModels] = useState<string[]>([]);

  useEffect(() => {
    let active = true;
    setPresetError(null);
    const load = formKey === "faiss_ivf"
      ? fetchFaissPresets().then((presets) => { if (active) setCpuPresets(presets); })
      : fetchGpuPresets().then((presets) => { if (active) setGpuPresets(presets); });
    load.catch((err) => { if (active) setPresetError(err instanceof Error ? err.message : String(err)); });
    if (formKey !== "faiss_ivf") fetchLlmModels().then((models) => { if (active) setLlmModels(models); }).catch(() => {});
    return () => { active = false; };
  }, [formKey, presetAttempt]);

  // FAISS params
  const [faiss, setFaiss] = useState({
    N: "1000000",
    d: "768",
    nlist: "1024",
    M: "96",
    nbits: "8",
    nprobe: "64",
    top_k: "10",
    batch_size: "1",
    parallel_mode: "0",
  });
  const [cpuPreset, setCpuPreset] = useState("m3_pro");

  const [llm, setLlm] = useState({
    component: "vllm",
    llm_model: "Qwen/Qwen2.5-3B-Instruct",
    avg_chunk_tokens: "256",
    avg_query_tokens: "32",
    avg_output_tokens: "128",
    prompt_template_tokens: "50",
    top_k: "5",
    batch_size: "1",
  });
  const [gpuPreset, setGpuPreset] = useState("A100_80GB_GPU");

  const [queryExpansion, setQueryExpansion] = useState({
    component: "hyde",
    llm_model: "Qwen/Qwen2.5-3B-Instruct",
    avg_query_tokens: "32",
    avg_chunk_tokens: "256",
    avg_output_tokens: "64",
    prompt_template_tokens: "0",
    batch_size: "1",
  });

  const [passageCompressor, setPassageCompressor] = useState({
    component: model.name === "passage_compressor_token_pruning_sim" ? "longllmlingua" : "tree_summarize",
    llm_model: "Qwen/Qwen2.5-3B-Instruct",
    avg_query_tokens: "32",
    avg_chunk_tokens: "256",
    avg_output_tokens: "128",
    prompt_template_tokens: "0",
    top_k: "5",
    batch_size: "1",
  });

  const [passageReranker, setPassageReranker] = useState({
    model_name: "ms_marco_minilm_l2",
    avg_query_tokens: "32",
    avg_chunk_tokens: "256",
    avg_output_tokens: "1",
    prompt_template_tokens: "0",
    top_k: "10",
    batch_size: "1",
  });

  const [result, setResult] = useState<EstimateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      let pipelineConfig: Record<string, unknown>;
      let hardware: Record<string, unknown>;

      if (formKey === "faiss_ivf") {
        const f = Object.fromEntries(
          Object.entries(faiss).map(([k, v]) => [k, Number(v)]),
        );
        pipelineConfig = buildFaissConfig(f);
        hardware = { cpu: cpuPreset };
      } else if (formKey === "llm_sim") {
        const f = {
          ...llm,
          avg_chunk_tokens: Number(llm.avg_chunk_tokens),
          avg_query_tokens: Number(llm.avg_query_tokens),
          avg_output_tokens: Number(llm.avg_output_tokens),
          prompt_template_tokens: Number(llm.prompt_template_tokens),
          top_k: Number(llm.top_k),
          batch_size: Number(llm.batch_size),
        };
        pipelineConfig = buildLlmConfig(f);
        hardware = { gpu: gpuPreset };
      } else if (formKey === "query_expansion_sim") {
        const f = {
          ...queryExpansion,
          avg_query_tokens: Number(queryExpansion.avg_query_tokens),
          avg_chunk_tokens: Number(queryExpansion.avg_chunk_tokens),
          avg_output_tokens: Number(queryExpansion.avg_output_tokens),
          prompt_template_tokens: Number(queryExpansion.prompt_template_tokens),
          batch_size: Number(queryExpansion.batch_size),
        };
        pipelineConfig = buildQueryExpansionConfig(f);
        hardware = { gpu: gpuPreset };
      } else if (
        formKey === "passage_compressor_sim" ||
        formKey === "passage_compressor_token_pruning_sim"
      ) {
        const f = {
          ...passageCompressor,
          avg_query_tokens: Number(passageCompressor.avg_query_tokens),
          avg_chunk_tokens: Number(passageCompressor.avg_chunk_tokens),
          avg_output_tokens: Number(passageCompressor.avg_output_tokens),
          prompt_template_tokens: Number(passageCompressor.prompt_template_tokens),
          top_k: Number(passageCompressor.top_k),
          batch_size: Number(passageCompressor.batch_size),
        };
        pipelineConfig = buildPassageCompressorConfig(f);
        hardware = { gpu: gpuPreset };
      } else {
        const f = {
          ...passageReranker,
          avg_query_tokens: Number(passageReranker.avg_query_tokens),
          avg_chunk_tokens: Number(passageReranker.avg_chunk_tokens),
          avg_output_tokens: Number(passageReranker.avg_output_tokens),
          prompt_template_tokens: Number(passageReranker.prompt_template_tokens),
          top_k: Number(passageReranker.top_k),
          batch_size: Number(passageReranker.batch_size),
        };
        pipelineConfig = buildPassageRerankerConfig(f);
        hardware = { gpu: gpuPreset };
      }

      hardware.rag_cm = { calibration_mode: calibrationMode };
      const r = await postEstimate(model.name, pipelineConfig, hardware);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  // ── Mutators ──────────────────────────────────────────────────────
  const updateFaiss = (k: string, v: string) =>
    setFaiss((p) => ({ ...p, [k]: v }));
  const updateLlm = (k: string, v: string) =>
    setLlm((p) => ({ ...p, [k]: v }));
  const updateQueryExpansion = (k: string, v: string) =>
    setQueryExpansion((p) => ({ ...p, [k]: v }));
  const updatePassageReranker = (k: string, v: string) =>
    setPassageReranker((p) => ({ ...p, [k]: v }));
  const updatePassageCompressor = (k: string, v: string) =>
    setPassageCompressor((p) => ({ ...p, [k]: v }));

  // ── Form bodies ───────────────────────────────────────────────────

  const formContent = (
    <div className="space-y-8">
      {formKey === "faiss_ivf" && (
        <>
          <FormSection title="Index">
            <ParamField
              label="N (vectors)"
              value={faiss.N}
              onChange={(v) => updateFaiss("N", v)}
            />
            <ParamField
              label="d (dims)"
              value={faiss.d}
              onChange={(v) => updateFaiss("d", v)}
            />
            <ParamField
              label="nlist"
              value={faiss.nlist}
              onChange={(v) => updateFaiss("nlist", v)}
            />
            <ParamField
              label="M (sub-quant)"
              value={faiss.M}
              onChange={(v) => updateFaiss("M", v)}
            />
            <ParamField
              label="nbits"
              value={faiss.nbits}
              onChange={(v) => updateFaiss("nbits", v)}
            />
          </FormSection>
          <FormSection title="Search">
            <ParamField
              label="nprobe"
              value={faiss.nprobe}
              onChange={(v) => updateFaiss("nprobe", v)}
            />
            <ParamField
              label="top_k"
              value={faiss.top_k}
              onChange={(v) => updateFaiss("top_k", v)}
            />
            <ParamField
              label="batch_size"
              value={faiss.batch_size}
              onChange={(v) => updateFaiss("batch_size", v)}
            />
            <FormSelect
              label="parallel_mode"
              value={faiss.parallel_mode}
              onChange={(v) => updateFaiss("parallel_mode", v)}
              options={[
                { key: "0", label: "0 — Inter-query" },
                { key: "1", label: "1 — Intra-query" },
              ]}
            />
          </FormSection>
          <FormSection title="Hardware">
            <FormSelect
              label="CPU Preset"
              value={cpuPreset}
              onChange={setCpuPreset}
              options={cpuPresets.map((preset) => ({
                key: preset.name,
                label: preset.description || preset.name,
              }))}
            />
          </FormSection>
        </>
      )}

      {formKey === "llm_sim" && (
        <>
          <FormSection title="Model">
            <FormSelect
              label="Module Type"
              value={llm.component}
              onChange={(v) => updateLlm("component", v)}
              options={[
                { key: "vllm", label: "vLLM" },
                { key: "openai_llm", label: "OpenAI LLM" },
                { key: "llama_index_llm", label: "LlamaIndex LLM" },
              ]}
            />
            <FormSelect
              label="LLM Model"
              value={llm.llm_model}
              onChange={(v) => updateLlm("llm_model", v)}
              options={llmModels.map((m) => ({ key: m, label: m }))}
            />
          </FormSection>
          <FormSection title="Token statistics">
            <ParamField
              label="avg_chunk_tokens"
              value={llm.avg_chunk_tokens}
              onChange={(v) => updateLlm("avg_chunk_tokens", v)}
            />
            <ParamField
              label="avg_query_tokens"
              value={llm.avg_query_tokens}
              onChange={(v) => updateLlm("avg_query_tokens", v)}
            />
            <ParamField
              label="avg_output_tokens"
              value={llm.avg_output_tokens}
              onChange={(v) => updateLlm("avg_output_tokens", v)}
            />
            <ParamField
              label="prompt_template_tokens"
              value={llm.prompt_template_tokens}
              onChange={(v) => updateLlm("prompt_template_tokens", v)}
            />
            <ParamField
              label="top_k"
              value={llm.top_k}
              onChange={(v) => updateLlm("top_k", v)}
            />
            <ParamField
              label="batch_size"
              value={llm.batch_size}
              onChange={(v) => updateLlm("batch_size", v)}
            />
          </FormSection>
          <FormSection title="Hardware">
            <FormSelect
              label="GPU"
              value={gpuPreset}
              onChange={setGpuPreset}
              options={gpuPresets.map((g) => ({
                key: g,
                label: g.replace(/_/g, " "),
              }))}
            />
          </FormSection>
        </>
      )}

      {formKey === "query_expansion_sim" && (
        <>
          <FormSection title="Module">
            <FormSelect
              label="Module Type"
              value={queryExpansion.component}
              onChange={(v) => updateQueryExpansion("component", v)}
              options={[
                { key: "hyde", label: "HyDE" },
                { key: "query_decompose", label: "Query Decompose" },
                { key: "multi_query_expansion", label: "Multi-Query" },
              ]}
            />
            <FormSelect
              label="LLM Model"
              value={queryExpansion.llm_model}
              onChange={(v) => updateQueryExpansion("llm_model", v)}
              options={llmModels.map((m) => ({ key: m, label: m }))}
            />
          </FormSection>
          <FormSection title="Parameters">
            <ParamField
              label="avg_query_tokens"
              value={queryExpansion.avg_query_tokens}
              onChange={(v) => updateQueryExpansion("avg_query_tokens", v)}
            />
            <ParamField
              label="avg_chunk_tokens"
              value={queryExpansion.avg_chunk_tokens}
              onChange={(v) => updateQueryExpansion("avg_chunk_tokens", v)}
            />
            <ParamField
              label="batch_size"
              value={queryExpansion.batch_size}
              onChange={(v) => updateQueryExpansion("batch_size", v)}
            />
          </FormSection>
          <FormSection title="Hardware">
            <FormSelect
              label="GPU"
              value={gpuPreset}
              onChange={setGpuPreset}
              options={gpuPresets.map((g) => ({
                key: g,
                label: g.replace(/_/g, " "),
              }))}
            />
          </FormSection>
        </>
      )}

      {(formKey === "passage_compressor_sim" ||
        formKey === "passage_compressor_token_pruning_sim") && (
        <>
          <FormSection title="Module">
            <FormSelect
              label="Module Type"
              value={passageCompressor.component}
              onChange={(v) => updatePassageCompressor("component", v)}
              options={
                formKey === "passage_compressor_token_pruning_sim"
                  ? [{ key: "longllmlingua", label: "longllmlingua" }]
                  : [
                      { key: "tree_summarize", label: "tree_summarize" },
                      { key: "refine", label: "refine" },
                    ]
              }
            />
            {formKey === "passage_compressor_token_pruning_sim" ? <p className="text-xs text-muted-foreground">LongLLMLingua uses the adapter’s fixed Llama 2 7B forward model.</p> : <FormSelect
              label="LLM Model"
              value={passageCompressor.llm_model}
              onChange={(v) => updatePassageCompressor("llm_model", v)}
              options={llmModels.map((m) => ({ key: m, label: m }))}
            />}
          </FormSection>
          <FormSection title="Token statistics">
            <ParamField
              label="avg_query_tokens"
              value={passageCompressor.avg_query_tokens}
              onChange={(v) => updatePassageCompressor("avg_query_tokens", v)}
            />
            <ParamField
              label="avg_chunk_tokens"
              value={passageCompressor.avg_chunk_tokens}
              onChange={(v) => updatePassageCompressor("avg_chunk_tokens", v)}
            />
            {formKey !== "passage_compressor_token_pruning_sim" && <ParamField
              label="avg_output_tokens"
              value={passageCompressor.avg_output_tokens}
              onChange={(v) => updatePassageCompressor("avg_output_tokens", v)}
            />}
            <ParamField
              label="prompt_template_tokens"
              value={passageCompressor.prompt_template_tokens}
              onChange={(v) =>
                updatePassageCompressor("prompt_template_tokens", v)
              }
            />
            <ParamField
              label="top_k"
              value={passageCompressor.top_k}
              onChange={(v) => updatePassageCompressor("top_k", v)}
            />
            <ParamField
              label="batch_size"
              value={passageCompressor.batch_size}
              onChange={(v) => updatePassageCompressor("batch_size", v)}
            />
          </FormSection>
          <FormSection title="Hardware">
            <FormSelect
              label="GPU"
              value={gpuPreset}
              onChange={setGpuPreset}
              options={gpuPresets.map((g) => ({
                key: g,
                label: g.replace(/_/g, " "),
              }))}
            />
          </FormSection>
        </>
      )}

      {formKey === "passage_reranker_sim" && (
        <>
          <FormSection title="Model">
            <div className="space-y-1.5 sm:col-span-2 md:col-span-3 min-w-0">
              <Label className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                Model Name
              </Label>
              <Input
                type="text"
                value={passageReranker.model_name}
                onChange={(e) =>
                  updatePassageReranker("model_name", e.target.value)
                }
                className="font-mono text-sm"
              />
            </div>
          </FormSection>
          <FormSection title="Parameters">
            <ParamField
              label="avg_query_tokens"
              value={passageReranker.avg_query_tokens}
              onChange={(v) => updatePassageReranker("avg_query_tokens", v)}
            />
            <ParamField
              label="avg_chunk_tokens"
              value={passageReranker.avg_chunk_tokens}
              onChange={(v) => updatePassageReranker("avg_chunk_tokens", v)}
            />
            <ParamField
              label="top_k"
              value={passageReranker.top_k}
              onChange={(v) => updatePassageReranker("top_k", v)}
            />
            <ParamField
              label="batch_size"
              value={passageReranker.batch_size}
              onChange={(v) => updatePassageReranker("batch_size", v)}
            />
          </FormSection>
          <FormSection title="Hardware">
            <FormSelect
              label="GPU"
              value={gpuPreset}
              onChange={setGpuPreset}
              options={gpuPresets.map((g) => ({
                key: g,
                label: g.replace(/_/g, " "),
              }))}
            />
          </FormSection>
        </>
      )}
    </div>
  );

  // ── Page ──────────────────────────────────────────────────────────

  return (
    <div className="flex-1 min-w-0 overflow-auto">
      <div className="mx-auto max-w-[1100px] px-8 pt-8 pb-20 space-y-10">
        {/* Section header */}
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <h2 className="text-xl font-semibold tracking-tight">
              Component estimate{" "}
              <span className="font-mono text-primary">{model.name}</span>
            </h2>
            <p className="text-sm text-muted-foreground">
              Estimate this component with explicit workload and hardware inputs.
              For a project’s compiled execution path, use CM Preview.
            </p>
          </div>
          <Button type="button" onClick={handleRun} disabled={loading || !!presetError || (formKey === "faiss_ivf" ? !cpuPresets.length : !gpuPresets.length)}>
            {loading ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Play className="size-4" />
            )}
            {loading ? "Running…" : "Run Estimate"}
          </Button>
        </div>

        {/* Configuration */}
        <section className="space-y-4">
          <div className="space-y-1">
            <h3 className="text-base font-semibold tracking-tight">
              Configuration
            </h3>
            <p className="text-xs text-muted-foreground">
              These inputs are passed to the selected component adapter. This estimate does not include project scheduling, communication, or end-to-end queueing.
            </p>
          </div>
          {formContent}
          <FormSection title="Calibration policy">
            <FormSelect label="Calibration mode" value={calibrationMode} onChange={setCalibrationMode}
              options={[{ key: "strict", label: "Strict · require matching calibration" }, { key: "uncalibrated_spec", label: "Allow specification fallback" }]} />
            <p className="text-xs leading-relaxed text-muted-foreground">{calibrationMode === "strict"
              ? "Missing or incompatible calibration blocks the estimate."
              : "Matching calibration takes priority. If it is missing, the adapter may estimate from hardware specifications."}</p>
          </FormSection>
        </section>

        {/* Error */}
        {(error || presetError) && (
          <div className="flex items-start gap-2.5 rounded-md border border-destructive/20 bg-destructive/5 p-3">
            <TriangleAlert className="size-4 shrink-0 text-destructive" />
            <div className="min-w-0 flex-1 space-y-1">
              <p className="text-xs font-semibold text-destructive">
                Estimate failed
              </p>
              <p className="break-words font-mono text-[11px] text-destructive">
                {(() => {
                  try {
                    return JSON.parse(error || presetError || "").detail || error || presetError;
                  } catch {
                    return error || presetError;
                  }
                })()}
              </p>
              {presetError && <Button variant="outline" size="sm" onClick={() => setPresetAttempt((value) => value + 1)}>Reload hardware presets</Button>}
            </div>
          </div>
        )}

        {/* Result */}
        <section className="space-y-4">
          <div className="space-y-1">
            <h3 className="text-base font-semibold tracking-tight">Result</h3>
            <p className="text-xs text-muted-foreground">
              Component latency and capacity for the supplied inputs.
            </p>
          </div>
          {loading ? (
            <div className="flex items-center justify-center gap-2 py-10">
              <Loader2 className="size-5 animate-spin text-muted-foreground" />
              <span className="text-sm text-muted-foreground">
                Estimating…
              </span>
            </div>
          ) : !result ? (
            <div className="flex flex-col items-center justify-center gap-3 py-10 text-center">
              <Zap className="size-8 text-muted-foreground/40" />
              <p className="text-sm text-muted-foreground">
                Configure parameters and click <b>Run Estimate</b>.
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <Metric
                  icon={<Clock className="size-3.5" />}
                  label="Component latency"
                  value={fmtLatency(result.latency_s)}
                  sub={`${result.latency_s.toExponential(3)} s`}
                />
                <Metric
                  icon={<Activity className="size-3.5" />}
                  label="Component capacity"
                  value={fmtQPS(result.throughput_qps)}
                  sub={`${result.throughput_qps.toFixed(2)} QPS`}
                />
              </div>

              {result.operators.length > 0 && (
                <div className="space-y-2">
                  <h4 className="border-b border-border pb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Operator breakdown
                  </h4>
                  <OperatorTable
                    operators={result.operators}
                    totalLatency={result.latency_s}
                  />
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function Metric({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        <span className="text-primary">{icon}</span>
        {label}
      </div>
      <div className="text-3xl font-semibold tabular-nums tracking-tight text-foreground">
        {value}
      </div>
      <div className="font-mono text-xs text-muted-foreground">{sub}</div>
    </div>
  );
}
