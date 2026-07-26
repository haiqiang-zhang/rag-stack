import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createDataClient } from "@/lib/config";
import { useProject } from "@/context/ProjectContext";
import { STAGE_BY_NAME, stageChipsColumn } from "@/lib/stages";
import type {
  AssemblyResult,
  PipelineNode,
  RagoSweepPoint,
  TaxonomyStages,
  SweepRow,
} from "@/types";

export interface StageInfo {
  name: string;
  qps: number;
  num_chips: number;
  batch_size: number;
  /** Synthetic proxy (batch/qps) kept for backward compat. Prefer `svc_s`. */
  latency_s: number;
  /** Real per-stage service time from best_hw_params svc_<stage>_s (seconds). */
  svc_s: number;
  /** Standalone throughput ceiling of this stage (qps_<stage>_capacity), if the
   *  cost model reported it; 0 when unavailable (older/collocated rows). */
  qps_capacity: number;
  /** Tensor / pipeline parallel degree chosen for this stage (0 if N/A). */
  tp: number;
  pp: number;
  is_bottleneck: boolean;
}

export interface EvalBreakdown {
  pipeline: PipelineNode[];
  stages: StageInfo[];
  ttft_s: number;
  e2e_s: number;
  decode_normalized_s: number;
  /** Time a request spends queued before service (best_hw_params
   *  latency_s_queue_wait). This is what the old waterfall wrongly folded
   *  into the decode segment. */
  queue_wait_s: number;
  qps_bound: string[];
  getStageSweep: (stage: string) => SweepRow[] | undefined;
  loading: boolean;
}

const EMPTY: EvalBreakdown = {
  pipeline: [],
  stages: [],
  ttft_s: 0,
  e2e_s: 0,
  decode_normalized_s: 0,
  queue_wait_s: 0,
  qps_bound: [],
  getStageSweep: () => undefined,
  loading: false,
};

const NUMERIC_DISPLAY_KEYS = new Set([
  "top_k",
  "model",
  "temperature",
  "bits",
  "embedding_model",
]);

function nodeDisplayParams(
  node: Record<string, unknown>,
): { label: string; value: string }[] {
  const out: { label: string; value: string }[] = [];
  if (node.top_k != null) {
    out.push({ label: "top_k", value: String(node.top_k) });
  }
  const modules = (node.modules as Array<Record<string, unknown>>) || [];
  const m = modules[0] || {};
  for (const key of ["model", "embedding_model", "temperature", "bits"]) {
    const v = m[key];
    if (v != null && v !== "") {
      const str = Array.isArray(v) ? String(v[0]) : String(v);
      out.push({ label: key, value: str });
    }
  }
  return out;
}

function nodeOverride(
  node: Record<string, unknown>,
): number | Record<string, number> | undefined {
  const sys = (node.system as Record<string, unknown>) || {};
  const v = sys.max_batch_size;
  if (v == null) return undefined;
  if (typeof v === "number") return v;
  if (typeof v === "object") {
    const out: Record<string, number> = {};
    for (const [k, n] of Object.entries(v as Record<string, unknown>)) {
      out[k] = Number(n);
    }
    return out;
  }
  return undefined;
}

function buildStageToPerformanceStages(
  taxonomy: TaxonomyStages,
): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const [stage, info] of Object.entries(taxonomy)) {
    out[stage] = info.performance_stages;
  }
  return out;
}

function derivePipeline(
  pipelineConfig: Record<string, unknown>,
  nodeTypeToStages: Record<string, string[]>,
): PipelineNode[] {
  const out: PipelineNode[] = [];
  const lines = (pipelineConfig.node_lines as Array<Record<string, unknown>>) || [];
  for (const nl of lines) {
    const nodes = (nl.nodes as Array<Record<string, unknown>>) || [];
    for (const node of nodes) {
      const nodeType = node.stage as string;
      const modules = (node.modules as Array<Record<string, unknown>>) || [];
      const moduleType = (modules[0]?.component as string) || "";
      out.push({
        node_line: (nl.node_line_name as string) || "",
        stage: nodeType,
        component: moduleType,
        display: nodeDisplayParams(node),
        stages: nodeTypeToStages[nodeType] ?? [],
        override: nodeOverride(node),
      });
    }
  }
  return out;
}

const STAGE_AGGREGATE_KEYS = new Set(["qps", "qps_per_chip", "qps_bound"]);

/** `qps_bound` may arrive as a JSON array (from assembly.json) or a Python
 *  repr string ("['decode']") from rago_sweep.csv. Normalize to string[]. */
function parseQpsBound(raw: unknown): string[] {
  if (raw == null) return [];
  if (Array.isArray(raw)) return raw.map(String);
  if (typeof raw !== "string") return [];
  const s = raw.trim();
  if (s === "") return [];
  try {
    const parsed = JSON.parse(s.replace(/'/g, '"'));
    if (Array.isArray(parsed)) return parsed.map(String);
  } catch {
    // ignore
  }
  return [];
}

function deriveStagesFromParams(p: Record<string, unknown> | null): StageInfo[] {
  if (!p) return [];
  // Iterative mode: result_df has no per-stage qps_* columns. Synthesize a
  // 3-stage view from the iterative columns instead.
  const mode = (p["pipeline_mode"] ?? "").toString();
  if (mode && mode !== "sequential") {
    const out: StageInfo[] = [];
    const stagesIterative: Array<{ name: string; chipsKey: string; batchKey: string }> = [
      { name: "semantic_retrieval_vectorsearch", chipsKey: "num_retrieval_servers", batchKey: "batch_size_iterative" },
      { name: "generator_prefill", chipsKey: "num_chips_generator_prefill", batchKey: "batch_size_iterative" },
      { name: "generator_decode", chipsKey: "num_chips_generator_decode", batchKey: "batch_size_decode" },
    ];
    for (const s of stagesIterative) {
      const chips = Number(p[s.chipsKey]);
      const batch = Number(p[s.batchKey]);
      out.push({
        name: s.name,
        qps: 0,
        num_chips: Number.isFinite(chips) ? chips : 0,
        batch_size: Number.isFinite(batch) ? batch : 0,
        latency_s: 0,
        svc_s: 0,
        qps_capacity: 0,
        tp: 0,
        pp: 0,
        is_bottleneck: false,
      });
    }
    return out;
  }

  const qpsBound = parseQpsBound(p.qps_bound);
  const out: StageInfo[] = [];
  for (const key of Object.keys(p)) {
    if (!key.startsWith("qps_")) continue;
    if (STAGE_AGGREGATE_KEYS.has(key)) continue;
    const stage = key.slice("qps_".length);
    // Skip the per-stage capacity columns (qps_<short>_capacity) — they are
    // ceilings, not stages, and would otherwise be misread as phantom stages.
    if (stage.endsWith("_capacity")) continue;
    const qps = Number(p[key]);
    const batch = Number(p[`batch_size_${stage}`]);
    const chips = Number(p[stageChipsColumn(stage)]);
    const latency = qps > 0 ? batch / qps : 0;
    const svcKey = STAGE_BY_NAME[stage]?.svcKey;
    const svc = svcKey != null ? Number(p[`svc_${svcKey}_s`]) : NaN;
    const cap = svcKey != null ? Number(p[`qps_${svcKey}_capacity`]) : NaN;
    // qps_bound mixes short service tokens ("decode", "retrieval") with full
    // stage names ("query_expansion_prefill") — match both, or the most
    // common bound stages never get flagged.
    const isBound =
      qpsBound.includes(stage) || (svcKey != null && qpsBound.includes(svcKey));
    const tp = Number(p[`tp_${stage}`]);
    const pp = Number(p[`pp_${stage}`]);
    out.push({
      name: stage,
      qps,
      num_chips: Number.isFinite(chips) ? chips : 0,
      batch_size: Number.isFinite(batch) ? batch : 0,
      latency_s: latency,
      svc_s: Number.isFinite(svc) ? svc : 0,
      qps_capacity: Number.isFinite(cap) ? cap : 0,
      tp: Number.isFinite(tp) ? tp : 0,
      pp: Number.isFinite(pp) ? pp : 0,
      is_bottleneck: isBound,
    });
  }
  return out;
}

export function useEvalBreakdown(
  evalId: number | null,
  pointOverride?: RagoSweepPoint | null,
): EvalBreakdown {
  const { projectBaseUrl } = useProject();
  const [pipelineConfig, setPipelineConfig] =
    useState<Record<string, unknown> | null>(null);
  const [assembly, setAssembly] = useState<AssemblyResult | null>(null);
  const [mapping, setMapping] = useState<TaxonomyStages | null>(null);
  const [loading, setLoading] = useState(false);

  const sweepCache = useRef<Map<string, SweepRow[]>>(new Map());
  const sweepInflight = useRef<Map<string, Promise<SweepRow[]>>>(new Map());
  const [sweepTick, setSweepTick] = useState(0);

  useEffect(() => {
    sweepCache.current.clear();
    sweepInflight.current.clear();
    if (evalId == null || !projectBaseUrl) {
      setPipelineConfig(null);
      setAssembly(null);
      return;
    }
    const client = createDataClient(projectBaseUrl);
    setLoading(true);
    Promise.all([
      client.getPipelineConfig(evalId),
      client.getAssembly(evalId),
      mapping ? Promise.resolve(mapping) : client.getTaxonomyStages(),
    ])
      .then(([pc, asm, m]) => {
        setPipelineConfig(pc);
        setAssembly(asm);
        if (!mapping && m) setMapping(m);
      })
      .finally(() => setLoading(false));
  }, [evalId, projectBaseUrl]);  // eslint-disable-line react-hooks/exhaustive-deps

  const nodeTypeToStages = useMemo(
    () => (mapping ? buildStageToPerformanceStages(mapping) : {}),
    [mapping],
  );

  const pipeline = useMemo(
    () =>
      pipelineConfig && mapping
        ? derivePipeline(pipelineConfig, nodeTypeToStages)
        : [],
    [pipelineConfig, mapping, nodeTypeToStages],
  );

  // Source of point-level params: explicit pointOverride > assembly.best_hw_params.
  const params: Record<string, unknown> | null = useMemo(() => {
    if (pointOverride) return pointOverride as Record<string, unknown>;
    return (assembly?.best_hw_params as Record<string, unknown>) ?? null;
  }, [pointOverride, assembly]);

  const stages = useMemo(() => deriveStagesFromParams(params), [params]);

  const ttft_s = Number(params?.latency_s_ttft ?? 0);
  const e2e_s = Number(params?.latency_s ?? 0);
  const decode_normalized_s = Math.max(0, e2e_s - ttft_s);
  const queue_wait_s = Number(params?.latency_s_queue_wait ?? 0);
  const qps_bound = useMemo(() => parseQpsBound(params?.qps_bound), [params]);

  const getStageSweep = useCallback(
    (stage: string): SweepRow[] | undefined => {
      if (evalId == null || !projectBaseUrl) return undefined;
      const cacheKey = `${projectBaseUrl}:${evalId}:${stage}`;
      const cached = sweepCache.current.get(cacheKey);
      if (cached) return cached;
      if (sweepInflight.current.has(cacheKey)) return undefined;
      const client = createDataClient(projectBaseUrl);
      const p = client
        .getSweepStage(evalId, stage)
        .catch(() => [])
        .then((rows) => {
          sweepCache.current.set(cacheKey, rows);
          sweepInflight.current.delete(cacheKey);
          setSweepTick((t) => t + 1);
          return rows;
        });
      sweepInflight.current.set(cacheKey, p);
      return undefined;
    },
    // sweepTick included so consumers re-render when a fetch resolves
    [evalId, projectBaseUrl, sweepTick],
  );

  if (evalId == null || !projectBaseUrl) return EMPTY;

  return {
    pipeline,
    stages,
    ttft_s,
    e2e_s,
    decode_normalized_s,
    queue_wait_s,
    qps_bound,
    getStageSweep,
    loading,
  };
}
