/**
 * SystemTab — the `system` section: hardware target, the cost-model RAGO
 * deployment sweep (cm_search_space), the measured-mode search space (shown
 * only when performance_source=measured), and the FAISS runtime knobs.
 * Deep CPU/GPU specs stay editable in the YAML tab; this form covers the
 * commonly-swept fields.
 */
import React from "react";
import { Cpu } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import type { SectionTabProps } from "./SetupTab";
import {
  SectionTitle, Row, TextField, NumberField, SelectField, BoolField, ChipMultiSelect, ListField,
} from "./SectionFields";

export function SystemTab({ config, patch, registries }: SectionTabProps) {
  const { enums } = registries;
  const sys = (config.system ?? {}) as Record<string, any>;
  const cm = (sys.cm_search_space ?? {}) as Record<string, any>;
  const ms = (sys.measured_search_space ?? {}) as Record<string, any>;
  const ret = (sys.retrieval ?? {}) as Record<string, any>;
  const po = (sys.performance_objectives ?? {}) as Record<string, any>;
  const measured = sys.performance_source === "measured";

  const setSys = (k: string, v: unknown) => patch((c) => { c.system = { ...(c.system ?? {}), [k]: v }; });
  const setNested = (block: string, k: string, v: unknown) =>
    patch((c) => {
      const s = { ...(c.system ?? {}) };
      s[block] = { ...(s[block] ?? {}), [k]: v };
      if (v === undefined) delete s[block][k];
      c.system = s;
    });

  return (
    <div className="mx-auto max-w-2xl space-y-7 p-7 pb-12">
      <div className="flex items-center gap-2.5">
        <Cpu size={18} className="text-primary" />
        <span className="text-lg font-semibold tracking-tight text-foreground">System</span>
      </div>

      <section className="space-y-2">
        <SectionTitle>Hardware & performance</SectionTitle>
        <Row label="Performance source" hint="system.performance_source">
          <SelectField value={sys.performance_source ?? "cost_model"} options={enums.performance_source}
            onChange={(v) => setSys("performance_source", v)} />
        </Row>
        <Row label="GPU" hint="system.gpu (GenZ preset name)">
          <TextField value={typeof sys.gpu === "string" ? sys.gpu : ""} onChange={(v) => setSys("gpu", v)}
            mono placeholder="A100_80GB_GPU" />
        </Row>
        <Row label="Selection" hint="system.performance_objectives.selection">
          <SelectField value={po.selection} options={enums.performance_selection}
            onChange={(v) => setNested("performance_objectives", "selection", v)} placeholder="max_throughput" />
        </Row>
      </section>

      <Separator />

      <section className="space-y-2">
        <SectionTitle>Cost-model deployment sweep (cm_search_space)</SectionTitle>
        <Row label="Placement policy" hint="placement_policy">
          <ChipMultiSelect options={["disaggregated", "collocated"]}
            value={Array.isArray(cm.placement_policy) ? cm.placement_policy : []}
            onChange={(v) => setNested("cm_search_space", "placement_policy", v)} />
        </Row>
        <Row label="Min / Max GPUs" hint="min_num_gpus / max_num_gpus">
          <div className="flex gap-2">
            <NumberField value={cm.min_num_gpus} onChange={(v) => setNested("cm_search_space", "min_num_gpus", v)} placeholder="min" />
            <NumberField value={cm.max_num_gpus} onChange={(v) => setNested("cm_search_space", "max_num_gpus", v)} placeholder="max" />
          </div>
        </Row>
        <Row label="Max GPUs / stage" hint="max_num_gpus_per_stage">
          <NumberField value={cm.max_num_gpus_per_stage} onChange={(v) => setNested("cm_search_space", "max_num_gpus_per_stage", v)} />
        </Row>
        <Row label="GPUs / server" hint="gpus_per_server">
          <NumberField value={cm.gpus_per_server} onChange={(v) => setNested("cm_search_space", "gpus_per_server", v)} />
        </Row>
        <Row label="Max batch / req" hint="max_batch_size_request">
          <NumberField value={cm.max_batch_size_request} onChange={(v) => setNested("cm_search_space", "max_batch_size_request", v)} />
        </Row>
        <Row label="No microbatching" hint="no_microbatching">
          <BoolField value={cm.no_microbatching !== false} onChange={(v) => setNested("cm_search_space", "no_microbatching", v)} />
        </Row>
      </section>

      {measured && (
        <>
          <Separator />
          <section className="space-y-2">
            <SectionTitle>Measured search space</SectionTitle>
            <Row label="Available GPUs" hint="available_gpus (comma-separated)">
              <ListField value={ms.available_gpus} onChange={(v) => setNested("measured_search_space", "available_gpus", v)}
                placeholder="cuda:0, cuda:1" />
            </Row>
            <Row label="Batch size / req" hint="batch_size_request (one global batch)">
              <ListField value={ms.batch_size_request} numeric
                onChange={(v) => setNested("measured_search_space", "batch_size_request", v)} placeholder="1, 8, 16, 32" />
            </Row>
            <Row label="KV cache dtype" hint="vllm_kv_cache_dtype">
              <TextField value={typeof ms.vllm_kv_cache_dtype === "string" ? ms.vllm_kv_cache_dtype : ""}
                onChange={(v) => setNested("measured_search_space", "vllm_kv_cache_dtype", v)} mono placeholder="auto" />
            </Row>
          </section>
        </>
      )}

      <Separator />

      <section className="space-y-2">
        <SectionTitle>Retrieval runtime knobs (system.retrieval)</SectionTitle>
        <Row label="FAISS threads" hint="faiss_num_threads (comma-separated to sweep)">
          <ListField value={Array.isArray(ret.faiss_num_threads) ? ret.faiss_num_threads : ret.faiss_num_threads != null ? [ret.faiss_num_threads] : undefined}
            numeric onChange={(v) => setNested("retrieval", "faiss_num_threads", v.length === 1 ? v[0] : v)} placeholder="1" />
        </Row>
        <Row label="FAISS parallel mode" hint="faiss_parallel_mode (0=inter, 1=intra)">
          <ListField value={Array.isArray(ret.faiss_parallel_mode) ? ret.faiss_parallel_mode : ret.faiss_parallel_mode != null ? [ret.faiss_parallel_mode] : undefined}
            numeric onChange={(v) => setNested("retrieval", "faiss_parallel_mode", v.length === 1 ? v[0] : v)} placeholder="0, 1" />
        </Row>
      </section>
    </div>
  );
}
