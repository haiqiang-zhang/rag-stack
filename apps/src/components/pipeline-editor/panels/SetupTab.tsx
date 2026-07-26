/**
 * SetupTab — the control plane: the three orthogonal switches (eval_backend /
 * performance_source / rag_ir_mode) plus global.n_iterations, the dataset
 * paths, and the optimizer choice. All enums come from the backend registry.
 */
import React from "react";
import { Settings2 } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import type { NonVisualConfig } from "../types";
import type { ConfigRegistries } from "../hooks/useConfigRegistries";
import {
  SectionTitle, Row, TextField, NumberField, SelectField,
} from "./SectionFields";

export interface SectionTabProps {
  config: NonVisualConfig;
  patch: (fn: (c: any) => void) => void;
  registries: ConfigRegistries;
}

const JOINT_INIT_OPTIMIZERS = new Set([
  "ax_qnehvi",
  "ax_qnehvi_mixedgp",
  "agent_qnehvi",
  "agent_qnehvi_slo",
]);

function defaultOptimizerParams(
  type: string,
  current: Record<string, unknown>,
): Record<string, unknown> {
  const params: Record<string, unknown> = { seed: current.seed ?? 42 };
  if (current.objectives !== undefined) params.objectives = current.objectives;
  if (current.composite_objectives !== undefined) {
    params.composite_objectives = current.composite_objectives;
  }
  if (type === "smac_mobo") params.n_gt_init = 10;
  else if (JOINT_INIT_OPTIMIZERS.has(type)) params.n_both_init = 10;
  return params;
}

export function SetupTab({ config, patch, registries, datasetInputsExternal = false }: SectionTabProps & { datasetInputsExternal?: boolean }) {
  const { enums, optimizers } = registries;
  const g = (config.global ?? {}) as Record<string, any>;
  const ds = (config.dataset ?? {}) as Record<string, any>;
  const sys = (config.system ?? {}) as Record<string, any>;
  const opt = (config.optimizer ?? {}) as Record<string, any>;
  const params = (opt.params ?? {}) as Record<string, any>;

  const setG = (k: string, v: unknown) => patch((c) => { c.global = { ...(c.global ?? {}), [k]: v }; });
  const setDs = (k: string, v: unknown) => patch((c) => { c.dataset = { ...(c.dataset ?? {}), [k]: v }; });
  const setSys = (k: string, v: unknown) => patch((c) => { c.system = { ...(c.system ?? {}), [k]: v }; });
  const setOptimizerType = (type: string) =>
    patch((c) => {
      const current = { ...(c.optimizer ?? {}) };
      const currentParams = { ...(current.params ?? {}) };
      c.optimizer = {
        ...current,
        type,
        params: defaultOptimizerParams(type, currentParams),
      };
    });
  const setParam = (k: string, v: unknown) =>
    patch((c) => {
      const o = { ...(c.optimizer ?? {}) };
      o.params = { ...(o.params ?? {}), [k]: v };
      c.optimizer = o;
    });

  return (
    <div className="mx-auto max-w-2xl space-y-7 p-7 pb-12">
      <div className="flex items-center gap-2.5">
        <Settings2 size={18} className="text-primary" />
        <span className="text-lg font-semibold tracking-tight text-foreground">Setup</span>
      </div>

      {/* Three orthogonal switches */}
      <section className="space-y-2">
        <SectionTitle>Mode (the three orthogonal switches)</SectionTitle>
        <Row label="Quality backend" hint="global.eval_backend">
          <SelectField value={g.eval_backend ?? "static_gt"} options={enums.eval_backend}
            onChange={(v) => setG("eval_backend", v)} />
        </Row>
        <Row label="Performance source" hint="system.performance_source">
          <SelectField value={sys.performance_source ?? "cost_model"} options={enums.performance_source}
            onChange={(v) => setSys("performance_source", v)} />
        </Row>
        <Row label="RAG-IR mode" hint="global.rag_ir_mode">
          <SelectField value={g.rag_ir_mode ?? "dynamic"} options={enums.rag_ir_mode}
            onChange={(v) => setG("rag_ir_mode", v)} />
        </Row>
        <Row label="Iterations" hint="global.n_iterations">
          <NumberField value={g.n_iterations} onChange={(v) => setG("n_iterations", v)} placeholder="30" />
        </Row>
      </section>

      <Separator />

      {/* Dataset */}
      <section className="space-y-2">
        <SectionTitle>Dataset</SectionTitle>
        {datasetInputsExternal ? (
          <p className="text-sm text-muted-foreground">Select the dataset, QA and corpus files in the run controls above.</p>
        ) : <>
        <Row label="QA parquet" hint="dataset.qa">
          <TextField value={ds.qa ?? ""} onChange={(v) => setDs("qa", v)} mono
            placeholder="datasets/dragonball/qa_en_sample_100.parquet" />
        </Row>
        <Row label="Corpus parquet" hint="dataset.corpus">
          <TextField value={ds.corpus ?? ""} onChange={(v) => setDs("corpus", v)} mono
            placeholder="datasets/dragonball/raw_corpus_en.parquet" />
        </Row>
        </>}
      </section>

      <Separator />

      {/* Optimizer */}
      <section className="space-y-2">
        <SectionTitle>Optimizer</SectionTitle>
        <Row label="Type" hint="optimizer.type">
          <SelectField value={opt.type} options={optimizers.map((o) => o.key)}
            onChange={setOptimizerType} placeholder="agent_qnehvi" />
        </Row>
        <Row label="Seed" hint="optimizer.params.seed">
          <NumberField value={params.seed} onChange={(v) => setParam("seed", v)} placeholder="42" />
        </Row>
        {JOINT_INIT_OPTIMIZERS.has(opt.type) && (
          <Row label="Joint init" hint="optimizer.params.n_both_init">
            <NumberField value={params.n_both_init} onChange={(v) => setParam("n_both_init", v)} placeholder="10" />
          </Row>
        )}
        {opt.type === "smac_mobo" && (
          <Row label="GT init" hint="optimizer.params.n_gt_init">
            <NumberField value={params.n_gt_init} onChange={(v) => setParam("n_gt_init", v)} placeholder="10" />
          </Row>
        )}
      </section>
    </div>
  );
}
