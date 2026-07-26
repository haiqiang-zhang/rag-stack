import type {
  Summary,
  Evaluation,
  RagoSweepPoint,
  AssemblyResult,
  SweepRow,
  TaxonomyStages,
} from "@/types";

/**
 * Abstract data client interface.
 *
 * All data access goes through this interface. Components and hooks
 * never know whether data comes from static files or a REST API.
 *
 * To swap implementations, change the export in config.ts.
 *
 * Note: there is deliberately no getPareto() — the server's
 * pareto_frontier.csv is its frozen default-policy frontier; the UI's
 * authoritative frontier is computed client-side (utils/pareto.computePareto)
 * under the live quality formula + performance policy. The evaluations list
 * also lives elsewhere (api/ragProject.fetchProjectEvaluations) because its
 * CSV cells need server-side Python-repr parsing.
 */
export interface DataClient {
  getSummary(): Promise<Summary | null>;
  getRagoSweep(evalId: number): Promise<RagoSweepPoint[]>;
  getAssembly(evalId: number): Promise<AssemblyResult | null>;
  getGtMetrics(evalId: number): Promise<Record<string, number> | null>;
  getPipelineConfig(evalId: number): Promise<Record<string, unknown> | null>;
  getTaxonomyStages(): Promise<TaxonomyStages | null>;
  getSweepStage(evalId: number, stage: string): Promise<SweepRow[]>;
}
