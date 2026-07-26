import type { DataClient } from "./client";
import type {
  Summary,
  RagoSweepPoint,
  AssemblyResult,
  SweepRow,
  TaxonomyStages,
} from "@/types";
import { fetchJson } from "@/loaders/fetchJson";
import { fetchCsv } from "@/loaders/fetchCsv";
import { fetchTaxonomyStages } from "./registry";

/**
 * Static file client — fetches raw files from a static server
 * and parses them client-side (CSV via papaparse, JSON natively).
 *
 * This is the default implementation. Swap to ApiClient for
 * backend-served data in the future.
 */
export class StaticClient implements DataClient {
  constructor(private baseUrl: string) {}

  async getSummary(): Promise<Summary | null> {
    return fetchJson<Summary>(`${this.baseUrl}/summary.json`);
  }

  async getRagoSweep(evalId: number): Promise<RagoSweepPoint[]> {
    const padded = String(evalId).padStart(4, "0");
    const rows = await fetchCsv<RagoSweepPoint>(
      `${this.baseUrl}/evaluations/eval_${padded}/rago_sweep.csv`
    );
    return rows || [];
  }

  async getAssembly(evalId: number): Promise<AssemblyResult | null> {
    const padded = String(evalId).padStart(4, "0");
    return fetchJson<AssemblyResult>(
      `${this.baseUrl}/evaluations/eval_${padded}/assembly.json`
    );
  }

  async getGtMetrics(
    evalId: number
  ): Promise<Record<string, number> | null> {
    const padded = String(evalId).padStart(4, "0");
    return fetchJson<Record<string, number>>(
      `${this.baseUrl}/evaluations/eval_${padded}/gt_metrics.json`
    );
  }

  async getPipelineConfig(
    evalId: number
  ): Promise<Record<string, unknown> | null> {
    const padded = String(evalId).padStart(4, "0");
    return fetchJson<Record<string, unknown>>(
      `${this.baseUrl}/evaluations/eval_${padded}/pipeline_config.json`
    );
  }

  async getTaxonomyStages(): Promise<TaxonomyStages | null> {
    // Single source of truth: the backend registry API. The stage↔node mapping
    // is global cost-model config (not per-project run data), so it is no
    // longer snapshotted into the project dir as JSON.
    try {
      return await fetchTaxonomyStages();
    } catch {
      return null;
    }
  }

  async getSweepStage(evalId: number, stage: string): Promise<SweepRow[]> {
    const padded = String(evalId).padStart(4, "0");
    const rows = await fetchCsv<Record<string, unknown>>(
      `${this.baseUrl}/evaluations/eval_${padded}/sweep_dfs/${stage}.csv`
    );
    if (!rows) return [];
    const parsed = rows
      .map((row) => {
        const chips =
          row["hardware.num_chips"] != null && row["hardware.num_chips"] !== ""
            ? Number(row["hardware.num_chips"])
            : Number(row["hardware.num_servers"]);
        const tp = row["winning_tp"];
        const pp = row["winning_pp"];
        const steps = row["model.dec_steps"];
        return {
          batch_size: Number(row["batch_size"]),
          num_chips: chips,
          time_ms: Number(row["time_ms"]),
          dec_steps:
            steps != null && steps !== "" ? Number(steps) : undefined,
          winning_tp: tp != null && tp !== "" ? Number(tp) : undefined,
          winning_pp: pp != null && pp !== "" ? Number(pp) : undefined,
        };
      })
      .filter((r) => r.batch_size > 0 && r.time_ms > 0);
    // Retrieval sweeps can carry one row per faiss parallel_mode for the same
    // (chips, batch) cell; the CM picks the best mode per batch, so keep the
    // fastest row per cell.
    const best = new Map<string, SweepRow>();
    for (const r of parsed) {
      const key = `${r.num_chips}:${r.batch_size}`;
      const cur = best.get(key);
      if (!cur || r.time_ms < cur.time_ms) best.set(key, r);
    }
    return [...best.values()];
  }
}
