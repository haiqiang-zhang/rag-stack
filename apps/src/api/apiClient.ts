import type { DataClient } from "./client";
import type {
  Summary,
  RagoSweepPoint,
  AssemblyResult,
  SweepRow,
  TaxonomyStages,
} from "@/types";
import { fetchJson } from "@/loaders/fetchJson";
import { fetchTaxonomyStages } from "./registry";

/**
 * REST API client — fetches pre-parsed JSON from a backend server.
 *
 * Future implementation. The backend returns typed JSON directly,
 * so no client-side CSV/YAML parsing is needed.
 *
 * To activate, change the export in config.ts:
 *   export const dataClient = new ApiClient('http://localhost:8000');
 */
export class ApiClient implements DataClient {
  constructor(private baseUrl: string) {}

  async getSummary(): Promise<Summary | null> {
    return fetchJson<Summary>(`${this.baseUrl}/api/summary`);
  }

  async getRagoSweep(evalId: number): Promise<RagoSweepPoint[]> {
    const data = await fetchJson<RagoSweepPoint[]>(
      `${this.baseUrl}/api/rago-sweep/${evalId}`
    );
    return data || [];
  }

  async getAssembly(evalId: number): Promise<AssemblyResult | null> {
    return fetchJson<AssemblyResult>(
      `${this.baseUrl}/api/evaluations/${evalId}/assembly`
    );
  }

  async getGtMetrics(evalId: number): Promise<Record<string, number> | null> {
    return fetchJson<Record<string, number>>(
      `${this.baseUrl}/api/evaluations/${evalId}/gt-metrics`
    );
  }

  async getPipelineConfig(
    evalId: number
  ): Promise<Record<string, unknown> | null> {
    return fetchJson<Record<string, unknown>>(
      `${this.baseUrl}/api/evaluations/${evalId}/pipeline-config`
    );
  }

  async getTaxonomyStages(): Promise<TaxonomyStages | null> {
    // Single source of truth: the backend registry API (not a per-project
    // snapshot). Global config, so it uses the registry endpoint directly.
    try {
      return await fetchTaxonomyStages();
    } catch {
      return null;
    }
  }

  async getSweepStage(evalId: number, stage: string): Promise<SweepRow[]> {
    const data = await fetchJson<SweepRow[]>(
      `${this.baseUrl}/api/evaluations/${evalId}/sweep/${stage}`
    );
    return data || [];
  }
}
