import { useState, useEffect, useCallback } from "react";
import { fetchFaissBenchmarks } from "@/api/costModelClient";
import type { BenchmarkRow, BenchmarkSummary } from "@/types/costModel";

export function useBenchmarkResults(folder?: string, enabled = true) {
  const [rows, setRows] = useState<BenchmarkRow[]>([]);
  const [summary, setSummary] = useState<BenchmarkSummary[]>([]);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState<Error | null>(null);

  const load = useCallback(async () => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    try {
      setLoading(true);
      const data = await fetchFaissBenchmarks(folder);
      setRows(data.rows);
      setSummary(data.summary);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [enabled, folder]);

  useEffect(() => {
    load();
  }, [load]);

  return { rows, summary, loading, error, refetch: load };
}
