import { useState, useEffect, useCallback, useRef } from "react";
import { fetchOptimizerBenchmarkResults, fetchOptimizerBenchmarkSummary } from "@/api/costModelClient";
import type { OptimizerBenchmarkRow, OptimizerBenchmarkSummary } from "@/types/optimizer";

/**
 * Stale-while-revalidate semantics:
 *   - `loading` is true only on the very first load (no data yet).
 *   - Subsequent refetches (e.g. after a row deletion) leave the existing
 *     data in place and flip `refreshing` instead, so consumers can
 *     render an inline indicator without unmounting their content. This
 *     keeps the page from collapsing and avoids the browser's scroll
 *     snap-back when content height temporarily shrinks to a spinner.
 */
export function useOptimizerBenchmark(selectedFunction?: string) {
  const [rows, setRows] = useState<OptimizerBenchmarkRow[]>([]);
  const [functions, setFunctions] = useState<string[]>([]);
  const [optimizers, setOptimizers] = useState<string[]>([]);
  const [summary, setSummary] = useState<OptimizerBenchmarkSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const hasLoadedRef = useRef(false);

  const load = useCallback(async () => {
    if (hasLoadedRef.current) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    try {
      const [resultsData, summaryData] = await Promise.all([
        fetchOptimizerBenchmarkResults(selectedFunction),
        fetchOptimizerBenchmarkSummary(),
      ]);
      setRows(resultsData.rows);
      setFunctions(resultsData.functions);
      setOptimizers(resultsData.optimizers);
      setSummary(summaryData);
      setError(null);
      hasLoadedRef.current = true;
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [selectedFunction]);

  useEffect(() => {
    load();
  }, [load]);

  return {
    rows,
    functions,
    optimizers,
    summary,
    loading,
    refreshing,
    error,
    refetch: load,
  };
}
