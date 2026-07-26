import { useCallback, useEffect, useRef, useState } from "react";
import { fetchEvalDistribution } from "@/api/optimizer";
import type { EvalDistributionPayload } from "@/types/optimizer";

/**
 * Stale-while-revalidate fetcher for the eval-distribution payload of one
 * synthetic function. Mirrors useOptimizerBenchmark: `loading` is true only
 * on the first call, refetches flip `refreshing` instead so consumers can
 * keep showing the previous data without a layout collapse.
 *
 * `enabled` (default true) gates the fetch entirely. The OptimizerScreen
 * flips it to true via IntersectionObserver only once the eval-distribution
 * section scrolls into view, so users who never reach that section never
 * pay the cost of reading hundreds of evals CSVs from disk on the backend
 * thread (which can saturate the FastAPI thread pool and make other tabs
 * feel sluggish during the wait).
 *
 * `refetch` re-runs the fetch against the current `functionName` so callers
 * (e.g. a Reload button or a delete handler) can pull in newly written
 * `__evals.csv` files without unmounting the chart. It bypasses `enabled`
 * so an explicit user reload always works.
 */
export function useEvalDistribution(
  functionName: string | undefined,
  options?: { enabled?: boolean },
) {
  const enabled = options?.enabled ?? true;
  const [data, setData] = useState<EvalDistributionPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const hasLoadedRef = useRef(false);
  const cancelRef = useRef<{ cancelled: boolean } | null>(null);

  const load = useCallback(async () => {
    if (!functionName) {
      setData(null);
      hasLoadedRef.current = false;
      return;
    }
    // Cancel any in-flight fetch so its setData doesn't clobber ours.
    if (cancelRef.current) cancelRef.current.cancelled = true;
    const ticket = { cancelled: false };
    cancelRef.current = ticket;

    if (hasLoadedRef.current) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    try {
      const payload = await fetchEvalDistribution(functionName);
      if (ticket.cancelled) return;
      setData(payload);
      setError(null);
      hasLoadedRef.current = true;
    } catch (err) {
      if (ticket.cancelled) return;
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      if (!ticket.cancelled) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, [functionName]);

  // Initial fetch on mount / functionName change — only when enabled.
  useEffect(() => {
    if (!enabled) return;
    load();
    return () => {
      if (cancelRef.current) cancelRef.current.cancelled = true;
    };
  }, [load, enabled]);

  // Poll while the server is still computing the true Pareto in a
  // background thread. Polling stops if the gate is closed.
  useEffect(() => {
    if (!enabled || !data?.true_pareto_pending) return;
    const id = setInterval(() => {
      load();
    }, 3000);
    return () => clearInterval(id);
  }, [enabled, data?.true_pareto_pending, load]);

  return { data, loading, refreshing, error, refetch: load };
}
