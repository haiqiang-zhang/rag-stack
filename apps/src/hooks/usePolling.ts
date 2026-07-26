import { useState, useEffect, useRef, useCallback } from "react";

interface UsePollingOptions {
  /** Polling interval in ms. Set to 0 to disable polling. */
  intervalMs: number;
  /** Whether polling is enabled */
  enabled?: boolean;
  /** When this key changes, data is reset and a fresh fetch fires immediately. */
  key?: string;
}

interface UsePollingResult<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Generic polling hook.
 *
 * Calls the fetcher function immediately, then re-calls at the given interval.
 * Stops polling when enabled=false or intervalMs=0.
 * When `key` changes, clears stale data and re-fetches immediately.
 */
export function usePolling<T>(
  fetcher: () => Promise<T | null>,
  options: UsePollingOptions
): UsePollingResult<T> {
  const { intervalMs, enabled = true, key = "" } = options;
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const doFetch = useCallback(async () => {
    try {
      const result = await fetcherRef.current();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;

    // Reset state so stale data from the previous key is cleared immediately.
    setData(null);
    setLoading(true);
    setError(null);

    doFetch();

    if (intervalMs <= 0) return;

    const timer = setInterval(doFetch, intervalMs);
    return () => clearInterval(timer);
  }, [doFetch, intervalMs, enabled, key]);

  return { data, loading, error, refetch: doFetch };
}
