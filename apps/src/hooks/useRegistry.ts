import { useState, useEffect, useCallback } from "react";
import { fetchRegistry } from "@/api/costModelClient";
import type { CostModelInfo } from "@/types/costModel";

export function useRegistry() {
  const [models, setModels] = useState<CostModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const data = await fetchRegistry();
      setModels(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return { models, loading, error, refetch: load };
}
