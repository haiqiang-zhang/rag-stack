import { useState, useEffect, useCallback } from "react";
import { fetchOptimizerRegistry } from "@/api/costModelClient";
import type { OptimizerInfo } from "@/types/optimizer";

export function useOptimizerRegistry() {
  const [optimizers, setOptimizers] = useState<OptimizerInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const data = await fetchOptimizerRegistry();
      setOptimizers(data);
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

  return { optimizers, loading, error, refetch: load };
}
