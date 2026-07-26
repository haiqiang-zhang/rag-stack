import { useState, useEffect, useRef } from "react";
import { createDataClient } from "@/lib/config";
import { useProject } from "@/context/ProjectContext";
import type { RagoSweepPoint, AssemblyResult } from "@/types";

interface RagoSweepData {
  sweepPoints: RagoSweepPoint[];
  assembly: AssemblyResult | null;
}

/**
 * Fetch RAGO sweep data for a given eval_id.
 * Caches results per (project, eval_id) since sweep data is immutable.
 */
export function useRagoSweep(evalId: number | null) {
  const { projectBaseUrl } = useProject();
  const [data, setData] = useState<RagoSweepData>({ sweepPoints: [], assembly: null });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const cache = useRef<Map<string, RagoSweepData>>(new Map());

  useEffect(() => {
    if (evalId == null || !projectBaseUrl) {
      setData({ sweepPoints: [], assembly: null });
      return;
    }

    const cacheKey = `${projectBaseUrl}:${evalId}`;
    const cached = cache.current.get(cacheKey);
    if (cached) {
      setData(cached);
      return;
    }

    (async () => {
      try {
        setLoading(true);
        const client = createDataClient(projectBaseUrl);
        const [sweepPoints, assembly] = await Promise.all([
          client.getRagoSweep(evalId),
          client.getAssembly(evalId),
        ]);
        const result = { sweepPoints, assembly };
        cache.current.set(cacheKey, result);
        setData(result);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err : new Error(String(err)));
      } finally {
        setLoading(false);
      }
    })();
  }, [evalId, projectBaseUrl]);

  return { ...data, loading, error };
}
