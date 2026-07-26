import { useState, useEffect, useCallback } from "react";
import { fetchFaissProfiles } from "@/api/costModelClient";
import type { CalibrationProfile } from "@/types/costModel";

export function useProfiles(enabled = true) {
  const [profiles, setProfiles] = useState<CalibrationProfile[]>([]);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState<Error | null>(null);

  const load = useCallback(async () => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    try {
      setLoading(true);
      const data = await fetchFaissProfiles();
      setProfiles(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [enabled]);

  useEffect(() => {
    load();
  }, [load]);

  return { profiles, loading, error, refetch: load };
}
