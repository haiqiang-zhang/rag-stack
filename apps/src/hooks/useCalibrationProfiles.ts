import { useCallback, useEffect, useState } from "react";
import { fetchCalibrationProfiles } from "@/api/costModelClient";
import type { CalibrationProfileRecord } from "@/types/costModel";

/**
 * Fetch calibration profiles for a given cost model. Returns ``null`` when
 * the cost model returns HTTP 501 (no on-disk profile store). Callers use
 * that signal to render a "not supported" placeholder.
 */
export function useCalibrationProfiles(costModel: string | null) {
  const [profiles, setProfiles] = useState<CalibrationProfileRecord[] | null>(null);
  const [supported, setSupported] = useState<boolean>(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const load = useCallback(async () => {
    if (!costModel) return;
    try {
      setLoading(true);
      const out = await fetchCalibrationProfiles(costModel);
      if (out == null) {
        setSupported(false);
        setProfiles(null);
      } else {
        setSupported(true);
        setProfiles(out);
      }
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [costModel]);

  useEffect(() => { load(); }, [load]);

  return { profiles, supported, loading, error, refetch: load };
}
