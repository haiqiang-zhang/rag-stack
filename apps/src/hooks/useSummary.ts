import { createDataClient, POLL_INTERVAL_MS } from "@/lib/config";
import { useProject } from "@/context/ProjectContext";
import { usePolling } from "./usePolling";
import type { Summary } from "@/types";

/**
 * Fetch summary.json with auto-polling for the current project.
 */
export function useSummary() {
  const { projectBaseUrl } = useProject();

  const result = usePolling<Summary>(
    () => {
      if (!projectBaseUrl) return Promise.resolve(null);
      return createDataClient(projectBaseUrl).getSummary();
    },
    { intervalMs: POLL_INTERVAL_MS, enabled: !!projectBaseUrl, key: projectBaseUrl ?? "" }
  );

  return {
    ...result,
    isRunning: result.data?.status === "running",
    isCompleted: result.data?.status === "completed",
  };
}
