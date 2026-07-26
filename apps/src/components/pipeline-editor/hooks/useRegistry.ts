import { useState, useEffect } from "react";
import { fetchNodeTypeRegistry, fetchVectorDbTypes } from "@/api/costModelClient";
import type { NodeTypeRegistry } from "../types";

export interface RegistryData {
  nodeTypes: NodeTypeRegistry;
  vectorDbTypes: string[];
  loading: boolean;
  error: string | null;
}

export function useRegistry(): RegistryData {
  const [nodeTypes, setNodeTypes] = useState<NodeTypeRegistry>({});
  const [vectorDbTypes, setVectorDbTypes] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchNodeTypeRegistry(), fetchVectorDbTypes()])
      .then(([nt, vdb]) => {
        if (cancelled) return;
        setNodeTypes(nt);
        setVectorDbTypes(vdb);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  return { nodeTypes, vectorDbTypes, loading, error };
}
