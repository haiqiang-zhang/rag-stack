import { useEffect, useMemo, useRef, useState } from "react";
import { createDataClient } from "@/lib/config";
import { useProject } from "@/context/ProjectContext";
import type { RagoSweepPoint } from "@/types";

const CONCURRENCY = 6;
// Batch setState calls while workers stream in. Without a flush window we
// fire one setCache per fetch — 200+ fetches produce 200+ renders in quick
// succession, which cascades through the context tree and can trip React's
// "Maximum update depth" guard via downstream effects.
const FLUSH_MS = 80;

/**
 * Eagerly load rago_sweep_*.csv for every eval id in `evalIds`. Cached by
 * (project, evalId) so switching views does not re-fetch.
 *
 * Returns a Map<evalId, sweepRows>; missing entries (still loading or 404)
 * simply won't appear. Callers should fall back to `ev.performance` when an
 * eval's sweep is not yet in the map.
 */
export function useAllRagoSweeps(evalIds: number[]) {
  const { projectBaseUrl } = useProject();
  const [cache, setCache] = useState<Map<number, RagoSweepPoint[]>>(new Map());
  // "Resolved", not "loaded": counts ids whose fetch finished either way —
  // CM-only evals legitimately have no sweep CSV and still count.
  const [resolvedCount, setResolvedCount] = useState(0);
  const cacheRef = useRef<Map<string, RagoSweepPoint[]>>(new Map());

  // Stable key so the effect only re-runs when the actual id set changes,
  // not on every upstream ref bump.
  const key = useMemo(() => evalIds.join(","), [evalIds]);

  useEffect(() => {
    if (!projectBaseUrl || evalIds.length === 0) {
      setCache((prev) => (prev.size === 0 ? prev : new Map()));
      setResolvedCount(0);
      return;
    }

    let cancelled = false;
    const client = createDataClient(projectBaseUrl);
    const nextCache = new Map<number, RagoSweepPoint[]>();

    // Track attempts separately from successes. `cache.size` only grows for
    // eval_ids whose sweep CSV exists, but most CM-only evals don't ship a
    // sweep — counting "attempted" lets `sweepsReady` actually fire when
    // every id has been resolved one way or the other.
    const attempted = new Set<number>();

    // Seed from ref cache. Cache hits count as both succeeded AND attempted.
    for (const id of evalIds) {
      const hit = cacheRef.current.get(`${projectBaseUrl}:${id}`);
      if (hit) {
        nextCache.set(id, hit);
        attempted.add(id);
      }
    }
    setCache((prev) => {
      if (prev.size === nextCache.size) {
        let same = true;
        for (const [k, v] of nextCache) {
          if (prev.get(k) !== v) {
            same = false;
            break;
          }
        }
        if (same) return prev;
      }
      return new Map(nextCache);
    });
    setResolvedCount(attempted.size);

    const pending = evalIds.filter(
      (id) => !cacheRef.current.has(`${projectBaseUrl}:${id}`),
    );
    if (pending.length === 0) return;

    // Debounced flush so that many sweeps landing in a short window only
    // produce a handful of renders instead of one per fetch.
    let flushTimer: ReturnType<typeof setTimeout> | null = null;
    const scheduleFlush = () => {
      if (flushTimer != null || cancelled) return;
      flushTimer = setTimeout(() => {
        flushTimer = null;
        if (cancelled) return;
        setCache(new Map(nextCache));
        setResolvedCount(attempted.size);
      }, FLUSH_MS);
    };

    let cursor = 0;
    const worker = async () => {
      while (!cancelled && cursor < pending.length) {
        const id = pending[cursor++];
        try {
          const rows = await client.getRagoSweep(id);
          if (cancelled) return;
          cacheRef.current.set(`${projectBaseUrl}:${id}`, rows);
          nextCache.set(id, rows);
        } catch {
          // skip — eval has no sweep CSV (CM-only iter or pre-assembly)
        }
        attempted.add(id);
        scheduleFlush();
      }
    };

    const workers = Array.from(
      { length: Math.min(CONCURRENCY, pending.length) },
      worker,
    );
    Promise.all(workers).then(() => {
      if (cancelled) return;
      if (flushTimer != null) {
        clearTimeout(flushTimer);
        flushTimer = null;
      }
      // Final reconciliation so the state is guaranteed complete.
      setCache(new Map(nextCache));
      setResolvedCount(attempted.size);
    });

    return () => {
      cancelled = true;
      if (flushTimer != null) clearTimeout(flushTimer);
    };
  }, [projectBaseUrl, key]);

  // totalCount derives from the unique input set so it matches what the
  // worker actually attempts (deduped by SweepCacheBridge upstream too,
  // belt + braces). 200-element set construction per render is negligible.
  const totalCount = new Set(evalIds).size;
  return { sweeps: cache, resolvedCount, totalCount };
}
