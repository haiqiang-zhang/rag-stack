// TODO(migrate): `@/lib/config` does not yet exist in apps-next. The original
// apps/src/config.ts uses react-native's Platform + localStorage; the web port
// should expose `getAuthToken()` backed by window.localStorage.
import { getAuthToken } from "@/lib/config";

/**
 * Fetch and parse a JSON file from the given URL.
 * Returns null if the file does not exist (404).
 */
export async function fetchJson<T>(url: string): Promise<T | null> {
  const headers: Record<string, string> = {};
  const token = getAuthToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(url, { cache: "no-store", headers });
  if (!res.ok) {
    if (res.status === 404) return null;
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}
