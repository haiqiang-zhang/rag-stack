import Papa from "papaparse";
import { getAuthToken } from "@/lib/config";

/**
 * Fetch and parse a CSV file from the given URL.
 * Returns an array of objects with column headers as keys.
 * Returns null if the file does not exist (404).
 */
export async function fetchCsv<T extends Record<string, unknown>>(
  url: string
): Promise<T[] | null> {
  const headers: Record<string, string> = {};
  const token = getAuthToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(url, { cache: "no-store", headers });
  if (!res.ok) {
    if (res.status === 404) return null;
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  const text = await res.text();
  const result = Papa.parse<T>(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  return result.data;
}
