/// <reference types="vite/client" />
/**
 * Application configuration.
 *
 * Backend URL and auth token are stored in localStorage and read dynamically.
 * Defaults to http://localhost:8000 for local dev (override via VITE_API_URL).
 *
 * Ported from apps/src/config.ts. Web-only — `Platform.OS === "web"` checks
 * have been collapsed to direct localStorage access since Vite never runs in
 * a non-browser env.
 */

import { StaticClient } from "@/api/staticClient";
import type { DataClient } from "@/api/client";

const DEFAULT_BACKEND_URL =
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";
const STORAGE_KEY_URL = "rag_stack_backend_url";
const STORAGE_KEY_TOKEN = "rag_stack_token";
const STORAGE_KEY_CONNECTED = "rag_stack_connected";

/** Get the current backend API URL. */
export function getApiUrl(): string {
  return localStorage.getItem(STORAGE_KEY_URL) || DEFAULT_BACKEND_URL;
}

/** Get the current auth token (null if not set). */
export function getAuthToken(): string | null {
  return localStorage.getItem(STORAGE_KEY_TOKEN) || null;
}

/** Save backend connection credentials. */
export function saveConnection(url: string, token: string): void {
  localStorage.setItem(STORAGE_KEY_URL, url.replace(/\/+$/, ""));
  localStorage.setItem(STORAGE_KEY_TOKEN, token);
  localStorage.setItem(STORAGE_KEY_CONNECTED, "1");
}

/** Clear connected flag. URL/token stay for pre-filling the form. */
export function clearConnection(): void {
  localStorage.removeItem(STORAGE_KEY_CONNECTED);
}

/** Check if should auto-reconnect (was previously connected). */
export function hasSavedConnection(): boolean {
  return localStorage.getItem(STORAGE_KEY_CONNECTED) === "1";
}

/** Base URL for project data files */
export function getDataBaseUrl(): string {
  return `${getApiUrl()}/api/data`;
}

/** Polling interval for live mode (ms) */
export const POLL_INTERVAL_MS = 5000;

/** Create a data client for a specific project base URL */
export function createDataClient(baseUrl: string): DataClient {
  return new StaticClient(baseUrl);
}
