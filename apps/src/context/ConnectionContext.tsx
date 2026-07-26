import React, { createContext, useContext, useState, useEffect, useCallback, useMemo } from "react";
import { getApiUrl, getAuthToken, saveConnection, clearConnection, hasSavedConnection } from "@/lib/config";

interface ConnectionState {
  backendUrl: string | null;
  token: string | null;
  isConnected: boolean;
  isChecking: boolean;
  authRequired: boolean | null;
  error: string | null;
  connect: (url: string, token: string) => Promise<boolean>;
  disconnect: () => void;
}

const ConnectionContext = createContext<ConnectionState>({
  backendUrl: null,
  token: null,
  isConnected: false,
  isChecking: true,
  authRequired: null,
  error: null,
  connect: async () => false,
  disconnect: () => {},
});

async function checkAuthStatus(url: string): Promise<{ authRequired: boolean }> {
  const res = await fetch(`${url}/api/auth/status`);
  if (!res.ok) throw new Error("Cannot reach server");
  const data = await res.json();
  return { authRequired: data.auth_required ?? false };
}

async function verifyToken(url: string, token: string): Promise<boolean> {
  const res = await fetch(`${url}/api/auth/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
  if (res.status === 401) return false;
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  const data = await res.json();
  return data.valid === true;
}

export function ConnectionProvider({ children }: { children: React.ReactNode }) {
  const [backendUrl, setBackendUrl] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isChecking, setIsChecking] = useState(true);
  const [authRequired, setAuthRequired] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  // On mount, check saved credentials with timeout
  useEffect(() => {
    if (!hasSavedConnection()) {
      setIsChecking(false);
      return;
    }

    let cancelled = false;
    const savedUrl = getApiUrl();
    const savedToken = getAuthToken();

    const timeout = setTimeout(() => {
      if (!cancelled) { cancelled = true; setIsChecking(false); }
    }, 8000);

    (async () => {
      try {
        const res = await fetch(`${savedUrl}/api/auth/status`);
        if (cancelled) return;
        const data = await res.json();
        const authReq = data.auth_required ?? false;
        setAuthRequired(authReq);

        if (!authReq) {
          setBackendUrl(savedUrl);
          setToken(savedToken);
          setIsConnected(true);
        } else if (savedToken) {
          const vRes = await fetch(`${savedUrl}/api/auth/verify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ token: savedToken }),
          });
          if (cancelled) return;
          if (vRes.ok) {
            const vData = await vRes.json();
            if (vData.valid) {
              setBackendUrl(savedUrl);
              setToken(savedToken);
              setIsConnected(true);
            }
          }
        }
      } catch {
        // Server unreachable
      } finally {
        if (!cancelled) { cancelled = true; clearTimeout(timeout); setIsChecking(false); }
      }
    })();

    return () => { cancelled = true; clearTimeout(timeout); };
  }, []);

  const connect = useCallback(async (url: string, tok: string): Promise<boolean> => {
    setError(null);
    let cleanUrl = url.replace(/\/+$/, "");
    if (!/^https?:\/\//i.test(cleanUrl)) {
      cleanUrl = `http://${cleanUrl}`;
    }
    try {
      // Simple connectivity check — just hit the auth/status endpoint
      const res = await Promise.race([
        fetch(`${cleanUrl}/api/auth/status`),
        new Promise<never>((_, reject) => setTimeout(() => reject(new Error("Connection timed out")), 10000)),
      ]);
      const data = await (res as Response).json();
      const authReq = data.auth_required ?? false;
      setAuthRequired(authReq);

      if (authReq) {
        if (!tok) {
          setError("This server requires an access token");
          return false;
        }
        const verifyRes = await fetch(`${cleanUrl}/api/auth/verify`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token: tok }),
        });
        if (verifyRes.status === 401) {
          setError("Invalid token");
          return false;
        }
      }

      saveConnection(cleanUrl, tok);
      setBackendUrl(cleanUrl);
      setToken(tok);
      setIsConnected(true);
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Cannot connect to server");
      return false;
    }
  }, []);

  const disconnect = useCallback(() => {
    clearConnection();
    setBackendUrl(null);
    setToken(null);
    setIsConnected(false);
    setIsChecking(false);
    setAuthRequired(null);
    setError(null);
  }, []);

  const value = useMemo(
    () => ({ backendUrl, token, isConnected, isChecking, authRequired, error, connect, disconnect }),
    [backendUrl, token, isConnected, isChecking, authRequired, error, connect, disconnect],
  );

  return (
    <ConnectionContext.Provider value={value}>
      {children}
    </ConnectionContext.Provider>
  );
}

export function useConnection() {
  return useContext(ConnectionContext);
}
