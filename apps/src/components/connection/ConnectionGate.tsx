import { useConnection } from "@/context/ConnectionContext";
import { ConnectionScreen } from "./ConnectionScreen";

interface ConnectionGateProps {
  children: React.ReactNode;
}

/**
 * Mirrors the original `ConnectionGate` from apps/app/_layout.tsx — show the
 * connection form (or "reconnecting…" spinner) until we know the backend is
 * reachable.
 */
export function ConnectionGate({ children }: ConnectionGateProps) {
  const { isConnected, isChecking } = useConnection();
  if (isChecking || !isConnected) return <ConnectionScreen />;
  return <>{children}</>;
}
