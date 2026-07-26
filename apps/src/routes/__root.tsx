import { Outlet, createRootRoute } from "@tanstack/react-router";
import { Toaster } from "@/components/ui/sonner";
import { ConnectionProvider } from "@/context/ConnectionContext";
import { ProjectProvider } from "@/context/ProjectContext";
import { SelectionProvider } from "@/context/SelectionContext";
import { ConnectionGate } from "@/components/connection/ConnectionGate";
import { AppShell } from "@/components/layout/AppShell";

export const Route = createRootRoute({
  component: () => (
    <ConnectionProvider>
      <ConnectionGate>
        <ProjectProvider>
          <SelectionProvider>
            <AppShell>
              <Outlet />
            </AppShell>
            <Toaster />
          </SelectionProvider>
        </ProjectProvider>
      </ConnectionGate>
    </ConnectionProvider>
  ),
});
