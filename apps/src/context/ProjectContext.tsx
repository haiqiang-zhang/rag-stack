import React, { createContext, useContext, useState, useEffect, useCallback, useRef, useMemo } from "react";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { getDataBaseUrl } from "@/lib/config";
import { fetchJson } from "@/loaders/fetchJson";
import type { Summary } from "@/types";

export interface ProjectInfo {
  name: string;
  status?: "created" | "running" | "completed" | "failed";
  n_evaluations?: number;
  timestamp?: string;
}

export type ProjectSortKey = "name" | "status" | "evaluations" | "time";
export type SortDirection = "asc" | "desc";

const STATUS_ORDER: Record<string, number> = {
  running: 0,
  created: 1,
  completed: 2,
  failed: 3,
};

function sortProjects(
  projects: ProjectInfo[],
  sortKey: ProjectSortKey,
  direction: SortDirection,
): ProjectInfo[] {
  const dir = direction === "asc" ? 1 : -1;
  return [...projects].sort((a, b) => {
    let cmp: number;
    switch (sortKey) {
      case "status":
        cmp = (STATUS_ORDER[a.status ?? ""] ?? 9) - (STATUS_ORDER[b.status ?? ""] ?? 9);
        break;
      case "evaluations":
        cmp = (a.n_evaluations ?? 0) - (b.n_evaluations ?? 0);
        break;
      case "time":
        cmp = (a.timestamp ?? "").localeCompare(b.timestamp ?? "");
        break;
      default:
        cmp = a.name.localeCompare(b.name);
    }
    return cmp * dir;
  });
}

interface ProjectState {
  projects: ProjectInfo[];
  currentProject: string | null;
  projectBaseUrl: string | null;
  setCurrentProject: (name: string) => void;
  refreshProjects: () => Promise<void>;
  loading: boolean;
  sortKey: ProjectSortKey;
  sortDirection: SortDirection;
  setSortKey: (key: ProjectSortKey) => void;
  setSortDirection: (dir: SortDirection) => void;
}

const ProjectContext = createContext<ProjectState>({
  projects: [],
  currentProject: null,
  projectBaseUrl: null,
  setCurrentProject: () => {},
  refreshProjects: async () => {},
  loading: true,
  sortKey: "name",
  sortDirection: "asc",
  setSortKey: () => {},
  setSortDirection: () => {},
});

/**
 * Discover project directories from the data endpoint.
 * A project is any folder with a summary.json (any status: created, running, completed, failed).
 */
async function discoverProjects(): Promise<ProjectInfo[]> {
  try {
    const res = await fetch(getDataBaseUrl(), { cache: "no-store" });
    if (!res.ok) return [];

    const entries: string[] = await res.json();
    const folders = entries.filter((e) => typeof e === "string" && !e.includes("."));

    if (folders.length === 0) {
      const rootSummary = await fetchJson<Summary>(`${getDataBaseUrl()}/summary.json`);
      if (rootSummary) {
        return [{ name: "", status: rootSummary.status as ProjectInfo["status"], n_evaluations: rootSummary.n_evaluations, timestamp: rootSummary.timestamp }];
      }
      return [];
    }

    const projects: ProjectInfo[] = [];
    await Promise.all(
      folders.map(async (name) => {
        const summary = await fetchJson<Summary>(`${getDataBaseUrl()}/${name}/summary.json`);
        if (summary) {
          projects.push({
            name,
            status: summary.status as ProjectInfo["status"],
            n_evaluations: summary.n_evaluations,
            timestamp: summary.timestamp,
          });
        }
      })
    );

    projects.sort((a, b) => a.name.localeCompare(b.name));
    return projects;
  } catch {
    return [];
  }
}

export function ProjectProvider({ children }: { children: React.ReactNode }) {
  const [rawProjects, setRawProjects] = useState<ProjectInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState<ProjectSortKey>("time");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  // URL-driven project selection: `/?project=foo` → currentProject = "foo".
  // The URL is the source of truth so each project view is bookmarkable
  // and the browser back/forward buttons work as expected. The selection
  // is only meaningful on the RAG Optimize route (`/`); on other routes
  // (e.g. /optimizer, /cost-models) `currentProject` reads as null.
  const navigate = useNavigate();
  const search = useRouterState({
    select: (s) => s.location.search as { project?: string },
  });
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const currentProject = pathname === "/" ? (search.project ?? null) : null;
  const setCurrentProject = useCallback(
    (name: string | null) => {
      navigate({
        to: "/",
        search: (prev: Record<string, unknown>) => ({
          ...prev,
          project: name && name.length > 0 ? name : undefined,
          preview: undefined,
        }),
      });
    },
    [navigate],
  );

  const projects = useMemo(() => sortProjects(rawProjects, sortKey, sortDirection), [rawProjects, sortKey, sortDirection]);

  const refreshProjects = useCallback(async () => {
    try {
      const found = await discoverProjects();
      setRawProjects(found);
    } catch {
      setRawProjects([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshProjects();
  }, []);

  // Poll: 5s when any project is running, 30s otherwise
  const hasRunning = projects.some((p) => p.status === "running");
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    const ms = hasRunning ? 5_000 : 30_000;
    intervalRef.current = setInterval(() => { refreshProjects(); }, ms);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [hasRunning, refreshProjects]);

  const projectBaseUrl =
    currentProject !== null
      ? currentProject === ""
        ? getDataBaseUrl()
        : `${getDataBaseUrl()}/${currentProject}`
      : null;

  return (
    <ProjectContext.Provider
      value={{ projects, currentProject, projectBaseUrl, setCurrentProject, refreshProjects, loading, sortKey, sortDirection, setSortKey, setSortDirection }}
    >
      {children}
    </ProjectContext.Provider>
  );
}

export function useProject() {
  return useContext(ProjectContext);
}
