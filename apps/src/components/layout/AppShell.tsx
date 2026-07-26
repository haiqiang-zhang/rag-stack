import { Link, useRouterState } from "@tanstack/react-router";
import {
  BarChart3,
  Cpu,
  Zap,
  Database,
  GitBranch,
  Settings,
  Layers,
  Archive,
  ChevronRight,
  PanelLeftClose,
  PanelLeftOpen,
  Wifi,
  Bot,
  Key,
  type LucideIcon,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarProvider,
  SidebarRail,
  useSidebar,
} from "@/components/ui/sidebar";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { GlobalJobsCenter } from "@/components/jobs/GlobalJobsCenter";

interface NavItem {
  path: string;
  title: string;
  icon: LucideIcon;
}

const NAV_ITEMS: NavItem[] = [
  { path: "/", title: "RAG Optimize", icon: BarChart3 },
  { path: "/pipeline-editor", title: "Opt Configs", icon: GitBranch },
  { path: "/cost-models", title: "Cost Models", icon: Cpu },
  { path: "/optimizer", title: "Benchmark", icon: Zap },
  { path: "/dataset-generator", title: "Dataset", icon: Database },
  { path: "/quality-stores", title: "Quality Stores", icon: Archive },
];

// Settings sub-sections — these match SECTIONS in SettingsScreen.tsx. Each
// links to /settings?section=<key>; SettingsScreen reads the search param
// and switches its active rail accordingly.
type SettingsSectionKey =
  | "connection"
  | "generators"
  | "keys"
  | "rag"
  | "dataset"
  | "cost-model"
  | "optimizer";

const SETTINGS_SECTIONS: {
  key: SettingsSectionKey;
  title: string;
  icon: LucideIcon;
}[] = [
  { key: "connection", title: "Connection", icon: Wifi },
  { key: "generators", title: "Generator Clients", icon: Bot },
  { key: "keys", title: "Keys & Secrets", icon: Key },
  { key: "rag", title: "RAG Optimization", icon: BarChart3 },
  { key: "dataset", title: "Dataset Generation", icon: Database },
  { key: "cost-model", title: "Cost Model", icon: Cpu },
  { key: "optimizer", title: "Optimizer", icon: Zap },
];

interface AppShellProps {
  children: React.ReactNode;
}

function CollapseToggle() {
  const { state, toggleSidebar, isMobile } = useSidebar();
  if (isMobile) return null;
  const collapsed = state === "collapsed";
  const Icon = collapsed ? PanelLeftOpen : PanelLeftClose;
  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        onClick={toggleSidebar}
        tooltip={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        <Icon />
        <span>{collapsed ? "Expand" : "Collapse"}</span>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}

/**
 * Settings nav — collapsible parent with a sub-list, matching the canonical
 * shadcn sidebar pattern. Sub-items link to `/settings?section=<key>`; the
 * Settings screen reads that param and switches its active rail.
 */
function SettingsNav({
  pathname,
  search,
}: {
  pathname: string;
  search: Record<string, unknown>;
}) {
  const isOnSettings = pathname === "/settings";
  const activeSection = (search.section as string | undefined) ?? "connection";
  return (
    <Collapsible
      asChild
      defaultOpen={isOnSettings}
      className="group/collapsible"
    >
      <SidebarMenuItem>
        <CollapsibleTrigger asChild>
          <SidebarMenuButton tooltip="Settings" isActive={isOnSettings}>
            <Settings />
            <span>Settings</span>
            <ChevronRight className="ml-auto size-4 transition-transform group-data-[state=open]/collapsible:rotate-90 group-data-[collapsible=icon]:hidden" />
          </SidebarMenuButton>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <SidebarMenuSub>
            {SETTINGS_SECTIONS.map((sec) => (
              <SidebarMenuSubItem key={sec.key}>
                <SidebarMenuSubButton
                  asChild
                  isActive={isOnSettings && activeSection === sec.key}
                >
                  <Link to="/settings" search={{ section: sec.key }}>
                    <span>{sec.title}</span>
                  </Link>
                </SidebarMenuSubButton>
              </SidebarMenuSubItem>
            ))}
          </SidebarMenuSub>
        </CollapsibleContent>
      </SidebarMenuItem>
    </Collapsible>
  );
}

export function AppShell({ children }: AppShellProps) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const search = useRouterState({ select: (s) => s.location.search });
  const isActive = (path: string) =>
    pathname === path || (path === "/" && pathname === "");

  return (
    // Default shadcn SidebarProvider (min-h-svh) — let pages decide their
    // own scroll model. Pages that need viewport-bounded internal scrolling
    // (RAG optimization, pipeline editor) lock at their own screen root;
    // pages that want classic document scrolling (settings, lists) keep
    // it. Earlier I locked here globally and it broke every other route.
    <SidebarProvider>
      <Sidebar collapsible="icon">
        <SidebarHeader>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton asChild size="lg" tooltip="RAG-Stack">
                <Link to="/">
                  <div className="bg-sidebar-primary text-sidebar-primary-foreground flex aspect-square size-8 items-center justify-center rounded-lg group-data-[collapsible=icon]:size-6 group-data-[collapsible=icon]:rounded-md">
                    <Layers className="size-4 group-data-[collapsible=icon]:size-3.5" />
                  </div>
                  <div className="grid flex-1 text-left text-sm leading-tight group-data-[collapsible=icon]:hidden">
                    <span className="truncate font-semibold">RAG-Stack</span>
                  </div>
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarHeader>

        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupLabel>Navigation</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {NAV_ITEMS.map((item) => (
                  <SidebarMenuItem key={item.path}>
                    <SidebarMenuButton
                      asChild
                      isActive={isActive(item.path)}
                      tooltip={item.title}
                    >
                      <Link to={item.path}>
                        <item.icon />
                        <span>{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
                {/* Settings — collapsible parent with sub-sections */}
                <SettingsNav pathname={pathname} search={search} />
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>

        <SidebarFooter>
          <SidebarMenu>
            <GlobalJobsCenter />
            <CollapseToggle />
          </SidebarMenu>
        </SidebarFooter>

        <SidebarRail />
      </Sidebar>

      <SidebarInset className="min-w-0">
        {/* Mobile sidebar trigger is rendered inline by each route's
            <PageHeaderTitle> (left-aligned with the page icon row). Settings
            — the only route without PageHeader — has its own inline trigger. */}
        <div className="flex-1 min-h-0 min-w-0 flex flex-col">{children}</div>
      </SidebarInset>
    </SidebarProvider>
  );
}
