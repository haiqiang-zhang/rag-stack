/**
 * PageHeader — top bar slot for routes. Forwards children + className
 * so callers can place left/right groups inside however they like.
 *
 * Companion helpers:
 *   - <PageHeaderTitle icon={Icon} title="…" badge={…} />  — left side
 *   - <PageHeaderActions>…</PageHeaderActions>             — right side
 *
 * Use these in every route header so icon color, title weight, and gap are
 * uniform. Don't pass inline `style={{ color }}` on header icons — the
 * shadcn neutral palette uses `text-foreground` everywhere.
 */
import * as React from "react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { SidebarTrigger } from "@/components/ui/sidebar";

export function PageHeader({
  className,
  children,
  ...rest
}: React.ComponentProps<"header">) {
  return (
    <header
      className={cn(
        // Fixed height (not `min-h`) so the chrome doesn't grow when a tab
        // strip / shadcn TabsList (h-9) appears in the right slot. 52px is
        // tight; bumped to 56px so tab strip (36px) + py-2.5 (20px) fits.
        "flex h-14 select-none items-center justify-between border-b border-border bg-card px-5 py-2",
        className,
      )}
      {...rest}
    >
      {children}
    </header>
  );
}

interface PageHeaderTitleProps {
  icon?: LucideIcon;
  title: React.ReactNode;
  /** Optional pill / count badge rendered after the title. */
  badge?: React.ReactNode;
  /** Optional content rendered before the icon (e.g. a back button). */
  prefix?: React.ReactNode;
  className?: string;
}

export function PageHeaderTitle({
  icon: Icon,
  title,
  badge,
  prefix,
  className,
}: PageHeaderTitleProps) {
  return (
    <div className={cn("flex flex-1 min-w-0 items-center gap-2.5", className)}>
      {/* Mobile-only sidebar drawer trigger — sits in the same flex row as
          icon/title so vertical alignment is handled by `items-center`. The
          `-ml-1` pulls it slightly left to compensate for the header's
          `px-5` padding without overlapping the title. */}
      <SidebarTrigger className="md:hidden -ml-1 size-8 shrink-0" />
      {prefix}
      {Icon && <Icon className="size-4 shrink-0 text-foreground" />}
      <span className="truncate text-base font-semibold tracking-tight text-foreground">
        {title}
      </span>
      {badge}
    </div>
  );
}

export function PageHeaderActions({
  className,
  children,
  ...rest
}: React.ComponentProps<"div">) {
  return (
    <div className={cn("flex items-center gap-1.5", className)} {...rest}>
      {children}
    </div>
  );
}
