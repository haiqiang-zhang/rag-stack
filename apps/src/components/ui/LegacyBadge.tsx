/**
 * Legacy Badge + BadgeText — small inline labels with semantic variants.
 *
 * NOTE on filename: this file is `LegacyBadge.tsx` (not `Badge.tsx`) to
 * avoid a case-only filename collision with shadcn's `badge.tsx` that
 * TypeScript rejects under `forceConsistentCasingInFileNames`. The exported
 * symbols (`Badge`, `BadgeText`) are unchanged, and `./index.ts` re-exports
 * them under those names so callers never see the rename.
 *
 * Inlined (i.e. NOT importing shadcn's `badge.tsx`) so the legacy variant
 * palette stays self-contained and doesn't depend on the shadcn cva variants.
 *
 * Legacy API exposed two separate components (`<Badge variant>` for the
 * wrapper background and `<BadgeText variant>` for the inner text color);
 * both names are kept and consumers typically pair them with the same variant.
 */
import * as React from "react";
import { cn } from "@/lib/utils";

type Variant = "accent" | "muted" | "warning" | "error" | "success";

interface BadgeProps extends React.ComponentProps<"span"> {
  variant?: Variant;
}

/** Map legacy variant → Tailwind background + text class pair. */
const WRAPPER_CLASS: Record<Variant, string> = {
  accent: "bg-primary/10 text-primary",
  muted: "bg-muted text-muted-foreground",
  warning:
    "bg-amber-100 text-amber-800 dark:bg-amber-500/15 dark:text-amber-300",
  error: "bg-destructive/10 text-destructive",
  success:
    "bg-emerald-100 text-emerald-800 dark:bg-emerald-500/15 dark:text-emerald-300",
};

export function Badge({
  variant = "accent",
  className,
  children,
  ...rest
}: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex w-fit shrink-0 items-center gap-1 rounded-[5px] px-2 py-0.5",
        "text-[10px] font-semibold whitespace-nowrap",
        WRAPPER_CLASS[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </span>
  );
}

interface BadgeTextProps extends React.ComponentProps<"span"> {
  variant?: Variant;
}

const TEXT_CLASS: Record<Variant, string> = {
  accent: "text-primary",
  muted: "text-muted-foreground",
  warning: "text-amber-700 dark:text-amber-300",
  error: "text-destructive",
  success: "text-emerald-700 dark:text-emerald-300",
};

export function BadgeText({
  variant = "accent",
  className,
  children,
  ...rest
}: BadgeTextProps) {
  return (
    <span
      className={cn(
        "text-[10px] font-semibold",
        TEXT_CLASS[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </span>
  );
}
