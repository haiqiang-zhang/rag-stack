/**
 * SectionHeader — small uppercase title with optional eyebrow + right-side action.
 */
import type { ReactNode } from "react";

interface SectionHeaderProps {
  title: string;
  eyebrow?: string;
  right?: ReactNode;
}

export function SectionHeader({ title, eyebrow, right }: SectionHeaderProps) {
  return (
    <div className="mb-2.5 flex items-end justify-between gap-3">
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        {eyebrow ? (
          <span className="text-[9px] font-semibold uppercase tracking-[0.9px] text-muted-foreground">
            {eyebrow}
          </span>
        ) : null}
        <span className="truncate text-[11px] font-semibold uppercase tracking-[0.5px] text-foreground">
          {title}
        </span>
      </div>
      {right ? <div className="flex items-center">{right}</div> : null}
    </div>
  );
}
