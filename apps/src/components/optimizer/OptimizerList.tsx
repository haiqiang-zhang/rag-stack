/**
 * OptimizerList — registered-optimizer cards (read-only).
 *
 * Web port of apps/src/components/optimizer/OptimizerList.tsx. Each row
 * shows an accent-tinted name prefix chip (first underscore-segment) plus
 * the canonical key, class name, and short description. Visual layout is
 * preserved; styling uses shadcn semantic tokens.
 */
import type { OptimizerInfo } from "@/types/optimizer";

interface Props {
  optimizers: OptimizerInfo[];
}

export function OptimizerList({ optimizers }: Props) {
  if (optimizers.length === 0) {
    return (
      <div className="flex items-center justify-center p-5">
        <span className="text-[13px] text-muted-foreground">
          No optimizers registered
        </span>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-card p-[18px] shadow-sm">
      <div className="mb-3 text-[11px] font-semibold tracking-[0.5px] text-muted-foreground">
        REGISTERED OPTIMIZERS
      </div>
      <div className="flex flex-col gap-2">
        {optimizers.map((o) => (
          <div
            key={o.key}
            className="rounded-[10px] bg-muted p-3"
          >
            <div className="flex items-center gap-2.5">
              <span className="rounded-md bg-primary/10 px-2 py-1 text-[10px] font-bold text-primary">
                {o.key.split("_")[0]}
              </span>
              <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                <div className="flex items-center gap-1.5">
                  <span className="text-[13px] font-semibold text-foreground">
                    {o.key}
                  </span>
                  <span className="text-[11px] text-muted-foreground">
                    {o.class_name}
                  </span>
                </div>
                <span className="line-clamp-2 text-xs text-muted-foreground">
                  {o.description}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
