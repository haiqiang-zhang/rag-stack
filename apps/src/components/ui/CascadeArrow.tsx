import { ChevronDown } from "lucide-react";

interface CascadeArrowProps {
  label?: string;
  onClick?: () => void;
}

export function CascadeArrow({ label, onClick }: CascadeArrowProps) {
  return (
    <div
      onClick={onClick}
      className={`flex flex-col items-center gap-1 py-2.5 ${
        onClick ? "cursor-pointer" : "cursor-default"
      }`}
    >
      <div className="h-2.5 w-px bg-border opacity-60" />
      <ChevronDown className="size-3.5 text-muted-foreground opacity-70" />
      {label ? (
        <span className="font-mono text-[10px] tracking-[0.4px] text-muted-foreground">
          {label}
        </span>
      ) : null}
    </div>
  );
}
