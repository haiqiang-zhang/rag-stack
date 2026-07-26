import { useEffect, useRef, useState } from "react";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface DebouncedSliderProps {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  /** Called after the user stops sliding for `delayMs` ms. */
  onCommit: (v: number) => void;
  delayMs?: number;
  width?: number;
}

/**
 * Range slider that updates its own visible value instantly but only invokes
 * `onCommit` after the user pauses for `delayMs` milliseconds. Use this
 * anywhere the committed value triggers expensive recomputation (e.g. global
 * Pareto recompute across hundreds of sweeps).
 *
 * Built on shadcn `<Slider>` (Radix Slider primitive) — keyboard-accessible,
 * theme-aware (track/range/thumb match the shadcn neutral palette), and far
 * better than the native `<input type="range">` whose `accent-color` was the
 * only thing we could style.
 */
export function DebouncedSlider({
  label,
  value,
  min = 0,
  max = 2,
  step = 0.1,
  onCommit,
  delayMs = 200,
  width = 100,
}: DebouncedSliderProps) {
  const [local, setLocal] = useState(value);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastCommitted = useRef(value);

  // Sync from parent when not actively editing
  useEffect(() => {
    if (timer.current == null && value !== local) {
      setLocal(value);
      lastCommitted.current = value;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  useEffect(() => {
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  const handleChange = (v: number) => {
    setLocal(v);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      timer.current = null;
      if (v !== lastCommitted.current) {
        lastCommitted.current = v;
        onCommit(v);
      }
    }, delayMs);
  };

  return (
    <div className="flex items-center gap-2.5">
      <span className="flex-1 truncate text-[11px] text-muted-foreground">
        {label}
      </span>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[local]}
        onValueChange={(vs) => handleChange(vs[0] ?? local)}
        style={{ width }}
      />
      <span className="w-8 text-right text-[11px] font-medium tabular-nums text-foreground">
        {local.toFixed(1)}
      </span>
    </div>
  );
}

interface DebouncedNumberInputProps {
  value: number | "";
  onCommit: (v: number | undefined) => void;
  step?: number;
  min?: number;
  delayMs?: number;
  width?: number;
  placeholder?: string;
  borderColor?: string;
}

/**
 * Number input that debounces commit. Keeps local string state so users can
 * type freely without triggering recompute on every keystroke.
 */
export function DebouncedNumberInput({
  value,
  onCommit,
  step = 0.01,
  min,
  delayMs = 400,
  width = 90,
  placeholder = "none",
  borderColor,
}: DebouncedNumberInputProps) {
  const [local, setLocal] = useState<string>(value === "" ? "" : String(value));
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastCommitted = useRef<number | "">(value);

  useEffect(() => {
    if (timer.current == null) {
      const next = value === "" ? "" : String(value);
      setLocal(next);
      lastCommitted.current = value;
    }
  }, [value]);

  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  const handleChange = (raw: string) => {
    setLocal(raw);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      timer.current = null;
      if (raw === "") {
        if (lastCommitted.current !== "") {
          lastCommitted.current = "";
          onCommit(undefined);
        }
      } else {
        const num = parseFloat(raw);
        if (!Number.isNaN(num) && num !== lastCommitted.current) {
          lastCommitted.current = num;
          onCommit(num);
        }
      }
    }, delayMs);
  };

  return (
    <Input
      type="number"
      step={step}
      min={min}
      value={local}
      onChange={(e) => handleChange(e.target.value)}
      placeholder={placeholder}
      style={{ width }}
      className={cn("h-7 px-2 py-1 text-[11px]", borderColor && "border")}
    />
  );
}
