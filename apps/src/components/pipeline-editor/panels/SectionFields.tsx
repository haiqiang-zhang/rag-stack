/**
 * Small reusable field primitives for the 7-section config forms
 * (Setup / System / Quality). Thin wrappers over shadcn inputs so the section
 * tabs stay declarative — they hold no schema rules (those come from the
 * registry + the server-side validator).
 */
import React from "react";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

export function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <span className="block text-[10px] font-bold uppercase tracking-[1px] text-muted-foreground">
      {children}
    </span>
  );
}

export function Row({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 py-1">
      <div className="w-[150px] shrink-0 pt-2">
        <span className="text-[11px] font-semibold tracking-[0.3px] text-muted-foreground">{label}</span>
        {hint && <span className="mt-0.5 block text-[10px] leading-tight text-muted-foreground/60">{hint}</span>}
      </div>
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  );
}

const inputCls =
  "w-full rounded-lg border border-border bg-background px-3 py-2 text-[13px] text-foreground outline-none focus:border-primary/50";

export function TextField({
  value, onChange, placeholder, mono,
}: { value: string; onChange: (v: string) => void; placeholder?: string; mono?: boolean }) {
  return (
    <input
      className={cn(inputCls, mono && "font-mono text-xs")}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
    />
  );
}

export function NumberField({
  value, onChange, placeholder,
}: { value: number | undefined; onChange: (v: number | undefined) => void; placeholder?: string }) {
  return (
    <input
      type="number"
      className={cn(inputCls, "font-mono text-xs")}
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value === "" ? undefined : Number(e.target.value))}
      placeholder={placeholder}
    />
  );
}

export function SelectField({
  value, options, onChange, placeholder,
}: { value: string | undefined; options: string[]; onChange: (v: string) => void; placeholder?: string }) {
  return (
    <Select value={value ?? ""} onValueChange={onChange}>
      <SelectTrigger className="w-full">
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {options.map((o) => (
          <SelectItem key={o} value={o}>{o}</SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export function BoolField({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="pt-1.5">
      <Switch checked={value} onCheckedChange={onChange} />
    </div>
  );
}

/** Toggle-chip multi-select over a fixed option set. */
export function ChipMultiSelect({
  options, value, onChange,
}: { options: string[]; value: string[]; onChange: (v: string[]) => void }) {
  const set = new Set(value);
  const toggle = (o: string) => {
    const next = new Set(set);
    next.has(o) ? next.delete(o) : next.add(o);
    onChange(options.filter((x) => next.has(x)));
  };
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map((o) => {
        const on = set.has(o);
        return (
          <button
            key={o}
            type="button"
            onClick={() => toggle(o)}
            className={cn(
              "cursor-pointer rounded-md border px-2 py-1 text-[11px] transition-colors",
              on
                ? "border-primary/40 bg-primary/10 font-medium text-primary"
                : "border-border bg-background text-muted-foreground hover:bg-muted",
            )}
          >
            {o}
          </button>
        );
      })}
      {options.length === 0 && <span className="text-[11px] italic text-muted-foreground">no options</span>}
    </div>
  );
}

/** Comma-separated list <-> array of numbers or strings. */
export function ListField({
  value, onChange, numeric, placeholder,
}: { value: unknown[] | undefined; onChange: (v: unknown[]) => void; numeric?: boolean; placeholder?: string }) {
  const text = Array.isArray(value) ? value.join(", ") : "";
  return (
    <input
      className={cn(inputCls, "font-mono text-xs")}
      value={text}
      onChange={(e) => {
        const parts = e.target.value.split(",").map((s) => s.trim()).filter(Boolean);
        onChange(numeric ? parts.map(Number).filter((n) => !Number.isNaN(n)) : parts);
      }}
      placeholder={placeholder}
    />
  );
}
