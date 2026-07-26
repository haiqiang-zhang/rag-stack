import { useEffect, useState } from "react";
import type { CSSProperties } from "react";

export function formatConfigValue(value: unknown): string {
  if (value === undefined) return "";
  if (typeof value !== "string") return JSON.stringify(value);
  // Quote strings that would otherwise be read as JSON numbers/booleans/etc.
  try { JSON.parse(value); return JSON.stringify(value); }
  catch { return !value || /^[\[{\"]/.test(value) ? JSON.stringify(value) : value; }
}

export function parseConfigValue(raw: string): unknown {
  const trimmed = raw.trim();
  if (!trimmed) return undefined;
  try { return JSON.parse(trimmed); }
  catch {
    // A partially typed container must never become a scalar config value.
    if (/^[\[{\"]/.test(trimmed)) throw new Error("Enter valid JSON for a list, range, or quoted string.");
    return raw;
  }
}

/** Commit on blur/Enter so intermediate keystrokes cannot erase a selector. */
export function ConfigValueInput({ value, onChange, validate, placeholder, style, label }: {
  value: unknown;
  onChange: (value: unknown) => void;
  validate?: (value: unknown) => boolean;
  placeholder?: string;
  style?: CSSProperties;
  label?: string;
}) {
  const formatted = formatConfigValue(value);
  const [draft, setDraft] = useState(formatted);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { setDraft(formatted); setError(null); }, [formatted]);
  const commit = () => {
    if (draft === formatted) return;
    try {
      const next = parseConfigValue(draft);
      if (validate && !validate(next)) throw new Error("Enter a number, list of numbers, or range object.");
      setError(null);
      onChange(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <input
        aria-label={label}
        aria-invalid={!!error}
        type="text"
        value={draft}
        onChange={(event) => { setDraft(event.target.value); setError(null); }}
        onBlur={commit}
        onKeyDown={(event) => { if (event.key === "Enter") commit(); }}
        placeholder={placeholder}
        style={{ width: "100%", ...style }}
      />
      {error && <div role="alert" style={{ color: "#B91C1C", fontSize: 10 }}>{error}</div>}
    </div>
  );
}
