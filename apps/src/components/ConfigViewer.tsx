/**
 * ConfigViewer — read-only, syntactically tinted YAML view of a config object.
 *
 * Renders one `<div>` per source line with a sticky left line-number gutter.
 * Key/colon/value get distinct text colors (using semantic Tailwind tokens
 * that play with both light and dark themes).
 */
import yaml from "js-yaml";

interface ConfigViewerProps {
  config: Record<string, unknown>;
  title?: string;
}

export function ConfigViewer({ config, title }: ConfigViewerProps) {
  if (!config) return null;
  const yamlStr = yaml.dump(config, { indent: 2, lineWidth: 120 });
  const lines = yamlStr.split("\n").filter((l) => l.length > 0);

  return (
    <div>
      {title && (
        <div className="mb-2 text-[13px] font-semibold text-foreground">
          {title}
        </div>
      )}
      <div className="rounded-[10px] border border-border bg-muted/40 p-3">
        <div className="max-h-[400px] overflow-auto">
          {lines.map((line, i) => {
            const trimmed = line.trimStart();
            const isKey = trimmed.includes(":") && !trimmed.startsWith("-");
            const colonIdx = trimmed.indexOf(":");
            const indent = line.substring(0, line.indexOf(trimmed));

            return (
              <div
                key={i}
                className="flex items-start py-px font-mono text-xs"
              >
                <span className="w-7 select-none pr-2 text-right text-[11px] text-muted-foreground/70">
                  {i + 1}
                </span>
                <span className="mx-2 w-px shrink-0 self-stretch bg-border" />
                {isKey ? (
                  <span className="flex-1 whitespace-pre text-muted-foreground">
                    {indent}
                    <span className="text-primary">
                      {trimmed.substring(0, colonIdx)}
                    </span>
                    <span className="text-muted-foreground">:</span>
                    <span className="text-foreground">
                      {trimmed.substring(colonIdx + 1)}
                    </span>
                  </span>
                ) : (
                  <span className="flex-1 whitespace-pre text-muted-foreground">
                    {line}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
