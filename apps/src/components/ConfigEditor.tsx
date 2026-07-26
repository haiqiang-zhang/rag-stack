/**
 * ConfigEditor — flush CodeMirror 6 YAML editor.
 *
 * Designed to fill its container edge-to-edge with no internal frame: the
 * panel's own border (from the parent layout) IS the visual boundary, and
 * the editor's `.cm-scroller` is the single source of overflow scrolling
 * for the entire panel.
 *
 * Self-contained sizing: this component renders its own `relative` wrapper
 * and pins CodeMirror absolutely inside it, so callers don't need to know
 * about the absolute-positioning trick. The only contract is that the
 * parent must give this component a definite height — either via flex
 * (`flex-1 min-h-0`) or a fixed `h-[Npx]` / `h-svh` etc. Without a
 * bounded height the editor collapses to 0px tall.
 */
import { useEffect, useState } from "react";
import CodeMirror, { EditorView } from "@uiw/react-codemirror";
import { yaml as yamlLang } from "@codemirror/lang-yaml";
import { githubLight, githubDark } from "@uiw/codemirror-theme-github";

interface ConfigEditorProps {
  value: string;
  onChangeText: (text: string) => void;
  /** Optional: render read-only. Default false. */
  readOnly?: boolean;
}

/** Detect Tailwind's `class="dark"` on <html> — the rest of the app uses
 *  the same class strategy via `dark:` utility variants. */
function useIsDark(): boolean {
  const [dark, setDark] = useState(() =>
    typeof document !== "undefined" &&
    document.documentElement.classList.contains("dark"),
  );
  useEffect(() => {
    if (typeof document === "undefined") return;
    const root = document.documentElement;
    const observer = new MutationObserver(() => {
      setDark(root.classList.contains("dark"));
    });
    observer.observe(root, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);
  return dark;
}

// Inline theme overrides — strip the github theme's residual decoration
// (focus outline, default borders, font) so the editor reads as one with
// the surrounding shadcn panel.
const FLUSH_OVERRIDES = EditorView.theme({
  "&": {
    height: "100%",
    fontSize: "12.5px",
  },
  "&.cm-focused": {
    outline: "none",
  },
  ".cm-scroller": {
    fontFamily:
      "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace",
  },
  ".cm-gutters": {
    backgroundColor: "transparent",
    border: "none",
  },
  ".cm-activeLineGutter, .cm-activeLine": {
    backgroundColor: "transparent",
  },
});

export function ConfigEditor({
  value,
  onChangeText,
  readOnly = false,
}: ConfigEditorProps) {
  const dark = useIsDark();

  return (
    // `relative` here gives CodeMirror's `absolute inset-0` a positioning
    // anchor regardless of where the caller embeds us — so a bare
    // `<div className="mt-4"><ConfigEditor /></div>` no longer makes the
    // editor escape to the viewport. Caller still must provide a definite
    // height on this wrapper (flex-1 min-h-0 or fixed h-…).
    <div className="relative h-full w-full overflow-hidden">
      <CodeMirror
        value={value}
        onChange={onChangeText}
        theme={dark ? githubDark : githubLight}
        height="100%"
        extensions={[yamlLang(), EditorView.lineWrapping, FLUSH_OVERRIDES]}
        basicSetup={{
          lineNumbers: true,
          foldGutter: true,
          highlightActiveLine: !readOnly,
          highlightActiveLineGutter: false,
          highlightSelectionMatches: true,
          bracketMatching: true,
          autocompletion: false,
          searchKeymap: true,
        }}
        readOnly={readOnly}
        className="absolute inset-0"
      />
    </div>
  );
}
