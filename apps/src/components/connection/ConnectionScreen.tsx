/**
 * ConnectionScreen — backend handshake before the app loads.
 *
 * Layout:
 *   - md+ : two-column. Left = brand panel (gradient mesh + product blurb +
 *           feature bullets). Right = the connect form.
 *   - <md : single column, brand block on top, form below.
 *
 * The connect input is a "command-bar" composed of [prompt prefix] + [URL
 * input] + [Connect button] welded into one bordered row — the novelty
 * borrows the same affordance used in Dataset's run prompt and the cost-
 * model edit tools, making "Connect" feel like submitting a command.
 *
 * No card chrome. Token field is hidden behind a disclosure link to keep
 * the cold path (URL only) one input wide.
 */
import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  KeyRound,
  Layers,
  Loader2,
  ShieldCheck,
  Terminal,
} from "lucide-react";
import { useConnection } from "@/context/ConnectionContext";
import { getApiUrl, getAuthToken } from "@/lib/config";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export function ConnectionScreen() {
  const { connect, disconnect, error, isChecking } = useConnection();
  const [url, setUrl] = useState(() => getApiUrl() || "");
  const [token, setToken] = useState(() => getAuthToken() || "");
  const [showToken, setShowToken] = useState(() => Boolean(getAuthToken()));
  const [connecting, setConnecting] = useState(false);

  const handleConnect = async () => {
    if (!url.trim()) return;
    setConnecting(true);
    await connect(url.trim(), token.trim());
    setConnecting(false);
  };

  if (isChecking) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <BrandMark pulse />
          <Loader2 className="size-5 animate-spin text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Reconnecting…</p>
          <Button variant="ghost" size="sm" onClick={() => disconnect()}>
            Cancel
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="grid min-h-screen grid-cols-1 bg-background md:grid-cols-[1.1fr_1fr]">
      {/* Left: brand panel */}
      <BrandPanel />

      {/* Right: connect form */}
      <div className="relative flex items-center justify-center overflow-hidden px-6 py-12 md:px-12">
        {/* Ambient backdrop — no text. Sparse dotted grid + two soft
            radial blurs in opposite corners give the panel weight that
            balances the active vector field on the left without
            competing with it. */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-[0.35]"
          style={{
            backgroundImage:
              "radial-gradient(circle, var(--color-border) 1px, transparent 1px)",
            backgroundSize: "32px 32px",
          }}
        />
        <div
          aria-hidden
          className="pointer-events-none absolute -right-32 -top-32 size-[440px] rounded-full bg-primary/[0.07] blur-3xl"
        />
        <div
          aria-hidden
          className="pointer-events-none absolute -bottom-40 -left-24 size-[380px] rounded-full bg-primary/[0.05] blur-3xl"
        />
        {/* Hairline horizon line — a single accent stroke that grounds
            the form without adding any glyph or copy. */}
        <div
          aria-hidden
          className="pointer-events-none absolute left-0 right-0 top-1/2 h-px -translate-y-1/2 bg-gradient-to-r from-transparent via-border to-transparent opacity-60"
        />

        <div className="relative w-full max-w-md space-y-8">
          <div className="space-y-2 md:hidden">
            <BrandMark />
            <h1 className="text-2xl font-semibold tracking-tight">RAG-Stack</h1>
          </div>

          <div className="space-y-1">
            <h2 className="text-2xl font-semibold tracking-tight">
              Connect a backend
            </h2>
            <p className="text-sm text-muted-foreground">
              Point the app at the API server you&apos;re running.
            </p>
          </div>

          {/* Command-bar for the URL */}
          <div className="space-y-3">
            <div className="group flex items-stretch overflow-hidden rounded-md border border-border focus-within:border-ring focus-within:ring-2 focus-within:ring-ring/40">
              <span className="flex shrink-0 items-center gap-1.5 border-r border-border bg-muted/60 px-3 font-mono text-xs font-medium text-muted-foreground">
                <Terminal className="size-3 text-primary" />
                connect&nbsp;→
              </span>
              <Input
                id="backend-url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="http://localhost:8000"
                autoCapitalize="none"
                autoCorrect="off"
                spellCheck={false}
                className="flex-1 rounded-none border-0 bg-transparent px-3 font-mono text-[13px] shadow-none focus-visible:ring-0"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleConnect();
                }}
              />
              <Button
                type="button"
                onClick={handleConnect}
                disabled={!url.trim() || connecting}
                className="rounded-none border-0"
              >
                {connecting ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <>
                    Connect
                    <ArrowRight className="size-4" strokeWidth={2.5} />
                  </>
                )}
              </Button>
            </div>

            {/* Quick-start suggestions */}
            {!url.trim() && (
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] uppercase tracking-wider text-muted-foreground/70">
                  Try
                </span>
                {["http://localhost:8000", "http://127.0.0.1:8000"].map(
                  (s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setUrl(s)}
                      className="rounded border border-border px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground"
                    >
                      {s}
                    </button>
                  ),
                )}
              </div>
            )}
          </div>

          {/* Token disclosure */}
          {showToken ? (
            <div className="space-y-2">
              <label
                htmlFor="access-token"
                className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
              >
                <KeyRound className="size-3" />
                Access token
              </label>
              <Input
                id="access-token"
                type="password"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="Bearer …"
                autoCapitalize="none"
                autoCorrect="off"
                spellCheck={false}
                className="font-mono text-[13px]"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleConnect();
                }}
              />
              {!token && (
                <button
                  type="button"
                  onClick={() => {
                    setShowToken(false);
                    setToken("");
                  }}
                  className="text-[11px] text-muted-foreground hover:text-foreground"
                >
                  No token needed
                </button>
              )}
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setShowToken(true)}
              className="inline-flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
            >
              <KeyRound className="size-3" />
              Backend requires a token
            </button>
          )}

          {/* Error */}
          {error && (
            <div
              role="alert"
              className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs font-medium text-destructive"
            >
              {error}
            </div>
          )}

          {/* Footer */}
          <div className="flex items-center gap-1.5 border-t border-border pt-4 text-[11px] text-muted-foreground">
            <ShieldCheck className="size-3" />
            <span>Token is stored in browser localStorage.</span>
          </div>
        </div>
      </div>
    </div>
  );
}

/* -- Brand left panel ----------------------------------------------- */

/**
 * Background dots become a vector field that points toward the cursor.
 * Each grid cell renders a dot when far away and gradually grows into a
 * directional line as the cursor approaches; intensity falls off with
 * distance and the field smoothly relaxes back to dots when the cursor
 * leaves the panel.
 */
function BrandPanel() {
  const cursorRef = useRef<{ x: number; y: number } | null>(null);

  const handleMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    cursorRef.current = {
      x: e.clientX - r.left,
      y: e.clientY - r.top,
    };
  };

  return (
    <div
      onMouseMove={handleMove}
      onMouseLeave={() => {
        cursorRef.current = null;
      }}
      className="relative hidden min-h-screen overflow-hidden border-r border-border bg-muted/30 md:block"
    >
      <VectorField cursorRef={cursorRef} />

      <div className="relative flex h-full flex-col justify-between p-12">
        {/* Brand — top-left anchor */}
        <div className="flex items-center gap-3">
          <BrandMark />
          <span className="text-base font-semibold tracking-tight text-foreground">
            RAG-Stack
          </span>
        </div>
      </div>
    </div>
  );
}

/* -- Interactive vector field --------------------------------------- */

function VectorField({
  cursorRef,
}: {
  cursorRef: React.MutableRefObject<{ x: number; y: number } | null>;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    // Read design-system colors so the field tracks the active theme.
    const styles = getComputedStyle(document.documentElement);
    const cssVar = (name: string, fallback: string) =>
      styles.getPropertyValue(name).trim() || fallback;
    const primary = cssVar("--color-primary", "oklch(0.205 0 0)");
    const idleColor = cssVar("--color-foreground", "#0a0a0a");

    const dpr = Math.max(1, window.devicePixelRatio || 1);
    let w = 0;
    let h = 0;

    const sync = () => {
      // Read CSS size from the element. Because the canvas has
      // `size-full + inset-0`, this tracks the parent panel and grows as
      // layout settles or the viewport resizes.
      const rect = canvas.getBoundingClientRect();
      const nw = Math.max(1, Math.floor(rect.width));
      const nh = Math.max(1, Math.floor(rect.height));
      if (nw === w && nh === h) return;
      w = nw;
      h = nh;
      // Only the drawing buffer is set imperatively; CSS handles display.
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const ro = new ResizeObserver(() => sync());
    ro.observe(canvas);

    const SPACING = 22;
    const RADIUS = 260;
    const MAX_LEN = 11;

    let smooth: { x: number; y: number } | null = null;
    let strength = 0; // 0 = idle (dots only), 1 = full vector field
    let frameId = 0;
    let running = true;

    const draw = () => {
      if (!running) return;
      sync();
      ctx.clearRect(0, 0, w, h);

      const target = cursorRef.current;

      // Smoothly interpolate cursor + activity strength.
      if (target) {
        if (!smooth || strength < 0.04) {
          smooth = { x: target.x, y: target.y };
        } else {
          smooth.x += (target.x - smooth.x) * 0.18;
          smooth.y += (target.y - smooth.y) * 0.18;
        }
      }
      const targetStrength = target ? 1 : 0;
      strength += (targetStrength - strength) * 0.06;

      const showVectors = smooth && strength > 0.01;

      ctx.lineWidth = 1;
      ctx.lineCap = "round";

      for (let x = SPACING / 2; x < w; x += SPACING) {
        for (let y = SPACING / 2; y < h; y += SPACING) {
          let drewVector = false;
          if (showVectors && smooth) {
            const dx = smooth.x - x;
            const dy = smooth.y - y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < RADIUS) {
              const t = (1 - dist / RADIUS) * strength;
              if (t > 0.02) {
                const angle = Math.atan2(dy, dx);
                const len = MAX_LEN * t;

                ctx.strokeStyle = primary;
                ctx.globalAlpha = 0.18 + 0.55 * t;
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(x + Math.cos(angle) * len, y + Math.sin(angle) * len);
                ctx.stroke();

                ctx.fillStyle = primary;
                ctx.globalAlpha = 0.32 + 0.45 * t;
                ctx.beginPath();
                ctx.arc(x, y, 1.1, 0, Math.PI * 2);
                ctx.fill();

                drewVector = true;
              }
            }
          }
          if (!drewVector) {
            ctx.fillStyle = idleColor;
            ctx.globalAlpha = 0.13;
            ctx.beginPath();
            ctx.arc(x, y, 0.9, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
      ctx.globalAlpha = 1;
      frameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      running = false;
      ro.disconnect();
      cancelAnimationFrame(frameId);
    };
  }, [cursorRef]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className="pointer-events-none absolute inset-0 size-full"
    />
  );
}

function BrandMark({ pulse = false }: { pulse?: boolean }) {
  return (
    <div className="relative inline-flex">
      {pulse && (
        <span
          aria-hidden
          className="absolute inset-0 animate-ping rounded-xl bg-primary/30"
        />
      )}
      <span className="relative flex size-12 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-lg">
        <Layers className="size-5" strokeWidth={2.5} />
      </span>
    </div>
  );
}

