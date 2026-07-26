import { Construction } from "lucide-react";

interface RouteStubProps {
  title: string;
  path: string;
}

/**
 * Placeholder shown while the route is still being migrated from the old
 * Expo + Tamagui app. Each migrated route replaces this with its real
 * component.
 */
export function RouteStub({ title, path }: RouteStubProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full p-12 gap-3">
      <Construction size={48} className="text-muted-foreground" />
      <h1 className="text-2xl font-bold">{title}</h1>
      <p className="text-sm text-muted-foreground text-center max-w-md">
        This route hasn&apos;t been migrated yet. The original lives at{" "}
        <code className="font-mono text-xs px-1.5 py-0.5 bg-muted rounded">
          {path}
        </code>{" "}
        in the old <code>apps/</code> tree.
      </p>
    </div>
  );
}
