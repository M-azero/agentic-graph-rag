import clsx from "clsx";
import type { ReactNode } from "react";

/** One figure in the rail across the top of a page. */
export function Stat({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: "positive" | "caution" | "danger";
}) {
  return (
    <div className="rounded-lg border border-border bg-surface px-4 py-3 shadow-card">
      <p className="eyebrow">{label}</p>
      <p
        className={clsx(
          "mt-1 text-2xl font-semibold tabular-nums",
          tone === "positive" && "text-positive",
          tone === "caution" && "text-caution",
          tone === "danger" && "text-danger",
          !tone && "text-strong",
        )}
      >
        {value}
      </p>
      {sub && <p className="mt-0.5 text-2xs text-muted">{sub}</p>}
    </div>
  );
}

/** The rail itself. Wraps rather than scrolls, so nothing hides off-screen. */
export function StatRail({ children }: { children: ReactNode }) {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">{children}</div>
  );
}

/** A service's up/down light. */
export function StatusDot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-sm text-body">
      <span
        aria-hidden
        className={clsx("h-2 w-2 rounded-full", ok ? "bg-positive" : "bg-danger")}
      />
      {label}
      <span className="sr-only">{ok ? "up" : "down"}</span>
    </span>
  );
}
