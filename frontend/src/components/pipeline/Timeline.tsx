// The waterfall. The diagram shows what connects to what; this shows what
// happened when — which is the only view where the three retrieval legs
// visibly overlap, and where a slow stage is obvious rather than inferred.

import clsx from "clsx";

import type { TraceStep } from "../../api";
import { formatMs } from "../../lib/pipeline";

const BAR_CLASS: Record<string, string> = {
  ok: "bg-accent",
  running: "bg-accent/50 animate-pulse",
  error: "bg-danger",
  skipped: "bg-border",
};

export function Timeline({
  steps,
  selected,
  onSelect,
}: {
  steps: TraceStep[];
  selected: string | null;
  onSelect: (name: string) => void;
}) {
  if (steps.length === 0) return null;

  // The scale is the whole run, so bars are comparable to each other rather
  // than each filling its own row.
  const span = Math.max(
    ...steps.map((s) => (s.ended_ms ?? s.started_ms) || 0),
    1,
  );

  return (
    <section className="glass rounded-xl p-4">
      <h2 className="mb-3 text-sm font-semibold text-strong">
        Timeline
        <span className="ml-2 font-normal text-muted">
          {steps.length} steps over {formatMs(span)}
        </span>
      </h2>
      <ol className="flex flex-col gap-0.5">
        {steps.map((step) => {
          const start = (step.started_ms / span) * 100;
          const end = ((step.ended_ms ?? step.started_ms) / span) * 100;
          return (
            <li key={step.id}>
              <button
                onClick={() => onSelect(step.name)}
                className={clsx(
                  "flex w-full items-center gap-3 rounded px-1.5 py-1 text-left transition-colors",
                  selected === step.name ? "bg-raised" : "hover:bg-raised",
                )}
              >
                <span
                  className="w-40 shrink-0 truncate text-xs text-body"
                  // Nesting depth is the cheapest way to show that a retrieval
                  // leg belongs to a tool call belongs to the agent.
                  style={{ paddingLeft: step.parent_id ? 10 : 0 }}
                >
                  {step.label}
                </span>
                <span className="relative h-3 min-w-0 flex-1 rounded bg-canvas">
                  <span
                    className={clsx(
                      "absolute inset-y-0 rounded",
                      BAR_CLASS[step.status] ?? "bg-border",
                    )}
                    style={{
                      left: `${start}%`,
                      // A floor, so a sub-millisecond step is still a visible
                      // mark rather than nothing at all.
                      width: `${Math.max(end - start, 0.6)}%`,
                    }}
                  />
                </span>
                <span className="w-16 shrink-0 text-right font-mono text-2xs text-muted">
                  {formatMs(step.duration_ms)}
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
