// What one block received and returned. The point of the whole page: a reader
// clicks a box on the diagram and sees the actual values, not a description of
// them.

import clsx from "clsx";
import { AlertTriangle, Check, Clock, MinusCircle } from "lucide-react";
import { useState } from "react";

import type { PipelineNode, TraceStep } from "../../api";
import { formatMs, type NodeRun } from "../../lib/pipeline";

interface Props {
  node: PipelineNode | null;
  run: NodeRun | undefined;
}

export function StepInspector({ node, run }: Props) {
  // Which call of a repeated block is showing. Reset by keying the component on
  // the node name at the call site, which is simpler than syncing an effect.
  const [index, setIndex] = useState(0);

  if (!node) {
    return (
      <aside className="glass rounded-xl p-4">
        <p className="text-sm text-muted">
          Pick a block in the diagram to see what it received and what it returned.
        </p>
      </aside>
    );
  }

  const steps = run?.steps ?? [];
  const step = steps[Math.min(index, steps.length - 1)];

  return (
    <aside className="glass flex flex-col gap-3 rounded-xl p-4">
      <header>
        <h2 className="text-base font-semibold text-strong">{node.label}</h2>
        <p className="mt-1 text-sm text-body">{node.summary}</p>
        <code className="mt-2 block text-2xs text-muted">{node.source}</code>
      </header>

      {steps.length === 0 ? (
        <p className="rounded-md bg-raised p-3 text-sm text-muted">
          {node.optional
            ? "This stage did not run. It is optional — check the configuration it depends on."
            : "This run has not reached this stage."}
        </p>
      ) : (
        <>
          {steps.length > 1 && (
            <div className="flex flex-wrap items-center gap-1">
              <span className="mr-1 text-xs text-muted">
                ran {steps.length} times:
              </span>
              {steps.map((s, i) => (
                <button
                  key={s.id}
                  onClick={() => setIndex(i)}
                  className={clsx(
                    "rounded px-1.5 py-0.5 text-2xs font-medium transition-colors",
                    i === Math.min(index, steps.length - 1)
                      ? "bg-accent text-accent-text"
                      : "bg-raised text-muted hover:text-body",
                    s.status === "error" && i !== index && "text-danger",
                  )}
                >
                  {i + 1}
                </button>
              ))}
            </div>
          )}

          <StatusLine step={step} />
          {step.error && (
            <p className="rounded-md border border-danger/40 bg-danger/5 p-2 text-xs text-danger">
              {step.error}
            </p>
          )}

          <Payload title="Input" values={step.input} />
          <Payload title="Output" values={step.output} />
          <Payload title="Detail" values={step.meta} />
        </>
      )}
    </aside>
  );
}

function StatusLine({ step }: { step: TraceStep }) {
  const icon = {
    ok: <Check className="h-3.5 w-3.5 text-positive" />,
    error: <AlertTriangle className="h-3.5 w-3.5 text-danger" />,
    running: <Clock className="h-3.5 w-3.5 animate-pulse text-accent" />,
    skipped: <MinusCircle className="h-3.5 w-3.5 text-muted" />,
  }[step.status];

  return (
    <div className="flex items-center gap-2 border-y border-border py-2 text-xs text-muted">
      {icon}
      <span className="font-medium text-body">{step.status}</span>
      <span>·</span>
      <span>{formatMs(step.duration_ms)}</span>
      <span>·</span>
      <span>started at {formatMs(step.started_ms)}</span>
    </div>
  );
}

function Payload({ title, values }: { title: string; values: Record<string, unknown> }) {
  const entries = Object.entries(values ?? {});
  if (entries.length === 0) return null;
  return (
    <section>
      <h3 className="mb-1.5 text-xs font-semibold text-muted">
        {title}
      </h3>
      <dl className="flex flex-col gap-1.5">
        {entries.map(([key, value]) => (
          <div key={key} className="rounded-md bg-raised p-2">
            <dt className="text-2xs font-medium text-muted">{key}</dt>
            <dd className="mt-0.5">
              <Value value={value} />
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function Value({ value }: { value: unknown }) {
  if (value === null || value === undefined) {
    return <span className="text-xs text-muted">—</span>;
  }
  if (typeof value === "boolean") {
    return (
      <span className={clsx("text-xs font-medium", value ? "text-positive" : "text-muted")}>
        {String(value)}
      </span>
    );
  }
  if (typeof value === "number") {
    return <span className="font-mono text-xs text-strong">{value}</span>;
  }
  if (typeof value === "string") {
    return (
      // whitespace-pre-wrap: retrieved chunks and answers carry their own line
      // breaks, and collapsing them makes a passage unreadable.
      <p className="whitespace-pre-wrap break-words text-xs text-body">{value}</p>
    );
  }
  if (Array.isArray(value)) {
    return (
      <ol className="flex flex-col gap-1">
        {value.map((item, i) => (
          <li key={i} className="rounded bg-surface p-1.5">
            <Value value={item} />
          </li>
        ))}
      </ol>
    );
  }
  return (
    <dl className="flex flex-col gap-0.5">
      {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
        <div key={k} className="flex gap-2">
          <dt className="shrink-0 text-2xs text-muted">{k}</dt>
          <dd className="min-w-0 flex-1">
            <Value value={v} />
          </dd>
        </div>
      ))}
    </dl>
  );
}
