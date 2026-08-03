import { useState } from "react";

import type { AuditEntry } from "../api";
import { Badge, EmptyState, relativeTime } from "./ui";

// Actions are coloured by how much they take away, not alphabetically — the
// eye should find a purge in a page of role changes without reading.
function toneFor(action: string): "danger" | "caution" | "accent" | "neutral" {
  if (action.includes("purge") || action.includes("delete")) return "danger";
  if (
    action.includes("suspend") ||
    action.includes("revoke") ||
    action.includes("force_reset")
  ) {
    return "caution";
  }
  if (action.includes("create") || action.includes("unlock")) return "accent";
  return "neutral";
}

function Entry({ entry }: { entry: AuditEntry }) {
  const [open, setOpen] = useState(false);
  const hasDetail = entry.detail && Object.keys(entry.detail).length > 0;

  return (
    <li className="relative pl-6">
      <span
        aria-hidden
        className="absolute left-[5px] top-2 h-1.5 w-1.5 rounded-full bg-border ring-4 ring-canvas"
      />
      <div className="flex flex-wrap items-center gap-2 py-1.5">
        <Badge tone={toneFor(entry.action)}>{entry.action}</Badge>
        <span className="text-sm text-body">
          {entry.actor ? (
            <span title={entry.actor}>by {entry.actor.slice(0, 8)}</span>
          ) : (
            // `require_admin_user` returns None for the X-Admin-Key path, so
            // there is genuinely no account behind these rows.
            <span className="text-muted">via admin key</span>
          )}
          {entry.target && (
            <span className="text-muted"> → {entry.target.slice(0, 8)}</span>
          )}
        </span>
        <span
          className="ml-auto text-2xs text-muted"
          title={new Date(entry.created_at).toLocaleString()}
        >
          {relativeTime(entry.created_at)}
        </span>
        {hasDetail && (
          <button
            onClick={() => setOpen((v) => !v)}
            className="text-2xs text-accent hover:underline"
          >
            {open ? "hide" : "detail"}
          </button>
        )}
      </div>
      {open && hasDetail && (
        <pre className="mb-2 overflow-x-auto rounded-md border border-border bg-raised p-2 text-2xs text-body">
          {JSON.stringify(entry.detail, null, 2)}
        </pre>
      )}
    </li>
  );
}

export function Timeline({ entries }: { entries: AuditEntry[] }) {
  if (!entries.length) {
    return (
      <EmptyState
        title="No admin actions recorded"
        detail="Reads are not logged — only changes."
      />
    );
  }
  return (
    <ol className="relative space-y-0 before:absolute before:bottom-2 before:left-[9px] before:top-2 before:w-px before:bg-border">
      {entries.map((e) => (
        <Entry key={e.id} entry={e} />
      ))}
    </ol>
  );
}
