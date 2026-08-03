import { useState } from "react";

import { Button, Input } from "./ui";

export const LIMIT_FIELDS = [
  { key: "messages_per_minute", label: "Messages / minute" },
  { key: "messages_per_day", label: "Messages / day" },
  { key: "tokens_per_day", label: "Tokens / day" },
  { key: "tokens_per_month", label: "Tokens / month" },
  { key: "max_files", label: "Documents" },
  { key: "max_file_mb", label: "Max file size (MB)" },
  { key: "max_storage_mb", label: "Total storage (MB)" },
  { key: "max_chunks", label: "Indexed chunks" },
  { key: "max_threads", label: "Conversations" },
] as const;

/** Editor for a set of limits.
 *
 *  In `override` mode an empty field means "inherit the global default" rather
 *  than zero, so the placeholder shows what would be inherited — otherwise a
 *  blank box looks like a limit of nothing. */
export function LimitsForm({
  values,
  inherited,
  mode,
  onSave,
  onClear,
  saving,
}: {
  values: Record<string, number | null>;
  inherited?: Record<string, number>;
  mode: "global" | "override";
  onSave: (values: Record<string, number | null>) => void;
  onClear?: () => void;
  saving?: boolean;
}) {
  const [draft, setDraft] = useState<Record<string, string>>(() =>
    Object.fromEntries(
      LIMIT_FIELDS.map(({ key }) => [key, values[key] == null ? "" : String(values[key])]),
    ),
  );

  function submit() {
    const out: Record<string, number | null> = {};
    for (const { key } of LIMIT_FIELDS) {
      const raw = draft[key]?.trim();
      if (!raw) {
        // Global limits have no "unset": leave the current value alone.
        if (mode === "override") out[key] = null;
        continue;
      }
      const parsed = Number(raw);
      if (Number.isFinite(parsed) && parsed >= 0) out[key] = Math.floor(parsed);
    }
    onSave(out);
  }

  return (
    <div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {LIMIT_FIELDS.map(({ key, label }) => (
          <div key={key}>
            <label className="mb-1 block text-xs text-muted" htmlFor={key}>
              {label}
            </label>
            <Input
              id={key}
              type="number"
              min={0}
              value={draft[key] ?? ""}
              onChange={(e) => setDraft((d) => ({ ...d, [key]: e.target.value }))}
              placeholder={
                mode === "override" && inherited
                  ? `${inherited[key] ?? 0} (inherited)`
                  : undefined
              }
              className="font-mono text-sm"
            />
          </div>
        ))}
      </div>

      <div className="mt-4 flex items-center gap-2">
        <Button variant="primary" onClick={submit} loading={saving}>
          Save
        </Button>
        {onClear && (
          <Button variant="ghost" onClick={onClear}>
            Reset to defaults
          </Button>
        )}
        {mode === "override" && (
          <p className="text-xs text-muted">Empty means inherit the default.</p>
        )}
      </div>
    </div>
  );
}
