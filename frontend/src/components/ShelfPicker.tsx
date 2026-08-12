import clsx from "clsx";
import { Check, Library, Plus, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import type { PresetOption, ShelfInfo } from "../api";
import { Alert, Button, Input, Modal, Select } from "./ui";

/** The active shelf, and a menu to switch, add or remove one.
 *
 *  A shelf is a subject — a maths textbook and a programming manual are two
 *  knowledge bases, not one — so this control decides which documents a
 *  question can reach. It sits at the top of the conversation list rather than
 *  in the composer because it scopes the whole conversation, not one message:
 *  a thread is pinned to its shelf when it is created.
 */
export function ShelfPicker({
  shelves,
  activeId,
  presets,
  maxShelves,
  disabled,
  onSelect,
  onCreate,
  onDelete,
}: {
  shelves: ShelfInfo[];
  activeId: string | null;
  presets: PresetOption[];
  maxShelves: number;
  disabled?: boolean;
  onSelect: (id: string | null) => void;
  onCreate: (name: string, preset: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState("");
  const [preset, setPreset] = useState("general");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const active = shelves.find((s) => (s.id ?? null) === activeId) ?? shelves[0];

  // Close on an outside click or Escape. Without both, the menu stays open
  // behind whatever the user clicked next.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  async function submit() {
    const trimmed = name.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setError("");
    try {
      await onCreate(trimmed, preset);
      setAdding(false);
      setName("");
      setPreset("general");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create the shelf.");
    } finally {
      setBusy(false);
    }
  }

  async function remove(shelf: ShelfInfo) {
    if (!shelf.id) return;
    // A native confirm rather than a second modal: this destroys a knowledge
    // graph that took an ingest to build, and the count makes the cost concrete.
    const ok = window.confirm(
      `Delete "${shelf.name}" and its ${shelf.files} document${
        shelf.files === 1 ? "" : "s"
      }? This can't be undone.`,
    );
    if (!ok) return;
    await onDelete(shelf.id);
  }

  const presetOf = (id: string) => presets.find((p) => p.id === id);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        className={clsx(
          "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left transition-colors",
          "hover:bg-raised disabled:pointer-events-none disabled:opacity-50",
        )}
      >
        <Library className="h-3.5 w-3.5 shrink-0 text-muted" />
        <span className="min-w-0 flex-1 truncate text-sm font-medium text-strong">
          {active?.name ?? "My documents"}
        </span>
        <span className="shrink-0 font-mono text-2xs text-muted">
          {active?.files ?? 0}
        </span>
      </button>

      {open && (
        <div
          role="listbox"
          className="absolute left-0 right-0 top-full z-30 mt-1 overflow-hidden rounded-lg border border-border bg-surface shadow-pop"
        >
          <ul className="max-h-64 overflow-y-auto py-1">
            {shelves.map((shelf) => {
              const id = shelf.id ?? null;
              const selected = id === (active?.id ?? null);
              const job = presetOf(shelf.preset);
              return (
                <li key={shelf.id ?? "default"} className="group flex items-center">
                  <button
                    role="option"
                    aria-selected={selected}
                    onClick={() => {
                      onSelect(id);
                      setOpen(false);
                    }}
                    className="flex min-w-0 flex-1 items-center gap-2 px-2.5 py-1.5 text-left hover:bg-raised"
                  >
                    <span className="w-3.5 shrink-0">
                      {selected && <Check className="h-3.5 w-3.5 text-accent" />}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-sm text-body">
                      {shelf.name}
                    </span>
                    {job && job.id !== "general" && (
                      <span className="shrink-0 text-2xs" title={job.label}>
                        {job.emoji}
                      </span>
                    )}
                    <span className="shrink-0 font-mono text-2xs text-muted">
                      {shelf.files}
                    </span>
                  </button>
                  {/* The default shelf has no delete: its corpus is where
                      everything uploaded before shelves existed lives. */}
                  {!shelf.is_default && (
                    <button
                      onClick={() => void remove(shelf)}
                      aria-label={`Delete ${shelf.name}`}
                      className="mr-1 rounded p-1 text-muted opacity-0 transition-opacity hover:text-danger focus-visible:opacity-100 group-hover:opacity-100"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  )}
                </li>
              );
            })}
          </ul>

          <div className="border-t border-border p-1">
            <button
              onClick={() => {
                setOpen(false);
                setAdding(true);
              }}
              disabled={shelves.length >= maxShelves}
              className="flex w-full items-center gap-2 rounded px-2.5 py-1.5 text-left text-sm text-muted hover:bg-raised hover:text-body disabled:pointer-events-none disabled:opacity-50"
            >
              <Plus className="h-3.5 w-3.5" />
              {shelves.length >= maxShelves ? "Shelf limit reached" : "New shelf"}
            </button>
          </div>
        </div>
      )}

      <Modal open={adding} title="New shelf" onClose={() => setAdding(false)}>
        <div className="space-y-4">
          <div>
            <label className="mb-1.5 block text-sm font-medium text-body">Name</label>
            <Input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && void submit()}
              placeholder="Maths textbook"
              maxLength={80}
            />
            <p className="mt-1.5 text-sm text-muted">
              Documents on a shelf are searched together and kept apart from
              every other shelf.
            </p>
          </div>

          <div>
            <label className="mb-1.5 block text-sm font-medium text-body">
              Default mode
            </label>
            <Select value={preset} onChange={(e) => setPreset(e.target.value)}>
              {presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.emoji} {p.label}
                </option>
              ))}
            </Select>
            <p className="mt-1.5 text-sm text-muted">
              {presetOf(preset)?.description ?? ""}
            </p>
          </div>

          {error && <Alert>{error}</Alert>}

          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setAdding(false)}>
              Cancel
            </Button>
            <Button variant="primary" loading={busy} onClick={() => void submit()}>
              Create shelf
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
