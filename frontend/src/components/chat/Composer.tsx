import { ArrowUp, Square } from "lucide-react";
import { useEffect, useRef, type KeyboardEvent } from "react";

import type { ModelOption, PresetOption } from "../../api";
import { Button, Select } from "../ui";

const MAX_HEIGHT = 200;

export function Composer({
  value,
  onChange,
  onSend,
  onStop,
  busy,
  preset,
  onPresetChange,
  presets,
  model,
  onModelChange,
  models,
  disabled,
}: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
  busy: boolean;
  // The job the assistant is doing — study, finance, legal… The list comes
  // from the server (`/auth/me`), so this component never hardcodes it.
  preset: string;
  onPresetChange: (v: string) => void;
  presets: PresetOption[];
  model: string;
  onModelChange: (v: string) => void;
  models: ModelOption[];
  disabled?: boolean;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);

  // Grow with the text, up to a point, then scroll — a fixed single line hides
  // what you typed, and unbounded growth eats the transcript.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
  }, [value]);

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends, Shift+Enter breaks the line.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!busy && value.trim()) onSend();
    }
  }

  return (
    <div>
      {/* A border rather than a ring, and it thickens to the accent on focus.
          Sitting on the canvas with no bar behind it, the input reads as part
          of the conversation rather than a toolbar bolted to the bottom. */}
      <div className="rounded-xl border border-border bg-surface shadow-card transition-colors focus-within:border-accent">
        <textarea
          ref={ref}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          rows={1}
          disabled={disabled}
          placeholder="Ask about your documents…"
          className="w-full resize-none bg-transparent px-4 py-3.5 text-base text-strong placeholder:text-muted focus:outline-none disabled:opacity-60"
        />

        <div className="flex items-center gap-2 px-2.5 pb-2.5">
          <Select
            value={preset}
            onChange={(e) => onPresetChange(e.target.value)}
            aria-label="Mode"
            title={presets.find((p) => p.id === preset)?.description}
            className="h-7 !w-auto py-0 text-xs"
          >
            {presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.emoji} {p.label}
              </option>
            ))}
          </Select>

          {models.length > 1 && (
            <Select
              value={model}
              onChange={(e) => onModelChange(e.target.value)}
              aria-label="Model"
              className="h-7 !w-auto py-0 text-xs"
            >
              {models.map((m) => (
                <option key={m.model} value={m.model}>
                  {m.label}
                </option>
              ))}
            </Select>
          )}

          <div className="ml-auto">
            {busy ? (
              <Button size="sm" onClick={onStop} aria-label="Stop generating">
                <Square className="h-3 w-3 fill-current" />
                Stop
              </Button>
            ) : (
              <Button
                size="sm"
                variant="primary"
                onClick={onSend}
                disabled={!value.trim() || disabled}
                aria-label="Send"
              >
                <ArrowUp className="h-3.5 w-3.5" />
              </Button>
            )}
          </div>
        </div>
      </div>

      <p className="mt-2 text-center text-2xs text-muted">
        Answers are grounded in your uploaded documents.
      </p>
    </div>
  );
}
