import { ChevronRight, FileText } from "lucide-react";
import { useState } from "react";

import type { Source } from "../../api";
import { Badge } from "../ui";

/** Citations under an answer, collapsed by default — evidence should be one
 *  click away, not competing with the answer for attention. */
export function Sources({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  if (!sources.length) return null;

  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1.5 text-xs font-medium text-muted transition-colors hover:text-body"
        aria-expanded={open}
      >
        <ChevronRight
          className={`h-3 w-3 transition-transform ${open ? "rotate-90" : ""}`}
        />
        {sources.length} source{sources.length === 1 ? "" : "s"}
      </button>

      {open && (
        <ul className="mt-2 space-y-1.5 animate-slide-up">
          {sources.map((s) => (
            <li
              key={s.chunk_id}
              className="rounded-lg border border-border bg-raised/60 p-2.5"
            >
              <div className="mb-1 flex items-center gap-2">
                <FileText className="h-3 w-3 shrink-0 text-muted" />
                <span className="truncate text-xs font-medium text-body">
                  {fileName(s.source)}
                </span>
                <Badge>{s.retriever}</Badge>
              </div>
              <p className="line-clamp-3 text-xs leading-relaxed text-muted">
                {s.snippet}
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function fileName(path: string): string {
  const base = path.split(/[\\/]/).pop() ?? path;
  // Uploads are stored as "<8-hex id>_<original name>"; the id is noise here.
  return base.replace(/^[0-9a-f]{8}_/, "");
}
