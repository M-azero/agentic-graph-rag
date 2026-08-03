import clsx from "clsx";
import { FileText, Trash2, Upload as UploadIcon, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, deleteFile, ingestStatus, listFiles, uploadFile, type StoredFile } from "../api";
import { Alert, Badge, Button, EmptyState, Spinner } from "./ui";
import { QuotaBanner } from "./chat/QuotaBanner";
import type { LimitDetail } from "../api";

const POLL_MS = 1500;
const ACCEPT = ".pdf,.docx,.txt,.md,.html,.csv,.png,.jpg,.jpeg";

interface Job {
  id: string;
  name: string;
  // "partial": stored and searchable, but entity extraction failed for some
  // chunks so the knowledge graph is incomplete. A degraded success — the
  // document works, it just answers less well from the graph.
  status: "uploading" | "queued" | "running" | "done" | "partial" | "error";
  detail?: string;
  chunks?: number;
}

export function DocumentsPanel({ onClose }: { onClose: () => void }) {
  const [files, setFiles] = useState<StoredFile[]>([]);
  const [used, setUsed] = useState(0);
  const [limit, setLimit] = useState(0);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [quota, setQuota] = useState<LimitDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const data = await listFiles();
      setFiles(data.files);
      setUsed(data.used);
      setLimit(data.limit);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load documents.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Poll only while something is actually in flight, and stop as soon as
  // nothing is — a permanent timer is a needless request every 1.5s forever.
  useEffect(() => {
    const pending = jobs.filter((j) => j.status === "queued" || j.status === "running");
    if (!pending.length) return;

    const timer = setInterval(async () => {
      for (const job of pending) {
        try {
          const status = await ingestStatus(job.id);
          setJobs((prev) =>
            prev.map((j) =>
              j.id === job.id
                ? {
                    ...j,
                    status: status.status as Job["status"],
                    detail: status.detail,
                    chunks: status.chunks,
                  }
                : j,
            ),
          );
          // A partial ingest still stored its chunks, so the file list needs
          // refreshing exactly as it does for a clean one.
          if (status.status === "done" || status.status === "partial") void refresh();
        } catch {
          /* a transient poll failure resolves on the next tick */
        }
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [jobs, refresh]);

  async function upload(fileList: FileList | null) {
    if (!fileList?.length) return;
    setError("");
    setQuota(null);

    for (const file of Array.from(fileList)) {
      const placeholder: Job = { id: `pending-${file.name}`, name: file.name, status: "uploading" };
      setJobs((prev) => [...prev, placeholder]);
      try {
        const { job_id } = await uploadFile(file);
        setJobs((prev) =>
          prev.map((j) =>
            j.id === placeholder.id ? { ...j, id: job_id, status: "queued" } : j,
          ),
        );
      } catch (err) {
        setJobs((prev) => prev.filter((j) => j.id !== placeholder.id));
        if (err instanceof ApiError && err.limit) setQuota(err.limit);
        else setError(err instanceof Error ? err.message : `Could not upload ${file.name}.`);
      }
    }
  }

  async function remove(fileId: string) {
    try {
      await deleteFile(fileId);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete the document.");
    }
  }

  const atLimit = limit > 0 && used >= limit;

  return (
    <aside className="flex w-80 shrink-0 flex-col border-l border-border bg-surface">
      <header className="flex h-12 items-center justify-between border-b border-border px-4">
        <h2 className="text-sm font-semibold text-strong">Documents</h2>
        <div className="flex items-center gap-2">
          {limit > 0 && (
            <span className="font-mono text-2xs text-muted">
              {used}/{limit}
            </span>
          )}
          <button
            onClick={onClose}
            aria-label="Close documents"
            className="rounded p-1 text-muted hover:bg-raised hover:text-body"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </header>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        {quota && <QuotaBanner detail={quota} />}
        {error && <Alert>{error}</Alert>}

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            void upload(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
          className={clsx(
            "cursor-pointer rounded-lg border border-dashed px-3 py-6 text-center transition-colors",
            atLimit && "pointer-events-none opacity-50",
            dragging
              ? "border-accent bg-accent/5"
              : "border-border hover:border-accent/50 hover:bg-raised/50",
          )}
        >
          <UploadIcon className="mx-auto mb-2 h-4 w-4 text-muted" />
          <p className="text-sm font-medium text-body">
            {atLimit ? "Document limit reached" : "Drop files or click to upload"}
          </p>
          <p className="mt-0.5 text-2xs text-muted">PDF, Word, text, CSV, images</p>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPT}
            className="hidden"
            onChange={(e) => {
              void upload(e.target.files);
              e.target.value = "";
            }}
          />
        </div>

        {jobs.length > 0 && (
          <ul className="space-y-1.5">
            {jobs.map((job) => (
              <li
                key={job.id}
                className="flex items-center gap-2 rounded-md bg-raised/60 px-2.5 py-2"
              >
                {job.status === "done" ? (
                  <Badge tone="positive">done</Badge>
                ) : job.status === "partial" ? (
                  <Badge tone="caution">partial</Badge>
                ) : job.status === "error" ? (
                  <Badge tone="danger">failed</Badge>
                ) : (
                  <Spinner className="h-3 w-3 text-muted" />
                )}
                <span className="min-w-0 flex-1 truncate text-xs text-body">
                  {job.name}
                </span>
                {(job.status === "done" || job.status === "partial") && job.chunks ? (
                  <span className="text-2xs text-muted">{job.chunks} chunks</span>
                ) : null}
                {(job.status === "error" || job.status === "partial") && (
                  <button
                    onClick={() => setJobs((prev) => prev.filter((j) => j.id !== job.id))}
                    className="text-muted hover:text-body"
                    aria-label="Dismiss"
                  >
                    <X className="h-3 w-3" />
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}

        {jobs.some((j) => j.status === "error") && (
          <Alert>
            {jobs.find((j) => j.status === "error")?.detail ?? "Ingestion failed."}
          </Alert>
        )}

        {/* Separate from the error alert: the document did import and is
            searchable, so this explains a degraded result rather than a
            failure the user has to act on. */}
        {jobs.some((j) => j.status === "partial") && (
          <Alert tone="caution">
            {jobs.find((j) => j.status === "partial")?.detail ??
              "Imported, but the knowledge graph is incomplete."}
          </Alert>
        )}

        {loading ? null : files.length === 0 ? (
          <EmptyState
            icon={<FileText className="h-5 w-5" />}
            title="No documents yet"
            description="Upload something to give the assistant material to work from."
          />
        ) : (
          <ul className="space-y-0.5">
            {files.map((file) => (
              <li key={file.file_id} className="group flex items-center gap-2 rounded-md px-1 py-1.5">
                <FileText className="h-3.5 w-3.5 shrink-0 text-muted" />
                <span className="min-w-0 flex-1 truncate text-xs text-body" title={file.name}>
                  {file.name}
                </span>
                <button
                  onClick={() => remove(file.file_id)}
                  aria-label={`Delete ${file.name}`}
                  className="rounded p-1 text-muted opacity-0 transition-opacity hover:text-danger focus-visible:opacity-100 group-hover:opacity-100"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}

export { Button };
