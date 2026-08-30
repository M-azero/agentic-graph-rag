// The pipeline inspector: ask a question, watch the blocks light up, click one
// to see what it actually received and returned.
//
// Three pipelines share this page. Query and review are driven live from
// `/trace/query`; ingestion is read back by job id, because an ingest runs in a
// background task — or another process entirely — and cannot be streamed from.

import { Play, RotateCcw, Square } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import {
  ApiError,
  trace as traceApi,
  traceQuery,
  type PipelineGraph,
  type TraceAnswer,
  type TraceRunSummary,
  type TraceStep,
} from "../api";
import { Diagram } from "../components/pipeline/Diagram";
import { StepInspector } from "../components/pipeline/StepInspector";
import { Timeline } from "../components/pipeline/Timeline";
import { Alert, Button, EmptyState, Input } from "../components/ui";
import { useAuth } from "../lib/auth";
import { formatMs, runStateByNode } from "../lib/pipeline";

const TITLES: Record<string, string> = {
  query: "Answering a question",
  ingest: "Ingesting a document",
  review: "The review loop",
};

const BLURBS: Record<string, string> = {
  query:
    "One question, from the guard that screens it to the tokens it costs. Blocks light up as the run reaches them.",
  ingest:
    "A document becoming searchable text and a knowledge graph. Upload a file in Chat, then open its run here.",
  review:
    "Research, check, and repair. Only runs when agent.review.enabled is on; otherwise the query pipeline answers directly.",
};

export default function Pipeline() {
  const { me } = useAuth();
  const [graphs, setGraphs] = useState<PipelineGraph[]>([]);
  const [pipeline, setPipeline] = useState("query");
  const [question, setQuestion] = useState("");
  const [preset, setPreset] = useState("general");
  const [steps, setSteps] = useState<TraceStep[]>([]);
  const [answer, setAnswer] = useState<TraceAnswer | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [runs, setRuns] = useState<TraceRunSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const graph = graphs.find((g) => g.pipeline === pipeline) ?? null;
  const presets = me?.presets ?? [];

  // A step is re-sent while it runs, so the latest copy of each id wins. Keyed
  // by id rather than appended, or a block would show its own running state
  // alongside its finished one.
  const upsert = useCallback((step: TraceStep) => {
    setSteps((current) => {
      const at = current.findIndex((s) => s.id === step.id);
      if (at === -1) return [...current, step];
      const next = current.slice();
      next[at] = step;
      return next;
    });
  }, []);

  useEffect(() => {
    traceApi
      .pipelines()
      .then((d) => setGraphs(d.pipelines))
      .catch((e) =>
        setError(
          e instanceof ApiError && e.status === 404
            ? "Tracing is turned off on this server (trace.enabled)."
            : "Could not load the pipeline diagram.",
        ),
      );
  }, []);

  const loadRuns = useCallback(() => {
    traceApi
      .runs()
      .then((d) => setRuns(d.runs))
      .catch(() => undefined); // the list is a convenience, not the page
  }, []);

  useEffect(loadRuns, [loadRuns]);

  const nodeRuns = useMemo(() => runStateByNode(steps), [steps]);
  const selectedNode = graph?.nodes.find((n) => n.name === selected) ?? null;

  async function run() {
    if (!question.trim() || busy) return;
    setBusy(true);
    setError("");
    setSteps([]);
    setAnswer(null);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await traceQuery(
        { question: question.trim(), preset },
        { onStep: upsert, onAnswer: setAnswer },
        controller.signal,
      );
      loadRuns();
    } catch (e) {
      if (!controller.signal.aborted) {
        setError(e instanceof Error ? e.message : "The run failed.");
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  const open = useCallback(async (id: string) => {
    setError("");
    setSteps([]);
    setAnswer(null);
    try {
      const record = await traceApi.read(id);
      setPipeline(record.pipeline);
      setSteps(record.steps);
    } catch {
      setError("That run has expired or is no longer available.");
    }
  }, []);

  // Deep link from a document in Chat: `/pipeline?run=<job id>`. An ingest is
  // traced under its job id and runs in a background task — it can never be
  // streamed here, so opening it by id is the only way in.
  const [params] = useSearchParams();
  const runParam = params.get("run");
  useEffect(() => {
    if (runParam) void open(runParam);
  }, [runParam, open]);

  return (
    <div className="mx-auto flex h-full max-w-[1400px] flex-col gap-4 overflow-y-auto p-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-strong">Pipeline inspector</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted">{BLURBS[pipeline]}</p>
        </div>
        <nav className="flex rounded-md bg-raised p-0.5" role="tablist">
          {graphs.map((g) => (
            <button
              key={g.pipeline}
              role="tab"
              aria-selected={pipeline === g.pipeline}
              onClick={() => {
                setPipeline(g.pipeline);
                setSelected(null);
              }}
              className={
                pipeline === g.pipeline
                  ? "rounded px-3 py-1.5 text-sm font-medium bg-surface text-strong shadow-card"
                  : "rounded px-3 py-1.5 text-sm font-medium text-muted hover:text-body"
              }
            >
              {TITLES[g.pipeline] ?? g.pipeline}
            </button>
          ))}
        </nav>
      </header>

      {error && <Alert tone="danger">{error}</Alert>}

      {pipeline === "ingest" ? (
        <p className="rounded-lg border border-border bg-surface p-3 text-sm text-muted">
          Ingest runs happen in the background, so they are read back rather than
          streamed. Upload a document in Chat, then pick its run below — the job
          id is the run id.
        </p>
      ) : (
        <form
          className="flex flex-wrap items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void run();
          }}
        >
          <Input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask something your documents cover…"
            className="min-w-0 flex-1"
            disabled={busy}
          />
          <select
            value={preset}
            onChange={(e) => setPreset(e.target.value)}
            className="h-9 rounded-md border border-border bg-surface px-2 text-sm text-body"
            disabled={busy}
          >
            {presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.emoji} {p.label}
              </option>
            ))}
          </select>
          {busy ? (
            <Button type="button" variant="secondary" onClick={stop}>
              <Square className="h-3.5 w-3.5" />
              Stop
            </Button>
          ) : (
            <Button type="submit" variant="primary" disabled={!question.trim()}>
              <Play className="h-3.5 w-3.5" />
              Run
            </Button>
          )}
        </form>
      )}

      {/* This run costs tokens like any other question — say so rather than
          letting the page read as a free simulator. */}
      {pipeline !== "ingest" && (
        <p className="-mt-2 text-2xs text-muted">
          A traced run is a real run: it calls the same models and counts against
          the same quota as asking in Chat.
        </p>
      )}

      {graph ? (
        <>
          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-strong">Diagram</h2>
            <Diagram
              graph={graph}
              runs={nodeRuns}
              selected={selected}
              onSelect={setSelected}
            />
            <p className="text-2xs text-muted">
              Click a block for its input and output. Blocks that ran are solid;
              a stacked block ran more than once.
            </p>
          </section>

          {/* items-start so the two columns size to their own content — without
              it a grid stretches both to the taller one, and the inspector
              (which can hold a long chunk) would drag the timeline's height
              with it. */}
          <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
            <div className="flex min-w-0 flex-col gap-4">
              <Timeline steps={steps} selected={selected} onSelect={setSelected} />
              {answer && <AnswerCard answer={answer} />}
              <RunList runs={runs} onOpen={open} onRefresh={loadRuns} />
            </div>
            <div className="lg:sticky lg:top-4 lg:max-h-[calc(100vh-2rem)] lg:overflow-y-auto">
              <StepInspector
                key={selected ?? "none"}
                node={selectedNode}
                run={selected ? nodeRuns[selected] : undefined}
              />
            </div>
          </div>
        </>
      ) : (
        !error && <EmptyState title="Loading the pipeline…" />
      )}
    </div>
  );
}

function AnswerCard({ answer }: { answer: TraceAnswer }) {
  return (
    <section className="rounded-lg border border-border bg-surface p-4">
      <h2 className="mb-2 text-sm font-semibold text-strong">
        Answer
        <span className="ml-2 font-normal text-muted">{answer.outcome}</span>
      </h2>
      <p className="whitespace-pre-wrap text-sm text-body">{answer.answer}</p>
      {answer.safety && (
        <p className="mt-2 rounded-md bg-raised p-2 text-xs text-muted">
          Safety: {answer.safety.action} at {answer.safety.stage}
          {answer.safety.reasons.length > 0 && ` — ${answer.safety.reasons.join(", ")}`}
        </p>
      )}
      {answer.sources.length > 0 && (
        <ul className="mt-2 flex flex-wrap gap-1.5">
          {answer.sources.map((s) => (
            <li
              key={s.chunk_id}
              className="rounded bg-raised px-1.5 py-0.5 text-2xs text-muted"
            >
              {s.source}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function RunList({
  runs,
  onOpen,
  onRefresh,
}: {
  runs: TraceRunSummary[];
  onOpen: (id: string) => void;
  onRefresh: () => void;
}) {
  return (
    <section className="rounded-lg border border-border bg-surface p-4">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-strong">Recent runs</h2>
        <Button size="sm" variant="ghost" onClick={onRefresh}>
          <RotateCcw className="h-3.5 w-3.5" />
          Refresh
        </Button>
      </div>
      {runs.length === 0 ? (
        <p className="text-sm text-muted">
          No runs yet. Ask a question above, or upload a document in Chat.
        </p>
      ) : (
        <ul className="flex flex-col gap-1">
          {runs.map((r) => (
            <li key={r.trace_id}>
              <button
                onClick={() => onOpen(r.trace_id)}
                className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left hover:bg-raised"
              >
                <span className="min-w-0 flex-1 truncate text-sm text-body">
                  {r.title || r.trace_id}
                </span>
                <span className="shrink-0 text-2xs text-muted">{r.pipeline}</span>
                <span className="shrink-0 text-2xs text-muted">
                  {r.steps} steps · {formatMs(r.duration_ms)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
