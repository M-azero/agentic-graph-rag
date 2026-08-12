import { FileText, PanelRightOpen } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import {
  ApiError,
  queryOnce,
  shelves as shelvesApi,
  threads as threadsApi,
  type LimitDetail,
  type ShelfInfo,
  type ThreadInfo,
} from "../api";
import { DocumentsPanel } from "../components/DocumentsPanel";
import { ShelfPicker } from "../components/ShelfPicker";
import { Composer } from "../components/chat/Composer";
import { Message, type Turn } from "../components/chat/Message";
import { QuotaBanner } from "../components/chat/QuotaBanner";
import { ThreadSidebar } from "../components/chat/ThreadSidebar";
import { Alert, Button, EmptyState } from "../components/ui";
import { useAuth } from "../lib/auth";

const MODEL_KEY = "graphrag_model";
const SHELF_KEY = "graphrag_shelf";

export default function Chat() {
  const { me } = useAuth();
  const navigate = useNavigate();
  const { threadId } = useParams();

  const [threads, setThreads] = useState<ThreadInfo[]>([]);
  const [loadingThreads, setLoadingThreads] = useState(true);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [quota, setQuota] = useState<LimitDetail | null>(null);
  const [showDocs, setShowDocs] = useState(false);
  const [shelves, setShelves] = useState<ShelfInfo[]>([]);
  const [maxShelves, setMaxShelves] = useState(0);
  const [shelfId, setShelfId] = useState<string | null>(
    () => localStorage.getItem(SHELF_KEY) || null,
  );
  // The job preset. Seeded from the active shelf's default, then whatever the
  // user last chose — a shelf's preset is a starting point, not a lock.
  const [preset, setPreset] = useState("general");
  const [model, setModel] = useState(
    () => localStorage.getItem(MODEL_KEY) ?? me?.default_model ?? "",
  );

  const presets = me?.presets ?? [];
  const activeShelf =
    shelves.find((s) => (s.id ?? null) === shelfId) ?? shelves[0] ?? null;

  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  // send() creates a thread inline and navigates to it, which fires the
  // transcript-load effect below for a thread whose first message isn't
  // persisted yet. Without this, that effect loads an empty transcript and
  // resets `turns` to [] mid-stream — and the next streamed token then reads
  // `.content` off an undefined turn and crashes the whole view. This holds the
  // id we're actively streaming into so the effect leaves its optimistic turns
  // alone.
  const streamingRef = useRef<string | null>(null);

  useEffect(() => localStorage.setItem(MODEL_KEY, model), [model]);
  useEffect(() => localStorage.setItem(SHELF_KEY, shelfId ?? ""), [shelfId]);

  const loadThreads = useCallback(async () => {
    try {
      setThreads((await threadsApi.list()).threads);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load conversations.");
    } finally {
      setLoadingThreads(false);
    }
  }, []);

  const loadShelves = useCallback(async () => {
    try {
      const data = await shelvesApi.list();
      setShelves(data.shelves);
      setMaxShelves(data.max_shelves);
      // A remembered shelf that no longer exists (deleted here or on another
      // device) must not leave the picker pointing at nothing — and must not be
      // sent to the server, which would 404 every question.
      setShelfId((current) =>
        data.shelves.some((s) => (s.id ?? null) === current)
          ? current
          : (data.shelves[0]?.id ?? null),
      );
    } catch {
      // Shelves need a database. Without one the server answers from the
      // default shelf anyway, so the chat stays fully usable — just without a
      // picker. Nothing to report.
    }
  }, []);

  useEffect(() => {
    void loadThreads();
    void loadShelves();
  }, [loadThreads, loadShelves]);

  // The active shelf seeds the mode. Reading the shelf rather than localStorage
  // is the point of a per-shelf default: opening the maths shelf should select
  // Study without the user re-picking it every session.
  useEffect(() => {
    if (activeShelf) setPreset(activeShelf.preset);
  }, [activeShelf?.id, activeShelf?.preset]);

  // Load a conversation's transcript when the route changes. History lives on
  // the server now, so a reload or another device shows the same thing.
  useEffect(() => {
    // Leaving a thread that's still streaming: abort it so its remaining tokens
    // don't land in whatever thread we're about to show. send()'s own teardown
    // is guarded, so this early cleanup won't fight a stream started afterward.
    if (streamingRef.current && streamingRef.current !== threadId) {
      abortRef.current?.abort();
      abortRef.current = null;
      streamingRef.current = null;
      setBusy(false);
    }

    if (!threadId) {
      setTurns([]);
      return;
    }
    // We just created this thread in send() and are streaming into it; its
    // optimistic turns are live and its first message isn't saved yet. Don't
    // reload it out from under the stream.
    if (streamingRef.current === threadId) return;
    let cancelled = false;
    (async () => {
      try {
        const data = await threadsApi.messages(threadId);
        if (cancelled) return;
        setTurns(
          data.messages.map((m) => ({
            role: m.role,
            content: m.content,
            sources: m.sources?.length ? m.sources : undefined,
          })),
        );
        // Follow the conversation to its shelf. A thread is pinned to the shelf
        // it was created on and the server answers from that one regardless, so
        // leaving the picker elsewhere would show documents this conversation
        // cannot actually reach.
        setShelfId(data.thread.shelf_id ?? null);
      } catch {
        if (!cancelled) navigate("/chat", { replace: true });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [threadId, navigate]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  async function createThread() {
    try {
      const thread = await threadsApi.create("New chat", shelfId);
      setThreads((prev) => [thread, ...prev]);
      navigate(`/chat/${thread.id}`);
    } catch (err) {
      if (err instanceof ApiError && err.limit) setQuota(err.limit);
      else setError(err instanceof Error ? err.message : "Could not start a conversation.");
    }
  }

  async function createShelf(name: string, newPreset: string) {
    const shelf = await shelvesApi.create(name, newPreset);
    await loadShelves();
    // Switch to it, and leave any open conversation behind: the new shelf has
    // no documents yet and the current thread belongs to a different one.
    setShelfId(shelf.id);
    if (threadId) navigate("/chat");
  }

  async function removeShelf(id: string) {
    try {
      await shelvesApi.remove(id);
      // Its conversations went with it, so both lists are stale.
      await Promise.all([loadShelves(), loadThreads()]);
      if (shelfId === id) setShelfId(null);
      if (threadId) navigate("/chat");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete the shelf.");
    }
  }

  function selectShelf(id: string | null) {
    setShelfId(id);
    // A conversation belongs to one shelf, so switching shelves means leaving
    // it — otherwise the picker and the thread would disagree about what the
    // next question searches, and the thread would silently win.
    if (threadId) navigate("/chat");
  }

  async function removeThread(id: string) {
    await threadsApi.remove(id);
    setThreads((prev) => prev.filter((t) => t.id !== id));
    if (id === threadId) navigate("/chat", { replace: true });
  }

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    streamingRef.current = null;
    setBusy(false);
  }

  async function send() {
    const question = input.trim();
    if (!question || busy) return;

    setError("");
    setQuota(null);
    setInput("");
    setBusy(true);

    // A conversation is created on the first message rather than up front, so
    // opening the app doesn't litter the sidebar with empty threads.
    let id = threadId;
    if (!id) {
      try {
        // Pinned to the shelf showing in the picker. This is the only moment
        // the client gets to choose; every later turn follows the thread.
        const thread = await threadsApi.create("New chat", shelfId);
        setThreads((prev) => [thread, ...prev]);
        id = thread.id;
        // Mark before navigating so the load effect this navigation triggers
        // skips the reload instead of wiping our optimistic turns.
        streamingRef.current = id;
        navigate(`/chat/${thread.id}`, { replace: true });
      } catch (err) {
        setBusy(false);
        if (err instanceof ApiError && err.limit) setQuota(err.limit);
        else setError(err instanceof Error ? err.message : "Could not start a conversation.");
        return;
      }
    }

    streamingRef.current = id;
    setTurns((prev) => [...prev, { role: "user", content: question }, { role: "assistant", content: "" }]);

    const controller = new AbortController();
    abortRef.current = controller;

    // Every stream updater guards against an empty `turns`: state resets can
    // still race the stream, and a token must never read `.content` off an
    // undefined turn.
    const patchLast = (patch: Partial<Turn>) =>
      setTurns((prev) => {
        if (prev.length === 0) return prev;
        const next = [...prev];
        next[next.length - 1] = { ...next[next.length - 1], ...patch };
        return next;
      });

    try {
      // Non-streaming: the server runs the output guard and enforces (block /
      // redact) before replying, then hands back the verdict to display. The
      // answer arrives whole rather than token-by-token — the price of letting
      // the guard hold an answer back instead of only flagging it after the fact.
      const result = await queryOnce(
        question, preset, id, model || undefined, controller.signal, shelfId,
      );
      patchLast({
        content: result.answer,
        sources: result.sources,
        safety: result.safety ?? undefined,
        activity: undefined,
      });
      // The title is set server-side from the first question; reflect it.
      void loadThreads();
    } catch (err) {
      if (controller.signal.aborted) {
        patchLast({ activity: undefined });
      } else if (err instanceof ApiError && err.limit) {
        setQuota(err.limit);
        setTurns((prev) => prev.slice(0, -1)); // drop the empty assistant turn
      } else {
        patchLast({
          error: err instanceof Error ? err.message : "Something went wrong.",
          activity: undefined,
        });
      }
    } finally {
      // Only tear down if we still own the live stream. A thread switch or the
      // stop button may have already aborted us — and by now a newer stream
      // could be running, whose refs and busy state we must not clobber.
      if (abortRef.current === controller) {
        abortRef.current = null;
        streamingRef.current = null;
        setBusy(false);
      }
    }
  }

  return (
    <div className="flex h-full">
      <ThreadSidebar
        threads={threads}
        activeId={threadId ?? null}
        loading={loadingThreads}
        // Above the conversation list, because it scopes the whole list: a
        // thread belongs to one shelf, and switching shelves leaves it.
        // Omitted entirely when there are no shelves (no database), so the
        // chat degrades to exactly what it was before.
        header={
          shelves.length > 0 ? (
            <ShelfPicker
              shelves={shelves}
              activeId={activeShelf?.id ?? null}
              presets={presets}
              maxShelves={maxShelves}
              disabled={busy}
              onSelect={selectShelf}
              onCreate={createShelf}
              onDelete={removeShelf}
            />
          ) : undefined
        }
        onSelect={(id) => navigate(`/chat/${id}`)}
        onCreate={createThread}
        onDelete={removeThread}
      />

      <section className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-11 shrink-0 items-center justify-end px-4">
          {!showDocs && (
            <Button size="sm" variant="ghost" onClick={() => setShowDocs(true)}>
              <PanelRightOpen className="h-3.5 w-3.5" />
              Documents
            </Button>
          )}
        </div>

        {/* One column, and the composer shares its width. A full-bleed input
            under a narrow transcript makes the page feel like two designs. */}
        <div className="min-h-0 flex-1 overflow-y-auto px-6">
          <div className="mx-auto max-w-[46rem] space-y-8 py-8">
            {quota && <QuotaBanner detail={quota} />}
            {error && <Alert>{error}</Alert>}

            {turns.length === 0 && !quota && (
              <EmptyState
                icon={<FileText className="h-6 w-6" />}
                title={
                  activeShelf ? `Ask “${activeShelf.name}” anything` : "Ask your documents anything"
                }
                description="Upload a file, then ask a question. Answers cite the passages they came from."
                action={
                  <Button variant="secondary" onClick={() => setShowDocs(true)}>
                    Add documents
                  </Button>
                }
              />
            )}

            {turns.map((turn, i) => (
              <Message key={i} turn={turn} />
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="mx-auto w-full max-w-[46rem] px-6 pb-5">
          <Composer
            value={input}
            onChange={setInput}
            onSend={send}
            onStop={stop}
            busy={busy}
            preset={preset}
            onPresetChange={setPreset}
            presets={presets}
            model={model}
            onModelChange={setModel}
            models={me?.models ?? []}
          />
        </div>
      </section>

      {showDocs && (
        <DocumentsPanel
          onClose={() => setShowDocs(false)}
          shelfId={activeShelf?.id ?? null}
          shelfName={activeShelf?.name ?? "Documents"}
          onChanged={loadShelves}
        />
      )}
    </div>
  );
}
