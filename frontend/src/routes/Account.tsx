import { Copy, KeyRound, Monitor, Plus, Trash2 } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";

import { auth, type ApiKeyInfo, type LimitsInfo, type SessionInfo } from "../api";
import {
  Alert,
  Badge,
  Button,
  Card,
  CardTitle,
  EmptyState,
  Field,
  Input,
  Meter,
  Modal,
  Skeleton,
} from "../components/ui";
import { useAuth } from "../lib/auth";

export default function Account() {
  const { me } = useAuth();
  const [info, setInfo] = useState<LimitsInfo | null>(null);
  const [keys, setKeys] = useState<ApiKeyInfo[]>([]);
  const [sessions, setSessions] = useState<SessionInfo[] | null>(null);
  const [newKey, setNewKey] = useState("");
  const [label, setLabel] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");

  // allSettled, not all: these three cards are independent, and with Promise.all
  // one failing endpoint blanks the other two. That is not hypothetical — during
  // a rollout the served bundle can be newer than the API it talks to, and a
  // 404 from one new route would take the whole page down with it.
  async function load() {
    const [limits, keyList, sessionList] = await Promise.allSettled([
      auth.limits(),
      auth.listKeys(),
      auth.sessions(),
    ]);
    if (limits.status === "fulfilled") setInfo(limits.value);
    if (keyList.status === "fulfilled") setKeys(keyList.value.keys);
    // Settled either way, so a failure shows the empty state rather than a
    // skeleton that never resolves.
    setSessions(sessionList.status === "fulfilled" ? sessionList.value.sessions : []);

    const failed = [limits, keyList, sessionList].find((r) => r.status === "rejected");
    setError(
      failed && failed.status === "rejected"
        ? failed.reason instanceof Error
          ? failed.reason.message
          : "Some of your account details could not be loaded."
        : "",
    );
  }

  useEffect(() => {
    void load();
  }, []);

  async function createKey() {
    setCreating(true);
    try {
      const created = await auth.createKey(label);
      setNewKey(created.api_key);
      setLabel("");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create the key.");
    } finally {
      setCreating(false);
    }
  }

  async function revoke(id: number) {
    await auth.revokeKey(id);
    await load();
  }

  const limits = info?.limits ?? {};
  const usage = info?.usage ?? {};

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl space-y-4 px-4 py-8">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-strong">Account</h1>
          <p className="mt-1 text-sm text-muted">{me?.email}</p>
        </div>

        {error && <Alert>{error}</Alert>}

        <Card>
          <CardTitle
            action={
              me?.role === "admin" ? <Badge tone="accent">admin</Badge> : undefined
            }
          >
            Usage
          </CardTitle>
          {!info ? (
            <div className="space-y-4">
              {[0, 1, 2].map((i) => (
                <Skeleton key={i} className="h-8 w-full" />
              ))}
            </div>
          ) : (
            <div className="grid gap-5 sm:grid-cols-2">
              <Meter
                label="Messages today"
                used={usage.messages_today ?? 0}
                max={limits.messages_per_day ?? 0}
              />
              <Meter
                label="Tokens today"
                used={usage.tokens_today ?? 0}
                max={limits.tokens_per_day ?? 0}
              />
              <Meter
                label="Documents"
                used={info.files_used}
                max={limits.max_files ?? 0}
              />
              <Meter
                label="Storage"
                used={Math.round(info.storage_used_mb)}
                max={limits.max_storage_mb ?? 0}
                unit=" MB"
              />
              <Meter
                label="Conversations"
                used={info.threads_used}
                max={limits.max_threads ?? 0}
              />
              <Meter
                label="Tokens this month"
                used={usage.tokens_this_month ?? 0}
                max={limits.tokens_per_month ?? 0}
              />
            </div>
          )}
        </Card>

        <PasswordCard onChanged={load} />

        <SessionsCard
          sessions={sessions}
          onChanged={load}
          onError={(message) => setError(message)}
        />

        <Card>
          <CardTitle>API keys</CardTitle>
          <p className="mb-4 text-sm text-muted">
            For scripts and integrations. A key carries your identity and your
            limits — treat it like a password.
          </p>

          <div className="mb-4 flex gap-2">
            <Input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="What is this key for?"
              maxLength={64}
            />
            <Button variant="secondary" onClick={createKey} loading={creating}>
              <Plus className="h-3.5 w-3.5" />
              Create
            </Button>
          </div>

          {keys.length === 0 ? (
            <EmptyState
              icon={<KeyRound className="h-5 w-5" />}
              title="No API keys"
              description="Create one to use the API outside this app."
            />
          ) : (
            <ul className="divide-y divide-border">
              {keys.map((key) => (
                <li key={key.id} className="flex items-center gap-3 py-2.5">
                  <KeyRound className="h-3.5 w-3.5 shrink-0 text-muted" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm text-body">
                      {key.label || "Untitled key"}
                    </p>
                    <p className="text-2xs text-muted">
                      Created {formatDate(key.created_at)}
                      {key.last_used_at
                        ? ` · last used ${formatDate(key.last_used_at)}`
                        : " · never used"}
                    </p>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => revoke(key.id)}
                    aria-label="Revoke key"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <Modal open={Boolean(newKey)} title="Copy your API key" onClose={() => setNewKey("")}>
        <p className="mb-3 text-sm text-muted">
          This is the only time it will be shown. Only its hash is stored, so a
          lost key is replaced, never recovered.
        </p>
        <div className="flex gap-2">
          <Input readOnly value={newKey} className="font-mono text-xs" />
          <Button
            variant="secondary"
            onClick={() => navigator.clipboard?.writeText(newKey)}
            aria-label="Copy"
          >
            <Copy className="h-3.5 w-3.5" />
          </Button>
        </div>
        <div className="mt-4 flex justify-end">
          <Button variant="primary" onClick={() => setNewKey("")}>
            Done
          </Button>
        </div>
      </Modal>
    </div>
  );
}

/** Change your own password. Succeeding signs every other device out. */
function PasswordCard({ onChanged }: { onChanged: () => Promise<void> }) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [done, setDone] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setDone("");
    if (next !== confirm) {
      setError("Those passwords don't match.");
      return;
    }
    setBusy(true);
    try {
      const result = await auth.changePassword(current, next);
      setDone(result.message);
      setCurrent("");
      setNext("");
      setConfirm("");
      // The server rotated our session token; reload so anything cached against
      // the old one is refreshed.
      await onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not change your password.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardTitle>Password</CardTitle>
      <p className="mb-4 text-sm text-muted">
        Changing it signs you out everywhere else. This tab stays signed in.
      </p>
      <form onSubmit={submit} className="grid gap-3 sm:grid-cols-3">
        <Field label="Current">
          <Input
            type="password"
            value={current}
            onChange={(e) => setCurrent(e.target.value)}
            autoComplete="current-password"
            required
          />
        </Field>
        <Field label="New">
          <Input
            type="password"
            value={next}
            onChange={(e) => setNext(e.target.value)}
            autoComplete="new-password"
            required
          />
        </Field>
        <Field label="Confirm">
          <Input
            type="password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            autoComplete="new-password"
            required
          />
        </Field>
        <div className="sm:col-span-3 space-y-3">
          {error && <Alert>{error}</Alert>}
          {done && <Alert tone="positive">{done}</Alert>}
          <Button type="submit" variant="secondary" loading={busy}>
            Change password
          </Button>
        </div>
      </form>
    </Card>
  );
}

/**
 * Signed-in devices.
 *
 * `ip` and `user_agent` have been recorded on every login since the first
 * migration and shown nowhere — which made "is someone else in my account?"
 * an unanswerable question. This is the answer, and the button that acts on it.
 */
function SessionsCard({
  sessions,
  onChanged,
  onError,
}: {
  sessions: SessionInfo[] | null;
  onChanged: () => Promise<void>;
  onError: (message: string) => void;
}) {
  const [busy, setBusy] = useState("");

  async function run(id: string, fn: () => Promise<unknown>) {
    setBusy(id);
    try {
      await fn();
      await onChanged();
    } catch (err) {
      onError(err instanceof Error ? err.message : "That didn't work.");
    } finally {
      setBusy("");
    }
  }

  const others = (sessions ?? []).filter((s) => !s.current).length;

  return (
    <Card>
      <CardTitle
        action={
          others > 0 ? (
            <Button
              size="sm"
              variant="ghost"
              loading={busy === "all"}
              onClick={() => run("all", auth.revokeOtherSessions)}
            >
              Sign out everywhere else
            </Button>
          ) : undefined
        }
      >
        Signed-in devices
      </CardTitle>

      {sessions === null ? (
        <Skeleton className="h-16 w-full" />
      ) : sessions.length === 0 ? (
        <EmptyState
          icon={<Monitor className="h-5 w-5" />}
          title="No active sessions"
          description="Sign in again to see this device listed."
        />
      ) : (
        <ul className="divide-y divide-border">
          {sessions.map((session) => (
            <li key={session.id} className="flex items-center gap-3 py-2.5">
              <Monitor className="h-3.5 w-3.5 shrink-0 text-muted" />
              <div className="min-w-0 flex-1">
                <p className="flex items-center gap-2 truncate text-sm text-body">
                  {describeAgent(session.user_agent)}
                  {session.current && <Badge tone="accent">this device</Badge>}
                </p>
                <p className="text-2xs text-muted">
                  {session.ip ?? "unknown address"} · last active{" "}
                  {session.last_seen_at ? formatDate(session.last_seen_at) : "never"}
                </p>
              </div>
              <Button
                size="sm"
                variant="ghost"
                loading={busy === session.id}
                onClick={() => run(session.id, () => auth.revokeSession(session.id))}
                aria-label="Sign this device out"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

/** A user-agent string is not something to show a person. This picks out the
 *  browser and platform, which is all that makes a session recognisable. */
function describeAgent(agent: string | null): string {
  if (!agent) return "Unknown device";
  const browser =
    /Edg\//.test(agent) ? "Edge"
    : /OPR\/|Opera/.test(agent) ? "Opera"
    : /Chrome\//.test(agent) ? "Chrome"
    : /Safari\//.test(agent) ? "Safari"
    : /Firefox\//.test(agent) ? "Firefox"
    : null;
  const platform =
    /Windows/.test(agent) ? "Windows"
    : /Android/.test(agent) ? "Android"
    : /iPhone|iPad|iOS/.test(agent) ? "iOS"
    : /Mac OS X|Macintosh/.test(agent) ? "macOS"
    : /Linux/.test(agent) ? "Linux"
    : null;
  if (browser && platform) return `${browser} on ${platform}`;
  if (browser || platform) return browser ?? platform ?? "";
  // Scripts and curl land here; the raw string is the only useful identifier.
  return agent.slice(0, 40);
}

function formatDate(value: string): string {
  if (!value) return "—";
  return new Date(value).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}
