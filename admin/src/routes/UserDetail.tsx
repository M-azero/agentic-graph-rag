import { ArrowLeft, KeyRound, Lock, Mail, Trash2 } from "lucide-react";
import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { admin, type AdminUserDetail, type GraphSample } from "../api";
import { LimitsForm } from "../components/LimitsForm";
import { Stat, StatRail } from "../components/Stat";
import {
  Alert,
  Badge,
  Button,
  Card,
  Modal,
  Skeleton,
  compactNumber,
  relativeTime,
} from "../components/ui";

// d3-force is only needed on this one page, below the fold.
const GraphView = lazy(() =>
  import("../components/GraphView").then((m) => ({ default: m.GraphView })),
);

export default function UserDetail() {
  const { userId = "" } = useParams();
  const navigate = useNavigate();
  const [detail, setDetail] = useState<AdminUserDetail | null>(null);
  const [graph, setGraph] = useState<GraphSample | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [saving, setSaving] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const load = useCallback(async () => {
    try {
      setDetail(await admin.user(userId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load this user.");
    }
  }, [userId]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    let alive = true;
    admin
      .graphSample(userId)
      .then((g) => alive && setGraph(g))
      // A tenant with nothing ingested has no graph; that is an empty state,
      // not an error worth interrupting the page for.
      .catch(() => alive && setGraph({ nodes: [], edges: [] }));
    return () => {
      alive = false;
    };
  }, [userId]);

  async function act(fn: () => Promise<unknown>, done: string) {
    setError("");
    setNotice("");
    try {
      const result = (await fn()) as { message?: string };
      setNotice(result?.message ?? done);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "That didn't work.");
    }
  }

  if (!detail) {
    return (
      <div className="space-y-3">
        {error && <Alert>{error}</Alert>}
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  const { user, limits, overrides, graph: stats } = detail;
  const suspended = user.status === "suspended";
  const locked = Boolean(
    user.locked_until && new Date(user.locked_until).getTime() > Date.now(),
  );

  return (
    <div className="space-y-5">
      <Link
        to="/users"
        className="inline-flex items-center gap-1.5 text-sm text-muted hover:text-body"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        All users
      </Link>

      {error && <Alert>{error}</Alert>}
      {notice && <Alert tone="positive">{notice}</Alert>}

      <Card>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-lg font-semibold text-strong">{user.email}</h2>
              <Badge
                tone={suspended ? "danger" : user.status === "active" ? "positive" : "caution"}
              >
                {user.status}
              </Badge>
              {user.role === "admin" && <Badge tone="accent">admin</Badge>}
              {locked && <Badge tone="caution">locked</Badge>}
            </div>
            <p className="mt-1 font-mono text-xs text-muted">{user.tenant_id}</p>
            <p className="mt-1 text-xs text-muted">
              Joined {formatDate(user.created_at)} · last seen{" "}
              {relativeTime(user.last_login_at)}
              {user.email_verified ? "" : " · email unverified"}
              {user.password_changed_at
                ? ` · password changed ${relativeTime(user.password_changed_at)}`
                : ""}
            </p>
            {(locked || user.failed_logins > 0) && (
              <p className="mt-1 text-xs text-caution">
                {user.failed_logins} failed sign-in attempt
                {user.failed_logins === 1 ? "" : "s"}
                {locked && user.locked_until
                  ? ` · locked until ${new Date(user.locked_until).toLocaleTimeString()}`
                  : ""}
              </p>
            )}
          </div>

          <div className="flex flex-wrap gap-2">
            <Button
              onClick={() =>
                act(
                  () => admin.patchUser(user.id, { status: suspended ? "active" : "suspended" }),
                  suspended ? "User reactivated." : "User suspended and signed out.",
                )
              }
            >
              {suspended ? "Reactivate" : "Suspend"}
            </Button>
            <Button
              onClick={() =>
                act(
                  () =>
                    admin.patchUser(user.id, {
                      role: user.role === "admin" ? "user" : "admin",
                    }),
                  "Role updated.",
                )
              }
            >
              {user.role === "admin" ? "Remove admin" : "Make admin"}
            </Button>
            {(locked || user.failed_logins > 0) && (
              <Button onClick={() => act(() => admin.unlock(user.id), "Sign-in unlocked.")}>
                <Lock className="h-3.5 w-3.5" />
                Unlock
              </Button>
            )}
            <Button
              onClick={() =>
                act(
                  () => admin.forcePasswordReset(user.id),
                  "Sessions revoked and a reset code sent.",
                )
              }
            >
              <Mail className="h-3.5 w-3.5" />
              Force reset
            </Button>
            <Button onClick={() => act(() => admin.revokeKeys(user.id), "Keys revoked.")}>
              <KeyRound className="h-3.5 w-3.5" />
              Revoke keys
            </Button>
            {!user.email_verified && (
              <Button
                onClick={() => act(() => admin.resendVerification(user.id), "Code sent.")}
              >
                <Mail className="h-3.5 w-3.5" />
                Resend code
              </Button>
            )}
            <Button variant="danger" onClick={() => setConfirmDelete(true)}>
              <Trash2 className="h-3.5 w-3.5" />
              Delete
            </Button>
          </div>
        </div>
      </Card>

      <StatRail>
        <Stat label="Documents" value={user.files} sub={`${detail.storage_used_mb} MB stored`} />
        <Stat label="Conversations" value={user.threads} />
        <Stat
          label="Messages 30d"
          value={user.messages_30d.toLocaleString()}
          sub={`${compactNumber(user.tokens_30d)} tokens`}
        />
        <Stat label="Chunks indexed" value={(stats.chunks ?? 0).toLocaleString()} />
      </StatRail>

      <StatRail>
        <Stat label="Entities" value={(stats.entities ?? 0).toLocaleString()} />
        <Stat label="Relations" value={(stats.relations ?? 0).toLocaleString()} />
        <Stat label="Communities" value={(stats.communities ?? 0).toLocaleString()} />
        <Stat label="Failed sign-ins" value={user.failed_logins} tone={locked ? "caution" : undefined} />
      </StatRail>

      <Card title="Limits">
        <LimitsForm
          mode="override"
          values={overrides}
          inherited={limits}
          saving={saving}
          onSave={async (values) => {
            setSaving(true);
            await act(() => admin.setUserLimits(user.id, values), "Limits updated.");
            setSaving(false);
          }}
          onClear={() => act(() => admin.clearUserLimits(user.id), "Overrides cleared.")}
        />
      </Card>

      <Card title="Knowledge graph">
        {graph === null ? (
          <Skeleton className="h-56 w-full" />
        ) : (
          <Suspense fallback={<Skeleton className="h-56 w-full" />}>
            <GraphView sample={graph} />
          </Suspense>
        )}
      </Card>

      {confirmDelete && (
        <Modal title={`Delete ${user.email}?`} onClose={() => setConfirmDelete(false)}>
          <p className="text-sm text-muted">
            This removes their account, conversations, documents, vectors and graph. It
            cannot be undone.
          </p>
          <div className="mt-4 flex justify-end gap-2">
            <Button onClick={() => setConfirmDelete(false)}>Cancel</Button>
            <Button
              variant="danger"
              onClick={async () => {
                await admin.deleteUser(user.id);
                navigate("/users", { replace: true });
              }}
            >
              Delete everything
            </Button>
          </div>
        </Modal>
      )}
    </div>
  );
}

function formatDate(value: string): string {
  if (!value) return "—";
  return new Date(value).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}
