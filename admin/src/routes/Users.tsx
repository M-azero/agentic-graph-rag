import { Lock, Mail, KeyRound, Plus, Search, Trash2, UserCog } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { admin, type AdminUser } from "../api";
import { DataTable, type Column } from "../components/DataTable";
import { RowMenu, type Action } from "../components/RowMenu";
import {
  Alert,
  Badge,
  Button,
  Card,
  Field,
  Input,
  Modal,
  Select,
  compactNumber,
  relativeTime,
} from "../components/ui";

const STATUS_TONE = {
  active: "positive",
  pending: "caution",
  suspended: "danger",
  deleted: "neutral",
} as const;

const PAGE_SIZE = 25;

function isLocked(user: AdminUser): boolean {
  return Boolean(user.locked_until && new Date(user.locked_until).getTime() > Date.now());
}

export default function Users() {
  const navigate = useNavigate();
  const [users, setUsers] = useState<AdminUser[] | null>(null);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [inviting, setInviting] = useState(false);
  const [confirm, setConfirm] = useState<{ user: AdminUser } | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await admin.users({ query, status, page, size: PAGE_SIZE });
      setUsers(data.users);
      setTotal(data.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load users.");
    }
  }, [query, status, page]);

  useEffect(() => {
    // Debounced so typing a search doesn't fire a request per keystroke.
    const timer = setTimeout(() => void load(), 250);
    return () => clearTimeout(timer);
  }, [load]);

  async function run(fn: () => Promise<unknown>, done: string) {
    setError("");
    setNotice("");
    try {
      await fn();
      setNotice(done);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "That didn't work.");
    }
  }

  function actionsFor(user: AdminUser): Action[] {
    const suspended = user.status === "suspended";
    return [
      {
        label: suspended ? "Reactivate" : "Suspend",
        icon: <UserCog className="h-3.5 w-3.5" />,
        onSelect: () =>
          run(
            () => admin.patchUser(user.id, { status: suspended ? "active" : "suspended" }),
            suspended ? "Account reactivated." : "Account suspended and signed out.",
          ),
      },
      {
        label: user.role === "admin" ? "Remove admin role" : "Make admin",
        icon: <UserCog className="h-3.5 w-3.5" />,
        onSelect: () =>
          run(
            () => admin.patchUser(user.id, { role: user.role === "admin" ? "user" : "admin" }),
            "Role updated.",
          ),
      },
      {
        label: "Unlock sign-in",
        icon: <Lock className="h-3.5 w-3.5" />,
        // Only offered when it would do something — an "unlock" on an unlocked
        // account reads as though the admin missed something.
        disabled: !isLocked(user) && user.failed_logins === 0,
        onSelect: () => run(() => admin.unlock(user.id), "Sign-in unlocked."),
      },
      {
        label: "Force password reset",
        icon: <Mail className="h-3.5 w-3.5" />,
        onSelect: () =>
          run(
            () => admin.forcePasswordReset(user.id),
            "Sessions revoked and a reset code sent.",
          ),
      },
      {
        label: "Revoke API keys",
        icon: <KeyRound className="h-3.5 w-3.5" />,
        onSelect: () => run(() => admin.revokeKeys(user.id), "API keys revoked."),
      },
      {
        label: "Resend verification",
        icon: <Mail className="h-3.5 w-3.5" />,
        disabled: user.status !== "pending",
        onSelect: () =>
          run(() => admin.resendVerification(user.id), "Verification code sent."),
      },
      {
        label: "Delete user and data",
        icon: <Trash2 className="h-3.5 w-3.5" />,
        danger: true,
        onSelect: () => setConfirm({ user }),
      },
    ];
  }

  const columns: Column<AdminUser>[] = [
    {
      key: "email",
      label: "User",
      sortable: true,
      value: (u) => u.email,
      render: (u) => (
        <span className="flex items-center gap-2">
          <span className="font-medium text-strong">{u.email}</span>
          {u.role === "admin" && <Badge tone="accent">admin</Badge>}
          {isLocked(u) && <Badge tone="caution">locked</Badge>}
        </span>
      ),
    },
    {
      key: "status",
      label: "Status",
      sortable: true,
      value: (u) => u.status,
      render: (u) => (
        <Badge tone={STATUS_TONE[u.status as keyof typeof STATUS_TONE] ?? "neutral"}>
          {u.status}
        </Badge>
      ),
    },
    {
      key: "last_login_at",
      label: "Last seen",
      sortable: true,
      value: (u) => u.last_login_at ?? "",
      render: (u) => <span className="text-muted">{relativeTime(u.last_login_at)}</span>,
      className: "text-xs",
    },
    {
      key: "files",
      label: "Docs",
      align: "right",
      sortable: true,
      value: (u) => u.files,
      render: (u) => u.files,
      className: "text-xs text-muted",
    },
    {
      key: "threads",
      label: "Chats",
      align: "right",
      sortable: true,
      value: (u) => u.threads,
      render: (u) => u.threads,
      className: "text-xs text-muted",
    },
    {
      key: "messages_30d",
      label: "Msgs 30d",
      align: "right",
      sortable: true,
      value: (u) => u.messages_30d,
      render: (u) => u.messages_30d.toLocaleString(),
      className: "text-xs text-muted",
    },
    {
      key: "tokens_30d",
      label: "Tokens 30d",
      align: "right",
      sortable: true,
      value: (u) => u.tokens_30d,
      render: (u) => compactNumber(u.tokens_30d),
      className: "text-xs text-muted",
    },
    {
      key: "actions",
      label: "",
      align: "right",
      render: (u) => <RowMenu actions={actionsFor(u)} label={`Actions for ${u.email}`} />,
    },
  ];

  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="space-y-4">
      {error && <Alert>{error}</Alert>}
      {notice && <Alert tone="positive">{notice}</Alert>}

      {/* One row: search grows, the filter and the action keep their natural
          width. `!w-auto` because the shared field class sets w-full, and
          Tailwind resolves that conflict by CSS order rather than by the order
          of names here — a plain `w-auto` loses and the toolbar wraps. */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-[200px] max-w-sm flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted" />
          <Input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setPage(1);
            }}
            placeholder="Search by email"
            className="pl-9"
          />
        </div>
        <Select
          value={status}
          onChange={(e) => {
            setStatus(e.target.value);
            setPage(1);
          }}
          className="!w-auto"
          aria-label="Filter by status"
        >
          <option value="">All statuses</option>
          <option value="active">Active</option>
          <option value="pending">Pending</option>
          <option value="suspended">Suspended</option>
        </Select>
        <Button variant="primary" className="ml-auto" onClick={() => setInviting(true)}>
          <Plus className="h-3.5 w-3.5" />
          Invite
        </Button>
      </div>

      <Card className="overflow-hidden p-0 [&>div]:p-0">
        <DataTable
          columns={columns}
          rows={users ?? []}
          rowKey={(u) => u.id}
          loading={users === null}
          empty={{ title: "No users match", detail: "Try a different search or filter." }}
          onRowClick={(u) => navigate(`/users/${u.id}`)}
        />
      </Card>

      {pages > 1 && (
        <div className="flex items-center justify-between text-sm text-muted">
          <span>
            Page {page} of {pages} · {total} users
          </span>
          <div className="flex gap-2">
            <Button disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
              Previous
            </Button>
            <Button disabled={page >= pages} onClick={() => setPage((p) => p + 1)}>
              Next
            </Button>
          </div>
        </div>
      )}

      {inviting && (
        <InviteModal
          onClose={() => setInviting(false)}
          onDone={async (message) => {
            setInviting(false);
            setNotice(message);
            await load();
          }}
        />
      )}

      {confirm && (
        <Modal title={`Delete ${confirm.user.email}?`} onClose={() => setConfirm(null)}>
          <p className="text-sm text-muted">
            Removes their account, documents, conversations and everything extracted from
            them — across Postgres, Neo4j and the vector store. This cannot be undone.
          </p>
          <div className="mt-4 flex justify-end gap-2">
            <Button onClick={() => setConfirm(null)}>Cancel</Button>
            <Button
              variant="danger"
              onClick={async () => {
                const target = confirm.user;
                setConfirm(null);
                await run(() => admin.deleteUser(target.id), `${target.email} deleted.`);
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

function InviteModal({
  onClose,
  onDone,
}: {
  onClose: () => void;
  onDone: (message: string) => void;
}) {
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("user");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const user = await admin.createUser(email.trim(), role);
      onDone(`Invited ${user.email}. They have been emailed a code to set a password.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create the account.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal title="Invite a user" onClose={onClose}>
      <form onSubmit={submit} className="space-y-3">
        <p className="text-xs text-muted">
          Creates a verified account and emails a code to set a password. No password is
          chosen here — the invitee picks their own.
        </p>
        <Field label="Email">
          <Input
            type="email"
            required
            autoFocus
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="person@example.com"
          />
        </Field>
        <Field label="Role">
          <Select value={role} onChange={(e) => setRole(e.target.value)}>
            <option value="user">User</option>
            <option value="admin">Admin</option>
          </Select>
        </Field>
        {error && <Alert>{error}</Alert>}
        <div className="flex justify-end gap-2 pt-1">
          <Button type="button" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" variant="primary" loading={busy}>
            Send invite
          </Button>
        </div>
      </form>
    </Modal>
  );
}
