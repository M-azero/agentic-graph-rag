// The console's API client. Deliberately covers only what the console calls —
// `/admin/*`, plus the two `/auth` endpoints needed to know who is looking and
// to sign them out. It shares no code with the chat app's client, so none of
// the chat DTOs (threads, messages, sources, SSE) exist here at all.
//
// Requests carry the session cookie. There is no dev `X-User-Id` escape hatch:
// the console is only meaningful against a server with auth on.

const API = "/api";

export class ApiError extends Error {
  status: number;
  code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    let code: string | undefined;
    try {
      const body = await res.json();
      const d = body?.detail;
      if (typeof d === "string") {
        detail = d;
      } else if (d && typeof d === "object") {
        detail = d.message ?? JSON.stringify(d);
        code = d.code;
      }
    } catch {
      /* non-JSON error body — keep the status line */
    }
    throw new ApiError(detail, res.status, code);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  return fetch(`${API}${path}`, { ...init, credentials: "include" }).then(jsonOrThrow<T>);
}

const body = (method: string) => (payload: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});

const post = body("POST");
const put = body("PUT");
const patch = body("PATCH");

// -- types --------------------------------------------------------------------

export interface Me {
  user_id: string;
  email: string;
  role: string;
  tenant_id: string;
}

export interface ModelOption {
  model: string;
  label: string;
  provider: string;
}

export interface AdminUser {
  id: string;
  email: string;
  role: string;
  status: string;
  tenant_id: string;
  created_at: string;
  last_login_at: string | null;
  email_verified: boolean;
  locked_until: string | null;
  failed_logins: number;
  password_changed_at: string | null;
  files: number;
  threads: number;
  messages_30d: number;
  tokens_30d: number;
}

export interface AdminUserDetail {
  user: AdminUser;
  limits: Record<string, number>;
  overrides: Record<string, number | null>;
  usage: Record<string, number>;
  storage_used_mb: number;
  graph: Record<string, number>;
  files: { file_id: string; name: string; source: string }[];
}

export interface UsageSeries {
  points: { bucket: string; messages: number; tokens: number; uploads: number }[];
  totals: Record<string, number>;
}

export interface SystemStatus {
  version: string;
  neo4j: boolean;
  redis: boolean;
  database: boolean;
  users: number;
  active_users: number;
  threads: number;
  files: number;
  vector_provider: string;
  memory_backend: string;
  default_model: string;
}

export interface GraphSample {
  nodes: { key: string; name: string; type: string; degree: number }[];
  edges: { source: string; target: string; type: string }[];
}

export interface AuditEntry {
  id: number;
  action: string;
  actor: string | null;
  target: string | null;
  detail: Record<string, unknown>;
  created_at: string;
}

interface Ack {
  ok: boolean;
  message: string;
}

// -- endpoints ----------------------------------------------------------------

export const session = {
  me: () => request<Me>("/auth/me"),
  logout: () => request<Ack>("/auth/logout", { method: "POST" }),
};

export const admin = {
  users: (params: { query?: string; status?: string; page?: number; size?: number } = {}) => {
    const q = new URLSearchParams();
    if (params.query) q.set("query", params.query);
    if (params.status) q.set("status", params.status);
    q.set("page", String(params.page ?? 1));
    q.set("size", String(params.size ?? 25));
    return request<{ users: AdminUser[]; total: number; page: number; size: number }>(
      `/admin/users?${q}`,
    );
  },
  user: (id: string) => request<AdminUserDetail>(`/admin/users/${id}`),
  createUser: (email: string, role: string) =>
    request<AdminUser>("/admin/users", post({ email, role })),
  patchUser: (id: string, changes: { status?: string; role?: string }) =>
    request<AdminUser>(`/admin/users/${id}`, patch(changes)),
  deleteUser: (id: string, keepAccount = false) =>
    request<{ tenant_id: string; errors: string[] }>(
      `/admin/users/${id}?keep_account=${keepAccount}`,
      { method: "DELETE" },
    ),
  revokeKeys: (id: string) =>
    request<Ack>(`/admin/users/${id}/revoke-keys`, { method: "POST" }),
  resendVerification: (id: string) =>
    request<Ack>(`/admin/users/${id}/resend-verification`, { method: "POST" }),
  forcePasswordReset: (id: string) =>
    request<Ack>(`/admin/users/${id}/force-password-reset`, { method: "POST" }),
  unlock: (id: string) => request<Ack>(`/admin/users/${id}/unlock`, { method: "POST" }),

  globalLimits: () => request<Record<string, number>>("/admin/limits"),
  setGlobalLimits: (values: Record<string, number>) =>
    request<Record<string, number>>("/admin/limits", put(values)),
  setUserLimits: (id: string, values: Record<string, number | null>) =>
    request<Record<string, number>>(`/admin/users/${id}/limits`, put(values)),
  clearUserLimits: (id: string) =>
    request<Ack>(`/admin/users/${id}/limits`, { method: "DELETE" }),
  bulkLimits: (payload: { set?: Record<string, number>; clear?: boolean }) =>
    request<Ack>("/admin/limits/bulk", post(payload)),

  usage: (days = 30, userId?: string) =>
    request<UsageSeries>(`/admin/usage?days=${days}${userId ? `&user_id=${userId}` : ""}`),
  graph: (id: string) => request<Record<string, number>>(`/admin/users/${id}/graph`),
  graphSample: (id: string, limit = 80) =>
    request<GraphSample>(`/admin/users/${id}/graph/sample?limit=${limit}`),
  system: () => request<SystemStatus>("/admin/system"),
  models: () => request<{ available: ModelOption[]; enabled: string[] }>("/admin/models"),
  setModels: (enabled: string[]) =>
    request<{ available: ModelOption[]; enabled: string[] }>("/admin/models", put({ enabled })),
  audit: (limit = 100) => request<AuditEntry[]>(`/admin/audit?limit=${limit}`),
};
