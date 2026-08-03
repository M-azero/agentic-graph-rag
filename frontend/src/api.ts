// API client for the chat app. Requests carry the session cookie
// (`credentials: "include"`), so identity is server-issued rather than a header
// the page picks.
//
// The admin console is a separate app with its own client (admin/src/api.ts),
// so nothing here knows the /admin endpoints exist.
//
// Queries are answered in one response rather than streamed: the output guard
// can block or redact an answer, and a streamed answer cannot be pulled back.
// An SSE client lived here until the guard made it unusable — see `streamQuery`
// in the history if streaming ever comes back.

export interface Source {
  chunk_id: string;
  source: string;
  snippet: string;
  score: number;
  retriever: string;
}

export interface SafetyInfo {
  action: string; // "block" | "flag" | "redacted"
  stage: string; // "input" | "output"
  reasons: string[];
}

export interface StoredFile {
  file_id: string;
  name: string;
  source: string;
}

export interface FileList {
  files: StoredFile[];
  used: number;
  limit: number;
}

export interface ModelOption {
  model: string;
  label: string;
  provider: string;
}

export interface Me {
  user_id: string;
  email: string;
  role: string;
  tenant_id: string;
  authenticated: boolean;
  models: ModelOption[];
  default_model: string;
}

export interface ThreadInfo {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface MessageInfo {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  model: string;
  created_at: string;
}

export interface LimitsInfo {
  limits: Record<string, number>;
  usage: Record<string, number>;
  files_used: number;
  storage_used_mb: number;
  threads_used: number;
}

export interface ApiKeyInfo {
  id: number;
  label: string;
  created_at: string;
  last_used_at: string | null;
}

const API = "/api";

/** A 429 carries the structured limit detail the quota banner renders. */
export interface LimitDetail {
  code: string;
  limit: string;
  used: number;
  max: number;
  retry_after: number;
  message: string;
}

export class ApiError extends Error {
  status: number;
  code?: string;
  limit?: LimitDetail;

  constructor(message: string, status: number, code?: string, limit?: LimitDetail) {
    super(message);
    this.status = status;
    this.code = code;
    this.limit = limit;
  }
}

// Dev builds may still identify with a header when the server has auth off.
// Production never sends it: with auth on the server ignores it entirely.
const DEV_USER_KEY = "graphrag_dev_user";

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const h: Record<string, string> = { ...extra };
  if (import.meta.env.DEV) {
    const devUser = localStorage.getItem(DEV_USER_KEY);
    if (devUser) h["X-User-Id"] = devUser;
  }
  return h;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    ...init,
    credentials: "include",
    headers: headers(init.headers as Record<string, string>),
  });
  return jsonOrThrow<T>(res);
}

function json(body: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

// Every non-streaming call goes through this. Without it an error response
// parses as success with all fields undefined — a rejected upload showed
// "queued" forever while polling /ingest/undefined for eternity.
async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    let code: string | undefined;
    let limit: LimitDetail | undefined;
    try {
      const body = await res.json();
      const d = body?.detail;
      if (typeof d === "string") {
        detail = d;
      } else if (d && typeof d === "object") {
        detail = d.message ?? JSON.stringify(d);
        code = d.code;
        if (d.code === "limit_exceeded") limit = d as LimitDetail;
      }
    } catch {
      /* non-JSON error body — keep the status line */
    }
    throw new ApiError(detail, res.status, code, limit);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export interface QueryResult {
  answer: string;
  sources: Source[];
  safety: SafetyInfo | null;
}

// Non-streaming query. Used when the server must fully enforce the output guard
// (block / redact) before the answer is shown — a streamed answer can't be pulled
// back, so hard enforcement needs the whole answer in one response. Trades
// token-by-token rendering for a firm safety guarantee plus a verdict to display.
export async function queryOnce(
  question: string,
  style: string,
  threadId: string,
  model?: string,
  signal?: AbortSignal,
): Promise<QueryResult> {
  const res = await fetch(`${API}/query`, {
    method: "POST",
    credentials: "include",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      question,
      style,
      thread_id: threadId,
      stream: false,
      ...(model ? { model } : {}),
    }),
    signal,
  });
  const data = await jsonOrThrow<{
    answer: string;
    sources: Source[];
    safety: SafetyInfo | null;
  }>(res);
  return {
    answer: data.answer,
    sources: data.sources ?? [],
    safety: data.safety ?? null,
  };
}

// -- auth ---------------------------------------------------------------------

export interface SessionInfo {
  id: string;
  created_at: string;
  last_seen_at: string | null;
  ip: string | null;
  user_agent: string | null;
  current: boolean;
}

interface Ack {
  ok: boolean;
  message: string;
}

export const auth = {
  me: () => request<Me>("/auth/me"),
  signup: (email: string, password: string) =>
    request<Ack>("/auth/signup", json({ email, password })),
  verify: (email: string, code: string) => request<Me>("/auth/verify", json({ email, code })),
  resend: (email: string) => request<Ack>("/auth/resend", json({ email })),
  login: (email: string, password: string) =>
    request<Me>("/auth/login", json({ email, password })),
  logout: () => request<{ ok: boolean }>("/auth/logout", { method: "POST" }),
  limits: () => request<LimitsInfo>("/auth/limits"),

  // Passwords. The server answers forgotPassword identically for every address,
  // so there is nothing here to branch on — and nothing to leak.
  forgotPassword: (email: string) => request<Ack>("/auth/forgot-password", json({ email })),
  resetPassword: (email: string, code: string, password: string) =>
    request<Ack>("/auth/reset-password", json({ email, code, password })),
  changePassword: (current_password: string, new_password: string) =>
    request<Ack>("/auth/change-password", json({ current_password, new_password })),

  // Signed-in devices.
  sessions: () => request<{ sessions: SessionInfo[] }>("/auth/sessions"),
  revokeSession: (id: string) =>
    request<Ack>(`/auth/sessions/${id}`, { method: "DELETE" }),
  revokeOtherSessions: () => request<Ack>("/auth/sessions/revoke-all", { method: "POST" }),

  listKeys: () => request<{ keys: ApiKeyInfo[] }>("/auth/keys"),
  createKey: (label: string) =>
    request<{ id: number; api_key: string }>("/auth/keys", json({ label })),
  revokeKey: (id: number) => request<{ ok: boolean }>(`/auth/keys/${id}`, { method: "DELETE" }),
};

// -- conversations ------------------------------------------------------------

export const threads = {
  list: () => request<{ threads: ThreadInfo[] }>("/threads"),
  create: (title = "New chat") => request<ThreadInfo>("/threads", json({ title })),
  rename: (id: string, title: string) =>
    request<ThreadInfo>(`/threads/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),
  remove: (id: string) => request<{ ok: boolean }>(`/threads/${id}`, { method: "DELETE" }),
  messages: (id: string) =>
    request<{ thread: ThreadInfo; messages: MessageInfo[] }>(`/threads/${id}/messages`),
};

// -- documents ----------------------------------------------------------------

export async function uploadFile(file: File): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("file", file);
  return request<{ job_id: string }>("/ingest/upload", { method: "POST", body: form });
}

export const ingestStatus = (jobId: string) =>
  request<{ status: string; chunks?: number; entities?: number; detail?: string }>(
    `/ingest/${jobId}`,
  );

export const listFiles = () => request<FileList>("/ingest/files");

export const deleteFile = (fileId: string) =>
  request<{ chunks_removed: number }>(`/ingest/files/${fileId}`, { method: "DELETE" });
