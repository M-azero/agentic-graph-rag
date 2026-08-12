"""Request/response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from graphrag.core.types import RetrievedChunk


class Source(BaseModel):
    chunk_id: str
    source: str
    snippet: str
    score: float
    retriever: str
    # Whether the answer actually cited this source, as opposed to retrieval
    # merely surfacing it. Retrieval over-fetches by design, so the two are
    # routinely different and a client can now show "evidence used" apart from
    # "also found". Default False keeps `/search`, which has no answer to check
    # against, honest rather than optimistic.
    cited: bool = False

    @classmethod
    def from_chunk(cls, c: RetrievedChunk, cited: bool = False) -> Source:
        snippet = c.text if len(c.text) <= 400 else c.text[:400] + "…"
        return cls(
            chunk_id=c.chunk_id, source=c.source, snippet=snippet,
            score=round(c.score, 4), retriever=c.retriever, cited=cited,
        )


# A question is a question. The cap is far above any real one and far below
# what makes the request expensive: the text is embedded, screened by the guard,
# and then resent on every turn of the agent's tool loop, so an unbounded field
# is an unbounded bill for one unit of message quota. Bounded here rather than
# only at the proxy so the API is safe when something else fronts it.
_MAX_QUESTION_CHARS = 8000
_MAX_THREAD_ID_CHARS = 128
_MAX_MODEL_ID_CHARS = 128
_MAX_PRESET_ID_CHARS = 32
# A UUID in any spelling anyone writes one. Bounded for the same reason the
# others are: these are looked up, and an unbounded string is an unbounded key.
_MAX_ID_CHARS = 64
_MAX_SHELF_NAME_CHARS = 80


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=_MAX_QUESTION_CHARS)
    style: str = Field("detailed", description="concise | detailed | technical | eli5")
    # The job preset (see agent/presets.py). Unknown values clamp to `general`,
    # which defers to `style` — so a client that sends neither behaves exactly
    # as it did before presets existed.
    preset: str | None = Field(None, max_length=_MAX_PRESET_ID_CHARS)
    thread_id: str = Field(
        "default",
        max_length=_MAX_THREAD_ID_CHARS,
        description="conversation id for multi-turn memory",
    )
    # Which shelf to search. Omitted -> the default shelf. Ignored when
    # `thread_id` names a conversation that is already pinned to one: a thread's
    # shelf is fixed at creation, and its memory is keyed on that shelf's corpus.
    shelf_id: str | None = Field(None, max_length=_MAX_ID_CHARS)
    # None -> the server default (api.stream in config) decides.
    stream: bool | None = None
    # Chat model id from the allowed list; unknown ids fall back to the default.
    model: str | None = Field(None, max_length=_MAX_MODEL_ID_CHARS)


class ToolCall(BaseModel):
    tool: str
    args: dict = {}


class SafetyInfo(BaseModel):
    """The guard's verdict, surfaced to the client when it did more than allow.

    `action` is block | flag | redacted; `stage` is input | output. Present only
    when the guard blocked, flagged, or redacted — a plain allow stays None.
    """

    action: str
    stage: str
    reasons: list[str] = []


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    tool_calls: list[ToolCall] = []
    safety: SafetyInfo | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=_MAX_QUESTION_CHARS)
    # Bounded on both sides: a negative k slices a result list from the wrong
    # end, and a huge one is only ever capped by chance (`candidate_k`).
    k: int = Field(8, ge=1, le=50)


class SearchResponse(BaseModel):
    results: list[Source] = []


class CompareRequest(BaseModel):
    """A side-by-side comparison.

    `subjects` is capped because it fans out: the composed question drives the
    agent's `compare` tool, which runs one full hybrid retrieval *per subject* —
    an embedding call, three Neo4j queries and a rerank each, and under a
    generative reranker that is `candidate_k` model calls each. Unbounded, one
    request costing one unit of message quota could issue thousands.
    """

    subjects: list[str] = Field(..., min_length=2, max_length=8)
    aspects: list[str] = Field(default=[], max_length=12)
    style: str = "detailed"
    preset: str | None = Field(None, max_length=_MAX_PRESET_ID_CHARS)
    thread_id: str = Field("default", max_length=_MAX_THREAD_ID_CHARS)
    shelf_id: str | None = Field(None, max_length=_MAX_ID_CHARS)
    model: str | None = Field(None, max_length=_MAX_MODEL_ID_CHARS)

    @field_validator("subjects", "aspects")
    @classmethod
    def _bound_each_entry(cls, values: list[str]) -> list[str]:
        """The list length is not the only dimension — one 5 MB subject is the
        same problem wearing a different hat."""
        for value in values:
            if len(value) > 200:
                raise ValueError("each entry must be 200 characters or fewer")
        return values


class IngestResponse(BaseModel):
    job_id: str
    status: str


class IngestStatus(BaseModel):
    job_id: str
    # queued | running | done | partial | error
    # `partial` = chunks stored and searchable, but graph extraction failed for
    # some of them, so the knowledge graph is incomplete. `detail` says how many.
    status: str
    detail: str = ""
    documents: int = 0
    chunks: int = 0
    entities: int = 0
    relations: int = 0


class StoredFile(BaseModel):
    file_id: str
    name: str
    source: str
    shelf_id: str | None = None


class FileList(BaseModel):
    files: list[StoredFile] = []
    # `used` and `limit` are per account, not per shelf: the file quota is
    # something you hold across every shelf, so a panel showing one shelf's
    # documents must still show the whole allowance or "3/10" would mean
    # something different on every shelf.
    used: int = 0
    limit: int = 0


# --- shelves -----------------------------------------------------------------

class ShelfInfo(BaseModel):
    """One shelf. `id` is null for the implicit default shelf of an account
    that has no rows yet — clients send it back as-is, and null means the
    default, so they never need to know the difference."""

    id: str | None = None
    name: str
    slug: str = ""
    preset: str = "general"
    is_default: bool = False
    files: int = 0


class ShelfList(BaseModel):
    shelves: list[ShelfInfo] = []
    max_shelves: int = 0


class ShelfCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=_MAX_SHELF_NAME_CHARS)
    preset: str | None = Field(None, max_length=_MAX_PRESET_ID_CHARS)


class ShelfUpdate(BaseModel):
    """Name and preset only. A shelf's slug is its storage namespace and is
    fixed at creation — changing it would leave every chunk, entity and summary
    the shelf holds under a corpus nothing queries."""

    name: str | None = Field(None, min_length=1, max_length=_MAX_SHELF_NAME_CHARS)
    preset: str | None = Field(None, max_length=_MAX_PRESET_ID_CHARS)


class ShelfDeleted(BaseModel):
    id: str
    chunks_removed: int = 0
    files_removed: int = 0


class PresetOption(BaseModel):
    """One entry of the job picker. The UI renders this list rather than
    carrying its own copy, so adding a preset server-side offers it."""

    id: str
    label: str
    emoji: str = ""
    description: str = ""


class DeleteResponse(BaseModel):
    file_id: str
    chunks_removed: int = 0


class SignupRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=1, max_length=128)


class VerifyRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    code: str = Field(..., min_length=4, max_length=12)


class EmailRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=1, max_length=128)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)


class ResetPasswordRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    code: str = Field(..., min_length=4, max_length=12)
    password: str = Field(..., min_length=1, max_length=128)


class ChangePasswordRequest(BaseModel):
    # Length is validated by `validate_password`, not here: a bound of 1 lets
    # the server return its own "too short" message rather than a 422 the UI
    # would have to translate.
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=1, max_length=128)


class SessionInfo(BaseModel):
    """One signed-in device. `ip` and `user_agent` are shown so a session the
    user does not recognise is recognisable as such."""

    id: str
    created_at: str = ""
    last_seen_at: str | None = None
    ip: str | None = None
    user_agent: str | None = None
    current: bool = False


class SessionList(BaseModel):
    sessions: list[SessionInfo] = []


class Acknowledged(BaseModel):
    ok: bool = True
    message: str = ""


class ModelOption(BaseModel):
    model: str
    label: str
    provider: str


class Me(BaseModel):
    """Everything the UI needs to render a signed-in session."""

    user_id: str
    email: str = ""
    role: str = "user"
    tenant_id: str = ""
    authenticated: bool = True
    models: list[ModelOption] = []
    default_model: str = ""
    presets: list[PresetOption] = []


class APIKeyInfo(BaseModel):
    id: int
    label: str = ""
    created_at: str = ""
    last_used_at: str | None = None


class APIKeyList(BaseModel):
    keys: list[APIKeyInfo] = []


class APIKeyCreate(BaseModel):
    label: str = Field("", max_length=64)


class APIKeyCreated(BaseModel):
    id: int
    api_key: str  # shown once


class ThreadInfo(BaseModel):
    id: str
    title: str
    created_at: str = ""
    updated_at: str = ""
    # The shelf this conversation asks about; null is the default shelf.
    shelf_id: str | None = None


class ThreadList(BaseModel):
    threads: list[ThreadInfo] = []


class ThreadCreate(BaseModel):
    title: str = Field("New chat", max_length=120)
    # Pinned at creation and never changed afterwards — see `Thread.shelf_id`.
    shelf_id: str | None = Field(None, max_length=_MAX_ID_CHARS)


class ThreadUpdate(BaseModel):
    title: str | None = Field(None, max_length=120)


class MessageInfo(BaseModel):
    id: int
    role: str
    content: str
    sources: list = []
    model: str = ""
    created_at: str = ""


class ThreadMessages(BaseModel):
    thread: ThreadInfo
    messages: list[MessageInfo] = []


class LimitsInfo(BaseModel):
    """A user's allowances and what they've used, for the account page."""

    limits: dict[str, int] = {}
    usage: dict[str, int] = {}
    files_used: int = 0
    storage_used_mb: float = 0.0
    threads_used: int = 0


class UserCreate(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=48)


class UserInfo(BaseModel):
    user_id: str


class UserCreated(BaseModel):
    user_id: str
    api_key: str | None = None  # returned once; only when auth is enabled


class UsersList(BaseModel):
    users: list[str] = []


class KeysRevoked(BaseModel):
    user_id: str
    revoked: int


class UsageReport(BaseModel):
    # user id -> streamed answer tokens (approximate; counted per SSE chunk)
    tokens: dict[str, int] = {}


# --- admin -------------------------------------------------------------------

class AdminUser(BaseModel):
    id: str
    email: str
    role: str
    status: str
    tenant_id: str
    created_at: str = ""
    last_login_at: str | None = None
    email_verified: bool = False
    # Lockout state, so the admin table can explain "they can't sign in"
    # without the admin having to guess.
    locked_until: str | None = None
    failed_logins: int = 0
    password_changed_at: str | None = None
    files: int = 0
    threads: int = 0
    messages_30d: int = 0
    tokens_30d: int = 0


class AdminUserList(BaseModel):
    users: list[AdminUser] = []
    total: int = 0
    page: int = 1
    size: int = 25


class AdminUserDetail(BaseModel):
    user: AdminUser
    limits: dict[str, int] = {}
    overrides: dict[str, int | None] = {}
    usage: dict[str, int] = {}
    storage_used_mb: float = 0.0
    graph: dict[str, int] = {}
    files: list[StoredFile] = []


class AdminUserCreate(BaseModel):
    """Invite an account into existence. No password field on purpose — the
    invitee sets their own via the emailed code, so no secret is ever typed by
    one person on behalf of another."""

    email: str = Field(..., min_length=3, max_length=320)
    role: str = Field("user", description="user | admin")


class UserPatch(BaseModel):
    status: str | None = Field(None, description="active | suspended")
    role: str | None = Field(None, description="user | admin")


class LimitsPatch(BaseModel):
    """Every field optional. Null clears an override back to the global default."""

    messages_per_minute: int | None = None
    messages_per_day: int | None = None
    tokens_per_day: int | None = None
    tokens_per_month: int | None = None
    max_files: int | None = None
    max_file_mb: int | None = None
    max_storage_mb: int | None = None
    max_chunks: int | None = None
    max_threads: int | None = None


class BulkLimits(BaseModel):
    """Apply to every user at once. `clear` drops all per-user overrides so
    everyone inherits the (possibly just-updated) global defaults."""

    set: LimitsPatch | None = None
    clear: bool = False


class UsagePoint(BaseModel):
    bucket: str
    messages: int = 0
    tokens: int = 0
    uploads: int = 0


class UsageSeries(BaseModel):
    points: list[UsagePoint] = []
    totals: dict[str, int] = {}


class GraphNode(BaseModel):
    key: str
    name: str = ""
    type: str = ""
    degree: int = 0


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str = ""


class GraphSample(BaseModel):
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []


class SystemStatus(BaseModel):
    version: str = ""
    neo4j: bool = False
    redis: bool = False
    database: bool = False
    users: int = 0
    active_users: int = 0
    threads: int = 0
    files: int = 0
    vector_provider: str = ""
    memory_backend: str = ""
    default_model: str = ""


class ModelSettings(BaseModel):
    """Which of the configured models the chat UI may offer."""

    available: list[ModelOption] = []
    enabled: list[str] = []


class ModelSettingsUpdate(BaseModel):
    enabled: list[str] = []


class PurgeResult(BaseModel):
    tenant_id: str = ""
    graph_nodes: int = 0
    files_removed: int = 0
    vectors_removed: bool = False
    rows_removed: bool = False
    errors: list[str] = []


class Health(BaseModel):
    status: str
    version: str


class Ready(BaseModel):
    ready: bool
    neo4j: bool
    redis: bool
    # Accounts, limits, usage, chat history. Gates `ready` only when auth is on;
    # reported either way so a probe shows the state without acting on it.
    database: bool = False
