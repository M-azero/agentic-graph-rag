import clsx from "clsx";
import { Mail } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";

import { Wordmark } from "../components/ui";
import { useAuth } from "../lib/auth";

/**
 * The page before sign-in. Its one moving part is the demo in the hero: a
 * question is typed, the path that answers it lights up across a small
 * knowledge graph, and the answer arrives with its citations. That loop *is*
 * the description of the product; the prose below only names the parts.
 */
export default function Landing() {
  const { me } = useAuth();

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex min-h-full max-w-6xl flex-col px-5 sm:px-8">
        <nav className="flex items-center gap-2 py-5" aria-label="Main">
          <Wordmark />
          <div className="ml-8 hidden items-center gap-6 text-sm font-medium text-muted lg:flex">
            <a href="#how-it-works" className="hover:text-strong">How it works</a>
            <a href="#features" className="hover:text-strong">Features</a>
            <a href="#contact-us" className="hover:text-strong">Contact</a>
          </div>
          <div className="ml-auto flex items-center gap-1 sm:gap-2">
            <a
              href={__SOURCE_URL__}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-medium text-muted transition-colors hover:text-strong"
            >
              <GitHubMark className="h-4 w-4" />
              <span className="hidden sm:inline">GitHub</span>
            </a>
            {me ? (
              <PillLink to="/chat" primary>
                Open the app
              </PillLink>
            ) : (
              <>
                <PillLink to="/login">Sign in</PillLink>
                <PillLink to="/signup" primary>
                  Sign up
                </PillLink>
              </>
            )}
          </div>
        </nav>

        <header className="grid items-center gap-10 py-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] lg:gap-14 lg:py-20">
          <div>
            <h1 className="max-w-[14ch] font-display text-[2.6rem] font-bold leading-[1.02] tracking-tight text-strong sm:text-6xl">
              Answers that know how your documents connect.
            </h1>
            <p className="mt-6 max-w-[54ch] text-lg leading-relaxed text-body">
              {__APP_NAME__} reads your files twice: once for what they mean, and
              once for the people, companies and events they mention and how those
              relate. An agent decides which view to search, and every answer cites
              the passages it came from.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              {me ? (
                <PillLink to="/chat" primary large>
                  Open the app
                </PillLink>
              ) : (
                <>
                  <PillLink to="/signup" primary large>
                    Sign up
                  </PillLink>
                  <PillLink to="/login" large>
                    Sign in
                  </PillLink>
                </>
              )}
            </div>
          </div>

          <GraphDemo />
        </header>

        <WhySection />
        <FlowSection />
        <GroundedSection />
        <FeaturesSection />
        <SelfHostSection />

        <section
          id="contact-us"
          aria-labelledby="contact"
          className="glass my-12 scroll-mt-6 flex flex-col gap-6 rounded-2xl p-7 sm:p-10 md:flex-row md:items-center md:justify-between"
        >
          <div>
            <h2
              id="contact"
              className="font-display text-2xl font-bold tracking-tight text-strong"
            >
              Questions, or want it for your team?
            </h2>
            <p className="mt-2 max-w-[52ch] text-body">
              {__CONTACT_EMAIL__
                ? "Write to us and we’ll reply by email. Found a bug or have an idea? Open an issue on GitHub."
                : "Open an issue or a discussion on GitHub and we’ll pick it up there."}
            </p>
          </div>
          <div className="flex shrink-0 flex-wrap gap-3">
            {__CONTACT_EMAIL__ && (
              <a href={`mailto:${__CONTACT_EMAIL__}`} className={pill(true, true)}>
                <Mail className="h-4 w-4" />
                Contact us
              </a>
            )}
            <a
              href={__SOURCE_URL__}
              target="_blank"
              rel="noreferrer"
              className={pill(!__CONTACT_EMAIL__, true)}
            >
              <GitHubMark className="h-4 w-4" />
              View on GitHub
            </a>
          </div>
        </section>

        <footer className="mt-auto flex flex-wrap items-center justify-between gap-3 border-t border-border py-6 text-sm text-muted">
          <span>
            © {new Date().getFullYear()} {__APP_NAME__}
          </span>
          <span className="flex gap-5">
            {__CONTACT_EMAIL__ && (
              <a href={`mailto:${__CONTACT_EMAIL__}`} className="hover:text-strong">
                {__CONTACT_EMAIL__}
              </a>
            )}
            <a href={__SOURCE_URL__} target="_blank" rel="noreferrer" className="hover:text-strong">
              Source code
            </a>
          </span>
        </footer>
      </div>
    </div>
  );
}

// -- pieces --------------------------------------------------------------------

const pill = (primary = false, large = false) =>
  clsx(
    "inline-flex items-center justify-center gap-2 rounded-full font-medium transition",
    large ? "h-11 px-5 text-base" : "h-9 px-4 text-sm",
    primary
      ? "bg-accent text-accent-text shadow-glow hover:brightness-110"
      : "border border-border bg-surface text-strong hover:bg-raised",
  );

function PillLink({
  to,
  primary,
  large,
  className,
  children,
}: {
  to: string;
  primary?: boolean;
  large?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <Link to={to} className={clsx(pill(primary, large), className)}>
      {children}
    </Link>
  );
}

/** A page section: a display heading, an optional lede, then its content. */
function Section({
  id,
  title,
  lede,
  children,
}: {
  id: string;
  title: string;
  lede?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section aria-labelledby={`${id}-title`} id={id} className="scroll-mt-6 py-14 lg:py-20">
      <h2
        id={`${id}-title`}
        className="max-w-[22ch] font-display text-3xl font-bold leading-tight tracking-tight text-strong sm:text-4xl"
      >
        {title}
      </h2>
      {lede && <p className="mt-4 max-w-[62ch] text-lg leading-relaxed text-body">{lede}</p>}
      <div className="mt-10">{children}</div>
    </section>
  );
}

// -- why it's different ----------------------------------------------------------

function WhySection() {
  return (
    <Section
      id="why"
      title="Search finds passages. Questions are often about connections."
      lede={
        <>
          Ordinary document search returns the text closest in meaning to your
          question. That works for “find the part about X” and fails at “how are X
          and Y related?”, because two things being connected in the world doesn’t
          make them sound alike on the page. So {__APP_NAME__} keeps your documents
          in two forms and uses whichever the question needs.
        </>
      }
    >
      <div className="grid gap-5 md:grid-cols-2">
        <View
          name="Meaning index"
          what="Every passage is turned into a vector, so a search matches ideas rather than exact words."
          example="Find what we wrote about warehouse robots"
        />
        <View
          name="Knowledge graph"
          what="People, companies, products, places and events become nodes, linked by the relationships your documents describe."
          example="Who founded the company that makes the Pallet Pilot?"
        />
      </div>
      <p className="mt-6 max-w-[62ch] text-body">
        An agent reads each question and decides how to search: by meaning, by
        following the graph, by exact keywords, or all three at once.
      </p>
    </Section>
  );
}

function View({ name, what, example }: { name: string; what: string; example: string }) {
  return (
    <div className="glass rounded-xl p-6">
      <h3 className="text-lg font-semibold text-strong">{name}</h3>
      <p className="mt-2 leading-relaxed text-body">{what}</p>
      <p className="mt-5 text-sm text-muted">Good for questions like</p>
      <p className="mt-1.5 rounded-lg border-l-2 border-accent bg-accent/10 px-3 py-2 text-strong">
        “{example}”
      </p>
    </div>
  );
}

// -- the two pipelines --------------------------------------------------------

const UPLOAD_STEPS = [
  {
    title: "Read",
    body: "PDF, Word, slides, Excel, CSV, JSON, HTML, text and images. Scans and photos of text go through OCR first.",
  },
  {
    title: "Split and index",
    body: "Documents are cut into passages and each one is embedded for meaning search.",
  },
  {
    title: "Map",
    body: "A language model pulls out entities and relationships and adds them to your knowledge graph, linked to the passages that mention them.",
  },
  {
    title: "Tidy up",
    body: "Duplicates are merged (“Acme” and “Acme Robotics”), and related clusters are summarised so broad questions have an answer too.",
  },
];

const ASK_STEPS = [
  {
    title: "Check relevance",
    body: "The question is tested against your documents first. If nothing is close enough, it’s refused before any model runs.",
  },
  {
    title: "Plan the search",
    body: "The agent picks its tools: graph neighbours, subgraph expansion, entity lookup, comparison, or whole-collection summaries.",
  },
  {
    title: "Retrieve",
    body: "Meaning, graph and keyword searches run in parallel. Their results are merged and reranked so the best evidence comes first.",
  },
  {
    title: "Answer with sources",
    body: "The answer cites the exact passages it used, and its citations are checked before it reaches you.",
  },
];

function FlowSection() {
  return (
    <Section
      id="how-it-works"
      title="From upload to answer"
      lede="Two pipelines do the work. One runs once, when you add a document. The other runs every time you ask."
    >
      <div className="space-y-12">
        <Flow label="When you upload a file" steps={UPLOAD_STEPS} />
        <Flow label="When you ask a question" steps={ASK_STEPS} />
      </div>
      <p className="mt-10 max-w-[62ch] text-body">
        You can watch this happen. The pipeline view draws each pipeline as a
        diagram, lights up every step as a real run passes through it, and shows
        what each step received, returned, how long it took and what it cost.
      </p>
    </Section>
  );
}

/** A pipeline drawn the way the product thinks about it: steps as nodes on
 *  one edge. Horizontal from md up, vertical below it. */
function Flow({ label, steps }: { label: string; steps: { title: string; body: string }[] }) {
  return (
    <div>
      <h3 className="text-lg font-semibold text-strong">{label}</h3>
      <ol className="relative mt-6 grid gap-8 md:grid-cols-4 md:gap-6">
        {/* The edge the nodes sit on. */}
        <span
          aria-hidden
          className="absolute left-[7px] top-2 h-[calc(100%-1rem)] w-px bg-gradient-to-b from-accent/70 to-accent/10 md:left-2 md:right-8 md:top-[7px] md:h-px md:w-auto md:bg-gradient-to-r"
        />
        {steps.map((step, i) => (
          <li key={step.title} className="relative pl-8 md:pl-0 md:pt-8">
            <span
              aria-hidden
              className={clsx(
                "absolute left-0 top-0.5 h-[15px] w-[15px] rounded-full border-2 border-accent md:top-0",
                i === steps.length - 1 ? "bg-accent shadow-glow" : "bg-canvas",
              )}
            />
            <p className="font-semibold text-strong">{step.title}</p>
            <p className="mt-1.5 text-sm leading-relaxed text-body">{step.body}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}

// -- grounded answering -------------------------------------------------------

// Measured, not illustrative: the README's calibration run on a set of
// software-engineering notes, reranked with Cohere rerank-v4. The gate sits
// at 0.5.
const GATE = 0.5;
const SCORES = [
  { q: "Use case diagram and actors", score: 0.89, inDocs: true },
  { q: "Software requirements engineering", score: 0.7, inDocs: true },
  { q: "Write me a Python web scraper", score: 0.49, inDocs: false },
  { q: "Weather forecast tomorrow", score: 0.2, inDocs: false },
  { q: "What is 55 × 88?", score: 0.17, inDocs: false },
];

function GroundedSection() {
  return (
    <Section
      id="grounded"
      title="It only answers from your documents"
      lede={
        <>
          The worst thing a document assistant can do is answer confidently from
          general knowledge and cite your file anyway. {__APP_NAME__} answers from
          what you uploaded or says it can’t find it. It never fills the gap with
          a plausible guess.
        </>
      }
    >
      <div className="grid items-start gap-10 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <figure className="glass rounded-xl p-6">
          <figcaption className="text-sm text-muted">
            Relevance scores from a real test on a set of software-engineering notes.
            Below the line, the question is refused without calling a model.
          </figcaption>
          <ul className="mt-6 space-y-4">
            {SCORES.map(({ q, score, inDocs }) => {
              const passes = score >= GATE;
              return (
                <li key={q}>
                  <div className="flex items-baseline justify-between gap-3 text-sm">
                    <span className="text-strong">
                      {q}
                      <span className="ml-2 text-muted">
                        {inDocs ? "in the notes" : "not in the notes"}
                      </span>
                    </span>
                    <span
                      className={clsx(
                        "shrink-0 font-medium",
                        passes ? "text-positive" : "text-muted",
                      )}
                    >
                      {passes ? "Answered" : "Refused"}
                    </span>
                  </div>
                  <div className="relative mt-1.5 h-2 rounded-full bg-raised">
                    <div
                      className={clsx(
                        "h-full rounded-full",
                        passes ? "bg-accent shadow-glow" : "bg-muted/40",
                      )}
                      style={{ width: `${score * 100}%` }}
                    />
                    {/* The gate. */}
                    <span
                      aria-hidden
                      className="absolute -top-1 h-4 w-0.5 rounded bg-strong/60"
                      style={{ left: `${GATE * 100}%` }}
                    />
                  </div>
                  <span className="sr-only">Score {score.toFixed(2)}</span>
                </li>
              );
            })}
          </ul>
        </figure>

        <div className="space-y-6">
          <Point title="A relevance check before the model">
            Each question is first scored against your documents. On-topic
            questions scored 0.70 to 0.89 in this test and off-topic ones 0.11 to
            0.20, so a threshold of 0.5 separates them with room to spare. A
            refused question costs no tokens.
          </Point>
          <Point title="A strict rule for the model">
            The agent has no outside knowledge. It must search before it answers,
            every claim has to trace back to a passage it can cite, and when the
            documents don’t cover something it says so.
          </Point>
          <Point title="Safety checks on the way in and out">
            Questions are screened for prompt injection and pasted secrets.
            Answers are checked for groundedness, and personal data is redacted.
            If an answer is blocked or changed, you’re told why.
          </Point>
        </div>
      </div>
    </Section>
  );
}

function Point({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="border-l-2 border-accent/50 pl-4">
      <h3 className="font-semibold text-strong">{title}</h3>
      <p className="mt-1.5 leading-relaxed text-body">{children}</p>
    </div>
  );
}

// -- features ------------------------------------------------------------------

const MODES = [
  "Study", "Finance", "Legal", "Research", "Medical",
  "Code", "Business", "Writing", "Teaching", "Summarize",
];

function FeaturesSection() {
  return (
    <Section id="features" title="What you get">
      <dl className="grid gap-x-12 gap-y-10 md:grid-cols-2">
        <Item title="Shelves for each subject">
          Keep a maths textbook and a programming manual apart. Each shelf is its
          own knowledge base, so questions search one subject and graphs never mix.
        </Item>
        <Item title="Modes for the job at hand">
          Pick how the assistant approaches your documents. Each shelf remembers
          its mode.
          <span className="mt-3 flex flex-wrap gap-1.5">
            {MODES.map((m) => (
              <span
                key={m}
                className="rounded-full border border-border bg-surface px-2.5 py-0.5 text-xs text-body"
              >
                {m}
              </span>
            ))}
          </span>
        </Item>
        <Item title="Answers you can check">
          Every answer lists its sources with the passage it used, and while it
          works you can see which search it’s running.
        </Item>
        <Item title="Comparisons and big-picture questions">
          Ask it to compare several subjects side by side, or ask “what are the
          main themes?” and it reads cluster summaries instead of one passage.
        </Item>
        <Item title="Your choice of model">
          Switch between the available language models from the message box.
        </Item>
        <Item title="Private by default">
          Each account has a fully separate knowledge base. Nobody else’s
          questions can reach your documents.
        </Item>
      </dl>
    </Section>
  );
}

function Item({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <dt className="flex items-center gap-2.5 text-lg font-semibold text-strong">
        <span aria-hidden className="h-2 w-2 rounded-full bg-accent shadow-glow" />
        {title}
      </dt>
      <dd className="mt-2 pl-[18px] leading-relaxed text-body">{children}</dd>
    </div>
  );
}

// -- self-hosting ----------------------------------------------------------------

function SelfHostSection() {
  return (
    <Section
      id="self-host"
      title="Open source, and yours to run"
      lede="Run the whole stack with Docker: graph database, cache, API and web app. Use local open models through Ollama with no API keys, or cloud models from Anthropic, Google, OpenAI and others."
    >
      <pre className="glass overflow-x-auto rounded-xl p-5 font-mono text-sm leading-relaxed text-strong">
        <code>
          <span className="text-muted"># get the code and set your passwords and keys</span>
          {"\n"}git clone {__SOURCE_URL__}.git
          {"\n"}cd agentic-graph-rag && cp .env.example .env
          {"\n\n"}
          <span className="text-muted"># local models, no keys (or PROFILE=production)</span>
          {"\n"}make setup PROFILE=local
          {"\n"}make up
          {"\n"}docker compose exec api alembic upgrade head
        </code>
      </pre>
      <p className="mt-5 text-body">
        Then open http://localhost. The README covers configuration, models and
        production setup.{" "}
        <a
          href={__SOURCE_URL__}
          target="_blank"
          rel="noreferrer"
          className="font-medium text-accent underline-offset-2 hover:underline"
        >
          Read it on GitHub
        </a>
      </p>
    </Section>
  );
}

/** Lucide dropped brand icons, so the mark is drawn here. */
function GitHubMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" className={className} fill="currentColor" aria-hidden>
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

// -- the demo ------------------------------------------------------------------

const QUESTION = "Who founded the company that makes the Pallet Pilot?";

type NodeKind = "product" | "company" | "person" | "place" | "topic";

const NODES: { id: string; label: string; kind: NodeKind; x: number; y: number }[] = [
  { id: "pilot", label: "Pallet Pilot", kind: "product", x: 78, y: 196 },
  { id: "nordlicht", label: "Nordlicht Robotics", kind: "company", x: 222, y: 118 },
  { id: "ruiz", label: "Ana Ruiz", kind: "person", x: 368, y: 58 },
  { id: "robots", label: "Warehouse robots", kind: "topic", x: 86, y: 62 },
  { id: "hamburg", label: "Hamburg", kind: "place", x: 372, y: 196 },
  { id: "lift", label: "Lift Assist", kind: "product", x: 214, y: 236 },
];

// Background edges: the rest of the graph the agent does not need to walk.
const EDGES: [string, string][] = [
  ["robots", "pilot"],
  ["robots", "nordlicht"],
  ["nordlicht", "hamburg"],
  ["nordlicht", "lift"],
  ["ruiz", "hamburg"],
];

// The answer's path, in the order it is walked.
const PATH: { from: string; to: string; label: string }[] = [
  { from: "pilot", to: "nordlicht", label: "made by" },
  { from: "nordlicht", to: "ruiz", label: "founded by" },
];

const node = (id: string) => NODES.find((n) => n.id === id)!;

// Phases of one loop. Each number is how long the phase lasts in ms.
const STEPS = [
  { name: "typing", ms: QUESTION.length * 38 + 300 },
  { name: "start", ms: 700 },
  { name: "hop1", ms: 900 },
  { name: "hop2", ms: 900 },
  { name: "answer", ms: 4200 },
] as const;

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(
    () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
  );
  useEffect(() => {
    const media = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!media) return;
    const on = () => setReduced(media.matches);
    media.addEventListener("change", on);
    return () => media.removeEventListener("change", on);
  }, []);
  return reduced;
}

function GraphDemo() {
  const reduced = usePrefersReducedMotion();
  const [step, setStep] = useState(0);
  const [typed, setTyped] = useState(0);

  // Advance through the phases, then start over.
  useEffect(() => {
    if (reduced) return;
    const timer = window.setTimeout(
      () => setStep((s) => (s + 1) % STEPS.length),
      STEPS[step].ms,
    );
    return () => window.clearTimeout(timer);
  }, [step, reduced]);

  // Type the question during the first phase.
  useEffect(() => {
    if (reduced) return;
    if (step !== 0) return;
    setTyped(0);
    const timer = window.setInterval(
      () => setTyped((t) => (t >= QUESTION.length ? t : t + 1)),
      38,
    );
    return () => window.clearInterval(timer);
  }, [step, reduced]);

  // Reduced motion shows the finished state and holds it.
  const at = reduced ? STEPS.length - 1 : step;
  const shown = reduced || at > 0 ? QUESTION : QUESTION.slice(0, typed);
  const lit = new Set<string>();
  if (at >= 1) lit.add("pilot");
  if (at >= 2) lit.add("nordlicht");
  if (at >= 3) lit.add("ruiz");
  const hops = Math.max(0, at - 1); // path edges drawn so far

  const status =
    at === 0 ? "" : at === 1 ? "Finding “Pallet Pilot”" : at < 4 ? "Following the graph" : "";

  return (
    <figure
      className="glass relative rounded-2xl p-5 sm:p-6"
      aria-label={`Example: asked “${QUESTION}”, the answer follows two links in the knowledge graph.`}
    >
      {/* The question, as it would be typed into the composer. */}
      <div className="flex items-center gap-3 rounded-xl border border-border bg-surface px-4 py-3">
        <p className="min-h-[1.5rem] flex-1 text-sm text-strong sm:text-base" aria-hidden>
          {shown}
          {at === 0 && !reduced && (
            <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-pulse bg-accent" />
          )}
        </p>
      </div>

      <svg viewBox="0 0 440 270" className="mt-4 w-full" role="img" aria-hidden>
        {EDGES.map(([a, b]) => (
          <line
            key={`${a}-${b}`}
            x1={node(a).x}
            y1={node(a).y}
            x2={node(b).x}
            y2={node(b).y}
            className="stroke-muted/30"
            strokeWidth={1}
          />
        ))}

        {PATH.map((edge, i) => {
          const a = node(edge.from);
          const b = node(edge.to);
          const length = Math.hypot(b.x - a.x, b.y - a.y);
          const drawn = i < hops;
          return (
            <g key={edge.label}>
              <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="stroke-muted/30" strokeWidth={1} />
              <line
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                className="stroke-accent"
                strokeWidth={2.5}
                strokeLinecap="round"
                strokeDasharray={length}
                strokeDashoffset={drawn ? 0 : length}
                style={{
                  transition: reduced ? "none" : "stroke-dashoffset 800ms ease-in-out",
                  filter: "drop-shadow(0 0 6px rgb(var(--accent) / 0.7))",
                }}
              />
              <text
                x={(a.x + b.x) / 2 + 8}
                y={(a.y + b.y) / 2 - 8}
                className={clsx(
                  "text-[11px] font-medium transition-opacity duration-500",
                  drawn ? "fill-accent opacity-100" : "fill-muted opacity-0",
                )}
              >
                {edge.label}
              </text>
            </g>
          );
        })}

        {NODES.map((n) => {
          const on = lit.has(n.id);
          return (
            <g key={n.id}>
              {on && (
                <circle
                  cx={n.x}
                  cy={n.y}
                  r={18}
                  className="fill-accent/15"
                  style={{ filter: "blur(4px)" }}
                />
              )}
              <circle
                cx={n.x}
                cy={n.y}
                r={on ? 7 : 5}
                className={clsx(
                  "transition-all duration-500",
                  on ? "fill-accent" : "fill-surface stroke-muted/60",
                )}
                strokeWidth={1.2}
              />
              <text
                x={n.x}
                y={n.y + (n.y > 150 ? 24 : -14)}
                textAnchor="middle"
                className={clsx(
                  "text-[12px] transition-colors duration-500",
                  on ? "fill-strong font-semibold" : "fill-muted",
                )}
              >
                {n.label}
              </text>
            </g>
          );
        })}
      </svg>

      <div className="min-h-[4.5rem]" aria-hidden>
        {status && (
          <p className="flex items-center gap-2 text-sm text-muted">
            <span className="h-2 w-2 animate-pulse rounded-full bg-accent" />
            {status}…
          </p>
        )}
        {at === 4 && (
        <div className={clsx(!reduced && "animate-slide-up")}>
          <p className="text-base leading-relaxed text-strong">
            <strong>Ana Ruiz</strong> founded Nordlicht Robotics, the company that makes
            the Pallet Pilot.
            <Cite n={1} /> <Cite n={2} />
          </p>
          <p className="mt-1.5 text-xs text-muted">
            From “product-catalog.pdf” and “company-history.docx”
          </p>
        </div>
        )}
      </div>

      <figcaption className="sr-only">
        The answer: Ana Ruiz founded Nordlicht Robotics, the company that makes the
        Pallet Pilot.
      </figcaption>
    </figure>
  );
}

function Cite({ n }: { n: number }) {
  return (
    <span className="ml-1 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-accent/15 px-1.5 align-[1px] text-[11px] font-semibold text-accent">
      {n}
    </span>
  );
}
