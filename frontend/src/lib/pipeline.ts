// Where each block sits on the canvas, and how a run's steps map onto it.
//
// The server owns *what* the blocks are and how they connect (built from
// `graphrag/trace/steps.py`, so the picture cannot drift from the code that
// emits it). This file owns only *where they sit* — a hand-placed layout beats
// an auto-layout here, because the shape of this pipeline is the thing being
// explained: the three retrieval legs have to read as parallel, and the review
// loop has to read as a loop.
//
// A node the server sends that has no position here still renders: it drops
// into the overflow lane rather than vanishing, which is what keeps a newly
// added block visible before anyone touches this file.

import type { PipelineGraph, TraceStep } from "../api";

export interface Point {
  x: number;
  y: number;
}

export const NODE_W = 168;
export const NODE_H = 56;

/** Column/row coordinates, multiplied up at render time. */
const GRID_X = 216;
const GRID_Y = 92;

const at = (col: number, row: number): Point => ({
  x: col * GRID_X,
  y: row * GRID_Y,
});

// Rows are lanes: 0 is the spine the request travels along, and anything above
// or below it is work that happens off to one side.
const LAYOUT: Record<string, Point> = {
  // -- query: the spine, left to right --------------------------------------
  "guard.input": at(0, 1),
  "gate.probe": at(1, 1),
  "agent.run": at(2, 1),
  "tool.call": at(3, 1),
  // the three legs, fanned out and run at the same time
  "retrieval.vector": at(4, 0),
  "retrieval.graph": at(4, 1),
  "retrieval.keyword": at(4, 2),
  "retrieval.fuse": at(5, 1),
  "retrieval.rerank": at(6, 1),
  // ...and back to the agent, which is why these two sit below the spine
  "check.citations": at(3, 3),
  "guard.output": at(4, 3),
  billing: at(5, 3),

  // -- review: the loop ------------------------------------------------------
  "review.research": at(0, 1),
  "review.check": at(1, 1),
  "review.widen": at(1, 0), // the cycle: check -> widen -> research
  "review.revise": at(2, 2),
  "review.finalize": at(3, 1),

  // -- ingest ----------------------------------------------------------------
  "ingest.load": at(0, 1),
  "ingest.chunk": at(1, 1),
  // the two halves of an ingest: vectors above, knowledge graph below
  "ingest.embed": at(2, 0),
  "ingest.vector_upsert": at(3, 0),
  "ingest.graph_chunks": at(4, 0),
  "ingest.link_sequence": at(5, 0),
  "ingest.extract": at(2, 2),
  "ingest.graph_write": at(3, 2),
  "ingest.resolve_entities": at(5, 2),
  "ingest.communities": at(6, 1),
};

/** Blocks whose two halves are worth separating visually. */
export const LANE_LABELS: Record<string, { row: number; label: string }[]> = {
  query: [
    { row: 0, label: "retrieval, in parallel" },
    { row: 3, label: "after the answer" },
  ],
  ingest: [
    { row: 0, label: "vector index" },
    { row: 2, label: "knowledge graph" },
  ],
};

export function positionOf(name: string, fallbackIndex: number): Point {
  return LAYOUT[name] ?? at(fallbackIndex % 7, 5); // the overflow lane
}

export interface PlacedNode {
  name: string;
  label: string;
  kind: string;
  source: string;
  summary: string;
  repeats: boolean;
  optional: boolean;
  pos: Point;
  placed: boolean;
}

export function place(graph: PipelineGraph): PlacedNode[] {
  return graph.nodes.map((n, i) => ({
    ...n,
    pos: positionOf(n.name, i),
    placed: n.name in LAYOUT,
  }));
}

export function extent(nodes: PlacedNode[]): { width: number; height: number } {
  const width = Math.max(...nodes.map((n) => n.pos.x)) + NODE_W + 24;
  const height = Math.max(...nodes.map((n) => n.pos.y)) + NODE_H + 24;
  return { width, height };
}

// -- run state ----------------------------------------------------------------

export type NodeState = "idle" | "running" | "ok" | "error" | "skipped";

export interface NodeRun {
  state: NodeState;
  /** Every step recorded against this block. A block that repeats — a tool
   *  call, a retrieval leg — has one entry per call. */
  steps: TraceStep[];
  totalMs: number;
}

/** Fold a run's steps onto the blocks of the diagram.
 *
 *  Several steps can share a block (the agent calls three tools; the retriever
 *  runs on each). The block shows the aggregate and the inspector shows them
 *  individually, which is the only way a repeated block can report both "ran 3
 *  times" and "the second one failed".
 */
export function runStateByNode(steps: TraceStep[]): Record<string, NodeRun> {
  const byNode: Record<string, NodeRun> = {};
  for (const step of steps) {
    const entry = (byNode[step.name] ??= { state: "idle", steps: [], totalMs: 0 });
    entry.steps.push(step);
    entry.totalMs += step.duration_ms ?? 0;
  }
  for (const entry of Object.values(byNode)) {
    entry.state = collapse(entry.steps);
  }
  return byNode;
}

/** One state for a block that ran several times.
 *
 *  Order matters and is not arbitrary: a failure is the thing worth surfacing
 *  even when four other calls succeeded, and "still running" outranks a
 *  finished sibling because the block is, in fact, still busy. `skipped` only
 *  wins when *every* call was skipped — one real call means the block ran.
 */
function collapse(steps: TraceStep[]): NodeState {
  if (steps.some((s) => s.status === "error")) return "error";
  if (steps.some((s) => s.status === "running")) return "running";
  if (steps.every((s) => s.status === "skipped")) return "skipped";
  return "ok";
}

/** A one-line summary for the face of a block: the number a reader wants
 *  without opening the inspector. */
export function headline(name: string, run: NodeRun): string {
  const last = run.steps[run.steps.length - 1];
  if (!last) return "";
  const out = last.output ?? {};
  const num = (key: string) => (typeof out[key] === "number" ? (out[key] as number) : null);

  const candidates = num("candidates");
  const results = num("results");
  const chunks = num("chunks");
  const score = num("top_score");

  if (name === "gate.probe" && score !== null) return `top ${score.toFixed(2)}`;
  if (name === "retrieval.fuse" && candidates !== null) return `${candidates} fused`;
  if (name === "retrieval.rerank" && results !== null) return `top ${results}`;
  if (name.startsWith("retrieval.") && results !== null) return `${results} hits`;
  if (name === "tool.call") return String(out.tool ?? last.input?.tool ?? "");
  if (name === "billing") {
    const i = num("input_tokens") ?? 0;
    const o = num("output_tokens") ?? 0;
    return `${i + o} tokens`;
  }
  if (chunks !== null) return `${chunks} chunks`;
  if (last.status === "skipped") return "skipped";
  return "";
}

export function formatMs(ms: number | null): string {
  if (ms === null) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
}
