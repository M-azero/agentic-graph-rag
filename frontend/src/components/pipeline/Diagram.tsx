// The diagram. Hand-rolled SVG rather than a graph library: the layout is
// fixed and hand-placed (see lib/pipeline.ts), so all a library would add is
// ~100 kB to a bundle this app has kept deliberately small.
//
// Colour comes entirely from the app's semantic tokens, so light and dark are
// one set of class names — and `currentColor` on the edges means a stroke
// follows whatever text colour its group carries.

import clsx from "clsx";

import type { PipelineGraph } from "../../api";
import {
  extent,
  headline,
  LANE_LABELS,
  NODE_H,
  NODE_W,
  place,
  type NodeRun,
  type NodeState,
  type PlacedNode,
} from "../../lib/pipeline";

const STATE_CLASS: Record<NodeState, string> = {
  // Dashed: this block has not been reached yet. The distinction from `skipped`
  // matters — one is "not yet", the other is "ran and had nothing to do".
  idle: "fill-surface stroke-border [stroke-dasharray:4_3]",
  running: "fill-accent-soft stroke-accent",
  ok: "fill-surface stroke-positive",
  error: "fill-surface stroke-danger",
  skipped: "fill-raised stroke-border",
};

const LABEL_CLASS: Record<NodeState, string> = {
  idle: "fill-muted",
  running: "fill-strong",
  ok: "fill-strong",
  error: "fill-danger",
  skipped: "fill-muted",
};

interface Props {
  graph: PipelineGraph;
  runs: Record<string, NodeRun>;
  selected: string | null;
  onSelect: (name: string) => void;
}

export function Diagram({ graph, runs, selected, onSelect }: Props) {
  const nodes = place(graph);
  const byName = new Map(nodes.map((n) => [n.name, n]));
  const { width, height } = extent(nodes);
  const lanes = LANE_LABELS[graph.pipeline] ?? [];

  return (
    // The wide diagram scrolls inside its own box; the page never scrolls
    // sideways. `max-h` keeps a tall pipeline from pushing the panels below it
    // off-screen — it scrolls vertically inside the box instead.
    <div className="max-h-[70vh] overflow-auto rounded-lg border border-border bg-canvas p-4">
      <svg
        viewBox={`-8 -28 ${width + 16} ${height + 36}`}
        // BOTH dimensions as attributes, 1:1 with the viewBox. Setting only
        // `width` and leaving height to CSS `auto` lets the box collapse while
        // the content still draws full size — the diagram then spills out and
        // overlaps whatever is below it, and re-renders to a different height.
        // An explicit height makes the box exactly bound the drawing.
        width={width + 16}
        height={height + 36}
        style={{ display: "block", maxWidth: "none" }}
        role="img"
        aria-label={`${graph.pipeline} pipeline`}
      >
        <defs>
          <marker
            id="arrow"
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M0,1 L7,4 L0,7 z" className="fill-border" />
          </marker>
        </defs>

        {lanes.map((lane) => (
          <text
            key={lane.row}
            x={-4}
            y={lane.row * 92 - 8}
            className="fill-muted text-[10px] uppercase tracking-wide"
          >
            {lane.label}
          </text>
        ))}

        {graph.edges.map((edge) => {
          const from = byName.get(edge.from);
          const to = byName.get(edge.to);
          if (!from || !to) return null;
          return (
            <path
              key={`${edge.from}->${edge.to}`}
              d={edgePath(from, to)}
              className={clsx(
                "fill-none stroke-border",
                // An edge into a block that ran is drawn solid; the rest of the
                // graph stays visible but recedes.
                runs[edge.to] ? "opacity-100" : "opacity-40",
              )}
              strokeWidth={1.5}
              markerEnd="url(#arrow)"
            />
          );
        })}

        {nodes.map((node) => (
          <Node
            key={node.name}
            node={node}
            run={runs[node.name]}
            selected={selected === node.name}
            onSelect={onSelect}
          />
        ))}
      </svg>
    </div>
  );
}

function Node({
  node,
  run,
  selected,
  onSelect,
}: {
  node: PlacedNode;
  run: NodeRun | undefined;
  selected: boolean;
  onSelect: (name: string) => void;
}) {
  const state: NodeState = run?.state ?? "idle";
  const count = run?.steps.length ?? 0;
  const summary = run ? headline(node.name, run) : "";

  return (
    <g
      transform={`translate(${node.pos.x}, ${node.pos.y})`}
      onClick={() => onSelect(node.name)}
      className="cursor-pointer"
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect(node.name);
        }
      }}
      aria-label={`${node.label}: ${state}`}
    >
      {/* A block that runs many times is drawn as a stack, so "this happened
          once" and "this happened nine times" are different at a glance. */}
      {node.repeats && count > 1 && (
        <rect
          x={4}
          y={4}
          width={NODE_W}
          height={NODE_H}
          rx={8}
          className={clsx(STATE_CLASS[state], "opacity-40")}
          strokeWidth={1.5}
        />
      )}
      <rect
        width={NODE_W}
        height={NODE_H}
        rx={8}
        className={STATE_CLASS[state]}
        strokeWidth={selected ? 2.5 : 1.5}
      />
      {state === "running" && (
        <rect
          width={NODE_W}
          height={NODE_H}
          rx={8}
          className="fill-none stroke-accent"
          strokeWidth={2}
        >
          <animate
            attributeName="opacity"
            values="1;0.25;1"
            dur="1.2s"
            repeatCount="indefinite"
          />
        </rect>
      )}

      <text x={12} y={22} className={clsx(LABEL_CLASS[state], "text-[12px] font-medium")}>
        {node.label}
      </text>
      <text x={12} y={39} className="fill-muted text-[10.5px]">
        {summary || node.kind}
      </text>
      {count > 1 && (
        <>
          <circle cx={NODE_W - 16} cy={18} r={9} className="fill-raised stroke-border" />
          <text
            x={NODE_W - 16}
            y={21.5}
            textAnchor="middle"
            className="fill-body text-[10px] font-semibold"
          >
            {count}
          </text>
        </>
      )}
      {run && (
        <text x={NODE_W - 10} y={47} textAnchor="end" className="fill-muted text-[10px]">
          {run.totalMs >= 1 ? `${Math.round(run.totalMs)} ms` : ""}
        </text>
      )}
    </g>
  );
}

/** A cubic between two blocks, leaving the right edge and entering the left.
 *
 *  Edges that run backwards (rerank returns to the agent; widen returns to
 *  research) bow underneath instead, or they would be drawn straight through
 *  the blocks between them. */
function edgePath(from: PlacedNode, to: PlacedNode): string {
  const x1 = from.pos.x + NODE_W;
  const y1 = from.pos.y + NODE_H / 2;
  const x2 = to.pos.x;
  const y2 = to.pos.y + NODE_H / 2;

  if (x2 < x1) {
    const dip = Math.max(y1, y2) + NODE_H;
    return `M ${from.pos.x + NODE_W / 2} ${from.pos.y + NODE_H} C ${
      from.pos.x + NODE_W / 2
    } ${dip}, ${to.pos.x + NODE_W / 2} ${dip}, ${to.pos.x + NODE_W / 2} ${
      to.pos.y + NODE_H
    }`;
  }
  const mid = (x1 + x2) / 2;
  return `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`;
}
