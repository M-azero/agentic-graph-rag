import {
  forceCenter,
  forceLink,
  forceManyBody,
  forceSimulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import { useEffect, useMemo, useRef, useState } from "react";

import type { GraphSample } from "../api";

interface Node extends SimulationNodeDatum {
  key: string;
  name: string;
  type: string;
  degree: number;
}

const WIDTH = 640;
const HEIGHT = 340;

/** A force-directed view of a tenant's entity graph.
 *
 *  Hand-rolled SVG over d3-force rather than a graph library: this is one
 *  admin-only widget, and the libraries that draw it well cost more bundle
 *  than the rest of the admin area combined. The simulation runs to a fixed
 *  tick count up front rather than animating — this is a diagram, not a toy. */
export function GraphView({ sample }: { sample: GraphSample }) {
  const [, force] = useState(0);
  const nodesRef = useRef<Node[]>([]);
  const linksRef = useRef<SimulationLinkDatum<Node>[]>([]);

  const maxDegree = useMemo(
    () => Math.max(1, ...sample.nodes.map((n) => n.degree)),
    [sample],
  );

  useEffect(() => {
    const nodes: Node[] = sample.nodes.map((n) => ({ ...n }));
    const byKey = new Map(nodes.map((n) => [n.key, n]));
    const links = sample.edges
      .map((e) => ({ source: byKey.get(e.source), target: byKey.get(e.target) }))
      .filter((l): l is { source: Node; target: Node } => Boolean(l.source && l.target));

    const simulation = forceSimulation(nodes)
      .force("charge", forceManyBody().strength(-120))
      .force("link", forceLink(links).id((d) => (d as Node).key).distance(60))
      .force("center", forceCenter(WIDTH / 2, HEIGHT / 2))
      .stop();

    simulation.tick(220);
    nodesRef.current = nodes;
    linksRef.current = links;
    force((n) => n + 1);
    return () => {
      simulation.stop();
    };
  }, [sample]);

  if (!sample.nodes.length) {
    return (
      <p className="py-12 text-center text-sm text-muted">
        No entities extracted yet.
      </p>
    );
  }

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="w-full"
      role="img"
      aria-label={`Entity graph with ${sample.nodes.length} entities`}
    >
      <g>
        {linksRef.current.map((link, i) => {
          const s = link.source as Node;
          const t = link.target as Node;
          return (
            <line
              key={i}
              x1={s.x}
              y1={s.y}
              x2={t.x}
              y2={t.y}
              stroke="rgb(var(--border))"
              strokeWidth={1}
            />
          );
        })}
      </g>
      <g>
        {nodesRef.current.map((node) => {
          // Radius tracks degree, so hubs read as hubs at a glance.
          const r = 3 + (node.degree / maxDegree) * 7;
          return (
            <g key={node.key}>
              <circle
                cx={node.x}
                cy={node.y}
                r={r}
                fill="rgb(var(--accent))"
                fillOpacity={0.75}
              />
              <title>{`${node.name} (${node.type}) — ${node.degree} connections`}</title>
              {node.degree > maxDegree * 0.4 && (
                <text
                  x={(node.x ?? 0) + r + 3}
                  y={(node.y ?? 0) + 3}
                  fontSize={9}
                  fill="rgb(var(--text-muted))"
                >
                  {node.name.slice(0, 22)}
                </text>
              )}
            </g>
          );
        })}
      </g>
    </svg>
  );
}
