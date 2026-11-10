// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The recorded road network as an SVG map. Every node is drawn in one of four states the caller
// derives from the node's round stamp: dark (not settled), lit (final cost), front (joins the
// current round's queue for the last time) and pending (not final under a budget).

import {memo, useMemo} from 'react';
import {SEARCH} from './search-data';

export type NodeState = 'dark' | 'lit' | 'front' | 'pending';

const VIEW_HEIGHT = 500;
const VIEW_WIDTH = Math.round(VIEW_HEIGHT * SEARCH.aspectRatio);
const PADDING = 8;

function getPoint(node: number): [number, number] {
  return [
    PADDING + SEARCH.positions[node * 2] * (VIEW_WIDTH - 2 * PADDING),
    PADDING + SEARCH.positions[node * 2 + 1] * (VIEW_HEIGHT - 2 * PADDING)
  ];
}

const POINTS = Array.from({length: SEARCH.nodeCount}, (_value, node) => getPoint(node));

function formatPoint([x, y]: [number, number]): string {
  return `${x.toFixed(1)} ${y.toFixed(1)}`;
}

/** One path of every undirected edge accepted by `keep`. */
function getEdgePath(keep: (from: number, to: number) => boolean): string {
  const parts: string[] = [];
  for (let edge = 0; edge < SEARCH.edges.length; edge += 2) {
    const from = SEARCH.edges[edge];
    const to = SEARCH.edges[edge + 1];
    if (keep(from, to)) parts.push(`M${formatPoint(POINTS[from])}L${formatPoint(POINTS[to])}`);
  }
  return parts.join('');
}

const ALL_EDGES = getEdgePath(() => true);

function isSettled(state: NodeState): boolean {
  return state === 'lit' || state === 'front';
}

const Edges = memo(function Edges({litPath}: {litPath: string}) {
  return (
    <>
      <path className="iteration-map__edges" d={ALL_EDGES} />
      <path className="iteration-map__edges" data-state="lit" d={litPath} />
    </>
  );
});

export function RoadMap({
  getState,
  stateKey,
  label
}: {
  /** State of one node. */
  getState: (node: number) => NodeState;
  /** Changes whenever `getState` would return something different (memo key). */
  stateKey: string;
  label: string;
}) {
  const states = useMemo(
    () => Array.from({length: SEARCH.nodeCount}, (_value, node) => getState(node)),
    [stateKey]
  );
  const litPath = useMemo(
    () => getEdgePath((from, to) => isSettled(states[from]) && isSettled(states[to])),
    [states]
  );
  return (
    <svg
      className="iteration-map"
      viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
      role="img"
      aria-label={label}
    >
      <Edges litPath={litPath} />
      {POINTS.map(([x, y], node) =>
        states[node] === 'front' ? null : (
          <circle
            key={node}
            className="iteration-map__node"
            data-state={states[node]}
            cx={x.toFixed(1)}
            cy={y.toFixed(1)}
            r={states[node] === 'pending' ? 2.2 : 1.9}
          />
        )
      )}
      {POINTS.map(([x, y], node) =>
        states[node] === 'front' ? (
          <circle
            key={node}
            className="iteration-map__node"
            data-state="front"
            cx={x.toFixed(1)}
            cy={y.toFixed(1)}
            r={3.4}
          />
        ) : null
      )}
      {SEARCH.startNodes.map(node => (
        <g key={node} className="iteration-map__start">
          <circle cx={POINTS[node][0]} cy={POINTS[node][1]} r={7} />
          <text x={POINTS[node][0] + 10} y={POINTS[node][1] + 4}>
            start
          </text>
        </g>
      ))}
    </svg>
  );
}
