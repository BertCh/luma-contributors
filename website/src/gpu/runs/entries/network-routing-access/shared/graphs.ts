// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COMMUTE_FLOWS, TRACT_LOCATIONS} from '../../../../data/commute-flows';
import {buildRows, type CsrWeights} from '../../../inputs';
import type {RunBuilder} from '../../../define-run';

export type CommunityGraph = {
  nodeCount: number;
  groups: Uint32Array;
  degrees: Uint32Array;
  csr: CsrWeights;
};

/** Directed home-to-work links, with optional ordering by north-to-south bands of tracts. */
export function createCommunityGraph(bandCount: number): CommunityGraph {
  const nodeCount = TRACT_LOCATIONS.length;
  const ordered = Array.from({length: nodeCount}, (_, node) => node).sort(
    (first, second) => TRACT_LOCATIONS[second][1] - TRACT_LOCATIONS[first][1]
  );
  const groups = new Uint32Array(nodeCount);
  ordered.forEach((node, rank) => {
    groups[node] = Math.floor((rank * bandCount) / nodeCount);
  });
  const rows = Array.from({length: nodeCount}, (): [number, number][] => []);
  for (const [home, work] of COMMUTE_FLOWS) rows[home].push([work, 1]);
  const csr = buildRows(nodeCount, row => rows[row]);
  return {nodeCount, groups, csr, degrees: Uint32Array.from(rows, row => row.length)};
}

/** Uploads the graph's CSR as a network adjacency. */
export function uploadAdjacency(builder: RunBuilder, graph: CommunityGraph) {
  return {
    offsets: builder.upload('offsets', 'uint32', graph.csr.offsets).view,
    neighbors: builder.upload('neighbors', 'uint32', graph.csr.neighbors).view,
    weights: builder.upload('weights', 'float32', graph.csr.weights).view
  };
}

/** Scales and centers packed (x, y) positions so their extent fills the drawing world. */
export function fitPositions(
  positions: Float32Array,
  world: {width: number; height: number},
  margin = 0.07
): Float32Array {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let index = 0; index < positions.length; index += 2) {
    minimumX = Math.min(minimumX, positions[index]);
    maximumX = Math.max(maximumX, positions[index]);
    minimumY = Math.min(minimumY, positions[index + 1]);
    maximumY = Math.max(maximumY, positions[index + 1]);
  }
  const scale = Math.min(
    (world.width * (1 - 2 * margin)) / Math.max(maximumX - minimumX, 1e-9),
    (world.height * (1 - 2 * margin)) / Math.max(maximumY - minimumY, 1e-9)
  );
  const offsetX = world.width / 2 - ((minimumX + maximumX) / 2) * scale;
  const offsetY = world.height / 2 - ((minimumY + maximumY) / 2) * scale;
  return Float32Array.from(positions, (value, index) =>
    index % 2 === 0 ? value * scale + offsetX : value * scale + offsetY
  );
}

/**
 * Symmetric tract graph with community structure: the commuter links among tracts of one band, plus
 * every sixth link that crosses bands, each in both directions. Grouped order therefore draws dense
 * blocks on the diagonal over a sparse background.
 */
export function createBandedGraph(bandCount: number): CommunityGraph {
  const base = createCommunityGraph(bandCount);
  const rows = Array.from({length: base.nodeCount}, () => new Set<number>());
  COMMUTE_FLOWS.forEach(([home, work], index) => {
    if (home === work) return;
    if (base.groups[home] !== base.groups[work] && index % 6 !== 0) return;
    rows[home].add(work);
    rows[work].add(home);
  });
  const csr = buildRows(base.nodeCount, row =>
    Array.from(rows[row], (work): [number, number] => [work, 1])
  );
  return {...base, csr, degrees: Uint32Array.from(rows, row => row.size)};
}
