// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A seeded community graph for the adjacency-matrix runs: points in Gaussian clusters joined to
// their nearest neighbors. Node numbers follow the random point order, so the raw matrix is noise
// until nodes are grouped.

import type {RunBuilder} from '../../../define-run';
import {
  createClusteredPoints,
  createKnnWeights,
  type CsrWeights,
  type World
} from '../../../inputs';

export type CommunityGraph = {
  nodeCount: number;
  positions: Float32Array;
  /** Cluster of each node, the community the order operation groups by. */
  groups: Uint32Array;
  /** Out-degree of each node. */
  degrees: Uint32Array;
  csr: CsrWeights;
};

export function createCommunityGraph(
  random: () => number,
  world: World,
  clusterCount: number,
  nodeCount = 256
): CommunityGraph {
  const points = createClusteredPoints(random, world, nodeCount, clusterCount, 0.04);
  const csr = createKnnWeights(points.positions, 5);
  const degrees = Uint32Array.from(
    {length: nodeCount},
    (_, node) => csr.offsets[node + 1] - csr.offsets[node]
  );
  return {nodeCount, positions: points.positions, groups: points.labels, degrees, csr};
}

/** Uploads the graph's CSR as a network adjacency. */
export function uploadAdjacency(builder: RunBuilder, graph: CommunityGraph) {
  return {
    offsets: builder.upload('offsets', 'uint32', graph.csr.offsets).view,
    neighbors: builder.upload('neighbors', 'uint32', graph.csr.neighbors).view,
    weights: builder.upload('weights', 'float32', graph.csr.weights).view
  };
}
