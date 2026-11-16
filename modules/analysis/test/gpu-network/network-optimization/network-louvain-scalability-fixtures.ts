// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** CSR rows shared by the Louvain scalability node and headless specs. */
export type NetworkLouvainScalabilityFixture = {
  nodeCount: number;
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
  resolution: number;
};

/**
 * Four-vertex weighted components spread across four ID ranges. Strong pairs stabilize before
 * the quotient merges them: gamma = componentCount keeps the moving problem identical at scale.
 * Unused low IDs (at most three) are isolates. Boundary variants add duplicate arcs and loops.
 */
export function createNetworkLouvainQuotientFixture(
  nodeCount: number,
  options: {duplicateArcs?: boolean; selfLoops?: boolean; directed?: boolean} = {}
): NetworkLouvainScalabilityFixture {
  const componentCount = Math.floor(nodeCount / 4);
  const isolateCount = nodeCount % 4;
  const slotsPerVertex = (options.duplicateArcs ? 6 : 3) + (options.selfLoops ? 1 : 0);
  const offsets = new Uint32Array(nodeCount + 1);
  const neighbors = new Uint32Array(4 * componentCount * slotsPerVertex);
  const weights = new Float32Array(neighbors.length);
  let slot = 0;
  for (let source = 0; source < nodeCount; source++) {
    offsets[source] = slot;
    if (source < isolateCount) continue;
    const localSource = Math.floor((source - isolateCount) / componentCount);
    const component = (source - isolateCount) % componentCount;
    for (let localDestination = 0; localDestination < 4; localDestination++) {
      if (localDestination === localSource) continue;
      const destination = isolateCount + localDestination * componentCount + component;
      const weight =
        (localDestination === (localSource ^ 1) ? 2 : 1.25) *
        (options.directed && localSource < localDestination ? 1.125 : 1);
      neighbors[slot] = destination;
      weights[slot++] = options.duplicateArcs ? weight * 0.25 : weight;
      if (options.duplicateArcs) {
        neighbors[slot] = destination;
        weights[slot++] = weight * 0.75;
      }
    }
    if (options.selfLoops) {
      neighbors[slot] = source;
      // Undirected loops are already doubled, as required by the input contract.
      weights[slot++] = options.directed ? 0.125 : 0.25;
    }
  }
  offsets[nodeCount] = slot;
  return {nodeCount, offsets, neighbors, weights, resolution: componentCount};
}

/**
 * 384 duplicates per directed pair, alternating very small and large exact binary weights.
 * Each arc still totals exactly 2 or 1.25. Symmetry and moving aggregate 768 weak records per
 * pair, crossing the 256-thread segment reducer's strided-partial-sum boundary.
 */
export function createNetworkLouvainMixedDuplicateFixture(): NetworkLouvainScalabilityFixture {
  const nodeCount = 4;
  const duplicatesPerPair = 384;
  const offsets = new Uint32Array(nodeCount + 1);
  const neighbors = new Uint32Array(nodeCount * 3 * duplicatesPerPair);
  const weights = new Float32Array(neighbors.length);
  let slot = 0;
  for (let source = 0; source < nodeCount; source++) {
    offsets[source] = slot;
    for (let destination = 0; destination < nodeCount; destination++) {
      if (source === destination) continue;
      const pairWeight = destination === (source ^ 1) ? 2 : 1.25;
      const smallWeight = 2 ** -20;
      const largeWeight = (pairWeight - 256 * smallWeight) / 128;
      for (let duplicate = 0; duplicate < duplicatesPerPair; duplicate++) {
        neighbors[slot] = destination;
        weights[slot++] = duplicate % 3 === 0 ? largeWeight : smallWeight;
      }
    }
  }
  offsets[nodeCount] = slot;
  return {nodeCount, offsets, neighbors, weights, resolution: 1};
}
