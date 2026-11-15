// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CPUNetworkLouvainOptions,
  CPUNetworkLouvainResult
} from './cpu-network-louvain-operation';

/** @internal */
export type NetworkLouvainCPUCSR = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
};

/** Float64 sparse twin of parallel proposals, community reservations and quotient rounds.
 * Keeps no GPU pair-key limit; the public function validates shape/seed/trace budgets.
 * @internal
 */
export function computeSparseNetworkLouvain(
  topology: NetworkLouvainCPUCSR,
  parameters: Float32Array,
  options: CPUNetworkLouvainOptions
): CPUNetworkLouvainResult {
  const {nodeCount} = options;
  const {offsets, neighbors, weights} = topology;
  if (offsets.length !== nodeCount + 1)
    throw new Error('topology.offsets length must equal nodeCount + 1');
  if (neighbors.length !== weights.length)
    throw new Error('topology.weights length must equal topology.neighbors length');
  if (parameters.length !== 2)
    throw new Error('parameters must contain resolution and minimum improvement');
  const communities = Uint32Array.from({length: nodeCount}, (_, node) => node);
  const maximumLevels = options.maximumLevels ?? 8;
  const maximumIterations = options.maximumIterations ?? 16;
  const objectiveTrace = new Float32Array(1 + maximumLevels * maximumIterations).fill(NaN);
  const summary = new Uint32Array(4);
  const result = {communities, objectiveTrace, summary};
  const [resolution, tolerance] = parameters;
  // Malformed CSR fails closed on GPU; retain the same identity/invalid output here.
  if (offsets[0] !== 0) return result;
  for (let node = 0; node < nodeCount; node++)
    if (offsets[node] > offsets[node + 1] || offsets[node + 1] > neighbors.length) return result;
  if (
    !Number.isFinite(resolution) ||
    resolution < 0 ||
    !Number.isFinite(tolerance) ||
    tolerance < 0
  )
    return result;
  let degreeOut = new Float64Array(nodeCount);
  let degreeIn = new Float64Array(nodeCount);
  let edges = Array.from({length: nodeCount}, () => new Map<number, number>());
  const directedEdges = Array.from({length: nodeCount}, () => new Map<number, number>());
  let total = 0;
  for (let source = 0; source < nodeCount; source++) {
    for (let slot = offsets[source]; slot < offsets[source + 1]; slot++) {
      const target = neighbors[slot];
      if (target >= nodeCount) continue;
      const weight = weights[slot];
      if (!Number.isFinite(weight) || weight < 0) return result;
      if (weight === 0) continue;
      directedEdges[source].set(target, (directedEdges[source].get(target) ?? 0) + weight);
      degreeOut[source] += weight;
      degreeIn[target] += weight;
      total += weight;
      edges[source].set(target, (edges[source].get(target) ?? 0) + weight);
      edges[target].set(source, (edges[target].get(source) ?? 0) + weight);
    }
  }
  if (!options.directed) {
    for (let source = 0; source < nodeCount; source++) {
      for (const [target, weight] of directedEdges[source]) {
        const reverseWeight = directedEdges[target].get(source) ?? 0;
        if (
          Math.abs(weight - reverseWeight) >
          1e-6 * Math.max(Math.abs(weight), Math.abs(reverseWeight))
        )
          return result;
      }
    }
  }
  summary[0] = 1;
  if (total === 0) {
    summary[1] = 1;
    objectiveTrace[0] = 0;
    return result;
  }
  for (let node = 0; node < nodeCount; node++) {
    degreeOut[node] /= total;
    degreeIn[node] /= total;
    for (const [target, weight] of edges[node]) edges[node].set(target, weight / total);
  }
  const labels = Uint32Array.from(communities);
  const groups = Uint32Array.from(communities);
  const collectTotals = () => {
    const outgoing = new Float64Array(nodeCount);
    const incoming = new Float64Array(nodeCount);
    for (let node = 0; node < nodeCount; node++) {
      outgoing[labels[node]] += degreeOut[node];
      incoming[labels[node]] += degreeIn[node];
    }
    return {outgoing, incoming};
  };
  let totals = collectTotals();
  const objective = () => {
    let internal = 0;
    let penalty = 0;
    for (let source = 0; source < nodeCount; source++) {
      for (const [target, weight] of edges[source])
        if (labels[source] === labels[target]) internal += weight / 2;
      penalty += resolution * totals.outgoing[source] * totals.incoming[source];
    }
    return internal - penalty;
  };
  objectiveTrace[0] = objective();
  let rounds = 0;
  for (let level = 0; level < maximumLevels; level++) {
    let levelChanged = false;
    for (let round = 0; round < maximumIterations; round++) {
      const proposals = new Uint32Array(nodeCount).fill(0xffffffff);
      const reservations = new Uint32Array(nodeCount).fill(0xffffffff);
      for (let source = 0; source < nodeCount; source++) {
        const original = labels[source];
        const connections = new Map<number, number>();
        for (const [target, weight] of edges[source]) {
          const community = labels[target];
          connections.set(
            community,
            (connections.get(community) ?? 0) + (source === target ? 0 : weight)
          );
        }
        const originalConnection = connections.get(original) ?? 0;
        let bestGain = 0;
        let best = 0xffffffff;
        for (const [target, weight] of connections) {
          if (target === original) continue;
          const penalty =
            degreeOut[source] *
              (totals.incoming[target] - totals.incoming[original] + degreeIn[source]) +
            degreeIn[source] *
              (totals.outgoing[target] - totals.outgoing[original] + degreeOut[source]);
          const gain = weight - originalConnection - resolution * penalty;
          if (gain > tolerance && (gain > bestGain || (gain === bestGain && target < best))) {
            bestGain = gain;
            best = target;
          }
        }
        proposals[source] = best;
        if (best !== 0xffffffff) {
          const priority = (source ^ (options.seed ?? 0)) >>> 0;
          reservations[original] = Math.min(reservations[original], priority);
          reservations[best] = Math.min(reservations[best], priority);
        }
      }
      const accepted = Uint32Array.from(labels);
      let changed = false;
      for (let source = 0; source < nodeCount; source++) {
        const target = proposals[source];
        const original = labels[source];
        const priority = (source ^ (options.seed ?? 0)) >>> 0;
        if (
          target !== 0xffffffff &&
          reservations[original] === priority &&
          reservations[target] === priority
        ) {
          accepted[source] = target;
          changed = true;
        }
      }
      labels.set(accepted);
      for (let node = 0; node < nodeCount; node++) communities[node] = labels[groups[node]];
      totals = collectTotals();
      objectiveTrace[++rounds] = objective();
      summary[2] = level + 1;
      summary[3] = rounds;
      if (!changed) {
        if (!levelChanged) {
          summary[1] = 1;
          return result;
        }
        break;
      }
      levelChanged = true;
      if (round + 1 === maximumIterations) return result;
    }
    if (level + 1 === maximumLevels) return result;
    const quotient = Array.from({length: nodeCount}, () => new Map<number, number>());
    for (let source = 0; source < nodeCount; source++) {
      for (const [target, weight] of edges[source]) {
        const coarseSource = labels[source];
        const coarseTarget = labels[target];
        quotient[coarseSource].set(
          coarseTarget,
          (quotient[coarseSource].get(coarseTarget) ?? 0) + weight
        );
      }
    }
    edges = quotient;
    degreeOut = totals.outgoing;
    degreeIn = totals.incoming;
    groups.set(communities);
    for (let node = 0; node < nodeCount; node++) labels[node] = node;
  }
  return result;
}
