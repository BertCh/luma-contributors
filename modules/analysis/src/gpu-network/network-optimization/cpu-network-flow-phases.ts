// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** CPU phase state follows the contributor's sink labels, forward pushes, excess returns and
 * separate cut search. Within each discharge, residual arcs use input-edge-row order.
 * GPU atomic scheduling may choose another equal-cost branch; a phase never reuses returned
 * source excess for another forward push. Inputs have already been validated by the caller.
 * @internal */
export function computeNetworkFlowPhasesOnCPU(input: {
  nodeCount: number;
  edges: Uint32Array;
  capacities: Uint32Array;
  costs?: Uint32Array;
  source: number;
  sink: number;
  required: number;
  maximumPhases: number;
  maximumRounds: number;
  flow: Uint32Array;
  cut: Uint32Array;
}) {
  const {
    nodeCount,
    edges,
    capacities,
    costs,
    source,
    sink,
    required,
    maximumPhases,
    maximumRounds,
    flow,
    cut
  } = input;
  const outgoing = Array.from({length: nodeCount}, () => [] as number[]);
  for (let edge = 0; edge < capacities.length; edge++) {
    outgoing[edges[edge * 2]].push(edge * 2);
    outgoing[edges[edge * 2 + 1]].push(edge * 2 + 1);
  }
  const getNeighbor = (arc: number) => edges[(arc >> 1) * 2 + (arc & 1 ? 0 : 1)];
  const getResidual = (arc: number) =>
    arc & 1 ? flow[arc >> 1] : capacities[arc >> 1] - flow[arc >> 1];
  const getCost = (arc: number) => (arc & 1 ? -1n : 1n) * BigInt(costs?.[arc >> 1] ?? 1);
  const findLabels = (root: number, reverse: boolean) => {
    let distances = new Array<bigint | undefined>(nodeCount);
    let hops = new Float64Array(nodeCount).fill(Infinity);
    distances[root] = 0n;
    hops[root] = 0;
    let converged = false;
    for (let round = 0; round < maximumRounds; round++) {
      const nextDistances = distances.slice();
      const nextHops = hops.slice();
      let changed = false;
      for (let vertex = 0; vertex < nodeCount; vertex++) {
        for (const outgoingArc of outgoing[vertex]) {
          const arc = reverse ? outgoingArc : outgoingArc ^ 1;
          if (!getResidual(arc)) continue;
          const neighbor = getNeighbor(outgoingArc);
          const distance = distances[neighbor];
          if (distance === undefined) continue;
          const candidate = distance + (reverse && costs ? getCost(arc) : 1n);
          const candidateHops = hops[neighbor] + 1;
          const current = nextDistances[vertex];
          if (
            current === undefined ||
            candidate < current ||
            (candidate === current && candidateHops < nextHops[vertex])
          ) {
            nextDistances[vertex] = candidate;
            nextHops[vertex] = candidateHops;
            changed = true;
          }
        }
      }
      distances = nextDistances;
      hops = nextHops;
      if (!changed) {
        converged = true;
        break;
      }
    }
    return {distances, hops, converged};
  };
  let value = 0;
  let phaseCount = 0;
  let complete = false;
  for (;;) {
    const labels = findLabels(sink, true);
    if (value === required) {
      complete = true;
      break;
    }
    if (!labels.converged) break;
    if (labels.distances[source] === undefined) {
      complete = true;
      break;
    }
    if (phaseCount >= maximumPhases) break;
    phaseCount++;
    const previousFlow = flow.slice();
    const increments = new Uint32Array(flow.length);
    const excess = new Float64Array(nodeCount);
    excess[source] = required - value;
    const isAdmissible = (vertex: number, arc: number, returning: boolean) => {
      const neighbor = getNeighbor(arc);
      if (labels.distances[vertex] === undefined || labels.distances[neighbor] === undefined)
        return false;
      if (returning)
        return (
          labels.hops[neighbor] === labels.hops[vertex] + 1 &&
          (!costs || labels.distances[neighbor] === labels.distances[vertex]! - getCost(arc))
        );
      return (
        labels.hops[vertex] === labels.hops[neighbor] + 1 &&
        (!costs || labels.distances[vertex] === labels.distances[neighbor]! + getCost(arc))
      );
    };
    for (const returning of [false, true]) {
      for (let round = 0; round < maximumRounds; round++) {
        const availableAtStart = excess.slice();
        let changed = false;
        for (let vertex = 0; vertex < nodeCount; vertex++) {
          if (vertex === sink || (returning && vertex === source)) continue;
          let available = availableAtStart[vertex];
          for (const arc of outgoing[vertex]) {
            if (!available) break;
            if (!isAdmissible(vertex, arc, returning)) continue;
            const edge = arc >> 1;
            const increment = Math.min(available, returning ? increments[edge] : getResidual(arc));
            if (!increment) continue;
            const direction = arc & 1 ? -1 : 1;
            flow[edge] += direction * increment;
            increments[edge] += (returning ? -1 : 1) * increment;
            excess[getNeighbor(arc)] += increment;
            excess[vertex] -= increment;
            available -= increment;
            changed = true;
          }
        }
        if (!changed) break;
      }
    }
    if (excess.some((amount, vertex) => vertex !== source && vertex !== sink && amount !== 0)) {
      flow.set(previousFlow);
      phaseCount--; // Only feasible batches are published in iterationCount.
      break;
    }
    value += excess[sink];
    if (!excess[sink]) break;
  }
  const cutLabels = findLabels(source, false);
  for (let vertex = 0; vertex < nodeCount; vertex++)
    cut[vertex] = Number(cutLabels.distances[vertex] !== undefined);
  if (cutLabels.converged && !cut[sink]) complete = true;
  return {value, phaseCount, complete, cutSearchConverged: cutLabels.converged};
}
