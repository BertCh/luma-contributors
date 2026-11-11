// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  NONE,
  type AccessibilityCSR,
  type OracleSnap
} from '../network-accessibility/network-accessibility-oracle';

/** Result of {@link getDriveTimeOracle}. */
export type DriveTimeOracle = {
  /** Cost from the nearest facility to every network node, `Infinity` when unreached. */
  nodeCosts: number[];
  /** Cost from the nearest facility to every demand point, `Infinity` when unsnapped or unreached. */
  demandTimes: number[];
};

/**
 * Drive times by the definition, in binary64: every snapped facility and demand point becomes a
 * node of its own on the straight road between the two end nodes of its edge, the road is cut
 * into pieces at those nodes with costs in proportion to their length, and one multi-source
 * Dijkstra runs from the facility nodes. A road is driven in a direction only when the network
 * has a passable edge that way, at the cost of the cheapest one. No formula for the demand time
 * appears here, so it checks the recipe's adapters independently.
 */
export function getDriveTimeOracle(
  csr: AccessibilityCSR,
  nodeCount: number,
  facilitySnaps: readonly OracleSnap[],
  demandSnaps: readonly OracleSnap[]
): DriveTimeOracle {
  const adjacency: {target: number; cost: number}[][] = Array.from({length: nodeCount}, () => []);
  for (let edge = 0; edge < csr.neighbors.length; edge++) {
    const weight = csr.weights[edge];
    if (csr.neighbors[edge] < nodeCount && weight >= 0 && Number.isFinite(weight)) {
      adjacency[csr.sources[edge]].push({target: csr.neighbors[edge], cost: weight});
    }
  }
  const getRoadCost = (fromNode: number, toNode: number): number =>
    adjacency[fromNode]
      .filter(({target}) => target === toNode)
      .reduce((best, {cost}) => Math.min(best, cost), Infinity);

  // Snapped points per road, keyed by the unordered pair of end nodes.
  type RoadPoint = {node: number; fromLow: number};
  const roads = new Map<string, {low: number; high: number; points: RoadPoint[]}>();
  const addPoint = (snap: OracleSnap): number => {
    if (snap.edge === NONE) {
      return -1;
    }
    const source = csr.sources[snap.edge];
    const target = csr.neighbors[snap.edge];
    const low = Math.min(source, target);
    const high = Math.max(source, target);
    const road = roads.get(`${low}-${high}`) ?? {low, high, points: []};
    roads.set(`${low}-${high}`, road);
    adjacency.push([]);
    road.points.push({
      node: adjacency.length - 1,
      fromLow: source === low ? snap.fraction : 1 - snap.fraction
    });
    return adjacency.length - 1;
  };
  const facilityNodes = facilitySnaps.map(addPoint);
  const demandNodes = demandSnaps.map(addPoint);
  for (const {low, high, points} of roads.values()) {
    const upCost = getRoadCost(low, high);
    const downCost = getRoadCost(high, low);
    const stops = [
      {node: low, fromLow: 0},
      ...points.sort((first, second) => first.fromLow - second.fromLow),
      {node: high, fromLow: 1}
    ];
    for (let stop = 0; stop + 1 < stops.length; stop++) {
      const share = stops[stop + 1].fromLow - stops[stop].fromLow;
      if (Number.isFinite(upCost)) {
        adjacency[stops[stop].node].push({target: stops[stop + 1].node, cost: share * upCost});
      }
      if (Number.isFinite(downCost)) {
        adjacency[stops[stop + 1].node].push({target: stops[stop].node, cost: share * downCost});
      }
    }
  }

  const distances = new Array<number>(adjacency.length).fill(Infinity);
  for (const node of facilityNodes) {
    if (node >= 0) {
      distances[node] = 0;
    }
  }
  const settled = new Uint8Array(adjacency.length);
  for (;;) {
    let current = -1;
    for (let node = 0; node < adjacency.length; node++) {
      if (!settled[node] && (current < 0 || distances[node] < distances[current])) {
        current = node;
      }
    }
    if (current < 0 || distances[current] === Infinity) {
      break;
    }
    settled[current] = 1;
    for (const {target, cost} of adjacency[current]) {
      distances[target] = Math.min(distances[target], distances[current] + cost);
    }
  }
  return {
    nodeCosts: distances.slice(0, nodeCount),
    demandTimes: demandNodes.map(node => (node < 0 ? Infinity : distances[node]))
  };
}
