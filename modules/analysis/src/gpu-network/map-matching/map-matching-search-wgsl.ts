// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Words per point of the hop table shared by reconstruction passes. @internal */
export const HOP_WORD_COUNT = 5;

/** Hash cells for a node table: the next power of two at least twice the table size. */
export function getHashSize(budget: number): number {
  let size = 16;
  while (size < 2 * budget) {
    size *= 2;
  }
  return size;
}

/** Module declarations of the forward Viterbi kernel: private route-search tables and search. */
export function getForwardDeclarations(
  budget: number,
  hashSize: number,
  candidateCount: number,
  edgeTableAccess: string
): string {
  return `${edgeTableAccess}
${getAccessDeclarations()}
${getRouteSearchDeclarations(budget, hashSize, candidateCount)}`;
}

/**
 * Declarations of the bounded route search alone: private node table, hash and `searchNetwork`.
 * Needs the `adjacency`, `edgeTable` and `nodePositions` bindings and the edge table accessors.
 */
export function getRouteSearchDeclarations(
  budget: number,
  hashSize: number,
  candidateCount: number,
  edgeTableAccess: string = ''
): string {
  return `${edgeTableAccess}
const S: u32 = ${budget}u;
const HASH_MASK: u32 = ${hashSize - 1}u;
var<private> tableNode: array<u32, ${budget}>;
var<private> tableDistance: array<f32, ${budget}>;
var<private> tableKey: array<f32, ${budget}>;
var<private> tableDone: array<u32, ${budget}>;
// Edge row that reached each table node with its current distance (NONE for the start).
var<private> tableParent: array<u32, ${budget}>;
var<private> tableHash: array<u32, ${hashSize}>;
var<private> pendingTargets: array<u32, ${candidateCount}>;
var<private> targetPositions: array<vec2<f32>, ${candidateCount}>;
var<private> targetTotal: u32 = 0u;
// Discovered nodes the full table could not keep in the last search.
var<private> droppedCount: u32 = 0u;

// Consistent A* heuristic: the distance to the nearest target node, planar or great-circle. Every
// edge is as long as the straight line (or great circle) between its nodes, so by the triangle
// inequality the heuristic never overestimates; the factor
// below 1 absorbs f32 rounding so it stays consistent in floating point.
fn getHeuristic(node: u32) -> f32 {
  if (targetTotal == 0u || node >= NODE_COUNT) {
    return 0.0;
  }
  let position = readNode(node);
  var nearest = 3.0e38;
  for (var slot = 0u; slot < targetTotal; slot++) {
    nearest = min(nearest, networkDistance(position, targetPositions[slot]));
  }
  return nearest * 0.999999;
}

fn getHashStart(node: u32) -> u32 {
  return ((node * 2654435761u) >> 7u) & HASH_MASK;
}
// Table slot of a node, or NONE. The hash has at least twice as many cells as table slots.
fn findEntry(node: u32) -> u32 {
  var cell = getHashStart(node);
  loop {
    let stored = tableHash[cell];
    if (stored == 0u) {
      return NONE;
    }
    if (tableNode[stored - 1u] == node) {
      return stored - 1u;
    }
    cell = (cell + 1u) & HASH_MASK;
  }
}
// Removes the hash cell of table slot 'slot' with backward-shift deletion: later cells of the
// probe run move into the hole unless their home cell lies cyclically after it, so every lookup
// still reaches its entry without tombstones. The hash always has an empty cell (it has at least
// twice as many cells as table slots), so the loops end.
fn removeHashEntry(slot: u32) {
  var hole = getHashStart(tableNode[slot]);
  while (tableHash[hole] != slot + 1u) {
    hole = (hole + 1u) & HASH_MASK;
  }
  tableHash[hole] = 0u;
  var cell = (hole + 1u) & HASH_MASK;
  loop {
    let stored = tableHash[cell];
    if (stored == 0u) {
      break;
    }
    let home = getHashStart(tableNode[stored - 1u]);
    if (((cell - home) & HASH_MASK) >= ((cell - hole) & HASH_MASK)) {
      tableHash[hole] = stored;
      tableHash[cell] = 0u;
      hole = cell;
    }
    cell = (cell + 1u) & HASH_MASK;
  }
}
// The open (unsettled) slot with the largest key, ties to the highest slot, or NONE.
fn findWorstOpenEntry(count: u32) -> u32 {
  var worst = NONE;
  for (var slot = 0u; slot < count; slot++) {
    if (tableDone[slot] == 0u && (worst == NONE || tableKey[slot] >= tableKey[worst])) {
      worst = slot;
    }
  }
  return worst;
}
fn addEntry(slot: u32, node: u32, distance: f32, parentEdge: u32) {
  tableNode[slot] = node;
  tableParent[slot] = parentEdge;
  tableDistance[slot] = distance;
  tableKey[slot] = distance + getHeuristic(node);
  tableDone[slot] = 0u;
  var cell = getHashStart(node);
  while (tableHash[cell] != 0u) {
    cell = (cell + 1u) & HASH_MASK;
  }
  tableHash[cell] = slot + 1u;
}

// Bounded A* from 'start' over the CSR network; distances within 'limit' land in the private
// table (at most S nodes; a full table counts every node it cannot keep in 'droppedCount' and keeps
// the open nodes with the best keys). Nodes are settled in order of distance plus the heuristic, so a node
// settled (done) has its exact network distance, and the search heads toward the targets instead of
// settling the whole disc of radius 'limit' as plain Dijkstra does. It stops once every node in
// 'pendingTargets' (NONE for unused slots; 'pendingCount' of them are set) is settled, or once the
// best key exceeds 'limit' (no remaining target can be within the bound). Returns the table size.
fn searchNetwork(start: u32, limit: f32, pendingCount: u32) -> u32 {
  var pending = pendingCount;
  droppedCount = 0u;
  targetTotal = 0u;
  for (var slot = 0u; slot < K; slot++) {
    if (pendingTargets[slot] != NONE) {
      targetPositions[targetTotal] = readNode(pendingTargets[slot]);
      targetTotal++;
    }
  }
  // A runtime trip count prevents the Metal back end from losing unrolled integer-table stores.
  let hashCellCount = (HASH_MASK + 1u) * min(arrayLength(&adjacency), 1u);
  for (var slot = 0u; slot < hashCellCount; slot++) {
    tableHash[slot] = 0u;
  }
  var count = 0u;
  addEntry(count, start, 0.0, NONE);
  count = 1u;
  loop {
    var best = NONE;
    // At large finite bounds, limit + 1 rounds back to limit. Select the first finite key
    // explicitly so a key exactly at the inclusive bound is still settled.
    var bestKey = limit;
    for (var slot = 0u; slot < count; slot++) {
      if (tableDone[slot] == 0u && bitcast<u32>(tableKey[slot]) < 0x7f800000u &&
          (best == NONE || tableKey[slot] < bestKey)) {
        best = slot;
        bestKey = tableKey[slot];
      }
    }
    if (best == NONE || bestKey > limit) {
      break;
    }
    let bestDistance = tableDistance[best];
    tableDone[best] = 1u;
    let node = tableNode[best];
    for (var slot = 0u; slot < K; slot++) {
      if (pendingTargets[slot] == node) {
        pendingTargets[slot] = NONE;
        pending--;
      }
    }
    if (pending == 0u) {
      break;
    }
    if (node >= NODE_COUNT) {
      continue;
    }
    let end = min(adjacency[adjacencyOffset + node + 1u], EDGE_COUNT);
    for (var edge = adjacency[adjacencyOffset + node]; edge < end; edge++) {
      if (!isEdgeValid(edge)) {
        continue;
      }
      let candidateDistance = bestDistance + edgeLength(edge);
      if (candidateDistance > limit) {
        continue;
      }
      let neighbor = edgeTarget(edge);
      let found = findEntry(neighbor);
      if (found != NONE) {
        if (tableDone[found] == 0u && candidateDistance < tableDistance[found]) {
          tableDistance[found] = candidateDistance;
          tableParent[found] = edge;
          tableKey[found] = candidateDistance + getHeuristic(neighbor);
        }
      } else if (count < S) {
        addEntry(count, neighbor, candidateDistance, edge);
        count++;
      } else {
        // Full table: the node is dropped, or replaces the open entry with the worst key when its
        // own key is better. Either way the search may now miss or overestimate a route.
        droppedCount++;
        let key = candidateDistance + getHeuristic(neighbor);
        let worst = findWorstOpenEntry(count);
        if (worst != NONE && key < tableKey[worst]) {
          removeHashEntry(worst);
          addEntry(worst, neighbor, candidateDistance, edge);
        }
      }
    }
  }
  return count;
}

fn lookupDistance(node: u32) -> f32 {
  let slot = findEntry(node);
  if (slot != NONE && tableDone[slot] != 0u) {
    return tableDistance[slot];
  }
  return -1.0;
}
`;
}

/** Candidate and point accessors shared by the route and Viterbi kernels. */
export function getAccessDeclarations(): string {
  return `fn readPoint(point: u32) -> vec2<f32> {
  return vec2<f32>(points[pointsOffset + point * 2u], points[pointsOffset + point * 2u + 1u]);
}
fn getCandidateEdge(point: u32, slot: u32) -> u32 {
  return candidates[candidatesOffset + (point * K + slot) * 3u];
}
fn getCandidateFraction(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 1u]);
}
fn getCandidateDistance(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 2u]);
}

`;
}

/**
 * Body of the route kernel: one thread per (point, candidate of the previous point). Writes the
 * route length to every candidate of the point, or -1 when there is none within the bound.
 */
export function getRoutesBody(hasRouteTable: boolean, hasFixFilter: boolean): string {
  return `let point = index / K;
  let fromSlot = index % K;
  ${
    hasRouteTable
      ? `// A non-negative trailing word marks a row the route table resolved (route-table kernel).
  if (routes[routesOffset + index * ROUTE_STRIDE + K] >= 0.0) {
    return;
  }
  `
      : ''
  }for (var to = 0u; to < K; to++) {
    routes[routesOffset + index * ROUTE_STRIDE + to] = -1.0;
  }
  routes[routesOffset + index * ROUTE_STRIDE + K] = 0.0;
  // Point 0 has no predecessor even when it lies before the first track (review B-23).
  ${getPreviousFixWGSL(hasFixFilter)}
  let fromEdge = getCandidateEdge(previous, fromSlot);
  if (fromEdge == NONE) {
    return;
  }
  let straight = networkDistance(readPoint(point), readPoint(previous));
  let bound = straight * parameters[parametersOffset + 3u] + parameters[parametersOffset + 4u];
  let fromLength = edgeLength(fromEdge);
  let fromFraction = getCandidateFraction(previous, fromSlot);
  let tail = (1.0 - fromFraction) * fromLength;
  let remaining = bound - tail;
  if (remaining >= 0.0) {
    var pendingCount = 0u;
    for (var to = 0u; to < K; to++) {
      let toEdge = getCandidateEdge(point, to);
      let isSameEdgeAhead = toEdge == fromEdge && getCandidateFraction(point, to) >= fromFraction;
      let isPending = toEdge != NONE && !isSameEdgeAhead;
      pendingTargets[to] = select(NONE, edgeSource(select(0u, toEdge, isPending)), isPending);
      pendingCount += select(0u, 1u, isPending);
    }
    searchNetwork(edgeTarget(fromEdge), remaining, pendingCount);
    routes[routesOffset + index * ROUTE_STRIDE + K] = f32(min(droppedCount, 16777216u));
  }
  for (var to = 0u; to < K; to++) {
    let toEdge = getCandidateEdge(point, to);
    if (toEdge == NONE) {
      continue;
    }
    let toFraction = getCandidateFraction(point, to);
    var route = -1.0;
    if (toEdge == fromEdge && toFraction >= fromFraction) {
      route = (toFraction - fromFraction) * fromLength;
    } else if (remaining >= 0.0) {
      let network = lookupDistance(edgeSource(toEdge));
      if (network >= 0.0) {
        route = tail + network + toFraction * edgeLength(toEdge);
      }
    }
    // The bound covers the whole route, including the partial edge at either end (review B-24).
    if (route > bound) {
      route = -1.0;
    }
    routes[routesOffset + index * ROUTE_STRIDE + to] = route;
  }`;
}

/**
 * Lines that return from a (point, candidate) kernel when `point` has no previous fix to come
 * from, and otherwise declare `previous`, the point the transition starts at: the preceding point,
 * or, with a fix filter, the previous retained fix.
 */
export function getPreviousFixWGSL(hasFixFilter: boolean): string {
  if (hasFixFilter) {
    return `let previous = predecessors[predecessorsOffset + point];
  if (previous >= DROPPED_FIX) {
    return;
  }`;
  }
  return `if (point == 0u || pointStarts[pointStartsOffset + point] != 0u) {
    return;
  }
  let previous = point - 1u;`;
}

/** Candidate accessors and the binary search of the `route-table` kernel. */
export function getRouteTableDeclarations(): string {
  return `fn getCandidateEdge(point: u32, slot: u32) -> u32 {
  return candidates[candidatesOffset + (point * K + slot) * 3u];
}
fn getCandidateFraction(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 1u]);
}
// Cost from 'row' to 'targetNode' by binary search of the row's ascending targets, or -1 when the
// table does not list it (the cost then exceeds the table's cost limit).
fn findTableCost(row: u32, targetNode: u32) -> f32 {
  var low = tableOffsets[tableOffsetsOffset + row];
  let end = tableOffsets[tableOffsetsOffset + row + 1u];
  var high = end;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (tableTargets[tableTargetsOffset + middle] < targetNode) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && tableTargets[tableTargetsOffset + low] == targetNode) {
    return tableCosts[tableCostsOffset + low];
  }
  return -1.0;
}`;
}

/**
 * Body of the route-table kernel: one thread per (point, candidate of the previous point). Resolves
 * the rows `route-bounds` marked (trailing word `>= 0`) exactly like the A* route kernel, with the
 * network cost read from the table, and marks them resolved (trailing word 0).
 */
export function getRouteTableBody(hasFixFilter: boolean): string {
  return `let base = routesOffset + index * ROUTE_STRIDE;
  let remaining = routes[base + K];
  if (!(remaining >= 0.0)) {
    return;
  }
  let bound = routes[base];
  let point = index / K;
  let previous = ${hasFixFilter ? 'predecessors[predecessorsOffset + point]' : 'point - 1u'};
  let fromEdge = getCandidateEdge(previous, index % K);
  let fromFraction = getCandidateFraction(previous, index % K);
  let fromLength = edgeLength(fromEdge);
  let tail = (1.0 - fromFraction) * fromLength;
  let startNode = edgeTarget(fromEdge);
  for (var to = 0u; to < K; to++) {
    let toEdge = getCandidateEdge(point, to);
    var route = -1.0;
    if (toEdge != NONE) {
      let toFraction = getCandidateFraction(point, to);
      if (toEdge == fromEdge && toFraction >= fromFraction) {
        route = (toFraction - fromFraction) * fromLength;
      } else {
        let network = findTableCost(startNode, edgeSource(toEdge));
        // The A* search only admits costs within the allowance; the table also lists larger ones.
        if (network >= 0.0 && network <= remaining) {
          route = tail + network + toFraction * edgeLength(toEdge);
        }
      }
      // The bound covers the whole route, including the partial edge at either end (review B-24).
      if (route > bound) {
        route = -1.0;
      }
    }
    routes[base + to] = route;
  }
  routes[base + K] = 0.0;`;
}

/** Private state and emit function of the `route-edges` kernel. */
export function getRouteEdgesDeclarations(budget: number): string {
  return `var<private> emittedCount: u32 = 0u;
var<private> pathEdges: array<u32, ${budget}>;
fn emitEdge(track: u32, edge: u32) {
  if (emittedCount < EDGE_CAPACITY) {
    traversedEdges[traversedEdgesOffset + track * EDGE_CAPACITY + emittedCount] = edge;
  } else {
    atomicStore(&traversedStatus[traversedStatusOffset], 1u);
  }
  emittedCount++;
}`;
}

/** Body of the route-edges kernel: one thread per track walks its hops in order. */
export function getRouteEdgesBody(): string {
  return `let first = trackOffsets[trackOffsetsOffset + index];
  let last = trackOffsets[trackOffsetsOffset + index + 1u];
  emittedCount = 0u;
  for (var point = first; point < last; point++) {
    let base = hopsOffset + point * ${HOP_WORD_COUNT}u;
    let kind = hops[base];
    if (kind == 0u || kind == 3u) {
      continue;
    }
    let toEdge = hops[base + 4u];
    if (kind == 1u) {
      if (emittedCount > 0u) {
        emitEdge(index, NONE);
      }
      emitEdge(index, toEdge);
      continue;
    }
    // Hop over the network: search again with the allowance of the route kernel, then walk the
    // parent edges from the target node back to the start node.
    let startNode = hops[base + 1u];
    let targetNode = hops[base + 2u];
    var pathLength = 0u;
    var isRecovered = true;
    if (startNode != targetNode) {
      for (var slot = 0u; slot < K; slot++) {
        pendingTargets[slot] = NONE;
      }
      pendingTargets[0] = targetNode;
      searchNetwork(startNode, bitcast<f32>(hops[base + 3u]), 1u);
      if (droppedCount > 0u) { atomicStore(&traversedStatus[traversedStatusOffset + 1u], 1u); }
      var node = targetNode;
      loop {
        if (node == startNode) {
          break;
        }
        let entry = findEntry(node);
        if (entry == NONE || tableDone[entry] == 0u || pathLength >= S) {
          isRecovered = false;
          break;
        }
        pathEdges[pathLength] = tableParent[entry];
        pathLength++;
        node = edgeSource(tableParent[entry]);
      }
    }
    if (!isRecovered) {
      atomicStore(&traversedStatus[traversedStatusOffset + 1u], 1u);
      emitEdge(index, NONE);
    } else {
      for (var step = 0u; step < pathLength; step++) {
        emitEdge(index, pathEdges[pathLength - 1u - step]);
      }
    }
    emitEdge(index, toEdge);
  }
  traversedCounts[traversedCountsOffset + index] = emittedCount;`;
}

/** Returns every read-only view of a map-matching contributor. */
