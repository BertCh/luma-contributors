// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateMaximumCount, validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {KeplerAllocationError} from '../runtime/dataset-registry';
import {makeLocalFrame} from './catalog-local-frame';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  getLineParts,
  getPointPositions,
  makeCatalogGeometryText,
  makeCatalogTable,
  measureDistance,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection,
  type CatalogOutputTable,
  type CatalogPosition
} from './catalog-geojson';

export type CatalogNetworkParameters = {
  coordinateSystem: CatalogCoordinateSystem;
  /** Vertex clustering distance in meters (EPSG:4326) or coordinate units (planar). */
  snapTolerance: number;
  /** Split lines at interior vertices shared with another vertex; crossings are not noded. */
  splitAtSharedVertices: boolean;
  costMode: 'length' | 'property';
  forwardCostProperty: string;
  /** Empty means the reverse cost equals the forward cost. */
  reverseCostProperty: string;
  /** Property naming one-way restriction: forward, reverse or both (default). */
  onewayProperty: string;
};

export type NetworkEdge = {
  edgeId: string;
  sourceIndex: number;
  sourceId: string | null;
  partIndex: number;
  fromNode: number;
  toNode: number;
  coordinates: CatalogPosition[];
  cumulative: number[];
  length: number;
  /** null means traversal in that direction is not allowed. */
  forwardCost: number | null;
  reverseCost: number | null;
};

export type NetworkGraph = {
  coordinateSystem: CatalogCoordinateSystem;
  nodes: CatalogPosition[];
  edges: NetworkEdge[];
  outgoing: {edge: number; to: number; forward: boolean; cost: number}[][];
  componentOfNode: Int32Array;
  componentCount: number;
  diagnostics: CatalogDiagnostic[];
};

export type CatalogNetworkResult = {
  tables: CatalogOutputTable[];
  diagnostics: CatalogDiagnostic[];
  summary: Record<string, string | number>;
};

const FORWARD_TOKENS = new Set(['yes', 'true', '1', 'forward', 'f', 'ft']);
const REVERSE_TOKENS = new Set(['-1', 'reverse', 'backward', 'tf']);

export type Projector = (position: CatalogPosition) => [number, number];

/** Local azimuthal-equidistant meters (shared catalog local frame) for EPSG:4326; identity for planar. */
export function makeProjector(
  coordinateSystem: CatalogCoordinateSystem,
  positions: readonly CatalogPosition[]
): Projector {
  if (coordinateSystem === 'planar') return position => [position[0], position[1]];
  const frame = makeLocalFrame(coordinateSystem, positions);
  return position => frame.forward(position);
}

/** Read a finite numeric cost without coercing null, blank text or booleans to zero. */
export function readNetworkCost(value: unknown): number | null {
  const cost = typeof value === 'string' && value.trim() ? Number(value) : value;
  return typeof cost === 'number' && Number.isFinite(cost) ? cost : null;
}

/**
 * Builds the routing graph. Nodes cluster vertices within the snap tolerance greedily in input
 * order (deterministic, order dependent). Every part keeps source feature index and identifier.
 */
export function buildNetworkGraph(
  lines: CatalogFeatureCollection,
  parameters: CatalogNetworkParameters,
  maximumEdges: number,
  bindingId = 'network-build-graph'
): NetworkGraph {
  const diagnostics: CatalogDiagnostic[] = [];
  if (
    parameters.costMode === 'length' &&
    (parameters.forwardCostProperty || parameters.reverseCostProperty)
  )
    addCatalogDiagnostic(
      diagnostics,
      'cost-properties-ignored',
      'warning',
      'costMode=length ignores forwardCostProperty and reverseCostProperty; set costMode=property to route using the named cost columns.'
    );
  const parts: {sourceIndex: number; partIndex: number; coordinates: CatalogPosition[]}[] = [];
  const allPositions: CatalogPosition[] = [];
  lines.features.forEach((feature, sourceIndex) => {
    const sourceParts = getLineParts(feature.geometry);
    if (!sourceParts.length)
      addCatalogDiagnostic(
        diagnostics,
        'non-line-feature',
        'warning',
        'Non-line features are ignored.'
      );
    sourceParts.forEach((part, partIndex) => {
      const coordinates = part.filter(
        (position, index) =>
          Number.isFinite(position[0]) &&
          Number.isFinite(position[1]) &&
          (index === 0 || position[0] !== part[index - 1][0] || position[1] !== part[index - 1][1])
      );
      if (coordinates.length < 2) {
        addCatalogDiagnostic(
          diagnostics,
          'degenerate-line',
          'warning',
          'Lines with fewer than two distinct finite vertices are ignored.'
        );
        return;
      }
      parts.push({sourceIndex, partIndex, coordinates});
      allPositions.push(...coordinates);
    });
  });
  const project = makeProjector(parameters.coordinateSystem, allPositions);
  const tolerance = parameters.snapTolerance;
  const nodes: CatalogPosition[] = [];
  const projectedNodes: [number, number][] = [];
  const cells = new Map<string, number[]>();
  const findOrCreateNode = (position: CatalogPosition): number => {
    const [x, y] = project(position);
    const cellSize = tolerance > 0 ? tolerance : 1;
    const cellX = tolerance > 0 ? Math.floor(x / cellSize) : x;
    const cellY = tolerance > 0 ? Math.floor(y / cellSize) : y;
    if (tolerance > 0) {
      let best = -1;
      let bestDistance = tolerance;
      for (let dx = -1; dx <= 1; dx++)
        for (let dy = -1; dy <= 1; dy++)
          for (const candidate of cells.get(`${cellX + dx}|${cellY + dy}`) ?? []) {
            const distance = Math.hypot(
              projectedNodes[candidate][0] - x,
              projectedNodes[candidate][1] - y
            );
            if (distance <= bestDistance) {
              best = candidate;
              bestDistance = distance;
            }
          }
      if (best >= 0) return best;
    } else {
      const existing = cells.get(`${cellX}|${cellY}`);
      if (existing) return existing[0];
    }
    nodes.push(position);
    projectedNodes.push([x, y]);
    const key = `${cellX}|${cellY}`;
    cells.set(key, [...(cells.get(key) ?? []), nodes.length - 1]);
    return nodes.length - 1;
  };
  const partNodes = parts.map(part => part.coordinates.map(findOrCreateNode));
  const visits = new Map<number, number>();
  for (const ids of partNodes) for (const id of ids) visits.set(id, (visits.get(id) ?? 0) + 1);
  const edges: NetworkEdge[] = [];
  let invalidCostCount = 0;
  parts.forEach((part, partNumber) => {
    const ids = partNodes[partNumber];
    const feature = lines.features[part.sourceIndex];
    const properties = feature.properties ?? {};
    const cutPositions = [0];
    if (parameters.splitAtSharedVertices)
      for (let index = 1; index < ids.length - 1; index++)
        if ((visits.get(ids[index]) ?? 0) > 1) cutPositions.push(index);
    cutPositions.push(ids.length - 1);
    const wholeLength = measureLine(part.coordinates, parameters.coordinateSystem);
    const direction = String(properties[parameters.onewayProperty] ?? '').toLowerCase();
    const forwardAllowed = !REVERSE_TOKENS.has(direction);
    const reverseAllowed = !FORWARD_TOKENS.has(direction);
    const readCost = (name: string, parameter: string): number | null => {
      const cost = readNetworkCost(properties[name]);
      validateRequirement(
        bindingId,
        cost === null || cost >= 0,
        `${parameter} "${name}" cost must be at least 0; correct negative costs before routing`
      );
      return cost;
    };
    const propertyForwardCost =
      parameters.costMode === 'property'
        ? readCost(parameters.forwardCostProperty, 'forwardCostProperty')
        : null;
    const propertyReverseCost =
      parameters.costMode === 'property' && parameters.reverseCostProperty
        ? readCost(parameters.reverseCostProperty, 'reverseCostProperty')
        : propertyForwardCost;
    for (let cut = 0; cut + 1 < cutPositions.length; cut++) {
      const coordinates = part.coordinates.slice(cutPositions[cut], cutPositions[cut + 1] + 1);
      const cumulative = [0];
      for (let index = 1; index < coordinates.length; index++)
        cumulative.push(
          cumulative[index - 1] +
            measureDistance(coordinates[index - 1], coordinates[index], parameters.coordinateSystem)
        );
      const length = cumulative[cumulative.length - 1];
      const share = wholeLength > 0 ? length / wholeLength : 1;
      let forwardCost: number | null;
      let reverseCost: number | null;
      if (parameters.costMode === 'length') {
        forwardCost = length;
        reverseCost = length;
      } else {
        // Property costs apply to the whole source part and are prorated by length share.
        forwardCost = propertyForwardCost === null ? null : propertyForwardCost * share;
        reverseCost = propertyReverseCost === null ? null : propertyReverseCost * share;
      }
      const sanitize = (cost: number | null, allowed: boolean): number | null => {
        if (!allowed) return null;
        if (cost === null || !Number.isFinite(cost) || cost < 0) {
          invalidCostCount++;
          return null;
        }
        return cost;
      };
      validateMaximumCount(
        bindingId,
        'edgeCount (reduce lines or shared vertices, or disable splitAtSharedVertices)',
        edges.length + 1,
        maximumEdges
      );
      edges.push({
        edgeId: `${part.sourceIndex}:${part.partIndex}:${cut}`,
        sourceIndex: part.sourceIndex,
        sourceId: getFeatureIdentity(feature),
        partIndex: part.partIndex,
        fromNode: ids[cutPositions[cut]],
        toNode: ids[cutPositions[cut + 1]],
        coordinates,
        cumulative,
        length,
        forwardCost: sanitize(forwardCost, forwardAllowed),
        reverseCost: sanitize(reverseCost, reverseAllowed)
      });
    }
  });
  if (invalidCostCount)
    addCatalogDiagnostic(
      diagnostics,
      'invalid-cost-arc-removed',
      'warning',
      'Arcs with missing or non-numeric cost are removed from routing; supply finite non-negative cost values to retain them.',
      invalidCostCount
    );
  const outgoing: NetworkGraph['outgoing'] = nodes.map(() => []);
  edges.forEach((edge, index) => {
    if (edge.forwardCost !== null)
      outgoing[edge.fromNode].push({
        edge: index,
        to: edge.toNode,
        forward: true,
        cost: edge.forwardCost
      });
    if (edge.reverseCost !== null)
      outgoing[edge.toNode].push({
        edge: index,
        to: edge.fromNode,
        forward: false,
        cost: edge.reverseCost
      });
  });
  const parent = nodes.map((_, index) => index);
  const findRoot = (node: number): number => {
    while (parent[node] !== node) node = parent[node] = parent[parent[node]];
    return node;
  };
  for (const edge of edges) parent[findRoot(edge.fromNode)] = findRoot(edge.toNode);
  const rootIds = new Map<number, number>();
  const componentOfNode = new Int32Array(nodes.length);
  nodes.forEach((_, node) => {
    const root = findRoot(node);
    if (!rootIds.has(root)) rootIds.set(root, rootIds.size);
    componentOfNode[node] = rootIds.get(root)!;
  });
  return {
    coordinateSystem: parameters.coordinateSystem,
    nodes,
    edges,
    outgoing,
    componentOfNode,
    componentCount: rootIds.size,
    diagnostics
  };
}

function measureLine(
  coordinates: readonly CatalogPosition[],
  system: CatalogCoordinateSystem
): number {
  let length = 0;
  for (let index = 1; index < coordinates.length; index++)
    length += measureDistance(coordinates[index - 1], coordinates[index], system);
  return length;
}

export type SnappedPoint = {
  edge: number;
  /** Fraction of edge length from fromNode, in 0..1. */
  fraction: number;
  distance: number;
  position: CatalogPosition;
};

/** Nearest edge point by planar distance in local meters; null beyond the maximum distance. */
export function snapPointToNetwork(
  graph: NetworkGraph,
  point: CatalogPosition,
  maximumDistance: number
): SnappedPoint | null {
  const project = makeProjector(graph.coordinateSystem, graph.nodes);
  const [px, py] = project(point);
  let best: SnappedPoint | null = null;
  graph.edges.forEach((edge, edgeIndex) => {
    for (let index = 1; index < edge.coordinates.length; index++) {
      const [ax, ay] = project(edge.coordinates[index - 1]);
      const [bx, by] = project(edge.coordinates[index]);
      const lengthSquared = (bx - ax) ** 2 + (by - ay) ** 2;
      const u =
        lengthSquared === 0
          ? 0
          : Math.max(
              0,
              Math.min(1, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / lengthSquared)
            );
      const distance = Math.hypot(px - (ax + u * (bx - ax)), py - (ay + u * (by - ay)));
      if (distance <= maximumDistance && (!best || distance < best.distance)) {
        const along =
          edge.cumulative[index - 1] + u * (edge.cumulative[index] - edge.cumulative[index - 1]);
        const start = edge.coordinates[index - 1];
        const end = edge.coordinates[index];
        best = {
          edge: edgeIndex,
          fraction: edge.length > 0 ? along / edge.length : 0,
          distance,
          position: [start[0] + (end[0] - start[0]) * u, start[1] + (end[1] - start[1]) * u]
        };
      }
    }
  });
  return best;
}

/** Coordinates between two fractions of an edge (start <= end), interpolated linearly. */
export function sliceEdge(
  edge: NetworkEdge,
  startFraction: number,
  endFraction: number
): CatalogPosition[] {
  const startDistance = startFraction * edge.length;
  const endDistance = endFraction * edge.length;
  const interpolate = (distance: number): CatalogPosition => {
    for (let index = 1; index < edge.cumulative.length; index++)
      if (distance <= edge.cumulative[index] || index === edge.cumulative.length - 1) {
        const span = edge.cumulative[index] - edge.cumulative[index - 1];
        const u =
          span > 0 ? Math.max(0, Math.min(1, (distance - edge.cumulative[index - 1]) / span)) : 0;
        const a = edge.coordinates[index - 1];
        const b = edge.coordinates[index];
        return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u];
      }
    return edge.coordinates[0];
  };
  const result: CatalogPosition[] = [interpolate(startDistance)];
  for (let index = 1; index < edge.cumulative.length - 1; index++)
    if (edge.cumulative[index] > startDistance && edge.cumulative[index] < endDistance)
      result.push(edge.coordinates[index]);
  result.push(interpolate(endDistance));
  return result;
}

export type NetworkSearch = {
  distance: Float64Array;
  parentNode: Int32Array;
  parentEdge: Int32Array;
  parentForward: Uint8Array;
  /** Origin leg that seeded a node, -1 when reached over the network. */
  seed: Int32Array;
};

/** Multi-source Dijkstra with a binary heap; seeds are origin legs onto the first nodes. */
export function searchNetwork(
  graph: NetworkGraph,
  seeds: {node: number; cost: number}[]
): NetworkSearch {
  const count = graph.nodes.length;
  const search: NetworkSearch = {
    distance: new Float64Array(count).fill(Infinity),
    parentNode: new Int32Array(count).fill(-1),
    parentEdge: new Int32Array(count).fill(-1),
    parentForward: new Uint8Array(count),
    seed: new Int32Array(count).fill(-1)
  };
  const heapNodes: number[] = [];
  const heapCosts: number[] = [];
  const push = (node: number, cost: number) => {
    let index = heapNodes.length;
    heapNodes.push(node);
    heapCosts.push(cost);
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (heapCosts[parent] <= cost) break;
      heapNodes[index] = heapNodes[parent];
      heapCosts[index] = heapCosts[parent];
      index = parent;
    }
    heapNodes[index] = node;
    heapCosts[index] = cost;
  };
  const pop = (): [number, number] => {
    const top: [number, number] = [heapNodes[0], heapCosts[0]];
    const node = heapNodes.pop()!;
    const cost = heapCosts.pop()!;
    if (heapNodes.length) {
      let index = 0;
      for (;;) {
        let child = index * 2 + 1;
        if (child >= heapNodes.length) break;
        if (child + 1 < heapNodes.length && heapCosts[child + 1] < heapCosts[child]) child++;
        if (heapCosts[child] >= cost) break;
        heapNodes[index] = heapNodes[child];
        heapCosts[index] = heapCosts[child];
        index = child;
      }
      heapNodes[index] = node;
      heapCosts[index] = cost;
    }
    return top;
  };
  seeds.forEach((seed, seedIndex) => {
    if (seed.cost < search.distance[seed.node]) {
      search.distance[seed.node] = seed.cost;
      search.seed[seed.node] = seedIndex;
      push(seed.node, seed.cost);
    }
  });
  while (heapNodes.length) {
    const [node, cost] = pop();
    if (cost > search.distance[node]) continue;
    for (const arc of graph.outgoing[node]) {
      const next = cost + arc.cost;
      if (next < search.distance[arc.to]) {
        search.distance[arc.to] = next;
        search.parentNode[arc.to] = node;
        search.parentEdge[arc.to] = arc.edge;
        search.parentForward[arc.to] = arc.forward ? 1 : 0;
        search.seed[arc.to] = -1;
        push(arc.to, next);
      }
    }
  }
  return search;
}

export type OriginLeg = {
  node: number;
  cost: number;
  edge: number;
  startFraction: number;
  forward: boolean;
};

/** Origin on an edge can leave over either endpoint if that direction is allowed. */
export function makeOriginLegs(graph: NetworkGraph, snapped: SnappedPoint): OriginLeg[] {
  const edge = graph.edges[snapped.edge];
  const legs: OriginLeg[] = [];
  if (edge.forwardCost !== null)
    legs.push({
      node: edge.toNode,
      cost: (1 - snapped.fraction) * edge.forwardCost,
      edge: snapped.edge,
      startFraction: snapped.fraction,
      forward: true
    });
  if (edge.reverseCost !== null)
    legs.push({
      node: edge.fromNode,
      cost: snapped.fraction * edge.reverseCost,
      edge: snapped.edge,
      startFraction: snapped.fraction,
      forward: false
    });
  return legs;
}

export function runNetworkBuild(
  lines: CatalogFeatureCollection,
  parameters: CatalogNetworkParameters,
  maximumEdges: number
): CatalogNetworkResult {
  const graph = buildNetworkGraph(lines, parameters, maximumEdges, 'network-build-graph');
  const table = makeCatalogTable(
    'network-edges',
    'Network edges',
    {
      edge_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      from_node: 'int32',
      to_node: 'int32',
      length: 'float64',
      forward_cost: 'float64',
      reverse_cost: 'float64',
      direction: 'string',
      component: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  for (const edge of graph.edges)
    appendCatalogRow(table, {
      edge_id: edge.edgeId,
      source_index: edge.sourceIndex,
      source_id: edge.sourceId,
      part_index: edge.partIndex,
      from_node: edge.fromNode,
      to_node: edge.toNode,
      length: edge.length,
      forward_cost: edge.forwardCost,
      reverse_cost: edge.reverseCost,
      direction:
        edge.forwardCost !== null && edge.reverseCost !== null
          ? 'both'
          : edge.forwardCost !== null
            ? 'forward'
            : edge.reverseCost !== null
              ? 'reverse'
              : 'none',
      component: graph.componentOfNode[edge.fromNode],
      geometry: makeCatalogGeometryText('LineString', edge.coordinates)
    });
  const diagnostics = [...graph.diagnostics];
  if (graph.componentCount > 1)
    addCatalogDiagnostic(
      diagnostics,
      'disconnected-components',
      'info',
      'The network has more than one connected component.',
      graph.componentCount
    );
  addCatalogDiagnostic(
    diagnostics,
    'crossings-not-noded',
    'info',
    'Lines that cross without a shared vertex are not connected.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {
      nodeCount: graph.nodes.length,
      edgeCount: graph.edges.length,
      componentCount: graph.componentCount
    }
  };
}

export type CatalogRoutingParameters = {
  maximumSnapDistance: number;
};

/**
 * Injection points for engines that reuse the CPU graph, snapping and output assembly: a prebuilt
 * graph and a replacement for the least-cost search (the GPU engine supplies one computed by
 * GPUNetworkReachability from the same seeds).
 */
export type CatalogNetworkRunOptions = {
  graph?: NetworkGraph;
  search?: (graph: NetworkGraph, seeds: {node: number; cost: number}[]) => NetworkSearch;
  /** Origin snap computed by the caller (for example on the GPU); defaults to the CPU snap. */
  snappedOrigin?: SnappedPoint | null;
  /** Destination snaps computed by the caller, one per destination point in feature order. */
  snappedDestinations?: readonly (SnappedPoint | null)[];
};

function appendPath(coordinates: CatalogPosition[], addition: readonly CatalogPosition[]): void {
  for (const position of addition) {
    const last = coordinates[coordinates.length - 1];
    if (!last || last[0] !== position[0] || last[1] !== position[1]) coordinates.push(position);
  }
}

/** One path row per destination. Costs follow the declared directed arc costs. */
export function runNetworkShortestPaths(
  lines: CatalogFeatureCollection,
  origins: CatalogFeatureCollection,
  destinations: CatalogFeatureCollection,
  parameters: CatalogNetworkParameters & CatalogRoutingParameters,
  limits: {maximumEdges: number; maximumDestinations: number},
  options: CatalogNetworkRunOptions = {}
): CatalogNetworkResult {
  const graph =
    options.graph ??
    buildNetworkGraph(lines, parameters, limits.maximumEdges, 'network-shortest-path');
  const originPosition = getPointPositions(origins.features[0]?.geometry ?? null)[0];
  // Exactly one origin point feature is required.
  if (origins.features.length !== 1 || !originPosition)
    throw new Error('One origin point required');
  const destinationPositions = destinations.features.flatMap((feature, featureIndex) =>
    getPointPositions(feature.geometry).map(position => ({position, featureIndex}))
  );
  // Destination capacity exceeded.
  if (destinationPositions.length > limits.maximumDestinations)
    throw new KeplerAllocationError('capacity', {message: 'Destination capacity exceeded'});
  const diagnostics = [...graph.diagnostics];
  const table = makeCatalogTable(
    'network-paths',
    'Shortest paths',
    {
      path_id: 'string',
      destination_index: 'int32',
      destination_source_id: 'string',
      status: 'string',
      total_cost: 'float64',
      snap_distance_origin: 'float64',
      snap_distance_destination: 'float64',
      edge_count: 'int32',
      edge_ids: 'string',
      source_indices: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  const snappedOrigin =
    options.snappedOrigin !== undefined
      ? options.snappedOrigin
      : snapPointToNetwork(graph, originPosition, parameters.maximumSnapDistance);
  const legs = snappedOrigin ? makeOriginLegs(graph, snappedOrigin) : [];
  const search = (options.search ?? searchNetwork)(
    graph,
    legs.map(leg => ({node: leg.node, cost: leg.cost}))
  );
  const statusCounts: Record<string, number> = {};
  destinationPositions.forEach(({position, featureIndex}, destinationIndex) => {
    const base = {
      path_id: `0:${destinationIndex}`,
      destination_index: destinationIndex,
      destination_source_id: getFeatureIdentity(destinations.features[featureIndex]),
      snap_distance_origin: snappedOrigin?.distance ?? null
    };
    const finish = (status: string, row: Record<string, string | number | null> = {}) => {
      statusCounts[status] = (statusCounts[status] ?? 0) + 1;
      appendCatalogRow(table, {...base, status, ...row});
    };
    const snapped = options.snappedDestinations
      ? (options.snappedDestinations[destinationIndex] ?? null)
      : snapPointToNetwork(graph, position, parameters.maximumSnapDistance);
    if (!snappedOrigin) return finish('origin-not-snapped');
    if (!snapped) return finish('destination-not-snapped');
    const edge = graph.edges[snapped.edge];
    // Candidate arrivals: over the start node going forward or over the end node going backward.
    type Candidate =
      | {cost: number; viaNode: number; forward: boolean}
      | {cost: number; direct: true};
    let best: Candidate | null = null;
    const consider = (candidate: Candidate) => {
      if (!best || candidate.cost < best.cost) best = candidate;
    };
    if (edge.forwardCost !== null && Number.isFinite(search.distance[edge.fromNode]))
      consider({
        cost: search.distance[edge.fromNode] + snapped.fraction * edge.forwardCost,
        viaNode: edge.fromNode,
        forward: true
      });
    if (edge.reverseCost !== null && Number.isFinite(search.distance[edge.toNode]))
      consider({
        cost: search.distance[edge.toNode] + (1 - snapped.fraction) * edge.reverseCost,
        viaNode: edge.toNode,
        forward: false
      });
    if (snapped.edge === snappedOrigin.edge) {
      if (snapped.fraction >= snappedOrigin.fraction && edge.forwardCost !== null)
        consider({
          cost: (snapped.fraction - snappedOrigin.fraction) * edge.forwardCost,
          direct: true
        });
      if (snapped.fraction < snappedOrigin.fraction && edge.reverseCost !== null)
        consider({
          cost: (snappedOrigin.fraction - snapped.fraction) * edge.reverseCost,
          direct: true
        });
    }
    const chosen = best as Candidate | null;
    if (!chosen) {
      const originEdge = graph.edges[snappedOrigin.edge];
      const sameComponent =
        graph.componentOfNode[originEdge.fromNode] === graph.componentOfNode[edge.fromNode];
      return finish(sameComponent ? 'unreachable-directed' : 'unreachable-disconnected', {
        snap_distance_destination: snapped.distance
      });
    }
    const coordinates: CatalogPosition[] = [];
    const edgeIds: string[] = [];
    const sourceIndices: number[] = [];
    const visit = (usedEdge: NetworkEdge, startFraction: number, endFraction: number) => {
      // Zero-length traversals (a point exactly on a node) do not use the edge.
      if (startFraction === endFraction) return;
      const forward = startFraction <= endFraction;
      const slice = sliceEdge(
        usedEdge,
        Math.min(startFraction, endFraction),
        Math.max(startFraction, endFraction)
      );
      appendPath(coordinates, forward ? slice : slice.reverse());
      edgeIds.push(usedEdge.edgeId);
      sourceIndices.push(usedEdge.sourceIndex);
    };
    if ('direct' in chosen) visit(edge, snappedOrigin.fraction, snapped.fraction);
    else {
      const trail: {edge: number; forward: boolean}[] = [];
      let node = chosen.viaNode;
      while (search.seed[node] === -1 && search.parentNode[node] !== -1) {
        trail.push({edge: search.parentEdge[node], forward: search.parentForward[node] === 1});
        node = search.parentNode[node];
      }
      const leg = legs[search.seed[node]];
      visit(graph.edges[leg.edge], leg.startFraction, leg.forward ? 1 : 0);
      for (const step of trail.reverse())
        visit(graph.edges[step.edge], step.forward ? 0 : 1, step.forward ? 1 : 0);
      visit(edge, chosen.forward ? 0 : 1, snapped.fraction);
    }
    if (coordinates.length < 2)
      coordinates.splice(0, coordinates.length, snappedOrigin.position, snapped.position);
    const uniqueIds = edgeIds.filter((id, index) => edgeIds[index - 1] !== id);
    finish('ok', {
      total_cost: chosen.cost,
      snap_distance_destination: snapped.distance,
      edge_count: uniqueIds.length,
      edge_ids: uniqueIds.join(','),
      source_indices: sourceIndices.join(','),
      geometry: makeCatalogGeometryText('LineString', coordinates)
    });
  });
  for (const [status, count] of Object.entries(statusCounts))
    if (status !== 'ok')
      addCatalogDiagnostic(diagnostics, status, 'warning', `Paths with status ${status}.`, count);
  addCatalogDiagnostic(
    diagnostics,
    'partial-edge-cost-proportional',
    'info',
    'Partial-edge cost is proportional to length along the snapped edge.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {
      destinationCount: destinationPositions.length,
      reachedCount: statusCounts['ok'] ?? 0,
      nodeCount: graph.nodes.length,
      edgeCount: graph.edges.length
    }
  };
}

/**
 * Edges reachable within the largest budget, with first-reach cost, band and reached geometry.
 * Unreachable edges can be retained with a status that separates disconnected from directed.
 */
export function runNetworkServiceArea(
  lines: CatalogFeatureCollection,
  origins: CatalogFeatureCollection,
  parameters: CatalogNetworkParameters &
    CatalogRoutingParameters & {budgets: readonly number[]; includeUnreachable: boolean},
  limits: {maximumEdges: number},
  options: CatalogNetworkRunOptions = {}
): CatalogNetworkResult {
  const graph =
    options.graph ??
    buildNetworkGraph(lines, parameters, limits.maximumEdges, 'network-service-area');
  const originPosition = getPointPositions(origins.features[0]?.geometry ?? null)[0];
  // Exactly one origin point feature is required.
  if (origins.features.length !== 1 || !originPosition)
    throw new Error('One origin point required');
  const budgets = [...parameters.budgets].sort((a, b) => a - b);
  const maximumBudget = budgets[budgets.length - 1];
  const diagnostics = [...graph.diagnostics];
  const table = makeCatalogTable(
    'network-service-area',
    'Service area edges',
    {
      edge_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      status: 'string',
      first_reach_cost: 'float64',
      band: 'int32',
      band_budget: 'float64',
      reached_fraction: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const snappedOrigin = snapPointToNetwork(graph, originPosition, parameters.maximumSnapDistance);
  if (!snappedOrigin) {
    addCatalogDiagnostic(
      diagnostics,
      'origin-not-snapped',
      'error',
      'The origin has no network edge within the maximum snap distance.'
    );
    return {
      tables: [table],
      diagnostics,
      summary: {reachedEdgeCount: 0, unreachableEdgeCount: graph.edges.length}
    };
  }
  const legs = makeOriginLegs(graph, snappedOrigin);
  const search = (options.search ?? searchNetwork)(
    graph,
    legs.map(leg => ({node: leg.node, cost: leg.cost}))
  );
  const originComponent = graph.componentOfNode[graph.edges[snappedOrigin.edge].fromNode];
  const counts = {full: 0, partial: 0, beyond: 0, disconnected: 0, directed: 0};
  graph.edges.forEach((edge, edgeIndex) => {
    const intervals: [number, number][] = [];
    let firstReach = Infinity;
    const startCost = search.distance[edge.fromNode];
    const endCost = search.distance[edge.toNode];
    if (edge.forwardCost !== null && startCost <= maximumBudget) {
      firstReach = Math.min(firstReach, startCost);
      intervals.push([
        0,
        edge.forwardCost > 0 ? Math.min(1, (maximumBudget - startCost) / edge.forwardCost) : 1
      ]);
    }
    if (edge.reverseCost !== null && endCost <= maximumBudget) {
      firstReach = Math.min(firstReach, endCost);
      intervals.push([
        edge.reverseCost > 0 ? Math.max(0, 1 - (maximumBudget - endCost) / edge.reverseCost) : 0,
        1
      ]);
    }
    if (edgeIndex === snappedOrigin.edge) {
      firstReach = 0;
      const low =
        edge.reverseCost !== null && edge.reverseCost > 0
          ? Math.max(0, snappedOrigin.fraction - maximumBudget / edge.reverseCost)
          : edge.reverseCost === 0
            ? 0
            : snappedOrigin.fraction;
      const high =
        edge.forwardCost !== null && edge.forwardCost > 0
          ? Math.min(1, snappedOrigin.fraction + maximumBudget / edge.forwardCost)
          : edge.forwardCost === 0
            ? 1
            : snappedOrigin.fraction;
      intervals.push([low, high]);
    }
    intervals.sort((a, b) => a[0] - b[0]);
    const merged: [number, number][] = [];
    for (const interval of intervals) {
      const last = merged[merged.length - 1];
      if (last && interval[0] <= last[1]) last[1] = Math.max(last[1], interval[1]);
      else merged.push([...interval]);
    }
    const reachedFraction = merged.reduce((sum, [low, high]) => sum + (high - low), 0);
    const base = {edge_id: edge.edgeId, source_index: edge.sourceIndex, source_id: edge.sourceId};
    if (!merged.length || (reachedFraction <= 0 && edgeIndex !== snappedOrigin.edge)) {
      const entryCost = Math.min(
        edge.forwardCost !== null ? startCost : Infinity,
        edge.reverseCost !== null ? endCost : Infinity
      );
      const component = graph.componentOfNode[edge.fromNode];
      const status = Number.isFinite(entryCost)
        ? 'unreachable-beyond-budget'
        : component === originComponent
          ? 'unreachable-directed'
          : 'unreachable-disconnected';
      if (status === 'unreachable-beyond-budget') counts.beyond++;
      else if (status === 'unreachable-directed') counts.directed++;
      else counts.disconnected++;
      if (parameters.includeUnreachable)
        appendCatalogRow(table, {
          ...base,
          status,
          first_reach_cost: Number.isFinite(entryCost) ? entryCost : null,
          reached_fraction: 0,
          geometry: makeCatalogGeometryText('LineString', edge.coordinates)
        });
      return;
    }
    const isFull = merged.length === 1 && merged[0][0] <= 0 && merged[0][1] >= 1;
    if (isFull) counts.full++;
    else counts.partial++;
    const bandIndex = budgets.findIndex(budget => firstReach <= budget);
    const pieces = merged.map(([low, high]) => sliceEdge(edge, low, high));
    appendCatalogRow(table, {
      ...base,
      status: isFull ? 'reached-full' : 'reached-partial',
      first_reach_cost: firstReach,
      band: bandIndex + 1,
      band_budget: budgets[bandIndex],
      reached_fraction: Math.min(1, reachedFraction),
      geometry:
        pieces.length === 1
          ? makeCatalogGeometryText('LineString', pieces[0])
          : makeCatalogGeometryText('MultiLineString', pieces)
    });
  });
  if (counts.directed)
    addCatalogDiagnostic(
      diagnostics,
      'unreachable-directed',
      'warning',
      'Edges in the origin component with no directed path.',
      counts.directed
    );
  if (counts.disconnected)
    addCatalogDiagnostic(
      diagnostics,
      'unreachable-disconnected',
      'warning',
      'Edges in a different connected component.',
      counts.disconnected
    );
  if (counts.beyond)
    addCatalogDiagnostic(
      diagnostics,
      'unreachable-beyond-budget',
      'info',
      'Edges whose first-reach cost exceeds the largest budget.',
      counts.beyond
    );
  addCatalogDiagnostic(
    diagnostics,
    'no-isochrone-polygons',
    'info',
    'Bands are edge attributes; no isochrone polygon boundary is generated.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {
      reachedEdgeCount: counts.full + counts.partial,
      fullEdgeCount: counts.full,
      partialEdgeCount: counts.partial,
      unreachableEdgeCount: counts.beyond + counts.directed + counts.disconnected,
      nodeCount: graph.nodes.length,
      edgeCount: graph.edges.length
    }
  };
}
