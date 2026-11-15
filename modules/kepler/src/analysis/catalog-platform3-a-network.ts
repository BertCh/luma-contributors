// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {KeplerAllocationError} from '../runtime/dataset-registry';
import {
  getNetworkParameters,
  MAXIMUM_NETWORK_EDGES,
  type CatalogParameterValue,
  type CatalogRunOutput
} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  getLineParts,
  getPointPositions,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogDiagnostic,
  type CatalogFeatureCollection,
  type CatalogPosition
} from './catalog-geojson';
import {
  buildNetworkGraph,
  makeOriginLegs,
  makeProjector,
  searchNetwork,
  sliceEdge,
  snapPointToNetwork,
  type NetworkGraph,
  type NetworkSearch,
  type OriginLeg,
  type SnappedPoint
} from './catalog-network';
import {
  collectPositions,
  readPropertyNumber,
  type Inputs,
  type PlanarPoint,
  type Values
} from './catalog-platform3-a-shared';
import {makeLocalFrame} from './catalog-local-frame';

// Match the catalog's declared edge budget, including edges created at shared vertices.
export const MAXIMUM_ROUTING_EDGES = MAXIMUM_NETWORK_EDGES;

export function buildRoutingGraph(
  lines: CatalogFeatureCollection,
  values: Values,
  bindingId = 'network-build-graph'
): NetworkGraph {
  return buildNetworkGraph(lines, getNetworkParameters(values), MAXIMUM_ROUTING_EDGES, bindingId);
}

type Arrival = {cost: number; viaNode: number; forward: boolean} | {cost: number; direct: true};

export type RouteStep = {edge: number; startFraction: number; endFraction: number};

export type Route = {cost: number; steps: RouteStep[]; coordinates: CatalogPosition[]};

function getBestArrival(
  graph: NetworkGraph,
  search: NetworkSearch,
  origin: SnappedPoint,
  destination: SnappedPoint
): Arrival | null {
  const edge = graph.edges[destination.edge];
  let best: Arrival | null = null;
  const consider = (candidate: Arrival) => {
    if (!best || candidate.cost < best.cost) best = candidate;
  };
  if (edge.forwardCost !== null && Number.isFinite(search.distance[edge.fromNode]))
    consider({
      cost: search.distance[edge.fromNode] + destination.fraction * edge.forwardCost,
      viaNode: edge.fromNode,
      forward: true
    });
  if (edge.reverseCost !== null && Number.isFinite(search.distance[edge.toNode]))
    consider({
      cost: search.distance[edge.toNode] + (1 - destination.fraction) * edge.reverseCost,
      viaNode: edge.toNode,
      forward: false
    });
  if (destination.edge === origin.edge) {
    if (destination.fraction >= origin.fraction && edge.forwardCost !== null)
      consider({cost: (destination.fraction - origin.fraction) * edge.forwardCost, direct: true});
    if (destination.fraction < origin.fraction && edge.reverseCost !== null)
      consider({cost: (origin.fraction - destination.fraction) * edge.reverseCost, direct: true});
  }
  return best;
}

/** Least cost from the origin of `search` to a snapped destination, or null when unreachable. */
export function getRouteCost(
  graph: NetworkGraph,
  search: NetworkSearch,
  origin: SnappedPoint,
  destination: SnappedPoint
): number | null {
  return getBestArrival(graph, search, origin, destination)?.cost ?? null;
}

export function traceRoute(
  graph: NetworkGraph,
  search: NetworkSearch,
  legs: readonly OriginLeg[],
  origin: SnappedPoint,
  destination: SnappedPoint
): Route | null {
  const chosen = getBestArrival(graph, search, origin, destination);
  if (!chosen) return null;
  const steps: RouteStep[] = [];
  const coordinates: CatalogPosition[] = [];
  const visit = (edgeIndex: number, startFraction: number, endFraction: number) => {
    if (startFraction === endFraction) return;
    const slice = sliceEdge(
      graph.edges[edgeIndex],
      Math.min(startFraction, endFraction),
      Math.max(startFraction, endFraction)
    );
    for (const position of startFraction <= endFraction ? slice : slice.reverse()) {
      const last = coordinates[coordinates.length - 1];
      if (!last || last[0] !== position[0] || last[1] !== position[1]) coordinates.push(position);
    }
    steps.push({edge: edgeIndex, startFraction, endFraction});
  };
  if ('direct' in chosen) visit(origin.edge, origin.fraction, destination.fraction);
  else {
    const trail: {edge: number; forward: boolean}[] = [];
    let node = chosen.viaNode;
    while (search.seed[node] === -1 && search.parentNode[node] !== -1) {
      trail.push({edge: search.parentEdge[node], forward: search.parentForward[node] === 1});
      node = search.parentNode[node];
    }
    const leg = legs[search.seed[node]];
    visit(leg.edge, leg.startFraction, leg.forward ? 1 : 0);
    for (const step of trail.reverse())
      visit(step.edge, step.forward ? 0 : 1, step.forward ? 1 : 0);
    visit(destination.edge, chosen.forward ? 0 : 1, destination.fraction);
  }
  if (coordinates.length < 2)
    coordinates.splice(0, coordinates.length, origin.position, destination.position);
  return {cost: chosen.cost, steps, coordinates};
}

function getStepLength(graph: NetworkGraph, step: RouteStep): number {
  return Math.abs(step.endFraction - step.startFraction) * graph.edges[step.edge].length;
}

function getStepCost(graph: NetworkGraph, step: RouteStep): number {
  const edge = graph.edges[step.edge];
  const cost = step.endFraction >= step.startFraction ? edge.forwardCost : edge.reverseCost;
  return Math.abs(step.endFraction - step.startFraction) * (cost ?? 0);
}

type SnappedFeature = {
  featureIndex: number;
  position: CatalogPosition;
  snapped: SnappedPoint | null;
};

function snapFeaturePoints(
  graph: NetworkGraph,
  collection: CatalogFeatureCollection,
  maximumDistance: number
): SnappedFeature[] {
  const result: SnappedFeature[] = [];
  collection.features.forEach((feature, featureIndex) => {
    for (const position of getPointPositions(feature.geometry))
      result.push({
        featureIndex,
        position,
        snapped: snapPointToNetwork(graph, position, maximumDistance)
      });
  });
  return result;
}

function searchFrom(graph: NetworkGraph, snapped: SnappedPoint) {
  const legs = makeOriginLegs(graph, snapped);
  return {
    legs,
    search: searchNetwork(
      graph,
      legs.map(leg => ({node: leg.node, cost: leg.cost}))
    )
  };
}

function makeEdgeLineText(graph: NetworkGraph, edgeIndex: number): string {
  return makeCatalogGeometryText('LineString', graph.edges[edgeIndex].coordinates);
}

// ---------------------------------------------------------------------------------------------
// Accessibility
// ---------------------------------------------------------------------------------------------

export function runNetworkAccessibility(inputs: Inputs, values: Values): CatalogRunOutput {
  const graph = buildRoutingGraph(inputs['lines'], values, 'network-accessibility');
  const diagnostics: CatalogDiagnostic[] = [...graph.diagnostics];
  const maximumSnap = values['maximumSnapDistance'] as number;
  const maximumCost = values['maximumCost'] as number;
  const scale = values['decayScale'] as number;
  const decay = values['decay'] as string;
  const weightProperty = values['weightProperty'] as string;
  const getDecay = (cost: number): number => {
    if (cost > maximumCost) return 0;
    if (decay === 'linear') return 1 - cost / maximumCost;
    if (decay === 'exponential') return Math.exp(-cost / scale);
    if (decay === 'gaussian') return Math.exp(-(cost * cost) / (2 * scale * scale));
    return 1;
  };
  // Read and validate weights once at ingestion, independently of reachability or origin count.
  const weights = inputs['opportunities'].features.map(feature => {
    const weight = weightProperty ? readPropertyNumber(feature, weightProperty) : 1;
    validateRequirement(
      'network-accessibility',
      weight === null || weight >= 0,
      `weightProperty "${weightProperty}" must be at least 0; correct negative opportunity weights`
    );
    return weight;
  });
  const opportunities = snapFeaturePoints(graph, inputs['opportunities'], maximumSnap);
  const missingWeights = opportunities.filter(item => weights[item.featureIndex] === null).length;
  if (missingWeights)
    addCatalogDiagnostic(
      diagnostics,
      'opportunity-weight-missing',
      'warning',
      'Opportunities with missing or non-numeric weightProperty values are excluded from reachable_count and accessibility.',
      missingWeights
    );
  const unsnappedOpportunities = opportunities.filter(item => !item.snapped).length;
  if (unsnappedOpportunities)
    addCatalogDiagnostic(
      diagnostics,
      'opportunity-not-snapped',
      'warning',
      'Opportunities farther than the maximum snap distance from the network are ignored.',
      unsnappedOpportunities
    );
  const origins = snapFeaturePoints(graph, inputs['origins'], maximumSnap);
  const table = makeCatalogTable(
    'accessibility',
    'Network accessibility',
    {
      origin_index: 'int32',
      origin_id: 'string',
      point_index: 'int32',
      status: 'string',
      reachable_count: 'int32',
      accessibility: 'float64',
      nearest_cost: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  origins.forEach((origin, pointIndex) => {
    const base = {
      origin_index: origin.featureIndex,
      origin_id: getFeatureIdentity(inputs['origins'].features[origin.featureIndex]),
      point_index: pointIndex,
      geometry: makeCatalogGeometryText('Point', origin.position)
    };
    if (!origin.snapped) {
      appendCatalogRow(table, {...base, status: 'origin-not-snapped'});
      return;
    }
    const {search} = searchFrom(graph, origin.snapped);
    let reachable = 0;
    let score = 0;
    let nearest = Infinity;
    for (const opportunity of opportunities) {
      const weight = weights[opportunity.featureIndex];
      if (!opportunity.snapped || weight === null) continue;
      const cost = getRouteCost(graph, search, origin.snapped, opportunity.snapped);
      if (cost === null || cost > maximumCost) continue;
      reachable++;
      score += weight * getDecay(cost);
      nearest = Math.min(nearest, cost);
    }
    appendCatalogRow(table, {
      ...base,
      status: 'ok',
      reachable_count: reachable,
      accessibility: score,
      nearest_cost: Number.isFinite(nearest) ? nearest : null
    });
  });
  addCatalogDiagnostic(
    diagnostics,
    'cost-units',
    'info',
    'Costs follow the declared arc costs (length in meters or coordinate units, or the cost properties); the decay is evaluated at the least network cost from each origin.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {
      originCount: origins.length,
      opportunityCount: opportunities.length,
      edgeCount: graph.edges.length
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Subgraph filter
// ---------------------------------------------------------------------------------------------

export function runNetworkSubgraphFilter(inputs: Inputs, values: Values): CatalogRunOutput {
  const graph = buildRoutingGraph(inputs['lines'], values, 'network-subgraph-filter');
  const diagnostics: CatalogDiagnostic[] = [...graph.diagnostics];
  const minimumCost = values['minimumCost'] as number;
  const maximumCost = values['maximumCost'] as number;
  const propertyName = values['propertyName'] as string;
  const propertyValue = values['propertyValue'] as string;
  const largestOnly = values['largestComponentOnly'] as boolean;
  const keptByRule = graph.edges.map(edge => {
    const cost = edge.forwardCost ?? edge.reverseCost ?? 0;
    if (cost < minimumCost || cost > maximumCost) return false;
    if (propertyName) {
      const value = inputs['lines'].features[edge.sourceIndex].properties?.[propertyName];
      if (String(value ?? '') !== propertyValue) return false;
    }
    return true;
  });
  // Connected components of the retained edges (undirected).
  const parent = graph.nodes.map((_, index) => index);
  const findRoot = (node: number): number => {
    while (parent[node] !== node) {
      parent[node] = parent[parent[node]];
      node = parent[node];
    }
    return node;
  };
  graph.edges.forEach((edge, edgeIndex) => {
    if (keptByRule[edgeIndex]) parent[findRoot(edge.fromNode)] = findRoot(edge.toNode);
  });
  const componentLength = new Map<number, number>();
  graph.edges.forEach((edge, edgeIndex) => {
    if (!keptByRule[edgeIndex]) return;
    const root = findRoot(edge.fromNode);
    componentLength.set(root, (componentLength.get(root) ?? 0) + edge.length);
  });
  let largestRoot = -1;
  let largestLength = -1;
  for (const [root, length] of [...componentLength.entries()].sort(
    ([first], [second]) => first - second
  ))
    if (length > largestLength) {
      largestLength = length;
      largestRoot = root;
    }
  const table = makeCatalogTable(
    'subgraph-edges',
    'Retained network edges',
    {
      edge_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      from_node: 'int32',
      to_node: 'int32',
      component: 'int32',
      length: 'float64',
      forward_cost: 'float64',
      reverse_cost: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  let removedCount = 0;
  graph.edges.forEach((edge, edgeIndex) => {
    const root = findRoot(edge.fromNode);
    if (!keptByRule[edgeIndex] || (largestOnly && root !== largestRoot)) {
      removedCount++;
      return;
    }
    appendCatalogRow(table, {
      edge_id: edge.edgeId,
      source_index: edge.sourceIndex,
      source_id: edge.sourceId,
      from_node: edge.fromNode,
      to_node: edge.toNode,
      component: root,
      length: edge.length,
      forward_cost: edge.forwardCost,
      reverse_cost: edge.reverseCost,
      geometry: makeEdgeLineText(graph, edgeIndex)
    });
  });
  if (removedCount)
    addCatalogDiagnostic(
      diagnostics,
      'edges-removed',
      'info',
      'Edges removed by the filter rules.',
      removedCount
    );
  return {
    tables: [table],
    diagnostics,
    summary: {keptCount: table.rowCount, removedCount, componentCount: componentLength.size}
  };
}

// ---------------------------------------------------------------------------------------------
// Coarsening (degree-two chain merging)
// ---------------------------------------------------------------------------------------------

export function runNetworkCoarsening(inputs: Inputs, values: Values): CatalogRunOutput {
  const graph = buildRoutingGraph(inputs['lines'], values, 'network-coarsening');
  const diagnostics: CatalogDiagnostic[] = [...graph.diagnostics];
  const incident: number[][] = graph.nodes.map(() => []);
  graph.edges.forEach((edge, edgeIndex) => {
    incident[edge.fromNode].push(edgeIndex);
    incident[edge.toNode].push(edgeIndex);
  });
  const visited = new Uint8Array(graph.edges.length);
  const table = makeCatalogTable(
    'coarse-edges',
    'Coarsened edges',
    {
      coarse_id: 'string',
      from_node: 'int32',
      to_node: 'int32',
      edge_count: 'int32',
      source_indices: 'string',
      length: 'float64',
      forward_cost: 'float64',
      reverse_cost: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const walk = (startNode: number, firstEdge: number) => {
    let node = startNode;
    let edgeIndex = firstEdge;
    const coordinates: CatalogPosition[] = [];
    const sources: number[] = [];
    let length = 0;
    let forwardCost: number | null = 0;
    let reverseCost: number | null = 0;
    let count = 0;
    for (;;) {
      visited[edgeIndex] = 1;
      const edge = graph.edges[edgeIndex];
      const forward = edge.fromNode === node;
      const ordered = forward ? edge.coordinates : [...edge.coordinates].reverse();
      coordinates.push(...(coordinates.length ? ordered.slice(1) : ordered));
      sources.push(edge.sourceIndex);
      length += edge.length;
      const costAlong = forward ? edge.forwardCost : edge.reverseCost;
      const costBack = forward ? edge.reverseCost : edge.forwardCost;
      forwardCost = forwardCost === null || costAlong === null ? null : forwardCost + costAlong;
      reverseCost = reverseCost === null || costBack === null ? null : reverseCost + costBack;
      count++;
      node = forward ? edge.toNode : edge.fromNode;
      const next =
        incident[node].length === 2
          ? incident[node].find(candidate => !visited[candidate])
          : undefined;
      if (next === undefined || node === startNode) break;
      edgeIndex = next;
    }
    appendCatalogRow(table, {
      coarse_id: `${table.rowCount}`,
      from_node: startNode,
      to_node: node,
      edge_count: count,
      source_indices: [...new Set(sources)].join(','),
      length,
      forward_cost: forwardCost,
      reverse_cost: reverseCost,
      geometry: makeCatalogGeometryText('LineString', coordinates)
    });
  };
  graph.nodes.forEach((_, node) => {
    if (incident[node].length === 2) return;
    for (const edgeIndex of incident[node]) if (!visited[edgeIndex]) walk(node, edgeIndex);
  });
  graph.edges.forEach((edge, edgeIndex) => {
    if (!visited[edgeIndex]) walk(edge.fromNode, edgeIndex);
  });
  addCatalogDiagnostic(
    diagnostics,
    'attributes-not-compared',
    'info',
    'Chains merge at every node with exactly two incident edges regardless of source feature or attributes; costs are summed per direction and become null when any member is not traversable that way.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {
      coarseEdgeCount: table.rowCount,
      originalEdgeCount: graph.edges.length,
      nodeCount: graph.nodes.length
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Flow aggregation
// ---------------------------------------------------------------------------------------------

export function runNetworkFlowAggregation(inputs: Inputs, values: Values): CatalogRunOutput {
  const graph = buildRoutingGraph(inputs['lines'], values, 'network-flow-aggregation');
  const diagnostics: CatalogDiagnostic[] = [...graph.diagnostics];
  const maximumSnap = values['maximumSnapDistance'] as number;
  const weightProperty = values['weightProperty'] as string;
  const origins = inputs['origins'].features;
  const destinations = inputs['destinations'].features;
  const pairCount = Math.min(origins.length, destinations.length);
  if (origins.length !== destinations.length)
    addCatalogDiagnostic(
      diagnostics,
      'unpaired-features',
      'warning',
      'Origins and destinations pair by feature order; surplus features are ignored.',
      Math.abs(origins.length - destinations.length)
    );
  const flow = new Float64Array(graph.edges.length);
  const flowCounts = new Int32Array(graph.edges.length);
  const pairTable = makeCatalogTable('flow-pairs', 'Routed origin-destination pairs', {
    pair_index: 'int32',
    origin_id: 'string',
    destination_id: 'string',
    weight: 'float64',
    status: 'string',
    cost: 'float64',
    edge_count: 'int32'
  });
  const searches = new Map<string, ReturnType<typeof searchFrom>>();
  for (let pairIndex = 0; pairIndex < pairCount; pairIndex++) {
    const originPosition = getPointPositions(origins[pairIndex].geometry)[0];
    const destinationPosition = getPointPositions(destinations[pairIndex].geometry)[0];
    const weight = weightProperty ? readPropertyNumber(origins[pairIndex], weightProperty) : 1;
    const base = {
      pair_index: pairIndex,
      origin_id: getFeatureIdentity(origins[pairIndex]),
      destination_id: getFeatureIdentity(destinations[pairIndex]),
      weight
    };
    if (!originPosition || !destinationPosition || weight === null || weight < 0) {
      appendCatalogRow(pairTable, {...base, status: 'invalid-pair'});
      continue;
    }
    const origin = snapPointToNetwork(graph, originPosition, maximumSnap);
    const destination = snapPointToNetwork(graph, destinationPosition, maximumSnap);
    if (!origin || !destination) {
      appendCatalogRow(pairTable, {
        ...base,
        status: origin ? 'destination-not-snapped' : 'origin-not-snapped'
      });
      continue;
    }
    const key = `${origin.edge}:${origin.fraction}`;
    if (!searches.has(key)) searches.set(key, searchFrom(graph, origin));
    const {legs, search} = searches.get(key)!;
    const route = traceRoute(graph, search, legs, origin, destination);
    if (!route) {
      appendCatalogRow(pairTable, {...base, status: 'unreachable'});
      continue;
    }
    const used = new Set(route.steps.map(step => step.edge));
    for (const edgeIndex of used) {
      flow[edgeIndex] += weight;
      flowCounts[edgeIndex]++;
    }
    appendCatalogRow(pairTable, {...base, status: 'ok', cost: route.cost, edge_count: used.size});
  }
  const edgeTable = makeCatalogTable(
    'flow-edges',
    'Edge flows',
    {
      edge_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      flow: 'float64',
      pair_count: 'int32',
      length: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  graph.edges.forEach((edge, edgeIndex) => {
    if (!flowCounts[edgeIndex]) return;
    appendCatalogRow(edgeTable, {
      edge_id: edge.edgeId,
      source_index: edge.sourceIndex,
      source_id: edge.sourceId,
      flow: flow[edgeIndex],
      pair_count: flowCounts[edgeIndex],
      length: edge.length,
      geometry: makeEdgeLineText(graph, edgeIndex)
    });
  });
  addCatalogDiagnostic(
    diagnostics,
    'whole-edge-flow',
    'info',
    'Every edge a route touches, including the partial edges at the snapped origin and destination, receives the full weight once.'
  );
  const failed = pairTable.columns['status'].filter(status => status !== 'ok').length;
  if (failed)
    addCatalogDiagnostic(
      diagnostics,
      'pairs-not-routed',
      'warning',
      'Pairs that could not be routed carry no flow.',
      failed
    );
  return {
    tables: [edgeTable, pairTable],
    diagnostics,
    summary: {
      pairCount,
      routedCount: pairCount - failed,
      loadedEdgeCount: edgeTable.rowCount,
      searchCount: searches.size
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Alternative routes (penalty method)
// ---------------------------------------------------------------------------------------------

export function runAlternativeRoutes(inputs: Inputs, values: Values): CatalogRunOutput {
  const baseGraph = buildRoutingGraph(inputs['lines'], values, 'network-alternative-routes');
  const diagnostics: CatalogDiagnostic[] = [...baseGraph.diagnostics];
  const maximumSnap = values['maximumSnapDistance'] as number;
  const routeCount = values['routeCount'] as number;
  const penalty = values['penalty'] as number;
  const maximumOverlap = values['maximumOverlap'] as number;
  const originPosition = getPointPositions(inputs['origin'].features[0]?.geometry ?? null)[0];
  const destinationPosition = getPointPositions(
    inputs['destination'].features[0]?.geometry ?? null
  )[0];
  // One origin and one destination point required.
  if (
    inputs['origin'].features.length !== 1 ||
    inputs['destination'].features.length !== 1 ||
    !originPosition ||
    !destinationPosition
  )
    throw new Error('One origin and one destination point required');
  const table = makeCatalogTable(
    'alternative-routes',
    'Alternative routes',
    {
      route_rank: 'int32',
      cost: 'float64',
      length: 'float64',
      detour_ratio: 'float64',
      overlap_with_best: 'float64',
      edge_count: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  const origin = snapPointToNetwork(baseGraph, originPosition, maximumSnap);
  const destination = snapPointToNetwork(baseGraph, destinationPosition, maximumSnap);
  if (!origin || !destination) {
    addCatalogDiagnostic(
      diagnostics,
      origin ? 'destination-not-snapped' : 'origin-not-snapped',
      'warning',
      'The point is farther than the maximum snap distance from the network; no route is produced.'
    );
    return {tables: [table], diagnostics, summary: {routeCount: 0}};
  }
  const multiplier = new Float64Array(baseGraph.edges.length).fill(1);
  const accepted: Route[] = [];
  const acceptedEdgeSets: Set<number>[] = [];
  for (let attempt = 0; attempt < routeCount * 4 && accepted.length < routeCount; attempt++) {
    const penalized: NetworkGraph = {
      ...baseGraph,
      edges: baseGraph.edges.map((edge, edgeIndex) => ({
        ...edge,
        forwardCost: edge.forwardCost === null ? null : edge.forwardCost * multiplier[edgeIndex],
        reverseCost: edge.reverseCost === null ? null : edge.reverseCost * multiplier[edgeIndex]
      })),
      outgoing: baseGraph.outgoing.map(arcs =>
        arcs.map(arc => ({...arc, cost: arc.cost * multiplier[arc.edge]}))
      )
    };
    const {legs, search} = searchFrom(penalized, origin);
    const route = traceRoute(penalized, search, legs, origin, destination);
    if (!route) break;
    const edges = new Set(route.steps.map(step => step.edge));
    const trueCost = route.steps.reduce((sum, step) => sum + getStepCost(baseGraph, step), 0);
    const routeLength = route.steps.reduce((sum, step) => sum + getStepLength(baseGraph, step), 0);
    const overlapWith = (other: Set<number>) =>
      routeLength > 0
        ? route.steps
            .filter(step => other.has(step.edge))
            .reduce((sum, step) => sum + getStepLength(baseGraph, step), 0) / routeLength
        : 1;
    const isDuplicate = acceptedEdgeSets.some(other => overlapWith(other) > maximumOverlap);
    for (const edgeIndex of edges) multiplier[edgeIndex] *= 1 + penalty;
    if (isDuplicate) continue;
    accepted.push({...route, cost: trueCost});
    acceptedEdgeSets.push(edges);
  }
  const bestCost = accepted[0]?.cost ?? 0;
  accepted.forEach((route, rank) => {
    const length = route.steps.reduce((sum, step) => sum + getStepLength(baseGraph, step), 0);
    const best = acceptedEdgeSets[0];
    const shared = route.steps
      .filter(step => best.has(step.edge))
      .reduce((sum, step) => sum + getStepLength(baseGraph, step), 0);
    appendCatalogRow(table, {
      route_rank: rank,
      cost: route.cost,
      length,
      detour_ratio: bestCost > 0 ? route.cost / bestCost : 1,
      overlap_with_best: length > 0 ? shared / length : 1,
      edge_count: acceptedEdgeSets[rank].size,
      geometry: makeCatalogGeometryText('LineString', route.coordinates)
    });
  });
  if (accepted.length < routeCount)
    addCatalogDiagnostic(
      diagnostics,
      'fewer-routes-than-requested',
      'info',
      'Fewer distinct alternatives than requested were found within the attempt budget.'
    );
  return {tables: [table], diagnostics, summary: {routeCount: accepted.length}};
}

// ---------------------------------------------------------------------------------------------
// Map matching (HMM with Viterbi)
// ---------------------------------------------------------------------------------------------

type MatchCandidate = {snapped: SnappedPoint; emission: number};

export function runMapMatching(inputs: Inputs, values: Values): CatalogRunOutput {
  const graph = buildRoutingGraph(inputs['lines'], values, 'network-map-matching');
  const diagnostics: CatalogDiagnostic[] = [...graph.diagnostics];
  const searchRadius = values['searchRadius'] as number;
  const candidateCount = values['candidatesPerPoint'] as number;
  const sigma = values['sigma'] as number;
  const beta = values['beta'] as number;
  const trace: {featureIndex: number; position: CatalogPosition}[] = [];
  inputs['trace'].features.forEach((feature, featureIndex) => {
    const positions = [
      ...getPointPositions(feature.geometry),
      ...getLineParts(feature.geometry).flat()
    ];
    for (const position of positions) trace.push({featureIndex, position});
  });
  // Trace point capacity exceeded.
  if (trace.length > 500)
    throw new KeplerAllocationError('capacity', {
      message:
        'network-map-matching trace point count must be at most 500; split or reduce the trace'
    });
  const project = makeProjector(graph.coordinateSystem, graph.nodes);
  const getCandidates = (position: CatalogPosition): MatchCandidate[] => {
    const [x, y] = project(position);
    const candidates: MatchCandidate[] = [];
    graph.edges.forEach((edge, edgeIndex) => {
      let best: SnappedPoint | null = null;
      for (let index = 1; index < edge.coordinates.length; index++) {
        const [ax, ay] = project(edge.coordinates[index - 1]);
        const [bx, by] = project(edge.coordinates[index]);
        const lengthSquared = (bx - ax) ** 2 + (by - ay) ** 2;
        const u =
          lengthSquared === 0
            ? 0
            : Math.max(
                0,
                Math.min(1, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / lengthSquared)
              );
        const distance = Math.hypot(x - (ax + u * (bx - ax)), y - (ay + u * (by - ay)));
        if (distance <= searchRadius && (!best || distance < best.distance)) {
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
      if (best)
        candidates.push({snapped: best, emission: -(best.distance ** 2) / (2 * sigma * sigma)});
    });
    return candidates
      .sort(
        (first, second) =>
          first.snapped.distance - second.snapped.distance ||
          first.snapped.edge - second.snapped.edge
      )
      .slice(0, candidateCount);
  };
  const candidates = trace.map(item => getCandidates(item.position));
  // Viterbi with chain breaks where no transition is possible.
  const score: number[][] = [];
  const back: number[][] = [];
  const segmentStart: boolean[] = [];
  const projectedTrace = trace.map(item => project(item.position));
  for (let step = 0; step < trace.length; step++) {
    score.push(new Array(candidates[step].length).fill(-Infinity));
    back.push(new Array(candidates[step].length).fill(-1));
    segmentStart.push(true);
    if (!candidates[step].length) continue;
    const previous = step > 0 ? candidates[step - 1] : [];
    let linked = false;
    if (previous.length && score[step - 1].some(Number.isFinite)) {
      const lineDistance = Math.hypot(
        projectedTrace[step][0] - projectedTrace[step - 1][0],
        projectedTrace[step][1] - projectedTrace[step - 1][1]
      );
      previous.forEach((from, fromIndex) => {
        if (!Number.isFinite(score[step - 1][fromIndex])) return;
        const {search} = searchFrom(graph, from.snapped);
        candidates[step].forEach((to, toIndex) => {
          const cost = getRouteCost(graph, search, from.snapped, to.snapped);
          if (cost === null) return;
          const total =
            score[step - 1][fromIndex] + to.emission - Math.abs(cost - lineDistance) / beta;
          if (total > score[step][toIndex]) {
            score[step][toIndex] = total;
            back[step][toIndex] = fromIndex;
            linked = true;
          }
        });
      });
    }
    if (linked) segmentStart[step] = false;
    else candidates[step].forEach((candidate, index) => (score[step][index] = candidate.emission));
  }
  // Backtrack per chain.
  const chosen = new Array<number>(trace.length).fill(-1);
  for (let step = trace.length - 1; step >= 0; ) {
    if (!candidates[step].length) {
      step--;
      continue;
    }
    let index = score[step].indexOf(Math.max(...score[step]));
    let current = step;
    for (;;) {
      chosen[current] = index;
      if (segmentStart[current] || back[current][index] < 0) break;
      index = back[current][index];
      current--;
    }
    step = current - 1;
  }
  const pointTable = makeCatalogTable(
    'matched-points',
    'Matched points',
    {
      point_index: 'int32',
      source_index: 'int32',
      status: 'string',
      route_segment: 'int32',
      edge_id: 'string',
      edge_source_id: 'string',
      fraction: 'float64',
      distance: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  let segmentIndex = -1;
  const routeCoordinates: CatalogPosition[][] = [];
  let previousSnapped: SnappedPoint | null = null;
  trace.forEach((item, step) => {
    const index = chosen[step];
    if (index < 0) {
      appendCatalogRow(pointTable, {
        point_index: step,
        source_index: item.featureIndex,
        status: 'unmatched',
        geometry: makeCatalogGeometryText('Point', item.position)
      });
      previousSnapped = null;
      return;
    }
    const snapped = candidates[step][index].snapped;
    if (segmentStart[step] || !previousSnapped) {
      segmentIndex++;
      routeCoordinates.push([]);
    }
    if (previousSnapped && !segmentStart[step]) {
      const {legs, search} = searchFrom(graph, previousSnapped);
      const route = traceRoute(graph, search, legs, previousSnapped, snapped);
      if (route) routeCoordinates[segmentIndex].push(...route.coordinates);
    } else routeCoordinates[segmentIndex].push(snapped.position);
    previousSnapped = snapped;
    const edge = graph.edges[snapped.edge];
    appendCatalogRow(pointTable, {
      point_index: step,
      source_index: item.featureIndex,
      status: 'matched',
      route_segment: segmentIndex,
      edge_id: edge.edgeId,
      edge_source_id: edge.sourceId,
      fraction: snapped.fraction,
      distance: snapped.distance,
      geometry: makeCatalogGeometryText('Point', snapped.position)
    });
  });
  const routeTable = makeCatalogTable(
    'matched-route',
    'Matched route',
    {route_segment: 'int32', vertex_count: 'int32', geometry: 'string'},
    'geometry'
  );
  routeCoordinates.forEach((coordinates, index) => {
    const deduplicated = coordinates.filter(
      (position, positionIndex) =>
        positionIndex === 0 ||
        position[0] !== coordinates[positionIndex - 1][0] ||
        position[1] !== coordinates[positionIndex - 1][1]
    );
    if (deduplicated.length < 2) return;
    appendCatalogRow(routeTable, {
      route_segment: index,
      vertex_count: deduplicated.length,
      geometry: makeCatalogGeometryText('LineString', deduplicated)
    });
  });
  const unmatched = pointTable.columns['status'].filter(status => status === 'unmatched').length;
  if (unmatched)
    addCatalogDiagnostic(
      diagnostics,
      'trace-point-unmatched',
      'warning',
      'Trace points with no network edge within the search radius are left unmatched and break the route.',
      unmatched
    );
  if (segmentIndex > 0)
    addCatalogDiagnostic(
      diagnostics,
      'route-broken',
      'info',
      'The trace was split into separate route segments where no connected transition existed.',
      segmentIndex
    );
  return {
    tables: [pointTable, routeTable],
    diagnostics,
    summary: {
      traceCount: trace.length,
      matchedCount: trace.length - unmatched,
      segmentCount: routeTable.rowCount
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Network K function
// ---------------------------------------------------------------------------------------------

export function runNetworkKFunction(inputs: Inputs, values: Values): CatalogRunOutput {
  validateRequirement(
    'network-k-function',
    values['costMode'] === 'length',
    'costMode must be length; property-cost radii are unsupported by the length-based K normalization; use length mode'
  );
  const graph = buildRoutingGraph(inputs['lines'], values, 'network-k-function');
  const diagnostics: CatalogDiagnostic[] = [...graph.diagnostics];
  const radii = [...(values['radii'] as readonly number[])].sort((first, second) => first - second);
  const snapped = snapFeaturePoints(
    graph,
    inputs['points'],
    values['maximumSnapDistance'] as number
  )
    .map(item => item.snapped)
    .filter((item): item is SnappedPoint => item !== null);
  const total = graph.edges.reduce((sum, edge) => sum + edge.length, 0);
  const counts = new Array<number>(radii.length).fill(0);
  snapped.forEach((from, fromIndex) => {
    const {search} = searchFrom(graph, from);
    snapped.forEach((to, toIndex) => {
      if (fromIndex === toIndex) return;
      const cost = getRouteCost(graph, search, from, to);
      if (cost === null) return;
      radii.forEach((radius, radiusIndex) => {
        if (cost <= radius) counts[radiusIndex]++;
      });
    });
  });
  const table = makeCatalogTable('network-k', 'Network K function', {
    radius: 'float64',
    pair_count: 'int32',
    k: 'float64',
    expected_k: 'float64',
    excess: 'float64'
  });
  const n = snapped.length;
  radii.forEach((radius, index) => {
    const k = n > 1 ? (total * counts[index]) / (n * (n - 1)) : null;
    appendCatalogRow(table, {
      radius,
      pair_count: counts[index],
      k,
      expected_k: 2 * radius,
      excess: k === null ? null : k - 2 * radius
    });
  });
  if (n < 2)
    addCatalogDiagnostic(
      diagnostics,
      'too-few-points',
      'warning',
      'Fewer than two points snapped to the network; K is undefined.'
    );
  addCatalogDiagnostic(
    diagnostics,
    'edge-effects',
    'info',
    'K(r) = L / (n (n - 1)) times the ordered pairs within network cost r, with L the total edge length. The reference 2r holds for an unbounded line; no edge correction or envelope is computed.'
  );
  return {tables: [table], diagnostics, summary: {pointCount: n, networkLength: total}};
}

// ---------------------------------------------------------------------------------------------
// Edge bundling (force-directed, FDEB)
// ---------------------------------------------------------------------------------------------

export function runEdgeBundling(
  lines: CatalogFeatureCollection,
  values: Readonly<Record<string, CatalogParameterValue>>
): CatalogRunOutput {
  const cycles = values['cycles'] as number;
  const iterations = values['iterations'] as number;
  const stiffness = values['stiffness'] as number;
  const threshold = values['compatibilityThreshold'] as number;
  const frame = makeLocalFrame(
    values['coordinateSystem'] as 'EPSG:4326' | 'planar',
    collectPositions(lines)
  );
  type Edge = {featureIndex: number; points: PlanarPoint[]};
  const edges: Edge[] = [];
  lines.features.forEach((feature, featureIndex) => {
    const parts = getLineParts(feature.geometry);
    const positions = parts[0] ?? [];
    if (positions.length < 2) return;
    const start = frame.forward(positions[0]);
    const end = frame.forward(positions[positions.length - 1]);
    if (start[0] === end[0] && start[1] === end[1]) return;
    edges.push({featureIndex, points: [start, end]});
  });
  const dot = (first: PlanarPoint, second: PlanarPoint) =>
    first[0] * second[0] + first[1] * second[1];
  const getLength = (edge: Edge) =>
    Math.hypot(
      edge.points[edge.points.length - 1][0] - edge.points[0][0],
      edge.points[edge.points.length - 1][1] - edge.points[0][1]
    );
  const chord = (edge: Edge): PlanarPoint => [
    edge.points[edge.points.length - 1][0] - edge.points[0][0],
    edge.points[edge.points.length - 1][1] - edge.points[0][1]
  ];
  const compatible: number[][] = edges.map(() => []);
  const compatibilityWeight: number[][] = edges.map(() => []);
  for (let first = 0; first < edges.length; first++)
    for (let second = first + 1; second < edges.length; second++) {
      const a = edges[first];
      const b = edges[second];
      const lengthA = getLength(a);
      const lengthB = getLength(b);
      const angle = Math.abs(dot(chord(a), chord(b))) / (lengthA * lengthB);
      const average = (lengthA + lengthB) / 2;
      const scale =
        2 / (average / Math.min(lengthA, lengthB) + Math.max(lengthA, lengthB) / average);
      const midpointA: PlanarPoint = [
        (a.points[0][0] + a.points[a.points.length - 1][0]) / 2,
        (a.points[0][1] + a.points[a.points.length - 1][1]) / 2
      ];
      const midpointB: PlanarPoint = [
        (b.points[0][0] + b.points[b.points.length - 1][0]) / 2,
        (b.points[0][1] + b.points[b.points.length - 1][1]) / 2
      ];
      const position =
        average / (average + Math.hypot(midpointA[0] - midpointB[0], midpointA[1] - midpointB[1]));
      const total = angle * scale * position;
      if (total >= threshold) {
        compatible[first].push(second);
        compatible[second].push(first);
        compatibilityWeight[first].push(total);
        compatibilityWeight[second].push(total);
      }
    }
  let segmentCount = 1;
  let stepSize = 0.04;
  let iterationCount = iterations;
  for (let cycle = 0; cycle < cycles; cycle++) {
    segmentCount *= 2;
    for (const edge of edges) {
      const dense: PlanarPoint[] = [];
      // Resample the current polyline into segmentCount + 1 equally spaced points.
      const cumulative = [0];
      for (let index = 1; index < edge.points.length; index++)
        cumulative.push(
          cumulative[index - 1] +
            Math.hypot(
              edge.points[index][0] - edge.points[index - 1][0],
              edge.points[index][1] - edge.points[index - 1][1]
            )
        );
      const total = cumulative[cumulative.length - 1];
      for (let index = 0; index <= segmentCount; index++) {
        const target = (total * index) / segmentCount;
        let segment = 1;
        while (segment < cumulative.length - 1 && cumulative[segment] < target) segment++;
        const span = cumulative[segment] - cumulative[segment - 1];
        const fraction = span > 0 ? (target - cumulative[segment - 1]) / span : 0;
        dense.push([
          edge.points[segment - 1][0] +
            (edge.points[segment][0] - edge.points[segment - 1][0]) * fraction,
          edge.points[segment - 1][1] +
            (edge.points[segment][1] - edge.points[segment - 1][1]) * fraction
        ]);
      }
      edge.points = dense;
    }
    for (let iteration = 0; iteration < iterationCount; iteration++) {
      const next = edges.map((edge, edgeIndex) => {
        const edgeLength = getLength(edge);
        const kappa = stiffness / edgeLength;
        return edge.points.map((point, pointIndex): PlanarPoint => {
          if (pointIndex === 0 || pointIndex === edge.points.length - 1) return point;
          const springX =
            kappa *
            (edge.points[pointIndex - 1][0] + edge.points[pointIndex + 1][0] - 2 * point[0]) *
            segmentCount;
          const springY =
            kappa *
            (edge.points[pointIndex - 1][1] + edge.points[pointIndex + 1][1] - 2 * point[1]) *
            segmentCount;
          let forceX = 0;
          let forceY = 0;
          compatible[edgeIndex].forEach((otherIndex, listIndex) => {
            const other = edges[otherIndex].points[pointIndex];
            const deltaX = other[0] - point[0];
            const deltaY = other[1] - point[1];
            const distance = Math.hypot(deltaX, deltaY);
            if (distance < 1e-12) return;
            const weight = compatibilityWeight[edgeIndex][listIndex] / distance;
            forceX += weight * deltaX;
            forceY += weight * deltaY;
          });
          return [
            point[0] + stepSize * edgeLength * (springX + forceX),
            point[1] + stepSize * edgeLength * (springY + forceY)
          ];
        });
      });
      edges.forEach((edge, edgeIndex) => (edge.points = next[edgeIndex]));
    }
    stepSize /= 2;
    iterationCount = Math.max(1, Math.round((iterationCount * 2) / 3));
  }
  const table = makeCatalogTable(
    'bundled-edges',
    'Bundled edges',
    {
      source_index: 'int32',
      source_id: 'string',
      compatible_count: 'int32',
      straightness: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  edges.forEach((edge, edgeIndex) => {
    let length = 0;
    for (let index = 1; index < edge.points.length; index++)
      length += Math.hypot(
        edge.points[index][0] - edge.points[index - 1][0],
        edge.points[index][1] - edge.points[index - 1][1]
      );
    appendCatalogRow(table, {
      source_index: edge.featureIndex,
      source_id: getFeatureIdentity(lines.features[edge.featureIndex]),
      compatible_count: compatible[edgeIndex].length,
      straightness: length > 0 ? getLength(edge) / length : null,
      geometry: makeCatalogGeometryText(
        'LineString',
        edge.points.map(point => frame.inverse(point[0], point[1]))
      )
    });
  });
  return {
    tables: [table],
    diagnostics: [
      {
        code: 'first-last-vertices',
        severity: 'info',
        count: 1,
        message:
          'Each line is bundled as the edge between its first and last vertex (force-directed edge bundling); intermediate vertices are dropped and lines with identical ends are skipped.'
      }
    ],
    summary: {edgeCount: edges.length, subdivisionCount: segmentCount}
  };
}
