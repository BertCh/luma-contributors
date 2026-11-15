// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
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
  searchNetwork,
  snapPointToNetwork,
  type CatalogNetworkParameters,
  type NetworkGraph,
  type SnappedPoint
} from './catalog-network';
import type {ExtraRunOutput} from './catalog-extra-geometry';

function collectPoints(collection: CatalogFeatureCollection) {
  return collection.features.flatMap((feature, featureIndex) =>
    getPointPositions(feature.geometry).map(position => ({feature, featureIndex, position}))
  );
}

/** Undirected incidence count per node; a self-loop counts twice. */
function countNodeDegrees(graph: NetworkGraph): Int32Array {
  const degrees = new Int32Array(graph.nodes.length);
  for (const edge of graph.edges) {
    degrees[edge.fromNode]++;
    degrees[edge.toNode]++;
  }
  return degrees;
}

export function runNetworkStructure(
  lines: CatalogFeatureCollection,
  parameters: CatalogNetworkParameters,
  maximumEdges: number
): ExtraRunOutput {
  const graph = buildNetworkGraph(lines, parameters, maximumEdges, 'network-structure');
  const degrees = countNodeDegrees(graph);
  const table = makeCatalogTable(
    'network-structure',
    'Network nodes',
    {
      node_id: 'int32',
      degree: 'int32',
      component: 'int32',
      node_kind: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  const counts = {isolated: 0, deadEnd: 0, passThrough: 0, junction: 0};
  graph.nodes.forEach((position, node) => {
    const degree = degrees[node];
    const kind =
      degree === 0
        ? 'isolated'
        : degree === 1
          ? 'dead-end'
          : degree === 2
            ? 'pass-through'
            : 'junction';
    if (kind === 'isolated') counts.isolated++;
    else if (kind === 'dead-end') counts.deadEnd++;
    else if (kind === 'pass-through') counts.passThrough++;
    else counts.junction++;
    appendCatalogRow(table, {
      node_id: node,
      degree,
      component: graph.componentOfNode[node],
      node_kind: kind,
      geometry: makeCatalogGeometryText('Point', position)
    });
  });
  return {
    tables: [table],
    diagnostics: [...graph.diagnostics],
    summary: {
      nodeCount: graph.nodes.length,
      edgeCount: graph.edges.length,
      componentCount: graph.componentCount,
      deadEndCount: counts.deadEnd,
      junctionCount: counts.junction
    }
  };
}

export function runNetworkCentrality(
  lines: CatalogFeatureCollection,
  parameters: CatalogNetworkParameters,
  limits: {maximumEdges: number; maximumNodes: number}
): ExtraRunOutput {
  const graph = buildNetworkGraph(lines, parameters, limits.maximumEdges, 'network-centrality');
  // All-sources search is quadratic; the node count is bounded before any search starts.
  if (graph.nodes.length > limits.maximumNodes)
    throw new KeplerAllocationError('capacity', {message: 'Node capacity exceeded'});
  const degrees = countNodeDegrees(graph);
  const nodeCount = graph.nodes.length;
  const table = makeCatalogTable(
    'network-centrality',
    'Node centrality',
    {
      node_id: 'int32',
      degree: 'int32',
      degree_centrality: 'float64',
      reachable_count: 'int32',
      closeness_centrality: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  graph.nodes.forEach((position, node) => {
    const search = searchNetwork(graph, [{node, cost: 0}]);
    let reached = 0;
    let total = 0;
    for (let other = 0; other < nodeCount; other++)
      if (other !== node && Number.isFinite(search.distance[other])) {
        reached++;
        total += search.distance[other];
      }
    // Wasserman-Faust closeness: scaled by the reachable share so disconnected nodes are not inflated.
    const closeness =
      reached > 0 && total > 0 && nodeCount > 1
        ? (reached / (nodeCount - 1)) * (reached / total)
        : 0;
    appendCatalogRow(table, {
      node_id: node,
      degree: degrees[node],
      degree_centrality: nodeCount > 1 ? degrees[node] / (nodeCount - 1) : 0,
      reachable_count: reached,
      closeness_centrality: closeness,
      geometry: makeCatalogGeometryText('Point', position)
    });
  });
  return {
    tables: [table],
    diagnostics: [...graph.diagnostics],
    summary: {nodeCount, edgeCount: graph.edges.length}
  };
}

function computeDestinationCost(
  graph: NetworkGraph,
  distance: Float64Array,
  origin: SnappedPoint,
  destination: SnappedPoint
): number | null {
  const edge = graph.edges[destination.edge];
  let best = Infinity;
  if (edge.forwardCost !== null && Number.isFinite(distance[edge.fromNode]))
    best = Math.min(best, distance[edge.fromNode] + destination.fraction * edge.forwardCost);
  if (edge.reverseCost !== null && Number.isFinite(distance[edge.toNode]))
    best = Math.min(best, distance[edge.toNode] + (1 - destination.fraction) * edge.reverseCost);
  if (destination.edge === origin.edge) {
    if (destination.fraction >= origin.fraction && edge.forwardCost !== null)
      best = Math.min(best, (destination.fraction - origin.fraction) * edge.forwardCost);
    if (destination.fraction < origin.fraction && edge.reverseCost !== null)
      best = Math.min(best, (origin.fraction - destination.fraction) * edge.reverseCost);
  }
  return Number.isFinite(best) ? best : null;
}

export function runNetworkCostMatrix(
  lines: CatalogFeatureCollection,
  origins: CatalogFeatureCollection,
  destinations: CatalogFeatureCollection,
  parameters: CatalogNetworkParameters & {maximumSnapDistance: number},
  limits: {maximumEdges: number}
): ExtraRunOutput {
  const graph = buildNetworkGraph(lines, parameters, limits.maximumEdges, 'network-cost-matrix');
  const originPoints = collectPoints(origins);
  const destinationPoints = collectPoints(destinations);
  const snappedDestinations = destinationPoints.map(point =>
    snapPointToNetwork(graph, point.position, parameters.maximumSnapDistance)
  );
  const table = makeCatalogTable('network-cost-matrix', 'Origin-destination costs', {
    pair_id: 'string',
    origin_index: 'int32',
    origin_source_id: 'string',
    destination_index: 'int32',
    destination_source_id: 'string',
    status: 'string',
    total_cost: 'float64'
  });
  const statusCounts: Record<string, number> = {};
  originPoints.forEach((origin, originIndex) => {
    const snappedOrigin = snapPointToNetwork(
      graph,
      origin.position,
      parameters.maximumSnapDistance
    );
    const search = snappedOrigin
      ? searchNetwork(
          graph,
          makeOriginLegs(graph, snappedOrigin).map(leg => ({node: leg.node, cost: leg.cost}))
        )
      : undefined;
    destinationPoints.forEach((destination, destinationIndex) => {
      const snapped = snappedDestinations[destinationIndex];
      let status = 'ok';
      let cost: number | null = null;
      if (!snappedOrigin || !search) status = 'origin-not-snapped';
      else if (!snapped) status = 'destination-not-snapped';
      else {
        cost = computeDestinationCost(graph, search.distance, snappedOrigin, snapped);
        if (cost === null) status = 'unreachable';
      }
      statusCounts[status] = (statusCounts[status] ?? 0) + 1;
      appendCatalogRow(table, {
        pair_id: `${originIndex}:${destinationIndex}`,
        origin_index: originIndex,
        origin_source_id: getFeatureIdentity(origin.feature),
        destination_index: destinationIndex,
        destination_source_id: getFeatureIdentity(destination.feature),
        status,
        total_cost: cost
      });
    });
  });
  const diagnostics = [...graph.diagnostics];
  for (const [status, count] of Object.entries(statusCounts))
    if (status !== 'ok')
      addCatalogDiagnostic(diagnostics, status, 'warning', `Pairs with status ${status}.`, count);
  return {
    tables: [table],
    diagnostics,
    summary: {
      originCount: originPoints.length,
      destinationCount: destinationPoints.length,
      reachedCount: statusCounts['ok'] ?? 0
    }
  };
}

export function runNetworkNeighborhood(
  lines: CatalogFeatureCollection,
  origins: CatalogFeatureCollection,
  parameters: CatalogNetworkParameters & {maximumSnapDistance: number; maximumCost: number},
  limits: {maximumEdges: number}
): ExtraRunOutput {
  const graph = buildNetworkGraph(lines, parameters, limits.maximumEdges, 'network-neighborhood');
  const originPosition = collectPoints(origins)[0]?.position;
  // Exactly one origin point feature is required.
  if (origins.features.length !== 1 || !originPosition)
    throw new Error('One origin point required');
  const table = makeCatalogTable(
    'network-neighborhood',
    'Neighborhood nodes',
    {node_id: 'int32', cost: 'float64', component: 'int32', geometry: 'string'},
    'geometry'
  );
  const diagnostics: CatalogDiagnostic[] = [...graph.diagnostics];
  const snapped = snapPointToNetwork(graph, originPosition, parameters.maximumSnapDistance);
  if (!snapped) {
    addCatalogDiagnostic(
      diagnostics,
      'origin-not-snapped',
      'error',
      'The origin has no network edge within the maximum snap distance.'
    );
    return {tables: [table], diagnostics, summary: {nodeCount: 0}};
  }
  const search = searchNetwork(
    graph,
    makeOriginLegs(graph, snapped).map(leg => ({node: leg.node, cost: leg.cost}))
  );
  const reached: {node: number; cost: number}[] = [];
  graph.nodes.forEach((_, node) => {
    if (search.distance[node] <= parameters.maximumCost)
      reached.push({node, cost: search.distance[node]});
  });
  reached.sort((a, b) => a.cost - b.cost || a.node - b.node);
  for (const {node, cost} of reached)
    appendCatalogRow(table, {
      node_id: node,
      cost,
      component: graph.componentOfNode[node],
      geometry: makeCatalogGeometryText('Point', graph.nodes[node] as CatalogPosition)
    });
  return {
    tables: [table],
    diagnostics,
    summary: {nodeCount: reached.length, totalNodeCount: graph.nodes.length}
  };
}

export function runNetworkLineGraph(
  lines: CatalogFeatureCollection,
  parameters: CatalogNetworkParameters,
  limits: {maximumEdges: number; maximumRows: number}
): ExtraRunOutput {
  const graph = buildNetworkGraph(lines, parameters, limits.maximumEdges, 'network-line-graph');
  const incident: number[][] = graph.nodes.map(() => []);
  graph.edges.forEach((edge, edgeIndex) => {
    incident[edge.fromNode].push(edgeIndex);
    if (edge.toNode !== edge.fromNode) incident[edge.toNode].push(edgeIndex);
  });
  const plannedRows = incident.reduce(
    (sum, list) => sum + (list.length * (list.length - 1)) / 2,
    0
  );
  // Link count would exceed the declared output capacity; stop before allocating rows.
  if (plannedRows > limits.maximumRows)
    throw new KeplerAllocationError('capacity', {message: 'Output capacity exceeded'});
  const table = makeCatalogTable('network-line-graph', 'Edge adjacency', {
    link_id: 'string',
    shared_node: 'int32',
    first_edge_id: 'string',
    second_edge_id: 'string',
    first_source_index: 'int32',
    second_source_index: 'int32'
  });
  incident.forEach((list, node) => {
    for (let first = 0; first < list.length; first++)
      for (let second = first + 1; second < list.length; second++) {
        const firstEdge = graph.edges[list[first]];
        const secondEdge = graph.edges[list[second]];
        appendCatalogRow(table, {
          link_id: `${node}:${table.rowCount}`,
          shared_node: node,
          first_edge_id: firstEdge.edgeId,
          second_edge_id: secondEdge.edgeId,
          first_source_index: firstEdge.sourceIndex,
          second_source_index: secondEdge.sourceIndex
        });
      }
  });
  return {
    tables: [table],
    diagnostics: [...graph.diagnostics],
    summary: {edgeCount: graph.edges.length, linkCount: table.rowCount}
  };
}
