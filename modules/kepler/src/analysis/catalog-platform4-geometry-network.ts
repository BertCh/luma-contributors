// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUNetworkCentrality, GPUNetworkStructure} from '@luma-contributors/analysis/gpu-network';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import {
  appendCatalogRow,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {buildCatalogDirectedCsr, getNetworkGraphArcs} from './catalog-gpu-network';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {
  buildNetworkGraph,
  type CatalogNetworkParameters,
  type NetworkGraph
} from './catalog-network';

const MAXIMUM_STRUCTURE_NODES = 1024;
const MAXIMUM_STRUCTURE_SLOTS = 65536;
const MAXIMUM_CENTRALITY_ROUNDS = 1024;

/** Undirected incidence count per node; a self-loop counts twice. */
function countNodeDegrees(graph: NetworkGraph): Int32Array {
  const degrees = new Int32Array(graph.nodes.length);
  for (const edge of graph.edges) {
    degrees[edge.fromNode]++;
    degrees[edge.toNode]++;
  }
  return degrees;
}

/**
 * GPUNetworkStructure on the symmetric adjacency of the built graph (one-way restrictions are
 * ignored, as in the CPU reference). Strongly connected components of a symmetric graph are its
 * connected components; the contributor labels each with its lowest node, which ranks to the CPU
 * component number. Degree and node kind come from the graph's incidence counts. Limited to 1024
 * nodes and 65536 adjacency slots by the contributor; larger networks use the CPU engine.
 */
export async function runGpuNetworkStructure(
  context: CatalogGpuContext,
  lines: CatalogFeatureCollection,
  parameters: CatalogNetworkParameters,
  maximumEdges: number
): Promise<CatalogRunOutput> {
  const graph = buildNetworkGraph(lines, parameters, maximumEdges, 'network-structure');
  const nodeCount = graph.nodes.length;
  const slotCount = graph.edges.length * 2;
  if (!nodeCount || nodeCount > MAXIMUM_STRUCTURE_NODES || slotCount > MAXIMUM_STRUCTURE_SLOTS)
    throw new CatalogGpuUnsupportedError('Network is outside the GPUNetworkStructure limits');
  const csr = buildCatalogDirectedCsr(
    nodeCount,
    graph.edges.flatMap(edge => [
      {from: edge.fromNode, to: edge.toNode, cost: 1},
      {from: edge.toNode, to: edge.fromNode, cost: 1}
    ])
  );
  const labels = await withCatalogGpuSession(
    context.device,
    'catalog-network-structure',
    async session => {
      const strongComponents = session.output('components', 'uint32', nodeCount);
      session.graph.add(
        new GPUNetworkStructure({
          topology: {
            offsets: session.input('offsets', csr.offsets, 'uint32'),
            neighbors: session.input('neighbors', csr.neighbors, 'uint32')
          },
          output: {strongComponents: strongComponents.view}
        })
      );
      session.run();
      return session.readUint32(strongComponents);
    }
  );
  const componentOfLabel = new Map<number, number>();
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
  const counts = {deadEnd: 0, junction: 0};
  graph.nodes.forEach((position, node) => {
    if (!componentOfLabel.has(labels[node]))
      componentOfLabel.set(labels[node], componentOfLabel.size);
    const degree = degrees[node];
    const kind =
      degree === 0
        ? 'isolated'
        : degree === 1
          ? 'dead-end'
          : degree === 2
            ? 'pass-through'
            : 'junction';
    if (kind === 'dead-end') counts.deadEnd++;
    else if (kind === 'junction') counts.junction++;
    appendCatalogRow(table, {
      node_id: node,
      degree,
      component: componentOfLabel.get(labels[node])!,
      node_kind: kind,
      geometry: makeCatalogGeometryText('Point', position)
    });
  });
  return {
    tables: [table],
    diagnostics: [...graph.diagnostics],
    summary: {
      nodeCount,
      edgeCount: graph.edges.length,
      componentCount: componentOfLabel.size,
      deadEndCount: counts.deadEnd,
      junctionCount: counts.junction
    },
    dispatchedContributors: ['GPUNetworkStructure']
  };
}

/**
 * GPUNetworkCentrality with every node as a pivot over the directed arc costs. The contributor
 * reports the reached count (pivot included) and closeness `(r - 1) / S`; the catalog's
 * Wasserman-Faust closeness is that value scaled by `(r - 1) / (n - 1)`. Costs are float32 (relative
 * error about 1e-6 per path). Rounds double until the reachability converges.
 */
export async function runGpuNetworkCentrality(
  context: CatalogGpuContext,
  lines: CatalogFeatureCollection,
  parameters: CatalogNetworkParameters,
  limits: {maximumEdges: number; maximumNodes: number}
): Promise<CatalogRunOutput> {
  const graph = buildNetworkGraph(lines, parameters, limits.maximumEdges, 'network-centrality');
  const nodeCount = graph.nodes.length;
  // The CPU engine raises the allocation error for oversized node counts.
  if (!nodeCount || nodeCount > limits.maximumNodes)
    throw new CatalogGpuUnsupportedError('Node count outside the centrality limits');
  const arcs = getNetworkGraphArcs(graph);
  const forward = buildCatalogDirectedCsr(nodeCount, arcs);
  const reverse = buildCatalogDirectedCsr(
    nodeCount,
    arcs.map(arc => ({from: arc.to, to: arc.from, cost: arc.cost}))
  );
  const pivots = Uint32Array.from({length: nodeCount}, (_, node) => node);
  let result: {closeness: Float32Array; reached: Uint32Array} | null = null;
  for (let rounds = 64; !result && rounds <= MAXIMUM_CENTRALITY_ROUNDS; rounds *= 2) {
    const attempt = await withCatalogGpuSession(
      context.device,
      'catalog-network-centrality',
      async session => {
        const closeness = session.output('closeness', 'float32', nodeCount);
        const reached = session.output('reached', 'uint32', nodeCount);
        const converged = session.output('converged', 'uint32', 1);
        const topology = (csr: typeof forward, name: string) => ({
          offsets: session.input(`${name}-offsets`, csr.offsets, 'uint32'),
          neighbors: session.input(`${name}-neighbors`, csr.neighbors, 'uint32'),
          weights: session.input(`${name}-weights`, csr.weights, 'float32')
        });
        session.graph.add(
          new GPUNetworkCentrality({
            topology: topology(forward, 'forward'),
            reverseTopology: topology(reverse, 'reverse'),
            pivotNodes: session.input('pivots', pivots, 'uint32'),
            maximumIterations: rounds,
            output: {
              closeness: closeness.view,
              reachedCounts: reached.view,
              status: {converged: converged.view}
            }
          })
        );
        session.run();
        return {
          converged: (await session.readScalar(converged)) !== 0,
          closeness: await session.readFloat32(closeness),
          reached: await session.readUint32(reached)
        };
      }
    );
    if (attempt.converged) result = attempt;
  }
  if (!result)
    throw new CatalogGpuUnsupportedError(
      'GPUNetworkCentrality did not converge within 1024 rounds'
    );
  const degrees = countNodeDegrees(graph);
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
    const reached = result.reached[node] - 1;
    appendCatalogRow(table, {
      node_id: node,
      degree: degrees[node],
      degree_centrality: nodeCount > 1 ? degrees[node] / (nodeCount - 1) : 0,
      reachable_count: reached,
      closeness_centrality:
        reached > 0 && nodeCount > 1 ? (reached / (nodeCount - 1)) * result.closeness[node] : 0,
      geometry: makeCatalogGeometryText('Point', position)
    });
  });
  return {
    tables: [table],
    diagnostics: [...graph.diagnostics],
    summary: {nodeCount, edgeCount: graph.edges.length},
    dispatchedContributors: ['GPUNetworkCentrality']
  };
}
