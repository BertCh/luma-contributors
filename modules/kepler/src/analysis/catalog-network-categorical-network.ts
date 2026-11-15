// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUNetworkPotentialPathArea,
  GPUNetworkTravelTimePercentiles
} from '@luma-contributors/analysis/gpu-network';
import {
  NETWORK_PARAMETERS,
  getNetworkParameters,
  type CatalogOperationBinding
} from './catalog-binding-types';
import {buildNetworkGraph, type NetworkGraph} from './catalog-network';
import {
  buildCatalogDirectedCsr,
  getNetworkGraphArcs,
  runGpuReachability
} from './catalog-gpu-network';
import {makeCatalogGeometryText, type CatalogFeatureCollection} from './catalog-geojson';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {
  makeInputSpec,
  makeOutputSpec,
  makeNumberParameter,
  makeStringParameter
} from './catalog-platform3-b-common';
import {
  getNumber,
  getString,
  makeBinding,
  makeResult,
  uploadWeights
} from './catalog-network-categorical-common';

/** Match arc order while refusing to invent a time-dependent model for split source parts. */
function prepareNetworkTimeSlopes(
  lines: CatalogFeatureCollection,
  graph: NetworkGraph,
  slopeProperty: string
): number[] {
  const pieceCounts = new Map<string, number>();
  for (const edge of graph.edges) {
    const key = `${edge.sourceIndex}:${edge.partIndex}`;
    pieceCounts.set(key, (pieceCounts.get(key) ?? 0) + 1);
  }
  return graph.outgoing.flatMap(outgoing =>
    outgoing.map(arc => {
      const edge = graph.edges[arc.edge];
      const feature = lines.features[edge.sourceIndex];
      const slope = slopeProperty ? feature.properties?.[slopeProperty] : 0;
      if (typeof slope !== 'number' || !Number.isFinite(Math.fround(slope)) || slope < 0)
        throw new Error('Duration slopes must be finite and nonnegative (FIFO)');
      if (slope !== 0 && pieceCounts.get(`${edge.sourceIndex}:${edge.partIndex}`)! > 1)
        throw new Error(
          `network-travel-time-percentiles: nonzero duration slope on split source feature ${edge.sourceIndex}, part ${edge.partIndex}; provide per-edge lines or disable splitAtSharedVertices`
        );
      return slope;
    })
  );
}

export const NETWORK_BINDINGS: readonly CatalogOperationBinding[] = (
  ['network-potential-path-area', 'network-travel-time-percentiles'] as const
).map(id => {
  const prism = id === 'network-potential-path-area';
  const title = prism ? 'Network potential path area' : 'Network travel-time percentiles';
  const contributor = prism ? 'GPUNetworkPotentialPathArea' : 'GPUNetworkTravelTimePercentiles';
  return makeBinding({
    id,
    title,
    domain: 'network',
    summary: prism
      ? 'Directed edges on a route between two anchor nodes within a travel budget.'
      : 'Nearest-rank travel-time percentiles over a regular departure window on an affine FIFO network.',
    contributorNames: [contributor, ...(prism ? ['GPUNetworkReachability'] : [])],
    inputs: [makeInputSpec('lines', 'Network lines', ['line'], 1024, 8192)],
    parameters: [
      ...NETWORK_PARAMETERS.filter(parameter => parameter.name !== 'crossings'),
      makeNumberParameter('originNode', 'Origin node index', 0, 0, 255, undefined, 'integer'),
      ...(prism
        ? [
            makeNumberParameter(
              'destinationNode',
              'Destination node index',
              1,
              0,
              255,
              undefined,
              'integer'
            ),
            makeNumberParameter('budget', 'Travel cost budget', 1000, 0, 1e12, 'cost')
          ]
        : [
            makeStringParameter('slopeProperty', 'Duration slope property (optional)', ''),
            makeNumberParameter('firstDeparture', 'First departure', 0, 0, 1e9, 'cost'),
            makeNumberParameter('departureStep', 'Departure step', 1, 0, 1e9, 'cost'),
            makeNumberParameter(
              'departureCount',
              'Departure count',
              5,
              1,
              60,
              undefined,
              'integer'
            ),
            {
              name: 'percentiles',
              label: 'Percentiles (ascending)',
              kind: 'number-list' as const,
              minimum: 0.001,
              maximum: 100,
              maximumLength: 16,
              defaultValue: [50, 90]
            },
            makeNumberParameter(
              'maximumRelaxations',
              'Maximum routing relaxations',
              256,
              0,
              65536,
              undefined,
              'integer'
            )
          ])
    ],
    outputs: [
      makeOutputSpec(
        id,
        title,
        prism ? 'line' : 'point',
        prism ? ['arc_index', 'source_index', 'source_id'] : ['node_index', 'percentile'],
        prism ? 4096 : 4096
      )
    ],
    declaredBehavior: [
      'The existing catalog graph builder splits at shared vertices, honors one-way and directional costs, and preserves original source edge identities. Crossing lines without a shared vertex remain disconnected.',
      'Anchors are graph node indices from Network build; unknown indices are rejected. At most 256 nodes and 4096 directed arcs after graph construction.',
      'Potential path area uses forward and reversed GPU reachability trees; each directed edge is tested whole, and node membership and slack are retained.',
      'Unreachable or undefined travel times and slack are null, with separate membership/reached-count status. Routing valid/converged status is reported for every departure.'
    ],
    limitation: prism
      ? 'Node anchors only; no point snapping or isochrone polygon. Float32 costs near the budget can change membership.'
      : 'Reference-tier routing: affine duration base+slope*departure, nonnegative slope. Source parts split at shared vertices must have zero slope; otherwise provide per-edge lines or disable splitAtSharedVertices. No restriction automaton or piecewise-arrival input. Percentiles include unreachable departures as infinity.',
    runGpu: async (context, inputs, values) => {
      const graph = buildNetworkGraph(inputs['lines'], getNetworkParameters(values), 4096);
      const arcs = getNetworkGraphArcs(graph);
      const nodeCount = graph.nodes.length;
      if (!nodeCount || nodeCount > 256 || arcs.length > 4096)
        throw new Error('Network must have 1-256 nodes and at most 4096 directed arcs');
      if (arcs.some(arc => !Number.isFinite(Math.fround(arc.cost))))
        throw new Error('Network costs must be finite in Float32');
      const origin = getNumber(values, 'originNode');
      const destination = prism ? getNumber(values, 'destinationNode') : 0;
      if (origin >= nodeCount || destination >= nodeCount)
        throw new Error('Anchor node index must be below network node count');
      const slopes = prism
        ? undefined
        : prepareNetworkTimeSlopes(inputs['lines'], graph, getString(values, 'slopeProperty'));
      const topology = buildCatalogDirectedCsr(nodeCount, arcs);
      const forward = prism
        ? await runGpuReachability(context, topology, nodeCount, [[{node: origin, cost: 0}]], false)
        : undefined;
      const reverse = prism
        ? await runGpuReachability(
            context,
            buildCatalogDirectedCsr(
              nodeCount,
              arcs.map(arc => ({from: arc.to, to: arc.from, cost: arc.cost}))
            ),
            nodeCount,
            [[{node: destination, cost: 0}]],
            false
          )
        : undefined;
      const output = await withCatalogGpuSession(context.device, `catalog-${id}`, async session => {
        if (prism) {
          const membership = session.output('membership', 'uint32', arcs.length);
          const slack = session.output('slack', 'float32', arcs.length);
          const nodes = session.output('nodes', 'uint32', nodeCount);
          const summary = session.output('summary', 'uint32', 2);
          session.graph.add(
            new GPUNetworkPotentialPathArea({
              topology: uploadWeights(session, topology),
              forwardCosts: session.input('forward', forward!.costs, 'float32'),
              reverseCosts: session.input('reverse', reverse!.costs, 'float32'),
              budget: session.input(
                'budget',
                new Float32Array([getNumber(values, 'budget')]),
                'float32'
              ),
              output: {
                edgeInPrism: membership.view,
                edgeSlack: slack.view,
                nodeInPrism: nodes.view,
                summary: summary.view
              }
            })
          );
          session.run();
          const included = await session.readUint32(membership);
          const margins = await session.readFloat32(slack);
          const nodeMembership = await session.readUint32(nodes);
          const counts = await session.readUint32(summary);
          // Arc order is the same stable node-major order used by buildCatalogDirectedCsr.
          let arcIndex = 0;
          const rows = graph.outgoing.flatMap((outgoing, from) =>
            outgoing.map(arc => {
              const edge = graph.edges[arc.edge];
              const slot = arcIndex++;
              return {
                arc_index: slot,
                source_index: edge.sourceIndex,
                source_id: edge.sourceId,
                from_node: from,
                to_node: arc.to,
                in_prism: included[slot] !== 0,
                from_in_prism: nodeMembership[from] !== 0,
                to_in_prism: nodeMembership[arc.to] !== 0,
                slack: margins[slot],
                geometry: makeCatalogGeometryText(
                  'LineString',
                  arc.forward ? edge.coordinates : [...edge.coordinates].reverse()
                )
              };
            })
          );
          return makeResult(
            id,
            title,
            {
              arc_index: 'int32',
              source_index: 'int32',
              source_id: 'string',
              from_node: 'int32',
              to_node: 'int32',
              in_prism: 'boolean',
              from_in_prism: 'boolean',
              to_in_prism: 'boolean',
              slack: 'float64',
              geometry: 'string'
            },
            rows,
            {prismNodeCount: counts[0], prismEdgeCount: counts[1]}
          );
        }
        const percentiles = values['percentiles'] as readonly number[];
        if (percentiles.some((value, index) => index > 0 && value <= percentiles[index - 1]))
          throw new Error('Percentiles must be strictly ascending');
        const departureCount = getNumber(values, 'departureCount');
        const times = session.output('times', 'float32', nodeCount * percentiles.length);
        const reached = session.output('reached', 'uint32', nodeCount);
        const status = session.output('status', 'uint32', departureCount * 2);
        session.graph.add(
          new GPUNetworkTravelTimePercentiles({
            stateVertices: session.input(
              'vertices',
              Uint32Array.from(graph.nodes.map((_, index) => index)),
              'uint32'
            ),
            transitions: session.input(
              'transitions',
              Uint32Array.from(arcs.flatMap(arc => [arc.from, arc.to])),
              'uint32x2'
            ),
            costs: session.input(
              'costs',
              Float32Array.from(arcs.flatMap((arc, index) => [arc.cost, slopes![index]])),
              'float32x2'
            ),
            vertexCount: nodeCount,
            initialState: session.input('origin', new Uint32Array([origin]), 'uint32'),
            departureCount,
            percentiles,
            window: session.input(
              'window',
              new Float32Array([
                getNumber(values, 'firstDeparture'),
                getNumber(values, 'departureStep')
              ]),
              'float32'
            ),
            maximumRelaxations: getNumber(values, 'maximumRelaxations'),
            output: {travelTimes: times.view, reachedCounts: reached.view, status: status.view}
          })
        );
        session.run();
        const travelTimes = await session.readFloat32(times);
        const reachedCounts = await session.readUint32(reached);
        const statuses = await session.readUint32(status);
        return makeResult(
          id,
          title,
          {
            node_index: 'int32',
            percentile: 'float64',
            travel_time: 'float64',
            reached_departures: 'int32',
            geometry: 'string'
          },
          graph.nodes.flatMap((position, node) =>
            percentiles.map((percentile, index) => ({
              node_index: node,
              percentile,
              travel_time: travelTimes[node * percentiles.length + index],
              reached_departures: reachedCounts[node],
              geometry: makeCatalogGeometryText('Point', position)
            }))
          ),
          Object.fromEntries(
            Array.from({length: departureCount}, (_, index) => [
              [`departure${index}Valid`, statuses[2 * index]],
              [`departure${index}Converged`, statuses[2 * index + 1]]
            ]).flat()
          )
        );
      });
      output.diagnostics.push(...graph.diagnostics);
      return output;
    }
  });
});
