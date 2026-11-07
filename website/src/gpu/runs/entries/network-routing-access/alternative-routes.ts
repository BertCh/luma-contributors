// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUAlternativeRoutes: the k cheapest simple routes between two intersections. Each route is
// drawn in its own color, from the vertex ids the operation wrote.

import {GPUAlternativeRoutes} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {createRoads, getRoadContext, uploadRoads} from './shared/roads';
import {addVertexSegments} from './shared/display';
import {parameters} from './alternative-routes.parameters';

const _MAXIMUM_ROUTES = 4;
const ROUTE_COLORS = [
  [1, 0.78, 0.25, 1],
  [0.35, 0.8, 1, 1],
  [1, 0.4, 0.6, 1],
  [0.6, 1, 0.5, 1]
] as const;

export default defineOperationRun({
  parameters,
  caption: 'The k cheapest simple routes between two intersections, one color per route.',
  createInputs: ({random, world}) => {
    const network = createRoads(random, world, 7);
    return {
      network,
      source: network.findNearestNode(world.width * 0.3, 0.25),
      target: network.findNearestNode(world.width * 0.7, 0.75),
      costs: new Float32Array(network.slotCount)
    };
  },
  build(builder, inputs, values) {
    const {network} = inputs;
    const routeCount = Math.round(values.routeCount);
    const {topology, weights} = uploadNetworkTopology(builder, network);
    const vertices = builder.allocate('vertices', 'uint32', routeCount * network.nodeCount);
    const edges = builder.allocate('edges', 'uint32', routeCount * network.nodeCount);
    const lengths = builder.allocate('lengths', 'uint32', routeCount);
    const costs = builder.allocate('costs', 'float32', routeCount);
    const counts = builder.allocate('counts', 'uint32', 1);
    const complete = builder.allocate('complete', 'uint32', 1);
    builder.graph.add(
      new GPUAlternativeRoutes({
        topology,
        sources: builder.upload('sources', 'uint32', Uint32Array.of(inputs.source)).view,
        targets: builder.upload('targets', 'uint32', Uint32Array.of(inputs.target)).view,
        routeCount,
        maximumExpansions: 1000000,
        maximumFrontierSize: 8192,
        output: {
          vertices: vertices.view,
          edges: edges.view,
          lengths: lengths.view,
          costs: costs.view,
          counts: counts.view,
          complete: complete.view
        }
      })
    );
    const nodePositions = builder.upload('node-positions', 'float32', network.nodes);
    const routes = Array.from({length: routeCount}, (_, routeIndex) =>
      addVertexSegments(builder, {
        id: `route-${routeIndex}`,
        vertices,
        nodePositions,
        routeIndex,
        stride: network.nodeCount
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        routes,
        endpoints: builder.upload(
          'endpoints',
          'float32x2',
          Float32Array.of(
            network.nodes[inputs.source * 2],
            network.nodes[inputs.source * 2 + 1],
            network.nodes[inputs.target * 2],
            network.nodes[inputs.target * 2 + 1]
          )
        )
      },
      update: current => {
        network.fillWeights(inputs.costs, current.congestion);
        weights.write(inputs.costs);
      }
    };
  },
  rebuildOn: ['routeCount'],
  draw: ({inputs, outputs}) => [
    getRoadContext(outputs),
    ...outputs.routes
      .map((segments, routeIndex) => ({
        layer: 'lines' as const,
        segments,
        count: inputs.network.nodeCount - 1,
        radius: 3.2 - routeIndex * 0.6,
        color: ROUTE_COLORS[routeIndex]
      }))
      .reverse(),
    {layer: 'dots', positions: outputs.endpoints, radius: 8, color: [1, 1, 1, 1]}
  ]
});
