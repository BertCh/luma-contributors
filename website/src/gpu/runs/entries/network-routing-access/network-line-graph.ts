// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkLineGraph: the turn-restriction graph of a road network (every directed road becomes a
// node, every legal turn an arc with a turn cost). Reachability over that graph prices turns, so
// the cost field below depends on the left, right and angle penalties.

import {
  GPUNetworkLineGraph,
  GPUNetworkReachability,
  getGPUNetworkLineGraphParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {
  createRoadNetwork,
  createRoads,
  getSlotValues,
  uploadRoadTopology,
  uploadSlotLines
} from './shared/roads';
import {addSlotValues} from './shared/display';
import {parameters} from './network-line-graph.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Cost to arrive along each directed road, with turn penalties from the line graph.',
  createInputs: ({random, world}) => {
    const network = createRoadNetwork(createRoads(random, world, 10), 0);
    const origin = network.findNearestNode(world.width * 0.5, 0.5);
    const originSlots = Uint32Array.from(
      {length: network.offsets[origin + 1] - network.offsets[origin]},
      (_, index) => network.offsets[origin] + index
    );
    let arcCapacity = 0;
    for (let slot = 0; slot < network.slotCount; slot++) {
      const head = network.neighbors[slot];
      arcCapacity += network.offsets[head + 1] - network.offsets[head];
    }
    return {network, originSlots, arcCapacity};
  },
  build(builder, {network, originSlots, arcCapacity}) {
    const {topology} = uploadRoadTopology(builder, network);
    const nodePositions = builder.upload('node-positions', 'float32x2', network.nodes);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const lineOffsets = builder.allocate('line-offsets', 'uint32', network.slotCount + 1);
    const lineNeighbors = builder.allocate('line-neighbors', 'uint32', arcCapacity);
    const lineWeights = builder.allocate('line-weights', 'float32', arcCapacity);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUNetworkLineGraph({
        topology: {...topology, nodePositions: nodePositions.view},
        parameters: parameterBuffer.view,
        output: {
          topology: {
            offsets: lineOffsets.view,
            neighbors: lineNeighbors.view,
            weights: lineWeights.view
          },
          status: {overflow: overflow.view}
        }
      })
    );
    const costs = builder.allocate('line-costs', 'float32', network.slotCount);
    builder.graph.add(
      new GPUNetworkReachability({
        topology: {
          offsets: lineOffsets.view,
          neighbors: lineNeighbors.view,
          weights: lineWeights.view
        },
        startNodes: builder.upload('start-edges', 'uint32', originSlots).view,
        startCosts: builder.upload(
          'start-costs',
          'float32',
          Float32Array.from(originSlots, slot => network.lengths[slot])
        ).view,
        maximumIterations: 64,
        localIterations: 16,
        output: {costs: costs.view}
      })
    );
    return {
      outputs: {
        slots: uploadSlotLines(builder, network, 0.006),
        costs: addSlotValues(builder, {id: 'line-costs', source: costs, count: network.slotCount})
      },
      update: values =>
        parameterBuffer.write(
          getGPUNetworkLineGraphParameterValues({
            angleCost: values.angleCost,
            leftTurnCost: values.leftTurnCost,
            rightTurnCost: values.rightTurnCost
          })
        )
    };
  },
  draw: ({outputs}) => [getSlotValues(outputs.slots, outputs.costs, {radius: 2.2})]
});
