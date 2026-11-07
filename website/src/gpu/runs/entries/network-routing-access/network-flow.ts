// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkFlow: integral min-cost flow from the left source to the right sink. Roads carry the
// computed flow (brighter is more); the dots mark intersections still reachable from the source in
// the residual graph, which exposes the bottleneck cut once the required flow cannot be met.

import {GPUNetworkFlow} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addSlotValues} from './shared/display';
import {
  createGridRoadNetwork,
  getRoadContext,
  getSlotValues,
  uploadRoads,
  uploadSlotLines
} from './shared/roads';
import {parameters} from './network-flow.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Flow carried by each directed road; dots mark the residual source side of the cut.',
  createInputs: ({random, world}) => {
    const network = createGridRoadNetwork(random, world, 9, 6);
    const edges = new Uint32Array(network.slotCount * 2);
    for (let slot = 0; slot < network.slotCount; slot++) {
      edges[slot * 2] = network.slotSources[slot];
      edges[slot * 2 + 1] = network.neighbors[slot];
    }
    // One base capacity per road, shared by both directions.
    const base = Uint32Array.from(
      {length: network.slotCount},
      (_, slot) =>
        1 + ((Math.imul(Math.min(slot, network.reverseSlots[slot]) + 1, 2654435761) >>> 0) % 3)
    );
    const costs = Uint32Array.from(network.lengths, length => Math.round(length * 100));
    return {
      network,
      edges,
      base,
      costs,
      source: network.findNearestNode(0, 0.5),
      sink: network.findNearestNode(world.width, 0.5)
    };
  },
  build(builder, inputs) {
    const {network} = inputs;
    const capacities = builder.allocate('capacities', 'uint32', network.slotCount);
    const parameterBuffer = builder.allocate('parameters', 'uint32', 3);
    const flow = builder.allocate('flow', 'uint32', network.slotCount);
    const cut = builder.allocate('cut', 'uint32', network.nodeCount);
    const summary = builder.allocate('summary', 'uint32', 8);
    builder.graph.add(
      new GPUNetworkFlow({
        nodeCount: network.nodeCount,
        edges: builder.upload('edges', 'uint32x2', inputs.edges).view,
        capacities: capacities.view,
        costs: builder.upload('costs', 'uint32', inputs.costs).view,
        parameters: parameterBuffer.view,
        output: {flow: flow.view, cut: cut.view, summary: summary.view}
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        slots: uploadSlotLines(builder, network, 0.006),
        flow: addSlotValues(builder, {id: 'flow', source: flow, count: network.slotCount}),
        cut
      },
      update: values => {
        capacities.write(Uint32Array.from(inputs.base, base => base * Math.round(values.capacity)));
        parameterBuffer.write(
          Uint32Array.of(inputs.source, inputs.sink, Math.round(values.requiredFlow))
        );
      }
    };
  },
  draw: ({outputs}) => [
    getRoadContext(outputs),
    getSlotValues(outputs.slots, outputs.flow, {radius: 2.2, range: [-2.5, 7]}),
    {
      layer: 'dots',
      positions: outputs.nodes,
      values: outputs.cut,
      ramp: 'sequential',
      range: [-1, 1],
      radius: 4
    }
  ]
});
