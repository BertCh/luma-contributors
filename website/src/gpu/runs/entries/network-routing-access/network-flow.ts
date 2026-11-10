// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkFlow: integral min-cost flow from the left source to the right sink. Roads carry the
// computed flow (brighter is more); the dots mark intersections still reachable from the source in
// the residual graph, which exposes the bottleneck cut once the required flow cannot be met.

import {GPUNetworkFlow} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addSlotValues} from './shared/display';
import {chooseSticky} from '../../pick';
import {
  createNodeIndex,
  createSmallRoadNetwork,
  getEndpointMarkers,
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
    const network = createSmallRoadNetwork(random, world, 54);
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
    const sink = network.findNearestNode(world.width, 0.5);
    return {
      network,
      edges,
      base,
      costs,
      source: network.findNearestNode(0, 0.5),
      defaultSink: sink,
      sink,
      index: createNodeIndex(network)
    };
  },
  build(builder, inputs) {
    const {network} = inputs;
    const capacities = builder.allocate('capacities', 'uint32', network.slotCount);
    const parameterBuffer = builder.allocate('parameters', 'uint32', 3);
    const flow = builder.allocate('flow', 'uint32', network.slotCount);
    const cut = builder.allocate('cut', 'uint32', network.nodeCount);
    const summary = builder.allocate('summary', 'uint32', 8);
    const sinkPosition = builder.allocate('sink-position', 'float32x2', 1);
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
        cut,
        summary,
        sinkPosition,
        sourcePosition: builder.upload(
          'source-position',
          'float32x2',
          Float32Array.of(network.nodes[inputs.source * 2], network.nodes[inputs.source * 2 + 1])
        )
      },
      update: values => {
        sinkPosition.write(
          Float32Array.of(network.nodes[inputs.sink * 2], network.nodes[inputs.sink * 2 + 1])
        );
        capacities.write(Uint32Array.from(inputs.base, base => base * Math.round(values.capacity)));
        parameterBuffer.write(
          Uint32Array.of(inputs.source, inputs.sink, Math.round(values.requiredFlow))
        );
      }
    };
  },
  // The pointer places the sink on the nearest intersection; the source stays on the left edge.
  interaction: {
    kind: 'place',
    hint: 'place the sink',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.network.nodes[inputs.defaultSink * 2],
        inputs.network.nodes[inputs.defaultSink * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const node = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.sink);
      const changed = node !== inputs.sink;
      inputs.sink = node;
      const summary = peek(outputs.summary);
      const flowText = summary
        ? `flow ${summary[2]} of ${Math.round(values.requiredFlow)} · cut capacity ${summary[6]}`
        : `required flow ${Math.round(values.requiredFlow)}`;
      return {
        rerun: changed,
        glyphs: [
          {
            shape: 'crosshair',
            at: [inputs.network.nodes[node * 2], inputs.network.nodes[node * 2 + 1]],
            size: 14
          }
        ],
        readout: flowText
      };
    }
  },
  draw: ({outputs, theme}) => [
    getRoadContext(outputs),
    getSlotValues(outputs.slots, outputs.flow, {radius: 2.2, range: [-2.5, 7]}),
    // Intersections still reachable from the source in the residual graph; the rest stay hidden.
    {
      layer: 'dots',
      positions: outputs.nodes,
      values: outputs.cut,
      ramp: 'categorical',
      categories: [
        [...theme.muted, 0],
        [...theme.categories[3], 1]
      ],
      radius: 4.5
    },
    ...getEndpointMarkers(outputs.sourcePosition, theme, {colorIndex: 4}),
    ...getEndpointMarkers(outputs.sinkPosition, theme, {colorIndex: 2})
  ]
});
