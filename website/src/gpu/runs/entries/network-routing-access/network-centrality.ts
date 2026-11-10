// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkCentrality: radius-bounded betweenness from sampled pivot nodes. Bright roads carry
// the most shortest paths inside the cost limit.

import {GPUNetworkCentrality} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {findParameter, snapToParameter} from '../../interaction';
import {
  createRoads,
  findReachPoint,
  getRoadContext,
  getRoadValues,
  pickNodes,
  uploadRoads
} from './shared/roads';
import {parameters} from './network-centrality.parameters';

const COST_LIMIT = findParameter(parameters, 'costLimit');

export default defineOperationRun({
  parameters,
  caption: 'Betweenness per intersection, sampled from the pivots within the cost limit.',
  createInputs: ({random, world}) => {
    const network = createRoads(random, world);
    const center = network.findNearestNode(world.width / 2, 0.5);
    return {
      network,
      pivots: pickNodes(network, world, 96),
      center: [network.nodes[center * 2], network.nodes[center * 2 + 1]] as [number, number]
    };
  },
  build(builder, {network, pivots}, values) {
    const pivotCount = Math.round(values.pivotNodes);
    const {topology} = uploadNetworkTopology(builder, network);
    const costLimit = builder.allocate('cost-limit', 'float32', 1);
    const betweenness = builder.allocate('betweenness', 'float32', network.nodeCount);
    builder.graph.add(
      new GPUNetworkCentrality({
        topology,
        reverseTopology: topology,
        pivotNodes: builder.upload('pivot-nodes', 'uint32', pivots.slice(0, pivotCount)).view,
        costLimit: costLimit.view,
        betweennessScale: network.nodeCount / pivotCount,
        maximumIterations: 48,
        output: {betweenness: betweenness.view}
      })
    );
    return {
      outputs: {...uploadRoads(builder, network), betweenness},
      update: current => costLimit.write(Float32Array.of(current.costLimit))
    };
  },
  rebuildOn: ['pivotNodes'],
  // Reach: the distance from the central intersection to the pointer is the cost limit.
  interaction: {
    kind: 'reach',
    hint: 'reach out',
    rest: ({inputs, world, toUnit}) =>
      toUnit(findReachPoint(inputs.center, COST_LIMIT.default, world)),
    resolve(pointer, {inputs, outputs, peek}) {
      const costLimit = snapToParameter(
        COST_LIMIT,
        Math.hypot(pointer.x - inputs.center[0], pointer.y - inputs.center[1])
      );
      const betweenness = peek(outputs.betweenness);
      let peak = 0;
      if (betweenness)
        for (const value of betweenness) if (Number.isFinite(value)) peak = Math.max(peak, value);
      return {
        values: {costLimit},
        glyphs: [
          {shape: 'ring', at: inputs.center, radius: costLimit, dashed: true},
          {shape: 'line', from: inputs.center, to: [pointer.x, pointer.y], dashed: true}
        ],
        readout: `costLimit ${costLimit.toFixed(2)} · peak betweenness ${peak.toFixed(0)}`
      };
    }
  },
  draw: ({outputs}) => [
    getRoadContext(outputs),
    getRoadValues(outputs, outputs.betweenness, {radius: 2.2, range: [0, 3000]})
  ]
});
