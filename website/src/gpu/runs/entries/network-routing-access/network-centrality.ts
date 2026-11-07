// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkCentrality: radius-bounded betweenness from sampled pivot nodes. Bright roads carry
// the most shortest paths inside the cost limit.

import {GPUNetworkCentrality} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {createRoads, getRoadContext, getRoadValues, pickNodes, uploadRoads} from './shared/roads';
import {parameters} from './network-centrality.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Betweenness per intersection, sampled from the pivots within the cost limit.',
  createInputs: ({random, world}) => {
    const network = createRoads(random, world);
    return {network, pivots: pickNodes(network, world, 96)};
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
  draw: ({outputs}) => [
    getRoadContext(outputs),
    getRoadValues(outputs, outputs.betweenness, {radius: 2.2})
  ]
});
