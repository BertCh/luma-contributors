// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkCostMatrix: one row of travel costs per origin, one column per destination. Pairs
// beyond `costLimit` are unreached and drawn muted.

import {GPUNetworkCostMatrix} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {createRoads, pickNodes} from './shared/roads';
import {parameters} from './network-cost-matrix.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Origin by destination travel costs; muted cells are beyond the cost limit.',
  createInputs: ({random, world}) => {
    const network = createRoads(random, world);
    return {
      network,
      origins: pickNodes(network, world, 14),
      destinations: pickNodes(network, world, 14, 20),
      costs: new Float32Array(network.slotCount)
    };
  },
  build(builder, inputs, values) {
    const {network} = inputs;
    const originCount = Math.round(values.startNodes);
    const destinationCount = Math.round(values.targetNodes);
    const {topology, weights} = uploadNetworkTopology(builder, network);
    const costLimit = builder.allocate('cost-limit', 'float32', 1);
    const matrix = builder.allocate('cost-matrix', 'float32', originCount * destinationCount);
    builder.graph.add(
      new GPUNetworkCostMatrix({
        topology,
        startNodes: builder.upload('start-nodes', 'uint32', inputs.origins.slice(0, originCount))
          .view,
        targetNodes: builder.upload(
          'target-nodes',
          'uint32',
          inputs.destinations.slice(0, destinationCount)
        ).view,
        costLimit: costLimit.view,
        maximumIterations: 48,
        output: {costs: matrix.view}
      })
    );
    return {
      outputs: {matrix},
      update: current => {
        network.fillWeights(inputs.costs, current.congestion);
        weights.write(inputs.costs);
        costLimit.write(Float32Array.of(current.costLimit));
      }
    };
  },
  rebuildOn: ['startNodes', 'targetNodes'],
  draw: ({outputs, values}) => [
    {
      layer: 'raster',
      values: outputs.matrix,
      columns: Math.round(values.targetNodes),
      rows: Math.round(values.startNodes),
      range: [0, 1.6]
    }
  ]
});
