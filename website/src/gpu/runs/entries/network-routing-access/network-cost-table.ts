// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkCostTable: a sparse table of every origin-destination pair within the radius. Each bar
// is one table row's travel cost; the number of rows grows with the cost limit.

import {GPUNetworkCostTable} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {createNetwork, uploadNetworkTopology} from '../../inputs';
import {parameters} from './network-cost-table.parameters';

const CAPACITY = 3000;

export default defineOperationRun({
  parameters,
  caption: 'Costs of the origin-destination pairs inside the radius, in table order.',
  createInputs: ({random, world}) => {
    const network = createNetwork(random, world, 9);
    return {network, costs: new Float32Array(network.slotCount)};
  },
  build(builder, inputs) {
    const {network} = inputs;
    const {topology, weights} = uploadNetworkTopology(builder, network);
    const costLimit = builder.allocate('cost-limit', 'float32', 1);
    const offsets = builder.allocate('offsets', 'uint32', network.nodeCount + 1);
    const targets = builder.allocate('targets', 'uint32', CAPACITY);
    const costs = builder.allocate('costs', 'float32', CAPACITY);
    builder.graph.add(
      new GPUNetworkCostTable({
        topology,
        costLimit: costLimit.view,
        maximumIterations: 48,
        output: {offsets: offsets.view, targets: targets.view, costs: costs.view}
      })
    );
    return {
      outputs: {costs},
      update: values => {
        network.fillWeights(inputs.costs, values.congestion);
        weights.write(inputs.costs);
        costs.write(new Float32Array(CAPACITY));
        costLimit.write(Float32Array.of(values.costLimit));
      }
    };
  },
  draw: ({outputs, values}) => [
    {layer: 'chart', kind: 'bars', values: outputs.costs, range: [0, values.costLimit]}
  ]
});
