// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkCostTable: a sparse table of every origin-destination pair within the radius. The
// picture draws the row of the hovered origin: its cost to each intersection inside the radius.

import {GPUNetworkCostTable} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {createNetwork, uploadNetworkTopology} from '../../inputs';
import {chooseSticky} from '../../pick';
import {createNodeIndex, getRoadContext, getRoadValues, uploadRoads} from './shared/roads';
import {parameters} from './network-cost-table.parameters';

const CAPACITY = 3000;

export default defineOperationRun({
  parameters,
  caption:
    'The cost table row of one origin, drawn as the cost to each intersection within the radius.',
  createInputs: ({random, world}) => {
    const network = createNetwork(random, world, 9);
    const defaultNode = network.findNearestNode(world.width / 2, 0.5);
    return {
      network,
      defaultNode,
      node: defaultNode,
      index: createNodeIndex(network),
      costs: new Float32Array(network.slotCount)
    };
  },
  build(builder, inputs) {
    const {network} = inputs;
    const {topology, weights} = uploadNetworkTopology(builder, network);
    const costLimit = builder.allocate('cost-limit', 'float32', 1);
    const offsets = builder.allocate('offsets', 'uint32', network.nodeCount + 1);
    const targets = builder.allocate('targets', 'uint32', CAPACITY);
    const costs = builder.allocate('costs', 'float32', CAPACITY);
    const complete = builder.allocate('complete', 'uint32', 1);
    builder.graph.add(
      new GPUNetworkCostTable({
        topology,
        costLimit: costLimit.view,
        maximumIterations: 48,
        output: {
          offsets: offsets.view,
          targets: targets.view,
          costs: costs.view,
          status: {complete: complete.view}
        }
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        offsets,
        targets,
        costs,
        complete,
        rowCosts: builder.upload(
          'row-costs',
          'float32',
          new Float32Array(network.nodeCount).fill(Number.NaN)
        )
      },
      update: values => {
        network.fillWeights(inputs.costs, values.congestion);
        weights.write(inputs.costs);
        costs.write(new Float32Array(CAPACITY));
        costLimit.write(Float32Array.of(values.costLimit));
      }
    };
  },
  // Probe: the hovered intersection's row of the table, drawn on the roads it reaches.
  interaction: {
    kind: 'probe',
    hint: 'hover an origin',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.network.nodes[inputs.defaultNode * 2],
        inputs.network.nodes[inputs.defaultNode * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.node = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.node);
      const origin = inputs.node;
      const offsets = peek(outputs.offsets);
      const targets = peek(outputs.targets);
      const costs = peek(outputs.costs);
      const complete = (peek(outputs.complete)?.[0] ?? 0) !== 0;
      const row = new Float32Array(inputs.network.nodeCount).fill(Number.NaN);
      let targetCount = 0;
      if (complete && offsets && targets && costs) {
        for (let entry = offsets[origin]; entry < offsets[origin + 1]; entry++) {
          row[targets[entry]] = costs[entry];
          targetCount++;
        }
      }
      outputs.rowCosts.write(row);
      return {
        glyphs: [
          {
            shape: 'crosshair',
            at: [inputs.network.nodes[origin * 2], inputs.network.nodes[origin * 2 + 1]],
            size: 14
          },
          {
            shape: 'ring',
            at: [inputs.network.nodes[origin * 2], inputs.network.nodes[origin * 2 + 1]],
            radius: values.costLimit,
            dashed: true
          }
        ],
        readout:
          complete && offsets && targets && costs
            ? `node ${origin} · ${targetCount} targets within ${values.costLimit.toFixed(2)}`
            : `node ${origin} · incomplete table`
      };
    }
  },
  draw: ({outputs, values}) => [
    getRoadContext(outputs),
    getRoadValues(outputs, outputs.rowCosts, {radius: 2.4, range: [0, values.costLimit]})
  ]
});
