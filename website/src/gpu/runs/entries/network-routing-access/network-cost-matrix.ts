// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkCostMatrix: one row of travel costs per origin, one column per destination. Pairs
// beyond `costLimit` are unreached and drawn muted.

import {GPUNetworkCostMatrix} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {createRoads, pickNodes} from './shared/roads';
import {getRasterCell, getRasterCellCenter} from '../../pick';
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
  // Probe: the cell under the pointer is one origin-destination pair; its row and column are outlined.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const columns = Math.round(values.targetNodes);
      const rows = Math.round(values.startNodes);
      const bounds = [0, 0, world.width, 1] as const;
      const cell = getRasterCell(columns, rows, bounds, pointer.x, pointer.y);
      const [centerX, centerY] = getRasterCellCenter(columns, rows, bounds, cell.column, cell.row);
      const halfWidth = world.width / columns / 2;
      const halfHeight = 1 / rows / 2;
      const costs = peek(outputs.matrix);
      const cost = costs ? costs[cell.index] : Number.NaN;
      return {
        spotlight: null,
        glyphs: [
          {
            shape: 'box',
            minimum: [0, centerY - halfHeight],
            maximum: [world.width, centerY + halfHeight],
            dashed: true
          },
          {
            shape: 'box',
            minimum: [centerX - halfWidth, 0],
            maximum: [centerX + halfWidth, 1],
            dashed: true
          },
          {
            shape: 'box',
            minimum: [centerX - halfWidth, centerY - halfHeight],
            maximum: [centerX + halfWidth, centerY + halfHeight]
          }
        ],
        readout: `origin ${inputs.origins[cell.row]} to destination ${inputs.destinations[cell.column]} · ${
          Number.isFinite(cost) && cost < 1e20 ? `cost ${cost.toFixed(2)}` : 'beyond costLimit'
        }`
      };
    }
  },
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
