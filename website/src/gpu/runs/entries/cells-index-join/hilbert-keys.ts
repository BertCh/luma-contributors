// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUHilbertKeys maps points to positions along a Hilbert curve. Dots are colored by their key; the
// polyline visits the points in the order the operation sorted them, so it follows the curve.

import {GPUHilbertKeys} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {addUintToFloat} from './support/kernels';
import {parameters} from './hilbert-keys.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Points keyed along a Hilbert curve; the line visits them in key order.',
  createInputs: ({random, world}) => createUniformPoints(random, world, 220, 0),
  build(builder, points, values) {
    const positions = builder.allocate('positions', 'float32x2', points.count);
    const keys = builder.allocate('keys', 'uint32', points.count);
    const sortedRows = builder.allocate('sorted-rows', 'uint32', points.count);
    builder.graph.add(
      new GPUHilbertKeys({
        order: values.order,
        positions: positions.view,
        output: {keys: keys.view, sortedRows: sortedRows.view}
      })
    );
    return {
      outputs: {positions, sortedRows, keyValues: addUintToFloat(builder, 'keys', keys)},
      update: current =>
        positions.write(
          Float32Array.from(points.positions, (value, index) => {
            const noise = Math.sin(index * 12.9898) * 43758.5453;
            return value + (noise - Math.floor(noise) - 0.5) * 2 * current.jitter;
          })
        )
    };
  },
  rebuildOn: ['order'],
  draw: ({inputs, outputs}) => [
    {
      layer: 'paths',
      nodes: outputs.positions,
      ids: outputs.sortedRows,
      count: inputs.count,
      radius: 1.2
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.keyValues,
      ramp: 'sequential',
      radius: 4.5
    }
  ]
});
