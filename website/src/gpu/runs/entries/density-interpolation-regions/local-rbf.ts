// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalRBF fits a radial basis function to the nearest samples of every grid cell. The kernel
// sets how the surface bends between samples; `smoothing` relaxes it away from exact interpolation.

import {GPULocalRBF} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './local-rbf.parameters';
import {addChannel} from './support/channel';
import {createInterpolationInputs} from './support/interpolation';

const KERNELS = ['thin-plate-spline', 'cubic', 'gaussian', 'linear', 'quintic'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Local radial basis interpolation of scattered samples (dots).',
  createInputs: ({random, world}) => createInterpolationInputs(random, world),
  build(builder, {grid, samples}, values) {
    const cellCount = grid.columns * grid.rows;
    const results = builder.allocate('results', 'float32x4', cellCount);
    const kernel = KERNELS[values.kernel];
    builder.graph.add(
      new GPULocalRBF({
        positions: builder.upload('positions', 'float32x2', samples.positions).view,
        values: builder.upload('values', 'float32', samples.values).view,
        queries: builder.upload('queries', 'float32x2', grid.centers).view,
        kernel,
        epsilon: kernel === 'gaussian' ? 6 : undefined,
        smoothing: values.smoothing,
        neighborCount: values.neighborCount,
        output: {values: results.view, status: builder.allocate('status', 'uint32', cellCount).view}
      })
    );
    return {
      outputs: {
        prediction: addChannel(builder, results, 0, 'prediction'),
        positions: builder.upload('sample-positions', 'float32x2', samples.positions)
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.prediction,
      columns: inputs.grid.columns,
      rows: inputs.grid.rows,
      range: [0, 1],
      smoothing: true
    },
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2.5, color: [1, 1, 1, 0.9]}
  ]
});
