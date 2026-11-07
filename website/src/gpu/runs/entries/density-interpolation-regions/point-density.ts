// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointDensity bins clustered points into square cells (count, sum or mean of the point
// weights); the dots are the input events, the raster is the operation's `values` output.

import {
  createGPUPointDensityGaussianKernel,
  GPUPointDensity
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {parameters} from './point-density.parameters';

const STATISTICS = ['count', 'sum', 'mean'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Events binned into cells; the dots are the input points.',
  createInputs: ({random, world}) => ({
    ...createClusteredPoints(random, world, 900, 4, 0.12),
    width: world.width
  }),
  build(builder, points, values) {
    const rows = values.gridSize;
    const columns = Math.round(rows * points.width);
    // Row 0 of the operation's grid is the lowest y; the raster layer draws row 0 at the top.
    const flipped = Float32Array.from(points.positions, (value, index) =>
      index % 2 === 1 ? 1 - value : value
    );
    const positions = builder.upload('positions', 'float32x2', flipped);
    const weights = builder.upload('weights', 'float32', points.values);
    const cells = builder.allocate('values', 'float32', columns * rows);
    const radius = values.smoothing;
    builder.graph.add(
      new GPUPointDensity({
        positions: positions.view,
        weights: weights.view,
        bounds: [0, 0, points.width, 1],
        gridSize: [columns, rows],
        statistic: STATISTICS[values.statistic],
        smoothing:
          radius > 0
            ? {
                kernel: builder.upload(
                  'kernel',
                  'float32',
                  createGPUPointDensityGaussianKernel(radius)
                ).view,
                kernelWidth: 2 * radius + 1,
                kernelHeight: 2 * radius + 1
              }
            : undefined,
        output: {values: cells.view}
      })
    );
    return {
      outputs: {cells, columns, rows, dots: builder.upload('dots', 'float32x2', points.positions)}
    };
  },
  draw: ({outputs}) => [
    {layer: 'raster', values: outputs.cells, columns: outputs.columns, rows: outputs.rows},
    {layer: 'dots', context: true, positions: outputs.dots, radius: 1.6}
  ]
});
