// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalRBF fits a radial basis function to the nearest samples of every grid cell. The kernel
// sets how the surface bends between samples; `smoothing` relaxes it away from exact interpolation.

import {GPULocalRBF} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getRasterCell, getRasterCellCenter} from '../../pick';
import {parameters} from './local-rbf.parameters';
import {addChannel} from './support/channel';
import {
  createInterpolationInputs,
  createNeighborhoodGlyphs,
  findNearestSamples,
  drawSampleDots
} from './support/interpolation';

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
        positions: builder.upload('sample-positions', 'float32x2', samples.positions),
        sampleValues: builder.upload('sample-values', 'float32', samples.values)
      }
    };
  },
  rebuildOn: ['kernel', 'smoothing', 'neighborCount'],
  // The hovered cell shows the neighborCount samples its radial basis function was fitted to.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const {grid, samples} = inputs;
      const cell = getRasterCell(grid.columns, grid.rows, grid.window, pointer.x, pointer.y);
      const at = getRasterCellCenter(grid.columns, grid.rows, grid.window, cell.column, cell.row);
      const prediction = peek(outputs.prediction)?.[cell.index];
      const kernel = KERNELS[values.kernel];
      // The fit reaches as far as its farthest neighbor.
      const farthestRow =
        findNearestSamples(samples.positions, at[0], at[1], values.neighborCount).at(-1) ?? 0;
      const farthest = Math.hypot(
        samples.positions[farthestRow * 2] - at[0],
        samples.positions[farthestRow * 2 + 1] - at[1]
      );
      return {
        glyphs: [
          ...createNeighborhoodGlyphs(samples.positions, at, values.neighborCount, true),
          {shape: 'ring', at, radius: farthest, dashed: true},
          {shape: 'crosshair', at}
        ],
        readout:
          prediction === undefined
            ? `${kernel} · ${values.neighborCount} neighbors`
            : `${kernel} · ${values.neighborCount} neighbors · ${prediction.toFixed(2)}`
      };
    }
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
    ...drawSampleDots(outputs.positions, outputs.sampleValues)
  ]
});
