// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalKriging predicts at every grid-cell center from the nearest `neighborCount` samples.
// The operation writes [prediction, variance, ...] rows; one channel is drawn.

import {GPULocalKriging} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getRasterCell, getRasterCellCenter} from '../../pick';
import {parameters} from './local-kriging.parameters';
import {addChannel} from './support/channel';
import {
  createInterpolationInputs,
  createNeighborhoodGlyphs,
  drawSampleDots
} from './support/interpolation';

export default defineOperationRun({
  parameters,
  caption:
    'Local kriging at every cell; hover a cell for its neighbors (spokes) and covariance range (ring).',
  createInputs: ({random, world}) => createInterpolationInputs(random, world),
  build(builder, {grid, samples, observations}, values) {
    const cellCount = grid.columns * grid.rows;
    const results = builder.allocate('results', 'float32x4', cellCount);
    builder.graph.add(
      new GPULocalKriging({
        positions: builder.upload('positions', 'float32x2', samples.positions).view,
        observations: builder.upload('observations', 'float32x4', observations).view,
        supports: builder.upload('supports', 'float32x2', grid.centers).view,
        mode: values.mode === 0 ? 'simple' : 'ordinary',
        means: [0.5],
        covariance: {model: 'exponential', range: values.range, sill: 0.06, nugget: 0.005},
        neighborCount: values.neighborCount,
        output: {values: results.view, status: builder.allocate('status', 'uint32', cellCount).view}
      })
    );
    return {
      outputs: {
        prediction: addChannel(builder, results, 0, 'prediction'),
        variance: addChannel(builder, results, 1, 'variance'),
        positions: builder.upload('sample-positions', 'float32x2', samples.positions),
        sampleValues: builder.upload('sample-values', 'float32', samples.values)
      }
    };
  },
  rebuildOn: ['mode', 'range', 'neighborCount'],
  // The hovered cell shows what is local: the neighborCount samples its kriging system was built from.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const {grid, samples} = inputs;
      const cell = getRasterCell(grid.columns, grid.rows, grid.window, pointer.x, pointer.y);
      const at = getRasterCellCenter(grid.columns, grid.rows, grid.window, cell.column, cell.row);
      const prediction = peek(outputs.prediction)?.[cell.index];
      const variance = peek(outputs.variance)?.[cell.index];
      return {
        glyphs: [
          ...createNeighborhoodGlyphs(samples.positions, at, values.neighborCount),
          {shape: 'ring', at, radius: values.range},
          {shape: 'crosshair', at}
        ],
        readout:
          prediction === undefined
            ? `neighborCount ${values.neighborCount}`
            : `neighborCount ${values.neighborCount} · ${prediction.toFixed(2)} ± ${Math.sqrt(Math.max(0, variance ?? 0)).toFixed(2)}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'raster',
      values: values.output === 1 ? outputs.variance : outputs.prediction,
      columns: inputs.grid.columns,
      rows: inputs.grid.rows,
      range: values.output === 1 ? 'auto' : [0, 1],
      smoothing: true
    },
    ...drawSampleDots(outputs.positions, outputs.sampleValues)
  ]
});
