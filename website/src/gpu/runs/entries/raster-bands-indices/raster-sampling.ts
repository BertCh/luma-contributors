// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterSampling: a coarse raster with nodata holes is sampled at scattered points. Dots take
// the sampled value; nearest reproduces cell blocks, bilinear and bicubic interpolate, and the
// nodata policy decides what happens next to a hole (strict drops the sample, renormalize
// reweights the valid neighbours). Method and policy are per-frame parameters.

import {
  getGPURasterSamplingParameterValues,
  GPURasterSampling,
  type GPURasterSamplingMethod
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createSmoothField, createUniformPoints} from '../../inputs';
import {parameters} from './raster-sampling.parameters';
import {drawRaster, makeWorldGrid} from './support/bands';
import {sampleField} from './support/zones';

const METHODS: GPURasterSamplingMethod[] = ['nearest', 'bilinear', 'bicubic'];

export default defineOperationRun({
  parameters,
  rebuildOn: ['gridWidth'],
  caption: 'A coarse raster in; its value interpolated at every point out.',
  createInputs: ({random, world}) => ({
    world,
    field: createSmoothField(random, 4, 3),
    holes: createSmoothField(random, 3, 4),
    points: createUniformPoints(random, world, 1800)
  }),
  build(builder, inputs, values) {
    const {grid, columns, rows} = makeWorldGrid(inputs.world, values.gridWidth);
    const raster = sampleField(inputs, columns, rows);
    const holes = sampleField({field: inputs.holes}, columns, rows);
    raster.forEach((_, index) => {
      if (holes[index] > 0.68) raster[index] = Number.NaN;
    });
    const rasterBuffer = builder.upload('raster', 'float32', raster);
    const positions = builder.upload('points', 'float32x2', inputs.points.positions);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const sampled = builder.allocate('sampled', 'float32', inputs.points.count);
    builder.graph.add(
      new GPURasterSampling({
        grid,
        values: rasterBuffer.view,
        positions: positions.view,
        parameters: parameterBuffer.view,
        output: {values: sampled.view}
      })
    );
    return {
      outputs: {rasterBuffer, positions, sampled, columns, rows},
      update: current =>
        parameterBuffer.write(
          getGPURasterSamplingParameterValues(
            {
              method: METHODS[Math.round(current.method)],
              noDataPolicy: current.noDataPolicy === 1 ? 'renormalize' : 'strict'
            },
            grid
          )
        )
    };
  },
  draw: ({outputs}) => [
    drawRaster(outputs, outputs.rasterBuffer, {context: true}),
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.sampled,
      radius: 4.5,
      range: [0, 1]
    }
  ]
});
