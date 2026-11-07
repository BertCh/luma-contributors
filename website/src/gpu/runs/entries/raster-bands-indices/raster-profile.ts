// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterProfile: three lines are resampled at a fixed spacing and the raster value is read at
// every sample. Dots show the samples on the map; the scatter plot is the elevation profile
// (value against distance along each line). Spacing and method are per-frame parameters.

import {
  getGPURasterProfileParameterValues,
  GPURasterProfile,
  type GPURasterSamplingMethod
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createSmoothField} from '../../inputs';
import {parameters} from './raster-profile.parameters';
import {drawRaster, makeWorldGrid} from './support/bands';
import {sampleField} from './support/zones';

const METHODS: GPURasterSamplingMethod[] = ['nearest', 'bilinear', 'bicubic'];
const SAMPLE_CAPACITY = 1500;

export default defineOperationRun({
  parameters,
  rebuildOn: ['gridWidth'],
  caption: 'A raster and three lines in; samples along each line and their profile out.',
  createInputs: ({random, world}) => {
    // Three gently winding lines of four vertices each, across the world.
    const positions: number[] = [];
    for (let line = 0; line < 3; line++) {
      const y = (line + 0.8) / 3.6;
      for (let vertex = 0; vertex < 4; vertex++) {
        const x = (0.04 + (0.92 * vertex) / 3) * world.width;
        positions.push(x, Math.min(0.97, Math.max(0.03, y + (random() - 0.5) * 0.18)));
      }
    }
    return {
      world,
      field: createSmoothField(random, 4, 3),
      positions: Float32Array.from(positions),
      lineOffsets: Uint32Array.of(0, 4, 8, 12)
    };
  },
  build(builder, inputs, values) {
    const {grid, columns, rows} = makeWorldGrid(inputs.world, values.gridWidth);
    const raster = builder.upload('raster', 'float32', sampleField(inputs, columns, rows));
    const positions = builder.upload('line-positions', 'float32x2', inputs.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', inputs.lineOffsets);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const count = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const requiredCount = builder.allocate('required-count', 'uint32', 1);
    const samplePositions = builder.allocate('sample-positions', 'float32x2', SAMPLE_CAPACITY);
    const sampleDistances = builder.allocate('sample-distances', 'float32', SAMPLE_CAPACITY);
    const sampleValues = builder.allocate('sample-values', 'float32', SAMPLE_CAPACITY);
    builder.graph.add(
      new GPURasterProfile({
        grid,
        values: raster.view,
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        sampleCapacity: SAMPLE_CAPACITY,
        parameters: parameterBuffer.view,
        output: {
          status: {count: count.view, overflow: overflow.view, requiredCount: requiredCount.view},
          samplePositions: samplePositions.view,
          sampleDistances: sampleDistances.view,
          sampleValues: sampleValues.view
        }
      })
    );
    return {
      outputs: {raster, samplePositions, sampleDistances, sampleValues, columns, rows},
      update: current =>
        parameterBuffer.write(
          getGPURasterProfileParameterValues(
            {spacing: current.spacing, method: METHODS[Math.round(current.method)]},
            grid
          )
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(outputs, outputs.raster, {context: true}),
    {
      layer: 'dots',
      positions: outputs.samplePositions,
      values: outputs.sampleValues,
      radius: 3.5,
      range: [0, 1]
    },
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.sampleValues,
      xValues: outputs.sampleDistances,
      xRange: [0, inputs.world.width],
      range: [0, 1],
      ramp: 'sequential',
      rectangle: [0.05, 0.03, 0.55, 0.3]
    }
  ]
});
