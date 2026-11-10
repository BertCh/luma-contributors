// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterProfile: a line is resampled at a fixed spacing and the raster value is read at every
// sample. Dots show the samples on the map; the framed scatter plot is the elevation profile (value
// against distance along the line). Spacing and method are per-frame parameters; the pointer drags
// the line's end vertex, a pointer-owned input, from an anchor at the left edge.

import {
  getGPURasterProfileParameterValues,
  GPURasterProfile,
  type GPURasterSamplingMethod
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {sampleElevation, normalizeElevation} from '../../../data/elevation';
import {formatNumber} from '../../interaction';
import {parameters} from './raster-profile.parameters';
import {drawRaster, makeWorldGrid} from './support/bands';

const METHODS: GPURasterSamplingMethod[] = ['nearest', 'bilinear', 'bicubic'];
const SAMPLE_CAPACITY = 1500;
const CHART = [0.05, 0.04, 0.5, 0.3] as const;
const MAXIMUM_LENGTH = 2.2;

export default defineOperationRun({
  parameters,
  rebuildOn: ['gridWidth'],
  caption: 'A raster and three lines in; samples along each line and their profile out.',
  createInputs: ({world}) => ({
    world,
    anchor: [0.04 * world.width, 0.5] as [number, number],
    end: [0.96 * world.width, 0.55] as [number, number]
  }),
  build(builder, inputs, values) {
    const {grid, columns, rows} = makeWorldGrid(inputs.world, values.gridWidth);
    const raster = builder.upload(
      'raster',
      'float32',
      normalizeElevation(sampleElevation(columns, rows).elevation)
    );
    const positions = builder.allocate('line-positions', 'float32x2', 2);
    const lineOffsets = builder.upload('line-offsets', 'uint32', Uint32Array.of(0, 2));
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
      outputs: {raster, samplePositions, sampleDistances, sampleValues, count, columns, rows},
      update: current => {
        positions.write(Float32Array.from([...inputs.anchor, ...inputs.end]));
        parameterBuffer.write(
          getGPURasterProfileParameterValues(
            {spacing: current.spacing, method: METHODS[Math.round(current.method)]},
            grid
          )
        );
      }
    };
  },
  interaction: {
    kind: 'drag',
    hint: 'drag the profile line',
    rest: [0.96, 0.45],
    resolve: (pointer, {inputs, outputs, world, peek}) => {
      inputs.end = [pointer.x, pointer.y];
      const count = peek(outputs.count);
      const sampleValues = peek(outputs.sampleValues);
      let relief = 0;
      if (count && sampleValues) {
        let low = Infinity;
        let high = -Infinity;
        for (let sample = 0; sample < count[0]; sample++) {
          low = Math.min(low, sampleValues[sample]);
          high = Math.max(high, sampleValues[sample]);
        }
        relief = high - low;
      }
      const length = Math.hypot(inputs.end[0] - inputs.anchor[0], inputs.end[1] - inputs.anchor[1]);
      return {
        rerun: true,
        glyphs: [
          {shape: 'line', from: inputs.anchor, to: inputs.end},
          {shape: 'crosshair', at: inputs.end},
          {
            shape: 'box',
            minimum: [world.width * CHART[0], world.height * CHART[1]],
            maximum: [world.width * CHART[2], world.height * CHART[3]]
          }
        ],
        readout: `length ${formatNumber(length)} | ${count ? count[0] : '-'} samples | relief ${formatNumber(relief)}`
      };
    }
  },
  draw: ({outputs}) => [
    drawRaster(outputs, outputs.raster, {ramp: 'sequential', range: [0, 1], alpha: 0.8}),
    {
      layer: 'dots',
      positions: outputs.samplePositions,
      values: outputs.sampleValues,
      count: SAMPLE_CAPACITY,
      radius: 4,
      range: [0, 1],
      ramp: 'diverging'
    },
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.sampleValues,
      xValues: outputs.sampleDistances,
      xRange: [0, MAXIMUM_LENGTH],
      range: [0, 1],
      color: [1, 1, 1, 0.95],
      radius: 3,
      rectangle: CHART
    }
  ]
});
