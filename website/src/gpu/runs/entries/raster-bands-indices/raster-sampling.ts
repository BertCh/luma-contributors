// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterSampling: a coarse raster with nodata holes is sampled at scattered points. Dots take
// the sampled value; nearest reproduces cell blocks, bilinear and bicubic interpolate, and the
// nodata policy decides what happens next to a hole (propagate drops the sample, renormalize
// reweights the valid neighbours). Method and policy are per-frame parameters. The pointer places
// the first sample point, a pointer-owned input; its support cells are outlined.

import {
  getGPURasterSamplingParameterValues,
  GPURasterSampling,
  type GPURasterSamplingMethod
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createSmoothField} from '../../inputs';
import {sampleElevation, normalizeElevation} from '../../../data/elevation';
import {formatNumber} from '../../interaction';
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
    holes: createSmoothField(random, 3, 4),
    query: [world.width / 2, world.height / 2] as [number, number],
    points: {
      count: 600,
      positions: Float32Array.from(
        {length: 1200},
        (_, index) => random() * (index % 2 === 0 ? world.width : world.height)
      )
    }
  }),
  build(builder, inputs, values) {
    const {grid, columns, rows} = makeWorldGrid(inputs.world, values.gridWidth);
    const raster = normalizeElevation(sampleElevation(columns, rows).elevation);
    const holes = sampleField({field: inputs.holes}, columns, rows);
    raster.forEach((_, index) => {
      if (holes[index] > 0.6) raster[index] = Number.NaN;
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
      update: current => {
        const points = Float32Array.from(inputs.points.positions);
        points.set(inputs.query, 0);
        positions.write(points);
        parameterBuffer.write(
          getGPURasterSamplingParameterValues(
            {
              method: METHODS[Math.round(current.method)],
              noDataPolicy: current.noDataPolicy === 1 ? 'renormalize' : 'propagate'
            },
            grid
          )
        );
      }
    };
  },
  interaction: {
    kind: 'place',
    hint: 'place a sample point',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, values, world, peek}) => {
      inputs.query = [pointer.x, pointer.y];
      const sampled = peek(outputs.sampled);
      const method = METHODS[Math.round(values.method)];
      // Support: 1 cell for nearest, 2 x 2 for bilinear, 4 x 4 for bicubic.
      const reach = method === 'nearest' ? 0 : method === 'bilinear' ? 1 : 2;
      const cellWidth = world.width / outputs.columns;
      const cellHeight = world.height / outputs.rows;
      const column = pointer.x / cellWidth;
      const row = (world.height - pointer.y) / cellHeight;

      const lowColumn =
        method === 'nearest' ? Math.floor(column) : Math.floor(column - 0.5) - (reach - 1);
      const lowRow = method === 'nearest' ? Math.floor(row) : Math.floor(row - 0.5) - (reach - 1);
      const span = method === 'nearest' ? 1 : 2 * reach;
      return {
        rerun: true,
        glyphs: [
          {shape: 'crosshair', at: inputs.query},
          {
            shape: 'box',
            minimum: [
              Math.max(0, lowColumn) * cellWidth,
              world.height - Math.min(outputs.rows, lowRow + span) * cellHeight
            ],
            maximum: [
              Math.min(outputs.columns, lowColumn + span) * cellWidth,
              world.height - Math.max(0, lowRow) * cellHeight
            ]
          }
        ],
        readout: `${method} ${sampled && Number.isFinite(sampled[0]) ? formatNumber(sampled[0]) : 'nodata'} | ${span * span} support cells`
      };
    }
  },
  draw: ({outputs}) => [
    drawRaster(outputs, outputs.rasterBuffer, {ramp: 'grey', range: [0, 1]}),
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.sampled,
      radius: 4.5,
      range: [0, 1]
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.sampled,
      count: 1,
      radius: 9,
      range: [0, 1]
    }
  ]
});
