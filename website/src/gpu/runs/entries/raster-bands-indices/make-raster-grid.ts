// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// makeGPURasterMetadata describes where a raster sits in the world. Here the Grand Canyon elevation raster
// is placed by the grid the sliders build, and a lattice of
// probe points reads the raster through GPURasterSampling, which takes the cell size, origin and
// row direction from that grid: cells under a probe light up, probes off the grid stay empty.
// `width` and `height` size the raster; the rest only changes the grid's affine. The pointer places
// the grid's origin; its outline and the row direction are drawn.

import {
  getGPURasterSamplingParameterValues,
  GPURasterSampling,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {sampleElevation, normalizeElevation} from '../../../data/elevation';
import {clamp, findParameter} from '../../interaction';
import {parameters} from './make-raster-grid.parameters';

const ORIGIN_X = findParameter(parameters, 'originX');
const ORIGIN_Y = findParameter(parameters, 'originY');

/** Hundredths, not the slider's step: the default origin (0.85) is not on that step grid. */
const getOrigin = (parameter: {min: number; max: number}, value: number) =>
  Math.round(100 * clamp(value, parameter.min, parameter.max)) / 100;

export default defineOperationRun({
  parameters,
  rebuildOn: ['width', 'height'],
  caption: 'Width, height, cell size, origin and row direction in; the placed raster out.',
  createInputs: ({world}) => {
    const columns = 110;
    const rows = Math.round(columns / world.width);
    const probes = new Float32Array(columns * rows * 2);
    for (let index = 0; index < columns * rows; index++) {
      probes[index * 2] = (((index % columns) + 0.5) / columns) * world.width;
      probes[index * 2 + 1] = 1 - (Math.floor(index / columns) + 0.5) / rows;
    }
    return {probes, probeCount: columns * rows, columns, rows};
  },
  build(builder, inputs, values) {
    const width = values.width;
    const height = values.height;
    const raster = normalizeElevation(sampleElevation(width, height).elevation);
    const rasterBuffer = builder.upload('raster', 'float32', raster);
    const probes = builder.upload('probes', 'float32x2', inputs.probes);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const sampled = builder.allocate('sampled', 'float32', inputs.probeCount);
    const initialGrid = makeGPURasterMetadata({width, height});
    builder.graph.add(
      new GPURasterSampling({
        grid: initialGrid,
        values: rasterBuffer.view,
        positions: probes.view,
        parameters: parameterBuffer.view,
        output: {values: sampled.view}
      })
    );
    return {
      outputs: {probes, sampled, columns: inputs.columns, rows: inputs.rows},
      update: current =>
        parameterBuffer.write(
          getGPURasterSamplingParameterValues(
            {method: 'nearest'},
            makeGPURasterMetadata({
              width,
              height,
              cellSize: current.cellSize,
              origin: [current.originX, current.originY],
              rowDirection: current.rowDirection === 1 ? 'north' : 'south'
            })
          )
        )
    };
  },
  interaction: {
    kind: 'place',
    hint: 'place the grid origin',
    rest: ({world}) => [ORIGIN_X.default / world.width, 1 - ORIGIN_Y.default],
    resolve: (pointer, {values, world}) => {
      const originX = getOrigin(ORIGIN_X, pointer.x);
      const originY = getOrigin(ORIGIN_Y, pointer.y);
      const spanX = values.width * values.cellSize;
      const spanY = values.height * values.cellSize;
      const south = values.rowDirection === 0;
      const top = south ? originY : originY + spanY;
      const bottom = south ? originY - spanY : originY;
      return {
        values: {originX, originY},
        glyphs: [
          {shape: 'crosshair', at: [originX, originY]},
          {shape: 'box', minimum: [originX, bottom], maximum: [originX + spanX, top]},
          {
            shape: 'arrow',
            from: [originX, originY],
            to: [originX, originY + (south ? -0.08 : 0.08)]
          }
        ],
        readout: `origin [${originX.toFixed(2)}, ${originY.toFixed(2)}] | ${values.width}x${values.height}`
      };
    }
  },
  // The probe lattice drawn as a raster: probes off the grid read nodata and stay empty.
  draw: ({outputs}) => [
    {
      layer: 'raster',
      values: outputs.sampled,
      columns: outputs.columns,
      rows: outputs.rows,
      ramp: 'sequential',
      range: [-0.35, 1]
    }
  ]
});
