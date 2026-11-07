// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// makeGPURasterMetadata describes where a raster sits in the world. Here a small gradient raster
// (its pixel (0, 0) corner is bright) is placed by the grid the sliders build, and a lattice of
// probe points reads the raster through GPURasterSampling, which takes the cell size, origin and
// row direction from that grid: cells under a probe light up, probes off the grid stay empty.
// `width` and `height` size the raster; the rest only changes the grid's affine.

import {
  getGPURasterSamplingParameterValues,
  GPURasterSampling,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './make-raster-grid.parameters';

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
      probes[index * 2 + 1] = (Math.floor(index / columns) + 0.5) / rows;
    }
    return {probes, probeCount: columns * rows};
  },
  build(builder, inputs, values) {
    const width = values.width;
    const height = values.height;
    const raster = Float32Array.from({length: width * height}, (_, index) => {
      const column = index % width;
      const row = Math.floor(index / width);
      // A gradient, with the pixel (0, 0) corner marked so the row direction is visible.
      return column < 2 && row < 2 ? 2 : 0.15 + 0.5 * (column / width) + 0.35 * (row / height);
    });
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
      outputs: {probes, sampled},
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
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.probes, values: outputs.sampled, radius: 3.2, range: [0, 1]}
  ]
});
