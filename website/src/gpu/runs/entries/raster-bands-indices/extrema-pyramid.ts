// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterExtremaPyramid: a min/max pyramid of a raster whose first level has blocks of
// `firstBlockSize` cells; every further level doubles the block. Level 0 is drawn (the pyramid
// stores levels back to back, so level 0 is the first rows of each output): coarse blocks hold the
// extremes of the cells they cover, so peaks survive downsampling. `bilinear` footprints widen
// each block so the extremes also cover bilinear resampling.

import {
  getGPURasterExtremaPyramidLayout,
  GPURasterExtremaPyramid,
  makeGPURasterBand
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './extrema-pyramid.parameters';
import {makeGridOfRaster} from './support/bands';

export default defineOperationRun({
  parameters,
  rebuildOn: ['firstBlockSize', 'footprint'],
  caption: 'A raster in; the maximum and minimum of every block out (level 0).',
  createInputs: ({random, world}) => createFractalRaster(random, world, 64, 6),
  build(builder, inputs, values) {
    const grid = makeGridOfRaster(inputs);
    const firstBlockSize = 2 ** Math.round(values.firstBlockSize);
    const options = {
      firstBlockSize,
      footprint: values.footprint === 1 ? ('bilinear' as const) : ('cell' as const)
    };
    const layout = getGPURasterExtremaPyramidLayout(grid, options);
    const source = builder.upload('source', 'float32', inputs.values);
    const maximum = builder.allocate('maximum', 'float32', layout.length);
    const minimum = builder.allocate('minimum', 'float32', layout.length);
    builder.graph.add(
      new GPURasterExtremaPyramid({
        grid,
        input: makeGPURasterBand(source.view, grid, {id: 'source'}),
        ...options,
        output: {maximum: maximum.view, minimum: minimum.view}
      })
    );
    return {
      outputs: {
        source,
        maximum,
        minimum,
        columns: layout.levels[0].width,
        rows: layout.levels[0].height
      }
    };
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.source,
      columns: inputs.columns,
      rows: inputs.rows,
      range: [0, 1]
    },
    {
      layer: 'raster',
      values: values.statistic === 0 ? outputs.maximum : outputs.minimum,
      columns: outputs.columns,
      rows: outputs.rows,
      range: [0, 1]
    }
  ]
});
