// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterReprojection resamples a raster (with a coordinate grid painted on it) into a
// destination grid. Here the destination grid is the source rotated and scaled about its center,
// so the output is the same scene turned on the screen; cells that fall outside the source are
// nodata. The destination grid is a construction prop, so the run rebuilds on every change.

import {
  GPURasterReprojection,
  makeGPURasterBand
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './raster-reprojection.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';
import {makeRotatedGrid, paintGridLines} from './support/reproject';

export default defineOperationRun({
  parameters,
  rebuildOn: ['rotation', 'scale', 'method'],
  caption: 'A raster in; resampled into a rotated, rescaled grid out.',
  createInputs: ({random, world}) => {
    const raster = createFractalRaster(random, world, 48, 5);
    return {...raster, painted: paintGridLines(raster)};
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const sourceGrid = makeGridOfRaster(inputs);
    const destinationGrid = makeRotatedGrid(inputs, values.rotation, values.scale);
    const source = builder.upload('source', 'float32', inputs.painted);
    const result = builder.allocate('values', 'float32', cellCount);
    const validity = builder.allocate('validity', 'uint32', cellCount);
    builder.graph.add(
      new GPURasterReprojection({
        sourceGrid,
        destinationGrid,
        input: makeGPURasterBand(source.view, sourceGrid, {id: 'source'}),
        method: values.method === 1 ? 'bilinear' : 'nearest',
        output: {values: result.view, validity: validity.view}
      })
    );
    return {outputs: {source, result}};
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.source, {context: true, range: [0, 1]}),
    drawRaster(inputs, outputs.result, {range: [0, 1]})
  ]
});
