// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterRankFilter on a speckled raster: every cell becomes the quantile of its window.
// Quantile 0.5 is the median (removes salt-and-pepper noise), 0 an erosion (minimum), 1 a
// dilation (maximum). `radius` and `quantile` are construction props, so the run rebuilds.

import {GPURasterRankFilter} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './rank-filter.parameters';
import {drawRaster, makeGridOfRaster, uploadBand} from './support/bands';

export default defineOperationRun({
  parameters,
  rebuildOn: ['radius', 'quantile'],
  caption: 'A speckled raster in; each cell replaced by a quantile of its neighborhood.',
  createInputs: ({random, world}) => {
    const raster = createFractalRaster(random, world, 64, 4);
    const draws = Float32Array.from(raster.values, () => random());
    const flips = Float32Array.from(raster.values, () => (random() < 0.5 ? 0 : 1));
    return {...raster, draws, flips};
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const source = uploadBand(builder, inputs, 'source', inputs.values);
    const result = builder.allocate('values', 'float32', cellCount);
    const validity = builder.allocate('validity', 'uint32', cellCount);
    builder.graph.add(
      new GPURasterRankFilter({
        grid: makeGridOfRaster(inputs),
        input: source.band,
        radius: values.radius,
        quantile: values.quantile,
        output: {values: result.view, validity: validity.view}
      })
    );
    return {
      outputs: {source: source.buffer, result},
      update: current =>
        source.buffer.write(
          Float32Array.from(inputs.values, (value, index) =>
            inputs.draws[index] < current.noise ? inputs.flips[index] : value
          )
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.source, {context: true, range: [0, 1]}),
    drawRaster(inputs, outputs.result, {range: [0, 1], smoothing: false})
  ]
});
