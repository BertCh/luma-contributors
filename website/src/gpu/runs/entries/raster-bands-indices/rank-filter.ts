// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterRankFilter on a speckled raster: every cell becomes the quantile of its window.
// Quantile 0.5 is the median (removes salt-and-pepper noise), 0 an erosion (minimum), 1 a
// dilation (maximum). `radius` and `quantile` are construction props, so the run rebuilds.

import {GPURasterRankFilter} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createElevationRaster} from '../../inputs';
import {parameters} from './rank-filter.parameters';
import {drawRaster, makeGridOfRaster, uploadBand} from './support/bands';
import {getSwipeResult, getCellAt, getCellBox, getCurtainX} from './support/gestures';

export default defineOperationRun({
  parameters,
  rebuildOn: ['radius', 'quantile'],
  caption: 'A speckled raster in; each cell replaced by a quantile of its neighborhood.',
  createInputs: ({random, world}) => {
    const raster = createElevationRaster(world, 64);
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
  // The pointer is a curtain between the speckled input and the filtered result, with the window.
  interaction: {
    kind: 'swipe',
    hint: 'swipe speckle and filter',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, values, world}) => {
      const cell = getCellAt(inputs, world, pointer);
      return getSwipeResult(
        pointer,
        `quantile ${values.quantile} of ${(2 * values.radius + 1) ** 2} cells`,
        [getCellBox(inputs, world, cell.column, cell.row, values.radius)]
      );
    }
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = getCurtainX(interaction, world);
    return [
      drawRaster(inputs, outputs.source, {range: [0, 1], clip: {side: 'left', x: curtain}}),
      drawRaster(inputs, outputs.result, {
        range: [0, 1],
        smoothing: false,
        clip: {side: 'right', x: curtain}
      })
    ];
  }
});
