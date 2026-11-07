// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterFillHoles on a binary mask (land above a threshold, peppered with pits): background
// regions not connected to the border are filled. Few iterations
// leave the result unpublished: it is only written once the fill has converged.

import {GPURasterFillHoles} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './fill-holes.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';

export default defineOperationRun({
  parameters,
  rebuildOn: ['maximumIterations'],
  caption: 'A binary mask in; enclosed holes filled out.',
  createInputs: ({random, world}) => {
    const raster = createFractalRaster(random, world, 72, 6);
    return {...raster, draws: Float32Array.from(raster.values, () => random())};
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const mask = builder.allocate('mask', 'uint32', cellCount);
    const result = builder.allocate('values', 'uint32', cellCount);
    const converged = builder.allocate('converged', 'uint32', 1);
    builder.graph.add(
      new GPURasterFillHoles({
        grid: makeGridOfRaster(inputs),
        input: mask.view,
        connectivity: 8,
        maximumIterations: values.maximumIterations,
        output: {values: result.view, status: {converged: converged.view}}
      })
    );
    return {
      outputs: {mask, result},
      update: current =>
        mask.write(
          Uint32Array.from(inputs.values, (value, index) =>
            value > current.threshold && inputs.draws[index] >= current.pits ? 1 : 0
          )
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.mask, {context: true}),
    drawRaster(inputs, outputs.result)
  ]
});
