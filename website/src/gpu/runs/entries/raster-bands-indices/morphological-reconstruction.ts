// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterReconstruction: the marker is the mask lowered (dilation) or raised (erosion) by an
// offset, and the operation grows it back under the mask. Dilation with an offset cuts every
// peak lower than the offset flat (h-maxima); erosion fills shallow pits. The result is
// published only once the growth has converged, so too few `maximumIterations` leave it blank.
// Method and iterations are compile-time.

import {GPURasterReconstruction} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './morphological-reconstruction.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';

export default defineOperationRun({
  parameters,
  rebuildOn: ['method', 'maximumIterations'],
  caption: 'A mask and a marker in; the marker reconstructed under the mask out.',
  createInputs: ({random, world}) => createFractalRaster(random, world, 36, 6),
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const mask = builder.upload('mask', 'float32', inputs.values);
    const marker = builder.allocate('marker', 'float32', cellCount);
    const result = builder.allocate('values', 'float32', cellCount);
    const validity = builder.allocate('validity', 'uint32', cellCount);
    const converged = builder.allocate('converged', 'uint32', 1);
    const erosion = values.method === 1;
    builder.graph.add(
      new GPURasterReconstruction({
        grid: makeGridOfRaster(inputs),
        marker: marker.view,
        mask: mask.view,
        method: erosion ? 'erosion' : 'dilation',
        connectivity: 8,
        maximumIterations: values.maximumIterations,
        output: {values: result.view, validity: validity.view, status: {converged: converged.view}}
      })
    );
    return {
      outputs: {mask, marker, result},
      update: current =>
        marker.write(
          inputs.values.map(value => (erosion ? value + current.height : value - current.height))
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.mask, {context: true, range: [0, 1]}),
    drawRaster(inputs, outputs.result, {range: [0, 1], hillshade: 0.6, elevation: 1.5})
  ]
});
