// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterSieve on a patch label raster: patches smaller than `minimumPixels` are removed (set
// to background) or merged into their largest neighbor. The size threshold is a per-frame
// parameter; mode and connectivity are construction props (the run rebuilds).

import {
  getGPURasterSieveParameterValues,
  GPURasterSieve
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './sieve.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';
import {createPatchSeeds, getPatchLabels} from './support/patches';

const PATCH_COUNT = 90;

export default defineOperationRun({
  parameters,
  rebuildOn: ['mode', 'connectivity'],
  caption: 'Patch labels in; small patches removed or merged into neighbors out.',
  createInputs: ({random, world}) => {
    const raster = createFractalRaster(random, world, 54, 1);
    return {...raster, seeds: createPatchSeeds(random, raster.columns, raster.rows, PATCH_COUNT)};
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const labels = builder.upload('labels', 'uint32', getPatchLabels(inputs.seeds, PATCH_COUNT));
    const parameterBuffer = builder.allocate('parameters', 'uint32', 4);
    const sieved = builder.allocate('sieved', 'uint32', cellCount);
    builder.graph.add(
      new GPURasterSieve({
        grid: makeGridOfRaster(inputs),
        labels: labels.view,
        patchCapacity: PATCH_COUNT + 1,
        mode: values.mode === 1 ? 'merge' : 'remove',
        connectivity: values.connectivity === 1 ? 8 : 4,
        parameters: parameterBuffer.view,
        output: {labels: sieved.view}
      })
    );
    return {
      outputs: {labels, sieved},
      update: current =>
        parameterBuffer.write(
          getGPURasterSieveParameterValues({minimumPixels: current.minimumPixels})
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.labels, {context: true}),
    drawRaster(inputs, outputs.sieved)
  ]
});
