// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTextureShading on a Grand Canyon DEM: fractional-Laplacian texture shading, a weighted sum of
// Gaussian-blurred detail levels that brings out ridge and valley texture without a sun. The detail
// and gain are per-frame values; the number of levels is a construction prop.

import {
  getGPUTextureShadingParameterValues,
  GPUTextureShading
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {bindAxes} from '../../interaction';
import {drawRelief} from './support/relief';
import {createTerrainInput, uploadTerrain} from './support/terrain';
import {parameters} from './texture-shading.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['levelCount'],
  caption: 'Elevation in; texture shading (sunless relief) out.',
  createInputs: ({world}) => createTerrainInput(world, 96),
  build(builder, terrain, values) {
    const {grid, band, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const result = builder.allocate('result', 'float32', cellCount);
    builder.graph.add(
      new GPUTextureShading({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        levelCount: values.levelCount,
        output: {textureShade: result.view}
      })
    );
    return {
      outputs: {result},
      update: current =>
        parameterBuffer.write(
          getGPUTextureShadingParameterValues({detail: current.detail, gain: current.gain})
        )
    };
  },
  // The pointer sets the two per-frame props: x is the detail exponent, y the gain.
  interaction: bindAxes(parameters, {
    kind: 'pad',
    hint: 'detail and gain',
    x: 'detail',
    y: 'gain',
    decorate: (values, pointer) => ({
      glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
      readout: `detail ${values.detail.toFixed(1)} · gain ${values.gain.toFixed(1)}`
    })
  }),
  draw: ({inputs, outputs}) => [drawRelief(inputs, outputs.result, [-250, 250])]
});
