// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTextureShading on a synthetic DEM: fractional-Laplacian texture shading, a weighted sum of
// Gaussian-blurred detail levels that brings out ridge and valley texture without a sun. The detail
// and gain are per-frame values; the number of levels is a construction prop.

import {
  getGPUTextureShadingParameterValues,
  GPUTextureShading
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './texture-shading.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['levelCount'],
  caption: 'Elevation in; texture shading (sunless relief) out.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
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
      outputs: {elevation, result},
      update: current =>
        parameterBuffer.write(
          getGPUTextureShadingParameterValues({detail: current.detail, gain: current.gain})
        )
    };
  },
  draw: ({inputs, outputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.result, {ramp: 'diverging', range: [-100, 100], smoothing: true})
  ]
});
