// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUReliefShading on a synthetic DEM: a cartographic shaded relief that blends a hillshade from one
// light with the sky view factor (ambient occlusion) in a single fused kernel. Everything the
// reader tunes is a per-frame setting.

import {
  getGPUReliefShadingParameterValues,
  GPUReliefShading
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {addTerrainHorizon} from './support/horizon';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './relief-shading.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Elevation and a sky view factor in; a lit, shaded relief out.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
  build(builder, terrain) {
    const uploaded = uploadTerrain(builder, terrain);
    const {grid, band, elevation, cellCount} = uploaded;
    const {skyViewFactor} = addTerrainHorizon(builder, uploaded, {
      directionCount: 16,
      maximumRadius: 30
    });
    const parameterBuffer = builder.allocate('parameters', 'float32', 80);
    const relief = builder.allocate('relief', 'float32', cellCount);
    const color = builder.allocate('color', 'uint32', cellCount);
    builder.graph.add(
      new GPUReliefShading({
        grid,
        elevation: band,
        skyViewFactor: skyViewFactor.view,
        parameters: parameterBuffer.view,
        output: {relief: relief.view, color: color.view}
      })
    );
    return {
      outputs: {elevation, relief},
      update: current =>
        parameterBuffer.write(
          getGPUReliefShadingParameterValues(
            {
              lights: [{azimuthDegrees: current.azimuthDegrees, altitudeDegrees: 40}],
              skyViewStrength: current.skyViewStrength,
              exposure: current.exposure,
              zFactor: current.zFactor
            },
            grid
          )
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.relief, {range: [0, 1], smoothing: true})
  ]
});
