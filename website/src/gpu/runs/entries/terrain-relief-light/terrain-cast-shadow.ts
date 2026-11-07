// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainCastShadow on a synthetic DEM: for every cell, the fraction of the sun's disk that
// terrain along the sun direction leaves visible (1 in full sun, 0 in full shadow, fractions in
// the penumbra). Every sun property is a per-frame value.

import {
  getGPUTerrainCastShadowParameterValues,
  GPUTerrainCastShadow
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './terrain-cast-shadow.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Elevation and a low sun in; the shadow terrain casts, with penumbra, out.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
  build(builder, terrain) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 16);
    const sunVisibility = builder.allocate('sun-visibility', 'float32', cellCount);
    builder.graph.add(
      new GPUTerrainCastShadow({
        grid,
        elevation: band,
        maximumRadius: 60,
        parameters: parameterBuffer.view,
        output: {sunVisibility: sunVisibility.view}
      })
    );
    return {
      outputs: {elevation, sunVisibility},
      update: current =>
        parameterBuffer.write(
          getGPUTerrainCastShadowParameterValues(
            {
              azimuthDegrees: current.azimuthDegrees,
              altitudeDegrees: current.altitudeDegrees,
              angularRadiusDegrees: current.angularRadiusDegrees,
              zFactor: current.zFactor
            },
            grid
          )
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.sunVisibility, {range: [0, 1], smoothing: true})
  ]
});
