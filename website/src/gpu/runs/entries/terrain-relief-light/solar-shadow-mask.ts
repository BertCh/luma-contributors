// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSolarShadowMask on a synthetic DEM: GPUTerrainHorizon computes 16 horizon angles per cell once,
// then the mask looks the sun up against them for any azimuth and altitude without tracing
// again. The output is the illumination of every cell (slope toward the sun, shadow and ambient).

import {
  getGPUSolarShadowMaskParameterValues,
  GPUSolarShadowMask
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {addTerrainHorizon, addTerrainSlopeAspect} from './support/horizon';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './solar-shadow-mask.parameters';

const DIRECTION_COUNT = 16;

export default defineOperationRun({
  parameters,
  caption:
    'Precomputed horizons and a sun in; sunlit and shadowed ground, with relief shading, out.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
  build(builder, terrain) {
    const uploaded = uploadTerrain(builder, terrain);
    const {grid, elevation, cellCount} = uploaded;
    const {horizon, skyViewFactor} = addTerrainHorizon(builder, uploaded, {
      directionCount: DIRECTION_COUNT,
      maximumRadius: 40
    });
    const {slope, aspect} = addTerrainSlopeAspect(builder, uploaded);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const sunVisibility = builder.allocate('sun-visibility', 'float32', cellCount);
    const illumination = builder.allocate('illumination', 'float32', cellCount);
    builder.graph.add(
      new GPUSolarShadowMask({
        grid,
        directionCount: DIRECTION_COUNT,
        horizon: horizon.view,
        slope: slope.view,
        aspect: aspect.view,
        skyViewFactor: skyViewFactor.view,
        parameters: parameterBuffer.view,
        output: {sunVisibility: sunVisibility.view, illumination: illumination.view}
      })
    );
    return {
      outputs: {elevation, illumination},
      update: current =>
        parameterBuffer.write(
          getGPUSolarShadowMaskParameterValues({
            azimuthDegrees: current.azimuthDegrees,
            altitudeDegrees: current.altitudeDegrees,
            ambientIntensity: current.ambientIntensity
          })
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.illumination, {range: [0, 1], smoothing: true})
  ]
});
