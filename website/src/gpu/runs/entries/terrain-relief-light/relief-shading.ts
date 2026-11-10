// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUReliefShading on a Grand Canyon DEM: a cartographic shaded relief that blends a hillshade from one
// light with the sky view factor (ambient occlusion) in a single fused kernel. Everything the
// reader tunes is a per-frame setting.

import {
  getGPUReliefShadingParameterValues,
  GPUReliefShading
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {aimAround} from '../../interaction';
import {addTerrainHorizon} from './support/horizon';
import {createTerrainInput, uploadTerrain} from './support/terrain';
import {drawRelief} from './support/relief';
import {parameters} from './relief-shading.parameters';

export default defineOperationRun({
  parameters,
  caption:
    'Elevation and a sky view factor in; a lit relief out, darkened in hollows by the sky view factor.',
  createInputs: ({world}) => createTerrainInput(world),
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
              lights: [
                {azimuthDegrees: current.azimuthDegrees, altitudeDegrees: current.altitudeDegrees}
              ],
              skyViewStrength: current.skyViewStrength,
              exposure: current.exposure,
              zFactor: current.zFactor
            },
            grid
          )
        )
    };
  },
  // The pointer is the light: its compass angle around the center is the azimuth, and the farther
  // out it sits the lower the sun.
  interaction: aimAround(parameters, {
    hint: 'aim the light',
    angle: 'azimuthDegrees',
    distance: {parameter: 'altitudeDegrees', invert: true},
    readout: values => `light ${values.azimuthDegrees}° · ${values.altitudeDegrees}° up`
  }),
  draw: ({inputs, outputs}) => [drawRelief(inputs, outputs.relief)]
});
