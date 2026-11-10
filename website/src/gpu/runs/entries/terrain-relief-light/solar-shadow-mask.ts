// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSolarShadowMask on a Grand Canyon DEM: GPUTerrainHorizon computes 16 horizon angles per cell once,
// then the mask looks the sun up against them for any azimuth and altitude without tracing
// again. The output is the illumination of every cell (slope toward the sun, shadow and ambient).

import {
  getGPUSolarShadowMaskParameterValues,
  GPUSolarShadowMask
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {aimAround} from '../../interaction';
import {addTerrainHorizon, addTerrainSlopeAspect} from './support/horizon';
import {drawRelief} from './support/relief';
import {createTerrainInput, uploadTerrain} from './support/terrain';

import {parameters} from './solar-shadow-mask.parameters';

const DIRECTION_COUNT = 16;

const aim = aimAround<unknown, {sunVisibility: RunBuffer}>(parameters, {
  hint: 'aim the sun',
  angle: 'azimuthDegrees',
  distance: {parameter: 'altitudeDegrees', invert: true}
});

export default defineOperationRun({
  parameters,
  caption:
    'Precomputed horizons and a sun in; sunlit and shadowed ground, with relief shading, out.',
  createInputs: ({world}) => createTerrainInput(world),
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
      outputs: {elevation, sunVisibility, illumination},
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
  // The pointer is the sun: compass angle is the azimuth, distance from the center the altitude.
  interaction: {
    ...aim,
    resolve(pointer, state) {
      const result = aim.resolve(pointer, state);
      const merged = {...state.values, ...result.values};
      const visibility = state.peek(state.outputs.sunVisibility);
      let shadow = '';
      if (visibility) {
        let shadowed = 0;
        for (const value of visibility) if (value < 0.5) shadowed++;
        shadow = ` · ${Math.round((100 * shadowed) / visibility.length)}% shadow`;
      }
      return {
        ...result,
        readout: `sun ${merged.azimuthDegrees}° · ${merged.altitudeDegrees}° up${shadow}`
      };
    }
  },
  draw: ({inputs, outputs}) => [drawRelief(inputs, outputs.illumination)]
});
