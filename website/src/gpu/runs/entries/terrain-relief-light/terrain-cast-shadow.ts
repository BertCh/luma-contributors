// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainCastShadow on a Grand Canyon DEM: for every cell, the fraction of the sun's disk that
// terrain along the sun direction leaves visible (1 in full sun, 0 in full shadow, fractions in
// the penumbra). Every sun property is a per-frame value.

import {
  getGPUTerrainCastShadowParameterValues,
  GPUTerrainCastShadow
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {aimAround} from '../../interaction';
import {drawRelief} from './support/relief';
import {createTerrainInput, uploadTerrain} from './support/terrain';

import {parameters} from './terrain-cast-shadow.parameters';

const aim = aimAround<unknown, {sunVisibility: RunBuffer}>(parameters, {
  hint: 'aim the sun',
  angle: 'azimuthDegrees',
  distance: {parameter: 'altitudeDegrees', invert: true}
});

export default defineOperationRun({
  parameters,
  caption: 'Elevation and a low sun in; the shadow terrain casts, with penumbra, out.',
  createInputs: ({world}) => createTerrainInput(world),
  build(builder, terrain) {
    const {grid, band, cellCount} = uploadTerrain(builder, terrain);
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
      outputs: {sunVisibility},
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
        shadow = ` · ${Math.round((100 * shadowed) / visibility.length)}% shadowed`;
      }
      return {
        ...result,
        readout: `sun ${merged.azimuthDegrees}° · ${merged.altitudeDegrees}° up${shadow}`
      };
    }
  },
  draw: ({inputs, outputs}) => [drawRelief(inputs, outputs.sunVisibility)]
});
