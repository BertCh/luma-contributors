// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainHorizon on a synthetic DEM: the horizon angle in every direction around every cell,
// reduced here to the sky view factor and the positive and negative openness. The directions and
// search radius are construction props; the z factor is written per frame.

import {
  getGPUTerrainHorizonParameterValues,
  GPUTerrainHorizon
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './horizon-angles.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['directionCount', 'maximumRadius'],
  caption: 'Elevation in; how much sky each cell sees, from its horizon in every direction, out.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const horizon = builder.allocate('horizon', 'float32', cellCount * values.directionCount);
    const skyViewFactor = builder.allocate('sky-view-factor', 'float32', cellCount);
    const positiveOpenness = builder.allocate('positive-openness', 'float32', cellCount);
    const negativeOpenness = builder.allocate('negative-openness', 'float32', cellCount);
    builder.graph.add(
      new GPUTerrainHorizon({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        directionCount: values.directionCount,
        maximumRadius: values.maximumRadius,
        output: {
          horizon: horizon.view,
          skyViewFactor: skyViewFactor.view,
          positiveOpenness: positiveOpenness.view,
          negativeOpenness: negativeOpenness.view
        }
      })
    );
    return {
      outputs: {elevation, skyViewFactor, positiveOpenness, negativeOpenness},
      update: current =>
        parameterBuffer.write(getGPUTerrainHorizonParameterValues({zFactor: current.zFactor}, grid))
    };
  },
  draw: ({inputs, outputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(
      inputs,
      [outputs.skyViewFactor, outputs.positiveOpenness, outputs.negativeOpenness][values.output],
      {smoothing: true}
    )
  ]
});
