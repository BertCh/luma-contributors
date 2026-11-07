// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSimpleLocalRelief on a synthetic DEM: elevation minus its local mean within a window radius,
// which flattens the regional trend and leaves hills and hollows. The radius and exaggeration are
// per-frame values; maximumRadius bounds them at construction.

import {
  getGPUSimpleLocalReliefParameterValues,
  GPUSimpleLocalRelief
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './simple-local-relief.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Elevation in; local relief (above or below the neighborhood mean) out.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const result = builder.allocate('result', 'float32', cellCount);
    builder.graph.add(
      new GPUSimpleLocalRelief({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        maximumRadius: 20,
        output: {relief: result.view}
      })
    );
    return {
      outputs: {elevation, result},
      update: current =>
        parameterBuffer.write(
          getGPUSimpleLocalReliefParameterValues({
            radius: current.radius,
            verticalExaggeration: current.verticalExaggeration
          })
        )
    };
  },
  draw: ({inputs, outputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.result, {ramp: 'diverging', range: [-90, 90], smoothing: true})
  ]
});
