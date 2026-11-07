// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainVectorRuggedness on a synthetic DEM: dispersion of surface normals in a moving window.
// The radius and z factor are per-frame values, so the sliders never recompile.

import {
  getGPUTerrainVectorRuggednessParameterValues,
  GPUTerrainVectorRuggedness
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './vector-ruggedness.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Elevation in; the vector ruggedness measure out: orientation scatter, not height.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
  build(builder, terrain) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const ruggedness = builder.allocate('ruggedness', 'float32', cellCount);
    builder.graph.add(
      new GPUTerrainVectorRuggedness({
        grid,
        elevation: band,
        maximumRadius: 7,
        parameters: parameterBuffer.view,
        output: {vectorRuggedness: ruggedness.view}
      })
    );
    return {
      outputs: {elevation, ruggedness},
      update: current =>
        parameterBuffer.write(
          getGPUTerrainVectorRuggednessParameterValues(
            {radius: current.radius, zFactor: current.zFactor},
            grid
          )
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.ruggedness, {smoothing: true})
  ]
});
