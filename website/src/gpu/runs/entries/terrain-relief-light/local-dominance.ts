// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalDominance on a synthetic DEM: how much an observer standing on each cell looks down on the
// ground within a ring of radii (archaeological relief visualization). The observer height is a
// per-frame value; the radii are construction props.

import {
  getGPULocalDominanceParameterValues,
  GPULocalDominance
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './local-dominance.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['minimumRadius', 'maximumRadius'],
  caption: 'Elevation in; the local dominance of every cell over its surroundings out.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const result = builder.allocate('result', 'float32', cellCount);
    builder.graph.add(
      new GPULocalDominance({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        minimumRadius: values.minimumRadius,
        maximumRadius: values.maximumRadius,
        radiusIncrement: 2,
        angularResolution: 20,
        output: {dominance: result.view}
      })
    );
    return {
      outputs: {elevation, result},
      update: current =>
        parameterBuffer.write(
          getGPULocalDominanceParameterValues({observerHeight: current.observerHeight})
        )
    };
  },
  draw: ({inputs, outputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.result, {smoothing: true})
  ]
});
