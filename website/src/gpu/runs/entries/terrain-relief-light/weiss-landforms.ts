// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainWeissLandforms on a synthetic DEM: ten landform classes from topographic position at two
// scales (canyons, drainages, plains, slopes, ridges, mountain tops). The slope threshold is per
// frame; the two radii and the standardization are construction props.

import {
  getGPUTerrainWeissLandformsParameterValues,
  GPUTerrainWeissLandforms
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './weiss-landforms.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['smallScale', 'largeScale', 'standardization'],
  caption: 'Elevation in; ten Weiss landform classes out.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const landforms = builder.allocate('landforms', 'uint32', cellCount);
    builder.graph.add(
      new GPUTerrainWeissLandforms({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        smallScale: {radius: values.smallScale},
        largeScale: {radius: values.largeScale},
        standardization: values.standardization === 1 ? 'local' : 'global',
        output: {landforms: landforms.view}
      })
    );
    return {
      outputs: {elevation, landforms},
      update: current =>
        parameterBuffer.write(
          getGPUTerrainWeissLandformsParameterValues(
            {slopeThresholdDegrees: current.slopeThresholdDegrees},
            grid
          )
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.landforms, {alpha: 0.85})
  ]
});
