// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMultiScaleRelief on a synthetic DEM: local relief averaged over a geometric series of window radii
// between featureMinimum and featureMaximum. The feature range and scaling factor set the radii at
// construction; the exaggeration is a per-frame value.

import {
  getGPUMultiScaleReliefParameterValues,
  GPUMultiScaleRelief
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './multi-scale-relief.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['featureMaximum', 'scalingFactor'],
  caption: 'Elevation in; relief averaged over many window sizes out.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const result = builder.allocate('result', 'float32', cellCount);
    builder.graph.add(
      new GPUMultiScaleRelief({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        resolution: 1,
        featureMinimum: 2,
        featureMaximum: values.featureMaximum,
        scalingFactor: values.scalingFactor,
        output: {relief: result.view}
      })
    );
    return {
      outputs: {elevation, result},
      update: current =>
        parameterBuffer.write(
          getGPUMultiScaleReliefParameterValues({
            verticalExaggeration: current.verticalExaggeration
          })
        )
    };
  },
  draw: ({inputs, outputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.result, {ramp: 'diverging', range: [-40, 40], smoothing: true})
  ]
});
