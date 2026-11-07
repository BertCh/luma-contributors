// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainDerivatives on a synthetic DEM: Horn's 3x3 slope, aspect (downslope direction) and
// hillshade from one pair of Sobel gradients. The output toggle picks which of the three buffers is
// drawn; the light and the z factor are per-frame settings, so dragging them re-runs the kernels
// without recompiling.

import {
  getGPUTerrainDerivativesParameterValues,
  GPUTerrainDerivatives
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './slope-aspect-hillshade.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Elevation in; slope, aspect and hillshade out, from the same Sobel gradients.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
  build(builder, terrain) {
    const {grid, band, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const slope = builder.allocate('slope', 'float32', cellCount);
    const aspect = builder.allocate('aspect', 'float32', cellCount);
    const hillshade = builder.allocate('hillshade', 'float32', cellCount);
    builder.graph.add(
      new GPUTerrainDerivatives({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        output: {slope: slope.view, aspect: aspect.view, hillshade: hillshade.view}
      })
    );
    return {
      outputs: {slope, aspect, hillshade},
      update: current =>
        parameterBuffer.write(
          getGPUTerrainDerivativesParameterValues(
            {
              zFactor: current.zFactor,
              azimuthDegrees: current.azimuthDegrees,
              altitudeDegrees: current.altitudeDegrees
            },
            grid
          )
        )
    };
  },
  draw: ({inputs, outputs, values}) => {
    if (values.view === 0) {
      return [drawGrid(inputs, outputs.slope, {ramp: 'sequential', range: [0, 60]})];
    }
    if (values.view === 1) {
      return [drawGrid(inputs, outputs.aspect, {ramp: 'cyclic', range: [0, 360]})];
    }
    return [drawGrid(inputs, outputs.hillshade, {range: [0, 1], smoothing: true})];
  }
});
