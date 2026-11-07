// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainCurvature on a synthetic DEM: the chosen curvature of the surface per cell. Profile
// curvature is along the slope (acceleration), plan curvature across it (convergence); the
// differencing method is a construction prop, the z factor is written per frame.

import {
  getGPUTerrainCurvatureParameterValues,
  GPUTerrainCurvature,
  type GPUTerrainCurvatureKind,
  type GPUTerrainCurvatureMethod
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './terrain-curvature.parameters';

const METHODS: GPUTerrainCurvatureMethod[] = ['evans-young', 'zevenbergen-thorne', 'florinsky'];
const KINDS: GPUTerrainCurvatureKind[] = ['profile', 'plan', 'mean', 'gaussian'];

export default defineOperationRun({
  parameters,
  rebuildOn: ['method', 'curvature'],
  caption: 'Elevation in; the surface curvature out: convex ridges against concave valleys.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const curvature = builder.allocate('curvature', 'float32', cellCount);
    builder.graph.add(
      new GPUTerrainCurvature({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        method: METHODS[values.method],
        output: {curvatures: {[KINDS[values.curvature]]: curvature.view}}
      })
    );
    return {
      outputs: {elevation, curvature},
      update: current =>
        parameterBuffer.write(
          getGPUTerrainCurvatureParameterValues({zFactor: current.zFactor}, grid)
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.curvature, {
      ramp: 'diverging',
      range: [-0.003, 0.003],
      smoothing: true
    })
  ]
});
