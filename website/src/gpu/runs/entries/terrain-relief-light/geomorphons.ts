// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeomorphons on a synthetic DEM: ten landform classes (flat, peak, ridge, shoulder, spur, slope,
// hollow, footslope, valley, pit) from line-of-sight patterns in eight directions. The flat
// threshold is per frame; the search radius and comparison rule are construction props.

import {
  getGPUGeomorphonsParameterValues,
  GPUGeomorphons,
  type GPUGeomorphonComparison
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './geomorphons.parameters';

// Index is the landform code (GPU_GEOMORPHON_FORMS); 0 marks an invalid edge cell and stays clear.
const FORM_COLORS = [
  [0, 0, 0, 0],
  [0.8, 0.8, 0.78, 1],
  [0.75, 0.1, 0.1, 1],
  [0.95, 0.4, 0.15, 1],
  [0.98, 0.7, 0.2, 1],
  [0.7, 0.8, 0.25, 1],
  [0.95, 0.93, 0.7, 1],
  [0.55, 0.8, 0.9, 1],
  [0.2, 0.65, 0.6, 1],
  [0.15, 0.35, 0.8, 1],
  [0.1, 0.1, 0.35, 1]
] as const;

/** Box-blurs the elevation so landforms are large and coherent instead of single-cell noise. */
function smoothElevation(terrain: ReturnType<typeof createTerrainInput>, passes: number): void {
  const {columns, rows, elevation} = terrain;
  for (let pass = 0; pass < passes; pass++) {
    const source = elevation.slice();
    for (let row = 1; row < rows - 1; row++) {
      for (let column = 1; column < columns - 1; column++) {
        let sum = 0;
        for (let offset = 0; offset < 9; offset++) {
          sum += source[(row + Math.floor(offset / 3) - 1) * columns + column + (offset % 3) - 1];
        }
        elevation[row * columns + column] = sum / 9;
      }
    }
  }
}

const COMPARISONS: GPUGeomorphonComparison[] = ['anglev1', 'anglev2', 'anglev2-distance'];

export default defineOperationRun({
  parameters,
  rebuildOn: ['searchRadius', 'comparison'],
  caption: 'Elevation in; one of ten landform classes per cell out.',
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(random, world);
    smoothElevation(terrain, 3);
    return terrain;
  },
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const forms = builder.allocate('forms', 'uint32', cellCount);
    builder.graph.add(
      new GPUGeomorphons({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        searchRadius: values.searchRadius,
        skipRadius: 1,
        comparison: COMPARISONS[values.comparison],
        output: {forms: forms.view}
      })
    );
    return {
      outputs: {elevation, forms},
      update: current =>
        parameterBuffer.write(
          getGPUGeomorphonsParameterValues(
            {flatThresholdDegrees: current.flatThresholdDegrees},
            grid
          )
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.forms, {ramp: 'categorical', categories: FORM_COLORS, alpha: 0.9})
  ]
});
