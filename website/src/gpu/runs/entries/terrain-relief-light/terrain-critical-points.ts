// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainCriticalPoints on a synthetic DEM with measurement noise: every cell is classified as
// regular, peak, pit, saddle or boundary from its neighbour sign changes, and a class histogram is
// reduced on the GPU. Connectivity is a construction prop; the noise slider rewrites the input.

import {GPUTerrainCriticalPoints} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {
  addTerrainNoise,
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  uploadTerrain
} from './support/terrain';
import {parameters} from './terrain-critical-points.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['connectivity'],
  caption: 'Noisy elevation in; peaks, pits and saddles per cell, and their counts, out.',
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(random, world, 48);
    return {...terrain, noise: Float32Array.from(terrain.elevation, () => random())};
  },
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const classes = builder.allocate('classes', 'uint32', cellCount);
    const counts = builder.allocate('counts', 'uint32', 6);
    builder.graph.add(
      new GPUTerrainCriticalPoints({
        grid,
        elevation: band,
        connectivity: values.connectivity === 1 ? 6 : 8,
        output: {classes: classes.view, counts: counts.view}
      })
    );
    return {
      outputs: {elevation, classes, counts},
      update: current => elevation.write(addTerrainNoise(terrain, terrain.noise, current.noise))
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    // Regular cells (class 0) fade out; peaks, pits and saddles stay.
    drawGrid(inputs, outputs.classes, {ramp: 'sequential', range: [0, 3], fade: true}),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.counts,
      rectangle: [0.62, 0.06, 0.97, 0.3],
      logarithmic: true,
      range: 'auto'
    }
  ]
});
