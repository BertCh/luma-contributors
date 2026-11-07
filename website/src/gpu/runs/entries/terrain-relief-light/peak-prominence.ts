// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainPeakProminence on a synthetic DEM: for every local maximum, how far it rises above the
// saddle that connects it to higher ground, from a GPU union-find over the elevation order. The
// operation has no props, so the sliders reshape the surface and pick the output shown.

import {GPUTerrainPeakProminence} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {
  addTerrainNoise,
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  uploadTerrain
} from './support/terrain';
import {parameters} from './peak-prominence.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Elevation in; prominence and key saddle height at every peak out.',
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(random, world, 28);
    return {...terrain, noise: Float32Array.from(terrain.elevation, () => random())};
  },
  build(builder, terrain) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const prominence = builder.allocate('prominence', 'float32', cellCount);
    const keySaddle = builder.allocate('key-saddle', 'float32', cellCount);
    const parentPeak = builder.allocate('parent-peak', 'uint32', cellCount);
    builder.graph.add(
      new GPUTerrainPeakProminence({
        grid,
        elevation: band,
        output: {
          prominence: prominence.view,
          keySaddle: keySaddle.view,
          parentPeak: parentPeak.view
        }
      })
    );
    return {
      outputs: {elevation, prominence, keySaddle},
      update: current => {
        const surface = addTerrainNoise(terrain, terrain.noise, current.noise);
        for (let index = 0; index < surface.length; index++) surface[index] *= current.relief;
        elevation.write(surface);
      }
    };
  },
  draw: ({inputs, outputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    // Cells that are not peaks have zero prominence and fade out.
    drawGrid(inputs, values.output === 0 ? outputs.prominence : outputs.keySaddle, {
      fade: true,
      range: values.output === 0 ? [0, 60] : 'auto'
    })
  ]
});
