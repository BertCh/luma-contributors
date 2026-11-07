// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainSummits on a synthetic DEM: a cell is a summit when it is the highest within the radius
// and drops by at least minimumDrop on every side. The radius and drop are per-frame values;
// the border rule is a construction prop.

import {
  getGPUTerrainSummitsParameterValues,
  GPUTerrainSummits
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {
  addTerrainNoise,
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  uploadTerrain
} from './support/terrain';
import {parameters} from './terrain-summits.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['incompleteNeighborhood'],
  caption: "Elevation in; the summit mask and each summit's drop out.",
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(random, world, 30);
    return {...terrain, noise: Float32Array.from(terrain.elevation, () => random())};
  },
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const summitMask = builder.allocate('summit-mask', 'uint32', cellCount);
    const drop = builder.allocate('drop', 'float32', cellCount);
    builder.graph.add(
      new GPUTerrainSummits({
        grid,
        elevation: band,
        maximumRadiusPixels: 16,
        incompleteNeighborhood: values.incompleteNeighborhood === 1 ? 'ignore' : 'reject',
        parameters: parameterBuffer.view,
        output: {summitMask: summitMask.view, drop: drop.view}
      })
    );
    return {
      outputs: {elevation, drop, summitMask},
      update: current => {
        elevation.write(addTerrainNoise(terrain, terrain.noise, current.noise));
        parameterBuffer.write(
          getGPUTerrainSummitsParameterValues(
            {radius: current.radius, minimumDrop: current.minimumDrop},
            grid
          )
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    // Cells that are not summits have zero drop and fade out.
    drawGrid(inputs, outputs.summitMask, {ramp: 'sequential', range: [0, 1], fade: true})
  ]
});
