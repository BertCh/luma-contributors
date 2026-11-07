// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainCumulativeViewshed on a synthetic DEM: how many of a set of observers (towers on the
// highest summits) can see each cell. The observer count is a construction prop; their height
// and range are per-frame values.

import {
  getGPUTerrainLineOfSightParameterValues,
  GPUTerrainCumulativeViewshed
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  findLocalMaxima,
  getCellWorldPosition,
  uploadTerrain
} from './support/terrain';
import {parameters} from './cumulative-viewshed.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['observerCount'],
  caption: 'Elevation and several towers in; how many towers see each cell out.',
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(random, world);
    return {...terrain, summits: findLocalMaxima(terrain, 8)};
  },
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const summits = terrain.summits.slice(0, values.observerCount);
    const observers = builder.upload(
      'observers',
      'float32x2',
      Float32Array.from(
        summits.flatMap(index => [index % terrain.columns, Math.floor(index / terrain.columns)])
      )
    );
    const towers = builder.upload(
      'towers',
      'float32x2',
      Float32Array.from(
        summits.flatMap(index =>
          getCellWorldPosition(
            terrain,
            terrain.world,
            index % terrain.columns,
            Math.floor(index / terrain.columns)
          )
        )
      )
    );
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const visibleCount = builder.allocate('visible-count', 'uint32', cellCount);
    builder.graph.add(
      new GPUTerrainCumulativeViewshed({
        grid,
        elevation: band,
        observers: observers.view,
        parameters: parameterBuffer.view,
        output: {visibleCount: visibleCount.view}
      })
    );
    return {
      outputs: {elevation, visibleCount, towers},
      update: current =>
        parameterBuffer.write(
          getGPUTerrainLineOfSightParameterValues(
            {observerHeight: current.observerHeight, maximumDistance: current.maximumDistance},
            grid
          )
        )
    };
  },
  draw: ({inputs, outputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.visibleCount, {
      ramp: 'sequential',
      range: [0, values.observerCount],
      fade: true,
      alpha: 0.85
    }),
    {layer: 'dots', positions: outputs.towers, radius: 6, color: [1, 1, 1, 1]}
  ]
});
