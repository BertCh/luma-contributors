// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainViewshed on a synthetic DEM: for every cell, whether a person of observerHeight standing
// at the observer can see it (visible, hidden behind terrain, or out of range). The observer
// position, height and range are per-frame values, so dragging them never recompiles.

import {
  getGPUTerrainViewshedParameterValues,
  GPUTerrainViewshed
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  getCellOfPercentage,
  getCellWorldPosition,
  uploadTerrain
} from './support/terrain';
import {parameters} from './viewshed.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Elevation and an observer in; what the observer can see, hide from or not reach, out.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
  build(builder, terrain) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 10);
    const visibility = builder.allocate('visibility', 'uint32', cellCount);
    const observerDot = builder.allocate('observer', 'float32x2', 1);
    builder.graph.add(
      new GPUTerrainViewshed({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        output: {visibility: visibility.view}
      })
    );
    return {
      outputs: {elevation, visibility, observerDot},
      update: current => {
        const [column, row] = getCellOfPercentage(
          terrain,
          current.observerColumn,
          current.observerRow
        );
        parameterBuffer.write(
          getGPUTerrainViewshedParameterValues(
            {
              observer: [column, row],
              observerHeight: current.observerHeight,
              maximumDistance: current.maximumDistance
            },
            grid
          )
        );
        observerDot.write(
          Float32Array.from(getCellWorldPosition(terrain, terrain.world, column, row))
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.visibility, {alpha: 0.5}),
    {layer: 'dots', positions: outputs.observerDot, radius: 7, color: [1, 1, 1, 1]}
  ]
});
