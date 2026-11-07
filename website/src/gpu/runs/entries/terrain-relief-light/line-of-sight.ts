// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainLineOfSight on a synthetic DEM: one observer and many targets, each pair tested against
// the terrain between them. The clearance (meters above the terrain along the sight line; negative
// when blocked) colors each target. Positions and heights are per-frame values.

import {
  getGPUTerrainLineOfSightParameterValues,
  GPUTerrainLineOfSight
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {
  createTerrainInput,
  drawElevationContext,
  getCellOfPercentage,
  getCellWorldPosition,
  uploadTerrain
} from './support/terrain';
import {parameters} from './line-of-sight.parameters';

const TARGET_COUNT = 90;

export default defineOperationRun({
  parameters,
  caption: 'Elevation and observer-target pairs in; the clearance of every sight line out.',
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(random, world);
    const targets = Array.from({length: TARGET_COUNT}, () => [
      Math.floor(random() * terrain.columns),
      Math.floor(random() * terrain.rows)
    ]);
    return {...terrain, targets};
  },
  build(builder, terrain) {
    const {grid, band, elevation} = uploadTerrain(builder, terrain);
    const pairs = builder.allocate('pairs', 'float32x4', TARGET_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const visibility = builder.allocate('visibility', 'uint32', TARGET_COUNT);
    const clearance = builder.allocate('clearance', 'float32', TARGET_COUNT);
    const targetPositions = builder.upload(
      'target-positions',
      'float32x2',
      Float32Array.from(
        terrain.targets.flatMap(([column, row]) =>
          getCellWorldPosition(terrain, terrain.world, column, row)
        )
      )
    );
    const rays = builder.allocate('rays', 'float32x4', TARGET_COUNT);
    builder.graph.add(
      new GPUTerrainLineOfSight({
        grid,
        elevation: band,
        pairs: pairs.view,
        parameters: parameterBuffer.view,
        output: {visibility: visibility.view, clearance: clearance.view}
      })
    );
    return {
      outputs: {elevation, clearance, targetPositions, rays},
      update: current => {
        const [column, row] = getCellOfPercentage(
          terrain,
          current.observerColumn,
          current.observerRow
        );
        const [observerX, observerY] = getCellWorldPosition(terrain, terrain.world, column, row);
        pairs.write(
          Float32Array.from(
            terrain.targets.flatMap(([targetColumn, targetRow]) => [
              column,
              row,
              targetColumn,
              targetRow
            ])
          )
        );
        rays.write(
          Float32Array.from(
            terrain.targets.flatMap(([targetColumn, targetRow]) => [
              observerX,
              observerY,
              ...getCellWorldPosition(terrain, terrain.world, targetColumn, targetRow)
            ])
          )
        );
        parameterBuffer.write(
          getGPUTerrainLineOfSightParameterValues(
            {observerHeight: current.observerHeight, targetHeight: current.targetHeight},
            grid
          )
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    {layer: 'lines', context: true, segments: outputs.rays, radius: 0.6, alpha: 0.5},
    {
      layer: 'dots',
      positions: outputs.targetPositions,
      values: outputs.clearance,
      ramp: 'diverging',
      range: [-80, 80],
      radius: 6
    }
  ]
});
