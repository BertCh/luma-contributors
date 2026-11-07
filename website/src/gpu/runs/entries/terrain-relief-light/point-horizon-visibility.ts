// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointHorizonVisibility on a synthetic DEM: is each summit visible from the observer? The
// observer's horizon is marched once per direction and every target is compared against it. Targets
// are colored by the code the GPU wrote (visible or hidden); the lines show the pairs.

import {
  getGPUPointHorizonVisibilityParameterValues,
  GPUPointHorizonVisibility
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {
  createTerrainInput,
  drawElevationContext,
  findLocalMaxima,
  getCellOfPercentage,
  getCellWorldPosition,
  TERRAIN_CELL_SIZE,
  uploadTerrain
} from './support/terrain';
import {parameters} from './point-horizon-visibility.parameters';

const TARGET_COUNT = 14;

export default defineOperationRun({
  parameters,
  caption: 'Elevation, an observer and summit targets in; visible or hidden per target out.',
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(random, world);
    const peaks = findLocalMaxima(terrain, TARGET_COUNT).map(index => [
      index % terrain.columns,
      Math.floor(index / terrain.columns)
    ]);
    return {...terrain, peaks};
  },
  build(builder, terrain) {
    const {grid, band, elevation} = uploadTerrain(builder, terrain);
    const count = terrain.peaks.length;
    const observers = builder.allocate('observers', 'float32x4', 1);
    const targets = builder.allocate('targets', 'float32x4', count);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const visibility = builder.allocate('visibility', 'uint32', count);
    const details = builder.allocate('details', 'float32x4', count);
    const peakPositions = builder.upload(
      'peak-positions',
      'float32x2',
      Float32Array.from(
        terrain.peaks.flatMap(([column, row]) =>
          getCellWorldPosition(terrain, terrain.world, column, row)
        )
      )
    );
    const rays = builder.allocate('rays', 'float32x4', count);
    const observerDot = builder.allocate('observer-dot', 'float32x2', 1);
    builder.graph.add(
      new GPUPointHorizonVisibility({
        grid,
        elevation: band,
        observers: observers.view,
        targets: targets.view,
        cellSize: TERRAIN_CELL_SIZE,
        maximumDistance: 3600,
        parameters: parameterBuffer.view,
        output: {visibility: visibility.view, details: details.view}
      })
    );
    return {
      outputs: {elevation, visibility, peakPositions, rays, observerDot},
      update: current => {
        const [column, row] = getCellOfPercentage(
          terrain,
          current.observerColumn,
          current.observerRow
        );
        const observer = getCellWorldPosition(terrain, terrain.world, column, row);
        observers.write(Float32Array.of(column, row, current.observerHeight, 0));
        targets.write(
          Float32Array.from(
            terrain.peaks.flatMap(([peakColumn, peakRow]) => [
              peakColumn,
              peakRow,
              current.targetHeight,
              0
            ])
          )
        );
        rays.write(
          Float32Array.from(
            terrain.peaks.flatMap(([peakColumn, peakRow]) => [
              ...observer,
              ...getCellWorldPosition(terrain, terrain.world, peakColumn, peakRow)
            ])
          )
        );
        observerDot.write(Float32Array.from(observer));
        parameterBuffer.write(
          getGPUPointHorizonVisibilityParameterValues({maximumDistance: 3600}, grid)
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    {layer: 'lines', context: true, segments: outputs.rays, radius: 0.6, alpha: 0.5},
    {layer: 'dots', positions: outputs.peakPositions, values: outputs.visibility, radius: 13},
    {layer: 'dots', positions: outputs.observerDot, radius: 7, color: [1, 1, 1, 1]}
  ]
});
