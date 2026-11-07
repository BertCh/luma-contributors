// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointHorizonProfile on a synthetic DEM: the skyline seen from one observer, the highest
// elevation angle of the terrain in each of 180 azimuths. The map shows where the observer
// stands; the chart is the panorama written by the GPU (azimuth left to right, angle up).

import {
  getGPUPointHorizonProfileParameterValues,
  GPUPointHorizonProfile
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {
  createTerrainInput,
  drawElevationContext,
  getCellOfPercentage,
  getCellWorldPosition,
  TERRAIN_CELL_SIZE,
  uploadTerrain
} from './support/terrain';
import {parameters} from './point-horizon-profile.parameters';

const AZIMUTH_COUNT = 180;

export default defineOperationRun({
  parameters,
  caption: 'Elevation and an observer in; the skyline angle in every direction out.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
  build(builder, terrain) {
    const {grid, band, elevation} = uploadTerrain(builder, terrain);
    const observers = builder.allocate('observers', 'float32x4', 1);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const skylineAngle = builder.allocate('skyline-angle', 'float32', AZIMUTH_COUNT);
    const distance = builder.allocate('distance', 'float32', AZIMUTH_COUNT);
    const observerDot = builder.allocate('observer-dot', 'float32x2', 1);
    builder.graph.add(
      new GPUPointHorizonProfile({
        grid,
        elevation: band,
        observers: observers.view,
        azimuthCount: AZIMUTH_COUNT,
        cellSize: TERRAIN_CELL_SIZE,
        maximumDistance: 3600,
        parameters: parameterBuffer.view,
        output: {skylineAngle: skylineAngle.view, distance: distance.view}
      })
    );
    return {
      outputs: {elevation, skylineAngle, observerDot},
      update: current => {
        const [column, row] = getCellOfPercentage(
          terrain,
          current.observerColumn,
          current.observerRow
        );
        observers.write(Float32Array.of(column, row, current.observerHeight, 0));
        observerDot.write(
          Float32Array.from(getCellWorldPosition(terrain, terrain.world, column, row))
        );
        parameterBuffer.write(
          getGPUPointHorizonProfileParameterValues({maximumDistance: current.maximumDistance}, grid)
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    {layer: 'dots', positions: outputs.observerDot, radius: 7, color: [1, 1, 1, 1]},
    {layer: 'chart', kind: 'bars', values: outputs.skylineAngle, rectangle: [0.04, 0.04, 0.96, 0.4]}
  ]
});
