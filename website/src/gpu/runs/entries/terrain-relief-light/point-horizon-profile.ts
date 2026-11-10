// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointHorizonProfile on a Grand Canyon DEM: the skyline seen from one observer, the highest
// elevation angle of the terrain in each of 180 azimuths. The map shows where the observer
// stands; the chart is the panorama written by the GPU (azimuth left to right, angle up).

import {
  getGPUPointHorizonProfileParameterValues,
  GPUPointHorizonProfile
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {getObserverRest, getObserverUnderPointer} from './support/pointer';
import {
  createTerrainInput,
  drawElevationContext,
  getCellOfPercentage,
  getCellWorldPosition,
  getWorldPerMeter,
  uploadTerrain
} from './support/terrain';
import {parameters} from './point-horizon-profile.parameters';

const AZIMUTH_COUNT = 180;
const CHART_RECTANGLE = [0.04, 0.04, 0.96, 0.4] as const;
// Skyline angles in degrees: the chart spans -20 to 40, so the horizon line sits a third up.
const CHART_RANGE = [-20, 40] as const;

export default defineOperationRun({
  parameters,
  caption: 'Elevation and an observer in; the skyline angle in every direction out.',
  createInputs: ({world}) => createTerrainInput(world),
  build(builder, terrain) {
    const {grid, band, elevation} = uploadTerrain(builder, terrain);
    const observers = builder.allocate('observers', 'float32x4', 1);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const skylineAngle = builder.allocate('skyline-angle', 'float32', AZIMUTH_COUNT);
    const distance = builder.allocate('distance', 'float32', AZIMUTH_COUNT);
    const observerDot = builder.allocate('observer-dot', 'float32x2', 1);
    // Guides under the panorama: the baseline and a tick at north, east, south, west and north.
    const [left, bottom, right, top] = CHART_RECTANGLE;
    const {width, height} = terrain.world;
    const horizon = bottom + (top - bottom) * (-CHART_RANGE[0] / (CHART_RANGE[1] - CHART_RANGE[0]));
    const guides = builder.upload(
      'guides',
      'float32x4',
      Float32Array.from([
        left * width,
        horizon * height,
        right * width,
        horizon * height,
        ...[0, 0.25, 0.5, 0.75, 1].flatMap(fraction => [
          (left + fraction * (right - left)) * width,
          bottom * height,
          (left + fraction * (right - left)) * width,
          top * height
        ])
      ])
    );
    builder.graph.add(
      new GPUPointHorizonProfile({
        grid,
        elevation: band,
        observers: observers.view,
        azimuthCount: AZIMUTH_COUNT,
        cellSize: terrain.cellSize,
        maximumDistance: 3600,
        parameters: parameterBuffer.view,
        output: {skylineAngle: skylineAngle.view, distance: distance.view}
      })
    );
    return {
      outputs: {elevation, skylineAngle, observerDot, guides},
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
  // The pointer is the observer; the ring is the distance the skyline search reaches.
  interaction: {
    kind: 'place',
    hint: 'move the observer',
    rest: getObserverRest(parameters),
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const observer = getObserverUnderPointer(inputs, world, pointer);
      const skyline = peek(outputs.skylineAngle);
      let readout = `observer [${observer.column}, ${observer.row}] · sees ${values.maximumDistance} m`;
      if (skyline) {
        let highest = 0;
        for (let index = 1; index < skyline.length; index++) {
          if (skyline[index] > skyline[highest]) highest = index;
        }
        readout = `skyline ${Math.round(skyline[highest])}° at ${Math.round((highest * 360) / AZIMUTH_COUNT)}° · [${observer.column}, ${observer.row}]`;
      }
      return {
        values: {observerColumn: observer.observerColumn, observerRow: observer.observerRow},
        glyphs: [
          {
            shape: 'ring',
            at: observer.position,
            radius: values.maximumDistance * getWorldPerMeter(inputs, world),
            dashed: true
          },
          {shape: 'crosshair', at: observer.position}
        ],
        readout
      };
    }
  },
  draw: ({outputs, inputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    {layer: 'dots', positions: outputs.observerDot, radius: 7, color: [1, 1, 1, 1]},
    {layer: 'lines', segments: outputs.guides, radius: 0.8, color: [0.7, 0.7, 0.7, 0.8]},
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.skylineAngle,
      rectangle: CHART_RECTANGLE,
      range: CHART_RANGE
    }
  ]
});
