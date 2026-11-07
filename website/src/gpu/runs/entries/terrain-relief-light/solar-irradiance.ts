// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSolarIrradiance on a synthetic DEM: a sun table for one day is stepped through every cell's
// precomputed horizon, so each cell accumulates the hours of direct sun and the energy received
// (shadow, slope and aspect included). The table is a per-frame input, so the latitude and the
// month change it without recompiling.

import {
  getGPUSolarIrradianceParameterValues,
  getGPUSolarIrradianceSunTable,
  GPUSolarIrradiance
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {addTerrainHorizon, addTerrainSlopeAspect} from './support/horizon';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './solar-irradiance.parameters';

const DIRECTION_COUNT = 16;
const SAMPLE_COUNT = 96;

export default defineOperationRun({
  parameters,
  caption: 'Elevation and a day of sun positions in; hours of sun and received energy out.',
  createInputs: ({random, world}) => createTerrainInput(random, world, 48),
  build(builder, terrain) {
    const uploaded = uploadTerrain(builder, terrain);
    const {grid, elevation, cellCount} = uploaded;
    const {horizon, skyViewFactor} = addTerrainHorizon(builder, uploaded, {
      directionCount: DIRECTION_COUNT,
      maximumRadius: 40
    });
    const {slope, aspect} = addTerrainSlopeAspect(builder, uploaded);
    const sunTable = builder.allocate('sun-table', 'float32x4', SAMPLE_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    parameterBuffer.write(getGPUSolarIrradianceParameterValues({sampleCount: SAMPLE_COUNT}));
    const sunHours = builder.allocate('sun-hours', 'float32', cellCount);
    const insolation = builder.allocate('insolation', 'float32', cellCount);
    builder.graph.add(
      new GPUSolarIrradiance({
        grid,
        directionCount: DIRECTION_COUNT,
        horizon: horizon.view,
        slope: slope.view,
        aspect: aspect.view,
        skyViewFactor: skyViewFactor.view,
        sunTable: builder.graph.createDataView(sunTable.view.buffer, {
          format: 'float32',
          length: SAMPLE_COUNT * 4
        }),
        sampleCapacity: SAMPLE_COUNT,
        parameters: parameterBuffer.view,
        output: {sunHours: sunHours.view, insolation: insolation.view}
      })
    );
    return {
      outputs: {elevation, sunHours, insolation},
      update: current => {
        const start = new Date(Date.UTC(2026, current.month - 1, 21));
        sunTable.write(
          getGPUSolarIrradianceSunTable({
            longitude: 7.9,
            latitude: current.latitude,
            start,
            end: new Date(start.getTime() + 24 * 3600 * 1000),
            stepMinutes: 15
          }).values
        );
      }
    };
  },
  draw: ({inputs, outputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, values.output === 0 ? outputs.sunHours : outputs.insolation, {
      range: values.output === 0 ? [0, 14] : 'auto',
      smoothing: true
    })
  ]
});
