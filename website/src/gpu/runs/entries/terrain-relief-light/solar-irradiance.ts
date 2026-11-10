// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSolarIrradiance on a Grand Canyon DEM: a sun table for one day is stepped through every cell's
// precomputed horizon, so each cell accumulates the hours of direct sun and the energy received
// (shadow, slope and aspect included). The table is a per-frame input, so the latitude and the
// month change it without recompiling.

import {
  getGPUSolarIrradianceParameterValues,
  getGPUSolarIrradianceSunTable,
  GPUSolarIrradiance
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {addTerrainHorizon, addTerrainSlopeAspect} from './support/horizon';

import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './solar-irradiance.parameters';

const DIRECTION_COUNT = 16;
const SAMPLE_COUNT = 96;
// The Grand Canyon tile: about 36.1 degrees north, 112.1 degrees west.
const LONGITUDE = -112.1;
const MONTH_NAMES = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec'
];

const INSOLATION_RANGE = [0, 7000] as const;

const axes = bindAxes<unknown, {sunHours: RunBuffer}>(parameters, {
  kind: 'pad',
  hint: 'month and latitude',
  x: 'month',
  y: 'latitude',
  decorate: (values, pointer, {outputs, peek}) => {
    const sunHours = peek(outputs.sunHours);
    let mean = '';
    if (sunHours) {
      let sum = 0;
      for (const hours of sunHours) sum += hours;
      mean = ` · mean ${(sum / sunHours.length).toFixed(1)} h sun`;
    }
    return {
      glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
      readout: `21 ${MONTH_NAMES[values.month - 1]} · ${values.latitude}°${mean}`
    };
  }
});

export default defineOperationRun({
  parameters,
  caption: 'Elevation and a day of sun positions in; hours of sun and received energy out.',
  createInputs: ({world}) => createTerrainInput(world, 48),
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
        // One local day: it starts at local solar midnight, which is 12 hours before noon there.
        const start = new Date(
          Date.UTC(2026, current.month - 1, 21) + (-LONGITUDE / 15) * 3600 * 1000
        );
        sunTable.write(
          getGPUSolarIrradianceSunTable({
            longitude: LONGITUDE,
            latitude: current.latitude,
            start,
            end: new Date(start.getTime() + 24 * 3600 * 1000),
            stepMinutes: 15
          }).values
        );
      }
    };
  },
  // The pointer picks the sun: x is the month (21st of it), y the latitude of the sun table.
  interaction: axes,
  draw: ({inputs, outputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, values.output === 0 ? outputs.sunHours : outputs.insolation, {
      range: values.output === 0 ? [0, 12] : INSOLATION_RANGE,
      smoothing: true
    })
  ]
});
