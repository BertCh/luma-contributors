// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSolarPosition on a world grid of longitude and latitude: the sun's altitude above every place
// at one instant, which draws the day and night sides and the terminator. The timestamp is a
// per-frame value; refraction is on.

import {
  getGPUSolarPositionParameterValues,
  GPUSolarPosition
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {parameters} from './solar-position.parameters';

const COLUMNS = 72;
const ROWS = 36;

export default defineOperationRun({
  parameters,
  caption: 'Longitude, latitude and a date in; the sun altitude of every place out.',
  createInputs: ({world}) => {
    // Places on a regular grid, row 0 at latitude +85 (top of the map).
    const places = new Float32Array(COLUMNS * ROWS * 2);
    const drawn = new Float32Array(COLUMNS * ROWS * 2);
    for (let row = 0; row < ROWS; row++) {
      for (let column = 0; column < COLUMNS; column++) {
        const index = row * COLUMNS + column;
        places[index * 2] = -180 + (360 * (column + 0.5)) / COLUMNS;
        places[index * 2 + 1] = 85 - (170 * (row + 0.5)) / ROWS;
        drawn[index * 2] = ((column + 0.5) / COLUMNS) * world.width;
        drawn[index * 2 + 1] = world.height - ((row + 0.5) / ROWS) * world.height;
      }
    }
    return {places, drawn};
  },
  build(builder, inputs) {
    const places = builder.upload('places', 'float32x2', inputs.places);
    const drawn = builder.upload('drawn', 'float32x2', inputs.drawn);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const altitude = builder.allocate('altitude', 'float32', COLUMNS * ROWS);
    const azimuth = builder.allocate('azimuth', 'float32', COLUMNS * ROWS);
    const daylight = builder.allocate('daylight', 'uint32', COLUMNS * ROWS);
    builder.graph.add(
      new GPUSolarPosition({
        positions: places.view,
        refraction: true,
        parameters: parameterBuffer.view,
        output: {altitude: altitude.view, azimuth: azimuth.view, daylight: daylight.view}
      })
    );
    return {
      outputs: {drawn, altitude},
      update: current =>
        parameterBuffer.write(
          getGPUSolarPositionParameterValues({
            timestamp: new Date(
              Date.UTC(
                2026,
                current.month - 1,
                21,
                Math.floor(current.hour),
                (current.hour % 1) * 60
              )
            )
          })
        )
    };
  },
  draw: ({outputs}) => [
    {
      layer: 'raster',
      values: outputs.altitude,
      columns: COLUMNS,
      rows: ROWS,
      ramp: 'diverging',
      range: [-90, 90],
      smoothing: true
    },
    {layer: 'dots', context: true, positions: outputs.drawn, radius: 1.2}
  ]
});
