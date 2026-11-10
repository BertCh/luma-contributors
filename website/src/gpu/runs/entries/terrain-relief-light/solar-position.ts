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
import {findParameter, snapToParameter} from '../../interaction';
import {createCoastlineSegments} from '../../../data/coastline';
import {cityCoordinates} from '../../../data/hydrography';
import {parameters} from './solar-position.parameters';

const COLUMNS = 72;
const ROWS = 36;
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
// Solar declination in degrees on the 21st of each month.
const DECLINATIONS = [-20.1, -10.8, 0.1, 11.6, 20, 23.4, 20.5, 12.3, 0.4, -10.4, -19.8, -23.4];
// Months of the half year from the December to the June solstice, which the latitude walks along.
const SOLSTICE_BRANCH = [12, 1, 2, 3, 4, 5, 6];

/** The subsolar point of a month and an hour (UTC): longitude then latitude. */
function getSubsolarPoint(month: number, hour: number): [number, number] {
  const longitude = (((((12 - hour) * 15 + 180) % 360) + 360) % 360) - 180;
  return [longitude, DECLINATIONS[month - 1]];
}

export default defineOperationRun({
  parameters,
  caption:
    'Solar altitude across the world and at eighteen Natural Earth cities on the 21st of each month.',
  createInputs: ({world}) => {
    // Places on a regular grid, row 0 at latitude +85 (top of the map).
    const places = new Float32Array(COLUMNS * ROWS * 2);
    for (let row = 0; row < ROWS; row++) {
      for (let column = 0; column < COLUMNS; column++) {
        const index = row * COLUMNS + column;
        places[index * 2] = -180 + (360 * (column + 0.5)) / COLUMNS;
        places[index * 2 + 1] = 85 - (170 * (row + 0.5)) / ROWS;
      }
    }
    const cities = Object.values(cityCoordinates);
    const drawn = Float32Array.from(
      cities.flatMap(([longitude, latitude]) => [
        ((longitude + 180) * world.width) / 360,
        ((latitude + 85) * world.height) / 170
      ])
    );
    const coastline = createCoastlineSegments(
      [world.width / 360, world.height / 170],
      [world.width / 2, world.height / 2]
    );
    return {places, cities: Float32Array.from(cities.flat()), drawn, coastline};
  },
  build(builder, inputs) {
    const places = builder.upload('places', 'float32x2', inputs.places);
    const coastline = builder.upload('coastline', 'float32x4', inputs.coastline);
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
    const cityAltitude = builder.allocate('city-altitude', 'float32', inputs.cities.length / 2);
    builder.graph.add(
      new GPUSolarPosition({
        id: 'city-solar-position',
        positions: builder.upload('cities', 'float32x2', inputs.cities).view,
        refraction: true,
        parameters: parameterBuffer.view,
        output: {altitude: cityAltitude.view}
      })
    );
    return {
      outputs: {drawn, altitude, cityAltitude, coastline},
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
  // The pointer is the point of the Earth directly under the sun: its longitude sets the hour (UTC)
  // and its latitude, up to the tropics, the date by solar declination.
  interaction: {
    kind: 'drag',
    hint: 'place the sun overhead',
    rest: ({world}) => {
      const [longitude, latitude] = getSubsolarPoint(
        findParameter(parameters, 'month').default,
        findParameter(parameters, 'hour').default
      );
      return [(longitude + 180) / 360, 1 - (latitude + 85) / 170].map(unit => unit) as [
        number,
        number
      ];
    },
    resolve(pointer, {world}) {
      const hourParameter = findParameter(parameters, 'hour');
      const monthParameter = findParameter(parameters, 'month');
      const longitude = (pointer.x / world.width) * 360 - 180;
      const latitude = (pointer.y / world.height) * 170 - 85;
      const hour = snapToParameter(hourParameter, (((12 - longitude / 15) % 24) + 24) % 24);
      let month = SOLSTICE_BRANCH[0];
      for (const candidate of SOLSTICE_BRANCH) {
        if (
          Math.abs(DECLINATIONS[candidate - 1] - latitude) <
          Math.abs(DECLINATIONS[month - 1] - latitude)
        ) {
          month = candidate;
        }
      }
      month = snapToParameter(monthParameter, month);
      const [overheadLongitude, overheadLatitude] = getSubsolarPoint(month, hour);
      const at: [number, number] = [
        ((overheadLongitude + 180) * world.width) / 360,
        ((overheadLatitude + 85) * world.height) / 170
      ];
      const hours = Math.floor(hour);
      const minutes = hour % 1 === 0 ? '00' : '30';
      return {
        values: {hour, month},
        glyphs: [
          {shape: 'ring', at, radius: world.height * 0.06},
          {shape: 'dot', at}
        ],
        readout: `21 ${MONTH_NAMES[month - 1]} ${String(hours).padStart(2, '0')}:${minutes} UTC · overhead ${Math.abs(Math.round(overheadLatitude))}°${overheadLatitude >= 0 ? 'N' : 'S'}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {
      layer: 'raster',
      values: outputs.altitude,
      columns: COLUMNS,
      rows: ROWS,
      ramp: 'diverging',
      range: [-90, 90],
      smoothing: true
    },
    {layer: 'lines', segments: outputs.coastline, radius: 0.8, color: [...theme.ink, 0.65]},
    {layer: 'dots', positions: outputs.drawn, radius: 5, color: [...theme.ink, 0.85]},
    {
      layer: 'dots',
      positions: outputs.drawn,
      values: outputs.cityAltitude,
      ramp: 'diverging',
      radius: 3.5,
      range: [-90, 90]
    }
  ]
});
