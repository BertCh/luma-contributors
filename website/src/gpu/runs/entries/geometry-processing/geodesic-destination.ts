// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeodesicDestination: from a few origins, a fan of bearings and one distance give the
// destination points. Lines join each origin to its destinations on an equirectangular map, so
// the geodesics bend with latitude; destination dots are colored by the final bearing.

import {GPUGeodesicDestination} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addEquirectangular, addPairSegments} from './shared/shape-segments';
import {parameters} from './geodesic-destination.parameters';

const ORIGINS = [
  [-100, 40],
  [10, 50],
  [105, 25],
  [-60, -25],
  [25, -20],
  [150, -30]
];
const BEARINGS_PER_ORIGIN = 18;
const ROW_COUNT = ORIGINS.length * BEARINGS_PER_ORIGIN;

export default defineOperationRun({
  parameters,
  caption: 'Where each bearing and distance lead on the globe.',
  createInputs: ({world}) => ({world}),
  build(builder, {world}) {
    const origins = new Float32Array(ROW_COUNT * 2);
    for (let row = 0; row < ROW_COUNT; row++) {
      const origin = ORIGINS[Math.floor(row / BEARINGS_PER_ORIGIN)];
      origins[row * 2] = origin[0];
      origins[row * 2 + 1] = origin[1];
    }
    const originBuffer = builder.upload('origins', 'float32x2', origins);
    const bearings = builder.allocate('bearings', 'float32', ROW_COUNT);
    const distances = builder.allocate('distances', 'float32', ROW_COUNT);
    const destinations = builder.allocate('destinations', 'float32x2', ROW_COUNT);
    const finalBearings = builder.allocate('final-bearings', 'float32', ROW_COUNT);
    builder.graph.add(
      new GPUGeodesicDestination({
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'ellipsoidal',
          units: 'meters'
        },
        origins: originBuffer.view,
        bearings: bearings.view,
        distances: distances.view,
        output: {destinations: destinations.view, finalBearings: finalBearings.view}
      })
    );
    const mapOrigins = addEquirectangular(builder, 'origins', originBuffer, world);
    const mapDestinations = addEquirectangular(builder, 'destinations', destinations, world);
    return {
      outputs: {
        rays: addPairSegments(builder, 'rays', mapOrigins, mapDestinations),
        mapOrigins,
        mapDestinations,
        finalBearings
      },
      update: values => {
        bearings.write(
          Float32Array.from(
            {length: ROW_COUNT},
            (_, row) => ((row % BEARINGS_PER_ORIGIN) / BEARINGS_PER_ORIGIN) * values.bearingSpan
          )
        );
        distances.write(new Float32Array(ROW_COUNT).fill(values.distance * 1000));
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.rays},
    {layer: 'dots', positions: outputs.mapOrigins, context: true, radius: 4},
    {
      layer: 'dots',
      positions: outputs.mapDestinations,
      values: outputs.finalBearings,
      ramp: 'diverging',
      range: [-180, 180],
      radius: 5
    }
  ]
});
