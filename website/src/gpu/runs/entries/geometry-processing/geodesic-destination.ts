// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeodesicDestination: from one origin, a fan of bearings and one distance give the destination
// points. The arcs are the same operation sampled at k/12 of the distance, so on the equirectangular
// map they bend with latitude; destination dots are colored by the final bearing.

import {GPUGeodesicDestination} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createCoastlineSegments} from '../../../data/coastline';
import {defineOperationRun} from '../../define-run';
import {clamp} from '../../interaction';
import {parameters} from './geodesic-destination.parameters';
import {addEquirectangular, addGeodesicArcs} from './shared/shape-segments';

const DEFAULT_ORIGIN = [2.35, 48.86] as const;
const BEARING_COUNT = 72;
const ARC_STEPS = 12;
const MAXIMUM_LATITUDE = 78;

const getMapX = (longitude: number, width: number) =>
  width * 0.04 + ((longitude + 180) / 360) * width * 0.92;
const getMapY = (latitude: number) => 0.04 + ((latitude + 90) / 180) * 0.92;

export default defineOperationRun({
  parameters,
  caption: 'Where each bearing and distance lead on the globe; the arcs are great circles.',
  createInputs: ({world}) => ({world, origin: [...DEFAULT_ORIGIN] as [number, number]}),
  build(builder, inputs) {
    const {world} = inputs;
    const origins = builder.allocate('origins', 'float32x2', BEARING_COUNT);
    const bearings = builder.allocate('bearings', 'float32', BEARING_COUNT);
    const distances = builder.allocate('distances', 'float32', BEARING_COUNT);
    const destinations = builder.allocate('destinations', 'float32x2', BEARING_COUNT);
    const finalBearings = builder.allocate('final-bearings', 'float32', BEARING_COUNT);
    builder.graph.add(
      new GPUGeodesicDestination({
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'ellipsoidal',
          units: 'meters'
        },
        origins: origins.view,
        bearings: bearings.view,
        distances: distances.view,
        output: {destinations: destinations.view, finalBearings: finalBearings.view}
      })
    );
    return {
      outputs: {
        coastline: builder.upload(
          'coastline',
          'float32x4',
          createCoastlineSegments([(world.width * 0.92) / 360, 0.92 / 180], [world.width / 2, 0.5])
        ),
        arcs: addGeodesicArcs(builder, 'arcs', {
          origins,
          bearings,
          distances,
          rowCount: BEARING_COUNT,
          steps: ARC_STEPS,
          world
        }),
        mapOrigins: addEquirectangular(builder, 'origins', origins, world),
        mapDestinations: addEquirectangular(builder, 'destinations', destinations, world),
        finalBearings
      },
      update: values => {
        origins.write(
          Float32Array.from(
            {length: BEARING_COUNT * 2},
            (_, component) => inputs.origin[component & 1]
          )
        );
        bearings.write(
          Float32Array.from(
            {length: BEARING_COUNT},
            (_, row) => (row / BEARING_COUNT) * values.bearingSpan
          )
        );
        distances.write(new Float32Array(BEARING_COUNT).fill(values.distance * 1000));
      }
    };
  },
  // The pointer is the origin of the whole fan; the distance stays the slider.
  interaction: {
    kind: 'place',
    hint: 'move the origin',
    rest: ({world, toUnit}) =>
      toUnit([getMapX(DEFAULT_ORIGIN[0], world.width), getMapY(DEFAULT_ORIGIN[1])]),
    resolve(pointer, {inputs, values, world}) {
      const longitude = clamp(
        ((pointer.x - world.width * 0.04) / (world.width * 0.92)) * 360 - 180,
        -180,
        180
      );
      const latitude = clamp(
        ((pointer.y - 0.04) / 0.92) * 180 - 90,
        -MAXIMUM_LATITUDE,
        MAXIMUM_LATITUDE
      );
      inputs.origin = [longitude, latitude];
      return {
        rerun: true,
        glyphs: [{shape: 'crosshair', at: [getMapX(longitude, world.width), getMapY(latitude)]}],
        readout: `origin ${Math.abs(latitude).toFixed(1)}°${latitude < 0 ? 'S' : 'N'} ${Math.abs(longitude).toFixed(1)}°${longitude < 0 ? 'W' : 'E'} · ${values.distance} km`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.coastline, radius: 0.8, alpha: 0.45},
    {layer: 'lines', segments: outputs.arcs, radius: 1.1, alpha: 0.7},
    {layer: 'dots', positions: outputs.mapOrigins, context: true, radius: 4},
    {
      layer: 'dots',
      positions: outputs.mapDestinations,
      values: outputs.finalBearings,
      ramp: 'diverging',
      range: [-180, 180],
      radius: 4
    }
  ]
});
