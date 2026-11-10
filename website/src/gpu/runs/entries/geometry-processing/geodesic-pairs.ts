// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeodesicPairs measures many origin-target pairs on the globe. Every city is paired with the
// target under the pointer; the arcs are the same pairs sampled along their great circle, and the
// dots are the geodesic midpoints the operation wrote, colored by the pair's distance in meters.

import {GPUGeodesicPairs} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createCoastlineSegments} from '../../../data/coastline';
import {cityCoordinates} from '../../../data/hydrography';
import {defineOperationRun} from '../../define-run';
import {clamp} from '../../interaction';
import {parameters} from './geodesic-pairs.parameters';
import {addEquirectangular, addGeodesicArcs} from './shared/shape-segments';

const CITY_NAMES = Object.keys(cityCoordinates);
const CITY_POSITIONS = Object.values(cityCoordinates);
const PAIR_COUNT = CITY_POSITIONS.length;
const DEFAULT_TARGET = cityCoordinates.Tokyo;
const ARC_STEPS = 16;

const getMapX = (longitude: number, width: number) =>
  width * 0.04 + ((longitude + 180) / 360) * width * 0.92;
const getMapY = (latitude: number) => 0.04 + ((latitude + 90) / 180) * 0.92;

/** Initial great-circle bearing in degrees clockwise from north (spherical, for placing the arcs). */
function getInitialBearing(origin: readonly number[], target: readonly number[]): number {
  const radians = Math.PI / 180;
  const deltaLongitude = (target[0] - origin[0]) * radians;
  const firstLatitude = origin[1] * radians;
  const secondLatitude = target[1] * radians;
  const bearing =
    Math.atan2(
      Math.sin(deltaLongitude) * Math.cos(secondLatitude),
      Math.cos(firstLatitude) * Math.sin(secondLatitude) -
        Math.sin(firstLatitude) * Math.cos(secondLatitude) * Math.cos(deltaLongitude)
    ) / radians;
  return (bearing + 360) % 360;
}

export default defineOperationRun({
  parameters,
  caption: 'Distance and midpoint of each city to the target, measured on the ellipsoid.',
  createInputs: ({world}) => ({world, target: [...DEFAULT_TARGET] as [number, number]}),
  build(builder, inputs) {
    const {world} = inputs;
    const originBuffer = builder.allocate('origins', 'float32x2', PAIR_COUNT);
    const targetBuffer = builder.allocate('targets', 'float32x2', PAIR_COUNT);
    const bearings = builder.allocate('bearings', 'float32', PAIR_COUNT);
    const distances = builder.allocate('distances', 'float32', PAIR_COUNT);
    const midpoints = builder.allocate('midpoints', 'float32x2', PAIR_COUNT);
    builder.graph.add(
      new GPUGeodesicPairs({
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'ellipsoidal',
          units: 'meters'
        },
        origins: originBuffer.view,
        targets: targetBuffer.view,
        output: {distances: distances.view, midpoints: midpoints.view}
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
          origins: originBuffer,
          bearings,
          distances,
          rowCount: PAIR_COUNT,
          steps: ARC_STEPS,
          world
        }),
        mapOrigins: addEquirectangular(builder, 'origins', originBuffer, world),
        midpoints: addEquirectangular(builder, 'midpoints', midpoints, world),
        distances
      },
      update: values => {
        // Cities past `cityCount` are paired with the target itself, so their arcs and midpoints collapse.
        const origins = CITY_POSITIONS.map((city, index) =>
          index < values.cityCount ? city : inputs.target
        );
        originBuffer.write(Float32Array.from(origins.flat()));
        targetBuffer.write(
          Float32Array.from(
            {length: PAIR_COUNT * 2},
            (_, component) => inputs.target[component & 1]
          )
        );
        bearings.write(
          Float32Array.from(origins, origin => getInitialBearing(origin, inputs.target))
        );
      }
    };
  },
  // The pointer is the target every city is measured to.
  interaction: {
    kind: 'place',
    hint: 'move the target',
    rest: ({world, toUnit}) =>
      toUnit([getMapX(DEFAULT_TARGET[0], world.width), getMapY(DEFAULT_TARGET[1])]),
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const longitude = clamp(
        ((pointer.x - world.width * 0.04) / (world.width * 0.92)) * 360 - 180,
        -179,
        179
      );
      const latitude = clamp(((pointer.y - 0.04) / 0.92) * 180 - 90, -80, 80);
      inputs.target = [longitude, latitude];
      const distances = peek(outputs.distances);
      let nearest = -1;
      if (distances) {
        for (let city = 0; city < Math.min(values.cityCount, PAIR_COUNT); city++) {
          if (nearest < 0 || distances[city] < distances[nearest]) nearest = city;
        }
      }
      return {
        rerun: true,
        glyphs: [{shape: 'crosshair', at: [getMapX(longitude, world.width), getMapY(latitude)]}],
        readout:
          nearest >= 0 && distances
            ? `nearest ${CITY_NAMES[nearest]} ${Math.round(distances[nearest] / 1000)} km`
            : `target ${Math.abs(latitude).toFixed(1)}°${latitude < 0 ? 'S' : 'N'} ${Math.abs(longitude).toFixed(1)}°${longitude < 0 ? 'W' : 'E'}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.coastline, radius: 0.8, alpha: 0.45},
    {layer: 'dots', positions: outputs.mapOrigins, context: true, radius: 3},
    {layer: 'lines', segments: outputs.arcs, radius: 1.1, alpha: 0.7},
    {layer: 'dots', positions: outputs.midpoints, values: outputs.distances, radius: 6}
  ]
});
