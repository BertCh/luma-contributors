// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import lakes from '../../../../data/natural-earth-geometry.json';
import {getUnitVector} from './polygons';

export const SUPERIOR_WINDOW = [-92.6, 45.9, -83.9, 49.5] as const;
export const MICHIGAN_WINDOW = [-88.6, 43.0, -85.4, 46.3] as const;

/** The kernels take open exterior rings. Traverse the shore counterclockwise. */
export function getLakeShore(name: 'Lake Superior' | 'Lake Michigan') {
  return lakes.greatLakes[name].slice().reverse();
}

/** Latitude of the modeled survey corridor's center. */
export const SWATH_LATITUDE = 47.7;

/** Modeled survey corridor centered at a geographic point; moving it changes its overlap with Lake Superior. */
export function getSurveySwath(longitude: number, latitude = SWATH_LATITUDE): number[][] {
  return [
    [longitude - 1.5, latitude - 1.47],
    [longitude - 0.1, latitude - 1.47],
    [longitude + 1.6, latitude + 1.47],
    [longitude + 0.2, latitude + 1.47]
  ];
}

/** Inscribed convex geodesic polygon centered at a geographic point, with a radius in kilometers. */
export function getSurveyRing(
  longitude: number,
  latitude: number,
  radius: number,
  sides: number,
  rotation = 0
) {
  const radians = Math.PI / 180;
  const center = getUnitVector(longitude, latitude);
  const east = [-Math.sin(longitude * radians), Math.cos(longitude * radians), 0];
  const north = [-center[2] * east[1], center[2] * east[0], Math.cos(latitude * radians)];
  const distance = radius / 6371.0088;
  return Array.from({length: sides}, (_, index) => {
    const angle = (2 * Math.PI * index) / sides + rotation;
    const vector = center.map(
      (component, axis) =>
        component * Math.cos(distance) +
        Math.sin(distance) * (Math.cos(angle) * east[axis] + Math.sin(angle) * north[axis])
    );
    return [Math.atan2(vector[1], vector[0]) / radians, Math.asin(vector[2]) / radians];
  });
}

/** Lake Superior's spherical convex hull: the supporting minor arcs have every shore vertex
 * on their left. Keep the endpoints in the source ring's counterclockwise order. */
export function getConvexShore(shore: number[][]) {
  const vectors = shore.map(([longitude, latitude]) => getUnitVector(longitude, latitude));
  return shore.filter((_, index) => {
    const first = vectors[index];
    return vectors.some((second, other) => {
      if (index === other) return false;
      const normal = [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0]
      ];
      return vectors.every(
        point => normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2] >= -1e-12
      );
    });
  });
}

export function getRingSegments(rings: number[][][]): Float32Array {
  return Float32Array.from(
    rings.flatMap(ring =>
      ring.flatMap((point, index) => [...point, ...ring[(index + 1) % ring.length]])
    )
  );
}
