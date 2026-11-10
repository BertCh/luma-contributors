// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import coastline from './world-coastline.json';

/** Natural Earth coastline in the same equirectangular frame as the geodesic output. */
export function createCoastlineSegments(
  scale: readonly [number, number],
  offset: readonly [number, number]
): Float32Array {
  const segments: number[] = [];
  for (const line of coastline) {
    for (let index = 1; index < line.length; index++) {
      const [firstLongitude, firstLatitude] = line[index - 1];
      const [longitude, latitude] = line[index];
      if (Math.abs(firstLongitude - longitude) > 180) continue;
      segments.push(
        firstLongitude * scale[0] + offset[0],
        firstLatitude * scale[1] + offset[1],
        longitude * scale[0] + offset[0],
        latitude * scale[1] + offset[1]
      );
    }
  }
  return Float32Array.from(segments);
}

/** Segments whose endpoints lie within the selected geographic projection window. */
export function getCoastlineWindow(
  west: number,
  south: number,
  east: number,
  north: number
): number[][] {
  const vertices: number[][] = [];
  const inside = ([longitude, latitude]: number[]) =>
    longitude >= west && longitude <= east && latitude >= south && latitude <= north;
  for (const line of coastline) {
    for (let index = 1; index < line.length; index++) {
      if (inside(line[index - 1]) && inside(line[index]))
        vertices.push(line[index - 1], line[index]);
    }
  }
  return vertices;
}
