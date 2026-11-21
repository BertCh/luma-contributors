// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Hand derivation: a zero-weight self-loop at (1,1), cost 3, radius 2 and zero walking
 * cost covers all four centers of a 2x2 raster over [0,0,2,2]. Breaks [2,4] put all
 * samples in band 1. The unit square between the centers has two CCW fan triangles,
 * four boundary edges, one closed five-vertex ring and one four-vertex polygon.
 * These literals are independent of the splat/isoband/ring implementation.
 */
export const RASTER_SQUARE = {
  values: [3, 3, 3, 3],
  bandClasses: [1, 1, 1, 1],
  triangles: [0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5, 0.5, 1.5],
  triangleBands: [1, 1],
  edges: [0.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 1.5, 1.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 0.5],
  corners: [0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5]
};

/**
 * Quadbin z=2 tile (1,1): west=-90, east=0, north=atan(sinh(pi/2))*180/pi,
 * south=0. Its Morton path is 3, its 48-bit footer is all ones; the uint64 key is
 * 0x4823ffffffffffff. Four reached points in this tile give exactly one table row.
 * Independent numeric derivation: /usr/bin/python3 isochrones-output-oracles.oracle.py
 */
export const CELL_SQUARE = {
  positions: [-70, 20, -60, 30, -50, 40, -40, 50],
  key: [4294967295, 1210318847],
  north: 66.51326044311186,
  area: 5986.193439880067,
  // Clockwise cell edge order: north, east, south, west.
  endpoints: [
    -90, 66.51326044311186, 0, 66.51326044311186, 0, 66.51326044311186, 0, 0, 0, 0, -90, 0, -90, 0,
    -90, 66.51326044311186
  ],
  // With normalizeWinding the shell is counter-clockwise.
  corners: [-90, 0, 0, 0, 0, 66.51326044311186, -90, 66.51326044311186]
};

/** Python oracle above: detached z4 tile (1,1), then a 3x3 block (7..9,7..9) without (8,8). */
export const CELL_HOLE = {
  positions: [
    -146.25, 81.09321385260839, -11.25, 11.178401873711781, 11.25, 11.178401873711781, 33.75,
    11.178401873711781, -11.25, -11.178401873711781, -11.25, -31.952162238024968, 33.75,
    -11.178401873711781, 11.25, -31.952162238024968, 33.75, -31.952162238024968
  ],
  keys: [
    4294967295, 1212170239, 4294967295, 1212415999, 4294967295, 1212592127, 4294967295, 1212596223,
    4294967295, 1212768255, 4294967295, 1212776447, 4294967295, 1212948479, 4294967295, 1212952575,
    4294967295, 1212956671
  ],
  rows: [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
  edgeIndices: [0, 1, 2, 3, 0, 3, 0, 2, 0, 1, 1, 3, 2, 3, 1, 3, 0, 2, 1, 2],
  segmentStarts: [
    -157.5, 82.67628497834903, -135, 82.67628497834903, -135, 79.17133464081945, -157.5,
    79.17133464081945, -22.5, 21.943045533438177, -22.5, 0, 0, 21.943045533438177, 22.5, 0, 22.5,
    21.943045533438177, 45, 21.943045533438177, 0, 0, -22.5, -21.943045533438177, 0,
    -40.97989806962013, -22.5, -40.97989806962013, 45, 0, 22.5, -21.943045533438177, 0,
    -21.943045533438177, 22.5, -40.97989806962013, 45, -21.943045533438177, 45, -40.97989806962013
  ],
  ringOffsets: [0, 5, 18, 23, 23],
  areas: [78.86138259441552, 4247.298693206435, -493.718524502359, 0],
  rings: [
    [
      -157.5, 79.17133464081945, -135, 79.17133464081945, -135, 82.67628497834903, -157.5,
      82.67628497834903
    ],
    [
      -22.5, -40.97989806962013, 0, -40.97989806962013, 22.5, -40.97989806962013, 45,
      -40.97989806962013, 45, -21.943045533438177, 45, 0, 45, 21.943045533438177, 22.5,
      21.943045533438177, 0, 21.943045533438177, -22.5, 21.943045533438177, -22.5, 0, -22.5,
      -21.943045533438177
    ],
    [0, -21.943045533438177, 0, 0, 22.5, 0, 22.5, -21.943045533438177]
  ]
};

/** Canonical starting vertex only; preserve winding, duplicates and all coordinates. */
export function rotateOracleRing(coordinates: number[]): number[] {
  const points = Array.from({length: coordinates.length / 2}, (_, index) =>
    coordinates.slice(2 * index, 2 * index + 2)
  );
  const first = points.reduce(
    (best, point, index) =>
      point[0] < points[best][0] || (point[0] === points[best][0] && point[1] < points[best][1])
        ? index
        : best,
    0
  );
  return [...points.slice(first), ...points.slice(0, first)].flat();
}
