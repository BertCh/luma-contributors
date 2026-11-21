// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Shell [0,10]^2 and clockwise hole [3,7]^2 meet at t=1.5 on the square [1.5,8.5]^2. */
export const SQUARE_HOLE_INPUT = {
  positions: Float32Array.of(0, 0, 10, 0, 10, 10, 0, 10, 3, 3, 3, 7, 7, 7, 7, 3),
  ringOffsets: Uint32Array.of(0, 4, 8),
  polygonOffsets: Uint32Array.of(0, 2),
  featureOffsets: Uint32Array.of(0, 1)
};

/**
 * The first reflex split visits the hole's lower-left corner, then its merged component visits
 * shell corner 0, the other hole corners, and shell corners 1–3. Terminal ridges follow supports
 * bottom, right, top, left. Coordinates follow unit-speed wavefront planes, independent of code.
 */
export const SQUARE_HOLE_SEGMENTS = [
  3, 3, 1.5, 1.5, 0, 0, 1.5, 1.5, 3, 7, 1.5, 8.5, 7, 7, 8.5, 8.5, 7, 3, 8.5, 1.5, 10, 0, 8.5, 1.5,
  10, 10, 8.5, 8.5, 0, 10, 1.5, 8.5, 8.5, 1.5, 1.5, 1.5, 8.5, 1.5, 8.5, 8.5, 8.5, 8.5, 1.5, 8.5,
  1.5, 8.5, 1.5, 1.5
];

/** Original boundary edges adjacent to each corner or facing each other at a terminal ridge. */
export const SQUARE_HOLE_SOURCE_EDGES = [
  7, 4, 3, 0, 4, 5, 5, 6, 6, 7, 0, 1, 1, 2, 2, 3, 7, 0, 1, 6, 2, 5, 3, 4
];
