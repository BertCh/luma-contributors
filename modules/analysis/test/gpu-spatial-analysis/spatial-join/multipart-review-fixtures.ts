// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {MultipartFixtureGeometry} from './multipart-predicates-test-utils';

export const REVIEW_DISTANCE_THRESHOLD = {
  left: {kind: 'points', positions: [200, 200, -2.25, 11.5], featureOffsets: [0, 2]},
  right: {kind: 'lines', positions: [0, 0, 5, 12], lineOffsets: [0, 2]}
} satisfies {left: MultipartFixtureGeometry; right: MultipartFixtureGeometry};

export const REVIEW_TRANSLATED_DISTANCE = {
  left: {
    kind: 'points',
    positions: [1000000.125, 1000000, 1000002, 1000002],
    featureOffsets: [0, 2]
  },
  right: {kind: 'lines', positions: [1000000, 1000000, 1000001, 1000000.25], lineOffsets: [0, 2]}
} satisfies {left: MultipartFixtureGeometry; right: MultipartFixtureGeometry};

export const REVIEW_NARROW_HOLE = {
  left: {
    kind: 'lines',
    positions: [-1, 0, 1, 0, -1, 0.5, -0.5, 0.5],
    lineOffsets: [0, 2, 4],
    featureOffsets: [0, 2]
  },
  right: {
    kind: 'polygons',
    positions: [-2, -2, 2, -2, 2, 2, -2, 2, 0, -1, 0, 1, 2 ** -28, 1, 2 ** -28, -1],
    featureOffsets: [0, 1],
    polygonOffsets: [0, 2],
    ringOffsets: [0, 4, 8]
  }
} satisfies {left: MultipartFixtureGeometry; right: MultipartFixtureGeometry};

const translatedSquare = {
  kind: 'polygons',
  positions: [
    1000000, 1000000, 1000000.0625, 1000000, 1000000.0625, 1000000.0625, 1000000, 1000000.0625
  ],
  featureOffsets: [0, 1],
  polygonOffsets: [0, 1],
  ringOffsets: [0, 4]
} satisfies MultipartFixtureGeometry;
export const REVIEW_EQUAL_SQUARE = {left: translatedSquare, right: translatedSquare};
