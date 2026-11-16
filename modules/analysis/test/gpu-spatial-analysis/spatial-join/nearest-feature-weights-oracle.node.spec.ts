// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {assertValidCSR} from '../spatial-weights/spatial-weights-oracle';
import {assertValidCrossWeights} from './nearest-feature-weights-oracle';

it.each([
  64, 65, 256, 1024
])('validates two query rows with %i feature columns per row', capacity => {
  // On a line with feature coordinates 1100 - featureRow and queries at 0 and -1,
  // both rows retain feature IDs 1100 - capacity through 1099, sorted ascending.
  const neighbors = Array.from(
    {length: 2 * capacity},
    (_, slot) => 1100 - capacity + (slot % capacity)
  );
  const distances = neighbors.map(
    (neighbor, slot) => 1100 - neighbor + Math.floor(slot / capacity)
  );
  const weights = {
    offsets: [0, capacity, 2 * capacity],
    neighbors,
    distances,
    weights: distances.map(distance => 1 / distance)
  };
  expect(() => assertValidCrossWeights(weights, 2, 1100, 'cross')).not.toThrow();
  // This is the exact square/self-join precondition mismatch from the headless failure.
  expect(() => assertValidCSR(weights, 2, 'square')).toThrow('neighbor out of range');
});

it('accepts matching numeric query and feature IDs in separate datasets', () => {
  expect(() =>
    assertValidCrossWeights(
      {offsets: [0, 1, 2], neighbors: [0, 1], weights: [1, 1], distances: [0, 0]},
      2,
      2,
      'cross'
    )
  ).not.toThrow();
});

it('rejects feature IDs outside the feature domain even when they fit the query domain', () => {
  expect(() =>
    assertValidCrossWeights(
      {offsets: [0, 0, 0, 1], neighbors: [2], weights: [1], distances: [1]},
      3,
      2,
      'cross'
    )
  ).toThrow('neighbor out of feature range');
});
