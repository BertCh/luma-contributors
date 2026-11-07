// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  choosePointWeightsPath,
  getStreamedWeightsSettings
} from '../../../examples/deck/kepler-compute/src/analysis5-streamed-weights';
import {requireKeplerWholePointWeights} from '../src/analysis/point-weights-chunked';

const limits = {maxStorageBufferBindingSize: 134217728, maxBufferSize: 268435456};
const band = {
  kind: 'distance-band' as const,
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude' as const,
  metric: 'great-circle' as const,
  units: 'meters' as const,
  radiusMeters: 1000,
  neighborCapacity: 1_000_000
};

test('streams only when the single CSR exceeds the storage binding limit (or is forced)', () => {
  expect(choosePointWeightsPath(limits, band).path).toBe('single-csr');
  // 4 bytes per slot: 20 million slots is 80 MB, above a 64 MiB binding.
  const smallLimits = {maxStorageBufferBindingSize: 67108864, maxBufferSize: 268435456};
  const large = choosePointWeightsPath(smallLimits, {...band, neighborCapacity: 20_000_000});
  expect(large).toMatchObject({path: 'streamed', reason: 'device-limit'});
  expect(large.maximumSlotsPerChunk).toBe(67108864 / 4);
  expect(choosePointWeightsPath(smallLimits, {...band, neighborCapacity: 16_000_000}).path).toBe(
    'single-csr'
  );
  const forced = choosePointWeightsPath(limits, band, {
    forceStreamed: true,
    maximumSlotsPerChunk: 500
  });
  expect(forced).toMatchObject({path: 'streamed', reason: 'forced', maximumSlotsPerChunk: 500});
  // kNN has no streamed build.
  expect(
    choosePointWeightsPath(limits, {
      kind: 'knn',
      k: 8,
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude',
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'meters'
    }).path
  ).toBe('single-csr');
});

test('only chunk-aware point operations opt in to chunked weights', () => {
  expect(getStreamedWeightsSettings({kind: 'hot-spots-points', weights: band} as never)).toBe(band);
  // Quantile LISA, spatial rates and multivariate Geary became chunk-aware in WEIGHTS6.
  for (const kind of [
    'quantile-lisa-points',
    'spatial-rate-points',
    'multivariate-local-geary-points'
  ])
    expect(getStreamedWeightsSettings({kind, weights: band} as never)).toBe(band);
  expect(
    getStreamedWeightsSettings({kind: 'dbscan-points', weights: band} as never)
  ).toBeUndefined();
});

test('single-CSR-only consumers refuse chunked point weights', () => {
  expect(() => requireKeplerWholePointWeights({rowCount: 1, chunks: []} as never)).toThrow();
});
