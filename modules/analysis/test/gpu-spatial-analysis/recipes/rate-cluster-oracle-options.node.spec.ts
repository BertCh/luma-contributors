// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeLocalMoranOracle} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {computeLocalPermutationOracle} from '../permutation-inference/permutation-oracle';
import {RATE_CLUSTER_ORACLE_OPTIONS} from './rate-cluster-oracle-options';

it('B3 both rate-cluster oracles restandardize after losing one of two neighbors', () => {
  const values = Float32Array.of(-1, 1, NaN);
  const weights = {
    offsets: Uint32Array.of(0, 1, 3, 4),
    neighbors: Uint32Array.of(1, 0, 2, 1),
    weights: Float32Array.of(1, 0.5, 0.5, 1)
  };
  const moran = computeLocalMoranOracle({weights, values, ...RATE_CLUSTER_ORACLE_OPTIONS});
  const permutation = computeLocalPermutationOracle({
    weights,
    values,
    seed: 11,
    permutationCount: 9,
    significanceLevel: 0.05,
    maximumNeighbors: 32,
    statistic: 'localMoran',
    ...RATE_CLUSTER_ORACLE_OPTIONS
  });
  // Hand calculation: surviving row-one weight=1, lag=-1, I=(2-1)*1*(-1)/2=-.5.
  expect(Number.isFinite(moran.spatialLag[1])).toBe(true);
  expect(moran.spatialLag[1]).toBe(-1);
  expect(moran.localI[1]).toBe(-0.5);
  expect(Number.isFinite(permutation.observed[1])).toBe(true);
  expect(permutation.observed[1]).toBe(-0.5);
});
