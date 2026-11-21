// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getGPURasterEmbeddingProjectionParameterValues,
  getRasterEmbeddingPrincipalComponents,
  getRasterEmbeddingStretchRanges
} from '@luma-contributors/analysis/raster-analysis';

// RT-121: a rank-deficient covariance (one varying band, two constant ones) has two zero
// eigenvalues. Its stretch ranges must still be accepted by the projection packer.
it('RT-121 PCA stretch ranges of constant components feed the projection packer', () => {
  const covariance = [1, 0, 0, 0, 0, 0, 0, 0, 0];
  const {eigenvalues} = getRasterEmbeddingPrincipalComponents(covariance, 3, 3);
  // Closed form: diag(1, 0, 0) has eigenvalues 1, 0, 0.
  expect(Array.from(eigenvalues)).toEqual([1, 0, 0]);
  const ranges = getRasterEmbeddingStretchRanges(eigenvalues, 2);
  // Leading component: +-2 sqrt(1). Constant components take the widest defined spread, +-2,
  // so their zero scores map to mid-grey 0.5.
  expect(ranges).toEqual([
    [-2, 2],
    [-2, 2],
    [-2, 2]
  ]);
  const parameters = getGPURasterEmbeddingProjectionParameterValues({stretchRanges: ranges});
  expect(Array.from(parameters)).toEqual([-2, 2, -2, 2, -2, 2]);
});

it('RT-121 an all-zero covariance gets unit stretch ranges', () => {
  const ranges = getRasterEmbeddingStretchRanges([0, 0, 0], 2);
  expect(ranges).toEqual([
    [-1, 1],
    [-1, 1],
    [-1, 1]
  ]);
  expect(() =>
    getGPURasterEmbeddingProjectionParameterValues({stretchRanges: ranges})
  ).not.toThrow();
});

it('RT-121 a spread that rounds to zero in float32 is replaced', () => {
  // 2 * sqrt(1e-100) = 2e-50, which is 0 in float32, so the packer would see high == low.
  const ranges = getRasterEmbeddingStretchRanges([4, 1e-100, 0], 2);
  expect(ranges).toEqual([
    [-4, 4],
    [-4, 4],
    [-4, 4]
  ]);
  expect(() =>
    getGPURasterEmbeddingProjectionParameterValues({stretchRanges: ranges})
  ).not.toThrow();
});

it('review N2 ignores spreads that overflow float32 when selecting a usable fallback', () => {
  // sqrt(1e100)*2 is finite in binary64 but infinite in float32. No component has
  // usable spread, so each range must take the documented all-undefined unit fallback.
  const ranges = getRasterEmbeddingStretchRanges([1e100, 0, 0]);
  expect(ranges).toEqual([
    [-1, 1],
    [-1, 1],
    [-1, 1]
  ]);
  expect(() =>
    getGPURasterEmbeddingProjectionParameterValues({stretchRanges: ranges})
  ).not.toThrow();
  const mixed = getRasterEmbeddingStretchRanges([1e100, 4, 0]);
  expect(mixed).toEqual([
    [-4, 4],
    [-4, 4],
    [-4, 4]
  ]);
  expect(() =>
    getGPURasterEmbeddingProjectionParameterValues({stretchRanges: mixed})
  ).not.toThrow();
});
