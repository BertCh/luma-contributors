// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {
  GPURasterEmbeddingKNearestClassifier,
  type GPURasterEmbeddingKNearestClassifierProps
} from '@luma-contributors/analysis/raster-analysis';

const NONE = 0xffffffff;

export function createPublicSearchClassifier(
  searchStructure: 'brute-force' | 'tiled',
  props: Omit<GPURasterEmbeddingKNearestClassifierProps, 'searchStructure' | 'id'>
) {
  return new GPURasterEmbeddingKNearestClassifier({
    ...props,
    id: `public-search-${searchStructure}`,
    searchStructure
  });
}

export function expectPublicSearchResults(
  result: Record<string, number[]>,
  metric: 'euclidean' | 'cosine',
  neighborCount: number
): void {
  const expectedIds =
    metric === 'euclidean'
      ? [0, 255, 256, 257, 0, 255, 256, 257, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE]
      : neighborCount === 1
        ? [255, 255, NONE, NONE]
        : [255, 256, 255, 256, NONE, NONE, NONE, NONE];
  expect(result['ids']).toEqual(expectedIds);
  expect(result['counts']).toEqual([neighborCount, neighborCount, 0, 0]);
  const expectedScores =
    metric === 'euclidean'
      ? [
          2,
          2,
          2,
          Infinity,
          Infinity,
          Infinity,
          Infinity,
          Infinity,
          Infinity,
          Infinity,
          Infinity,
          Infinity,
          Infinity,
          Infinity,
          Infinity,
          Infinity
        ]
      : neighborCount === 1
        ? [1, 1, -Infinity, -Infinity]
        : [1, 1, 1, 1, -Infinity, -Infinity, -Infinity, -Infinity];
  expect(result['scores']).toHaveLength(expectedScores.length);
  for (const [index, expected] of expectedScores.entries()) {
    if (Number.isFinite(expected)) {
      expect(Number.isFinite(result['scores'][index])).toBe(true);
      expect(result['scores'][index]).toBeCloseTo(expected, 6);
    } else expect(result['scores'][index]).toBe(expected);
  }
  for (const route of ['brute-force', 'tiled']) {
    expect(result[`${route}-confidence`]).toHaveLength(4);
    expect(result[`${route}-shares`]).toHaveLength(16);
    expect(result[`${route}-labels`]).toEqual(
      metric === 'euclidean'
        ? [0, NONE, NONE, NONE]
        : neighborCount === 1
          ? [1, 1, NONE, NONE]
          : [0, 0, NONE, NONE]
    );
    for (let pixel = 0; pixel < 4; pixel++) {
      const missing = pixel >= (metric === 'euclidean' ? 1 : 2);
      if (missing) {
        expect(result[`${route}-confidence`][pixel]).toBeNaN();
        for (let category = 0; category < 4; category++)
          expect(result[`${route}-shares`][pixel * 4 + category]).toBeNaN();
      } else {
        const shares =
          metric === 'euclidean'
            ? [1 / 3, 1 / 3, 1 / 3, 0]
            : neighborCount === 1
              ? [0, 1, 0, 0]
              : [1 / 2, 1 / 2, 0, 0];
        expect(Number.isFinite(result[`${route}-confidence`][pixel])).toBe(true);
        expect(result[`${route}-confidence`][pixel]).toBeCloseTo(Math.max(...shares), 6);
        for (const [category, share] of shares.entries()) {
          expect(Number.isFinite(result[`${route}-shares`][pixel * 4 + category])).toBe(true);
          expect(result[`${route}-shares`][pixel * 4 + category]).toBeCloseTo(share, 6);
        }
      }
    }
  }
}
