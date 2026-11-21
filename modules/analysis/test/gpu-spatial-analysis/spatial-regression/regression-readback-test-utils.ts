// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {GPUSpatialMaximumLikelihoodReadback} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-model-result';

export function getMaximumLikelihoodReadback(
  table: number[],
  summary: number[],
  status: number[]
): GPUSpatialMaximumLikelihoodReadback {
  return {
    table: Float32Array.from(table),
    summary: Float32Array.from(summary),
    status: Uint32Array.from(status)
  };
}

export function close(actual: ArrayLike<number>, expected: number[], tolerance = 2e-4) {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, index) =>
    Number.isNaN(value)
      ? expect(actual[index]).toBeNaN()
      : expect(
          Math.abs(actual[index] - value),
          `element ${index}: ${actual[index]} vs ${value}`
        ).toBeLessThan(tolerance)
  );
}
