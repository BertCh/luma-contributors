// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import type {GPUSpatialWeightsKernel} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';

/** Numeric used-slot allowance; guard words are compared separately and exactly. */
export function getTransformAuditTolerance(
  operation: string,
  scale: number,
  precision: 'cpu' | 'gpu' = 'gpu'
) {
  const allowance = precision === 'cpu' ? 1e-14 : 2e-6;
  return {
    absoluteTolerance: operation === 'average-reciprocal' ? allowance * Math.abs(scale) : allowance,
    relativeTolerance: allowance
  };
}

export function expectTransformAuditResult(
  actual: ArrayLike<number>,
  expected: readonly number[],
  operation: string,
  scale: number,
  label?: string
): void {
  const values = Array.from(actual);
  expectNumericArrayClose(values.slice(0, expected.length), expected, {
    label,
    ...getTransformAuditTolerance(operation, scale)
  });
  expect(values.slice(expected.length)).toEqual([-99, -99]);
}

// Independent NumPy definition and command: weights-transform-audit.oracle.py.
export const TRANSFORM_AUDIT_INPUT = {
  offsets: [0, 2, 4, 7, 8, 9, 10, 11, 11],
  neighbors: [1, 2, 0, 2, 0, 1, 3, 2, 5, 4, 7],
  weights: [3, 4, 1, 2, 0, 2, 1, 0, 2, 2, 0],
  distances: [0, 0.25, 0.5, 1, 1.5, 2, -0.5, NaN, Infinity, -Infinity, 0]
};

export const TRANSFORM_AUDIT_EXPECTED = {
  row: [3 / 7, 4 / 7, 1 / 3, 2 / 3, 0, 2 / 3, 1 / 3, 0, 1, 1, 0],
  binary: [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
  doubleOne: [3 / 17, 4 / 17, 1 / 17, 2 / 17, 0, 2 / 17, 1 / 17, 0, 2 / 17, 2 / 17, 0],
  doubleRows: [24 / 17, 32 / 17, 8 / 17, 16 / 17, 0, 16 / 17, 8 / 17, 0, 16 / 17, 16 / 17, 0],
  variance: [
    0.7890478095415622, 1.0520637460554163, 0.5881215132107468, 1.1762430264214936, 0,
    1.1762430264214936, 0.5881215132107468, 0, 1.3150796825692703, 1.3150796825692703, 0
  ],
  'average-reciprocal': [2, 2, 2, 2, 2, 2, 0.5, 0.5, 2, 2, 0]
};

export const KERNEL_AUDIT_EXPECTED = {
  triangular: [1, 0.75, 0.5, 0, 0, 0, 0, 0, 0, 0, 1],
  epanechnikov: [0.75, 0.703125, 0.5625, 0, 0, 0, 0, 0, 0, 0, 0.75],
  bisquare: [0.9375, 0.823974609375, 0.52734375, 0, 0, 0, 0, 0, 0, 0, 0.9375],
  quartic: [0.9375, 0.823974609375, 0.52734375, 0, 0, 0, 0, 0, 0, 0, 0.9375],
  uniform: [0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0.5],
  gaussian: [
    0.3989422804014327, 0.3866681168028493, 0.3520653267642995, 0.24197072451914337,
    0.12951759566589174, 0.05399096651318806, 0, 0, 0, 0, 0.3989422804014327
  ]
} satisfies Record<GPUSpatialWeightsKernel, readonly number[]>;
