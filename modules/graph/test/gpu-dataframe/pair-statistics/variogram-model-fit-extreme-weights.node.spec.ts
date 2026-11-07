// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {COOPERATIVE_VARIOGRAM_FIT_WGSL} from '../../../src/gpu-dataframe/pair-statistics/variogram-model-fit-cooperative';

const round = Math.fround;

/** f32 mirror of the normal-equation guard in `fitLinearCooperative`. */
function getJointDeterminantGuard(weights: number[], shapes: number[], normalize: boolean) {
  const total = weights.reduce((sum, weight) => round(sum + weight), 0);
  const scale = normalize && total > 0 && total < 3.0e38 ? round(1 / total) : 1;
  let weightSum = 0;
  let shapeSum = 0;
  let squareSum = 0;
  weights.forEach((rawWeight, index) => {
    const weight = round(rawWeight * scale);
    weightSum = round(weightSum + weight);
    shapeSum = round(shapeSum + round(weight * shapes[index]));
    squareSum = round(squareSum + round(round(weight * shapes[index]) * shapes[index]));
  });
  const determinant = round(round(weightSum * squareSum) - round(shapeSum * shapeSum));
  return {determinant, threshold: round(round(1e-6 * weightSum) * squareSum)};
}

it('cooperative variogram fit normalizes extreme Cressie weights before the determinant guard', () => {
  // Cressie weights are 1 / max(fitted^2, 1e-20): near-zero fitted values give 1e20 per bin.
  const weights = [1e20, 1e20, 1e20, 1e20, 4, 1, 0.5, 0.25].map(round);
  const shapes = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1].map(round);

  const raw = getJointDeterminantGuard(weights, shapes, false);
  expect(Number.isFinite(raw.threshold) && Number.isFinite(raw.determinant)).toBe(false);

  const normalized = getJointDeterminantGuard(weights, shapes, true);
  expect(Number.isFinite(normalized.determinant)).toBe(true);
  expect(Number.isFinite(normalized.threshold)).toBe(true);
  expect(normalized.determinant).toBeGreaterThan(normalized.threshold);
});

it('cooperative variogram fit shader scales weights by their total and restores the residual', () => {
  expect(COOPERATIVE_VARIOGRAM_FIT_WGSL).toContain('weightScale');
  expect(COOPERATIVE_VARIOGRAM_FIT_WGSL.match(/weight\(bin\) \* weightScale/g)).toHaveLength(2);
  expect(COOPERATIVE_VARIOGRAM_FIT_WGSL).toContain('best.w = best.w / weightScale');
});
