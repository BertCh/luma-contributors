// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect} from 'vitest';

export function getSquareInput() {
  return {
    positions: Float32Array.of(0, 0, 1, 0, 0, 1, 1, 1),
    times: Float32Array.of(0.55, 0.7, 0.85, 1),
    parameters: Float32Array.of(16, 1, 2, 4, 1e-6, 0.1, 0.5, 0.1, 0.8)
  };
}

/** Direct dense intensity evaluation, independent of the fitter's log-domain responsibilities. */
export function evaluateSquare(estimates: ArrayLike<number>): number {
  const {positions, times} = getSquareInput();
  const [background, branching, decay, deviation] = Array.from(estimates);
  let result = -background * 16;
  for (let row = 0; row < 4; row++) {
    let intensity = background;
    for (let parent = 0; parent < row; parent++) {
      const squared =
        (positions[2 * row] - positions[2 * parent]) ** 2 +
        (positions[2 * row + 1] - positions[2 * parent + 1]) ** 2;
      intensity +=
        ((branching * decay) / (2 * Math.PI * deviation ** 2)) *
        Math.exp(-decay * (times[row] - times[parent]) - squared / (2 * deviation ** 2));
    }
    result += Math.log(intensity) - branching * -Math.expm1(-decay * (1 - times[row]));
  }
  return result;
}

export function expectSquareFit(
  estimates: ArrayLike<number>,
  fitted: number,
  converged: number
): void {
  expect(Number.isFinite(fitted)).toBe(true);
  expect(Math.abs(fitted - evaluateSquare(estimates))).toBeLessThan(2e-5);
  expect(converged).toBe(1);
  // SciPy optimization of mu, alpha=omega*beta, beta, variance gives the boundary
  // beta -> 0: mu=.0940703, alpha=2.772084, variance=.5805653, logL=-7.9753850.
  // Evaluate the numerical score in log parameters; at the constrained zero-decay
  // boundary its log-beta score tends to zero, while the one-sided beta score is negative.
  expect(Math.abs(fitted - -7.975385)).toBeLessThan(5e-4);
  for (let parameter = 0; parameter < 4; parameter++) {
    const lower = Array.from(estimates);
    const upper = Array.from(estimates);
    lower[parameter] *= Math.exp(-0.001);
    upper[parameter] *= Math.exp(0.001);
    const score = (evaluateSquare(upper) - evaluateSquare(lower)) / 0.002;
    expect(Number.isFinite(score)).toBe(true);
    expect(Math.abs(score)).toBeLessThan(0.003);
  }
}
