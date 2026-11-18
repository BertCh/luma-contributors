// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type NumericArrayTolerance = {
  /** Absolute error allowance, optionally computed per expected element. Defaults to zero. */
  absoluteTolerance?: number | ((expected: number, index: number) => number);
  /** Additional allowance multiplied by `abs(expected)`. Defaults to zero. */
  relativeTolerance?: number;
  /** Context included in a failed comparison. */
  label?: string;
};

/**
 * Checks equal lengths and `abs(actual - expected) <= absolute + relative * abs(expected)`.
 * Nonfinite values match only the same explicit expected value (including expected NaN).
 * Missing elements always fail. Tolerances must be finite and nonnegative.
 */
export function expectNumericArrayClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  options: NumericArrayTolerance = {}
): void {
  const label = options.label ?? 'numeric array';
  if (actual.length !== expected.length) {
    throw new Error(`${label}: length ${actual.length} != ${expected.length}`);
  }
  const relative = options.relativeTolerance ?? 0;
  if (!Number.isFinite(relative) || relative < 0)
    throw new Error(`${label}: invalid relative tolerance`);
  for (let index = 0; index < expected.length; index++) {
    const observed = actual[index];
    const target = expected[index];
    const mismatch = () => new Error(`${label}: row ${index}: ${observed} != ${target}`);
    if (typeof observed !== 'number' || typeof target !== 'number') throw mismatch();
    if (!Number.isFinite(observed) || !Number.isFinite(target)) {
      if (!Object.is(observed, target)) throw mismatch();
      continue;
    }
    const absolute =
      typeof options.absoluteTolerance === 'function'
        ? options.absoluteTolerance(expected[index], index)
        : (options.absoluteTolerance ?? 0);
    if (!Number.isFinite(absolute) || absolute < 0)
      throw new Error(`${label}: invalid absolute tolerance`);
    const tolerance = absolute + relative * Math.abs(target);
    if (Math.abs(observed - target) > tolerance) throw mismatch();
  }
}
