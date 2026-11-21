// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export function getParityScale(expected: ArrayLike<number>): number {
  let scale = 1;
  for (let index = 0; index < expected.length; index++) {
    if (Number.isFinite(expected[index])) scale = Math.max(scale, Math.abs(expected[index]));
  }
  return scale;
}

/** Count missing/extra rows once and compare nonfinite masks exactly in every wave. */
export function measureParity(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  bound: (target: number) => number,
  relativeError: (error: number, target: number) => number
): {maxAbs: number; maxRel: number; mismatches: number} {
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  for (let index = 0; index < Math.min(actual.length, expected.length); index++) {
    const observed = actual[index];
    const target = expected[index];
    if (!Number.isFinite(observed) || !Number.isFinite(target)) {
      if (!Object.is(observed, target)) mismatches++;
      continue;
    }
    const error = Math.abs(observed - target);
    maxAbs = Math.max(maxAbs, error);
    maxRel = Math.max(maxRel, relativeError(error, target));
    if (error > bound(target)) mismatches++;
  }
  return {maxAbs, maxRel, mismatches};
}
