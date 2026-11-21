// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {measureParity} from './parity-measurements';

/** S profile, per-element output scale; exact integer counts and non-finite masks. */
export function compareParity(
  className: string,
  caseName: string,
  reference: string,
  rowCount: number,
  metric: string,
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  exact = false
): void {
  const {maxAbs, maxRel, mismatches} = measureParity(
    actual,
    expected,
    target => (exact ? 0 : 1e-6 * Math.max(1, Math.abs(target)) + 1e-4 * Math.abs(target)),
    (error, target) => (target === 0 ? (error === 0 ? 0 : Infinity) : error / Math.abs(target))
  );
  const tolerance = exact
    ? {profile: 'E', absolute: 0, relative: 0}
    : {profile: 'S', absolute: '1e-6 * max(1, abs(reference element))', relative: 1e-4};
  console.log(
    'PARITY',
    JSON.stringify({
      class: className,
      case: caseName,
      reference,
      n: rowCount,
      metric,
      maxAbs,
      maxRel: Number.isFinite(maxRel) ? maxRel : 'Infinity',
      mismatches,
      tolerance
    })
  );
  // Soft assertions keep the remaining full-array measurements visible on a failing run.
  expect.soft(mismatches, `${className}/${caseName}/${metric}`).toBe(0);
}
