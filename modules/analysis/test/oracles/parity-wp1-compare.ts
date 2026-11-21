// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {getParityScale, measureParity} from './parity-measurements';

export function compareParity(
  className: string,
  testCase: {name: string; reference: string; n: number},
  metric: string,
  actual: readonly number[],
  expected: readonly number[],
  profile = 'S',
  absoluteOverride?: number,
  relativeOverride?: number
): void {
  const scale = getParityScale(expected);
  const absolute = absoluteOverride ?? (profile === 'E' ? 0 : 1e-6 * scale);
  const relative = relativeOverride ?? (profile === 'E' ? 0 : 1e-4);
  const {maxAbs, maxRel, mismatches} = measureParity(
    actual,
    expected,
    target => absolute + relative * Math.abs(target),
    (error, target) => error / Math.max(Math.abs(target), Number.MIN_VALUE)
  );
  console.log(
    'PARITY',
    JSON.stringify({
      class: className,
      case: testCase.name,
      reference: testCase.reference,
      n: testCase.n,
      metric,
      maxAbs,
      maxRel,
      mismatches,
      tolerance: {profile, absolute, relative}
    })
  );
  expect.soft(mismatches, `${className} ${testCase.name}: ${metric}`).toBe(0);
}
