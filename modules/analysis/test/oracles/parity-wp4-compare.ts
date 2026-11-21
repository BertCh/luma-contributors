// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {getParityScale, measureParity} from './parity-measurements';

/** Full-array measurement; NaN/Infinity masks and lengths are always exact. */
export function compareRasterParity(
  className: string,
  caseName: string,
  reference: string,
  metric: string,
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  profile: 'L' | 'S' | 'E'
): void {
  const scale = getParityScale(expected);
  const absolute = profile === 'E' ? 0 : 1e-6 * scale;
  const relative = profile === 'E' ? 0 : 1e-4;
  const {maxAbs, maxRel, mismatches} = measureParity(
    actual,
    expected,
    target => absolute + relative * Math.abs(target),
    (error, target) => error / Math.max(Math.abs(target), 1e-300)
  );
  const scaledResidual = maxAbs / scale;
  const tolerance = {profile, absolute, relative, scaledResidual: profile === 'L' ? 1e-4 : 0};
  console.log(
    'PARITY',
    JSON.stringify({
      class: className,
      case: caseName,
      reference,
      n: expected.length,
      metric,
      maxAbs,
      maxRel,
      mismatches,
      tolerance
    })
  );
  expect(mismatches, `${className}/${caseName}/${metric}`).toBe(0);
  if (profile === 'L')
    expect(scaledResidual, `${metric} scaled residual`).toBeLessThanOrEqual(1e-4);
}
