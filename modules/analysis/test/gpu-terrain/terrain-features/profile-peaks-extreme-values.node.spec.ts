// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {computeProfilePeaksOnCPU} from '../../../src/gpu-terrain/terrain-features/cpu-profile-peaks-operation';

it('N2 a converged kept peak can have overflowing float32 prominence', () => {
  // Largest finite binary32 is (2 - 2^-23) * 2^127. The exact prominence is twice
  // that value, so output rounding overflows even though the input samples are finite.
  const maximum = 3.4028234663852886e38;
  const rows = computeProfilePeaksOnCPU(
    Float32Array.of(-maximum, maximum, -maximum),
    Uint32Array.of(0, 3),
    Float32Array.of(1),
    {window: 1, minimumSide: 1, suppressionRadius: 0, capacity: 1}
  );
  expect(Array.from(rows.peakMask)).toEqual([0, 1, 0]);
  expect(rows.prominence.length).toBe(3);
  expect(rows.prominence[1]).toBe(Infinity);
  expect(Array.from(rows.count)).toEqual([1]);
  expect(Array.from(rows.ids)).toEqual([1]);
  expect(Array.from(rows.converged)).toEqual([1]);
  expect(Array.from(rows.iterationLimitReached)).toEqual([0]);
});

it('N2 sentinel documentation distinguishes overflowing kept values from unresolved peaks', () => {
  const source = readFileSync(
    new URL('../../../src/gpu-terrain/terrain-features/gpu-profile-peaks.ts', import.meta.url),
    'utf8'
  );
  const end = source.indexOf('export const GPU_PROFILE_PEAKS_UNRESOLVED_VALUE');
  const comment = source.slice(source.lastIndexOf('/**', end), end);
  expect(comment).toContain('overflow');
  expect(comment).toContain('output.peakMask');
  expect(comment).toContain('output.status');
  expect(comment).not.toContain('so it never');
});
