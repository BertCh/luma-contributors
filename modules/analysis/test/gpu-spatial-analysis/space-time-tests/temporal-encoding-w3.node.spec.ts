// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUSpaceTimeGuardedRelativeTimes} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-bands';
import {getGPUSpaceTimeRelativeTimes} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';

it('UNI-14 rejects encoded overflow and subnormal collapse', () => {
  expect(() => getGPUSpaceTimeGuardedRelativeTimes([0, 1e40], 1e35)).toThrow(/float32/);
  // IEEE-754 binary32 has no positive value below 2^-149; 1e-46 rounds to zero.
  expect(() => getGPUSpaceTimeGuardedRelativeTimes([0, 1e-46], 1e-46)).toThrow(/resolution/);
});

it('UNI-14 measures spacing at the encoded magnitude and floors subnormal spacing', () => {
  // The double rounds up to 2, whose binade has spacing 2^-22.
  expect(getGPUSpaceTimeRelativeTimes([0, 2 - 2 ** -25]).resolution).toBe(2 ** -22);
  expect(getGPUSpaceTimeRelativeTimes([0, 2 ** -149]).resolution).toBe(2 ** -149);
  expect(getGPUSpaceTimeRelativeTimes([0, 1e-46]).resolution).toBe(2 ** -149);
  expect(getGPUSpaceTimeRelativeTimes([0, 0]).resolution).toBe(0);
});

it('UNI-14 validates the rebased signed Int64 epoch and retains default-origin precision', () => {
  expect(() =>
    getGPUSpaceTimeGuardedRelativeTimes([1, 2], 4, {
      temporalContext: {unit: 'nanosecond', epoch: 2n ** 63n - 1n}
    })
  ).toThrow(/epoch/);
  expect([...getGPUSpaceTimeGuardedRelativeTimes([2 ** 24, 2 ** 24 + 1], 1).times]).toEqual([0, 1]);
});

it('S4 rejects unknown temporal units and invalid time zones on guarded preparation', () => {
  expect(() =>
    getGPUSpaceTimeGuardedRelativeTimes([0, 1], 4, {
      temporalContext: {unit: 'fortnight' as never, epoch: 0n}
    })
  ).toThrow(/temporal unit.*fortnight/);
  expect(() =>
    getGPUSpaceTimeGuardedRelativeTimes([0, 1], 4, {
      temporalContext: {unit: 'second', epoch: 0n, timeZone: ''}
    })
  ).toThrow(/timeZone.*non-empty/);
});
