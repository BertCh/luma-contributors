// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {runExpansion} from '../../../src/computation-lenses/scenarios/gpu-network/expansion.lens';

test('directed SCC lens captures minimum-ID labels and convergence with padding and an isolated node', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runExpansion(device, 'network-strongly-connected-components');
  const captures = result.captures!.snapshots;
  const values = (name: string) => {
    const capture = captures.find(candidate => candidate.name === name);
    expect(
      capture,
      `Missing capture ${name}; available: ${captures.map(candidate => candidate.name).join(', ')}`
    ).toBeDefined();
    return Array.from(capture!.values);
  };
  // 0 <-> 1 -> 2, 2 -> 2, and isolated 3. The fifth neighbor slot is unused padding.
  expect(values('offsets')).toEqual([0, 1, 3, 4, 4]);
  expect(values('neighbors')).toEqual([1, 0, 2, 2, 0xffffffff]);
  expect(values('strongComponents')).toEqual([0, 0, 2, 3]);
  expect(values('converged')).toEqual([1]);
  expect(values('iterationLimitReached')).toEqual([0]);
  const [iterationCount] = values('iterationCount');
  expect(iterationCount).toBeGreaterThan(0);
  expect(iterationCount).toBeLessThanOrEqual(32);
});
