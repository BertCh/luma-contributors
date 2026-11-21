// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createCostDistanceRoundRun,
  runCostDistanceRoundMatrix
} from './cost-distance-round-diagnostic';

it('checks exact corridor costs and controlled visibility counts without requiring parallel count parity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The 40-repetition matrix lives in the benchmark until its GPU duration is measured.
  await runCostDistanceRoundMatrix(device, 1);
}, 240000);

it('a one-round single-tile submission cannot certify the improving round', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const run = createCostDistanceRoundRun(device, 16, 1, 'forward');
  try {
    const first = await run.submit();
    expect(first.status).toEqual({iterationCount: 1, converged: 0, iterationLimitReached: 1});
    expect(first.costs.every(Number.isNaN)).toBe(true);
    const second = await run.submit();
    expect(second.status).toEqual({iterationCount: 2, converged: 1, iterationLimitReached: 0});
    expect(second.costs).toEqual(Array.from({length: 16}, (_, column) => column));
  } finally {
    run.destroy();
  }
});

it('continued sweeps use the cumulative phase and preserve ordered writes within the right sweep', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const run = createCostDistanceRoundRun(device, 128, 1, 'snapshot', 1, {
    sweepAfterIteration: 1,
    snapshotPlainOnly: true
  });
  try {
    // Rounds 0 and 1 are tiled. The right sweep at cumulative round index 2 carries unit
    // cost along the entire line. The left sweep at index 3 certifies the unchanged surface.
    // Values are integer sums below 128, so all expectations are exact in float32.
    for (const [execution, tail] of [Infinity, Infinity, 127, 127].entries()) {
      const result = await run.submit();
      expect(result.status).toEqual({
        iterationCount: execution + 1,
        converged: execution === 3 ? 1 : 0,
        iterationLimitReached: execution === 3 ? 0 : 1
      });
      expect((await run.readWorking())[127]).toBe(tail);
      if (execution < 3) expect(result.costs.every(Number.isNaN)).toBe(true);
      else expect(result.costs).toEqual(Array.from({length: 128}, (_, column) => column));
    }
  } finally {
    run.destroy();
  }
});
