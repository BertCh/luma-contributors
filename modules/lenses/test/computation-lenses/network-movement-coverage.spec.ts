// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '@luma-contributors/graph/testing';
import {runNetworkOptimizationCoverage} from '../../src/computation-lenses/scenarios/gpu-network/optimization-coverage.lens';
import {runTrajectoryCoverage} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/trajectory-coverage.lens';

for (const kind of [
  'point-routing',
  'flow',
  'louvain',
  'facility-optimization',
  'time-routing',
  'rational-access'
] as const) {
  it(`network ${kind} lens executes and captures actual results`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runNetworkOptimizationCoverage(device, kind);
    expect(result.captures?.snapshots.length).toBeGreaterThan(0);
  });
}
for (const kind of [
  'trajectory-kalman',
  'trajectory-restriction',
  'track-similarity-search'
] as const) {
  it(`movement ${kind} lens executes and captures actual results`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runTrajectoryCoverage(device, kind);
    const captures = result.captures?.snapshots ?? [];
    expect(captures.length).toBeGreaterThan(0);
    if (kind === 'trajectory-kalman')
      expect(captures.find(capture => capture.name === 'smoothedState')?.values[0]).toBeCloseTo(
        2 / 3,
        5
      );
    if (kind === 'trajectory-restriction')
      expect(captures.find(capture => capture.name === 'intervals')?.values[0]).toBeCloseTo(1, 5);
    if (kind === 'track-similarity-search')
      expect(Array.from(captures.find(capture => capture.name === 'matched')!.values)).toEqual([
        1, 0
      ]);
  });
}
