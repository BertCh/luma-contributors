// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {runExpansion} from '../../src/computation-lenses/scenarios/gpu-network/expansion.lens';
import {runNetworkOptimizationCoverage} from '../../src/computation-lenses/scenarios/gpu-network/optimization-coverage.lens';
import {runTrajectoryCoverage} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/trajectory-coverage.lens';

for (const kind of [
  'point-routing',
  'flow',
  'louvain',
  'louvain-sparse',
  'facility-optimization',
  'time-routing',
  'rational-access'
] as const) {
  it(`network ${kind} lens executes and captures actual results`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runNetworkOptimizationCoverage(device, kind);
    expect(result.captures?.snapshots.length).toBeGreaterThan(0);
    if (kind === 'facility-optimization' || kind === 'rational-access') {
      const snapshots = result.captures?.snapshots ?? [];
      const values = (name: string) =>
        Array.from(snapshots.find(capture => capture.name === name)!.values);
      expect(values('summary').slice(0, 3)).toEqual([1, 1, 1]);
      expect(snapshots.some(capture => capture.name === 'status')).toBe(false);
      if (kind === 'facility-optimization') {
        expect(values('assignments')).toEqual([1, 0]);
        expect(values('objective')).toEqual([3]);
      } else {
        expect(values('assignments')).toEqual([5, 5, 0, 0]);
        expect(values('unservedDemand')).toEqual([0, 3]);
      }
    }
    if (kind === 'louvain' || kind === 'louvain-sparse') {
      const snapshots = result.captures?.snapshots ?? [];
      const communities = snapshots.find(capture => capture.name === 'communities')!.values;
      expect(communities[0]).toBe(communities[1]);
      expect(communities[2]).toBe(communities[3]);
      expect(communities[0]).not.toBe(communities[2]);
      const trace = Array.from(
        snapshots.find(capture => capture.name === 'objectiveTrace')!.values
      ).filter(Number.isFinite);
      expect(trace[trace.length - 1]).toBeCloseTo(0.5, 6);
      if (kind === 'louvain-sparse') {
        expect(Array.from(communities)).toEqual([1, 1, 3, 3]);
        expect(trace).toEqual([-0.25, 0.5, 0.5, 0.5]);
        expect(snapshots.find(capture => capture.name === 'objectiveTrace')!.values).toHaveLength(
          9
        );
        expect(Array.from(snapshots.find(capture => capture.name === 'summary')!.values)).toEqual([
          1, 1, 2, 3
        ]);
      }
    }
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

it('k-shortest paths lens captures two directed diamond routes and counts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runExpansion(device, 'k-shortest-paths');
  const snapshots = result.captures?.snapshots ?? [];
  const values = (name: string) =>
    Array.from(snapshots.find(capture => capture.name === name)!.values);
  expect(values('costs')).toEqual([2, 3]);
  expect(values('lengths')).toEqual([3, 3]);
  expect(values('counts')).toEqual([2]);
  expect(values('vertices')).toEqual([0, 1, 3, 0xffffffff, 0, 2, 3, 0xffffffff]);
  expect(values('edges')).toEqual([0, 2, 0xffffffff, 0xffffffff, 1, 3, 0xffffffff, 0xffffffff]);
  expect(values('converged')).toEqual([1]);
  expect(values('iterationLimitReached')).toEqual([0]);
  expect(values('candidateOverflow')).toEqual([0]);
});
