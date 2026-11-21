// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Opt-in timing comparison: yarn test-benchmarks. The -bench.spec.ts suffix excludes this
// file from ordinary correctness projects; the deterministic access-count case stays there.
import {expect, it} from 'vitest';
import {computeTrajectoryLocationsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';

it('benchmarks 4000 singleton locations with bounded and exact CPU execution', () => {
  const total = 4000;
  const input = {
    centroids: new Float32Array(total * 2),
    tracks: Uint32Array.from({length: total}, (_, row) => row),
    count: Uint32Array.of(total),
    parameters: Float32Array.of(1, 0, 0, 0)
  };
  const started = performance.now();
  const bounded = computeTrajectoryLocationsOnCPU(input, {maximumIterations: 4});
  const boundedMilliseconds = performance.now() - started;
  const exactStarted = performance.now();
  const exact = computeTrajectoryLocationsOnCPU(input);
  console.info('locations singleton benchmark', {
    total,
    boundedMilliseconds,
    exactMilliseconds: performance.now() - exactStarted
  });
  // Every track is a singleton: the hand-derived location identity is the input row.
  expect(Array.from(bounded.locationIds)).toEqual(Array.from({length: total}, (_, row) => row));
  expect(Array.from(exact.locationIds)).toEqual(Array.from({length: total}, (_, row) => row));
  expect(Array.from(bounded.locationCount)).toEqual([4000]);
  expect(Array.from(exact.locationCount)).toEqual([4000]);
});
