// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeMantelTestOnCPU} from '../../../src/gpu-spatial-analysis/space-time-tests/cpu-mantel-test-operation';
import {
  getGPUSpaceTimeParameterValues,
  GPU_SPACE_TIME_SUMMARY
} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';

const input = {
  offsets: Uint32Array.of(0, 2, 3, 3),
  neighbors: Uint32Array.of(1, 2, 2),
  distances: Float32Array.of(1, 1, 1),
  weights: Float32Array.of(1, 1, 1),
  times: Float32Array.of(0, 1, 3),
  parameters: getGPUSpaceTimeParameterValues({seed: 0, permutationCount: 19})
};

it('NUM-03 reports undefined observed correlations without usable inference, in full and band populations', () => {
  for (const fixture of [
    input,
    {...input, offsets: Uint32Array.of(0, 1, 1, 1)},
    {...input, offsets: new Uint32Array(4)},
    {...input, distances: Float32Array.of(1, 2, 3), times: new Float32Array(3)}
  ]) {
    const result = computeMantelTestOnCPU(fixture, {maximumPermutations: 19});
    const pairCount = fixture.offsets[3];
    expect(result.summary[GPU_SPACE_TIME_SUMMARY.pairCount]).toBe(pairCount);
    expect([...result.statistics]).toEqual(Array(20).fill(NaN));
    expect(result.summary[GPU_SPACE_TIME_SUMMARY.observed]).toBeNaN();
    expect([...result.summary.slice(4)]).toEqual(Array(8).fill(NaN));
    expect([...result.permutationCount]).toEqual([0]);
  }
});

it('NUM-03 excludes undefined null draws from both tails and the denominator', () => {
  // Two spatial distances [1,2] and observed gaps [1,2] give Pearson r=1.
  // Under all six labelings, two focal-middle assignments have gaps [1,1]
  // and undefined r; the four remaining correlations are exactly +/-1.
  const result = computeMantelTestOnCPU(
    {
      ...input,
      offsets: Uint32Array.of(0, 2, 2, 2),
      distances: Float32Array.of(1, 2, 0),
      times: Float32Array.of(0, 1, 2)
    },
    {maximumPermutations: 19}
  );
  expect(result.statistics[0]).toBe(1);
  const usable = [...result.statistics.slice(1)].filter(Number.isFinite);
  // Independent Python integer shuffle in scratch/null-draws.py gives these 19 labels;
  // hand Pearson signs give 16 usable draws, six upper ties and p=(6+1)/(16+1).
  expect([...result.statistics]).toEqual([
    1,
    -1,
    NaN,
    -1,
    1,
    1,
    -1,
    NaN,
    1,
    NaN,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1
  ]);
  expect(usable.length).toBe(16);
  expect(usable.every(value => value === 1 || value === -1)).toBe(true);
  expect(result.permutationCount[0]).toBe(19);
  expect(result.usablePermutationCount[0]).toBe(16);
  expect(result.summary[GPU_SPACE_TIME_SUMMARY.pseudoPGreater]).toBe(Math.fround(7 / 17));
});

it('TWN-11 checks shapes then suppresses loss before interpreting offsets', () => {
  for (const terminal of [1, 2]) {
    const fixture = {
      ...input,
      offsets: Uint32Array.of(0, 1, terminal),
      neighbors: Uint32Array.of(1),
      weights: Float32Array.of(1),
      distances: Float32Array.of(1),
      times: Float32Array.of(0, 1),
      status: {overflow: Uint32Array.of(1)}
    };
    const result = computeMantelTestOnCPU(fixture, {maximumPermutations: 3});
    expect([...result.statistics]).toEqual([NaN, NaN, NaN, NaN]);
    expect([...result.summary]).toEqual(Array(12).fill(NaN));
    expect([...result.permutationCount]).toEqual([0]);
    if (terminal === 2)
      expect(() =>
        computeMantelTestOnCPU({...fixture, status: undefined}, {maximumPermutations: 3})
      ).toThrow(/offsets/);
    expect(() =>
      computeMantelTestOnCPU({...fixture, weights: new Float32Array(0)}, {maximumPermutations: 3})
    ).toThrow(/weights/);
  }
});

it('NUM-03 leaves inference unavailable with zero usable nulls and stops over usable draws', () => {
  const fixture = {
    ...input,
    offsets: Uint32Array.of(0, 2, 2, 2),
    distances: Float32Array.of(1, 2, 0),
    times: Float32Array.of(0, 1, 2)
  };
  // Independent Python: seed 5's first order is [1,2,0], giving equal gaps [1,1].
  const emptyNull = computeMantelTestOnCPU(
    {...fixture, parameters: getGPUSpaceTimeParameterValues({seed: 5, permutationCount: 1})},
    {maximumPermutations: 1}
  );
  expect([...emptyNull.statistics]).toEqual([1, NaN]);
  expect(emptyNull.summary[0]).toBe(1);
  expect([...emptyNull.summary.slice(4)].every(Number.isNaN)).toBe(true);
  expect(emptyNull.permutationCount[0]).toBe(1);
  expect(emptyNull.usablePermutationCount[0]).toBe(0);
  // Seed 0's first four draws are [-1,NaN,-1,+1]; h=1 stops at attempt 4,
  // using three correlations, so the sequential probability is 1/3, not 1/4.
  const stopped = computeMantelTestOnCPU(fixture, {
    maximumPermutations: 19,
    stopAfterExceedances: 1
  });
  expect([...stopped.statistics.slice(0, 5)]).toEqual([1, -1, NaN, -1, 1]);
  expect(stopped.statistics.slice(5).every(value => value === 0)).toBe(true);
  expect(stopped.permutationCount[0]).toBe(4);
  expect(stopped.usablePermutationCount[0]).toBe(3);
  expect(stopped.summary[8]).toBe(Math.fround(1 / 3));
});
