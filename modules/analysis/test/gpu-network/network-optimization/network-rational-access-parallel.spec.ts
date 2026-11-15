// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkRationalAccess} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOptimizationHarness} from './optimization-test-harness';

for (const originCount of [1, 17, 33, 64]) {
  for (const maximumCycles of [0, 1, 4096]) {
    it(`balances ${originCount} tied origins with ${maximumCycles} cycles, including inactive lanes`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createOptimizationHarness(device);
      const facilityCount = 32;
      const costs = new Float32Array(originCount * facilityCount).fill(Infinity);
      for (let origin = 0; origin < originCount; origin++) {
        costs[origin * facilityCount] = 0;
        costs[origin * facilityCount + 1] = 0;
      }
      const status = {
        invalidCount: harness.output('uint32', 1),
        iterationCount: harness.output('uint32', 1),
        converged: harness.output('uint32', 1),
        iterationLimitReached: harness.output('uint32', 1)
      };
      const output = {
        assignments: harness.output('float32', costs.length),
        scores: harness.output('float32', originCount),
        unservedDemand: harness.output('float32', originCount),
        summary: harness.output('uint32', 4),
        status
      };
      const supply = harness.input('float32', new Float32Array(facilityCount).fill(originCount));
      const result = await harness.run(
        new GPUNetworkRationalAccess({
          costs: harness.input('float32', costs),
          demand: harness.input('float32', new Float32Array(originCount).fill(2)),
          supply,
          parameters: harness.input('float32', Float32Array.of(1, 1, 1, 1, 50, 0)),
          maximumCycles,
          initialLimitCycles: 1,
          output
        }),
        first => {
          const solved = maximumCycles > 0;
          // Initially all origins choose facility zero (cost 2). Simultaneous requests to
          // facility one are capped by its capacity, so every origin moves one unit.
          // Both loads then equal originCount, giving cost 1 and gap zero in one cycle.
          const assignments = first(output.assignments);
          for (let origin = 0; origin < originCount; origin++) {
            for (let facility = 0; facility < facilityCount; facility++) {
              expect(assignments[origin * facilityCount + facility]).toBe(
                facility === 0 ? (solved ? 1 : 2) : facility === 1 && solved ? 1 : 0
              );
            }
          }
          // Float32 division on the GPU can differ from the closed form by one ULP
          // (for example, 66 / 33). Use the CPU twin's documented relative 2e-5 tolerance.
          const expectedScore = solved ? 1 : 2;
          const scores = first(output.scores);
          expect(scores).toHaveLength(originCount);
          for (const score of scores)
            expect(Math.abs(score - expectedScore)).toBeLessThanOrEqual(2e-5 * expectedScore);
          expect(first(output.unservedDemand)).toEqual(new Array(originCount).fill(0));
          const summary = first(output.summary);
          expect(summary).toHaveLength(4);
          expect(summary.slice(0, 3)).toEqual([1, Number(solved), Number(solved)]);
          const gap = new Float32Array(Uint32Array.of(summary[3]).buffer)[0];
          const expectedGap = solved ? 0 : 2;
          expect(Math.abs(gap - expectedGap)).toBeLessThanOrEqual(2e-5 * expectedGap);
          expect(first(status.invalidCount)).toEqual([0]);
          expect(first(status.iterationCount)).toEqual([Number(solved)]);
          expect(first(status.converged)).toEqual([Number(solved)]);
          expect(first(status.iterationLimitReached)).toEqual([Number(!solved)]);
          // Reusing the compiled graph must leave no workgroup state or stale scalar ports.
          harness.rewrite(supply, new Float32Array(facilityCount));
        }
      );
      expect(result(status.invalidCount)).toEqual([1]);
      expect(result(status.iterationCount)).toEqual([0]);
      expect(result(status.converged)).toEqual([0]);
      expect(result(status.iterationLimitReached)).toEqual([0]);
      expect(result(output.assignments)).toEqual(new Array(costs.length).fill(0));
      expect(result(output.scores).every(Number.isNaN)).toBe(true);
      expect(result(output.unservedDemand)).toEqual(new Array(originCount).fill(2));
    });
  }
}

it('publishes only scores preceding the first nonfinite origin in a cooperative evaluation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const originCount = 17;
  const costs = new Float32Array(originCount * 2).fill(Infinity);
  for (let origin = 0; origin < originCount; origin++) costs[origin * 2] = 0;
  // Travel at origin eight overflows after normalization. Later origins must retain NaN
  // scores even though their lanes can finish evaluation before the failing lane.
  costs[16] = new Float32Array(Uint32Array.of(0x7f7fffff).buffer)[0];
  const status = {
    invalidCount: harness.output('uint32', 1),
    iterationCount: harness.output('uint32', 1),
    converged: harness.output('uint32', 1),
    iterationLimitReached: harness.output('uint32', 1)
  };
  const output = {
    assignments: harness.output('float32', costs.length),
    scores: harness.output('float32', originCount),
    unservedDemand: harness.output('float32', originCount),
    summary: harness.output('uint32', 4),
    status
  };
  const result = await harness.run(
    new GPUNetworkRationalAccess({
      costs: harness.input('float32', costs),
      demand: harness.input('float32', new Float32Array(originCount).fill(1)),
      supply: harness.input('float32', Float32Array.of(32, 32)),
      parameters: harness.input('float32', Float32Array.of(0.5, 1, 1, 1, 50, 0)),
      output
    })
  );
  const scores = result(output.scores);
  expect(scores.slice(0, 8)).toEqual(new Array(8).fill(17 / 32));
  expect(scores.slice(8).every(Number.isNaN)).toBe(true);
  expect(result(output.assignments)).toEqual(
    Array.from({length: costs.length}, (_, index) => Number(index % 2 === 0))
  );
  expect(result(output.unservedDemand)).toEqual(new Array(originCount).fill(0));
  expect(result(output.summary)).toEqual([0, 0, 0, 0]);
  expect(result(status.invalidCount)).toEqual([1]);
  expect(result(status.iterationCount)).toEqual([0]);
  expect(result(status.converged)).toEqual([0]);
  expect(result(status.iterationLimitReached)).toEqual([0]);
});

it('balances fractional demand across both subgroups while keeping zero and unreachable scores missing', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const originCount = 33;
  const facilityCount = 32;
  const costs = new Float32Array(originCount * facilityCount).fill(Infinity);
  for (let origin = 0; origin < originCount - 1; origin++) {
    costs[origin * facilityCount] = 0;
    costs[origin * facilityCount + 1] = 0;
  }
  const demand = new Float32Array(originCount).fill(0.5);
  demand[0] = 0;
  const output = {
    assignments: harness.output('float32', costs.length),
    scores: harness.output('float32', originCount),
    unservedDemand: harness.output('float32', originCount),
    summary: harness.output('uint32', 4)
  };
  // 31 served origins, each with demand 1/2. Each of two facilities gets load 31/4
  // after one early-limited shift, exactly matching its supply and giving unit congestion.
  const result = await harness.run(
    new GPUNetworkRationalAccess({
      costs: harness.input('float32', costs),
      demand: harness.input('float32', demand),
      supply: harness.input('float32', new Float32Array(facilityCount).fill(31 / 4)),
      parameters: harness.input('float32', Float32Array.of(1, 1, 1, 1, 50, 1e-4)),
      demandMode: 'fractional',
      initialLimitCycles: 1,
      maximumCycles: 4,
      output
    })
  );
  const assignments = result(output.assignments);
  const scores = result(output.scores);
  for (let origin = 0; origin < originCount; origin++) {
    const served = origin > 0 && origin < originCount - 1;
    for (let facility = 0; facility < facilityCount; facility++) {
      expect(assignments[origin * facilityCount + facility]).toBeCloseTo(
        served && facility < 2 ? 0.25 : 0,
        5
      );
    }
    if (served) expect(scores[origin]).toBeCloseTo(1, 5);
    else expect(Number.isNaN(scores[origin])).toBe(true);
  }
  expect(result(output.unservedDemand)).toEqual([...new Array(originCount - 1).fill(0), 0.5]);
  expect(result(output.summary).slice(0, 3)).toEqual([1, 1, 1]);
  const gap = new Float32Array(Uint32Array.of(result(output.summary)[3]).buffer)[0];
  expect(gap).toBeLessThanOrEqual(1e-4);
});
