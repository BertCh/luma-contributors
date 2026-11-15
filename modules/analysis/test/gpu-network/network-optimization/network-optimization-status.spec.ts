// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkRationalAccess} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access';
import {GPUNetworkFlow} from '../../../src/gpu-network/network-optimization/gpu-network-flow';
import {GPUNetworkTimeRouting} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOptimizationHarness} from './optimization-test-harness';

for (const maximumAugmentations of [0, 1, 2]) {
  it(`flow publishes a feasible incumbent and named stopping flags at ${maximumAugmentations} augmentations`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createOptimizationHarness(device);
    const status = {
      invalidCount: harness.output('uint32', 1),
      iterationCount: harness.output('uint32', 1),
      converged: harness.output('uint32', 1),
      iterationLimitReached: harness.output('uint32', 1)
    };
    const output = {
      flow: harness.output('uint32', 4),
      cut: harness.output('uint32', 4),
      summary: harness.output('uint32', 8),
      exactCost: harness.output('uint32x2', 1),
      status
    };
    const parameters = harness.input('uint32', Uint32Array.of(0, 3, 0xffffffff));
    const result = await harness.run(
      new GPUNetworkFlow({
        nodeCount: 4,
        edges: harness.input('uint32x2', Uint32Array.of(0, 1, 1, 3, 0, 2, 2, 3)),
        capacities: harness.input('uint32', Uint32Array.of(1, 1, 1, 1)),
        costs: harness.input('uint32', Uint32Array.of(1, 1, 1, 1)),
        maximumAugmentations,
        parameters,
        output
      }),
      first => {
        expect(first(status.invalidCount)).toEqual([0]);
        expect(first(status.iterationCount)).toEqual([maximumAugmentations]);
        expect(first(status.converged)).toEqual([Number(maximumAugmentations === 2)]);
        expect(first(status.iterationLimitReached)).toEqual([Number(maximumAugmentations < 2)]);
        expect(first(output.flow)).toEqual([
          Number(maximumAugmentations > 0),
          Number(maximumAugmentations > 0),
          Number(maximumAugmentations > 1),
          Number(maximumAugmentations > 1)
        ]);
        expect(first(output.exactCost)).toEqual([2 * maximumAugmentations, 0]);
        harness.rewrite(parameters, Uint32Array.of(0, 0, 0xffffffff));
      }
    );
    expect(result(status.invalidCount)).toEqual([1]);
    expect(result(status.iterationCount)).toEqual([0]);
    expect(result(status.converged)).toEqual([0]);
    expect(result(status.iterationLimitReached)).toEqual([0]);
    expect(result(output.flow)).toEqual([0, 0, 0, 0]);
    expect(result(output.exactCost)).toEqual([0, 0]);
  });
}

for (const maximumRelaxations of [0, 1, 2, 3]) {
  it(`time routing retains partial labels and counts ${maximumRelaxations} bounded sweeps`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createOptimizationHarness(device);
    const status = {
      invalidCount: harness.output('uint32', 1),
      iterationCount: harness.output('uint32', 1),
      converged: harness.output('uint32', 1),
      iterationLimitReached: harness.output('uint32', 1)
    };
    const output = {
      arrivals: harness.output('float32', 3),
      predecessorTransitions: harness.output('uint32', 3),
      summary: harness.output('uint32', 4),
      status
    };
    const departure = harness.input('float32', Float32Array.of(0));
    const result = await harness.run(
      new GPUNetworkTimeRouting({
        stateVertices: harness.input('uint32', Uint32Array.of(0, 1, 2)),
        transitions: harness.input('uint32x2', Uint32Array.of(1, 2, 0, 1)),
        costs: harness.input('float32x2', Float32Array.of(1, 0, 1, 0)),
        endpoints: harness.input('uint32', Uint32Array.of(0, 2)),
        maximumRelaxations,
        departure,
        output
      }),
      first => {
        expect(first(status.invalidCount)).toEqual([0]);
        expect(first(status.iterationCount)).toEqual([maximumRelaxations]);
        expect(first(status.converged)).toEqual([Number(maximumRelaxations === 3)]);
        expect(first(status.iterationLimitReached)).toEqual([Number(maximumRelaxations < 3)]);
        expect(first(output.arrivals)).toEqual([
          0,
          maximumRelaxations >= 1 ? 1 : Infinity,
          maximumRelaxations >= 2 ? 2 : Infinity
        ]);
        const summary = first(output.summary);
        expect(summary.slice(0, 3)).toEqual([
          1,
          Number(maximumRelaxations === 3),
          maximumRelaxations >= 2 ? 2 : 0xffffffff
        ]);
        expect(new Float32Array(Uint32Array.of(summary[3]).buffer)[0]).toBe(
          maximumRelaxations >= 2 ? 2 : Infinity
        );
        harness.rewrite(departure, Float32Array.of(-1));
      }
    );
    expect(result(status.invalidCount)).toEqual([1]);
    expect(result(status.iterationCount)).toEqual([0]);
    expect(result(status.converged)).toEqual([0]);
    expect(result(status.iterationLimitReached)).toEqual([0]);
    expect(result(output.arrivals)).toEqual([Infinity, Infinity, Infinity]);
    expect(result(output.predecessorTransitions)).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
  });
}

for (const kind of ['equilibrium', 'zero-budget', 'rounding-stall'] as const) {
  it(`rational access distinguishes ${kind} and resets its named ports`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createOptimizationHarness(device);
    const status = {
      invalidCount: harness.output('uint32', 1),
      iterationCount: harness.output('uint32', 1),
      converged: harness.output('uint32', 1),
      iterationLimitReached: harness.output('uint32', 1)
    };
    const output = {
      assignments: harness.output('float32', 2),
      scores: harness.output('float32', 1),
      unservedDemand: harness.output('float32', 1),
      summary: harness.output('uint32', 4),
      status
    };
    const supply = harness.input('float32', Float32Array.of(1, 1));
    const maximumCycles = kind === 'zero-budget' ? 0 : 2;
    const result = await harness.run(
      new GPUNetworkRationalAccess({
        costs: harness.input('float32', Float32Array.of(0, 0)),
        demand: harness.input('float32', Float32Array.of(kind === 'rounding-stall' ? 1 : 10)),
        supply,
        parameters: harness.input('float32', Float32Array.of(1, 1, 1, 1, 50, 0)),
        initialLimitCycles: 0,
        maximumCycles,
        output
      }),
      first => {
        expect(first(status.invalidCount)).toEqual([0]);
        expect(first(status.converged)).toEqual([Number(kind === 'equilibrium')]);
        expect(first(status.iterationLimitReached)).toEqual([Number(kind !== 'equilibrium')]);
        expect(first(status.iterationCount)).toEqual([kind === 'equilibrium' ? 1 : maximumCycles]);
        expect(first(output.assignments)).toEqual(
          kind === 'equilibrium' ? [5, 5] : kind === 'zero-budget' ? [10, 0] : [1, 0]
        );
        harness.rewrite(supply, Float32Array.of(0, 1));
      }
    );
    expect(result(status.invalidCount)).toEqual([1]);
    expect(result(status.converged)).toEqual([0]);
    expect(result(status.iterationLimitReached)).toEqual([0]);
    expect(result(status.iterationCount)).toEqual([0]);
  });
}

it('rational access retains a completed shift when the next congestion evaluation overflows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const originCount = 17;
  const facilityCount = 18;
  const costs = new Float32Array(originCount * facilityCount).fill(Infinity);
  for (let origin = 0; origin < originCount; origin++) {
    costs[origin * facilityCount + origin] = 0;
    costs[origin * facilityCount + 17] = 1;
  }
  const demand = harness.input('float32', new Float32Array(originCount).fill(1));
  const status = {
    invalidCount: harness.output('uint32', 1),
    iterationCount: harness.output('uint32', 1),
    converged: harness.output('uint32', 1),
    iterationLimitReached: harness.output('uint32', 1)
  };
  const output = {
    assignments: harness.output('float32', originCount * facilityCount),
    scores: harness.output('float32', originCount),
    unservedDemand: harness.output('float32', originCount),
    summary: harness.output('uint32', 4),
    status
  };
  // Each private facility initially costs 2^125 and the shared facility costs one.
  // Pairwise equalization shifts 1/2 per origin; the shared load 8.5 divided by
  // 2^-125 then exceeds the largest finite float32 on the next gap evaluation.
  const result = await harness.run(
    new GPUNetworkRationalAccess({
      costs: harness.input('float32', costs),
      demand,
      supply: harness.input('float32', new Float32Array(facilityCount).fill(2 ** -125)),
      parameters: harness.input('float32', Float32Array.of(1, 1, 1, 1, 50, 0)),
      demandMode: 'fractional',
      initialLimitCycles: 0,
      maximumCycles: 2,
      output
    }),
    first => {
      expect(first(status.invalidCount)).toEqual([1]);
      expect(first(status.iterationCount)).toEqual([1]);
      expect(first(status.converged)).toEqual([0]);
      expect(first(status.iterationLimitReached)).toEqual([0]);
      expect(first(output.summary).slice(0, 3)).toEqual([0, 0, 1]);
      const assignments = first(output.assignments);
      for (let origin = 0; origin < originCount; origin++) {
        for (let facility = 0; facility < facilityCount; facility++) {
          expect(assignments[origin * facilityCount + facility]).toBeCloseTo(
            facility === origin || facility === 17 ? 0.5 : 0,
            5
          );
        }
      }
      harness.rewrite(demand, new Float32Array(originCount).fill(-1));
    }
  );
  expect(result(status.invalidCount)).toEqual([1]);
  expect(result(status.iterationCount)).toEqual([0]);
  expect(result(status.converged)).toEqual([0]);
  expect(result(status.iterationLimitReached)).toEqual([0]);
  expect(result(output.summary).slice(0, 3)).toEqual([0, 0, 0]);
});

it('rejects an extra status field before shader compilation and still publishes the supported flow port', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const invalidCount = harness.output('uint32', 1);
  const extendedStatus = {invalidCount, overflow: harness.output('uint32', 1)};
  const props = {
    id: 'extra-flow',
    nodeCount: 2,
    edges: harness.input('uint32x2', Uint32Array.of(0, 1)),
    capacities: harness.input('uint32', Uint32Array.of(3)),
    parameters: harness.input('uint32', Uint32Array.of(0, 1, 5)),
    output: {
      flow: harness.output('uint32', 1),
      cut: harness.output('uint32', 2),
      summary: harness.output('uint32', 8),
      status: extendedStatus
    }
  };
  expect(() => new GPUNetworkFlow(props)).toThrow(
    /^extra-flow output.status.overflow must be a supported field/
  );
  const result = await harness.run(
    new GPUNetworkFlow({...props, output: {...props.output, status: {invalidCount}}})
  );
  expect(result(invalidCount)).toEqual([0]);
  expect(result(props.output.flow)).toEqual([3]);
  expect(result(props.output.summary).slice(0, 5)).toEqual([1, 1, 3, 2, 1]);
});
