// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkFacilityOptimization} from '../../../src/gpu-network/network-optimization/gpu-network-facility-optimization';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOptimizationHarness} from './optimization-test-harness';

// The four assignment codes are [0,0], [1,0], [0,1], [1,1]. Only [1,0] is feasible:
// the inflexible second demand needs facility 0, so the flexible first demand uses facility 1.
for (const maximumCandidates of [0, 1, 2, 4]) {
  it(`reports exact search completion independently of feasibility at ${maximumCandidates} candidates`, async () => {
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
      selectedFacilities: harness.output('uint32', 2),
      assignments: harness.output('uint32', 2),
      objective: harness.output('float32', 1),
      summary: harness.output('uint32', 4),
      status
    };
    const result = await harness.run(
      new GPUNetworkFacilityOptimization({
        operation: 'capacitated-p-median',
        facilityCount: 2,
        selectCount: 2,
        maximumCandidates,
        costs: harness.input('float32', Float32Array.of(1, 2, 1, Infinity)),
        demand: harness.input('float32', Float32Array.of(1, 1)),
        capacities: harness.input('float32', Float32Array.of(1, 1)),
        threshold: harness.input('float32', Float32Array.of(0)),
        output
      })
    );
    const haveIncumbent = maximumCandidates >= 2;
    expect(result(output.summary)).toEqual([
      1,
      Number(haveIncumbent),
      Number(maximumCandidates === 4),
      maximumCandidates
    ]);
    expect(result(status.invalidCount)).toEqual([0]);
    expect(result(status.iterationCount)).toEqual([Number(maximumCandidates > 0)]);
    expect(result(status.converged)).toEqual([Number(maximumCandidates === 4)]);
    expect(result(status.iterationLimitReached)).toEqual([Number(maximumCandidates < 4)]);
    expect(result(output.assignments)).toEqual(haveIncumbent ? [1, 0] : [0xffffffff, 0xffffffff]);
    expect(result(output.selectedFacilities)).toEqual(haveIncumbent ? [1, 1] : [0, 0]);
    if (haveIncumbent) expect(result(output.objective)).toEqual([3]);
    else expect(result(output.objective)[0]).toBeNaN();
  });
}

it('clears every named port on valid and invalid re-encodings', async () => {
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
    selectedFacilities: harness.output('uint32', 2),
    assignments: harness.output('uint32', 2),
    objective: harness.output('float32', 1),
    summary: harness.output('uint32', 4),
    status
  };
  const demand = harness.input('float32', Float32Array.of(1, 1));
  const contributor = new GPUNetworkFacilityOptimization({
    operation: 'capacitated-p-median',
    facilityCount: 2,
    selectCount: 2,
    costs: harness.input('float32', Float32Array.of(1, 2, 1, Infinity)),
    demand,
    capacities: harness.input('float32', Float32Array.of(1, 1)),
    threshold: harness.input('float32', Float32Array.of(0)),
    output
  });
  const result = await harness.run(contributor, first => {
    expect(first(output.summary)).toEqual([1, 1, 1, 4]);
    expect(first(status.invalidCount)).toEqual([0]);
    expect(first(status.converged)).toEqual([1]);
    expect(first(status.iterationCount)).toEqual([1]);
    harness.rewrite(demand, Float32Array.of(-1, 1));
  });
  expect(result(output.summary)).toEqual([0, 0, 0, 0]);
  expect(result(status.invalidCount)).toEqual([1]);
  expect(result(status.converged)).toEqual([0]);
  expect(result(status.iterationLimitReached)).toEqual([0]);
  expect(result(status.iterationCount)).toEqual([0]);
  expect(result(output.assignments)).toEqual([0xffffffff, 0xffffffff]);
  expect(result(output.selectedFacilities)).toEqual([0, 0]);
  expect(result(output.objective)[0]).toBeNaN();
});

for (const kind of ['infeasible', 'empty', 'overflow'] as const) {
  it(`distinguishes a ${kind} problem from a budget exit`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createOptimizationHarness(device);
    const demandCount = kind === 'empty' ? 0 : 1;
    const status = {
      invalidCount: harness.output('uint32', 1),
      iterationCount: harness.output('uint32', 1),
      converged: harness.output('uint32', 1),
      iterationLimitReached: harness.output('uint32', 1)
    };
    const output = {
      selectedFacilities: harness.output('uint32', 1),
      assignments: harness.output('uint32', demandCount),
      objective: harness.output('float32', 1),
      summary: harness.output('uint32', 4),
      status
    };
    const result = await harness.run(
      new GPUNetworkFacilityOptimization({
        operation: 'capacitated-p-median',
        facilityCount: 1,
        selectCount: 1,
        costs: harness.input(
          'float32',
          Float32Array.from(kind === 'empty' ? [] : [kind === 'overflow' ? 3e38 : Infinity])
        ),
        demand: harness.input('float32', Float32Array.from(kind === 'empty' ? [] : [2])),
        capacities: harness.input('float32', Float32Array.of(2)),
        threshold: harness.input('float32', Float32Array.of(0)),
        output
      })
    );
    expect(result(status.invalidCount)).toEqual([Number(kind === 'overflow')]);
    expect(result(status.converged)).toEqual([Number(kind !== 'overflow')]);
    expect(result(status.iterationLimitReached)).toEqual([0]);
    expect(result(status.iterationCount)).toEqual([1]);
    if (kind === 'empty') {
      expect(result(output.selectedFacilities)).toEqual([1]);
      expect(result(output.assignments)).toEqual([]);
      expect(result(output.objective)).toEqual([0]);
    } else {
      expect(result(output.selectedFacilities)).toEqual([0]);
      expect(result(output.objective)[0]).toBeNaN();
    }
  });
}

it('retains two executed batches after objective overflow and resets the counter when inputs are rejected', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const demand = harness.input('float32', new Float32Array(9).fill(2));
  const status = {
    invalidCount: harness.output('uint32', 1),
    converged: harness.output('uint32', 1),
    iterationCount: harness.output('uint32', 1),
    iterationLimitReached: harness.output('uint32', 1)
  };
  const output = {
    selectedFacilities: harness.output('uint32', 4),
    assignments: harness.output('uint32', 9),
    objective: harness.output('float32', 1),
    summary: harness.output('uint32', 4),
    status
  };
  const result = await harness.run(
    new GPUNetworkFacilityOptimization({
      operation: 'capacitated-p-median',
      facilityCount: 4,
      selectCount: 4,
      maximumCandidates: 65537,
      costs: harness.input('float32', new Float32Array(36).fill(3e38)),
      demand,
      capacities: harness.input('float32', new Float32Array(4).fill(18)),
      threshold: harness.input('float32', Float32Array.of(0)),
      output
    }),
    first => {
      // All assignments have an overflowing objective. The partial second batch still scores one
      // candidate; padding slots do not add iterations. A rejected result is not zero executed work.
      expect(first(status.invalidCount)).toEqual([1]);
      expect(first(status.iterationCount)).toEqual([2]);
      expect(first(status.converged)).toEqual([0]);
      expect(first(status.iterationLimitReached)).toEqual([0]);
      expect(first(output.summary)).toEqual([0, 0, 0, 0]);
      expect(first(output.objective)[0]).toBeNaN();
      harness.rewrite(demand, new Float32Array(9).fill(-1));
    }
  );
  expect(result(status.invalidCount)).toEqual([1]);
  expect(result(status.iterationCount)).toEqual([0]);
  expect(result(status.converged)).toEqual([0]);
  expect(result(status.iterationLimitReached)).toEqual([0]);
});
