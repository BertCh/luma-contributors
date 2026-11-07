// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUBoundedFacilityAllocation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-bounded-facility-allocation';
import {GPUNetworkFacilityOptimization} from '../../../src/gpu-network/network-optimization/gpu-network-facility-optimization';
import {createOptimizationHarness} from '../../gpu-network/network-optimization/optimization-test-harness';
import {requireTestDevice} from '../../utils/require-test-device';

it('cooperative p-center covers more than 4096 demand rows and resets its winner on re-encoding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const facilityCount = 10;
  const demandCount = 8193;
  const preferred = [3, 5, 7];
  const costs = new Float32Array(demandCount * facilityCount);
  for (let row = 0; row < demandCount; row++) {
    for (let facility = 0; facility < facilityCount; facility++) {
      costs[row * facilityCount + facility] = facility === preferred[row % 3] ? 0 : facility + 1;
    }
  }
  const costView = harness.input('float32', costs);
  const output = {
    selectedFacilities: harness.output('uint32', facilityCount),
    assignments: harness.output('uint32', demandCount),
    objective: harness.output('float32', 1),
    status: harness.output('uint32', 3)
  };
  const result = await harness.run(
    new GPUBoundedFacilityAllocation({
      operation: 'p-center',
      facilityCount,
      selectCount: 3,
      costs: costView,
      demand: harness.input('float32', new Float32Array(demandCount).fill(1)),
      capacities: harness.input('float32', new Float32Array(facilityCount)),
      output
    }),
    first => {
      expect(first(output.status)).toEqual([0, 1, 1]);
      expect(first(output.objective)).toEqual([0]);
      expect(first(output.selectedFacilities)).toEqual(
        Array.from({length: facilityCount}, (_, facility) => Number(preferred.includes(facility)))
      );
      expect(first(output.assignments)).toEqual(
        Array.from({length: demandCount}, (_, row) => preferred[row % 3])
      );
      // All subset scores tie; original subset rank and facility rank must win independently.
      harness.rewrite(costView, costs.fill(1));
    }
  );
  expect(result(output.selectedFacilities)).toEqual([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]);
  expect(result(output.assignments)).toEqual(new Array(demandCount).fill(0));
  expect(result(output.objective)).toEqual([1]);
  expect(result(output.status)).toEqual([0, 1, 1]);
});

it('hierarchical facility selection retains the greedy capacity objective and original subset ties', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  // 12 choose 2 = 66 candidates, crossing the first winner-reduction workgroup.
  const costs = new Float32Array(24).fill(Infinity);
  costs[0] = 1;
  costs[1] = 2;
  costs[12] = 1;
  costs[13] = 100;
  const output = {
    selectedFacilities: harness.output('uint32', 12),
    assignments: harness.output('uint32', 2),
    objective: harness.output('float32', 1),
    status: harness.output('uint32', 3)
  };
  const result = await harness.run(
    new GPUBoundedFacilityAllocation({
      operation: 'capacitated-p-median',
      facilityCount: 12,
      selectCount: 2,
      costs: harness.input('float32', costs),
      demand: harness.input('float32', Float32Array.of(1, 1)),
      capacities: harness.input('float32', new Float32Array(12).fill(1)),
      output
    })
  );
  expect(result(output.selectedFacilities)).toEqual([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  expect(result(output.assignments)).toEqual([0, 1]);
  expect(result(output.objective)).toEqual([101]);
  expect(result(output.status)).toEqual([0, 1, 0]);
});

it('network enumeration respects the exact prefix and finds a winner in a partial second batch', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const maximumCandidates of [0, 65536, 65538]) {
    const harness = createOptimizationHarness(device);
    const costs = new Float32Array(36).fill(1);
    for (let row = 0; row < 8; row++) costs[row * 4] = 0;
    costs[32] = 100;
    costs[33] = 0;
    // Candidate 65537 ties 65536: demand row zero may choose either facility at zero cost.
    costs[1] = 0;
    const costView = harness.input('float32', costs);
    const output = {
      selectedFacilities: harness.output('uint32', 4),
      assignments: harness.output('uint32', 9),
      objective: harness.output('float32', 1),
      status: harness.output('uint32', 4)
    };
    const result = await harness.run(
      new GPUNetworkFacilityOptimization({
        operation: 'capacitated-p-median',
        facilityCount: 4,
        selectCount: 2,
        maximumCandidates,
        costs: costView,
        demand: harness.input('float32', new Float32Array(9).fill(1)),
        capacities: harness.input('float32', new Float32Array(4).fill(9)),
        threshold: harness.input('float32', Float32Array.of(0)),
        output
      }),
      maximumCandidates === 65538
        ? first => {
            expect(first(output.assignments)).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 1]);
            expect(first(output.objective)).toEqual([0]);
            expect(first(output.status)).toEqual([1, 1, 0, maximumCandidates]);
            harness.rewrite(costView, costs.fill(0));
          }
        : undefined
    );
    expect(result(output.status)).toEqual([1, Number(maximumCandidates > 0), 0, maximumCandidates]);
    if (maximumCandidates === 0) {
      expect(result(output.objective)[0]).toBeNaN();
      expect(result(output.assignments)).toEqual(new Array(9).fill(0xffffffff));
      expect(result(output.selectedFacilities)).toEqual([0, 0, 0, 0]);
    } else {
      expect(result(output.assignments)).toEqual(new Array(9).fill(0));
      expect(result(output.selectedFacilities)).toEqual([1, 1, 0, 0]);
      expect(result(output.objective)).toEqual([maximumCandidates === 65536 ? 100 : 0]);
    }
  }
});

it('network winner reductions keep signed-zero ties and invalidate any feasible overflowing candidate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const operation of ['p-dispersion', 'capacitated-p-median'] as const) {
    const harness = createOptimizationHarness(device);
    const dispersion = operation === 'p-dispersion';
    const output = {
      selectedFacilities: harness.output('uint32', 3),
      assignments: harness.output('uint32', 2),
      objective: harness.output('float32', 1),
      status: harness.output('uint32', 4)
    };
    const result = await harness.run(
      new GPUNetworkFacilityOptimization({
        operation,
        facilityCount: 3,
        selectCount: 2,
        costs: harness.input(
          'float32',
          dispersion
            ? Float32Array.of(0, -0, 0, -0, 0, 0, 0, 0, 0)
            : Float32Array.of(1, 3e38, 1, 1, 3e38, 1)
        ),
        demand: harness.input('float32', Float32Array.of(1, 1)),
        capacities: harness.input('float32', Float32Array.of(2, 2, 2)),
        threshold: harness.input('float32', Float32Array.of(0)),
        output
      })
    );
    if (dispersion) {
      expect(result(output.selectedFacilities)).toEqual([1, 1, 0]);
      expect(result(output.objective)[0]).toBeCloseTo(0);
      expect(result(output.status)).toEqual([1, 1, 1, 8]);
    } else {
      // Candidate zero has finite cost 2, but candidate four is feasible and its sum overflows.
      expect(result(output.selectedFacilities)).toEqual([0, 0, 0]);
      expect(result(output.objective)[0]).toBeNaN();
      expect(result(output.status)).toEqual([0, 0, 0, 0]);
    }
  }
});
