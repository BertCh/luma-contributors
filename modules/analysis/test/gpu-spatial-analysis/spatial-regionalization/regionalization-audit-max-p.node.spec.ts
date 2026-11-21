// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeMaxPRegionsOnCPU,
  CPUMaxPRegionsOperation
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  createMaxPGrowthInput,
  createMaxPRefinementInput,
  getRegionalizationOperationInputs
} from './regionalization-operation-test-utils';
import {
  createAuditMaxPRemainderInput,
  evaluateAuditMaxPRemainder
} from './regionalization-audit-max-p-fixtures';
import {
  createAuditMaxPInput,
  createAuditMaxPProps,
  createAuditMaxPSingletons
} from './regionalization-audit-max-p-fixtures';

it('counts budget fallback singletons that already meet the floor as feasible', () => {
  // Closed form: eight islands of extensive mass one at floor one force eight feasible
  // singleton regions, independent of seed waves or their iteration budget. Each SSD is zero.
  const result = computeMaxPRegionsOnCPU(createAuditMaxPSingletons(), {
    standardize: false,
    startCount: 1,
    maximumGrowthSteps: 1
  });
  expect(Array.from(result.labels)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  expect(Array.from(result.objective)).toEqual([0]);
  expect(Array.from(result.status)).toEqual([8, 1, 0, 0, 0, 0]);
  const input = createAuditMaxPSingletons();
  input.values[0] = NaN;
  const invalid = computeMaxPRegionsOnCPU(input, {
    standardize: false,
    startCount: 1,
    maximumGrowthSteps: 1
  });
  expect(Array.from(invalid.labels)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  expect(Array.from(invalid.objective)).toEqual([0]);
  expect(Array.from(invalid.status)).toEqual([0, 0, 8, 1, 0, 0]);
});

it('audits Max-P objective, dense labels and infeasibility against enumerated NumPy partitions', () => {
  for (const island of [false, true]) {
    const input = createAuditMaxPInput(island);
    for (const maximumIterations of [0, 4]) {
      const result = computeMaxPRegionsOnCPU(input, {
        ...createAuditMaxPProps(input.values.length),
        maximumIterations
      });
      expect(Array.from(result.labels)).toEqual([0, 0, 1, 1, 2, 2, 3, 3, ...(island ? [4] : [])]);
      expect(Array.from(result.objective)).toEqual([28]);
      expect(Array.from(result.status)).toEqual(island ? [5, 0, 1, 0, 0, 0] : [4, 1, 0, 0, 0, 0]);
    }
  }
});

it('preserves Max-P population-standardized affine invariance and raw scale equivariance', () => {
  for (const [scale, shift, standardize, expected] of [
    [1, 64, false, 28],
    [3, 0, false, 252],
    [1, 0, true, 0.25],
    [3, 64, true, 0.25]
  ] as const) {
    const input = createAuditMaxPInput();
    input.values = input.values.map(value => value * scale + shift);
    const result = computeMaxPRegionsOnCPU(input, {...createAuditMaxPProps(), standardize});
    expect(Array.from(result.labels)).toEqual([0, 0, 1, 1, 2, 2, 3, 3]);
    expect(Array.from(result.status)).toEqual([4, 1, 0, 0, 0, 0]);
    expect(result.objective[0]).toBe(expected);
  }
});

it('retains nonconstant Max-P columns and active refinement at extreme finite scales', () => {
  for (const exponent of [-80, 64, 120]) {
    const pairs = createAuditMaxPInput();
    pairs.values = pairs.values.map(value => value * 2 ** exponent);
    const forced = computeMaxPRegionsOnCPU(pairs, {...createAuditMaxPProps(), standardize: true});
    expect(Array.from(forced.labels)).toEqual([0, 0, 1, 1, 2, 2, 3, 3]);
    expect(Array.from(forced.status)).toEqual([4, 1, 0, 0, 0, 0]);
    expect(Array.from(forced.objective)).toEqual([0.25]);

    const input = createMaxPRefinementInput(2);
    input.values = input.values.map(value => value * 2 ** exponent);
    const refined = computeMaxPRegionsOnCPU(input, {
      standardize: true,
      startCount: 1,
      maximumGrowthSteps: 64,
      maximumIterations: 8
    });
    // NumPy direct SSD / population variance: 174 / (2000/81) = 7.047.
    expect(Array.from(refined.labels)).toEqual([0, 1, 1, 0, 1, 2, 0, 2, 2]);
    expect(Array.from(refined.status)).toEqual([3, 1, 0, 0, 0, 1]);
    expect(Array.from(refined.objective)).toEqual([Math.fround(7.047)]);
  }
});

it('runs active tiny-scale Max-P refinement through the CPU graph operation with oracle ports', async () => {
  // Only CPU stages execute; the null device does not provide GPU numerical evidence.
  const device = createNullWebGPUDevice();
  try {
    const input = createMaxPRefinementInput(2);
    input.values = input.values.map(value => value * 2 ** -80);
    const result = await runOperationGraph(device, {
      inputs: getRegionalizationOperationInputs(input),
      operations: [
        new CPUMaxPRegionsOperation({
          ...createAuditMaxPProps(9),
          standardize: true,
          startCount: 1,
          maximumIterations: 8
        })
      ],
      outputs: ['labels', 'objective', 'status']
    });
    expect(Array.from(result['labels'])).toEqual([0, 1, 1, 0, 1, 2, 0, 2, 2]);
    expect(Array.from(result['status'])).toEqual([3, 1, 0, 0, 0, 1]);
    expect(Array.from(result['objective'])).toEqual([Math.fround(7.047)]);
  } finally {
    device.destroy();
  }
});

it('suppresses a Max-P partition when the supplied adjacency reports lost edges', () => {
  const input = {...createAuditMaxPInput(), status: {candidateOverflow: Uint32Array.of(1)}};
  const result = computeMaxPRegionsOnCPU(input, createAuditMaxPProps());
  expect(Array.from(result.labels)).toEqual(new Array(8).fill(4294967295));
  expect(Number.isNaN(result.objective[0])).toBe(true);
  expect(Array.from(result.status)).toEqual([0, 0, 0, 0, 0, 0]);
});

it('publishes a nonzero selected start from an independent integer Philox and clique-score oracle', () => {
  const input = {...createMaxPGrowthInput(), parameters: Uint32Array.of(2)};
  const result = computeMaxPRegionsOnCPU(input, {
    columnCount: 2,
    standardize: false,
    startCount: 8
  });
  // Python integer Philox keys give roots [0,7] at start zero (one outlier) and [0,4] at
  // start one (both core). Growth SSD is zero for start one, final SSD is 75 per clique.
  expect(Array.from(result.labels)).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
  expect(Array.from(result.objective)).toEqual([150]);
  expect(Array.from(result.status)).toEqual([2, 1, 0, 0, 1, 0]);
});

it('keeps budget-limited enclave remainders contiguous and evaluates their actual floor and SSD', () => {
  const result = computeMaxPRegionsOnCPU(createAuditMaxPRemainderInput(), {
    standardize: false,
    startCount: 1,
    maximumGrowthSteps: 4
  });
  const oracle = evaluateAuditMaxPRemainder(result.labels);
  expect(oracle.connected).toBe(true);
  expect(result.labels[1]).not.toBe(result.labels[6]);
  expect(Array.from(new Set(result.labels)).sort((first, second) => first - second)).toEqual(
    Array.from({length: oracle.regionCount}, (_, row) => row)
  );
  expect(Array.from(result.status)).toEqual([
    oracle.regionCount,
    0,
    oracle.infeasibleRegionCount,
    0,
    0,
    0
  ]);
  expect(oracle.infeasibleRegionCount).toBeGreaterThan(0);
  expect(result.objective[0]).toBe(Math.fround(oracle.objective));
});
