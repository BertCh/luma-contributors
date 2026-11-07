// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  buildCoverageTrancheOneScenario,
  runCoverageTrancheOne
} from '../../../src/computation-lenses/scenarios/gpu-raster/coverage-tranche-one-runner';
import {COVERAGE_TRANCHE_ONE_SCENARIOS} from '../../../src/computation-lenses/scenarios/gpu-raster/coverage-tranche-one.scenarios';
import {submitGraph} from '@luma-contributors/analysis/testing';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {createCoverageTrancheOneHarness} from './coverage-tranche-one-harness';

it.each(
  COVERAGE_TRANCHE_ONE_SCENARIOS
)('$element executable example has independently checkable numerical outputs', async ({
  element
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCoverageTrancheOneHarness(device, element);
  try {
    await buildCoverageTrancheOneScenario(harness.session, element);
    harness.upload();
    const compiled = harness.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const read = (name: string) => harness.read(compiled, name);
      if (element === 'GPURasterFractionalZonalDistribution') {
        expect(await read('validity')).toEqual([1, 1]);
        expect(await read('status')).toEqual([0, 0]);
        expect(await read('weightSums')).toEqual([4.25, 2.5]);
        expect(await read('coveredArea')).toEqual([425, 250]);
        const means = await read('means');
        expect(means[0]).toBeCloseTo(40 / 4.25, 5);
        expect(means[1]).toBeCloseTo(18 / 2.5, 5);
        expect(await read('quantileValidity')).toEqual([1, 1, 1, 1]);
      } else if (element === 'GPURasterFractionalCategoricalStatistics') {
        expect(await read('coverageSums')).toEqual([1.5, 1.5, 0.25, 0.5, 1, 1]);
        expect(await read('coveredArea')).toEqual([425, 250]);
        expect(await read('missingArea')).toEqual([100, 0]);
        expect(await read('majority')).toEqual([1, 2]);
        expect(await read('minority')).toEqual([2, 0]);
        expect(await read('validity')).toEqual([1, 1]);
      } else if (element === 'GPURasterFocalPairs') {
        expect(await read('pairedCount')).toEqual([4, 6, 4, 6, 9, 6, 4, 6, 4]);
        expect(await read('focalStatus')).toEqual(new Array(9).fill(0));
        for (const [name, expected] of [
          ['slope', 2],
          ['intercept', 3],
          ['correlation', 1],
          ['rSquared', 1],
          ['residualSumSquares', 0]
        ] as const) {
          for (const actual of await read(name)) expect(actual, name).toBeCloseTo(expected, 5);
        }
      } else if (element === 'addDasymetricRedistributionRecipe') {
        expect(await read('complete')).toEqual([1]);
        expect(await read('invalidTupleCount')).toEqual([0]);
        expect(await read('sourceUnallocatedTotals')).toEqual([0, 0]);
        expect(await read('sourceAllocatedTotals')).toEqual([120, 90]);
        const totals = await read('targetTotals');
        expect(totals[0]).toBeCloseTo((120 * 2.75) / 4.25 + 90 / 3, 4);
        expect(totals.reduce((sum, value) => sum + value, 0)).toBeCloseTo(210, 4);
      } else {
        expect(await read('indexValidity')).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 0]);
        const values = await read('index');
        const expected =
          element === 'GPURasterNDWI'
            ? -5 / 7
            : element === 'GPURasterNBR'
              ? 5 / 7
              : element === 'GPURasterEVI'
                ? 1.25 / 1.825
                : 0.75 / 1.2;
        expect(values[0]).toBeCloseTo(expected, 6);
        expect(values[8]).toBeNaN();
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    harness.destroy();
  }
});

it.each(
  COVERAGE_TRANCHE_ONE_SCENARIOS
)('$element runs the actual lens record and capture validation', async ({
  element,
  slug,
  domain
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const elementId = domain === 'gpu-raster' ? `raster-${slug}` : 'dasymetric-redistribution-recipe';
  const result = await runCoverageTrancheOne(device, element, `${domain}.${elementId}`);
  expect(result.captures?.snapshots.length).toBeGreaterThan(0);
  expect(result.captures?.runId).toBe(result.runId);
  if (element === 'addDasymetricRedistributionRecipe') {
    expect(
      result.captures?.snapshots.some(snapshot => snapshot.name === 'sourceAllocatedTotals')
    ).toBe(true);
    expect(
      result.captures?.snapshots.some(snapshot => snapshot.name === 'eligibleCellWeights')
    ).toBe(true);
  }
});
