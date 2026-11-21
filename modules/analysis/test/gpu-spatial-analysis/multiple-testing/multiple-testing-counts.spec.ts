// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUMultipleTestingAdjustment} from '../../../src/gpu-spatial-analysis/multiple-testing/gpu-multiple-testing-adjustment';
import {
  GPURasterFalseDiscoveryRateMask,
  getGPURasterFalseDiscoveryRateMaskParameterValues
} from '../../../src/gpu-spatial-analysis/multiple-testing/gpu-raster-false-discovery-rate-mask';
import {createRasterTimeSeriesHarness} from '../../gpu-raster/raster-time-series/raster-time-series-test-utils';
import {GPUPettittTest} from '../../../src/gpu-spatial-analysis/change-points/gpu-pettitt-test';
import {createPettittCompositionCountOutputs} from './pettitt-composition-fixture';

it('DGN-01 Pettitt to correction uses the actual finite family and preserves tested constants', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const control of [
    new Array<number>(12).fill(NaN),
    [7, ...new Array<number>(11).fill(NaN)],
    new Array<number>(12).fill(7)
  ]) {
    const constant = control.every(Number.isFinite);
    const harness = createRasterTimeSeriesHarness(device);
    const pValues = harness.output('p-values', 'float32', 2);
    const countOutputs = createPettittCompositionCountOutputs(name =>
      harness.output(name, 'uint32', 1)
    );
    const result = await harness.run([
      new GPUPettittTest({
        values: harness.floats([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, ...control]),
        cellCount: 2,
        sliceCount: 12,
        layout: 'cell-major',
        output: {
          pValues,
          statistic: harness.output('statistic', 'uint32', 2),
          changeSlice: harness.output('change-slice', 'uint32', 2)
        }
      }),
      new GPURasterFalseDiscoveryRateMask({
        cellCount: 2,
        pValues,
        method: 'bonferroni',
        parameters: harness.floats(
          getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: 0.05})
        ),
        output: {
          adjustedPValues: harness.output('adjusted', 'float32', 2),
          significant: harness.output('significant', 'uint32', 2),
          ...countOutputs
        }
      })
    ]);
    // Independent cross-split count K=36 and n=12 gives p=.03140780113025614.
    expect(result['statistic']).toEqual([36, 0]);
    expect(Number.isFinite(result['p-values'][0])).toBe(true);
    expect(result['p-values'][0]).toBeCloseTo(0.03140780113025614, 7);
    expect(Number.isFinite(result['adjusted'][0])).toBe(true);
    expect(result['adjusted'][0]).toBeCloseTo(
      constant ? 0.06281560226051228 : 0.03140780113025614,
      7
    );
    if (constant) {
      expect(result['p-values'][1]).toBe(1);
      expect(result['adjusted'][1]).toBe(1);
    } else {
      expect(result['p-values'][1]).toBeNaN();
      expect(result['adjusted'][1]).toBeNaN();
    }
    expect(result['significant']).toEqual([constant ? 0 : 1, 0]);
    const countResults = {
      finite: result['finite'],
      effective: result['effective'],
      tested: result['effective']
    };
    for (const port of ['finite', 'effective', 'tested'] as const)
      expect(countResults[port]).toEqual([constant ? 2 : 1]);
  }
});

it('CMP-14 independently publishes observed and padded counts on GPU and raster routes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [port, expected] of [
    ['finiteTestCount', 1],
    ['effectiveFamilySize', 10]
  ] as const) {
    for (const route of ['none', 'bonferroni', 'raster'] as const) {
      const harness = createRasterTimeSeriesHarness(device);
      const output = {
        adjustedPValues: harness.output('adjusted', 'float32', 2),
        [port]: harness.output('count', 'uint32', 1)
      };
      const pValues = harness.floats([0.02, NaN]);
      const producer =
        route === 'raster'
          ? new GPURasterFalseDiscoveryRateMask({
              cellCount: 2,
              pValues,
              minimumFamilySize: 10,
              method: 'bonferroni',
              parameters: harness.floats(
                getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: 0.05})
              ),
              output: {...output, significant: harness.output('significant', 'uint32', 2)}
            })
          : new GPUMultipleTestingAdjustment({
              pValues,
              minimumFamilySize: 10,
              method: route,
              output
            });
      const result = await harness.run(producer);
      // One observed test, ten in the effective family. Bonferroni is .02 * 10 = .20.
      expect(result['count']).toEqual([expected]);
      expect(result['adjusted']).toHaveLength(2);
      expect(Number.isFinite(result['adjusted'][0])).toBe(true);
      expect(result['adjusted'][0]).toBeCloseTo(route === 'none' ? 0.02 : 0.2, 6);
      expect(result['adjusted'][1]).toBeNaN();
    }
  }
});

it('CMP-14 raster finiteTestCount respects the per-cell validity mask', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const result = await harness.run(
    new GPURasterFalseDiscoveryRateMask({
      cellCount: 2,
      pValues: harness.floats([1 / 512, 0.03]),
      validity: harness.integers([7, 0]),
      minimumFamilySize: 10,
      method: 'bonferroni',
      parameters: harness.floats(getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: 0.05})),
      output: {
        significant: harness.output('significant', 'uint32', 2),
        finiteTestCount: harness.output('observed', 'uint32', 1),
        effectiveFamilySize: harness.output('family', 'uint32', 1)
      }
    })
  );
  expect(result['observed']).toEqual([1]);
  expect(result['family']).toEqual([10]);
  // 10/512 < .05, so the nonzero validity word must retain the first significant cell.
  expect(result['significant']).toEqual([1, 0]);
});

it('W3-01 preserves BY adjustment on both sides of the exact harmonic transition', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Rational H_m computed with Python fractions, multiplied by m / 2**16.
  for (const [familySize, expected] of [
    [63, 0.004545299559531608],
    [64, 0.004632705960650165],
    [65, 0.004720350780347824]
  ]) {
    const harness = createRasterTimeSeriesHarness(device);
    const result = await harness.run(
      new GPUMultipleTestingAdjustment({
        pValues: harness.floats([2 ** -16]),
        method: 'benjamini-yekutieli',
        minimumFamilySize: familySize,
        output: {adjustedPValues: harness.output('adjusted', 'float32', 1)}
      })
    );
    expect(result['adjusted']).toHaveLength(1);
    expect(Number.isFinite(result['adjusted'][0])).toBe(true);
    expect(Math.abs(result['adjusted'][0] - expected) / expected).toBeLessThan(2e-6);
  }
});

it('review B3 raster excludes negative subnormal words before counting and adjustment', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const result = await harness.run(
    new GPURasterFalseDiscoveryRateMask({
      cellCount: 2,
      method: 'bonferroni',
      pValues: harness.floats(new Float32Array(Uint32Array.of(0x80000001, 0x3e000000).buffer)),
      parameters: harness.floats(getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: 0.2})),
      output: {
        adjustedPValues: harness.output('adjusted', 'float32', 2),
        finiteTestCount: harness.output('observed', 'uint32', 1),
        effectiveFamilySize: harness.output('family', 'uint32', 1),
        significant: harness.output('significant', 'uint32', 2)
      }
    })
  );
  // One valid test at .125; the negative subnormal is outside [0,1] regardless of FTZ arithmetic.
  expect(result['observed']).toEqual([1]);
  expect(result['family']).toEqual([1]);
  expect(result['adjusted']).toHaveLength(2);
  expect(result['adjusted'][0]).toBeNaN();
  expect(Number.isFinite(result['adjusted'][1])).toBe(true);
  expect(result['adjusted'][1]).toBe(0.125);
  expect(result['significant']).toEqual([0, 1]);
});

it('review B2 publishes raw and padded test counts together', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const raster of [false, true]) {
    const harness = createRasterTimeSeriesHarness(device);
    const observed = harness.output('observed', 'uint32', 1);
    const family = harness.output('family', 'uint32', 1);
    const settings = {
      pValues: harness.floats([0.02, NaN]),
      minimumFamilySize: 10,
      method: 'bonferroni' as const,
      output: {
        adjustedPValues: harness.output('adjusted', 'float32', 2),
        finiteTestCount: observed,
        effectiveFamilySize: family
      }
    };
    const producer = raster
      ? new GPURasterFalseDiscoveryRateMask({
          ...settings,
          cellCount: 2,
          parameters: harness.floats(
            getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: 0.05})
          ),
          output: {...settings.output, significant: harness.output('significant', 'uint32', 2)}
        })
      : new GPUMultipleTestingAdjustment(settings);
    const result = await harness.run(producer);
    expect(result['observed']).toEqual([1]);
    expect(result['family']).toEqual([10]);
    expect(result['adjusted']).toHaveLength(2);
    expect(Number.isFinite(result['adjusted'][0])).toBe(true);
    expect(result['adjusted'][0]).toBeCloseTo(0.2, 6);
    expect(result['adjusted'][1]).toBeNaN();
  }
});
