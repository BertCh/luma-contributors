// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  getGPURasterFalseDiscoveryRateMaskParameterValues,
  GPURasterFalseDiscoveryRateMask,
  type GPURasterFalseDiscoveryRateMethod
} from '../../../src/gpu-spatial-analysis/multiple-testing/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRasterTimeSeriesHarness} from '../../gpu-raster/raster-time-series/raster-time-series-test-utils';
import {MULTIPLE_TESTING_FIXTURES} from './multiple-testing-fixtures';

async function getDevice() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

type MaskRun = {
  significant: number[];
  adjusted: number[];
  testedCount: number;
  significantCount: number;
  criticalPValue: number;
  histogram: number[];
};

async function runMask(
  pValues: ArrayLike<number>,
  options: {
    method?: GPURasterFalseDiscoveryRateMethod;
    alpha?: number;
    validity?: ArrayLike<number>;
    minimumFamilySize?: number;
    binCount?: number;
  } = {}
): Promise<MaskRun> {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const cellCount = pValues.length;
  const binCount = options.binCount ?? 10;
  const result = await harness.run(
    new GPURasterFalseDiscoveryRateMask({
      cellCount,
      pValues: harness.floats(pValues),
      validity: options.validity ? harness.integers(options.validity) : undefined,
      method: options.method,
      minimumFamilySize: options.minimumFamilySize,
      parameters: harness.floats(
        getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: options.alpha ?? 0.05})
      ),
      output: {
        significant: harness.output('significant', 'uint32', cellCount),
        adjustedPValues: harness.output('adjusted', 'float32', cellCount),
        testedCount: harness.output('testedCount', 'uint32', 1),
        significantCount: harness.output('significantCount', 'uint32', 1),
        criticalPValue: harness.output('criticalPValue', 'float32', 1),
        histogram: harness.output('histogram', 'uint32', binCount)
      }
    })
  );
  return {
    significant: result['significant'],
    adjusted: result['adjusted'],
    testedCount: result['testedCount'][0],
    significantCount: result['significantCount'][0],
    criticalPValue: result['criticalPValue'][0],
    histogram: result['histogram']
  };
}

/**
 * Independent binary64 decision rules written from the textbook definitions, not from the adjusted
 * p-value formulas of the code under test. Returns a 0/1 mask over `pValues`; NaN is untested.
 */
function decide(
  pValues: ArrayLike<number>,
  method: GPURasterFalseDiscoveryRateMethod,
  alpha: number,
  minimumFamilySize = 0
): number[] {
  const tested = Array.from(pValues, (p, row) => ({p, row})).filter(
    ({p}) => Number.isFinite(p) && p >= 0 && p <= 1
  );
  const m = Math.max(tested.length, minimumFamilySize);
  tested.sort((a, b) => a.p - b.p);
  const mask = new Array<number>(pValues.length).fill(0);
  let rejectedCount = 0;
  if (method === 'bonferroni') {
    tested.forEach(({p, row}) => {
      mask[row] = p <= alpha / m ? 1 : 0;
    });
    return mask;
  }
  if (method === 'holm') {
    // Step-down: reject while p_(k) <= alpha / (m - k + 1), stop at the first failure.
    while (
      rejectedCount < tested.length &&
      tested[rejectedCount].p <= alpha / (m - rejectedCount)
    ) {
      rejectedCount++;
    }
  } else {
    // Step-up: the largest k with p_(k) <= k alpha / (m c(m)); everything up to k is rejected.
    let harmonic = 1;
    if (method === 'benjamini-yekutieli') {
      harmonic = 0;
      for (let i = 1; i <= m; i++) harmonic += 1 / i;
    }
    for (let k = tested.length; k >= 1; k--) {
      if (tested[k - 1].p <= (k * alpha) / (m * harmonic)) {
        rejectedCount = k;
        break;
      }
    }
  }
  for (let rank = 0; rank < rejectedCount; rank++) mask[tested[rank].row] = 1;
  return mask;
}

/** The 15 p-values of Benjamini and Hochberg (1995), in a scrambled order. */
const BENJAMINI_HOCHBERG_1995 = [
  0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.324, 0.4262, 0.5719,
  0.6528, 0.759, 1.0
];
const SCRAMBLE = [9, 3, 14, 0, 11, 6, 1, 13, 4, 8, 2, 12, 5, 10, 7];

it('RTS-01 reproduces the Benjamini-Hochberg 1995 worked example (4 rejections at q = 0.05)', async () => {
  const pValues = SCRAMBLE.map(index => BENJAMINI_HOCHBERG_1995[index]);
  const result = await runMask(pValues, {method: 'benjamini-hochberg', alpha: 0.05});
  // Hand values: p_(k) m / k = 0.0015, 0.003, 0.0095, 0.035625, 0.0603, ... so the first four pass.
  expect(result.significant).toEqual(SCRAMBLE.map(index => (index < 4 ? 1 : 0)));
  expect(result.significantCount).toBe(4);
  expect(result.testedCount).toBe(15);
  expect(result.criticalPValue).toBeCloseTo(0.0095, 7);
  const adjustedByIndex = new Map(SCRAMBLE.map((index, row) => [index, result.adjusted[row]]));
  expect(adjustedByIndex.get(0)).toBeCloseTo(0.0015, 6);
  expect(adjustedByIndex.get(3)).toBeCloseTo(0.035625, 6);
  expect(result.histogram.reduce((sum, count) => sum + count, 0)).toBe(15);
});

it('RTS-01 Bonferroni and Holm reject 3 of the 15 and the level is a per-frame parameter', async () => {
  const pValues = SCRAMBLE.map(index => BENJAMINI_HOCHBERG_1995[index]);
  for (const method of ['bonferroni', 'holm'] as const) {
    const result = await runMask(pValues, {method, alpha: 0.05});
    expect(result.significantCount, method).toBe(3);
    expect(result.criticalPValue, method).toBeCloseTo(0.0019, 7);
  }
  const strict = await runMask(pValues, {method: 'benjamini-hochberg', alpha: 0.01});
  expect(strict.significantCount).toBe(3);
  const none = await runMask(pValues, {method: 'benjamini-hochberg', alpha: 0.0001});
  expect(none.significantCount).toBe(0);
  expect(none.criticalPValue).toBeNaN();
});

for (const fixture of MULTIPLE_TESTING_FIXTURES) {
  for (const method of [
    'bonferroni',
    'holm',
    'benjamini-hochberg',
    'benjamini-yekutieli'
  ] as const) {
    it(`RTS-01 ${method} mask equals statsmodels reject on ${fixture.name}`, async () => {
      const alpha = 0.1;
      const result = await runMask(fixture.pValues, {
        method,
        alpha,
        minimumFamilySize: fixture.minimumFamilySize,
        binCount: 4
      });
      // statsmodels reject is adjusted <= alpha; skip cells within float32 rounding of the level.
      const expected = fixture.expected[method];
      fixture.pValues.forEach((p, row) => {
        const adjusted = expected[row];
        if (Number.isNaN(adjusted)) {
          expect(result.significant[row], `row ${row} untested`).toBe(0);
        } else if (Math.abs(adjusted - alpha) > 1e-5) {
          expect(result.significant[row], `row ${row} p ${p}`).toBe(adjusted <= alpha ? 1 : 0);
        }
      });
    });
  }
}

it('RTS-01 masks agree with the textbook step-up and step-down rules on a mixed raster', async () => {
  let state = 20251109;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const cellCount = 4000;
  // 15 percent of cells carry signal (small p), the rest are uniform; some cells are NaN.
  const pValues = Float32Array.from({length: cellCount}, (_, row) =>
    row % 53 === 0 ? Number.NaN : random() < 0.15 ? random() ** 6 : random()
  );
  for (const method of [
    'benjamini-hochberg',
    'benjamini-yekutieli',
    'holm',
    'bonferroni'
  ] as const) {
    const result = await runMask(pValues, {method, alpha: 0.05, binCount: 20});
    const expected = decide(pValues, method, 0.05);
    // float32 rounds adjusted p-values; accept disagreement only on cells whose own decision margin is tiny.
    let disagreements = 0;
    for (let row = 0; row < cellCount; row++) {
      if (result.significant[row] !== expected[row]) {
        disagreements++;
        expect(Math.abs(result.adjusted[row] - 0.05), `${method} row ${row}`).toBeLessThan(1e-5);
      }
    }
    expect(disagreements, method).toBeLessThanOrEqual(2);
    expect(result.significantCount, method).toBe(
      result.significant.reduce((sum, flag) => sum + flag, 0)
    );
    expect(result.testedCount).toBe(pValues.filter(Number.isFinite).length);
  }
});

it('RTS-01 zero validity removes a cell from the family and the histogram', async () => {
  const pValues = [0.001, 0.002, 0.03, 0.04, 0.5, 0.6, 0.0005, 0.9];
  const validity = [1, 1, 1, 1, 1, 1, 0, 0];
  const masked = await runMask(pValues, {validity, alpha: 0.05, binCount: 10});
  const reference = await runMask(pValues.slice(0, 6), {alpha: 0.05, binCount: 10});
  expect(masked.testedCount).toBe(6);
  expect(masked.significant.slice(0, 6)).toEqual(reference.significant);
  expect(masked.significant.slice(6)).toEqual([0, 0]);
  expect(masked.adjusted[6]).toBeNaN();
  expect(masked.histogram).toEqual(reference.histogram);
  // Bins of width 0.1: three values below 0.1, 0.5 and 0.6 in their own bins.
  expect(masked.histogram).toEqual([4, 0, 0, 0, 0, 1, 1, 0, 0, 0]);
});

it('RTS-01 keeps a cell whose adjusted p-value equals alpha exactly (reject is adjusted <= alpha)', async () => {
  // Dyadic values are exact in float32: 8 * 0.0625 = 0.5.
  const result = await runMask([0.0625, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75], {
    method: 'bonferroni',
    alpha: 0.5
  });
  expect(result.adjusted[0]).toBe(0.5);
  expect(result.significant[0]).toBe(1);
  expect(result.significantCount).toBe(1);
});

it('RTS-01 rejects an out of range level and an unknown method at declaration', () => {
  expect(() => getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: 0})).toThrow(
    /alpha must be a finite number/
  );
  expect(() => getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: 1.5})).toThrow(
    /alpha must be a finite number/
  );
});
