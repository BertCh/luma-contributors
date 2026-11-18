// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {readFileSync} from 'node:fs';
import {Worker} from 'node:worker_threads';
import typescript from 'typescript';
import {
  getGeographicalDetectorPValue,
  readGeographicalDetectorPValue,
  getNonCentralFUpperTail
} from '../../../src/gpu-spatial-analysis/geographical-detector/geographical-detector-p-value';
import {GPU_GEOGRAPHICAL_DETECTOR_LAYOUT} from '../../../src/gpu-spatial-analysis/geographical-detector/gpu-geographical-detector';
import fixture from '../../oracles/fixtures/stx08-spatial-methods.json';

it('reports optional p-value availability without suppressing unrelated summary read errors', () => {
  const layout = GPU_GEOGRAPHICAL_DETECTOR_LAYOUT;
  const summary = new Float32Array(layout.length);
  summary[layout.fStatistic] = 3;
  summary[layout.numeratorDegreesOfFreedom] = 1;
  summary[layout.denominatorDegreesOfFreedom] = 2;
  summary[layout.nonCentrality] = 106595;
  expect(readGeographicalDetectorPValue(summary)).toEqual({
    pValue: null,
    unavailableReason: expect.stringMatching(/nonCentrality.*10000/)
  });
  expect(() => getGeographicalDetectorPValue(summary)).toThrow(/nonCentrality.*10000/);
  summary[layout.fStatistic] = Infinity;
  expect(readGeographicalDetectorPValue(summary)).toEqual({pValue: 0, unavailableReason: null});
  summary[layout.fStatistic] = 3;
  summary[layout.nonCentrality] = 1;
  const strictProbability = getGeographicalDetectorPValue(summary);
  expect(Number.isFinite(strictProbability)).toBe(true);
  expect(readGeographicalDetectorPValue(summary)).toEqual({
    pValue: strictProbability,
    unavailableReason: null
  });
  summary[layout.fStatistic] = NaN;
  expect(readGeographicalDetectorPValue(summary)).toEqual({
    pValue: null,
    unavailableReason: expect.stringContaining('undefined statistics')
  });
  const readError = new TypeError('summary read failed');
  const brokenSummary = new Proxy(summary, {
    get() {
      throw readError;
    }
  });
  expect(() => readGeographicalDetectorPValue(brokenSummary)).toThrow(readError);
});

async function runIsolatedProbe(program: string, deadlineMilliseconds = 2000): Promise<unknown> {
  const worker = new Worker(program, {eval: true});
  let deadline: ReturnType<typeof setTimeout> | undefined;
  try {
    return await new Promise((resolve, reject) => {
      deadline = setTimeout(
        () => reject(new Error('upper-tail worker exceeded its parent deadline')),
        deadlineMilliseconds
      );
      worker.once('message', resolve);
      worker.once('error', reject);
      worker.once('exit', code =>
        reject(new Error(`upper-tail worker exited without a result (${code})`))
      );
    });
  } finally {
    clearTimeout(deadline);
    await worker.terminate();
  }
}

it('returns zero for a perfect detector even above the noncentrality mixture limit', () => {
  const layout = GPU_GEOGRAPHICAL_DETECTOR_LAYOUT;
  const summary = new Float32Array(layout.length);
  // Constant strata [100] and [101,101,101,101].
  summary[layout.q] = 1;
  summary[layout.fStatistic] = Infinity;
  summary[layout.numeratorDegreesOfFreedom] = 1;
  summary[layout.denominatorDegreesOfFreedom] = 3;
  summary[layout.nonCentrality] = 12251.25;
  expect(getGeographicalDetectorPValue(summary)).toBe(0);
  expect(getNonCentralFUpperTail(0, 1, 3, 12251.25)).toBe(1);
  expect(() => getNonCentralFUpperTail(100, 1, 3, 12251.25)).toThrow(/nonCentrality.*10000/);
});

it('keeps the small beta argument when its complement rounds to one', () => {
  // F(d,2): survival = 1 - (d*F/(d*F+2))^(d/2); here (2^-64)^(1/64) = 1/2.
  expect(getNonCentralFUpperTail(2 ** -58, 1 / 32, 2, 0)).toBeCloseTo(0.5, 14);
  // F(1,1): CDF = (2/pi)*atan(sqrt(F)).
  const value = 2 ** -54;
  const expected = 1 - (2 / Math.PI) * Math.atan(Math.sqrt(value));
  expect(getNonCentralFUpperTail(value, 1, 1, 0)).toBeCloseTo(expected, 15);
});

it('rounds a noncentral subnormal tail only after summing weighted terms', () => {
  const layout = GPU_GEOGRAPHICAL_DETECTOR_LAYOUT;
  const summary = new Float32Array(layout.length);
  summary[layout.fStatistic] = 140000;
  summary[layout.numeratorDegreesOfFreedom] = 2;
  summary[layout.denominatorDegreesOfFreedom] = 300;
  summary[layout.nonCentrality] = 500;
  // Independent high-precision Poisson mixture: 5.615585947927128e-323.
  expect(getNonCentralFUpperTail(140000, 2, 300, 500)).toBe(11 * Number.MIN_VALUE);
  expect(getGeographicalDetectorPValue(summary)).toBe(11 * Number.MIN_VALUE);
});

it('normalizes disparate beta shapes without subtracting large log-gamma values', () => {
  const denominatorDegrees = 1e15;
  const expected = Math.exp((-denominatorDegrees / 2) * Math.log1p(2 / denominatorDegrees));
  expect(getNonCentralFUpperTail(1, 2, denominatorDegrees, 0)).toBeCloseTo(expected, 13);
  // F(4,d): I_z(d/2,2) = z^(d/2) * (1 + (d/2)*(1-z)).
  const ratio = 4 / denominatorDegrees;
  const expectedFour =
    Math.exp((-denominatorDegrees / 2) * Math.log1p(ratio)) *
    (1 + ((denominatorDegrees / 2) * ratio) / (1 + ratio));
  expect(getNonCentralFUpperTail(1, 4, denominatorDegrees, 0)).toBeCloseTo(expectedFour, 12);
  // Independent 80-digit mpmath complemented-beta fixtures cover both Stirling branches.
  for (const [numeratorDegrees, expectedTail] of [
    [3, 0.3916251762710894180183498473690561],
    [16, 0.45296080948699560214539928177329359],
    [32, 0.46674489138772233717886962116171918],
    [100, 0.48119168452795953435941055846666245]
  ]) {
    const actual = getNonCentralFUpperTail(1, numeratorDegrees, denominatorDegrees, 0);
    expect(Number.isFinite(actual)).toBe(true);
    expect(Math.abs(actual - expectedTail)).toBeLessThan(1e-12);
  }
});

it('preserves a central F(2,2) upper tail below machine epsilon', () => {
  const value = 2 ** 60;
  const actual = getNonCentralFUpperTail(value, 2, 2, 0);
  expect(Number.isFinite(actual)).toBe(true);
  expect(Math.abs(actual / (1 / (1 + value)) - 1)).toBeLessThan(1e-12);
});

it('retains the F(4,2) tail when scaling the F statistic would overflow', () => {
  // I_z(1,2) = z*(2-z), z = 1/(1+2*value); form z without overflowing 2*value.
  const value = 1e308;
  const complement = 0.5 / value / (1 + 0.5 / value);
  const expected = complement * (2 - complement);
  const actual = getNonCentralFUpperTail(value, 4, 2, 0);
  expect(Number.isFinite(actual)).toBe(true);
  expect(Math.abs(actual / expected - 1)).toBeLessThan(1e-12);
});

it('preserves the noncentral tail for three strata, 303 observations and q = 0.4', () => {
  const layout = GPU_GEOGRAPHICAL_DETECTOR_LAYOUT;
  const summary = new Float32Array(layout.length);
  summary[layout.q] = 0.4;
  summary[layout.strataCount] = 3;
  summary[layout.observationCount] = 303;
  summary[layout.fStatistic] = 100; // (300 / 2) * 0.4 / 0.6
  summary[layout.numeratorDegreesOfFreedom] = 2;
  summary[layout.denominatorDegreesOfFreedom] = 300;
  summary[layout.nonCentrality] = 1.2;
  // scipy.stats.ncf.sf(100, 2, 300, float32(1.2)), SciPy 1.13.1.
  const actual = getGeographicalDetectorPValue(summary);
  expect(Number.isFinite(actual)).toBe(true);
  expect(Math.abs(actual / 6.70509172607348e-30 - 1)).toBeLessThan(1e-10);
});

it('agrees with an ordinary noncentral SciPy fixture', () => {
  expect(getNonCentralFUpperTail(3, 5, 40, 8)).toBeCloseTo(0.35681241324281293, 12);
});

it('diagnoses noncentrality outside its supported range before a cursor can stop advancing', async () => {
  // Execute the actual helper in isolation. Its only import supplies the summary layout;
  // transpilation keeps the production numerical code intact without loading GPU modules.
  const source = readFileSync(
    new URL(
      '../../../src/gpu-spatial-analysis/geographical-detector/geographical-detector-p-value.ts',
      import.meta.url
    ),
    'utf8'
  );
  const javascript = typescript.transpileModule(source, {
    compilerOptions: {module: typescript.ModuleKind.CommonJS}
  }).outputText;
  for (const nonCentrality of ['2 ** 55', 'Infinity']) {
    const result = await runIsolatedProbe(`
      const {parentPort} = require('node:worker_threads');
      const library = {};
      new Function('exports', 'require', ${JSON.stringify(javascript)})(library,
        () => ({GPU_GEOGRAPHICAL_DETECTOR_LAYOUT: ${JSON.stringify(GPU_GEOGRAPHICAL_DETECTOR_LAYOUT)}}));
      try {
        const value = library.getNonCentralFUpperTail(100, 2, 300, ${nonCentrality});
        parentPort.postMessage({value});
      } catch (error) {
        parentPort.postMessage({name: error.name, message: error.message});
      }
    `);
    expect(result).toEqual({name: 'Error', message: expect.stringMatching(/nonCentrality.*10000/)});
  }
  expect(() => getNonCentralFUpperTail(100, 2, 300, 10001)).toThrow(/nonCentrality/);
  expect(getNonCentralFUpperTail(5000, 2, 300, 10000)).toBeCloseTo(0.5108717881837139, 10);
});

it('terminates a stalled implementation from the parent instead of hanging the test worker', async () => {
  // The original cursor cannot advance at 2^54. A regression to that loop must fail within
  // a parent-enforced deadline, even though a synchronous Vitest timeout cannot interrupt it.
  await expect(
    runIsolatedProbe(
      `
    const rate = 2 ** 54;
    for (let index = rate; index < rate + 100000; index++) {
      if (index > rate) break;
    }
  `,
      100
    )
  ).rejects.toThrow('upper-tail worker exceeded its parent deadline');
});

it('retains a deep upper tail at the supported noncentrality boundary', () => {
  const actual = getNonCentralFUpperTail(20000, 2, 300, 10000);
  expect(Number.isFinite(actual)).toBe(true);
  expect(Math.abs(actual / 1.773044300452359e-42 - 1)).toBeLessThan(1e-9);
});

it('matches all existing geographical detector oracle tails on the host', () => {
  for (const reference of fixture.geographicalDetector) {
    const actual = getNonCentralFUpperTail(
      reference.fStatistic,
      reference.strataCount - 1,
      reference.values.length - reference.strataCount,
      reference.nonCentrality
    );
    expect(Number.isFinite(actual)).toBe(true);
    expect(Math.abs(actual - reference.pValue)).toBeLessThan(1e-9);
  }
});
