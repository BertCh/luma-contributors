// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpaceTimeCrossCorrelation} from '../../../src/gpu-spatial-analysis/space-time-statistics/index';
import type {GPUSpaceTimeWeightsLayout} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {SPACE_TIME_STATISTICS_REFERENCE as REFERENCE} from './space-time-statistics-reference';

// Oracles: test/oracles/space-time-statistics.py (numpy corrcoef, scipy norm.sf Fisher p-values,
// scipy false_discovery_control) and the independent double implementation below, which also
// covers untested pairs (NaN values, constant series) that the fixture does not contain.

const CELLS = REFERENCE.n;
const PERIODS = REFERENCE.periods;
const LAGS = REFERENCE.crossCorrelation.lags;

function layoutCube(periodMajor: number[][], layout: GPUSpaceTimeWeightsLayout): Float32Array {
  const cube = new Float32Array(CELLS * PERIODS);
  for (let period = 0; period < PERIODS; period++) {
    for (let cell = 0; cell < CELLS; cell++) {
      cube[layout === 'period-major' ? period * CELLS + cell : cell * PERIODS + period] =
        periodMajor[period][cell];
    }
  }
  return cube;
}

/** Complementary error function in double precision (Lentz continued fraction, series for small x). */
function erfc(x: number): number {
  if (x < 2) {
    // erf series: 2/sqrt(pi) sum (-1)^n x^(2n+1) / (n! (2n+1))
    let term = x;
    let sum = x;
    for (let n = 1; n < 100; n++) {
      term *= (-x * x) / n;
      sum += term / (2 * n + 1);
    }
    return 1 - (2 / Math.sqrt(Math.PI)) * sum;
  }
  // erfc continued fraction: exp(-x^2)/(x sqrt(pi)) / (1 + 1/(2x^2) / (1 + 2/(2x^2) / (1 + ...)))
  let fraction = 0;
  for (let n = 200; n >= 1; n--) {
    fraction = n / 2 / (x + fraction);
  }
  return Math.exp(-x * x) / (Math.sqrt(Math.PI) * (x + fraction));
}

type Reference = {r: number; p: number; n: number};

/** Double-precision Pearson r and Fisher p of the overlap pairs `(first[t], second[t + lag])`. */
function computeReference(first: number[], second: number[], lag: number): Reference {
  const start = Math.max(0, -lag);
  const stop = Math.min(first.length, first.length - lag);
  const n = Math.max(stop - start, 0);
  const missing = {r: NaN, p: NaN, n};
  if (n < 4) return missing;
  const a = first.slice(start, stop);
  const b = second.slice(start + lag, stop + lag);
  if (![...a, ...b].every(Number.isFinite)) return missing;
  const meanA = a.reduce((total, value) => total + value, 0) / n;
  const meanB = b.reduce((total, value) => total + value, 0) / n;
  let saa = 0;
  let sbb = 0;
  let sab = 0;
  for (let index = 0; index < n; index++) {
    saa += (a[index] - meanA) ** 2;
    sbb += (b[index] - meanB) ** 2;
    sab += (a[index] - meanA) * (b[index] - meanB);
  }
  if (!(saa > 0 && sbb > 0)) return missing;
  const r = Math.max(-1, Math.min(1, sab / Math.sqrt(saa * sbb)));
  const z = Math.abs(Math.atanh(r)) * Math.sqrt(n - 3);
  return {r, p: erfc(z / Math.SQRT2), n};
}

/** Benjamini-Hochberg over the finite p-values (NaN stays NaN). */
function adjustBenjaminiHochberg(pValues: number[]): number[] {
  const tested = pValues.map((p, index) => ({p, index})).filter(entry => Number.isFinite(entry.p));
  tested.sort((left, right) => left.p - right.p);
  const adjusted = new Array<number>(pValues.length).fill(NaN);
  let running = 1;
  for (let rank = tested.length; rank >= 1; rank--) {
    running = Math.min(running, (tested[rank - 1].p * tested.length) / rank);
    adjusted[tested[rank - 1].index] = running;
  }
  return adjusted;
}

async function runCrossCorrelation(
  layout: GPUSpaceTimeWeightsLayout,
  first: Float32Array,
  second: Float32Array,
  multipleTesting?: 'none' | 'benjamini-hochberg' | 'bonferroni'
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const rows = CELLS * LAGS.length;
  const correlations = rig.output('float32', rows);
  const pValues = rig.output('float32', rows);
  const adjustedPValues = rig.output('float32', rows);
  const pairCounts = rig.output('uint32', rows);
  const testedCount = rig.output('uint32', 1);
  rig.run(
    new GPUSpaceTimeCrossCorrelation({
      first: rig.input(first, 'float32', first.length),
      second: rig.input(second, 'float32', second.length),
      layout,
      cellCount: CELLS,
      periodCount: PERIODS,
      lags: LAGS,
      multipleTesting,
      output: {
        correlations: correlations.view,
        pValues: pValues.view,
        adjustedPValues: adjustedPValues.view,
        pairCounts: pairCounts.view,
        testedCount: testedCount.view
      }
    })
  );
  const result = {
    correlations: await readFloat32(correlations.buffer, rows),
    pValues: await readFloat32(pValues.buffer, rows),
    adjustedPValues: await readFloat32(adjustedPValues.buffer, rows),
    pairCounts: await readUint32(pairCounts.buffer, rows),
    testedCount: (await readUint32(testedCount.buffer, 1))[0]
  };
  rig.destroy();
  return result;
}

function expectCloseP(actual: number, expected: number, label: string): void {
  expect(Math.abs(actual - expected), label).toBeLessThanOrEqual(1e-5 + 1e-3 * expected);
}

for (const layout of ['period-major', 'cell-major'] as const) {
  it(`GPUSpaceTimeCrossCorrelation ${layout} matches numpy, scipy and the double reference with BH across cells and lags`, async () => {
    const result = await runCrossCorrelation(
      layout,
      layoutCube(REFERENCE.cube, layout),
      layoutCube(REFERENCE.second, layout)
    );
    const fixture = REFERENCE.crossCorrelation;
    const rows = CELLS * LAGS.length;
    expect(result.testedCount).toBe(rows);
    for (let row = 0; row < rows; row++) {
      const cell = Math.floor(row / LAGS.length);
      const lag = LAGS[row % LAGS.length];
      const first = REFERENCE.cube.map(period => period[cell]);
      const second = REFERENCE.second.map(period => period[cell]);
      const reference = computeReference(first, second, lag);
      const label = `${layout} cell ${cell} lag ${lag}`;
      expect(result.pairCounts[row], `${label} pairs`).toBe(fixture.pairCounts[row]);
      // numpy corrcoef, then the double twin.
      expect(
        Math.abs(result.correlations[row] - fixture.correlations[row]),
        `${label} r`
      ).toBeLessThanOrEqual(1e-4 + 1e-4 * Math.abs(fixture.correlations[row]));
      expect(Math.abs(reference.r - fixture.correlations[row]), `${label} twin r`).toBeLessThan(
        1e-12
      );
      expectCloseP(result.pValues[row], fixture.fisherPValues[row], `${label} p vs scipy`);
      expect(Math.abs(reference.p - fixture.fisherPValues[row]), `${label} twin p`).toBeLessThan(
        1e-9
      );
      expectCloseP(
        result.adjustedPValues[row],
        fixture.adjustedFisherBH[row],
        `${label} BH vs scipy`
      );
    }
  });
}

it('GPUSpaceTimeCrossCorrelation leaves NaN, constant and short pairs untested and out of the FDR family', async () => {
  const layout = 'cell-major';
  const first = layoutCube(REFERENCE.cube, layout);
  const second = layoutCube(REFERENCE.second, layout);
  // Cell 0: a NaN in the second series (every lag overlaps it or not, per its period).
  second[0 * PERIODS + 3] = NaN;
  // Cell 1: constant first series.
  for (let period = 0; period < PERIODS; period++) first[1 * PERIODS + period] = 2.5;
  const result = await runCrossCorrelation(layout, first, second);
  const expectedP: number[] = [];
  for (let cell = 0; cell < CELLS; cell++) {
    const firstSeries = Array.from(
      {length: PERIODS},
      (_, period) => first[cell * PERIODS + period]
    );
    const secondSeries = Array.from(
      {length: PERIODS},
      (_, period) => second[cell * PERIODS + period]
    );
    for (const lag of LAGS) {
      const reference = computeReference(firstSeries, secondSeries, lag);
      expectedP.push(reference.p);
      const row = expectedP.length - 1;
      if (Number.isNaN(reference.p)) {
        expect(result.correlations[row], `cell ${cell} lag ${lag} r`).toBeNaN();
        expect(result.pValues[row]).toBeNaN();
        expect(result.adjustedPValues[row]).toBeNaN();
      } else {
        expectCloseP(result.pValues[row], reference.p, `cell ${cell} lag ${lag} p`);
      }
    }
  }
  const untested = expectedP.filter(Number.isNaN).length;
  expect(untested).toBeGreaterThan(LAGS.length); // cell 1 fully, cell 0 for the lags that touch period 3
  const familySize = expectedP.length - untested;
  expect(result.testedCount).toBe(familySize);
  // The family is only the tested pairs: BH over the double p-values, not over all rows.
  const expectedAdjusted = adjustBenjaminiHochberg(expectedP);
  expectedAdjusted.forEach((value, row) => {
    if (!Number.isNaN(value))
      expectCloseP(result.adjustedPValues[row], value, `adjusted row ${row}`);
  });
});

it('GPUSpaceTimeCrossCorrelation recovers a planted lag (largest mean |r| at lag 1) and Bonferroni is more conservative than BH', async () => {
  // second[t] ~ 0.7 first[t - 1], so the strongest correlation sits at lag 1 for the typical cell.
  const layout = 'period-major';
  const first = layoutCube(REFERENCE.cube, layout);
  const second = layoutCube(REFERENCE.second, layout);
  const bonferroni = await runCrossCorrelation(layout, first, second, 'bonferroni');
  const benjamini = await runCrossCorrelation(layout, first, second, 'benjamini-hochberg');
  const meanAbsolute = LAGS.map((_, index) => {
    let total = 0;
    for (let cell = 0; cell < CELLS; cell++) {
      total += Math.abs(benjamini.correlations[cell * LAGS.length + index]);
      expect(bonferroni.adjustedPValues[cell * LAGS.length + index]).toBeGreaterThanOrEqual(
        benjamini.adjustedPValues[cell * LAGS.length + index] - 1e-6
      );
    }
    return total / CELLS;
  });
  const lagOne = LAGS.indexOf(1);
  meanAbsolute.forEach((value, index) => {
    if (index !== lagOne) expect(meanAbsolute[lagOne], `lag ${LAGS[index]}`).toBeGreaterThan(value);
  });
});

it('GPUSpaceTimeCrossCorrelation refuses lags without enough overlap and duplicates', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const cube = rig.input(layoutCube(REFERENCE.cube, 'period-major'), 'float32', CELLS * PERIODS);
  const other = rig.input(layoutCube(REFERENCE.second, 'period-major'), 'float32', CELLS * PERIODS);
  const build = (lags: number[]) => () =>
    new GPUSpaceTimeCrossCorrelation({
      first: cube,
      second: other,
      layout: 'period-major',
      cellCount: CELLS,
      periodCount: PERIODS,
      lags,
      output: {
        correlations: rig.output('float32', CELLS * lags.length).view,
        pValues: rig.output('float32', CELLS * lags.length).view
      }
    });
  expect(build([0, 3, -3])).not.toThrow();
  expect(build([4])).toThrow(/lag/);
  expect(build([-4])).toThrow(/lag/);
  expect(build([1, 1])).toThrow(/distinct/);
  expect(build([])).toThrow(/lags length/);
  rig.destroy();
});
