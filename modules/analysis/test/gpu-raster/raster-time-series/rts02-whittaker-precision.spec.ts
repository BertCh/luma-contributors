// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPURasterWhittakerSmoother,
  MAXIMUM_WHITTAKER_SLICE_COUNT,
  MAXIMUM_WHITTAKER_SMOOTHING_PARAMETER
} from '@luma-contributors/analysis/raster-analysis';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createRasterTimeSeriesHarness,
  getMaximumAbsoluteDifference
} from './raster-time-series-test-utils';
import sparseReference from '../../oracles/fixtures/rts02-whittaker-sparse-exact.json';

/**
 * Binary64 reference: `(W + lambda D'D) z = W y` on a regular axis, with `D` written out from the
 * binomial coefficients and a dense Cholesky solve. It shares no arithmetic with the kernel (which
 * builds `D` by divided differences and factors a float32 band).
 */
function solveWhittakerInBinary64(
  values: readonly number[],
  smoothingParameter: number,
  differenceOrder: number
): number[] {
  const count = values.length;
  const stencil = [
    [-1, 1],
    [1, -2, 1],
    [-1, 3, -3, 1]
  ][differenceOrder - 1];
  const matrix = Array.from({length: count}, () => new Float64Array(count));
  const rightSide = new Float64Array(count);
  for (let slice = 0; slice < count; slice++) {
    if (Number.isFinite(values[slice])) {
      matrix[slice][slice] = 1;
      rightSide[slice] = values[slice];
    }
  }
  for (let row = 0; row + differenceOrder < count; row++) {
    for (let left = 0; left <= differenceOrder; left++) {
      for (let right = 0; right <= differenceOrder; right++) {
        matrix[row + left][row + right] += smoothingParameter * stencil[left] * stencil[right];
      }
    }
  }
  // Dense Cholesky, lower triangle in place.
  for (let column = 0; column < count; column++) {
    for (let row = column; row < count; row++) {
      let sum = matrix[row][column];
      for (let inner = Math.max(0, row - differenceOrder); inner < column; inner++) {
        sum -= matrix[row][inner] * matrix[column][inner];
      }
      matrix[row][column] = row === column ? Math.sqrt(sum) : sum / matrix[column][column];
    }
  }
  const solution = Array.from(rightSide);
  for (let row = 0; row < count; row++) {
    for (let inner = Math.max(0, row - differenceOrder); inner < row; inner++) {
      solution[row] -= matrix[row][inner] * solution[inner];
    }
    solution[row] /= matrix[row][row];
  }
  for (let row = count - 1; row >= 0; row--) {
    for (let inner = row + 1; inner <= Math.min(count - 1, row + differenceOrder); inner++) {
      solution[row] -= matrix[inner][row] * solution[inner];
    }
    solution[row] /= matrix[row][row];
  }
  return solution;
}

/** `offset + 0.3 sin(...)` with one gap in five: the series of finding raster-imagery-02. */
function getSeries(sliceCount: number, offset: number): number[] {
  return Array.from({length: sliceCount}, (_, slice) =>
    slice % 5 === 3 ? Number.NaN : offset + 0.3 * Math.sin((2 * Math.PI * slice) / 46 + 0.4)
  );
}

async function runSmoother(values: number[], smoothingParameter: number, differenceOrder: number) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const result = await harness.run(
    new GPURasterWhittakerSmoother({
      cellCount: 1,
      sliceCount: values.length,
      differenceOrder,
      reweightPassCount: 0,
      values: harness.floats(values),
      // Written directly so a value above the accepted cap can be probed.
      parameters: harness.floats([smoothingParameter, 0.01]),
      output: {
        smoothed: harness.output('smoothed', 'float32', values.length),
        status: harness.output('status', 'uint32', 1)
      }
    })
  );
  return {smoothed: result['smoothed'], status: result['status'][0]};
}

// Finding raster-imagery-02. Before the fix this series at 300 was 0.37 off at lambda 1e4 with
// status 0 (difference order 2), and lambda 1e7 returned values of order 1e9 with status 0.
for (const differenceOrder of [1, 2, 3]) {
  for (const offset of [0.5, 300, 3000]) {
    it(`Whittaker order ${differenceOrder} around ${offset} matches binary64 up to the accepted cap`, async () => {
      const values = getSeries(MAXIMUM_WHITTAKER_SLICE_COUNT, offset);
      // One float32 step of the published value plus 1e-4 of the signal amplitude (0.3).
      const tolerance = offset * 2 ** -23 + 3e-5;
      for (const smoothingParameter of [1e2, 1e4, 1e6, MAXIMUM_WHITTAKER_SMOOTHING_PARAMETER]) {
        const expected = solveWhittakerInBinary64(
          values.map(value => Math.fround(value)),
          smoothingParameter,
          differenceOrder
        );
        const {smoothed, status} = await runSmoother(values, smoothingParameter, differenceOrder);
        expect(status, `status at lambda ${smoothingParameter}`).toBe(0);
        expect(
          getMaximumAbsoluteDifference(smoothed, expected),
          `error at lambda ${smoothingParameter}`
        ).toBeLessThan(tolerance);
      }
    });
  }
}

it('Whittaker returns a constant series unchanged and fills its gaps with the constant', async () => {
  const values = Array.from({length: 64}, (_, slice) => (slice % 7 === 2 ? Number.NaN : 287.15));
  for (const differenceOrder of [1, 2, 3]) {
    const {smoothed, status} = await runSmoother(values, 1e5, differenceOrder);
    expect(status).toBe(0);
    expect(smoothed).toEqual(new Array(64).fill(Math.fround(287.15)));
  }
});

it('Whittaker reports status 2 when the system is singular', async () => {
  // No penalty and a gap: the gap slice has no equation.
  const {smoothed, status} = await runSmoother([1, 2, Number.NaN, 4, 5, 6], 0, 2);
  expect(status).toBe(2);
  expect(smoothed.every(Number.isNaN)).toBe(true);
});

// Review follow-up of raster-imagery-02: very sparse support. The reference is an exact rational
// solve (`test/oracles/rts02-whittaker-sparse-exact.py`), because a binary64 factorisation of the
// normal equations is itself unusable here. 2, 3 and 5 valid observations out of 256 slices (the
// slice cap), at the start, the middle, the end or spread over the series; lambda 1 to 1e7; every
// order the support allows. Every system is non-singular, so every case must return status 0.
// (A status 2 threshold on the ratio of the factor's diagonals rejected five of these: five
// adjacent observations at the start or the middle, order 3, lambda from 1e4.)
//
// Tolerances are what float32 gives on the problem, measured and rounded up: one float32 step of
// the values (near 300) for order 1; for orders 2 and 3 the curve is continued as a line or a
// parabola across gaps of up to 250 slices, and the error is a fraction of the spread of the
// curve (largest measured 5.2e-4 for order 2 and 1.24e-2 for order 3).
const SPARSE_RELATIVE_TOLERANCE = [0, 0, 1e-3, 2.5e-2];

for (const placement of ['start', 'middle', 'end', 'spread']) {
  it(`Whittaker solves 2, 3 and 5 observations at the ${placement} of 256 slices for every lambda`, async () => {
    const entries = sparseReference.cases.filter(entry => entry.name.includes(`-${placement}-`));
    expect(entries.length).toBe(30);
    for (const entry of entries) {
      const values = new Array<number>(sparseReference.sliceCount).fill(Number.NaN);
      entry.positions.forEach((position, index) => {
        values[position] = entry.observations[index];
      });
      const {smoothed, status} = await runSmoother(values, entry.smoothing, entry.order);
      expect(status, `${entry.name} status`).toBe(0);
      const spread = Math.max(...entry.expected) - Math.min(...entry.expected);
      expect(
        getMaximumAbsoluteDifference(smoothed, entry.expected),
        `${entry.name} error (spread ${spread})`
      ).toBeLessThanOrEqual(6e-5 + SPARSE_RELATIVE_TOLERANCE[entry.order] * spread);
    }
  });
}
