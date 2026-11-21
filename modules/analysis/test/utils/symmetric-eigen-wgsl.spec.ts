// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {DENSE_SOLVE_STATUS} from '../../src/analysis-core/kernels/dense-solve-wgsl';
import {getSymmetricEigenWGSL} from '../../src/analysis-core/kernels/symmetric-eigen-wgsl';
import {IS_FINITE_VALUE_WGSL} from '../../src/analysis-core/kernels/wgsl-common';
import {requireTestDevice} from './require-test-device';
import {createSeededRandom, getFloatWords, runWGSLModuleKernel} from './wgsl-module-harness';

// getSymmetricEigenWGSL against a float64 cyclic Jacobi reference: eigenvalues, eigenpair
// residuals, orthonormal vectors, and the NONFINITE and NOT_CONVERGED statuses.

const CAPACITY = 6;
const SQUARE = CAPACITY * CAPACITY;
// Per case: status, sweeps, values, vectors.
const OUTPUT_STRIDE = 2 + CAPACITY + SQUARE;

type Matrix = number[][];

/**
 * Float64 cyclic Jacobi eigenvalues, ascending. The stopping test is relative to the Frobenius
 * norm, so the reference is invariant to the matrix scale (1e-25 and 1e20 entries included).
 */
function getReferenceEigenvalues(input: Matrix): number[] {
  const size = input.length;
  const matrix = input.map(row => [...row]);
  const frobenius = matrix.flat().reduce((sum, value) => sum + value * value, 0);
  for (let sweep = 0; sweep < 100; sweep++) {
    let offDiagonal = 0;
    for (let row = 0; row < size; row++) {
      for (let column = row + 1; column < size; column++) {
        offDiagonal += matrix[row][column] ** 2;
      }
    }
    if (offDiagonal <= 1e-30 * frobenius) {
      break;
    }
    for (let lower = 0; lower < size - 1; lower++) {
      for (let upper = lower + 1; upper < size; upper++) {
        if (matrix[lower][upper] === 0) {
          continue;
        }
        const theta = (matrix[upper][upper] - matrix[lower][lower]) / (2 * matrix[lower][upper]);
        const tangent = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const cosine = 1 / Math.sqrt(tangent * tangent + 1);
        const sine = tangent * cosine;
        for (let k = 0; k < size; k++) {
          const lowerValue = matrix[k][lower];
          const upperValue = matrix[k][upper];
          matrix[k][lower] = cosine * lowerValue - sine * upperValue;
          matrix[k][upper] = sine * lowerValue + cosine * upperValue;
        }
        for (let k = 0; k < size; k++) {
          const lowerValue = matrix[lower][k];
          const upperValue = matrix[upper][k];
          matrix[lower][k] = cosine * lowerValue - sine * upperValue;
          matrix[upper][k] = sine * lowerValue + cosine * upperValue;
        }
      }
    }
  }
  return matrix.map((row, index) => row[index]).sort((left, right) => left - right);
}

function createSymmetric(random: () => number, size: number): Matrix {
  const matrix = Array.from({length: size}, () => new Array<number>(size).fill(0));
  for (let row = 0; row < size; row++) {
    for (let column = 0; column <= row; column++) {
      matrix[row][column] = matrix[column][row] = 2 * random() - 1;
    }
  }
  return matrix;
}

async function runEigenCases(
  cases: {matrix: Matrix; maximumSweeps: number}[]
): Promise<{status: number; sweeps: number; values: number[]; vectors: Matrix}[]> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const matrices = new Float32Array(cases.length * SQUARE).fill(NaN);
  const settings = new Uint32Array(cases.length * 2);
  cases.forEach(({matrix, maximumSweeps}, caseIndex) => {
    matrix.forEach((row, rowIndex) =>
      row.forEach((value, column) => {
        // Only the lower triangle is read: poison the strict upper triangle.
        matrices[caseIndex * SQUARE + rowIndex * CAPACITY + column] =
          column > rowIndex ? NaN : value;
      })
    );
    settings.set([matrix.length, maximumSweeps], caseIndex * 2);
  });
  const words = await runWGSLModuleKernel(device, {
    id: 'symmetric-eigen',
    inputs: {matrices, settings},
    outputLength: cases.length * OUTPUT_STRIDE,
    invocationCount: cases.length,
    declarations: `${IS_FINITE_VALUE_WGSL}
${getSymmetricEigenWGSL(CAPACITY)}`,
    body: `var matrix: array<f32, ${SQUARE}>;
  var vectors: array<f32, ${SQUARE}>;
  for (var slot = 0u; slot < ${SQUARE}u; slot++) { matrix[slot] = matrices[matricesOffset + index * ${SQUARE}u + slot]; }
  let eigen = getSymmetricEigen_${CAPACITY}(&matrix, &vectors, settings[settingsOffset + index * 2u], settings[settingsOffset + index * 2u + 1u], 1e-6);
  let outputBase = outputOffset + index * ${OUTPUT_STRIDE}u;
  output[outputBase] = eigen.status;
  output[outputBase + 1u] = eigen.sweeps;
  for (var slot = 0u; slot < ${CAPACITY}u; slot++) { output[outputBase + 2u + slot] = bitcast<u32>(eigen.values[slot]); }
  for (var slot = 0u; slot < ${SQUARE}u; slot++) { output[outputBase + ${2 + CAPACITY}u + slot] = bitcast<u32>(vectors[slot]); }`
  });
  const floats = getFloatWords(words);
  return cases.map(({matrix}, caseIndex) => {
    const base = caseIndex * OUTPUT_STRIDE;
    const size = matrix.length;
    return {
      status: words[base],
      sweeps: words[base + 1],
      values: Array.from(floats.slice(base + 2, base + 2 + size)),
      vectors: Array.from({length: size}, (_, row) =>
        Array.from(
          floats.slice(
            base + 2 + CAPACITY + row * CAPACITY,
            base + 2 + CAPACITY + row * CAPACITY + size
          )
        )
      )
    };
  });
}

/** Rounds every entry to float32, the precision the kernel reads. */
function roundToFloat32(matrix: Matrix): Matrix {
  return matrix.map(row => row.map(value => Math.fround(value)));
}

/** A scaled identity plus a rank-one term: eigenvalue `2 scale` three times and `4 scale` once. */
function createRepeatedEigenvalues(scale: number): Matrix {
  return Array.from({length: 4}, (_, row) =>
    Array.from({length: 4}, (_, column) => ((row === column ? 2 : 0) + 0.5) * scale)
  );
}

it('symmetric Jacobi eigen decomposition matches a float64 reference', async () => {
  const random = createSeededRandom(31);
  const cases: {matrix: Matrix; maximumSweeps: number}[] = [];
  for (let size = 1; size <= CAPACITY; size++) {
    for (let repeat = 0; repeat < 6; repeat++) {
      cases.push({matrix: createSymmetric(random, size), maximumSweeps: 16});
    }
  }
  // Repeated eigenvalues (a scaled identity plus a rank-one term).
  cases.push({matrix: createRepeatedEigenvalues(1), maximumSweeps: 16});
  await expectEigenCasesMatchReference(cases);
});

it('symmetric Jacobi eigen decomposition is invariant to the matrix scale', async () => {
  // Regression: the stopping test summed squares of unscaled entries, which underflow to 0 below
  // about 1e-19 and overflow to infinity above about 1.8e19 in float32, so the solver stopped
  // before the first sweep and returned OK with the unrotated diagonal.
  const random = createSeededRandom(47);
  const cases: {matrix: Matrix; maximumSweeps: number}[] = [];
  for (const scale of [1e-25, 1e-12, 1e12, 1e20]) {
    cases.push({
      matrix: roundToFloat32([
        [scale, scale],
        [scale, scale]
      ]),
      maximumSweeps: 16
    });
    cases.push({matrix: roundToFloat32(createRepeatedEigenvalues(scale)), maximumSweeps: 16});
    for (let size = 2; size <= CAPACITY; size++) {
      cases.push({
        matrix: roundToFloat32(
          createSymmetric(random, size).map(row => row.map(value => value * scale))
        ),
        maximumSweeps: 16
      });
    }
  }
  const results = await expectEigenCasesMatchReference(cases);
  // Every scaled case has nonzero off-diagonal entries, so at least one sweep must run.
  expect(results.map(result => result.sweeps).filter(sweeps => sweeps === 0)).toEqual([]);
});

it('symmetric Jacobi eigen decomposition of a zero matrix needs no sweep', async () => {
  const zero = Array.from({length: 4}, () => new Array<number>(4).fill(0));
  const [result] = await runEigenCases([{matrix: zero, maximumSweeps: 16}]);
  expect(result.status).toBe(DENSE_SOLVE_STATUS.OK);
  expect(result.sweeps).toBe(0);
  expect(result.values).toEqual([0, 0, 0, 0]);
  expect(result.vectors).toEqual(
    zero.map((row, rowIndex) => row.map((_, column) => (rowIndex === column ? 1 : 0)))
  );
});

/**
 * Runs the cases and expects every one to return OK with eigenvalues within `1e-5` of the largest
 * reference eigenvalue magnitude, eigenpair residuals as small, and orthonormal vectors.
 */
async function expectEigenCasesMatchReference(
  cases: {matrix: Matrix; maximumSweeps: number}[]
): Promise<{status: number; sweeps: number; values: number[]; vectors: Matrix}[]> {
  const results = await runEigenCases(cases);
  expect(getEigenMismatches(cases, results).slice(0, 10)).toEqual([]);
  return results;
}

/**
 * Mismatches of eigen results against the float64 reference. Every bound is written so that NaN
 * fails, and missing or extra values, vectors or vector entries are mismatches (TOOL-109).
 */
function getEigenMismatches(
  cases: {matrix: Matrix; maximumSweeps: number}[],
  results: {status: number; sweeps: number; values: number[]; vectors: Matrix}[]
): string[] {
  const mismatches: string[] = [];
  if (results.length !== cases.length) {
    mismatches.push(`${results.length} results for ${cases.length} cases`);
  }
  cases.forEach(({matrix}, caseIndex) => {
    const result = results[caseIndex];
    if (!result) {
      mismatches.push(`case ${caseIndex}: no result`);
      return;
    }
    const {status, values, vectors} = result;
    const size = matrix.length;
    const reference = getReferenceEigenvalues(matrix);
    const scale = Math.max(...reference.map(Math.abs), 1e-30);
    if (status !== DENSE_SOLVE_STATUS.OK) {
      mismatches.push(`case ${caseIndex}: status ${status}`);
    }
    if (values.length !== size) {
      mismatches.push(`case ${caseIndex}: ${values.length} values for size ${size}`);
    }
    if (vectors.length !== size || vectors.some(row => row.length !== size)) {
      mismatches.push(`case ${caseIndex}: vectors are not ${size} by ${size}`);
      return;
    }
    reference.forEach((expected, index) => {
      const value = values[index];
      if (!Number.isFinite(value) || !(Math.abs(value - expected) <= 1e-5 * scale)) {
        mismatches.push(`case ${caseIndex}: value ${index} ${value} versus ${expected}`);
      }
    });
    for (let column = 0; column < size; column++) {
      const vector = vectors.map(row => row[column]);
      const residual = Math.hypot(
        ...matrix.map(
          (row, rowIndex) =>
            row.reduce((sum, entry, k) => sum + entry * vector[k], 0) -
            values[column] * vector[rowIndex]
        )
      );
      if (!(residual <= 1e-5 * scale)) {
        mismatches.push(`case ${caseIndex}: residual of vector ${column} is ${residual}`);
      }
      for (let other = 0; other < size; other++) {
        const dot = vectors.reduce((sum, row) => sum + row[column] * row[other], 0);
        if (!(Math.abs(dot - (column === other ? 1 : 0)) <= 1e-5)) {
          mismatches.push(`case ${caseIndex}: vectors ${column}, ${other} dot ${dot}`);
        }
      }
    }
  });
  return mismatches;
}

it('symmetric Jacobi eigen reports non-finite input and non-convergence', async () => {
  const results = await runEigenCases([
    {
      matrix: [
        [1, 0],
        [NaN, 2]
      ],
      maximumSweeps: 16
    },
    {
      matrix: [
        [2, 1, 0],
        [1, 2, 1],
        [0, 1, 2]
      ],
      maximumSweeps: 0
    },
    {
      matrix: [
        [3, 0],
        [0, 1]
      ],
      maximumSweeps: 0
    }
  ]);
  expect(results.map(result => result.status)).toEqual([
    DENSE_SOLVE_STATUS.NONFINITE,
    DENSE_SOLVE_STATUS.NOT_CONVERGED,
    DENSE_SOLVE_STATUS.OK
  ]);
  // NONFINITE writes nothing: the function-local vectors keep their zero initial value.
  expect(results[0].vectors).toEqual([
    [0, 0],
    [0, 0]
  ]);
  // A diagonal input needs no sweep and comes back sorted with permuted unit vectors.
  expect(results[2].values).toEqual([1, 3]);
  expect(results[2].vectors).toEqual([
    [0, 1],
    [1, 0]
  ]);
});

// CORE-106: diagonals of opposite sign near the float32 limit overflowed theta's difference, and a
// coupling above 1.7e38 overflowed twice the coupling; the rotation then cleared the coupling with
// a zero (or 45 degree) angle and returned OK with residuals near 0.03. Reference eigenvalues:
// numpy.linalg.eigh in float64 of the float32 entries (/usr/bin/python3).
it('symmetric Jacobi eigen keeps a coupling when the diagonal difference overflows (CORE-106)', async () => {
  const cases = [
    {
      matrix: roundToFloat32([
        [-3e38, 1e37],
        [1e37, 3e38]
      ]),
      values: [-3.0016662094333818e38, 3.0016662094333818e38]
    },
    {
      matrix: roundToFloat32([
        [0, 2e38],
        [2e38, 1e38]
      ]),
      values: [-1.5615527628837527e38, 2.561552730912322e38]
    }
  ];
  const results = await runEigenCases(cases.map(({matrix}) => ({matrix, maximumSweeps: 16})));
  results.forEach((result, caseIndex) => {
    const {matrix, values} = cases[caseIndex];
    expect(result.status).toBe(DENSE_SOLVE_STATUS.OK);
    expect([...result.values, ...result.vectors.flat()].every(Number.isFinite)).toBe(true);
    result.values.forEach((value, index) => {
      expect(Math.abs(value - values[index])).toBeLessThanOrEqual(1e-5 * Math.abs(values[1]));
      // Residual |A v - lambda v| / |lambda| in float64.
      const x = result.vectors[0][index];
      const y = result.vectors[1][index];
      const residual = Math.hypot(
        matrix[0][0] * x + matrix[0][1] * y - value * x,
        matrix[1][0] * x + matrix[1][1] * y - value * y
      );
      expect(residual / Math.abs(value)).toBeLessThan(1e-5);
    });
  });
});
