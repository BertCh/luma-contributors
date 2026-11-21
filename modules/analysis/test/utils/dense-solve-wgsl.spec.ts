// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  DENSE_SOLVE_STATUS,
  getDenseCholeskyWGSL,
  getDenseLUWGSL
} from '../../src/analysis-core/kernels/dense-solve-wgsl';
import {IS_FINITE_VALUE_WGSL} from '../../src/analysis-core/kernels/wgsl-common';
import {requireTestDevice} from './require-test-device';
import {createSeededRandom, getFloatWords, runWGSLModuleKernel} from './wgsl-module-harness';

// The dense solvers of analysis-core/kernels/dense-solve-wgsl.ts against a float64 CPU reference:
// solutions, determinants and inverses on well-conditioned systems, and every status code.

const CAPACITY = 6;
const SQUARE = CAPACITY * CAPACITY;
// Per case: status, condition estimate, log |det|, det sign, solution, inverse (row-major).
const OUTPUT_STRIDE = 4 + CAPACITY + SQUARE;

type Matrix = number[][];

type SolveCase = {
  name: string;
  matrix: Matrix;
  vector: number[];
  /** Cholesky: 1 to equilibrate. LU: equilibration passes. */
  mode: number;
  /** LU only: pivot tolerance. */
  pivotTolerance?: number;
  minimumPivotRatio: number;
  expectedStatus: number | number[];
};

type Reference = {solution: number[]; inverse: Matrix; logAbsDeterminant: number; sign: number};

/** Float64 Gauss-Jordan with partial pivoting: solution, inverse and determinant. */
function solveReference(matrix: Matrix, vector: number[]): Reference {
  const size = matrix.length;
  const work = matrix.map((row, rowIndex) => [
    ...row,
    ...row.map((_, column) => (column === rowIndex ? 1 : 0)),
    vector[rowIndex]
  ]);
  let logAbsDeterminant = 0;
  let sign = 1;
  for (let column = 0; column < size; column++) {
    let pivotRow = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(work[row][column]) > Math.abs(work[pivotRow][column])) {
        pivotRow = row;
      }
    }
    if (pivotRow !== column) {
      [work[column], work[pivotRow]] = [work[pivotRow], work[column]];
      sign = -sign;
    }
    const pivot = work[column][column];
    logAbsDeterminant += Math.log(Math.abs(pivot));
    sign *= Math.sign(pivot);
    for (let entry = 0; entry < work[column].length; entry++) {
      work[column][entry] /= pivot;
    }
    for (let row = 0; row < size; row++) {
      if (row !== column) {
        const factor = work[row][column];
        for (let entry = 0; entry < work[row].length; entry++) {
          work[row][entry] -= factor * work[column][entry];
        }
      }
    }
  }
  return {
    solution: work.map(row => row[2 * size]),
    inverse: work.map(row => row.slice(size, 2 * size)),
    logAbsDeterminant,
    sign
  };
}

function multiply(left: Matrix, right: Matrix): Matrix {
  return left.map(row =>
    right[0].map((_, column) => row.reduce((sum, value, k) => sum + value * right[k][column], 0))
  );
}

function transpose(matrix: Matrix): Matrix {
  return matrix[0].map((_, column) => matrix.map(row => row[column]));
}

function createRandomMatrix(random: () => number, rows: number, columns: number): Matrix {
  return Array.from({length: rows}, () => Array.from({length: columns}, () => 2 * random() - 1));
}

/** Symmetric positive definite: `B B^T + size I`, condition number in the tens. */
function createPositiveDefinite(random: () => number, size: number): Matrix {
  const factor = createRandomMatrix(random, size, size);
  return multiply(factor, transpose(factor)).map((row, rowIndex) =>
    row.map((value, column) => value + (rowIndex === column ? size : 0))
  );
}

/** Diagonally dominant general matrix with mixed signs. */
function createGeneral(random: () => number, size: number): Matrix {
  return createRandomMatrix(random, size, size).map((row, rowIndex) =>
    row.map((value, column) => value + (rowIndex === column ? (rowIndex % 2 ? -3 : 3) : 0))
  );
}

/** `D A D`: scales row and column `i` by `scales[i]`. */
function scaleSymmetric(matrix: Matrix, scales: number[]): Matrix {
  return matrix.map((row, rowIndex) =>
    row.map((value, column) => value * scales[rowIndex] * scales[column])
  );
}

function getRelativeError(actual: number[], expected: number[]): number {
  const difference = Math.hypot(...actual.map((value, index) => value - expected[index]));
  return difference / Math.max(Math.hypot(...expected), 1e-30);
}

async function runSolveCases(
  method: 'cholesky' | 'lu',
  cases: SolveCase[]
): Promise<
  {status: number; ratio: number; log: number; sign: number; solution: number[]; inverse: Matrix}[]
> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const matrices = new Float32Array(cases.length * SQUARE).fill(NaN);
  const vectors = new Float32Array(cases.length * CAPACITY).fill(NaN);
  const settings = new Float32Array(cases.length * 4);
  cases.forEach((solveCase, caseIndex) => {
    const size = solveCase.matrix.length;
    solveCase.matrix.forEach((row, rowIndex) =>
      row.forEach((value, column) => {
        matrices[caseIndex * SQUARE + rowIndex * CAPACITY + column] = value;
      })
    );
    solveCase.vector.forEach((value, rowIndex) => {
      vectors[caseIndex * CAPACITY + rowIndex] = value;
    });
    settings.set(
      [size, solveCase.mode, solveCase.pivotTolerance ?? 0, solveCase.minimumPivotRatio],
      caseIndex * 4
    );
  });
  const isCholesky = method === 'cholesky';
  const factor = isCholesky
    ? `var factorization = factorDenseCholesky_${CAPACITY}(&matrix, size, settings[settingsOffset + base4 + 1u] > 0.0, settings[settingsOffset + base4 + 3u]);
  let ratio = factorization.smallestPivotRatio;
  let usable = factorization.status == 0u || factorization.status == 3u;
  var logAbsDeterminant = 0.0;
  var sign = 1.0;
  if (usable) { logAbsDeterminant = getDenseCholeskyLogDeterminant_${CAPACITY}(&matrix, &factorization); }`
    : `var factorization = factorDenseLU_${CAPACITY}(&matrix, size, u32(settings[settingsOffset + base4 + 1u]), settings[settingsOffset + base4 + 2u], settings[settingsOffset + base4 + 3u]);
  let ratio = factorization.pivotRatio;
  let usable = factorization.status == 0u || factorization.status == 3u;
  let logAbsDeterminant = factorization.logAbsDeterminant;
  let sign = factorization.determinantSign;`;
  const solveName = isCholesky ? `DenseCholesky_${CAPACITY}` : `DenseLU_${CAPACITY}`;
  const words = await runWGSLModuleKernel(device, {
    id: `dense-solve-${method}`,
    inputs: {matrices, vectors, settings},
    outputLength: cases.length * OUTPUT_STRIDE,
    invocationCount: cases.length,
    declarations: `${IS_FINITE_VALUE_WGSL}
${isCholesky ? getDenseCholeskyWGSL(CAPACITY) : getDenseLUWGSL(CAPACITY)}`,
    body: `let base4 = index * 4u;
  let size = u32(settings[settingsOffset + base4]);
  var matrix: array<f32, ${SQUARE}>;
  var vector: array<f32, ${CAPACITY}>;
  for (var slot = 0u; slot < ${SQUARE}u; slot++) { matrix[slot] = matrices[matricesOffset + index * ${SQUARE}u + slot]; }
  for (var slot = 0u; slot < ${CAPACITY}u; slot++) { vector[slot] = vectors[vectorsOffset + index * ${CAPACITY}u + slot]; }
  ${factor}
  let outputBase = outputOffset + index * ${OUTPUT_STRIDE}u;
  output[outputBase] = factorization.status;
  output[outputBase + 1u] = bitcast<u32>(ratio);
  output[outputBase + 2u] = bitcast<u32>(logAbsDeterminant);
  output[outputBase + 3u] = bitcast<u32>(sign);
  if (usable) {
    var inverse: array<f32, ${SQUARE}>;
    invert${solveName}(&matrix, &factorization, &inverse);
    solve${solveName}(&matrix, &factorization, &vector);
    for (var slot = 0u; slot < ${CAPACITY}u; slot++) { output[outputBase + 4u + slot] = bitcast<u32>(vector[slot]); }
    for (var slot = 0u; slot < ${SQUARE}u; slot++) { output[outputBase + ${4 + CAPACITY}u + slot] = bitcast<u32>(inverse[slot]); }
  }`
  });
  const floats = getFloatWords(words);
  return cases.map((solveCase, caseIndex) => {
    const base = caseIndex * OUTPUT_STRIDE;
    const size = solveCase.matrix.length;
    return {
      status: words[base],
      ratio: floats[base + 1],
      log: floats[base + 2],
      sign: floats[base + 3],
      solution: Array.from(floats.slice(base + 4, base + 4 + size)),
      inverse: Array.from({length: size}, (_, row) =>
        Array.from(
          floats.slice(
            base + 4 + CAPACITY + row * CAPACITY,
            base + 4 + CAPACITY + row * CAPACITY + size
          )
        )
      )
    };
  });
}

function checkStatuses(cases: SolveCase[], results: {status: number}[]): void {
  const mismatches = cases
    .map((solveCase, index) => {
      const expected = [solveCase.expectedStatus].flat();
      return expected.includes(results[index].status)
        ? ''
        : `${solveCase.name}: status ${results[index].status}, expected ${expected.join(' or ')}`;
    })
    .filter(Boolean);
  expect(mismatches).toEqual([]);
}

function checkSolutions(
  cases: SolveCase[],
  results: {status: number; log: number; sign: number; solution: number[]; inverse: Matrix}[],
  tolerance: number
): void {
  expect(getSolutionMismatches(cases, results, tolerance)).toEqual([]);
}

/**
 * Mismatches of the OK results against the float64 reference. Every actual value must be finite,
 * solution and inverse must have the case's size, and each bound is written `!(error <= bound)`
 * so that a NaN error fails: `NaN > tolerance` is false and used to let a NaN through (CORE-117).
 */
function getSolutionMismatches(
  cases: SolveCase[],
  results: {status: number; log: number; sign: number; solution: number[]; inverse: Matrix}[],
  tolerance: number
): string[] {
  const mismatches: string[] = [];
  if (results.length !== cases.length) {
    mismatches.push(`${results.length} results for ${cases.length} cases`);
  }
  cases.forEach((solveCase, index) => {
    const result = results[index];
    if (!result || result.status !== DENSE_SOLVE_STATUS.OK) {
      return;
    }
    const size = solveCase.matrix.length;
    const actualValues = [result.log, result.sign, ...result.solution, ...result.inverse.flat()];
    if (
      result.solution.length !== size ||
      result.inverse.length !== size ||
      result.inverse.some(row => row.length !== size)
    ) {
      mismatches.push(`${solveCase.name}: solution or inverse is not of size ${size}`);
      return;
    }
    if (!actualValues.every(Number.isFinite)) {
      mismatches.push(`${solveCase.name}: OK status with a non-finite output`);
      return;
    }
    const reference = solveReference(solveCase.matrix, solveCase.vector);
    const solutionError = getRelativeError(result.solution, reference.solution);
    const inverseError = getRelativeError(result.inverse.flat(), reference.inverse.flat());
    const logError = Math.abs(result.log - reference.logAbsDeterminant);
    if (
      !(solutionError <= tolerance) ||
      !(inverseError <= tolerance) ||
      !(logError <= tolerance * size * 10)
    ) {
      mismatches.push(
        `${solveCase.name}: solution ${solutionError.toExponential(2)} inverse ${inverseError.toExponential(2)} log ${logError.toExponential(2)}`
      );
    }
    if (result.sign !== reference.sign) {
      mismatches.push(
        `${solveCase.name}: determinant sign ${result.sign}, expected ${reference.sign}`
      );
    }
  });
  return mismatches;
}

it('dense Cholesky solves, inverts and takes log-determinants of positive definite systems', async () => {
  const random = createSeededRandom(11);
  const cases: SolveCase[] = [];
  for (let size = 1; size <= CAPACITY; size++) {
    for (const mode of [0, 1]) {
      for (let repeat = 0; repeat < 4; repeat++) {
        const matrix = createPositiveDefinite(random, size);
        // Column scales spanning six decades: equilibration makes the factor scale-free.
        const scales = Array.from({length: size}, () => 10 ** (6 * random() - 3));
        cases.push({
          name: `spd size ${size} mode ${mode} repeat ${repeat}`,
          matrix: mode ? scaleSymmetric(matrix, scales) : matrix,
          vector: Array.from({length: size}, () => 2 * random() - 1),
          mode,
          minimumPivotRatio: 1e-6,
          expectedStatus: DENSE_SOLVE_STATUS.OK
        });
      }
    }
  }
  const results = await runSolveCases('cholesky', cases);
  checkStatuses(cases, results);
  checkSolutions(cases, results, 2e-4);
});

it('dense Cholesky reports non-finite, singular and ill-conditioned systems', async () => {
  const random = createSeededRandom(12);
  const base = createPositiveDefinite(random, 4);
  const withEntry = (row: number, column: number, value: number): Matrix =>
    base.map((values, rowIndex) =>
      values.map((entry, columnIndex) =>
        (rowIndex === row && columnIndex === column) || (rowIndex === column && columnIndex === row)
          ? value
          : entry
      )
    );
  // Gram matrix of [x, x + 3e-3 noise, z]: the second column has 1 - R^2 near 1e-5 (FX-09).
  const design = Array.from({length: 40}, () => {
    const first = 2 * random() - 1;
    return [first, first + 3e-3 * (2 * random() - 1), 2 * random() - 1];
  });
  const collinear = multiply(transpose(design), design);
  const vector = [1, -1, 0.5, 2];
  const cases: SolveCase[] = [
    {
      name: 'NaN entry',
      matrix: withEntry(2, 1, NaN),
      vector,
      mode: 1,
      minimumPivotRatio: 0,
      expectedStatus: DENSE_SOLVE_STATUS.NONFINITE
    },
    {
      name: 'infinite diagonal',
      matrix: withEntry(3, 3, Infinity),
      vector,
      mode: 0,
      minimumPivotRatio: 0,
      expectedStatus: DENSE_SOLVE_STATUS.NONFINITE
    },
    {
      name: 'zero diagonal',
      matrix: withEntry(0, 0, 0),
      vector,
      mode: 1,
      minimumPivotRatio: 0,
      expectedStatus: DENSE_SOLVE_STATUS.SINGULAR
    },
    {
      name: 'indefinite',
      matrix: [
        [1, 2],
        [2, 1]
      ],
      vector: [1, 1],
      mode: 0,
      minimumPivotRatio: 0,
      expectedStatus: DENSE_SOLVE_STATUS.SINGULAR
    },
    {
      name: 'rank deficient',
      matrix: [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
      ],
      vector: [1, 1, 1],
      mode: 1,
      minimumPivotRatio: 1e-6,
      expectedStatus: [DENSE_SOLVE_STATUS.SINGULAR, DENSE_SOLVE_STATUS.ILL_CONDITIONED]
    },
    {
      name: 'nearly collinear, limit 1e-3',
      matrix: collinear,
      vector: [1, 2, 3],
      mode: 1,
      minimumPivotRatio: 1e-3,
      expectedStatus: DENSE_SOLVE_STATUS.ILL_CONDITIONED
    },
    {
      name: 'nearly collinear, no limit',
      matrix: collinear,
      vector: [1, 2, 3],
      mode: 1,
      minimumPivotRatio: 0,
      expectedStatus: DENSE_SOLVE_STATUS.OK
    },
    {
      name: 'NaN outside the leading block',
      matrix: base.slice(0, 2).map(row => row.slice(0, 2)),
      vector: [1, 2],
      mode: 1,
      minimumPivotRatio: 1e-6,
      expectedStatus: DENSE_SOLVE_STATUS.OK
    }
  ];
  const results = await runSolveCases('cholesky', cases);
  checkStatuses(cases, results);
  const collinearRatio = results[5].ratio;
  expect(collinearRatio).toBeGreaterThan(1e-7);
  expect(collinearRatio).toBeLessThan(1e-3);
  // The unflagged run still solves accurately for its conditioning (float32, cond near 1e5).
  checkSolutions(cases.slice(6), results.slice(6), 5e-2);
});

it('dense LU solves, inverts and takes signed determinants of general systems', async () => {
  const random = createSeededRandom(21);
  const cases: SolveCase[] = [];
  for (let size = 1; size <= CAPACITY; size++) {
    for (const mode of [0, 4]) {
      for (let repeat = 0; repeat < 4; repeat++) {
        const matrix = createGeneral(random, size);
        const rowScales = Array.from({length: size}, () => 10 ** (8 * random() - 4));
        cases.push({
          name: `general size ${size} passes ${mode} repeat ${repeat}`,
          // Rows spanning eight decades need the Ruiz passes.
          matrix: mode
            ? matrix.map((row, rowIndex) => row.map(value => value * rowScales[rowIndex]))
            : matrix,
          vector: Array.from({length: size}, () => 2 * random() - 1),
          mode,
          pivotTolerance: 1e-6,
          minimumPivotRatio: 1e-4,
          expectedStatus: DENSE_SOLVE_STATUS.OK
        });
      }
    }
  }
  cases.push({
    name: 'zero leading entry needs a row swap',
    matrix: [
      [0, 2, 1],
      [1, 1, 0],
      [3, 0, 1]
    ],
    vector: [1, 2, 3],
    mode: 0,
    minimumPivotRatio: 0,
    expectedStatus: DENSE_SOLVE_STATUS.OK
  });
  const results = await runSolveCases('lu', cases);
  checkStatuses(cases, results);
  checkSolutions(cases, results, 2e-4);
});

it('dense LU reports non-finite, singular and ill-conditioned systems', async () => {
  const hilbert = Array.from({length: 6}, (_, row) =>
    Array.from({length: 6}, (_, column) => 1 / (row + column + 1))
  );
  const vector = [1, 2, 3, 4, 5, 6];
  const cases: SolveCase[] = [
    {
      name: 'NaN entry',
      matrix: [
        [1, 2],
        [NaN, 4]
      ],
      vector: [1, 1],
      mode: 4,
      minimumPivotRatio: 0,
      expectedStatus: DENSE_SOLVE_STATUS.NONFINITE
    },
    {
      name: 'duplicate rows',
      matrix: [
        [1, 2, 3],
        [1, 2, 3],
        [0, 1, 5]
      ],
      vector: [1, 1, 1],
      mode: 0,
      pivotTolerance: 1e-6,
      minimumPivotRatio: 0,
      expectedStatus: DENSE_SOLVE_STATUS.SINGULAR
    },
    {
      name: 'zero row, equilibrated',
      matrix: [
        [1, 2],
        [0, 0]
      ],
      vector: [1, 1],
      mode: 4,
      minimumPivotRatio: 0,
      expectedStatus: DENSE_SOLVE_STATUS.SINGULAR
    },
    {
      name: 'zero row, unscaled',
      matrix: [
        [1, 2],
        [0, 0]
      ],
      vector: [1, 1],
      mode: 0,
      minimumPivotRatio: 0,
      expectedStatus: DENSE_SOLVE_STATUS.SINGULAR
    },
    {
      name: 'Hilbert 6, ratio limit 1e-4',
      matrix: hilbert,
      vector,
      mode: 4,
      minimumPivotRatio: 1e-4,
      expectedStatus: DENSE_SOLVE_STATUS.ILL_CONDITIONED
    },
    {
      name: 'Hilbert 4, no limit',
      matrix: hilbert.slice(0, 4).map(row => row.slice(0, 4)),
      vector: vector.slice(0, 4),
      mode: 4,
      minimumPivotRatio: 0,
      expectedStatus: DENSE_SOLVE_STATUS.OK
    }
  ];
  const results = await runSolveCases('lu', cases);
  checkStatuses(cases, results);
  // Hilbert 4 has condition 1.6e4: float32 keeps about three digits.
  checkSolutions(cases.slice(5), results.slice(5), 2e-2);
});
