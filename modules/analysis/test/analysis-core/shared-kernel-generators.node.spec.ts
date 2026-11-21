// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createHash} from 'node:crypto';
import {
  getEquilibratedCholeskyWGSL,
  EQUILIBRATED_CHOLESKY_MAXIMUM_SOLVE_SIZE
} from '../../src/analysis-core/kernels/dense-solve-wgsl';
import {getEquilibratedCholeskyWGSL as getRegressionEquilibratedCholeskyWGSL} from '../../src/gpu-spatial-analysis/spatial-regression/local-weighted-least-squares-wgsl';
import {GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE} from '../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-parameters';
import {COMPENSATED_SUM_WGSL} from '../../src/analysis-core/kernels/compensated-sum-wgsl';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../src/analysis-core/kernels/addition-rounding-error-wgsl';
import {getCholeskyWGSL} from '../../src/gpu-spatial-analysis/spatial-regression/spatial-regression-solve';
import {getWGSLModuleUsage} from '../../src/testing/wgsl-memory-usage';

// SHA-256 of pre-extraction output from baseline 85a29fcc, independent of either current wrapper.
// Reproduction: own scratch/freeze-equilibrated.mjs reads both functions with git show and
// evaluates only those baseline functions. These digests stay fixed after family delegation.
const BASELINE_SOURCE_DIGESTS = [
  [1, 'd2745f43330c91b28d71cb21fe18519f666cebc44b2ec7d7c990f8067f815e0d'],
  [3, '838f8bc4ceffb54f81065de70035a95db09148a9e8aa3bf36fbddabe55c5af9d'],
  [8, 'a98712bb0532a8d12d4d3df10d748cd96e960f7a6531d5a1722a8501ab0393f5'],
  [12, 'd6c3fd401f0abb47c61168b1a401105b3d071af9176a984a31209ed2f1a57583'],
  [25, '198297db5a14282492e87c8e5075b735185b2561f4d822add4ccf073637d8dd6'],
  [44, '0587990a247ef025346c8b45dc05bba634b26887460c74301ec7d77239e3bead'],
  [45, 'd475ffe68be175d366ade3dc7bd986c95b8dd54feff61d7f03512a3feeab9982']
] as const;

it.each(
  BASELINE_SOURCE_DIGESTS
)('review S3 size %i matches independent pre-extraction source', (size, digest) => {
  expect(createHash('sha256').update(getEquilibratedCholeskyWGSL(size, 1e-3)).digest('hex')).toBe(
    digest
  );
});

function completeCaller(size: number): string {
  return `${getCholeskyWGSL(size, 45)}\n${getEquilibratedCholeskyWGSL(size, 1e-3)}
@compute @workgroup_size(1) fn main() {
  var matrix: array<f32, ${size * size}>;
  var scale: array<f32, ${size}>;
  var rhs: array<f32, ${size}>;
  if (factorEquilibrated_${size}(&matrix, &scale)) {
    solveEquilibrated_${size}(&matrix, &scale, &rhs);
    _ = inverseDiagonalEquilibrated_${size}(&matrix, &scale, 0u);
  }
}`;
}

it('review S3 complete-solve size distinguishes aggregate storage from a single array', () => {
  expect(EQUILIBRATED_CHOLESKY_MAXIMUM_SOLVE_SIZE).toBe(44);
  const supported = getWGSLModuleUsage(completeCaller(EQUILIBRATED_CHOLESKY_MAXIMUM_SOLVE_SIZE));
  expect(supported.functionBytes).toBe(8096); // 4 * (44^2 + 44 + 44)
  expect(supported.functionBytes).toBeLessThanOrEqual(8192);
  expect(getWGSLModuleUsage(completeCaller(45)).functionBytes).toBe(8460); // 4 * (45^2 + 45 + 45)
});

it.each([
  1, 3, 8, 12, 25, 45
])('SSR-126 preserves the regression generator byte for byte at size %i', size => {
  expect(getEquilibratedCholeskyWGSL(size, GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE)).toBe(
    getRegressionEquilibratedCholeskyWGSL(size)
  );
  const source = getEquilibratedCholeskyWGSL(size, 0.125);
  // The caller passes a factor-pivot tolerance; the dense matrix-pivot limit is its square.
  expect(source).toContain(`factorDenseCholesky_${size}(a, ${size}u, true, 0.015625)`);
  expect(source).toContain(`fn solveEquilibrated_${size}`);
  expect(source).toContain(`fn inverseDiagonalEquilibrated_${size}`);
  expect(size * size * 4).toBeLessThanOrEqual(8192);
});

it('SSR-126 refuses unsupported function storage and invalid pivot tolerances', () => {
  for (const size of [0, -1, 1.5, 46, 64])
    expect(() => getEquilibratedCholeskyWGSL(size, 0.001)).toThrow(/size/);
  for (const tolerance of [-1, NaN, Infinity, 1e30]) {
    expect(() => getEquilibratedCholeskyWGSL(3, tolerance)).toThrow(/pivotTolerance/);
  }
});

it('GEO-131 shares integer error recovery and preserves sum then correction evaluation', () => {
  expect(COMPENSATED_SUM_WGSL.startsWith(ADDITION_ROUNDING_ERROR_WGSL)).toBe(true);
  expect(COMPENSATED_SUM_WGSL).toContain(`let previous = accumulator.x;
  let total = previous + value;
  let correction = accumulator.y + getAdditionRoundingError(previous, value, total);
  return vec2<f32>(total, correction);`);
  expect(COMPENSATED_SUM_WGSL.match(/fn getAdditionRoundingError\(/g)).toHaveLength(1);
  expect(COMPENSATED_SUM_WGSL).toContain(
    '*accumulator = addCompensatedValue(*accumulator, value);'
  );
});
