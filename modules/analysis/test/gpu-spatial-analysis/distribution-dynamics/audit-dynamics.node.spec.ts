// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeTransitionDiagnosticsOnCPU} from '../../../src/gpu-spatial-analysis/distribution-dynamics/cpu-transition-diagnostics-operation';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {
  AUDIT_SUPPORT_CASES,
  AUDIT_TRANSITIONS,
  AUDIT_PARALLEL_DIAGNOSTICS
} from './audit-dynamics-fixture';

it('retains near-identical regime evidence below one float32 count ulp', () => {
  // 80-digit Decimal centered deviance; see audit-review.oracle.py.
  const actual = computeTransitionDiagnosticsOnCPU(
    Uint32Array.of(4721, 4722, 0, 0, 4720, 4721, 0, 0),
    {classCount: 2, conditionCount: 2}
  );
  expectNumericArrayClose([actual.homogeneity[2]], [2.375953939555561e-12], {
    relativeTolerance: 2e-6
  });
  expectNumericArrayClose([actual.homogeneity[3]], [0.9999987701303906], {absoluteTolerance: 1e-7});
});

it('audits every CPU diagnostic output against independent SciPy literals', () => {
  for (const expected of AUDIT_TRANSITIONS) {
    const actual = computeTransitionDiagnosticsOnCPU(Uint32Array.from(expected.counts), {
      classCount: 3,
      conditionCount: 2
    });
    for (const port of ['homogeneity', 'steadyStates', 'stateSparsity', 'converged'] as const) {
      // Float64 calculation rounded once into the public float32 output.
      expectNumericArrayClose(actual[port], expected[port], {
        absoluteTolerance: 1e-7,
        relativeTolerance: 1e-7,
        label: `${port}, lag ${expected.periodLag}`
      });
    }
  }
});

it('rejects reducible active chains independently of state numbering', () => {
  for (const expected of AUDIT_SUPPORT_CASES) {
    const actual = computeTransitionDiagnosticsOnCPU(Uint32Array.from(expected.counts), {
      classCount: 2
    });
    expectNumericArrayClose(actual.homogeneity, [0, NaN, 0, NaN, 0]);
    expectNumericArrayClose(actual.steadyStates, expected.steadyStates);
    expect(Array.from(actual.stateSparsity)).toEqual(expected.stateSparsity);
    expect(Array.from(actual.converged)).toEqual(expected.converged);
  }
});

it('checks the CPU form against the same 257-condition SciPy oracle as the parallel graph', () => {
  const expected = AUDIT_PARALLEL_DIAGNOSTICS;
  const actual = computeTransitionDiagnosticsOnCPU(Uint32Array.from(expected.counts), {
    classCount: 3,
    conditionCount: 257
  });
  for (const port of ['homogeneity', 'steadyStates', 'stateSparsity', 'converged'] as const) {
    expectNumericArrayClose(actual[port], expected[port], {
      absoluteTolerance: 0,
      relativeTolerance: 1e-7,
      label: port
    });
  }
});

it('retains the integer residual at 2^27 in near-independent regimes', () => {
  // Decimal(100): 2*sum(n*ln(n*N/(rowTotal*columnTotal))) for [[k,k+1],[k-1,k]].
  // LR=1.0339757656912846e-25, df=1; sf=0.9999999999997434 (rounds to 1 in f32).
  const count = 2 ** 27;
  const result = computeTransitionDiagnosticsOnCPU(
    Uint32Array.of(count, count + 1, 0, 0, count - 1, count, 0, 0),
    {classCount: 2, conditionCount: 2}
  );
  expect(Number.isFinite(result.homogeneity[2])).toBe(true);
  expect(result.homogeneity[2]).toBeGreaterThan(0);
  expectNumericArrayClose([result.homogeneity[2]], [1.0339757656912846e-25], {
    relativeTolerance: 2e-6
  });
  // Q equals LR to leading order here (Decimal: 1.0339757656912846e-25); both tails round to one.
  expect(result.homogeneity[0]).toBeGreaterThan(0);
  expectNumericArrayClose([result.homogeneity[0]], [1.0339757656912846e-25], {
    relativeTolerance: 2e-6
  });
  expect(result.homogeneity[1]).toBe(1);
  expect(result.homogeneity[3]).toBe(1);
  expect(result.homogeneity[4]).toBe(1);
});
