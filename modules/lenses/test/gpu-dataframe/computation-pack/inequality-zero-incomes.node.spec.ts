// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {computeInequalityOnCPU} from '../../../../analysis/src/dataframe-analysis/composite-indicators/cpu-inequality-operation';
import {getGPUInequalityParameterValues} from '../../../../analysis/src/dataframe-analysis/composite-indicators/inequality-parameters';
import {INDICATORS_OPERATORS} from '../../../src/gpu-dataframe/computation-pack/operators/indicators';

test.each(
  [
    {label: 'unit weights', incomes: [0, 2], weights: [1, 1]},
    {label: 'unequal weights', incomes: [4, 0], weights: [1, 3]}
  ].flatMap(input => [1, 1.5, 2].map(epsilon => ({...input, epsilon})))
)('NUM-09: epsilon $epsilon extends Atkinson to one with $label', ({epsilon, incomes, weights}) => {
  const result = computeInequalityOnCPU({
    values: Float32Array.of(...incomes, 0, 0),
    zoneIds: Uint32Array.of(0, 0, 1, 1),
    weights: Float32Array.of(...weights, 1, 1),
    zoneCount: 2,
    parameters: getGPUInequalityParameterValues({epsilon})
  });
  // Both populations have mean 1. Replacing zero by delta makes the equally distributed
  // equivalent tend to 0 as delta -> 0+ at epsilon >= 1, hence A = 1 - EDE/mean -> 1.
  // For [0,0], mean=0, so the normalization is undefined. Theil L keeps its NaN policy.
  expect(result.atkinson).toHaveLength(2);
  expect(result.theilL).toHaveLength(2);
  expect(Number.isNaN(result.atkinson[1])).toBe(true);
  expect(Number.isNaN(result.theilL[0])).toBe(true);
  expect(Number.isFinite(result.atkinson[0])).toBe(true);
  expect(result.atkinson[0]).toBe(1);
});

test('NUM-09: the rendered inequality formula distinguishes Atkinson from Theil L', () => {
  const formula = INDICATORS_OPERATORS['GPUInequality'].formula;
  if ('formula' in formula) throw new Error('Inequality requires a curated formula');
  expect(formula.ascii).toContain('Atkinson = 1');
  expect(formula.ascii).toContain('epsilon >= 1');
  expect(formula.ascii).toContain('all-zero population remains undefined');
  expect(formula.ascii).toContain('Theil L remains NaN');
  expect(formula.tex).toContain('A_\\epsilon=1');
});
