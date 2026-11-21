// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

// Two independent y=1 observations with sill=1, nugget=.1 and zero measurement error:
// L=-log(2*pi*1.1)-1/1.1. For log parameters, a=dv/dtheta=[1,0,.1].
// Summed score=-(a/v)*(1-1/v), information=a*a'/v^2, v=1.1.
// These closed forms do not call a Vecchia evaluator or share its numerical implementation.
export const VECCHIA_MARGINAL_EXPECTED = [
  -2.842278155304579,
  -10 / 121,
  0,
  -1 / 121,
  100 / 121,
  0,
  10 / 121,
  0,
  0,
  1 / 121
] as const;

export function expectVecchiaMarginalTotals(values: ArrayLike<number>, fisher: boolean): void {
  expect(values.length).toBe(fisher ? 20 : 2);
  const cellCount = fisher ? 10 : 1;
  for (let cell = 0; cell < cellCount; cell++) {
    const actual = values[cell * 2] + values[cell * 2 + 1];
    expect(Number.isFinite(actual), `finite recovered cell ${cell}`).toBe(true);
    expect(actual, `recovered cell ${cell}`).toBeCloseTo(VECCHIA_MARGINAL_EXPECTED[cell], 5);
  }
}

export function expectVecchiaSuppressedTotals(values: ArrayLike<number>, fisher: boolean): void {
  expect(values.length).toBe(fisher ? 20 : 2);
  for (let component = 0; component < values.length; component++)
    expect(Number.isNaN(values[component])).toBe(true);
}
