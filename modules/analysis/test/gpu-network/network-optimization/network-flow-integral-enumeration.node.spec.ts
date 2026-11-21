// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {expectEnumeratedIntegralFlow} from './network-flow-integral-enumeration.test-utils';

function createLiteralResult(required: number) {
  // Independently hand-worked paths: finite 0-1-3-5 costs 1; maximum reroutes via
  // 0-1-4-5 and 0-2-3-5, cost 4. Float bits are exactly 1.0 and 4.0.
  return required === 1
    ? {
        flow: [1, 0, 1, 0, 0, 0, 1, 0],
        cut: [1, 1, 1, 1, 1, 1],
        summary: [1, 1, 1, 0, 1, 0x3f800000, 0, 0],
        exactCost: [1, 0]
      }
    : {
        flow: [1, 1, 0, 1, 1, 0, 1, 1],
        cut: [1, 0, 0, 0, 0, 0],
        summary: [1, 1, 2, 0, 2, 0x40800000, 2, 1],
        exactCost: [4, 0]
      };
}

for (const required of [1, 0xffffffff]) {
  it(`B2/B3 accepts the hand-worked result for request ${required}`, () => {
    expectEnumeratedIntegralFlow(createLiteralResult(required), required);
  });
  for (const bits of [0x7fc00000, 0x7f800000, 0])
    it(`B2 rejects public objective bits ${bits} for request ${required}`, () => {
      const result = createLiteralResult(required);
      result.summary[5] = bits;
      expect(() => expectEnumeratedIntegralFlow(result, required)).toThrow();
    });
  for (const length of [7, 9])
    it(`B2 rejects ${length} summary words for request ${required}`, () => {
      const result = createLiteralResult(required);
      result.summary.length = length;
      if (length === 9) result.summary[8] = 0;
      expect(() => expectEnumeratedIntegralFlow(result, required)).toThrow();
    });
  for (const shape of ['empty', 'hole', 'extra', 'nan', 'nonbinary'] as const)
    it(`B3 rejects ${shape} cut rows for request ${required}`, () => {
      const result = createLiteralResult(required);
      if (shape === 'empty') result.cut = [];
      if (shape === 'hole') delete result.cut[2];
      if (shape === 'extra') result.cut.push(0);
      if (shape === 'nan') result.cut[2] = NaN;
      if (shape === 'nonbinary') result.cut[2] = 2;
      expect(() => expectEnumeratedIntegralFlow(result, required)).toThrow();
    });
}

it('B3 rejects a capacity-valid partition that is not residual source reachability', () => {
  const result = createLiteralResult(0xffffffff);
  result.cut = [1, 1, 1, 1, 1, 0];
  expect(() => expectEnumeratedIntegralFlow(result, 0xffffffff)).toThrow();
});
