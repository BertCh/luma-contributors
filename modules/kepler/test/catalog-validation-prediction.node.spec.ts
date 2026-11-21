// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  summarizeConformalStatuses,
  VALIDATION_PREDICTION_BINDINGS
} from '../src/analysis/catalog-validation-prediction';

test('geographic conformal summary retains independent truncation and interval status bits', () => {
  const words = Uint32Array.of(16, 17, 8);
  expect(summarizeConformalStatuses(words, true)).toEqual({
    unbounded_rows: 1,
    invalid_query_rows: 0,
    no_calibration_rows: 1,
    neighbors_truncated_rows: 2,
    input_incomplete_rows: 0
  });
  expect(Array.from(words)).toEqual([16, 17, 8]);
  expect(summarizeConformalStatuses(words, false)).toEqual({
    unbounded_rows: 1,
    invalid_query_rows: 0,
    no_calibration_rows: 1
  });
});

test('geographic conformal reading legend distinguishes input loss from internal truncation', () => {
  const binding = VALIDATION_PREDICTION_BINDINGS.find(
    entry => entry.id === 'validation-geo-conformal'
  )!;
  const legend = binding.declaredBehavior.join(' ');
  expect(legend).toContain('32 input incomplete');
  expect(legend).toContain('input_incomplete_rows');
  expect(legend).toContain('independent');
  expect(legend).toContain('16 neighbors truncated');
});

test('geographic conformal summary names incomplete input alongside independent row conditions', () => {
  const words = Uint32Array.of(3, 36, 18, 48, 0);
  // 36=32|4, 18=16|2, 48=32|16: two incomplete, two truncated, two unbounded, one invalid.
  expect(summarizeConformalStatuses(words, true)).toEqual({
    unbounded_rows: 2,
    invalid_query_rows: 1,
    no_calibration_rows: 0,
    neighbors_truncated_rows: 2,
    input_incomplete_rows: 2
  });
  expect(Array.from(words)).toEqual([3, 36, 18, 48, 0]);
});
