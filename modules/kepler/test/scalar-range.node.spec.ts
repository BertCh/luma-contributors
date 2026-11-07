// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {normalizeKeplerScalarRange} from '../src/selection/scalar-range';

it('normalizes fractional and out-of-storage integer slider endpoints exactly', () => {
  expect(normalizeKeplerScalarRange('uint32', 0.2, 1.8)).toEqual({
    minimum: 1,
    maximum: 1,
    empty: false
  });
  expect(normalizeKeplerScalarRange('uint32', -100, 2.9)).toEqual({
    minimum: 0,
    maximum: 2,
    empty: false
  });
  expect(normalizeKeplerScalarRange('sint32', -1.8, 1.8)).toEqual({
    minimum: -1,
    maximum: 1,
    empty: false
  });
  expect(normalizeKeplerScalarRange('uint32', -10, -1).empty).toBe(true);
  expect(normalizeKeplerScalarRange('uint32', 0.1, 0.9).empty).toBe(true);
  expect(normalizeKeplerScalarRange('sint32', 0x80000000, Number.MAX_VALUE).empty).toBe(true);
  expect(normalizeKeplerScalarRange('uint32', 0, Number.MAX_VALUE)).toEqual({
    minimum: 0,
    maximum: 0xffffffff,
    empty: false
  });
  expect(normalizeKeplerScalarRange('sint32', -Number.MAX_VALUE, Number.MAX_VALUE)).toEqual({
    minimum: -0x80000000,
    maximum: 0x7fffffff,
    empty: false
  });
  expect(normalizeKeplerScalarRange('uint32', 0, 1, true)).toEqual({
    minimum: 0,
    maximum: 0,
    empty: false
  });
  expect(normalizeKeplerScalarRange('uint32', 0, 1.5, true)).toEqual({
    minimum: 0,
    maximum: 1,
    empty: false
  });
});

it('uses directed Float32 boundaries without admitting an excluded rounded tie', () => {
  const nextAboveOne = 1 + 2 ** -23;
  const previousBelowOne = 1 - 2 ** -24;
  expect(normalizeKeplerScalarRange('float32', 1 + Number.EPSILON, 2)).toEqual({
    minimum: nextAboveOne,
    maximum: 2,
    empty: false
  });
  expect(normalizeKeplerScalarRange('float32', 0, 1 - Number.EPSILON)).toEqual({
    minimum: 0,
    maximum: previousBelowOne,
    empty: false
  });
  expect(normalizeKeplerScalarRange('float32', 1 + Number.EPSILON, 1 + Number.EPSILON).empty).toBe(
    true
  );
  expect(normalizeKeplerScalarRange('float32', 0, 1, true)).toEqual({
    minimum: 0,
    maximum: previousBelowOne,
    empty: false
  });
  expect(normalizeKeplerScalarRange('float32', 0, 1 + Number.EPSILON, true)).toEqual({
    minimum: 0,
    maximum: 1,
    empty: false
  });
  expect(normalizeKeplerScalarRange('float32', -1 + Number.EPSILON, 0)).toEqual({
    minimum: -previousBelowOne,
    maximum: 0,
    empty: false
  });
  expect(normalizeKeplerScalarRange('float32', -2, -1 - Number.EPSILON)).toEqual({
    minimum: -2,
    maximum: -nextAboveOne,
    empty: false
  });
});

it('handles finite Float32 limits and zero neighbors without infinite shader parameters', () => {
  const maximum = 3.4028234663852886e38;
  expect(normalizeKeplerScalarRange('float32', -Number.MAX_VALUE, Number.MAX_VALUE)).toEqual({
    minimum: -maximum,
    maximum,
    empty: false
  });
  expect(normalizeKeplerScalarRange('float32', maximum * 2, Number.MAX_VALUE).empty).toBe(true);
  expect(normalizeKeplerScalarRange('float32', -Number.MAX_VALUE, -maximum * 2).empty).toBe(true);
  expect(normalizeKeplerScalarRange('float32', Number.MIN_VALUE, 1).minimum).toBe(2 ** -149);
  expect(normalizeKeplerScalarRange('float32', -1, 0, true).maximum).toBe(-(2 ** -149));
  expect(normalizeKeplerScalarRange('float32', 0, 0, true).empty).toBe(true);
  expect(() => normalizeKeplerScalarRange('float32', Number.NaN, 1)).toThrow('finite');
  expect(() => normalizeKeplerScalarRange('uint32', 0, Infinity)).toThrow('finite');
  expect(() => normalizeKeplerScalarRange('sint32', 2, 1)).toThrow('Reversed');
});
