// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getGPUTimeWindowFilterParameterValues,
  splitTimestamps
} from '../../../src/dataframe-analysis/time-window-filter/index';

it('t1 sorted empty-window fixture uses exact dyadic endpoints between half-integer rows', () => {
  // 4.25 = 17/4 and 4.375 = 35/8: neither is a half-integer source value.
  expect(Array.from(getGPUTimeWindowFilterParameterValues({start: 4.25, end: 4.375}))).toEqual([
    4.25, 0, 4.375, 0, 0, 0, 0, 0
  ]);
  expect(() => getGPUTimeWindowFilterParameterValues({start: 4.25, end: 4.4})).toThrow(
    /exactly representable/
  );
});

it('getGPUTimeWindowFilterParameterValues packs high and low window parts', () => {
  expect(Array.from(getGPUTimeWindowFilterParameterValues({start: 10, end: 20}))).toEqual([
    10, 0, 20, 0, 0, 0, 0, 0
  ]);
  const start = 1_700_000_015;
  const values = getGPUTimeWindowFilterParameterValues({
    start,
    end: start + 1,
    startFadeDuration: 2
  });
  expect(values[0]).toBe(Math.fround(start));
  expect(values[0] + values[1]).toBe(start);
  expect(values[4]).toBe(2);

  const target = new Float32Array(8);
  expect(getGPUTimeWindowFilterParameterValues({start: 1, end: 2}, target)).toBe(target);
  expect(() =>
    getGPUTimeWindowFilterParameterValues({start: 1, end: 2}, new Float32Array(4))
  ).toThrow();
  expect(() =>
    getGPUTimeWindowFilterParameterValues({start: 1, end: 2, endFadeDuration: -1})
  ).toThrow();
  expect(() => getGPUTimeWindowFilterParameterValues({start: Number.NaN, end: 2})).toThrow();
});

it('splitTimestamps splits values into exact float32 pairs', () => {
  const values = [1_700_000_010, 0.5];
  const {high, low} = splitTimestamps(values);
  for (let index = 0; index < values.length; index++) {
    expect(high[index] + low[index]).toBe(values[index]);
    expect(Math.fround(high[index])).toBe(high[index]);
  }
  expect(low[1]).toBe(0);
});

it('DPC-108: validates packed float time lanes and preserves target on rejection', () => {
  // IEEE binary32: 1e40 -> Infinity, 1e-50 -> 0. Near 1 + 2^-25, adjacent doubles
  // have equal high and rounded-low lanes, collapsing a strictly ordered window.
  const adjacent = 1 + 2 ** -25;
  const invalid = [
    {start: 1e40, end: 1e40},
    {start: -1e40, end: 0},
    {start: adjacent, end: adjacent + 2 ** -52},
    {start: adjacent + 2 ** -52, end: adjacent},
    ...(['startFadeDuration', 'endFadeDuration'] as const).flatMap(name =>
      [1e-50, 1e40].map(value => ({start: 0, end: 1, [name]: value}))
    )
  ];
  for (const settings of invalid) {
    const target = new Float32Array(10).fill(17);
    expect(() => getGPUTimeWindowFilterParameterValues(settings, target)).toThrow();
    expect(Array.from(target)).toEqual(new Array(10).fill(17));
  }
  for (const [start, end] of [
    [1, 1],
    [2, 1],
    [1, 2]
  ]) {
    const target = new Float32Array(10).fill(17);
    expect(
      getGPUTimeWindowFilterParameterValues(
        {start, end, startFadeDuration: 0, endFadeDuration: 0},
        target
      )
    ).toBe(target);
    expect(Array.from(target)).toEqual([start, 0, end, 0, 0, 0, 0, 0, 17, 17]);
  }
});

it('B3: rejects rounding across either inclusive float boundary before writing the target', () => {
  // Row 1+2^-25 is exactly representable as two f32 lanes (1,2^-25).
  // End just below it must exclude it; start just above it must exclude it too.
  const row = 1 + 2 ** -25;
  for (const settings of [
    {start: 0, end: row - 2 ** -52},
    {start: row + 2 ** -52, end: 2}
  ]) {
    const target = new Float32Array(8).fill(17);
    expect(() => getGPUTimeWindowFilterParameterValues(settings, target)).toThrow();
    expect(Array.from(target)).toEqual(new Array(8).fill(17));
  }
  expect(
    Array.from(getGPUTimeWindowFilterParameterValues({start: row, end: row})).slice(0, 4)
  ).toEqual([1, 2 ** -25, 1, 2 ** -25]);
});

it('strict timestamp splitting rejects the first rounded row (PCK-06)', () => {
  // 2^-25 fits the low f32 lane; the additional binary64 ulp does not.
  const end = 1 + 2 ** -25;
  expect(splitTimestamps([end], {requireExact: true})).toEqual(splitTimestamps([end]));
  expect(() => splitTimestamps([end, end + Number.EPSILON], {requireExact: true})).toThrow(
    /row 1.*exactly representable/
  );
  expect(splitTimestamps([end + Number.EPSILON])).toEqual(splitTimestamps([end]));
  for (const value of [NaN, Infinity, 1e100]) {
    expect(() => splitTimestamps([value], {requireExact: true})).toThrow(/row 0/);
  }
});
