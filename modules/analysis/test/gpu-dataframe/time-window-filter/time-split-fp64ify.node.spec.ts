// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fp64ify} from '@luma.gl/shaders';
import {expect, it} from 'vitest';
import {
  getGPUTimeWindowFilterParameterValues,
  splitTimestamps
} from '../../../src/dataframe-analysis/time-window-filter/index';

/** The split the time window filter relied on before it used the published `fp64ify`. */
function splitReference(value: number): [number, number] {
  const high = Math.fround(value);
  return [high, Math.fround(value - high)];
}

function getSampleValues(): number[] {
  const values: number[] = [0, -0, 1, -1, 0.1, -0.1, 1e-45, 3.4e38, -3.4e38, 1e300, -1e300];
  values.push(Infinity, -Infinity, NaN, Number.MAX_VALUE, Number.MIN_VALUE, 2 ** 24 + 1, 2 ** 53);
  let state = 12345;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  for (let index = 0; index < 4000; index++) {
    values.push(1_700_000_000 + next() * 1e6);
    values.push(-(next() * 1e12));
    values.push(1_700_000_000_000 + Math.floor(next() * 1e9));
    // Near float32 rounding boundaries: halfway points between neighboring float32 values.
    const base = Math.fround(next() * 2 ** (1 + Math.floor(next() * 60)));
    const spacing = Math.fround(base * 2 ** -23);
    values.push(base + spacing / 2, base + spacing / 2 + spacing * 2 ** -30, base - spacing / 2);
  }
  return values;
}

it('fp64ify stored into Float32Array gives the same pair as the reference split', () => {
  const published = new Float32Array(2);
  const reference = new Float32Array(2);
  for (const value of getSampleValues()) {
    published.set(fp64ify(value));
    reference.set(splitReference(value));
    expect(Array.from(new Uint32Array(published.buffer))).toEqual(
      Array.from(new Uint32Array(reference.buffer))
    );
  }
});

it('splitTimestamps covers every sample and strict packing admits exactly lossless endpoint pairs', () => {
  const values = getSampleValues();
  const {high, low} = splitTimestamps(values);
  const highWords = new Uint32Array(high.buffer, high.byteOffset, high.length);
  const lowWords = new Uint32Array(low.buffer, low.byteOffset, low.length);
  const expected = new Float32Array(2);
  const expectedWords = new Uint32Array(expected.buffer);
  const parameters = new Float32Array(8);
  const parameterWords = new Uint32Array(parameters.buffer, parameters.byteOffset, 8);
  const expectedParameters = new Float32Array(8);
  const expectedParameterWords = new Uint32Array(expectedParameters.buffer);
  for (const [index, value] of values.entries()) {
    expected.set(splitReference(value));
    expect(highWords[index]).toBe(expectedWords[0]);
    expect(lowWords[index]).toBe(expectedWords[1]);
    // Use an exactly representable control in both endpoint positions: a start-only check
    // must not certify a rounded end. Checking the whole target also pins fade/reserved zeros.
    const exact = 1_700_000_000.125;
    for (const [start, end] of [
      [value, exact],
      [exact, value]
    ]) {
      expectedParameters.fill(0);
      expectedParameters.set(splitReference(start), 0);
      expectedParameters.set(splitReference(end), 2);
      const representable = [start, end].every(endpoint => {
        const highLane = Math.fround(endpoint);
        const residual = endpoint - highLane;
        const lowLane = Math.fround(residual);
        return Number.isFinite(highLane) && Number.isFinite(lowLane) && lowLane === residual;
      });
      parameters.fill(123.25);
      const before = Array.from(parameterWords);
      if (representable) {
        getGPUTimeWindowFilterParameterValues({start, end}, parameters);
        expect(Array.from(parameterWords)).toEqual(Array.from(expectedParameterWords));
      } else {
        expect(() => getGPUTimeWindowFilterParameterValues({start, end}, parameters)).toThrow();
        expect(Array.from(parameterWords)).toEqual(before);
      }
    }
  }
});

it('DPC-108 exactly representable equal and reversed windows remain valid', () => {
  for (const [start, end] of [
    [1.25, 1.25],
    [2.5, -1.25]
  ]) {
    expect(Array.from(getGPUTimeWindowFilterParameterValues({start, end}))).toEqual([
      start,
      0,
      end,
      0,
      0,
      0,
      0,
      0
    ]);
  }
});
