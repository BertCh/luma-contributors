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

it('splitTimestamps and the parameter packing give the reference pair bit for bit', () => {
  const values = getSampleValues();
  const {high, low} = splitTimestamps(values);
  const highWords = new Uint32Array(high.buffer, high.byteOffset, high.length);
  const lowWords = new Uint32Array(low.buffer, low.byteOffset, low.length);
  const expected = new Float32Array(2);
  const expectedWords = new Uint32Array(expected.buffer);
  const parameters = new Float32Array(8);
  const parameterWords = new Uint32Array(parameters.buffer, parameters.byteOffset, 4);
  const expectedParameters = new Float32Array(4);
  const expectedParameterWords = new Uint32Array(expectedParameters.buffer);
  for (const [index, value] of values.entries()) {
    expected.set(splitReference(value));
    expect(highWords[index]).toBe(expectedWords[0]);
    expect(lowWords[index]).toBe(expectedWords[1]);
    if (!Number.isFinite(value)) continue;
    const end = value === 0 ? 1_700_000_000.125 : -value;
    expectedParameters.set(splitReference(value), 0);
    expectedParameters.set(splitReference(end), 2);
    getGPUTimeWindowFilterParameterValues({start: value, end}, parameters);
    expect(Array.from(parameterWords)).toEqual(Array.from(expectedParameterWords));
  }
});
