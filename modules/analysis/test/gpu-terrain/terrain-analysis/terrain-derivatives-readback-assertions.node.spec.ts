// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {expectDerivativeArraysClose} from './terrain-derivatives-readback-assertions';

it('rejects impossible angles before comparing circular distance', () => {
  for (const actual of [810, -270, Infinity, -Infinity, NaN, 360]) {
    expect(() => expectDerivativeArraysClose([actual], [90], true)).toThrow();
  }
  expect(() => expectDerivativeArraysClose([359], [-1], true)).toThrow();
  expect(() => expectDerivativeArraysClose([-1], [-1], true)).not.toThrow();
  expect(() => expectDerivativeArraysClose([359.99999], [0], true)).not.toThrow();
  expect(() => expectDerivativeArraysClose([90], [90], true)).not.toThrow();
});
