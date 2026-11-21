// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {readRegressionFloat32Word, readRegressionUint32Word} from './regression-example-readback';

test('GPU fixture scalar reads reinterpret byte slices including their offsets', () => {
  const bytes = new Uint8Array(12);
  const view = new DataView(bytes.buffer);
  view.setFloat32(4, 1, true);
  expect(readRegressionFloat32Word(bytes.subarray(4, 8))).toBe(1);
  view.setFloat32(4, NaN, true);
  expect(Number.isNaN(readRegressionFloat32Word(bytes.subarray(4, 8)))).toBe(true);
  view.setUint32(4, 0x12345678, true);
  expect(readRegressionUint32Word(bytes.subarray(4, 8))).toBe(0x12345678);
});

test('explorer GPU recovery assertions accept the empty-matrix scaling norm and require zero count', () => {
  const source = readFileSync(
    'modules/kepler/test/regression-example-certificates.spec.ts',
    'utf8'
  );
  const start = source.indexOf('      if (radius > 1)');
  const end = source.indexOf('\n    }\n  } finally', start);
  const check = new Function(
    'expect',
    'radius',
    'value',
    'requiredCount',
    source.slice(start, end)
  );
  // The traces producer defines the scaling norm of the empty matrix as 1.
  expect(() => check(expect, 0.1, 1, 0)).not.toThrow();
  expect(() => check(expect, 0.1, 1, 1)).toThrow();
});
