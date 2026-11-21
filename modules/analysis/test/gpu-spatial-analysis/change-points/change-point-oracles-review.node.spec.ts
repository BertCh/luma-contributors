// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {expect, it} from 'vitest';
import {computePettittReference} from './stx06-change-point-reference';
import {computePettittTestOnCPU} from '../../../src/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation';

it('S1 Python oracle regeneration distinguishes untested and constant Pettitt rows', () => {
  const output = execFileSync(
    '/usr/bin/python3',
    [fileURLToPath(new URL('./audit-change-points.oracle.py', import.meta.url))],
    {encoding: 'utf8'}
  );
  const rows = output
    .split('\n')
    .filter(line => line.startsWith('pettitt '))
    .map(
      line =>
        JSON.parse(line.slice(8).replace(/\bNaN\b/g, 'null')) as {
          statistic: number;
          changeSlice: number;
          pValues: number | null;
        }
    );
  expect(rows).toHaveLength(6);
  expect(rows[2]).toEqual({statistic: 0, changeSlice: 1, pValues: 1});
  for (const row of rows.slice(3, 5))
    expect(row).toEqual({statistic: 0, changeSlice: 0xffffffff, pValues: null});
});

it('N1 reference keeps the first eligible split of a tested constant', () => {
  // Only the two finite observations are compared: K=0 and the first split's right slice is 3.
  expect(computePettittReference(Float32Array.of(NaN, 5, NaN, 5))).toEqual({
    statistic: 0,
    changeSlice: 3,
    pValue: 1
  });
  const actual = computePettittTestOnCPU(Float32Array.of(NaN, 5, NaN, 5), {
    cellCount: 1,
    sliceCount: 4,
    layout: 'cell-major'
  });
  expect(Array.from(actual.changeSlice)).toEqual([3]);
  expect(Array.from(actual.pValues)).toEqual([1]);
});
