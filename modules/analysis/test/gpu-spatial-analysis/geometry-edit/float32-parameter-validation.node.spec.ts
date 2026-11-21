// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUGeometryCleanupParameterValues} from '../../../src/gpu-spatial-analysis/geometry-edit';
import {getGPULineSimplificationParameterValues} from '../../../src/gpu-spatial-analysis/line-simplification/line-simplification-parameters';
import {getGPURectangleClipParameterValues} from '../../../src/gpu-spatial-analysis/rectangle-clip';

it('UGE-08 rejects float32 overflow and positive disable-switch underflow before writing', () => {
  const target = Float32Array.of(7, 7, 7, 7);
  for (const setting of ['gridSize', 'tolerance'] as const) {
    for (const value of [1e40, 1e-50]) {
      expect(() => getGPUGeometryCleanupParameterValues({[setting]: value}, target)).toThrow(
        new RegExp(
          `GPUGeometryCleanup.*${setting}.*${String(value).replace('+', '\\+').replace('-', '\\-')}`
        )
      );
      expect(Array.from(target)).toEqual([7, 7, 7, 7]);
    }
  }
  for (const tolerance of [1e40, 1e-50]) {
    expect(() => getGPULineSimplificationParameterValues({tolerance}, target)).toThrow(
      /GPULineSimplification.*tolerance/
    );
    expect(Array.from(target)).toEqual([7, 7, 7, 7]);
  }
  for (const setting of ['minX', 'minY', 'maxX', 'maxY'] as const) {
    expect(() =>
      getGPURectangleClipParameterValues(
        {minX: 0, minY: 0, maxX: 1, maxY: 1, [setting]: 1e40},
        target
      )
    ).toThrow(new RegExp(`GPURectangleClip.*${setting}`));
    expect(Array.from(target)).toEqual([7, 7, 7, 7]);
  }
  expect(Array.from(getGPUGeometryCleanupParameterValues({gridSize: 0, tolerance: 0}))).toEqual([
    0, 0, 1, 0
  ]);
  expect(Array.from(getGPULineSimplificationParameterValues({tolerance: 0}))).toEqual([0, 0, 0, 0]);
  expect(
    Array.from(getGPURectangleClipParameterValues({minX: 0, minY: 0, maxX: 0, maxY: 0}))
  ).toEqual([0, 0, 0, 0]);
});
