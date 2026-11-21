// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import * as density from '@luma-contributors/analysis/gpu-spatial-analysis';
import {computeKernelDensityField} from '../src/analysis/catalog-platform3-b-grids';
import type {GPUKernelDensityKernelName} from '@luma-contributors/analysis/gpu-spatial-analysis';

const sample = {
  x: Float64Array.of(1.5),
  y: Float64Array.of(0.5),
  value: Float64Array.of(1),
  sourceIndex: Int32Array.of(0),
  partIndex: Int32Array.of(0),
  count: 1,
  droppedCount: 0
};
const grid = {minX: 0, minY: 0, cellSize: 1, columns: 1, rows: 1};

test('catalog KDE bisquare is quartic with area normalization, distinct from Epanechnikov', () => {
  // Center=(.5,.5), d=1, h=2. Quartic: 3/(pi*4)*(1-1/4)^2=27/(64*pi).
  // Epanechnikov: 2/(pi*4)*(1-1/4)=3/(8*pi). Independent closed forms.
  const value = (kernel: GPUKernelDensityKernelName) => {
    const values = computeKernelDensityField(sample, grid, {kernel, bandwidth: 2}).values;
    expect(values).toHaveLength(1);
    return values[0];
  };
  expect(Number.isFinite(value('quartic'))).toBe(true);
  expect(value('quartic')).toBeCloseTo(0.13428698323378668, 14);
  expect(value('bisquare')).toBe(value('quartic'));
  expect(Number.isFinite(value('epanechnikov'))).toBe(true);
  expect(value('epanechnikov')).toBeCloseTo(0.1193662073189215, 14);
  expect(value('epanechnikov')).not.toBe(value('quartic'));
  // /usr/bin/python3: math.exp(-1/8)/(8*math.pi), the normalized Gaussian at d=1,h=2.
  expect(Number.isFinite(value('gaussian'))).toBe(true);
  expect(value('gaussian')).toBeCloseTo(0.0351134360774063, 14);
  for (const kernel of ['quartic', 'bisquare', 'epanechnikov'] as const) {
    const values = computeKernelDensityField(sample, grid, {kernel, bandwidth: 1}).values;
    expect(values).toHaveLength(1);
    expect(values[0]).toBe(0);
  }
  const gaussianCutoff = computeKernelDensityField(sample, grid, {
    kernel: 'gaussian',
    bandwidth: 0.2
  }).values;
  expect(gaussianCutoff).toHaveLength(1);
  expect(gaussianCutoff[0]).toBe(0);
  expect(() => value('typo' as GPUKernelDensityKernelName)).toThrow(
    /gaussian.*quartic.*epanechnikov/
  );
  // An unknown name must also fail when there are no samples to evaluate.
  expect(() =>
    computeKernelDensityField({...sample, count: 0}, grid, {
      kernel: 'typo' as GPUKernelDensityKernelName,
      bandwidth: 2
    })
  ).toThrow();
});

test('catalog KDE resolves the public evaluator once per field instead of per sample/cell', () => {
  const evaluator = vi.spyOn(density, 'evaluateGPUKernelDensityKernel');
  try {
    const values = computeKernelDensityField(
      sample,
      {...grid, columns: 2, rows: 2},
      {
        kernel: 'bisquare',
        bandwidth: 2
      }
    ).values;
    expect(values).toHaveLength(4);
    expect(evaluator).toHaveBeenCalledTimes(1);
  } finally {
    evaluator.mockRestore();
  }
});
