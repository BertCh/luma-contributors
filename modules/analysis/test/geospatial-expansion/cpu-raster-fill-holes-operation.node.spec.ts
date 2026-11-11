// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  computeRasterFillHolesOnCPU,
  CPURasterFillHolesOperation
} from '../../src/raster-analysis/raster-morphology/cpu-raster-fill-holes-operation';
import {
  GPURasterFillHolesOperation,
  type RasterFillHolesOperationProps
} from '../../src/raster-analysis/raster-morphology/gpu-raster-fill-holes-operation';
import {constructRasterOperationContributor} from '../utils/raster-operation-test-utils';
import fixtures from './raster-morphology-operation-fixtures.json';

const GRID: GPURasterMetadata = {
  width: 9,
  height: 7,
  affine: [1, 0, 0, 0, -1, 7],
  pixelInterpretation: 'area'
};
const PROPS: RasterFillHolesOperationProps = {
  id: 'fill',
  grid: GRID,
  input: 'input',
  output: {
    values: 'filled',
    status: {converged: 'converged', iterationCount: 'rounds', iterationLimitReached: 'limit'}
  }
};

it('matches independent SciPy component filling for both connectivities and nonbinary foreground', () => {
  for (const fixture of fixtures.fillHoles) {
    const result = computeRasterFillHolesOnCPU(Uint32Array.from(fixture.input), {
      grid: {...GRID, width: fixture.width, height: fixture.height},
      connectivity: fixture.connectivity as 4 | 8
    });
    expect(Array.from(result.values)).toEqual(fixture.expected);
    expect(Array.from(result.converged)).toEqual([1]);
    expect(Array.from(result.iterationLimitReached)).toEqual([0]);
  }
});

it('counts shortest boundary-background layers, including the verifying round', () => {
  const empty = computeRasterFillHolesOnCPU(new Uint32Array(63), {grid: GRID});
  expect(empty.values.every(value => value === 0)).toBe(true);
  // The most distant pixel is 3 edges from a boundary of the 9 by 7 rectangle.
  expect(empty.iterationCount[0]).toBe(4);
  const full = computeRasterFillHolesOnCPU(new Uint32Array(63).fill(15), {grid: GRID});
  expect(full.values.every(value => value === 1)).toBe(true);
  expect(full.iterationCount[0]).toBe(1);
  expect(() => computeRasterFillHolesOnCPU(new Uint32Array(62), {grid: GRID})).toThrow(/input/);
});

it('validates both operations at construction and drops only the CPU command budget', () => {
  const inputs = {input: {format: 'uint32' as const, length: 63}};
  expect(() => new CPURasterFillHolesOperation({...PROPS, maximumIterations: 0})).toThrow(
    /^fill .*maximumIterations/
  );
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterFillHolesOperation({...PROPS, maximumIterations: 0}),
      inputs
    )
  ).toThrow(/^fill .*maximumIterations/);
  for (const Operation of [CPURasterFillHolesOperation, GPURasterFillHolesOperation])
    expect(() => new Operation({...PROPS, grid: {...GRID, width: 0}})).toThrow(/^fill /);
  expect(() => new CPURasterFillHolesOperation({...PROPS, maximumIterations: 30000})).not.toThrow();
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterFillHolesOperation({...PROPS, maximumIterations: 30000}),
      inputs
    )
  ).toThrow(/^fill /);
  expect(() =>
    constructRasterOperationContributor(new GPURasterFillHolesOperation(PROPS), inputs)
  ).not.toThrow();
  expect(() =>
    constructRasterOperationContributor(new GPURasterFillHolesOperation(PROPS), {
      input: {format: 'uint32', length: 62}
    })
  ).toThrow(/^fill .*input/);
});

it('maps optional status names and the output callback exactly', async () => {
  const operation = new CPURasterFillHolesOperation({...PROPS, maximumIterations: 1});
  const values = new Uint32Array(63);
  const result = await operation.execute({
    inputs: {
      input: {format: 'uint32', length: 63, chunks: [{values, length: 63}], pack: () => values}
    }
  });
  expect(Object.keys(result).sort()).toEqual(['converged', 'filled', 'limit', 'rounds']);
  expect(result['rounds']).toEqual(Uint32Array.of(4));
  const minimal = new CPURasterFillHolesOperation({
    ...PROPS,
    output: {values: 'filled', status: {converged: 'done', iterationCount: undefined}}
  });
  expect(
    Object.keys(
      await minimal.execute({
        inputs: {
          input: {format: 'uint32', length: 63, chunks: [{values, length: 63}], pack: () => values}
        }
      })
    ).sort()
  ).toEqual(['done', 'filled']);
});

it('ignores surplus input rows and returns only the unit-grid boundary background', () => {
  const grid = {...GRID, width: 1, height: 1, affine: [1, 0, 0, 0, -1, 1] as const};
  const result = computeRasterFillHolesOnCPU(Uint32Array.of(0, 7), {grid});
  // The sole grid cell is background on the boundary, so it is never filled.
  expect(result.values).toEqual(Uint32Array.of(0));
  expect(result.converged).toEqual(Uint32Array.of(1));
  expect(result.iterationCount).toEqual(Uint32Array.of(1));
  expect(result.iterationLimitReached).toEqual(Uint32Array.of(0));
  expect(() =>
    constructRasterOperationContributor(new GPURasterFillHolesOperation({...PROPS, grid}), {
      input: {format: 'uint32', length: 2}
    })
  ).not.toThrow();
  expect(() => computeRasterFillHolesOnCPU(new Uint32Array(0), {grid})).toThrow(/input/);
});
