// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {expect, it} from 'vitest';
import {
  computeRasterRegionGraphOnCPU,
  CPURasterRegionGraphOperation
} from '../../../src/raster-analysis/remote-sensing/cpu-raster-region-graph-operation';
import {
  GPURasterRegionGraphOperation,
  type RasterRegionGraphOperationProps
} from '../../../src/raster-analysis/remote-sensing/gpu-raster-region-graph-operation';
import {constructRasterOperationContributor} from '../../utils/raster-operation-test-utils';

const GRID = makeGPURasterMetadata({width: 3, height: 2});
const PROPS: RasterRegionGraphOperationProps = {
  id: 'regions',
  grid: GRID,
  labels: 'labels',
  values: 'values',
  regionCount: 3,
  mergeThreshold: 2,
  maximumIterations: 4,
  output: {
    adjacency: 'adjacency',
    weights: 'weights',
    mapping: 'mapping',
    status: {iterationLimitReached: 'iterationLimitReached'}
  }
};

it('matches hand-counted shared edges and distinguishes fixed linkage from adaptive means', () => {
  // Each column is a region. Means 0, 2, 4; two edges between successive columns.
  const labels = new Uint32Array([0, 1, 2, 0, 1, 2]);
  const values = new Float32Array([0, 2, 4, 0, 2, 4]);
  const fixed = computeRasterRegionGraphOnCPU(labels, values, {...PROPS, maximumIterations: 1});
  expect(Array.from(fixed.adjacency)).toEqual([0, 2, 0, 2, 0, 2, 0, 2, 0]);
  expect(Array.from(fixed.weights)).toEqual([NaN, 2, NaN, 2, NaN, 2, NaN, 2, NaN]);
  expect(Array.from(fixed.mapping)).toEqual([0, 0, 0]);
  expect(fixed.iterationLimitReached[0]).toBe(0);
  // Lexicographic tie picks 0+1, mean 1; distance to region 2 becomes 3, stopping the merge.
  const adaptive = computeRasterRegionGraphOnCPU(labels, values, {...PROPS, mergeMode: 'adaptive'});
  expect(Array.from(adaptive.mapping)).toEqual([0, 0, 2]);
  expect(Array.from(adaptive.weights)).toEqual([NaN, NaN, NaN, NaN, NaN, 3, NaN, 3, NaN]);
});

it('excludes background and missing values from means, retaining boundary counts', () => {
  const result = computeRasterRegionGraphOnCPU(
    new Uint32Array([0, 1, 2, 0, 1, 0xffffffff]),
    new Float32Array([NaN, 1, 2, NaN, 3, 100]),
    PROPS
  );
  expect(Array.from(result.mapping)).toEqual([0, 1, 1]);
  expect(result.adjacency[1]).toBe(2);
  expect(result.weights[1]).toBeNaN();
  expect(result.weights[5]).toBe(0);
});

it('validates both operation forms and drops GPU-only size and round limits on CPU', () => {
  for (const Operation of [CPURasterRegionGraphOperation, GPURasterRegionGraphOperation]) {
    expect(() => new Operation({...PROPS, regionCount: 0})).toThrow(/^regions /);
  }
  const inputs = {
    labels: {format: 'uint32' as const, length: 6},
    values: {format: 'float32' as const, length: 6}
  };
  for (const override of [
    {mergeThreshold: -1},
    {maximumIterations: 0},
    {mergeMode: 'invalid' as never}
  ]) {
    expect(() => new CPURasterRegionGraphOperation({...PROPS, ...override})).toThrow(/^regions /);
    expect(() =>
      constructRasterOperationContributor(
        new GPURasterRegionGraphOperation({...PROPS, ...override}),
        inputs
      )
    ).toThrow(/^regions /);
  }
  expect(
    () => new CPURasterRegionGraphOperation({...PROPS, regionCount: 257, maximumIterations: 257})
  ).not.toThrow();
  // Auto now accepts fixed-linkage Boruvka above the dense reference maximum.
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterRegionGraphOperation({...PROPS, regionCount: 257}),
      inputs
    )
  ).not.toThrow();
  for (const override of [{searchStructure: 'dense' as const}, {mergeMode: 'adaptive' as const}]) {
    expect(() =>
      constructRasterOperationContributor(
        new GPURasterRegionGraphOperation({...PROPS, ...override, regionCount: 257}),
        inputs
      )
    ).toThrow(/^regions regionCount must be an integer in \[1, 256\]$/);
  }
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterRegionGraphOperation({...PROPS, regionCount: 65536}),
      inputs
    )
  ).toThrow(/^regions outputs\.adjacency\.length must be an integer in \[1, 4294967295\]$/);
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterRegionGraphOperation({...PROPS, maximumIterations: 257}),
      inputs
    )
  ).toThrow(/^regions maximumIterations must be an integer in \[1, 256\]$/);
  expect(() =>
    computeRasterRegionGraphOnCPU(new Uint32Array(5), new Float32Array(6), PROPS)
  ).toThrow(/labels/);
  expect(() =>
    computeRasterRegionGraphOnCPU(new Uint32Array(6), new Float32Array(5), PROPS)
  ).toThrow(/values/);
});
