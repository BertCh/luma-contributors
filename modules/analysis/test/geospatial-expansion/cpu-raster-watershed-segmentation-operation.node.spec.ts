// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {expect, it} from 'vitest';
import {
  computeRasterWatershedSegmentationOnCPU,
  CPURasterWatershedSegmentationOperation
} from '../../src/raster-analysis/raster-watershed/cpu-raster-watershed-segmentation-operation';
import {
  GPURasterWatershedSegmentationOperation,
  type RasterWatershedSegmentationOperationProps
} from '../../src/raster-analysis/raster-watershed/gpu-raster-watershed-segmentation-operation';
import {constructRasterOperationContributor} from '../utils/raster-operation-test-utils';
import {calculateWatershedScanReference} from './watershed-scan-reference';

const GRID = makeGPURasterMetadata({width: 7, height: 1});
const PROPS: RasterWatershedSegmentationOperationProps = {
  id: 'watershed',
  grid: GRID,
  input: {id: 'band', format: 'float32', storage: {kind: 'buffer', values: 'values'}},
  markers: 'markers',
  output: {
    labels: 'labels',
    status: {converged: 'converged', iterationLimitReached: 'iterationLimitReached'}
  }
};

it('FIFO heap agrees with the independent full-grid priority scan', () => {
  for (const connectivity of [4, 8] as const) {
    const grid = makeGPURasterMetadata({width: 5, height: 4});
    const values = Float32Array.from({length: 20}, (_, cell) => ((cell * 7) % 11) - 5);
    const markers = new Uint32Array(20);
    markers[0] = 3;
    markers[19] = 5;
    markers[9] = 0xffffffff;
    const validity = new Uint32Array(20).fill(1);
    validity[8] = 0;
    const result = computeRasterWatershedSegmentationOnCPU(
      {id: 'band', format: 'float32', values, validity},
      markers,
      {grid, connectivity}
    );
    const expected = calculateWatershedScanReference({
      width: 5,
      height: 4,
      values,
      markers,
      validity,
      connectivity,
      maximumSteps: 20
    });
    expect(Array.from(result.labels)).toEqual(expected.labels);
    expect(Array.from(result.converged)).toEqual(expected.converged);
    expect(result.iterationLimitReached[0]).toBe(0);
  }
});

it('matches hand-derived FIFO and canonical-edge forest plateau ownership', () => {
  const values = new Float32Array([5, 2, 2, 2, 2, 2, 5]);
  const markers = new Uint32Array([1, 0, 0, 0, 0, 0, 2]);
  const input = {id: 'band', format: 'float32' as const, values};
  expect(
    Array.from(computeRasterWatershedSegmentationOnCPU(input, markers, {grid: GRID}).labels)
  ).toEqual([1, 1, 1, 1, 2, 2, 2]);
  // Internal elevation-2 edges join first. The left elevation-5 edge has lower ID than the right.
  const forest = computeRasterWatershedSegmentationOnCPU(input, markers, {
    grid: GRID,
    method: 'minimum-spanning-forest',
    maximumIterations: 32
  });
  expect(Array.from(forest.labels)).toEqual([1, 1, 1, 1, 1, 1, 2]);
  expect(forest.converged[0]).toBe(1);
  const absent = computeRasterWatershedSegmentationOnCPU(input, new Uint32Array(7), {grid: GRID});
  expect(Array.from(absent.labels)).toEqual([0, 0, 0, 0, 0, 0, 0]);
});

it('validates constructors and array arguments, allowing CPU-only larger domains', () => {
  const inputs = {
    values: {format: 'float32' as const, length: 7},
    markers: {format: 'uint32' as const, length: 7}
  };
  for (const override of [
    {connectivity: 6 as never},
    {maximumSteps: 0},
    {maximumIterations: 2},
    {method: 'invalid' as never}
  ]) {
    expect(() => new CPURasterWatershedSegmentationOperation({...PROPS, ...override})).toThrow(
      /^watershed /
    );
    expect(() =>
      constructRasterOperationContributor(
        new GPURasterWatershedSegmentationOperation({...PROPS, ...override}),
        inputs
      )
    ).toThrow(/^watershed /);
  }
  for (const Operation of [
    CPURasterWatershedSegmentationOperation,
    GPURasterWatershedSegmentationOperation
  ]) {
    expect(() => new Operation({...PROPS, grid: {...GRID, width: 0}})).toThrow(/^watershed /);
  }
  expect(
    () => new CPURasterWatershedSegmentationOperation({...PROPS, grid: {...GRID, width: 16385}})
  ).not.toThrow();
  expect(() =>
    computeRasterWatershedSegmentationOnCPU(
      {id: 'band', format: 'float32', values: new Float32Array(6)},
      new Uint32Array(7),
      {grid: GRID}
    )
  ).toThrow(/input.values/);
  expect(() =>
    computeRasterWatershedSegmentationOnCPU(
      {id: 'band', format: 'float32', values: new Float32Array(7)},
      new Uint32Array(6),
      {grid: GRID}
    )
  ).toThrow(/markers/);
});

it('compares nodata in the input scalar format for both watershed methods', () => {
  const grid = makeGPURasterMetadata({width: 2, height: 1});
  for (const method of ['fifo', 'minimum-spanning-forest'] as const) {
    for (const input of [
      {
        id: 'band',
        format: 'float32' as const,
        values: new Float32Array([0.1, 2]),
        noDataValue: 0.1
      },
      {
        id: 'band',
        format: 'uint32' as const,
        values: new Uint32Array([0xffffffff, 2]),
        noDataValue: 0xffffffff
      },
      {
        id: 'band',
        format: 'sint32' as const,
        values: new Int32Array([-2147483647, 2]),
        noDataValue: -2147483647
      }
    ]) {
      // The only marker is nodata. No valid seed remains, so both cells stay unlabelled.
      const result = computeRasterWatershedSegmentationOnCPU(input, new Uint32Array([7, 0]), {
        grid,
        method
      });
      expect(Array.from(result.labels)).toEqual([0, 0]);
      expect(Array.from(result.converged)).toEqual([1]);
      expect(Array.from(result.iterationLimitReached)).toEqual([0]);
    }
  }
});

it('rejects trailing band samples and validity flags for a one-pixel grid', () => {
  const grid = makeGPURasterMetadata({width: 1, height: 1});
  for (const method of ['fifo', 'minimum-spanning-forest'] as const) {
    for (const input of [
      {id: 'band', format: 'float32' as const, values: new Float32Array([3, 100])},
      {
        id: 'band',
        format: 'float32' as const,
        values: new Float32Array([3]),
        validity: new Uint32Array([1, 1])
      }
    ]) {
      expect(() =>
        computeRasterWatershedSegmentationOnCPU(input, new Uint32Array([7]), {grid, method})
      ).toThrow(/input\.(values|validity) must contain exactly one/);
    }
  }
});
