// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import {
  computeNeighborhoodSummaryOnCPU,
  CPUNeighborhoodSummaryOperation
} from '../../../src/gpu-spatial-analysis/neighborhood-summary/cpu-neighborhood-summary-operation';
import {
  GPUNeighborhoodSummaryOperation,
  type NeighborhoodSummaryOperationProps
} from '../../../src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary-operation';
import type {GPUNeighborhoodSummaryStatistic} from '../../../src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary';

const STATISTICS: GPUNeighborhoodSummaryStatistic[] = [
  'count',
  'weightSum',
  'sum',
  'mean',
  'min',
  'max',
  'standardDeviation',
  'median'
];
const INPUT = {
  offsets: new Uint32Array([0, 2, 2, 2]),
  neighbors: new Uint32Array([1, 2]),
  weights: new Float32Array([1, 3]),
  values: new Float32Array([0, 2, 6]),
  categories: new Uint32Array([5, 5, 3])
};
const PROPS: NeighborhoodSummaryOperationProps = {
  id: 'summary-check',
  rowCount: 3,
  weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
  values: 'values',
  categories: 'categories',
  statistics: STATISTICS,
  output: {table: 'table', modes: 'modes', entropy: 'entropy', status: {overflow: 'overflow'}}
};

it('matches a hand-computed weighted neighborhood, including an actual mode tie', () => {
  const result = computeNeighborhoodSummaryOnCPU(INPUT, {
    statistics: STATISTICS,
    includeFocal: true,
    focalWeight: 2
  });
  const expected = [3, 6, 20, 10 / 3, 0, 6, Math.sqrt(68) / 3, 2];
  expected.forEach((value, column) => expect(result.table[column]).toBeCloseTo(value, 6));
  expect(Array.from(result.modes)).toEqual([3, 5, 3]);
  expect(result.entropy[0]).toBeCloseTo(Math.log(2), 7);
  expect(result.overflow[0]).toBe(0);
  const standardized = computeNeighborhoodSummaryOnCPU(INPUT, {
    statistics: STATISTICS,
    includeFocal: true,
    focalWeight: 2,
    standardized: true
  });
  [3, 3, 5, 5 / 3, 0, 6, Math.sqrt(59) / 3, 2].forEach((value, column) =>
    expect(standardized.table[column]).toBeCloseTo(value, 6)
  );
  expect(standardized.modes[0]).toBe(5);
  expect(standardized.entropy[0]).toBeCloseTo(-0.75 * Math.log(0.75) - 0.25 * Math.log(0.25), 7);
});

it('delegates GPU limits to contributor construction and drops that GPU-only cap on CPU', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {
      offsets: {format: 'uint32', length: 4},
      neighbors: {format: 'uint32', length: 2},
      weights: {format: 'float32', length: 2},
      values: {format: 'float32', length: 3},
      categories: {format: 'uint32', length: 3},
      table: {format: 'float32', length: 24},
      modes: {format: 'uint32', length: 3},
      entropy: {format: 'float32', length: 3},
      overflow: {format: 'uint32', length: 1}
    });
    const props = {...PROPS, maximumNeighbors: 65};
    expect(() => new GPUNeighborhoodSummaryOperation(props).createContributor(getView)).toThrow(
      /maximumNeighbors/
    );
    expect(() => new CPUNeighborhoodSummaryOperation(props)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('sorts medians beyond 64 members without overflow', () => {
  const offsets = new Uint32Array(67).fill(65);
  offsets[0] = 0;
  const result = computeNeighborhoodSummaryOnCPU(
    {
      offsets,
      neighbors: Uint32Array.from({length: 65}, (_, index) => index + 1),
      weights: new Float32Array(65).fill(1),
      values: Float32Array.from({length: 66}, (_, row) => row)
    },
    {statistics: ['count', 'median']}
  );
  expect(Array.from(result.table.subarray(0, 2))).toEqual([65, 33]);
  expect(result.overflow[0]).toBe(0);
});

it('counts a zero-weight focal member while keeping its weighted mean and entropy undefined', () => {
  const result = computeNeighborhoodSummaryOnCPU(
    {
      offsets: new Uint32Array([0, 0]),
      neighbors: Uint32Array.of(0),
      weights: Float32Array.of(0),
      values: Float32Array.of(6),
      categories: Uint32Array.of(9)
    },
    {statistics: STATISTICS, includeFocal: true, focalWeight: 0}
  );
  expect(Array.from(result.table)).toEqual([1, 0, 0, NaN, 6, 6, NaN, 6]);
  expect(result.modes[0]).toBe(9);
  expect(result.entropy[0]).toBeNaN();
});

it('keeps categorical membership separate and handles masks, truncation and clamped offsets', () => {
  const nonfinite = computeNeighborhoodSummaryOnCPU(
    {...INPUT, values: new Float32Array([0, NaN, 6])},
    {statistics: ['count', 'mean'], standardized: true}
  );
  expect(Array.from(nonfinite.table.subarray(0, 2))).toEqual([1, 6]);
  expect(nonfinite.modes[0]).toBe(3);
  expect(nonfinite.entropy[0]).toBeCloseTo(-0.25 * Math.log(0.25) - 0.75 * Math.log(0.75), 7);
  const truncated = computeNeighborhoodSummaryOnCPU(
    {...INPUT, mask: new Uint32Array([1, 0, 1]), status: {overflow: Uint32Array.of(1)}},
    {statistics: ['count', 'mean']}
  );
  expect(Array.from(truncated.table)).toEqual([NaN, NaN, 0, NaN, NaN, NaN]);
  expect(Array.from(truncated.modes)).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
  const clamped = computeNeighborhoodSummaryOnCPU(
    {...INPUT, offsets: new Uint32Array([0, 99, 1, 99])},
    {statistics: ['count']}
  );
  expect(Array.from(clamped.table)).toEqual([2, 0, 0]);
  const converted = computeNeighborhoodSummaryOnCPU(
    {...INPUT, values: new Uint32Array([0, 16777217, 16777217])},
    {statistics: ['mean']}
  );
  expect(converted.table[0]).toBe(16777216);
});

it('checks inconsistent arrays and construction-time props', () => {
  for (const replacement of [
    {offsets: new Uint32Array(1)},
    {weights: new Float32Array(1)},
    {values: new Float32Array(2)},
    {categories: new Uint32Array(2)},
    {mask: new Uint32Array(2)},
    {status: {overflow: new Uint32Array(0)}}
  ]) {
    expect(() =>
      computeNeighborhoodSummaryOnCPU({...INPUT, ...replacement}, {statistics: STATISTICS})
    ).toThrow();
  }
  for (const Operation of [GPUNeighborhoodSummaryOperation, CPUNeighborhoodSummaryOperation]) {
    expect(() => new Operation({...PROPS, rowCount: 0})).toThrow(/^summary-check/);
    expect(new Operation(PROPS).id).toBe('summary-check');
  }
  for (const replacement of [
    {statistics: ['mean', 'mean'] as const},
    {focalWeight: -1},
    {maximumNeighbors: 0},
    {values: undefined},
    {categories: undefined},
    {output: {table: 'table'}}
  ]) {
    expect(() => new CPUNeighborhoodSummaryOperation({...PROPS, ...replacement})).toThrow(
      /^summary-check/
    );
  }
});
