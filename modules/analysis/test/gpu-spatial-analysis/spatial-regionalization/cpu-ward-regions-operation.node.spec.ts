// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeWardRegionsOnCPU,
  CPUWardRegionsOperation
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-ward-regions-operation';
import {
  GPUWardRegionsOperation,
  type WardRegionsOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions-operation';
import {
  getGPUWardRegionsParameterValues,
  GPU_WARD_STATUS_LAYOUT
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions';
import {
  WARD_FIXTURE,
  createCompleteAdjacency,
  createLatticeAdjacency,
  getCSR
} from './ward-reference';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';

const PROPS: WardRegionsOperationProps = {
  id: 'ward-check',
  weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
  values: 'values',
  parameters: 'parameters',
  rowCount: 4,
  targetRegionCount: 2,
  maximumIterations: 2,
  output: {labels: 'labels', objective: 'objective', status: 'status'}
};

it('matches external SciPy and scikit-learn fixture cuts and direct within-region objectives', () => {
  for (const [scene, adjacency, schedules] of [
    [
      WARD_FIXTURE.separated,
      createCompleteAdjacency(WARD_FIXTURE.separated.rowCount),
      ['sequential', 'mutual-nearest']
    ],
    [WARD_FIXTURE.random, createCompleteAdjacency(WARD_FIXTURE.random.rowCount), ['sequential']],
    [WARD_FIXTURE.lattice, createLatticeAdjacency(12, 12), ['sequential']]
  ] as const) {
    const pattern = getCSR(adjacency);
    for (const cut of scene.cuts) {
      for (const mergeSchedule of schedules) {
        const result = computeWardRegionsOnCPU(
          {
            ...pattern,
            values: Float32Array.from(scene.values),
            parameters: getGPUWardRegionsParameterValues()
          },
          {
            columnCount: scene.columnCount,
            standardize: false,
            targetRegionCount: cut.targetRegionCount,
            mergeSchedule
          }
        );
        expect(Array.from(result.labels)).toEqual(cut.labels);
        expect(Math.abs(result.objective[0] - cut.objective)).toBeLessThanOrEqual(
          1e-5 * Math.max(1, cut.objective)
        );
        expect(Array.from(result.status)).toEqual([
          scene.rowCount - cut.targetRegionCount,
          1,
          0,
          0
        ]);
      }
    }
  }
});

it('uses population standardization, directed adjacency union, and exact constant columns', () => {
  // Directed chain only; raw one-dimensional merge objective is 1. Population variance is 25.25.
  const pattern = getCSR([[1], [2], [3], []]);
  for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
    const result = computeWardRegionsOnCPU(
      {
        ...pattern,
        values: Float32Array.of(0, 7, 1, 7, 10, 7, 11, 7),
        parameters: getGPUWardRegionsParameterValues({seed: 7})
      },
      {columnCount: 2, targetRegionCount: 2, mergeSchedule}
    );
    expect(Array.from(result.labels)).toEqual([0, 0, 2, 2]);
    expect(result.objective[0]).toBeCloseTo(1 / 25.25, 7);
  }
});

it('keeps the pair-key tie rule and rejects malformed arrays while reporting invalid entries', () => {
  // Every pair has cost 0. For seed 0 the independently precomputed smallest pair hash is (1,2).
  // Verify the tie with literal keys, independent of the production hash implementation.
  const pattern = getCSR(createCompleteAdjacency(3));
  const zero = computeWardRegionsOnCPU(
    {...pattern, values: new Float32Array(3), parameters: getGPUWardRegionsParameterValues()},
    {standardize: false, targetRegionCount: 2}
  );
  expect(Array.from(zero.labels)).toEqual([0, 1, 1]);
  const invalid = computeWardRegionsOnCPU(
    {
      offsets: Uint32Array.of(0, 1, 1),
      neighbors: Uint32Array.of(7),
      values: Float32Array.of(NaN, Infinity),
      parameters: Uint32Array.of(0)
    },
    {targetRegionCount: 1}
  );
  expect(Array.from(invalid.labels)).toEqual([0, 1]);
  expect(Array.from(invalid.status)).toEqual([0, 0, 0, 3]);
  expect(invalid.objective[0]).toBe(0);
  expect(() =>
    computeWardRegionsOnCPU(
      {...pattern, values: new Float32Array(2), parameters: Uint32Array.of(0)},
      {targetRegionCount: 1}
    )
  ).toThrow(/values length/);
  expect(() =>
    computeWardRegionsOnCPU(
      {...pattern, values: new Float32Array(3), parameters: new Uint32Array(0)},
      {targetRegionCount: 1}
    )
  ).toThrow(/parameters/);
  expect(() =>
    computeWardRegionsOnCPU(
      {
        offsets: Uint32Array.of(0, 7),
        neighbors: new Uint32Array(1),
        values: Float32Array.of(0),
        parameters: Uint32Array.of(0)
      },
      {targetRegionCount: 1}
    )
  ).toThrow(/offsets/);
});

it('stops at disconnected components, supports nine columns and honors explicit round budgets', () => {
  const pattern = getCSR([[1], [0], [3], [2]]);
  const values = Float32Array.from({length: 36}, (_, index) => Math.floor(index / 9));
  const result = computeWardRegionsOnCPU(
    {...pattern, values, parameters: Uint32Array.of(0)},
    {columnCount: 9, targetRegionCount: 1, standardize: false}
  );
  expect(Array.from(result.labels)).toEqual([0, 0, 2, 2]);
  expect(result.objective[0]).toBe(9);
  expect(Array.from(result.status)).toEqual([2, 0, 0, 0]);
  const mutual = computeWardRegionsOnCPU(
    {...pattern, values, parameters: Uint32Array.of(0)},
    {
      columnCount: 9,
      targetRegionCount: 1,
      standardize: false,
      mergeSchedule: 'mutual-nearest',
      statusLength: 5
    }
  );
  expect(Array.from(mutual.status)).toEqual([2, 0, 0, 0, 2]);
  const invalid = computeWardRegionsOnCPU(
    {...pattern, values: new Float32Array(4).fill(NaN), parameters: Uint32Array.of(0)},
    {targetRegionCount: 1, mergeSchedule: 'mutual-nearest', statusLength: 5}
  );
  expect(Array.from(invalid.status)).toEqual([0, 0, 0, 4, 1]);
  expect(
    () => new CPUWardRegionsOperation({...PROPS, columnCount: 9, maximumIterations: 0})
  ).not.toThrow();
  const stopped = computeWardRegionsOnCPU(
    {...pattern, values, parameters: Uint32Array.of(0)},
    {columnCount: 9, targetRegionCount: 1, standardize: false, maximumIterations: 0}
  );
  expect(Array.from(stopped.labels)).toEqual([0, 1, 2, 3]);
  expect(Array.from(stopped.status)).toEqual([0, 0, 1, 0]);
});

it('checks construction-time CPU limits and delegates GPU checks to contributor construction', () => {
  for (const Operation of [CPUWardRegionsOperation, GPUWardRegionsOperation]) {
    expect(() => new Operation({...PROPS, output: {...PROPS.output, labels: 'values'}})).toThrow(
      /^ward-check outputs.labels/
    );
  }
  expect(() => new CPUWardRegionsOperation({...PROPS, rowCount: 0})).toThrow(
    /^ward-check rowCount/
  );
  expect(() => new GPUWardRegionsOperation({...PROPS, rowCount: 0})).toThrow(
    /^ward-check outputs.labels.length/
  );
  expect(() => new CPUWardRegionsOperation({...PROPS, targetRegionCount: 5})).toThrow(
    /^ward-check targetRegionCount/
  );
  expect(() => new CPUWardRegionsOperation({...PROPS, columnCount: 0})).toThrow(
    /^ward-check columnCount/
  );
  expect(() => new CPUWardRegionsOperation({...PROPS, maximumIterations: -1})).toThrow(
    /^ward-check maximumIterations/
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {
      offsets: {format: 'uint32', length: 5},
      neighbors: {format: 'uint32', length: 6},
      weights: {format: 'float32', length: 6},
      values: {format: 'float32', length: 4},
      parameters: {format: 'uint32', length: 1},
      labels: {format: 'uint32', length: 4},
      objective: {format: 'float32', length: 1},
      status: {format: 'uint32', length: GPU_WARD_STATUS_LAYOUT.length}
    });
    expect(() =>
      new GPUWardRegionsOperation({...PROPS, targetRegionCount: 5}).createContributor(getView)
    ).toThrow(/^ward-check targetRegionCount/);
    expect(() =>
      new GPUWardRegionsOperation({...PROPS, columnCount: 9}).createContributor(getView)
    ).toThrow(/^ward-check columnCount/);
    expect(() =>
      new GPUWardRegionsOperation({...PROPS, maximumIterations: 3}).createContributor(getView)
    ).toThrow(/^ward-check maximumIterations/);
  } finally {
    device.destroy();
  }
});
