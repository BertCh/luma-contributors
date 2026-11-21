// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeTerrainWatershedsOnCPU,
  CPUTerrainWatershedsOperation
} from '../../../src/gpu-terrain/hydrology/cpu-terrain-watersheds-operation';
import {
  GPUTerrainWatershedsOperation,
  type TerrainWatershedsOperationProps
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-watersheds-operation';

const GRID = {
  width: 5,
  height: 1,
  pixelInterpretation: 'area' as const,
  affine: [1, 0, 0, 0, -1, 0] as const
};
const NONE = 0xffffffff;
const UNRESOLVED = 0xfffffffe;

it('labels hand-drawn chains and nearest inclusive duplicate pour points', () => {
  // 0 -> 1 -> 2 <- 3 <- 4, two nested pour points (1 and 2).
  const directions = new Uint32Array([1, 1, 0, 16, 16]);
  const options = {grid: GRID};
  expect(Array.from(computeTerrainWatershedsOnCPU(directions, options).labels)).toEqual([
    2, 2, 2, 2, 2
  ]);
  const result = computeTerrainWatershedsOnCPU(directions, options, new Uint32Array([2, 1, 1, 99]));
  expect(Array.from(result.labels)).toEqual([1, 1, 0, 0, 0]);
  expect(Array.from(result.converged)).toEqual([1]);
  expect(Array.from(result.iterationLimitReached)).toEqual([0]);
  expect(Array.from(result.iterationCount)).toEqual([0]);
  expect(
    Array.from(computeTerrainWatershedsOnCPU(directions, options, new Uint32Array()).labels)
  ).toEqual([NONE, NONE, NONE, NONE, NONE]);
});

it('honors all direction numberings, terminals, unknown paths, invalid barriers and cycles', () => {
  for (const [directionEncoding, words] of [
    ['esri', [1, 1, 0, 16, 16]],
    ['whitebox', [2, 2, 0, 32, 32]],
    ['taudem', [1, 1, 0, 5, 5]]
  ] as const) {
    expect(
      Array.from(
        computeTerrainWatershedsOnCPU(new Uint32Array(words), {grid: GRID, directionEncoding})
          .labels
      )
    ).toEqual([2, 2, 2, 2, 2]);
  }
  const options = {grid: GRID};
  expect(
    Array.from(
      computeTerrainWatershedsOnCPU(new Uint32Array([1, NONE, 1, UNRESOLVED, 1]), options).labels
    )
  ).toEqual([NONE, NONE, UNRESOLVED, UNRESOLVED, 4]);
  const cycle = computeTerrainWatershedsOnCPU(new Uint32Array([1, 16, 0, 0, 0]), options);
  expect(Array.from(cycle.labels)).toEqual([UNRESOLVED, UNRESOLVED, 2, 3, 4]);
  expect(cycle.converged[0]).toBe(0);
  expect(cycle.iterationLimitReached[0]).toBe(0);
  const intercepted = computeTerrainWatershedsOnCPU(
    new Uint32Array([1, 16, 0, 0, 0]),
    options,
    new Uint32Array([1])
  );
  expect(Array.from(intercepted.labels)).toEqual([0, 0, NONE, NONE, NONE]);
  expect(intercepted.converged[0]).toBe(1);
});

it('explicit pointer budgets suppress unfinished labels and require an unchanged confirmation', () => {
  // Initial [1,2,3,4,4], then [2,3,4,4,4], then [4,4,4,4,4], then unchanged.
  for (const [maximumIterations, labels, status] of [
    [1, [UNRESOLVED, UNRESOLVED, 4, 4, 4], [0, 1, 1]],
    [2, [4, 4, 4, 4, 4], [0, 1, 2]],
    [3, [4, 4, 4, 4, 4], [1, 0, 3]]
  ] as const) {
    const result = computeTerrainWatershedsOnCPU(Uint32Array.of(1, 1, 1, 1, 0), {
      grid: GRID,
      maximumIterations
    });
    expect(Array.from(result.labels)).toEqual(labels);
    expect([
      result.converged[0],
      result.iterationLimitReached[0],
      result.iterationCount[0]
    ]).toEqual(status);
  }
});

it('validates both operation declarations and inconsistent array lengths', () => {
  const props: TerrainWatershedsOperationProps = {
    id: 'watershed-validation',
    grid: GRID,
    flowDirections: 'directions',
    output: {labels: 'labels'}
  };
  for (const Operation of [GPUTerrainWatershedsOperation, CPUTerrainWatershedsOperation]) {
    expect(() => new Operation({...props, maximumIterations: 1024})).not.toThrow();
    expect(() => new Operation({...props, maximumIterations: 1025})).toThrow(
      /watershed-validation.*maximumIterations.*1024/
    );
    expect(() => new Operation({...props, grid: {...GRID, width: 0}})).toThrow(
      /^watershed-validation/
    );
    expect(() => new Operation({...props, output: {labels: 'directions'}})).toThrow(
      /^watershed-validation/
    );
  }
  expect(() => new CPUTerrainWatershedsOperation({...props, maximumIterations: 0})).toThrow(
    /watershed-validation.*maximumIterations/
  );
  const cycleGrid = {...GRID, width: 2, height: 2};
  const cycle = Uint32Array.of(1, 8, 64, 0);
  // The three-cycle rotates forever under synchronous doubling; every permitted round runs.
  const boundedCycle = computeTerrainWatershedsOnCPU(cycle, {
    grid: cycleGrid,
    maximumIterations: 1024
  });
  expect([
    boundedCycle.converged[0],
    boundedCycle.iterationLimitReached[0],
    boundedCycle.iterationCount[0]
  ]).toEqual([0, 1, 1024]);
  expect(() =>
    computeTerrainWatershedsOnCPU(cycle, {grid: cycleGrid, maximumIterations: 1025})
  ).toThrow(/computeTerrainWatershedsOnCPU.*maximumIterations.*1024/);
  expect(() => computeTerrainWatershedsOnCPU(new Uint32Array(4), {grid: GRID})).toThrow(
    /flowDirections/
  );
});
