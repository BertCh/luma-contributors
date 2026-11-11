// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeTerrainStreamOrderOnCPU,
  CPUTerrainStreamOrderOperation
} from '../../../src/gpu-terrain/hydrology/cpu-terrain-stream-order-operation';
import {
  GPUTerrainStreamOrderOperation,
  type TerrainStreamOrderOperationProps
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-order-operation';

const GRID = {
  width: 7,
  height: 3,
  pixelInterpretation: 'area' as const,
  affine: [1, 0, 0, 0, -1, 0] as const
};
const NONE = 0xffffffff;

it('recovers a hand-drawn balanced order-three tree', () => {
  const directions = new Uint32Array(21);
  const streams = new Uint32Array(21);
  // Four order-one leaves merge pairwise at 8 and 12, then at 17.
  for (const [cell, code] of [
    [0, 2],
    [2, 8],
    [4, 2],
    [6, 8],
    [8, 2],
    [12, 8],
    [16, 1],
    [18, 16],
    [17, 0]
  ]) {
    directions[cell] = code;
    streams[cell] = 1;
  }
  const result = computeTerrainStreamOrderOnCPU(directions, streams, {grid: GRID});
  const expected = new Uint32Array(21);
  for (const cell of [0, 2, 4, 6]) expected[cell] = 1;
  for (const cell of [8, 12, 16, 18]) expected[cell] = 2;
  expected[17] = 3;
  expect(result.streamOrder).toEqual(expected);
  expect(Array.from(result.converged)).toEqual([1]);
  expect(Array.from(result.iterationLimitReached)).toEqual([0]);
  expect(Array.from(result.iterationCount)).toEqual([0]);
});

it('propagates unknown donors, possible unknown-direction donors, invalid cells and cycles', () => {
  const grid = {...GRID, width: 5, height: 1};
  const directions = new Uint32Array([1, 1, 0, 16, NONE]);
  const result = computeTerrainStreamOrderOnCPU(directions, new Uint32Array([NONE, 1, 1, 1, 1]), {
    grid
  });
  expect(Array.from(result.streamOrder)).toEqual([NONE, NONE, NONE, 1, 0]);
  expect(result.converged[0]).toBe(1);
  const unknownDirection = computeTerrainStreamOrderOnCPU(
    new Uint32Array([0xfffffffe, 1, 0, 16, NONE]),
    new Uint32Array([1, 1, 1, 1, 1]),
    {grid}
  );
  expect(Array.from(unknownDirection.streamOrder)).toEqual([NONE, NONE, NONE, 1, 0]);
  const cycle = computeTerrainStreamOrderOnCPU(
    new Uint32Array([1, 1, 16, 16, NONE]),
    new Uint32Array([1, 1, 1, 1, 1]),
    {grid}
  );
  expect(Array.from(cycle.streamOrder)).toEqual([NONE, NONE, NONE, NONE, 0]);
  expect(cycle.converged[0]).toBe(0);
  expect(cycle.iterationLimitReached[0]).toBe(0);
  expect(
    Array.from(computeTerrainStreamOrderOnCPU(directions, new Uint32Array(5), {grid}).streamOrder)
  ).toEqual([0, 0, 0, 0, 0]);
});

it('validates both declarations, CPU static options, and inconsistent arrays', () => {
  const props: TerrainStreamOrderOperationProps = {
    id: 'order-validation',
    grid: GRID,
    flowDirections: 'directions',
    streams: 'streams',
    output: {streamOrder: 'order'}
  };
  for (const Operation of [GPUTerrainStreamOrderOperation, CPUTerrainStreamOrderOperation]) {
    expect(() => new Operation({...props, grid: {...GRID, width: 0}})).toThrow(/^order-validation/);
    expect(() => new Operation({...props, output: {streamOrder: 'streams'}})).toThrow(
      /^order-validation/
    );
  }
  expect(() => new CPUTerrainStreamOrderOperation({...props, maximumIterations: 0})).toThrow(
    /order-validation.*maximumIterations/
  );
  expect(
    () => new CPUTerrainStreamOrderOperation({...props, maximumIterations: 2000})
  ).not.toThrow();
  expect(() =>
    computeTerrainStreamOrderOnCPU(new Uint32Array(20), new Uint32Array(21), {grid: GRID})
  ).toThrow(/flowDirections/);
  expect(() =>
    computeTerrainStreamOrderOnCPU(new Uint32Array(21), new Uint32Array(20), {grid: GRID})
  ).toThrow(/streams/);
});
