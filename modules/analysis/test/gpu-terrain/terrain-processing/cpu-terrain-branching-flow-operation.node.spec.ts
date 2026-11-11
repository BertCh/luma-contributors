// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeTerrainBranchingFlowOnCPU,
  CPUTerrainBranchingFlowOperation
} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-branching-flow-operation';
import {GPUTerrainBranchingFlowOperation} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {
  BRANCHING_GRID,
  makeBranchingInput,
  makeBranchingProps
} from './terrain-branching-flow-operation-fixture';

it('matches hand-computed weighted split/merge transport and all four downstream recurrences', () => {
  // Transport at 4: 11 + (3 + 2/4)/2 + (7 + 3*2/4) = 21.25.
  // The three paths from 0 have probabilities 1/8, 1/8, 3/4 and distances 5, 18, 22.
  // Expected distance is 5/8 + 18/8 + 3*22/4 = 19.375; min=5, max=22.
  const expected = {
    transport: [2, 3.5, 6.75, 8.5, 21.25, 34.25],
    'expected-distance': [19.375, 9.5, 0, 18, 11, 0],
    'minimum-distance': [5, 3, 0, 18, 11, 0],
    'maximum-distance': [22, 16, 0, 18, 11, 0],
    'fractional-catchment': [0.6875, 0.5, 0.25, 0.75, 0.75, 0.75]
  };
  for (const mode of Object.keys(expected) as (keyof typeof expected)[]) {
    const input = makeBranchingInput();
    if (mode === 'fractional-catchment') input.values = Float32Array.of(1, 0, 0.25, 0, 0, 0.75);
    const result = computeTerrainBranchingFlowOnCPU(input, {grid: BRANCHING_GRID, mode});
    expect(Array.from(result.values)).toEqual(expected[mode]);
    expect(Array.from(result.status)).toEqual([1, 1, 1, 1, 1, 1]);
    expect(Array.from(result.unresolvedCount)).toEqual([0]);
  }
});

it('retains the GPU local-invalid and neighbor-failure distinctions', () => {
  const grid = {...BRANCHING_GRID, width: 4, height: 1};
  const fractions = new Float32Array(32);
  fractions[8] = 1;
  fractions[16] = 1;
  const input = {fractions, values: Float32Array.of(NaN, 1, 1, 1)};
  const result = computeTerrainBranchingFlowOnCPU(input, {grid, mode: 'transport'});
  // An invalid cell with a zero row donates nothing and does not block its neighbor.
  expect(Array.from(result.values)).toEqual([NaN, 1, 2, 3]);
  expect(Array.from(result.status)).toEqual([0, 1, 1, 1]);
  expect(result.unresolvedCount[0]).toBe(1);
  fractions[0] = NaN;
  const blocked = computeTerrainBranchingFlowOnCPU(input, {grid, mode: 'transport'});
  expect(Array.from(blocked.status)).toEqual([0, 2, 2, 2]);
  expect(blocked.unresolvedCount[0]).toBe(4);
  fractions.fill(0);
  fractions[0] = 1;
  fractions[12] = 1;
  const cycle = computeTerrainBranchingFlowOnCPU(
    {...input, values: new Float32Array(4)},
    {grid, mode: 'expected-distance'}
  );
  expect(Array.from(cycle.status)).toEqual([2, 2, 1, 1]);
  expect(Array.from(cycle.values)).toEqual([NaN, NaN, 0, 0]);
});

it('checks fractional membership, off-grid edges, used lengths and final float32 overflow', () => {
  const grid = {...BRANCHING_GRID, width: 2, height: 1};
  const fractions = new Float32Array(16);
  fractions[0] = 1;
  const input = {fractions, values: Float32Array.of(0, 2)};
  expect(
    Array.from(computeTerrainBranchingFlowOnCPU(input, {grid, mode: 'fractional-catchment'}).status)
  ).toEqual([2, 0]);
  expect(
    Array.from(
      computeTerrainBranchingFlowOnCPU(
        {...input, edgeLengths: new Float32Array(16).fill(-1)},
        {grid, mode: 'expected-distance'}
      ).status
    )
  ).toEqual([2, 1]);
  fractions[8] = 1;
  expect(
    Array.from(computeTerrainBranchingFlowOnCPU(input, {grid, mode: 'transport'}).status)
  ).toEqual([1, 0]);
  fractions[8] = 0;
  const overflow = computeTerrainBranchingFlowOnCPU(
    {...input, values: new Float32Array(2).fill(3e38)},
    {grid, mode: 'transport'}
  );
  expect(Array.from(overflow.status)).toEqual([1, 2]);
  expect(overflow.values[1]).toBeNaN();
});

it('uses affine vectors and the midpoint WGS84 row scale', () => {
  const grid = {...BRANCHING_GRID, width: 2, height: 1, affine: [3, 1, 0, 4, -2, 0] as const};
  const fractions = new Float32Array(16);
  fractions[0] = 1;
  const input = {fractions, values: new Float32Array(2)};
  expect(computeTerrainBranchingFlowOnCPU(input, {grid, mode: 'expected-distance'}).values[0]).toBe(
    5
  );
  // At latitude 0 the east-west WGS84 radius is exactly the semi-major axis.
  const geographic = {...grid, affine: [1, 0, 0, 0, -1, 0.5] as const};
  expect(
    computeTerrainBranchingFlowOnCPU(input, {
      grid: geographic,
      mode: 'expected-distance',
      cellSizeMode: 'geographic'
    }).values[0]
  ).toBe(Math.fround((6378137 * Math.PI) / 180));
});

it('resolves a 4097-cell chain with the exact closed-form accumulation', () => {
  const count = 4097;
  const fractions = new Float32Array(count * 8);
  for (let cell = 0; cell + 1 < count; cell++) fractions[cell * 8] = 1;
  const result = computeTerrainBranchingFlowOnCPU(
    {fractions, values: new Float32Array(count).fill(0.25)},
    {grid: {...BRANCHING_GRID, width: count, height: 1}, mode: 'transport'}
  );
  expect(result.unresolvedCount[0]).toBe(0);
  for (let cell = 0; cell < count; cell++) {
    expect(result.values[cell]).toBe((cell + 1) / 4);
    expect(result.status[cell]).toBe(1);
  }
});

it('validates both operation declarations and rejects inconsistent arrays', () => {
  const props = makeBranchingProps('transport');
  for (const Operation of [GPUTerrainBranchingFlowOperation, CPUTerrainBranchingFlowOperation]) {
    expect(() => new Operation({...props, grid: {...props.grid, width: 0}})).toThrow(
      /^branching-test/
    );
    expect(() => new Operation({...props, output: {...props.output, values: 'values'}})).toThrow(
      /^branching-test/
    );
    expect(() => new Operation({...props, output: {...props.output, status: 'result'}})).toThrow(
      /^branching-test/
    );
  }
  expect(() => new CPUTerrainBranchingFlowOperation({...props, maximumWalkLength: 0})).toThrow(
    /branching-test.*maximumWalkLength/
  );
  expect(() => new CPUTerrainBranchingFlowOperation({...props, maximumIterations: 0})).toThrow(
    /branching-test.*maximumIterations/
  );
  expect(
    () =>
      new CPUTerrainBranchingFlowOperation({
        ...props,
        maximumWalkLength: 2000,
        maximumIterations: 2000
      })
  ).not.toThrow();
  const input = makeBranchingInput();
  for (const [argument, array] of Object.entries({
    fractions: new Float32Array(),
    values: new Float32Array(),
    edgeLengths: new Float32Array(),
    validity: new Uint32Array()
  })) {
    expect(() => computeTerrainBranchingFlowOnCPU({...input, [argument]: array}, props)).toThrow(
      new RegExp(argument)
    );
  }
});

it('constructs the deferred GPU contributor and retains its GPU-only budget checks', () => {
  const props = makeBranchingProps('transport');
  const shapes = {
    fractions: {format: 'float32' as const, length: 48},
    values: {format: 'float32' as const, length: 6},
    edgeLengths: {format: 'float32' as const, length: 48}
  };
  expect(() =>
    constructNetworkOperationContributor(new GPUTerrainBranchingFlowOperation(props), shapes)
  ).not.toThrow();
  for (const overrides of [{maximumWalkLength: 2000}, {maximumIterations: 2000}])
    expect(() =>
      constructNetworkOperationContributor(
        new GPUTerrainBranchingFlowOperation({...props, ...overrides}),
        shapes
      )
    ).toThrow(/^branching-test/);
});
