// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeTerrainFlowDistanceOnCPU,
  CPUTerrainFlowDistanceOperation
} from '../../../src/gpu-terrain/hydrology/cpu-terrain-flow-distance-operation';
import {
  GPUTerrainFlowDistanceOperation,
  type TerrainFlowDistanceOperationProps
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow-distance-operation';

const GRID = {
  width: 2,
  height: 2,
  pixelInterpretation: 'area' as const,
  affine: [3, 0, 0, 0, -4, 0] as const
};
const DIRECTIONS = new Uint32Array([2, 4, 1, 0]);
const NONE = 0xffffffff;
const UNRESOLVED = 0xfffffffe;

for (const maximumSteps of [3, 2]) {
  it(`sums a two-edge chain and counts its terminal cell, maximumSteps=${maximumSteps}`, () => {
    const result = computeTerrainFlowDistanceOnCPU(
      new Uint32Array([1, 1, 0]),
      {grid: {...GRID, width: 3, height: 1}, maximumSteps},
      undefined,
      new Float32Array([2, 3, NaN])
    );
    // 0 -> 1 -> 2 costs 2 + 3 = 5, and contains three cells; the terminal costs zero.
    expect(Array.from(result.distance)).toEqual([maximumSteps === 3 ? 5 : NaN, 3, 0]);
    expect(Array.from(result.validity)).toEqual([maximumSteps === 3 ? 1 : 0, 1, 1]);
    expect(Array.from(result.terminalCells)).toEqual([maximumSteps === 3 ? 2 : UNRESOLVED, 2, 2]);
  });
}

it('matches a 3-4-5 triangle and hand-summed outgoing costs, counting terminal cells', () => {
  const result = computeTerrainFlowDistanceOnCPU(DIRECTIONS, {grid: GRID});
  expect(Array.from(result.distance)).toEqual([5, 4, 3, 0]);
  expect(Array.from(result.validity)).toEqual([1, 1, 1, 1]);
  expect(Array.from(result.terminalCells)).toEqual([3, 3, 3, 3]);
  const grid = {...GRID, width: 4, height: 1};
  const directions = new Uint32Array([1, 1, 1, 0]);
  const costs = new Float32Array([0.25, 2, 7, NaN]);
  const weighted = computeTerrainFlowDistanceOnCPU(
    directions,
    {grid, maximumSteps: 3},
    undefined,
    costs
  );
  expect(Array.from(weighted.distance)).toEqual([NaN, 9, 7, 0]);
  expect(Array.from(weighted.validity)).toEqual([0, 1, 1, 1]);
  expect(Array.from(weighted.terminalCells)).toEqual([UNRESOLVED, 3, 3, 3]);
  const streams = computeTerrainFlowDistanceOnCPU(
    directions,
    {grid},
    new Uint32Array([0, 1, 0, 0]),
    costs
  );
  expect(Array.from(streams.distance)).toEqual([0.25, 0, 7, 0]);
  expect(Array.from(streams.terminalCells)).toEqual([1, 1, 3, 3]);
});

it('uses affine vector lengths on sheared grids and a closed-form equatorial WGS84 scale', () => {
  const sheared = computeTerrainFlowDistanceOnCPU(DIRECTIONS, {
    grid: {...GRID, affine: [3, 1, 0, 0, -4, 0]}
  });
  expect(sheared.distance[0]).toBe(Math.fround(Math.sqrt(32)));
  expect(sheared.distance[1]).toBe(Math.fround(Math.sqrt(17)));
  const grid = {
    ...GRID,
    width: 2,
    height: 1,
    pixelInterpretation: 'point' as const,
    affine: [1, 0, 0, 0, -1, 0] as const
  };
  const geographic = computeTerrainFlowDistanceOnCPU(new Uint32Array([1, 0]), {
    grid,
    cellSizeMode: 'geographic'
  });
  // At the equator a longitude degree is exactly a * pi / 180 in this ellipsoid differential.
  expect(geographic.distance[0]).toBe(Math.fround((6378137 * Math.PI) / 180));
  const mercator = computeTerrainFlowDistanceOnCPU(new Uint32Array([1, 0]), {
    grid,
    cellSizeMode: 'web-mercator'
  });
  expect(mercator.distance[0]).toBe(1);
});

it('preserves bad-code, invalid, unknown-stream, bad-cost and cycle outcomes', () => {
  const grid = {...GRID, width: 5, height: 1};
  const directions = new Uint32Array([1, 1, 16, NONE, UNRESOLVED]);
  const result = computeTerrainFlowDistanceOnCPU(directions, {grid});
  expect(result.distance.every(Number.isNaN)).toBe(true);
  expect(Array.from(result.validity)).toEqual([0, 0, 0, 0, 0]);
  expect(Array.from(result.terminalCells)).toEqual([
    UNRESOLVED,
    UNRESOLVED,
    UNRESOLVED,
    NONE,
    UNRESOLVED
  ]);
  expect(
    computeTerrainFlowDistanceOnCPU(
      new Uint32Array([1, 0]),
      {grid: {...grid, width: 2}},
      new Uint32Array([0, NONE])
    ).validity[0]
  ).toBe(0);
  expect(
    computeTerrainFlowDistanceOnCPU(
      new Uint32Array([1, 0]),
      {grid: {...grid, width: 2}},
      undefined,
      new Float32Array([-1, 0])
    ).validity[0]
  ).toBe(0);
  // An outward direction is unresolved even if this cell is in the stream mask.
  expect(
    computeTerrainFlowDistanceOnCPU(
      new Uint32Array([1, 1]),
      {grid: {...grid, width: 2}},
      new Uint32Array([0, 1])
    ).validity[1]
  ).toBe(0);
});

it('validates both declarations, path bounds and inconsistent input arrays', () => {
  const props: TerrainFlowDistanceOperationProps = {
    id: 'distance-validation',
    grid: GRID,
    flowDirections: 'directions',
    output: {distance: 'distance', validity: 'validity', terminalCells: 'terminalCells'}
  };
  for (const Operation of [CPUTerrainFlowDistanceOperation, GPUTerrainFlowDistanceOperation]) {
    expect(() => new Operation({...props, grid: {...GRID, width: 0}})).toThrow(
      /^distance-validation/
    );
    expect(
      () => new Operation({...props, output: {...props.output, distance: 'directions'}})
    ).toThrow(/^distance-validation/);
  }
  expect(() => new CPUTerrainFlowDistanceOperation({...props, maximumSteps: 0})).toThrow(
    /distance-validation.*maximumSteps/
  );
  expect(
    () =>
      new CPUTerrainFlowDistanceOperation({
        ...props,
        grid: {...GRID, affine: [1, 1, 0, 0, 1, 0]},
        cellSizeMode: 'geographic'
      })
  ).toThrow(/distance-validation.*axis-aligned/);
  expect(() => computeTerrainFlowDistanceOnCPU(new Uint32Array(3), {grid: GRID})).toThrow(
    /flowDirections/
  );
  expect(() =>
    computeTerrainFlowDistanceOnCPU(DIRECTIONS, {grid: GRID}, new Uint32Array(3))
  ).toThrow(/streams/);
  expect(() =>
    computeTerrainFlowDistanceOnCPU(DIRECTIONS, {grid: GRID}, undefined, new Float32Array(3))
  ).toThrow(/transitionCosts/);
});
