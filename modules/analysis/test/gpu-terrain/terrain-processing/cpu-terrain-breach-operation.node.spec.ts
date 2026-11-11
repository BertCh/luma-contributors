// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeTerrainBreachOnCPU,
  CPUTerrainBreachOperation
} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-breach-operation';
import {GPUTerrainBreachOperation} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-breach-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {makeBreachInput, makeBreachProps} from './terrain-breach-operation-fixture';

const NONE = 0xffffffff;

it('constructs the deferred GPU contributor and checks its numeric validation', () => {
  const props = makeBreachProps();
  const shapes = {
    elevation: {format: 'float32' as const, length: 9},
    validity: {format: 'uint32' as const, length: 9},
    burnDepth: {format: 'float32' as const, length: 9},
    pitCells: {format: 'uint32' as const, length: 2}
  };
  expect(() =>
    constructNetworkOperationContributor(new GPUTerrainBreachOperation(props), shapes)
  ).not.toThrow();
  for (const overrides of [{maximumSteps: 0}, {maximumCutDepth: -1}, {elevationDecrement: 0}])
    expect(() =>
      constructNetworkOperationContributor(
        new GPUTerrainBreachOperation({...props, ...overrides}),
        shapes
      )
    ).toThrow(/^breach-test/);
});

it('matches hand-drawn multi-step pits, the lowest-outlet tie, breach and burn flags', () => {
  const result = computeTerrainBreachOnCPU(makeBreachInput(), makeBreachProps());
  expect(Array.from(result.pathCells)).toEqual([2, 1, 0, NONE, 6, 5, 4, NONE]);
  expect(Array.from(result.pathLengths)).toEqual([3, 3]);
  expect(Array.from(result.pitStatus)).toEqual([1, 1]);
  expect(Array.from(result.unresolvedCount)).toEqual([0]);
  expect(Array.from(result.elevation)).toEqual([0, 0.5, 1, 5, 0, 1.75, 2, 3.5, 0]);
  expect(Array.from(result.cutDepth)).toEqual([0, 3.5, 0, 0, 0, 2.25, 0, 0.5, 0]);
  expect(Array.from(result.modificationMask)).toEqual([0, 3, 0, 0, 0, 1, 0, 2, 0]);
});

it('agrees with exhaustive path enumeration on tiny two-dimensional search problems', () => {
  // This oracle enumerates walks; it uses no layered shortest-path recurrence or shared helper.
  const grid = {...makeBreachProps().grid, width: 5, height: 3};
  const options = {...makeBreachProps(), grid, maximumSteps: 3, maximumCutDepth: 9};
  for (const elevation of [
    Float32Array.of(0, 8, 9, 6, 0, 0, 4, 2, 5, 0, 0, 7, 9, 3, 0),
    Float32Array.of(0, 5, 9, 4, 0, 0, 4, 2, 3, 0, 0, 7, 9, 5, 0),
    Float32Array.of(9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9)
  ]) {
    const candidates: {path: number[]; cost: number}[] = [];
    const enumerateWalks = (path: number[], cost: number) => {
      if (path.length > options.maximumSteps) return;
      const last = path[path.length - 1];
      for (let cell = 0; cell < elevation.length; cell++) {
        if (
          cell === last ||
          Math.abs((cell % grid.width) - (last % grid.width)) > 1 ||
          Math.abs(Math.floor(cell / grid.width) - Math.floor(last / grid.width)) > 1
        )
          continue;
        const level = elevation[7] - path.length * options.elevationDecrement;
        const cut = Math.max(0, elevation[cell] - level);
        if (cut > options.maximumCutDepth) continue;
        const next = [...path, cell],
          nextCost = cost + cut;
        if (elevation[cell] <= level) candidates.push({path: next, cost: nextCost});
        enumerateWalks(next, nextCost);
      }
    };
    enumerateWalks([7], 0);
    candidates.sort(
      (first, second) =>
        first.cost - second.cost ||
        first.path.length - second.path.length ||
        first.path[first.path.length - 1] - second.path[second.path.length - 1]
    );
    const result = computeTerrainBreachOnCPU({elevation, pitCells: Uint32Array.of(7)}, options);
    if (!candidates.length) {
      expect(result.pitStatus[0]).toBe(2);
      expect(result.pathLengths[0]).toBe(0);
      continue;
    }
    const path = Array.from(result.pathCells.subarray(0, result.pathLengths[0]));
    const cost = path
      .slice(1)
      .reduce(
        (sum, cell, index) =>
          sum +
          Math.max(0, elevation[cell] - (elevation[7] - (index + 1) * options.elevationDecrement)),
        0
      );
    expect(cost).toBe(candidates[0].cost);
    expect(path.length).toBe(candidates[0].path.length);
    expect(path.length).toBeGreaterThan(2);
    expect(path[path.length - 1]).toBe(candidates[0].path[candidates[0].path.length - 1]);
  }
});

it('chooses the lowest predecessor ID when two cheapest routes share an outlet', () => {
  const props = makeBreachProps();
  const result = computeTerrainBreachOnCPU(
    {
      elevation: Float32Array.of(0, 4, 4, 4, 0, 0, 4, 2, 4, 0, 0, 4, 4, 4, 0),
      pitCells: Uint32Array.of(7)
    },
    {...props, grid: {...props.grid, width: 5, height: 3}}
  );
  expect(Array.from(result.pathCells)).toEqual([7, 1, 0, NONE]);
});

it('preserves paths when simultaneous edits or invalid burning break final descent', () => {
  const props = makeBreachProps(),
    input = makeBreachInput();
  const overlap = computeTerrainBreachOnCPU(
    {elevation: Float32Array.of(0, 4, 2, 1), pitCells: Uint32Array.of(2, 3)},
    {...props, grid: {...props.grid, width: 4}}
  );
  expect(Array.from(overlap.pathCells)).toEqual([2, 3, NONE, NONE, 3, 2, 1, 0]);
  expect(Array.from(overlap.pitStatus)).toEqual([2, 1]);
  expect(overlap.unresolvedCount[0]).toBe(1);
  input.burnDepth[1] = -1;
  const burned = computeTerrainBreachOnCPU(input, props);
  expect(burned.elevation[1]).toBeNaN();
  expect(burned.cutDepth[1]).toBeNaN();
  expect(burned.modificationMask[1]).toBe(5);
  expect(Array.from(burned.pitStatus)).toEqual([2, 1]);
  expect(burned.pathLengths[0]).toBe(3);
});

it('handles invalid pits, barriers, infeasible searches, empty pits and unrepresentable output steps', () => {
  const input = makeBreachInput(),
    props = makeBreachProps();
  expect(
    Array.from(computeTerrainBreachOnCPU(input, {...props, maximumCutDepth: 1}).pitStatus)
  ).toEqual([2, 2]);
  input.validity[2] = 0;
  input.pitCells[1] = NONE;
  const invalid = computeTerrainBreachOnCPU(input, props);
  expect(Array.from(invalid.pitStatus)).toEqual([0, 0]);
  expect(invalid.elevation[2]).toBeNaN();
  expect(invalid.pathCells.every(cell => cell === NONE)).toBe(true);
  const burnOnly = computeTerrainBreachOnCPU(
    {...makeBreachInput(), pitCells: new Uint32Array()},
    props
  );
  expect(burnOnly.pathCells.length).toBe(0);
  expect(burnOnly.elevation[1]).toBe(3.75);
  expect(burnOnly.modificationMask[1]).toBe(2);
  const fine = computeTerrainBreachOnCPU(
    {elevation: Float32Array.of(1e8, 1e8 + 8, 1e8 - 8), pitCells: Uint32Array.of(0)},
    {...props, grid: {...props.grid, width: 3}, elevationDecrement: 0.1, maximumCutDepth: 10}
  );
  expect(fine.pathLengths[0]).toBe(3);
  expect(fine.pitStatus[0]).toBe(2);
  expect(fine.cutDepth[1]).toBeCloseTo(8.1, 5);
});

it('validates both operation declarations, CPU options and malformed arrays', () => {
  const props = makeBreachProps();
  for (const Operation of [GPUTerrainBreachOperation, CPUTerrainBreachOperation]) {
    expect(() => new Operation({...props, pitCount: 0})).toThrow(/^breach-test/);
    expect(
      () => new Operation({...props, output: {...props.output, elevation: 'elevation'}})
    ).toThrow(/^breach-test/);
  }
  expect(() => new CPUTerrainBreachOperation({...props, maximumSteps: 0})).toThrow(
    /breach-test.*maximumSteps/
  );
  expect(() => new CPUTerrainBreachOperation({...props, maximumCutDepth: -1})).toThrow(
    /breach-test.*maximumCutDepth/
  );
  expect(() => new CPUTerrainBreachOperation({...props, elevationDecrement: 0})).toThrow(
    /breach-test.*elevationDecrement/
  );
  expect(() => new CPUTerrainBreachOperation({...props, elevationDecrement: 1e-50})).not.toThrow();
  const input = makeBreachInput();
  for (const name of ['elevation', 'validity', 'burnDepth'] as const)
    expect(() =>
      computeTerrainBreachOnCPU({...input, [name]: input[name].subarray(1)}, props)
    ).toThrow(new RegExp(name));
});
