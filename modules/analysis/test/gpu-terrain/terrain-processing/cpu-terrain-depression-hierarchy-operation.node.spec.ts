// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeTerrainDepressionHierarchyOnCPU,
  CPUTerrainDepressionHierarchyOperation
} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-depression-hierarchy-operation';
import {GPUTerrainDepressionHierarchyOperation} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {
  makeHierarchyInput,
  makeHierarchyProps
} from './terrain-depression-hierarchy-operation-fixture';

const NONE = 0xffffffff;

it('constructs the deferred GPU contributor and rejects inconsistent stage-dependent shapes', () => {
  const props = makeHierarchyProps();
  const shapes = {
    elevation: {format: 'float32' as const, length: 7},
    validity: {format: 'uint32' as const, length: 7},
    outlets: {format: 'uint32' as const, length: 7},
    stages: {format: 'float32' as const, length: 4}
  };
  expect(() =>
    constructNetworkOperationContributor(new GPUTerrainDepressionHierarchyOperation(props), shapes)
  ).not.toThrow();
  expect(() =>
    constructNetworkOperationContributor(new GPUTerrainDepressionHierarchyOperation(props), {
      ...shapes,
      stages: {...shapes.stages, length: 3}
    })
  ).toThrow(/hierarchy-test.*output.curves/);
});

it('matches a hand-drawn four-leaf, three-merge tree and exact stage areas/volumes', () => {
  // 0--3--1--5--0--3--2: the two 3-level saddles merge at 5; stable ties pick cell 0.
  const result = computeTerrainDepressionHierarchyOnCPU(makeHierarchyInput(), makeHierarchyProps());
  expect(Array.from(result.nodes)).toEqual([
    4,
    0,
    0,
    1,
    5,
    4,
    NONE,
    1,
    4,
    2,
    NONE,
    1,
    5,
    6,
    6,
    1,
    6,
    0,
    0,
    1,
    6,
    4,
    6,
    1,
    NONE,
    0,
    0,
    1,
    NONE,
    NONE,
    NONE,
    0
  ]);
  expect(Array.from(result.levels)).toEqual([
    0,
    3,
    0,
    3,
    1,
    3,
    2,
    3,
    3,
    5,
    3,
    5,
    5,
    Infinity,
    NaN,
    NaN
  ]);
  expect(Array.from(result.cellNodes)).toEqual([0, 4, 2, 6, 1, 5, 3]);
  expect(Array.from(result.status)).toEqual([7, 7, 0]);
  expect(Array.from(result.requiredCount)).toEqual([7]);
  expect(Array.from(result.overflow)).toEqual([0]);
  expect(Array.from(result.curves)).toEqual([
    2,
    0,
    2,
    4,
    2,
    6,
    2,
    10,
    2,
    0,
    2,
    4,
    2,
    6,
    2,
    10,
    0,
    0,
    2,
    2,
    2,
    4,
    2,
    8,
    0,
    0,
    2,
    0,
    2,
    2,
    2,
    6,
    2,
    0,
    4,
    6,
    6,
    10,
    6,
    22,
    2,
    0,
    4,
    4,
    6,
    8,
    6,
    20,
    4,
    0,
    8,
    10,
    12,
    18,
    14,
    42,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN
  ]);
});

it('agrees with independent sublevel connectivity counts and direct root-volume integration', () => {
  const input = makeHierarchyInput(),
    props = makeHierarchyProps();
  const result = computeTerrainDepressionHierarchyOnCPU(input, props);
  for (let stageIndex = 0; stageIndex < input.stages.length; stageIndex++) {
    const stage = input.stages[stageIndex];
    // On a line, a new connected sublevel component starts after each gap.
    let components = 0;
    for (let cell = 0; cell < input.elevation.length; cell++)
      if (input.elevation[cell] <= stage && (cell === 0 || input.elevation[cell - 1] > stage))
        components++;
    let liveNodes = 0;
    for (let node = 0; node < result.status[0]; node++)
      if (result.levels[node * 2] <= stage && result.levels[node * 2 + 1] > stage) liveNodes++;
    expect(liveNodes).toBe(components);
    const wetHeights = Array.from(input.elevation).filter(height => height <= stage);
    expect(result.curves[(6 * 4 + stageIndex) * 2]).toBe(2 * wetHeights.length);
    expect(result.curves[(6 * 4 + stageIndex) * 2 + 1]).toBe(
      wetHeights.reduce((volume, height) => volume + 2 * (stage - height), 0)
    );
  }
});

it('invalidates all nodes on overflow, retains ownership, and supports barriers and empty queries', () => {
  const input = makeHierarchyInput(),
    props = makeHierarchyProps(2);
  const result = computeTerrainDepressionHierarchyOnCPU(input, props);
  expect(Array.from(result.status)).toEqual([2, 7, 1]);
  expect(Array.from(result.nodes)).toEqual([NONE, NONE, NONE, 0, NONE, NONE, NONE, 0]);
  expect(result.levels.every(Number.isNaN)).toBe(true);
  expect(result.curves.every(Number.isNaN)).toBe(true);
  expect(Array.from(result.cellNodes)).toEqual([0, 4, 2, 6, 1, 5, 3]);
  expect(result.requiredCount[0]).toBe(7);
  expect(result.overflow[0]).toBe(1);
  input.validity.fill(0);
  const empty = computeTerrainDepressionHierarchyOnCPU(input, props);
  expect(Array.from(empty.status)).toEqual([0, 0, 0]);
  expect(empty.cellNodes.every(word => word === NONE)).toBe(true);
  expect(empty.curves.every(Number.isNaN)).toBe(true);
  input.validity.fill(1);
  input.validity[3] = 0;
  const split = computeTerrainDepressionHierarchyOnCPU(input, makeHierarchyProps());
  expect(split.cellNodes[3]).toBe(NONE);
  expect(
    Array.from(split.nodes).filter((word, index) => index % 4 === 0 && word === NONE)
  ).toHaveLength(4); // Two roots and two unused nodes.
  const noStages = computeTerrainDepressionHierarchyOnCPU(
    {...input, stages: new Float32Array()},
    makeHierarchyProps()
  );
  expect(noStages.curves.length).toBe(0);
});

it('handles equal elevations and validates operation declarations and malformed inputs', () => {
  const input = makeHierarchyInput(),
    props = makeHierarchyProps();
  input.elevation.fill(-2);
  const result = computeTerrainDepressionHierarchyOnCPU(input, props);
  expect(Array.from(result.status)).toEqual([1, 1, 0]);
  expect(Array.from(result.cellNodes)).toEqual([0, 0, 0, 0, 0, 0, 0]);
  expect(Array.from(result.nodes.subarray(0, 4))).toEqual([NONE, 0, 0, 1]);
  for (const Operation of [
    GPUTerrainDepressionHierarchyOperation,
    CPUTerrainDepressionHierarchyOperation
  ]) {
    expect(() => new Operation({...props, nodeCapacity: 0})).toThrow(/^hierarchy-test/);
    expect(() => new Operation({...props, stageCount: 0})).toThrow(/^hierarchy-test/);
    expect(() => new Operation({...props, output: {...props.output, curves: 'stages'}})).toThrow(
      /^hierarchy-test/
    );
  }
  for (const name of ['elevation', 'outlets', 'validity'] as const)
    expect(() =>
      computeTerrainDepressionHierarchyOnCPU({...input, [name]: input[name].subarray(1)}, props)
    ).toThrow(new RegExp(name));
});
