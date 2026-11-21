// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addVisibilityGraphRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';

// URB-103. Five viewpoints: a path (0,0), (1,0), (2,0) with radius 1.1, a viewpoint on the wall
// x = 2.5 (classification 2, half a unit from the third viewpoint) and a non-finite viewpoint
// (classification 3). Hand-worked: the only sightlines within 1.1 are 0-1 and 1-2 (length 1), and
// the wall lies beyond both. The dense reference marks only the two invalid adjacency rows and
// publishes the two valid pairs with invalid count 2; the spatial CSR route now does the same:
// two empty rows, four directed slots, invalidCount 2. Before the fix it published count 0 and
// every offset 0 as soon as one viewpoint was invalid.
const POSITIONS = new Float32Array([0, 0, 1, 0, 2, 0, 2.5, 0, Number.NaN, 0]);
const WALL = new Float32Array([2.5, -1, 2.5, 1]);

async function createFixture(name: string) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return new RecipeTestFixture(device, name);
}

it('URB-103 an invalid viewpoint empties only its own CSR row', async () => {
  const fixture = await createFixture('csr-invalid-rows');
  try {
    const offsets = fixture.output('offsets', 'uint32', 6);
    const neighbors = fixture.output('neighbors', 'uint32', 16);
    const degree = fixture.output('degree', 'uint32', 5);
    const classifications = fixture.output('classifications', 'uint32', 5);
    const count = fixture.output('count', 'uint32', 1);
    const invalidCount = fixture.output('invalid-count', 'uint32', 1);
    const overflow = fixture.output('overflow', 'uint32', 1);
    addVisibilityGraphRecipe(fixture.graph, {
      mode: 'csr',
      positions: fixture.input('positions', POSITIONS, 'float32x2', 5),
      segments: fixture.input('wall', WALL, 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 16,
      output: {
        offsets: offsets.view,
        neighbors: neighbors.view,
        degree: degree.view,
        viewpointClassifications: classifications.view,
        status: {count: count.view, invalidCount: invalidCount.view, overflow: overflow.view}
      }
    });
    fixture.run();
    expect(await fixture.readUint32(classifications, 5)).toEqual([0, 0, 0, 2, 3]);
    expect(await fixture.readUint32(invalidCount, 1)).toEqual([2]);
    expect(await fixture.readUint32(overflow, 1)).toEqual([0]);
    expect(await fixture.readUint32(count, 1)).toEqual([4]);
    const offsetValues = await fixture.readUint32(offsets, 6);
    expect(offsetValues).toEqual([0, 1, 3, 4, 4, 4]);
    const neighborValues = await fixture.readUint32(neighbors, 16);
    const rows = offsetValues
      .slice(0, -1)
      .map((offset, row) =>
        neighborValues.slice(offset, offsetValues[row + 1]).sort((first, second) => first - second)
      );
    expect(rows).toEqual([[1], [0, 2], [1], [], []]);
    expect(await fixture.readUint32(degree, 5)).toEqual([1, 2, 1, 0, 0]);
  } finally {
    fixture.destroy();
  }
});

it('URB-103 the dense reference keeps the same valid pairs beside invalid viewpoints', async () => {
  const fixture = await createFixture('dense-invalid-rows');
  try {
    const edges = fixture.output('edges', 'uint32x2', 4);
    const status = fixture.output('status', 'uint32x4', 1);
    addVisibilityGraphRecipe(fixture.graph, {
      positions: fixture.input('positions', POSITIONS, 'float32x2', 5),
      segments: fixture.input('wall', WALL, 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 4,
      output: {edges: edges.view, status: status.view}
    });
    fixture.run();
    // [count, requiredCount, overflow, invalid]: two undirected pairs, the CSR route's four slots.
    expect(await fixture.readUint32(status, 4)).toEqual([2, 2, 0, 2]);
    expect((await fixture.readUint32(edges, 8)).slice(0, 4)).toEqual([0, 1, 1, 2]);
  } finally {
    fixture.destroy();
  }
});
