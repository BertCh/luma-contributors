// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addVisibilityGraphRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/visibility-graph';
import {
  addUrbanPointGrid,
  getUrbanPointGridDeclarations,
  getUrbanPointGridRadiusWalk
} from '../../../src/gpu-spatial-analysis/urban-morphology/visibility-graph-point-grid';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';

type SearchStructure = 'grid' | 'brute-force';

async function readVisibility(
  positions: number[],
  searchStructure: SearchStructure,
  edgeCapacity = 8,
  maximumDistance = 2,
  walls = [1, -1, 1, 1, 5, 5, 6, 5]
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `visibility-${searchStructure}`);
  try {
    const result = addVisibilityGraphRecipe(fixture.graph, {
      positions: fixture.input(
        'positions',
        Float32Array.from(positions),
        'float32x2',
        positions.length / 2
      ),
      segments: fixture.input('walls', Float32Array.from(walls), 'float32x4', walls.length / 4),
      maximumDistance,
      edgeCapacity,
      obstacleIndex: 'bvh',
      searchStructure
    });
    const views: Record<string, GraphDataView> = {...result.outputs, status: result.status};
    const readbacks = Object.entries(views).map(([name, source]) => {
      const output = fixture.output(`read-${name}`, source.format as 'float32', source.length);
      const components = source.format.endsWith('x4') ? 4 : source.format.endsWith('x2') ? 2 : 1;
      const type = source.format.startsWith('float') ? 'f32' : 'u32';
      fixture.graph.add(
        createWGSLKernelNode(fixture.graph, {
          id: `read-${name}`,
          operation: 'visibility-grid-readback',
          invocationCount: source.length * components,
          bindings: [
            {name: 'source', view: source, type, access: 'read'},
            {name: 'output', view: output.view, type, access: 'read_write'}
          ],
          body: 'output[outputOffset+index]=source[sourceOffset+index];'
        })
      );
      return {name, output, components, source};
    });
    fixture.run();
    const values: Record<string, number[]> = {};
    for (const {name, output, components, source} of readbacks) {
      const length = source.length * components;
      values[name] = source.format.startsWith('float')
        ? await fixture.readFloat32(output, length)
        : await fixture.readUint32(output, length);
    }
    return values;
  } finally {
    fixture.destroy();
  }
}

it.each([
  {positions: [-2, 0, 0, 0, 2, 0, 4, 0, 20, 20]},
  {positions: [-2, 0, 0, 0, 1, 0, 4, 0, Number.NaN, 0]},
  {positions: [Number.NaN, 0, Number.POSITIVE_INFINITY, 0]},
  {positions: [0, 0, 0, 0, 0, 0]},
  {positions: [1000000, 1000000, 1000002, 1000000, 1000004, 1000000]}
])('grid sightlines preserve adjacency, compact edges and all exact metrics for %j', async ({
  positions
}) => {
  const reference = await readVisibility(positions, 'brute-force');
  const indexed = await readVisibility(positions, 'grid');
  expect(indexed).toEqual(reference);
});

it('grid visibility preserves the unclamped edge requirement when the output overflows', async () => {
  const positions = [0, 0, 0, 0.5, 0, 1, 0, 1.5];
  const reference = await readVisibility(positions, 'brute-force', 1);
  expect(reference['status'].slice(0, 3)).toEqual([1, 6, 1]);
  expect(await readVisibility(positions, 'grid', 1)).toEqual(reference);
});

it('the point-grid stage finds exact radius neighbours for ten thousand city viewpoints', async () => {
  // Tests the scalable candidate stage independently: the full recipe still has a 256-point
  // dense adjacency/Brandes contract. This is not a city-scale run of the full visibility recipe.
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'visibility-city-grid');
  try {
    const positions = new Float32Array(20000);
    const expected: number[] = [];
    for (let row = 0; row < 10000; row++) {
      const horizontal = row % 100;
      const vertical = Math.floor(row / 100);
      positions[2 * row] = horizontal;
      positions[2 * row + 1] = vertical;
      expected.push(
        Number(horizontal > 0) +
          Number(horizontal < 99) +
          Number(vertical > 0) +
          Number(vertical < 99)
      );
    }
    const points = fixture.input('city', positions, 'float32x2', 10000);
    const grid = addUrbanPointGrid(fixture.graph, {id: 'city', positions: points});
    const counts = fixture.output('neighbour-counts', 'uint32', 10000);
    fixture.graph.add(
      createWGSLKernelNode(fixture.graph, {
        id: 'radius-query',
        operation: 'visibility-city-grid',
        invocationCount: 10000,
        declarations: getUrbanPointGridDeclarations(grid),
        bindings: [
          {name: 'pointGrid', view: grid.data, type: 'u32', access: 'read'},
          {name: 'positions', view: points, type: 'f32', access: 'read'},
          {name: 'counts', view: counts.view, type: 'u32', access: 'read_write'}
        ],
        body: `let point=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
var total=0u;
${getUrbanPointGridRadiusWalk(
  grid,
  'point',
  '1.0',
  'other',
  `let candidate=vec2<f32>(positions[positionsOffset+2u*other],positions[positionsOffset+2u*other+1u]);
if(other!=index&&distance(point,candidate)<=1.0){total++;}`
)}
counts[countsOffset+index]=total;`
      })
    );
    fixture.run();
    expect(await fixture.readUint32(counts, 10000)).toEqual(expected);
  } finally {
    fixture.destroy();
  }
});

it.each([
  1, 255, 256
])('dense grid/reference compare every output at adjacent float32 radii (%i points)', async count => {
  const positions = Array.from({length: count}, (_, row) => [100 + 4 * row, 100]).flat();
  positions[0] = 0;
  positions[1] = 0;
  if (count > 1) {
    positions[2] = 1;
    positions[3] = 0;
    positions[4] = 0;
    positions[5] = new Float32Array(new Uint32Array([0x3f800001]).buffer)[0];
    positions[6] = -new Float32Array(new Uint32Array([0x3f7fffff]).buffer)[0];
    positions[7] = 0;
  }
  const exact = await readVisibility(positions, 'brute-force', 8, 1, [10000, 10000, 10001, 10000]);
  const indexed = await readVisibility(positions, 'grid', 8, 1, [10000, 10000, 10001, 10000]);
  expect(indexed).toEqual(exact);
  const expectedAdjacency = Array.from({length: count * count}, (_, pair) =>
    Number(
      count > 1 &&
        [
          [0, 1],
          [1, 0],
          [0, 3],
          [3, 0]
        ].some(([first, second]) => pair === first * count + second)
    )
  );
  expect(indexed['adjacency']).toEqual(expectedAdjacency);
  expect(indexed['status']).toEqual(count === 1 ? [0, 0, 0, 0] : [2, 2, 0, 0]);
  expect(indexed['viewpointClassifications']).toEqual(Array(count).fill(0));
  expect(indexed['choice']).toEqual(
    Array.from({length: count}, (_, row) => Number(count > 1 && row === 0))
  );
  expect(indexed['metrics'].slice(0, 4)).toEqual(
    count === 1 ? [0, 0, 0, Number.NaN] : [2, 2, 2, 1]
  );
  if (count > 1) expect(indexed['integration'].slice(0, 4)).toEqual([1, 0, 0, Infinity]);
}, 120000);
