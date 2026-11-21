// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  addIsovistsRecipe,
  addVisibilityGraphRecipe,
  type UrbanFreeSpaceRegion
} from '../../../src/gpu-spatial-analysis/urban-morphology/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';

async function createFixture(name: string) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return new RecipeTestFixture(device, name);
}

function getComponentCount(format: string): number {
  return format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
}

/** Copies every view to a readable output and returns the raw words of each. */
async function readAll(fixture: RecipeTestFixture, views: Record<string, GraphDataView>) {
  const published = Object.entries(views).map(([name, view]) => {
    const output = fixture.output(`read-${name}`, view.format as 'float32', view.length);
    fixture.graph.add(
      createWGSLKernelNode(fixture.graph, {
        id: `read-${name}`,
        operation: 'visibility-index-readback',
        invocationCount: view.length * getComponentCount(view.format),
        bindings: [
          {
            name: 'source',
            view,
            type: view.format.startsWith('float') ? 'f32' : 'u32',
            access: 'read'
          },
          {
            name: 'output',
            view: output.view,
            type: view.format.startsWith('float') ? 'f32' : 'u32',
            access: 'read_write'
          }
        ],
        body: 'output[outputOffset+index]=source[sourceOffset+index];'
      })
    );
    return [name, view, output] as const;
  });
  fixture.run();
  const words: Record<string, number[]> = {};
  for (const [name, view, output] of published) {
    const count = view.length * getComponentCount(view.format);
    words[name] = view.format.startsWith('float')
      ? Array.from(await fixture.readFloat32(output, count))
      : Array.from(await fixture.readUint32(output, count));
  }
  return words;
}

function createLCG(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

/** Closed rectangular obstacles (counterclockwise rings) and viewpoints that avoid their interiors. */
function createCity(seed: number) {
  const next = createLCG(seed);
  const rectangles: number[][] = [];
  const segments: number[] = [];
  for (let building = 0; building < 25; building++) {
    const x = 90 * next();
    const y = 90 * next();
    const width = 2 + 6 * next();
    const depth = 2 + 6 * next();
    rectangles.push([x, y, width, depth]);
    segments.push(x, y, x + width, y, x + width, y, x + width, y + depth);
    segments.push(x + width, y + depth, x, y + depth, x, y + depth, x, y);
  }
  const positions: number[] = [];
  while (positions.length < 2 * 96) {
    const x = 100 * next();
    const y = 100 * next();
    const insideBuilding = rectangles.some(
      ([left, bottom, width, depth]) =>
        x > left && x < left + width && y > bottom && y < bottom + depth
    );
    if (!insideBuilding) positions.push(x, y);
  }
  // One viewpoint on a wall (boundary, 2) and one inside the first building (blocked with a region).
  const [left, bottom, width, depth] = rectangles[0];
  positions.splice(0, 4, left + 0.5 * width, bottom, left + 0.5 * width, bottom + 0.5 * depth);
  return {positions, segments};
}

type VisibilityCase = {positions: number[]; segments: number[]; region?: UrbanFreeSpaceRegion};

async function runVisibilityGraph(
  testCase: VisibilityCase,
  obstacleIndex: 'brute-force' | 'bvh' | 'auto',
  name: string
) {
  const fixture = await createFixture(name);
  try {
    const pointCount = testCase.positions.length / 2;
    const result = addVisibilityGraphRecipe(fixture.graph, {
      positions: fixture.input(
        'positions',
        Float32Array.from(testCase.positions),
        'float32x2',
        pointCount
      ),
      segments: fixture.input(
        'segments',
        Float32Array.from(testCase.segments),
        'float32x4',
        testCase.segments.length / 4
      ),
      maximumDistance: 30,
      edgeCapacity: 8192,
      tolerance: 0.002,
      region: testCase.region,
      obstacleIndex
    });
    const words = await readAll(fixture, {...result.outputs, status: result.status});
    return {words, provenance: result.provenance};
  } finally {
    fixture.destroy();
  }
}

it('visibility graph and viewpoint classification agree between the segment BVH and every segment', async () => {
  const city = createCity(5);
  const regions: (UrbanFreeSpaceRegion | undefined)[] = [
    undefined,
    {fillRule: 'even-odd', freeSpace: 'outside'},
    {fillRule: 'nonzero', freeSpace: 'inside'}
  ];
  for (const [regionIndex, region] of regions.entries()) {
    const testCase = {...city, region};
    const bruteForce = await runVisibilityGraph(testCase, 'brute-force', `vg-brute-${regionIndex}`);
    const bvh = await runVisibilityGraph(testCase, 'bvh', `vg-bvh-${regionIndex}`);
    expect(bruteForce.provenance.obstacleIndex).toBe('brute-force');
    expect(bvh.provenance.obstacleIndex).toBe('bvh');
    const classes = bruteForce.words['viewpointClassifications'];
    // Free viewpoints, the wall viewpoint (boundary) and, with a region, blocked ones all occur.
    expect(classes.some(word => word === 0)).toBe(true);
    expect(classes[0]).toBe(2);
    if (region?.freeSpace === 'inside') expect(classes.some(word => word === 1)).toBe(true);
    // With free space 'inside' almost every viewpoint is outside the rings, hence blocked.
    if (region?.freeSpace !== 'inside') {
      expect(bruteForce.words['edges'].some(word => word !== 0xffffffff)).toBe(true);
    }
    for (const name of Object.keys(bruteForce.words)) {
      expect(bvh.words[name], `${name} with region ${regionIndex}`).toEqual(bruteForce.words[name]);
    }
  }
});

it('visibility graph picks the segment BVH above the brute-force segment count', async () => {
  const city = createCity(5);
  expect((await runVisibilityGraph(city, 'auto', 'vg-auto-small')).provenance.obstacleIndex).toBe(
    'brute-force'
  );
  // 4097 segments: the 100 city walls plus degenerate far-away segments that block nothing.
  const far = Array.from({length: 4097 - city.segments.length / 4}, (_, row) => [
    1000 + row,
    1000,
    1000 + row,
    1001
  ]).flat();
  const large = {positions: city.positions, segments: [...city.segments, ...far]};
  const automatic = await runVisibilityGraph(large, 'auto', 'vg-auto-large');
  expect(automatic.provenance.obstacleIndex).toBe('bvh');
  const bruteForce = await runVisibilityGraph(large, 'brute-force', 'vg-auto-brute');
  for (const name of Object.keys(bruteForce.words)) {
    expect(automatic.words[name], name).toEqual(bruteForce.words[name]);
  }
});

it('viewpoint classification marks every viewpoint non-finite when one segment is non-finite, with either index', async () => {
  const city = createCity(9);
  const segments = [...city.segments];
  segments[10] = Number.NaN;
  for (const obstacleIndex of ['brute-force', 'bvh'] as const) {
    const {words} = await runVisibilityGraph(
      {positions: city.positions, segments},
      obstacleIndex,
      `vg-nan-${obstacleIndex}`
    );
    expect(new Set(words['viewpointClassifications'])).toEqual(new Set([3]));
  }
});

it('isovists classify the same viewpoints with the segment BVH as with every segment, per fill rule', async () => {
  const city = createCity(13);
  const pointCount = city.positions.length / 2;
  for (const region of [
    {fillRule: 'even-odd', freeSpace: 'outside'},
    {fillRule: 'nonzero', freeSpace: 'inside'}
  ] as const) {
    const results: Record<string, number[]> = {};
    for (const obstacleIndex of ['brute-force', 'bvh'] as const) {
      const fixture = await createFixture(`isovist-class-${region.fillRule}-${obstacleIndex}`);
      try {
        const result = addIsovistsRecipe(fixture.graph, {
          positions: fixture.input(
            'positions',
            Float32Array.from(city.positions),
            'float32x2',
            pointCount
          ),
          segments: fixture.input(
            'segments',
            Float32Array.from(city.segments),
            'float32x4',
            city.segments.length / 4
          ),
          rayCount: 8,
          maximumDistance: 20,
          tolerance: 0.002,
          region,
          obstacleIndex
        });
        const words = await readAll(fixture, {
          viewpointClassifications: result.outputs.viewpointClassifications,
          measures: result.outputs.localMeasures
        });
        results[obstacleIndex] = [...words['viewpointClassifications'], ...words['measures']];
      } finally {
        fixture.destroy();
      }
    }
    expect(results['bvh']).toEqual(results['brute-force']);
  }
});

it('visibility normalizedChoice is NaN for every viewpoint when one is invalid or fewer than 3 exist, else finite', async () => {
  // WEB-PIC-17: the NaN is the documented contract, not a division by zero. Viewpoints 0 and 1 are
  // separated by a wall and 2 sees both around its tip; viewpoint 3 sits on the wall (invalid).
  const withInvalid = await runVisibilityGraph(
    {positions: [-1, 0, 1, 0, 0, 1, 0, 0.2], segments: [0, -1, 0, 0.5]},
    'brute-force',
    'vg-normalized-invalid'
  );
  expect(withInvalid.words['viewpointClassifications']).toEqual([0, 0, 0, 2]);
  expect(withInvalid.words['status'][3]).toBe(1);
  expect(withInvalid.words['normalizedChoice'].every(Number.isNaN)).toBe(true);
  expect(withInvalid.words['choice'].every(Number.isNaN)).toBe(true);
  // The same viewpoints without the invalid one are finite: 2 * 1 / ((3 - 1) * (3 - 2)).
  const valid = await runVisibilityGraph(
    {positions: [-1, 0, 1, 0, 0, 1], segments: [0, -1, 0, 0.5]},
    'brute-force',
    'vg-normalized-valid'
  );
  expect(valid.words['normalizedChoice']).toEqual([0, 0, 1]);
  // A valid viewpoint that sees nothing scores 0, not NaN.
  const isolated = await runVisibilityGraph(
    {positions: [-1, 0, 1, 0, 0, 1, 9, 9], segments: [8, 8, 8, 10, 8, 10, 10, 10]},
    'brute-force',
    'vg-normalized-isolated'
  );
  expect(isolated.words['status'][3]).toBe(0);
  expect(isolated.words['normalizedChoice'].every(Number.isFinite)).toBe(true);
  // Fewer than 3 viewpoints: the normalisation is undefined.
  const pair = await runVisibilityGraph(
    {positions: [-1, 0, 1, 0], segments: [0, 5, 0, 6]},
    'brute-force',
    'vg-normalized-pair'
  );
  expect(pair.words['normalizedChoice']).toEqual([Number.NaN, Number.NaN]);
});
