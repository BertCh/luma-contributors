// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  CPUPolygonTriangulationOperation,
  computePolygonTriangulationOnCPU,
  type CPUPolygonTriangulationInput
} from '../../../src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation';
import {GPUPolygonTriangulationOperation} from '../../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  OPERATION_POLYGONS,
  createPaddingTriangulationInput,
  CLAMPED_RING_END_CASES,
  createClampedRingEndInput,
  getTriangulationOperationProps,
  packOperationPolygons
} from './polygon-triangulation-operation-cases';
import {
  getTilingDefect,
  getTriangleAreaSum,
  type SearchPoint
} from './polygon-triangulation-search-cases';
import touchingFixture from '../../oracles/fixtures/polygon-triangulation-touching-rings.json';

function checkPolygonTilings(polygons: SearchPoint[][][], expectedAreas?: number[]): void {
  const input = packOperationPolygons(polygons);
  const props = getTriangulationOperationProps(input);
  const result = computePolygonTriangulationOnCPU(input, props);
  let vertexStart = 0;
  for (let polygon = 0; polygon < polygons.length; polygon++) {
    expect(result.valid[polygon], `polygon ${polygon}`).toBe(1);
    const triangles: number[][] = [];
    const start = result.triangleOffsets[polygon] * 3;
    for (let triangle = 0; triangle < result.triangleCount[polygon]; triangle++) {
      triangles.push(
        Array.from(
          result.indices.subarray(start + 3 * triangle, start + 3 * triangle + 3),
          vertex => vertex - vertexStart
        )
      );
    }
    // An independent boundary-chain/coverage invariant, not a second Earcut implementation.
    expect(getTilingDefect(polygons[polygon], triangles), `polygon ${polygon}`).toBeUndefined();
    if (expectedAreas)
      expect(getTriangleAreaSum(polygons[polygon], triangles)).toBe(expectedAreas[polygon]);
    vertexStart += polygons[polygon].flat().length;
  }
}

it('tiles a concave shell with a hole, another polygon and a hole touching an edge', () => {
  // Hand areas: (10*5 + 4*3) - 2*2, shoelace area 12, 100 - 3.
  checkPolygonTilings(OPERATION_POLYGONS, [58, 12, 97]);
});

it('tiles every committed Shapely-valid touching-ring scene and preserves its Shapely area', () => {
  const fixture = touchingFixture as unknown as {
    review: {rings: SearchPoint[][]; area: number}[];
    families: {rings: SearchPoint[][]; area: number}[];
  };
  const scenes = [...fixture.review, ...fixture.families];
  expect(scenes.length).toBe(1059);
  checkPolygonTilings(
    scenes.map(scene => scene.rings),
    scenes.map(scene => scene.area)
  );
});

it('keeps exact orientation on finite coordinates across a span the GPU cannot certify', () => {
  const input = packOperationPolygons([
    [
      [
        [0, 0],
        [2 ** 100, 0],
        [0, 2 ** -100]
      ]
    ]
  ]);
  const result = computePolygonTriangulationOnCPU(input, {indexCapacity: 3});
  expect(Array.from(result.valid)).toEqual([1]);
  expect(Array.from(result.triangleCount)).toEqual([1]);
  // Unique triangle, positive orientation, and area 1/2 regardless of clipping order.
  expect([...result.indices].sort((first, second) => first - second)).toEqual([0, 1, 2]);
});

it('keeps empty and collapsed polygons empty and pads failed slices with their first vertex', () => {
  const input = packOperationPolygons([
    [],
    [
      [
        [0, 0],
        [1, 0],
        [2, 0]
      ]
    ],
    [
      [
        [10, 0],
        [14, 0],
        [14, 4],
        [10, 4]
      ]
    ]
  ]);
  const result = computePolygonTriangulationOnCPU(input, {indexCapacity: 3});
  expect(Array.from(result.valid)).toEqual([1, 1, 0]);
  expect(Array.from(result.triangleCount)).toEqual([0, 0, 0]);
  expect(Array.from(result.triangleOffsets)).toEqual([0, 0, 0, 2]);
  expect(Array.from(result.indices)).toEqual([3, 3, 3]);
});

it('keeps spare capacity zero, repeated closing vertices and degenerate holes out of real counts', () => {
  const input = packOperationPolygons([
    [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4],
        [0, 0]
      ],
      [
        [1, 1],
        [2, 1],
        [3, 1]
      ]
    ]
  ]);
  const result = computePolygonTriangulationOnCPU(input, {indexCapacity: 30});
  expect(Array.from(result.valid)).toEqual([1]);
  expect(Array.from(result.triangleOffsets)).toEqual([0, 3]);
  expect(Array.from(result.triangleCount)).toEqual([2]);
  expect(Array.from(result.indices.subarray(6))).toEqual(new Array(24).fill(0));
});

it('rejects inconsistent arrays with the argument name and flags a nonfinite live ring', () => {
  const input = packOperationPolygons(OPERATION_POLYGONS);
  const options = getTriangulationOperationProps(input);
  const malformed: [keyof CPUPolygonTriangulationInput, Float32Array | Uint32Array][] = [
    ['positions', new Float32Array(3)],
    ['featureOffsets', new Uint32Array()],
    ['polygonOffsets', Uint32Array.of(0, 2, 1)],
    ['ringOffsets', Uint32Array.of(0, 4, 2)]
  ];
  for (const [argument, values] of malformed) {
    expect(() => computePolygonTriangulationOnCPU({...input, [argument]: values}, options)).toThrow(
      argument
    );
  }
  const nonfinite = packOperationPolygons([
    [
      [
        [0, 0],
        [4, 0],
        [Infinity, 4],
        [0, 4]
      ]
    ]
  ]);
  const result = computePolygonTriangulationOnCPU(nonfinite, {indexCapacity: 6});
  expect(Array.from(result.valid)).toEqual([0]);
  expect(Array.from(result.triangleCount)).toEqual([0]);
  expect(Array.from(result.indices)).toEqual([0, 0, 0, 0, 0, 0]);
});

it('clamps a ring end for reads but preserves its original slice capacity', () => {
  const input = packOperationPolygons([
    [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4]
      ]
    ]
  ]);
  input.ringOffsets[1] = 7;
  const result = computePolygonTriangulationOnCPU(input, {indexCapacity: 15});
  expect(Array.from(result.valid)).toEqual([1]);
  expect(Array.from(result.triangleCount)).toEqual([2]);
  expect(Array.from(result.triangleOffsets)).toEqual([0, 5]);
  expect(Array.from(result.indices.subarray(6))).toEqual(new Array(9).fill(0));
});

for (const scene of CLAMPED_RING_END_CASES) {
  it(`preserves failure padding at the signed slot boundary for ring end ${scene.ringEnd}`, () => {
    const result = computePolygonTriangulationOnCPU(createClampedRingEndInput(scene.ringEnd), {
      indexCapacity: 6
    });
    expect(Array.from(result.valid)).toEqual([0]);
    expect(Array.from(result.triangleCount)).toEqual([0]);
    expect(Array.from(result.triangleOffsets)).toEqual([0, scene.triangleEnd]);
    expect(Array.from(result.indices)).toEqual(scene.indices);
  });
}

it('validates both operation shapes at construction and contributor props when views are bound', () => {
  const input = packOperationPolygons(OPERATION_POLYGONS);
  const props = getTriangulationOperationProps(input);
  for (const Operation of [CPUPolygonTriangulationOperation, GPUPolygonTriangulationOperation]) {
    expect(() => new Operation({...props, indexCapacity: 0})).toThrow(
      /^polygon-twin.*indices.length/
    );
    expect(() => new Operation({...props, polygonCount: -1})).toThrow(
      /^polygon-twin.*valid.length/
    );
    expect(
      () => new Operation({...props, output: {...props.output, indices: 'positions'}})
    ).toThrow(/^polygon-twin/);
  }
  // The GPU operation adds no contributor validation: construction with resolved views owns it.
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const views = {
    positions: createTransientView(graph, 'positions', 'float32x2', input.positions.length / 2),
    featureOffsets: createTransientView(graph, 'features', 'uint32', input.featureOffsets.length),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', input.polygonOffsets.length),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', input.ringOffsets.length),
    indices: createTransientView(graph, 'indices', 'uint32', props.indexCapacity),
    valid: createTransientView(graph, 'valid', 'uint32', props.polygonCount),
    triangleCount: createTransientView(graph, 'counts', 'uint32', props.polygonCount),
    triangleOffsets: createTransientView(graph, 'offsets', 'uint32', props.polygonCount + 1)
  };
  const operation = new GPUPolygonTriangulationOperation({...props, maximumWork: 0});
  expect(() =>
    operation.createContributor(port => views[port as keyof typeof views] as never)
  ).toThrow(/^polygon-twin.*maximumWork/);
  expect(
    () =>
      new CPUPolygonTriangulationOperation({
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      })
  ).toThrow(/polygon-twin/);
  // A GPU work budget is deliberately unused by the CPU twin.
  expect(() => new CPUPolygonTriangulationOperation({...props, maximumWork: 0})).not.toThrow();
});

it('CPU triangulation ignores garbage before the first ring and after the last', () => {
  const reference = computePolygonTriangulationOnCPU(createPaddingTriangulationInput(), {
    indexCapacity: 128
  });
  expect(Array.from(reference.valid)).toEqual([1, 1, 1]);
  expect(Array.from(reference.triangleCount)).toEqual([0, 8, 2]);
  for (const value of [Number.NaN, 3e30, -3e30]) {
    const input = createPaddingTriangulationInput(2, 3, value);
    const result = computePolygonTriangulationOnCPU(input, {indexCapacity: 128});
    const usedWords = 3 * result.triangleOffsets[3];
    expect(Array.from(result.indices.slice(0, usedWords), vertex => vertex - 2)).toEqual(
      Array.from(reference.indices.slice(0, usedWords))
    );
    expect(result.valid).toEqual(reference.valid);
    expect(result.triangleCount).toEqual(reference.triangleCount);
    expect(result.triangleOffsets).toEqual(reference.triangleOffsets);
  }
});
