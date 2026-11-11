// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeConstrainedDelaunayTriangulationOnCPU,
  CPUConstrainedDelaunayTriangulationOperation
} from '../../../src/gpu-spatial-analysis/polygon-triangulation/cpu-constrained-delaunay-triangulation-operation';
import {GPUConstrainedDelaunayTriangulationOperation} from '../../../src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation-operation';
import {computePolygonTriangulationOnCPU} from '../../../src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {checkTriangulation, toTriangleRows} from '../delaunay-tessellation/delaunay-reference';
import {getTilingDefect} from './polygon-triangulation-search-cases';
import {
  OPERATION_POLYGONS,
  CLAMPED_RING_END_CASES,
  createClampedRingEndInput,
  packOperationPolygons,
  getTriangulationOperationProps
} from './polygon-triangulation-operation-cases';

it('flips the illegal diagonal of a hand-computed noncocircular quadrilateral', () => {
  const input = packOperationPolygons([
    [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 1]
      ]
    ]
  ]);
  const initial = computePolygonTriangulationOnCPU(input, {indexCapacity: 6});
  const result = computeConstrainedDelaunayTriangulationOnCPU(input, {indexCapacity: 6});
  expect(Array.from(result.indices)).not.toEqual(Array.from(initial.indices));
  // The circle through rows 0,2,3 is centred at (3.5,0.5), radius sqrt(12.5): row 1 is inside.
  // The correct diagonal is 1--3 and the kernel's pair/side scan writes these two triangles.
  expect(Array.from(result.indices)).toEqual([3, 1, 2, 1, 3, 0]);
  expect(checkTriangulation(input.positions, toTriangleRows(result.indices))).toEqual({
    degenerate: 0,
    circleViolations: 0,
    duplicateEdges: 0,
    areaDifference: 0n
  });
  expect(Array.from(result.converged)).toEqual([1]);
});

it('preserves hole boundaries and polygon slices on the multi-polygon tiling', () => {
  const input = packOperationPolygons(OPERATION_POLYGONS);
  const props = getTriangulationOperationProps(input);
  const result = computeConstrainedDelaunayTriangulationOnCPU(input, props);
  let vertexStart = 0;
  OPERATION_POLYGONS.forEach((rings, polygon) => {
    const start = 3 * result.triangleOffsets[polygon];
    const triangles = toTriangleRows(
      result.indices.subarray(start, start + 3 * result.triangleCount[polygon])
    ).map(triangle => triangle.map(vertex => vertex - vertexStart));
    expect(getTilingDefect(rings, triangles)).toBeUndefined();
    vertexStart += rings.flat().length;
  });
  expect(Array.from(result.converged)).toEqual([1, 1, 1]);
});

it('leaves cocircular diagonals unchanged, reports failed slices and rejects inconsistent arrays', () => {
  const input = packOperationPolygons([
    [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4]
      ]
    ],
    []
  ]);
  const options = {indexCapacity: 6};
  const initial = computePolygonTriangulationOnCPU(input, options);
  const result = computeConstrainedDelaunayTriangulationOnCPU(input, options);
  expect(Array.from(result.indices)).toEqual(Array.from(initial.indices));
  expect(Array.from(result.converged)).toEqual([1, 1]);
  const overflow = computeConstrainedDelaunayTriangulationOnCPU(input, {indexCapacity: 3});
  expect(Array.from(overflow.valid)).toEqual([0, 1]);
  expect(Array.from(overflow.converged)).toEqual([0, 1]);
  expect(() =>
    computeConstrainedDelaunayTriangulationOnCPU(
      {...input, positions: new Float32Array(3)},
      options
    )
  ).toThrow('positions');
  expect(() =>
    computeConstrainedDelaunayTriangulationOnCPU(
      {...input, polygonOffsets: Uint32Array.of(0, 99)},
      options
    )
  ).toThrow('polygonOffsets');
});

for (const scene of CLAMPED_RING_END_CASES) {
  it(`retains signed-slot failure padding and nonconvergence for ring end ${scene.ringEnd}`, () => {
    const result = computeConstrainedDelaunayTriangulationOnCPU(
      createClampedRingEndInput(scene.ringEnd),
      {indexCapacity: 6}
    );
    expect(Array.from(result.valid)).toEqual([0]);
    expect(Array.from(result.triangleCount)).toEqual([0]);
    expect(Array.from(result.triangleOffsets)).toEqual([0, scene.triangleEnd]);
    expect(Array.from(result.indices)).toEqual(scene.indices);
    expect(Array.from(result.converged)).toEqual([0]);
  });
}

it('validates both operation shapes and GPU maximumFlips when contributor views are bound', () => {
  const input = packOperationPolygons([
    [
      [
        [0, 0],
        [4, 0],
        [4, 1],
        [0, 4]
      ]
    ]
  ]);
  const props = {
    ...getTriangulationOperationProps(input),
    output: {...getTriangulationOperationProps(input).output, status: {converged: 'converged'}}
  };
  for (const Operation of [
    CPUConstrainedDelaunayTriangulationOperation,
    GPUConstrainedDelaunayTriangulationOperation
  ]) {
    expect(() => new Operation({...props, indexCapacity: 0})).toThrow(/^polygon-twin/);
    expect(
      () => new Operation({...props, output: {...props.output, status: {converged: 'positions'}}})
    ).toThrow(/^polygon-twin/);
  }
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const operation = new GPUConstrainedDelaunayTriangulationOperation({...props, maximumFlips: 0});
  expect(() =>
    operation.createContributor(
      port =>
        createTransientView(
          graph,
          port,
          port === 'positions' ? 'float32x2' : 'uint32',
          port === 'positions' ? 4 : port.endsWith('Offsets') ? 2 : port === 'indices' ? 6 : 1
        ) as never
    )
  ).toThrow(/maximumFlips/);
});
