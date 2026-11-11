// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeRatioConcaveHullOnCPU,
  CPURatioConcaveHullOperation
} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-ratio-concave-hull-operation';
import {
  GPURatioConcaveHullOperation,
  type RatioConcaveHullOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-ratio-concave-hull-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createLatticeFixture} from '../delaunay-tessellation/delaunay-reference';

function computeRetainedArea(
  positions: Float32Array,
  triangles: Uint32Array,
  validity: Uint32Array
): number {
  let area = 0;
  validity.forEach((valid, row) => {
    if (!valid) return;
    const [first, second, third] = triangles.subarray(row * 3, row * 3 + 3);
    area +=
      ((positions[2 * second] - positions[2 * first]) *
        (positions[2 * third + 1] - positions[2 * first + 1]) -
        (positions[2 * second + 1] - positions[2 * first + 1]) *
          (positions[2 * third] - positions[2 * first])) /
      2;
  });
  return area;
}

it('breaks equal-edge candidate ties by row and removes a hand-computed bottom notch', () => {
  const positions = Float32Array.of(0, 0, 4, 0, 4, 4, 0, 4, 2, 1);
  const result = computeRatioConcaveHullOnCPU(positions, {ratio: 0.2});
  // Four border candidates have equal size 4. Their common interior site lets only row 0 win.
  // Removing (0,1,4) takes area 2 from the square; all vertices then lie on the boundary.
  expect(Array.from(result.triangleValidity)).toEqual([0, 1, 1, 1, 0]);
  expect(computeRetainedArea(positions, result.triangles, result.triangleValidity)).toBe(14);
  expect(result.converged[0]).toBe(1);
  const boundary = [...result.edgeValidity].flatMap((valid, slot) =>
    valid ? [[...result.boundaryEdges.subarray(2 * slot, 2 * slot + 2)].join(',')] : []
  );
  expect(boundary.sort()).toEqual(['0,4', '1,2', '2,3', '3,0', '4,1']);
  const convex = computeRatioConcaveHullOnCPU(positions, {ratio: 1});
  expect(computeRetainedArea(positions, convex.triangles, convex.triangleValidity)).toBe(16);
});

it('hole seeding changes a lattice only when enabled, preserving all sites and a closed boundary', () => {
  const positions = createLatticeFixture(25, 5);
  const solid = computeRatioConcaveHullOnCPU(positions, {ratio: 0.2});
  const holes = computeRatioConcaveHullOnCPU(positions, {ratio: 0.2, allowHoles: true});
  expect(computeRetainedArea(positions, solid.triangles, solid.triangleValidity)).toBe(16);
  expect(computeRetainedArea(positions, holes.triangles, holes.triangleValidity)).toBeLessThan(16);
  const vertices = new Set<number>();
  const undirectedEdges = new Set<string>();
  let faceCount = 0;
  holes.triangleValidity.forEach((valid, row) => {
    if (!valid) return;
    faceCount++;
    const triangle = holes.triangles.subarray(3 * row, 3 * row + 3);
    for (let side = 0; side < 3; side++) {
      const first = triangle[side];
      const second = triangle[(side + 1) % 3];
      vertices.add(first);
      undirectedEdges.add(
        [first, second].sort((firstRow, secondRow) => firstRow - secondRow).join(',')
      );
    }
  });
  expect(vertices.size).toBe(25);
  // Euler characteristic of the retained planar triangle complex: at least one hole.
  expect(vertices.size - undirectedEdges.size + faceCount).toBeLessThan(1);
  const balance = new Int32Array(25);
  holes.edgeValidity.forEach((valid, slot) => {
    if (!valid) return;
    balance[holes.boundaryEdges[2 * slot]]++;
    balance[holes.boundaryEdges[2 * slot + 1]]--;
  });
  expect([...balance].every(value => value === 0)).toBe(true);
});

it('handles collinear/nonfinite sites and names malformed positions and ratio', () => {
  const line = computeRatioConcaveHullOnCPU(Float32Array.of(0, 0, 1, 0, 2, 0), {ratio: 0});
  expect(Array.from(line.triangleValidity)).toEqual([0]);
  expect(Array.from(line.edgeValidity)).toEqual([0, 0, 0]);
  expect(line.converged[0]).toBe(1);
  const nonfinite = computeRatioConcaveHullOnCPU(Float32Array.of(0, 0, 1, 0, Infinity, 1), {
    ratio: 0.2
  });
  expect(nonfinite.converged[0]).toBe(0);
  expect(() => computeRatioConcaveHullOnCPU(new Float32Array(3), {ratio: 0})).toThrow('positions');
  expect(() => computeRatioConcaveHullOnCPU(new Float32Array(6), {ratio: NaN})).toThrow('ratio');
});

it('validates both operation shapes and the GPU contributor ratio at binding', () => {
  const props: RatioConcaveHullOperationProps = {
    id: 'hull-check',
    positions: 'positions',
    pointCount: 5,
    ratio: 0.2,
    output: {
      triangles: 'triangles',
      triangleValidity: 'triangleValidity',
      boundaryEdges: 'boundaryEdges',
      edgeValidity: 'edgeValidity',
      status: {converged: 'converged'}
    }
  };
  for (const Operation of [CPURatioConcaveHullOperation, GPURatioConcaveHullOperation]) {
    expect(() => new Operation({...props, pointCount: 2})).toThrow();
    expect(
      () => new Operation({...props, output: {...props.output, triangles: 'positions'}})
    ).toThrow(/^hull-check/);
  }
  expect(() => new CPURatioConcaveHullOperation({...props, ratio: -0.1})).toThrow(
    /^hull-check.*ratio/
  );
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const operation = new GPURatioConcaveHullOperation({...props, ratio: 2});
  expect(() =>
    operation.createContributor(
      port =>
        createTransientView(
          graph,
          port,
          port === 'positions'
            ? 'float32x2'
            : port === 'triangles'
              ? 'uint32x3'
              : port === 'boundaryEdges'
                ? 'uint32x2'
                : 'uint32',
          port === 'converged' ? 1 : port === 'boundaryEdges' || port === 'edgeValidity' ? 15 : 5
        ) as never
    )
  ).toThrow(/ratio/);
});
