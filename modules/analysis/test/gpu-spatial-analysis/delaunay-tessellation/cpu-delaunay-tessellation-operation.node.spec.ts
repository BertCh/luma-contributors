// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeDelaunayTessellationOnCPU,
  CPUDelaunayTessellationOperation
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/cpu-delaunay-tessellation-operation';
import {
  GPUDelaunayTessellationOperation,
  type DelaunayTessellationOperationProps
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  SCIPY_FIXTURES,
  createUniformFixture,
  digestTriangleSet,
  toTriangleRows,
  checkTriangulation,
  createLatticeFixture,
  countHullBoundarySites
} from './delaunay-reference';

it('matches frozen SciPy triangle sets on generic and very thin point clouds', () => {
  for (const fixture of Object.values(SCIPY_FIXTURES)) {
    const positions = createUniformFixture(
      fixture.seed,
      fixture.count,
      fixture.width,
      fixture.height
    );
    const result = computeDelaunayTessellationOnCPU({positions}, {capacity: 2 * fixture.count});
    const triangles = toTriangleRows(result.triangles.subarray(0, 3 * result.count[0]));
    expect(result.requiredCount[0]).toBe(fixture.triangles);
    expect(digestTriangleSet(triangles)).toBe(fixture.digest);
    expect(result.overflow[0]).toBe(0);
    expect(result.converged[0]).toBe(1);
  }
});

it('tiles a cocircular lattice with empty circumcircles and every collinear hull site retained', () => {
  const positions = createLatticeFixture(25, 5);
  const result = computeDelaunayTessellationOnCPU({positions}, {capacity: 50});
  const triangles = toTriangleRows(result.triangles.subarray(0, 3 * result.count[0]));
  expect(result.count[0]).toBe(2 * 25 - 2 - countHullBoundarySites(positions));
  expect(checkTriangulation(positions, triangles)).toEqual({
    degenerate: 0,
    circleViolations: 0,
    duplicateEdges: 0,
    areaDifference: 0n
  });
  // Hand-derived lift cofactor: in the square, row 0's raised lift selects diagonal 1--3.
  const square = computeDelaunayTessellationOnCPU(
    {positions: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1)},
    {capacity: 2}
  );
  expect(Array.from(square.triangles)).toEqual([0, 1, 3, 1, 2, 3]);
});

it('handles groups, ignored labels, duplicates, overflow and per-group nonfinite cancellation', () => {
  const positions = Float32Array.of(
    0,
    0,
    2,
    0,
    0,
    2,
    10,
    0,
    12,
    0,
    10,
    2,
    0,
    -0,
    99,
    99,
    Infinity,
    1
  );
  const groupIds = Uint32Array.of(0, 0, 0, 1, 1, 1, 0, 9, 2);
  const result = computeDelaunayTessellationOnCPU(
    {positions, groupIds},
    {capacity: 1, groupCount: 3}
  );
  expect(Array.from(result.triangles)).toEqual([0, 1, 2]);
  expect(result.count[0]).toBe(1);
  expect(result.requiredCount[0]).toBe(2);
  expect(result.overflow[0]).toBe(1);
  expect(result.duplicateCount[0]).toBe(1);
  expect(result.invalidCount[0]).toBe(0);
  expect(Array.from(result.groupTriangleCounts)).toEqual([1, 1, 0]);
  expect(Array.from(result.groupTriangleOffsets)).toEqual([0, 1, 2]);
  expect(Array.from(result.groupConverged)).toEqual([1, 1, 0]);
  expect(result.iterationLimitReached[0]).toBe(0);
});

it('uses the row-order lift on eight exactly cocircular sites and retains a late lone apex', () => {
  const positions = Float32Array.of(5, 0, 3, 4, 0, 5, -3, 4, -5, 0, -3, -4, 0, -5, 3, -4);
  const result = computeDelaunayTessellationOnCPU({positions}, {capacity: 6});
  // Raising the lifts in row order selects the fan about the last vertex of this cyclic order.
  expect(toTriangleRows(result.triangles)).toEqual(
    Array.from({length: 6}, (_, row) => [row, row + 1, 7])
  );
  const line = Float32Array.from({length: 40}, (_, component) =>
    component === 38 ? 9.25 : component === 39 ? 1 : component % 2 === 0 ? component / 2 : 0
  );
  const apex = computeDelaunayTessellationOnCPU({positions: line}, {capacity: 18});
  expect(apex.count[0]).toBe(18);
  expect(new Set(apex.triangles).size).toBe(20);
  expect(checkTriangulation(line, toTriangleRows(apex.triangles))).toEqual({
    degenerate: 0,
    circleViolations: 0,
    duplicateEdges: 0,
    areaDifference: 0n
  });
});

it('accepts finite wide exponent spans, and leaves collinear and empty inputs empty', () => {
  const wide = computeDelaunayTessellationOnCPU(
    {positions: Float32Array.of(0, 0, 2 ** 100, 0, 0, 2 ** -100)},
    {capacity: 1}
  );
  expect(wide.count[0]).toBe(1);
  expect(wide.invalidCount[0]).toBe(0);
  for (const positions of [new Float32Array(), Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0)]) {
    const result = computeDelaunayTessellationOnCPU({positions}, {capacity: 0});
    expect(result.count[0]).toBe(0);
    expect(result.requiredCount[0]).toBe(0);
    expect(result.converged[0]).toBe(1);
  }
});

it('names inconsistent arrays and validates CPU and GPU operation construction', () => {
  expect(() =>
    computeDelaunayTessellationOnCPU({positions: new Float32Array(3)}, {capacity: 1})
  ).toThrow('positions');
  expect(() =>
    computeDelaunayTessellationOnCPU(
      {positions: new Float32Array(6), groupIds: Uint32Array.of(0)},
      {capacity: 1, groupCount: 1}
    )
  ).toThrow('groupIds');
  expect(() =>
    computeDelaunayTessellationOnCPU({positions: new Float32Array()}, {capacity: 1, groupCount: 1})
  ).toThrow('groupIds');
  const props: DelaunayTessellationOperationProps = {
    id: 'delaunay-check',
    positions: 'positions',
    capacity: 4,
    output: {
      triangles: 'triangles',
      status: {count: 'count', requiredCount: 'requiredCount', overflow: 'overflow'}
    }
  };
  for (const Operation of [CPUDelaunayTessellationOperation, GPUDelaunayTessellationOperation]) {
    expect(() => new Operation({...props, capacity: -1})).toThrow(/^delaunay-check/);
    expect(
      () => new Operation({...props, output: {...props.output, triangles: 'positions'}})
    ).toThrow(/^delaunay-check/);
  }
  expect(
    () => new CPUDelaunayTessellationOperation({...props, groups: {ids: 'ids', count: 0}})
  ).toThrow(/groups.count/);
  // Contributor validation is performed at view binding, as required by the operation contract.
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const operation = new GPUDelaunayTessellationOperation({
    ...props,
    groups: {ids: 'ids', count: 1}
  });
  expect(() =>
    operation.createContributor(
      port =>
        createTransientView(
          graph,
          port,
          port === 'positions' ? 'float32x2' : port === 'triangles' ? 'uint32x3' : 'uint32',
          port === 'positions' ? 3 : 1
        ) as never
    )
  ).toThrow(/groups.ids/);
});

it('reports distinct-site iteration counts through both the pure function and the CPU port', async () => {
  const ordinaryPositions = Float32Array.of(
    0,
    0,
    4,
    0,
    4,
    4,
    0,
    4,
    2,
    1,
    10,
    0,
    13,
    0,
    13,
    3,
    10,
    3,
    10,
    0
  );
  // Hand counts: ordinary has ten rows with row 9 repeating row 5; the review's collinear
  // input has four rows with row 3 repeating row 2; four equal rows own just one site.
  const scenes = [
    {positions: ordinaryPositions, expectedIterations: 9},
    {positions: Float32Array.of(0, 0, 1, 0, 2, 0, 2, 0), expectedIterations: 3},
    {positions: Float32Array.of(2, 2, 2, 2, 2, 2, 2, 2), expectedIterations: 1}
  ];
  for (const {positions, expectedIterations} of scenes) {
    const result = computeDelaunayTessellationOnCPU({positions}, {capacity: 20});
    expect(Array.from(result.iterationCount)).toEqual([expectedIterations]);
    const operation = new CPUDelaunayTessellationOperation({
      positions: 'positions',
      capacity: 20,
      output: {
        triangles: 'triangles',
        status: {
          count: 'count',
          requiredCount: 'requiredCount',
          overflow: 'overflow',
          iterationCount: 'iterationCount'
        }
      }
    });
    const outputs = await operation.execute({
      inputs: {
        positions: {
          format: 'float32x2',
          length: positions.length / 2,
          chunks: [{values: positions, length: positions.length / 2}],
          pack: () => positions
        }
      }
    });
    expect(outputs['iterationCount']).toEqual(Uint32Array.of(expectedIterations));
  }
});

it('admits count-only Delaunay operations and keeps scalar status mandatory', async () => {
  // Three noncollinear sites have exactly one triangle, regardless of output capacity.
  const positions = Float32Array.of(0, 0, 2, 0, 0, 2);
  for (const capacity of [0, 1]) {
    const props: DelaunayTessellationOperationProps = {
      positions: 'positions',
      capacity,
      output: {
        triangles: 'triangles',
        status: {count: 'count', requiredCount: 'requiredCount', overflow: 'overflow'}
      }
    };
    const gpuOperation = new GPUDelaunayTessellationOperation(props);
    expect(gpuOperation.outputs['triangles'].length).toBe(capacity);
    for (const port of ['count', 'requiredCount', 'overflow']) {
      expect(gpuOperation.outputs[port].length).toBe(1);
    }
    const operation = new CPUDelaunayTessellationOperation(props);
    const outputs = await operation.execute({
      inputs: {
        positions: {
          format: 'float32x2',
          length: 3,
          chunks: [{values: positions, length: 3}],
          pack: () => positions
        }
      }
    });
    expect(outputs['triangles']).toHaveLength(3 * capacity);
    expect(outputs['requiredCount']).toEqual(Uint32Array.of(1));
    expect(outputs['count']).toEqual(Uint32Array.of(capacity));
    expect(outputs['overflow']).toEqual(Uint32Array.of(1 - capacity));
  }
});

for (const port of ['count', 'requiredCount', 'overflow'] as const) {
  it(`rejects omission of mandatory Delaunay status.${port} in both named forms`, () => {
    for (const Operation of [GPUDelaunayTessellationOperation, CPUDelaunayTessellationOperation]) {
      const status = {count: 'count', requiredCount: 'requiredCount', overflow: 'overflow'};
      delete (status as Partial<typeof status>)[port];
      expect(
        () =>
          new Operation({
            id: 'required-delaunay',
            positions: 'positions',
            capacity: 0,
            output: {triangles: 'triangles', status}
          })
      ).toThrow(new RegExp(`required-delaunay output.status.${port}`));
    }
  });
}
