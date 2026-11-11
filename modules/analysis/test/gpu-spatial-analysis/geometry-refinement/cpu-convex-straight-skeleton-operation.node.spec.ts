// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeConvexStraightSkeletonOnCPU,
  CPUConvexStraightSkeletonOperation
} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-convex-straight-skeleton-operation';
import {
  GPUConvexStraightSkeletonOperation,
  type ConvexStraightSkeletonOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-convex-straight-skeleton-operation';
import fixtures from '../../oracles/fixtures/geometry-straight-skeleton.json';

function getProps(): ConvexStraightSkeletonOperationProps {
  return {
    id: 'convex-test',
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    polygonCount: 1,
    maximumVertices: 5,
    offsetDistance: 1.8,
    output: {
      segments: 'segments',
      sourceEdges: 'sourceEdges',
      offsetPositions: 'offsetPositions',
      status: 'status'
    }
  };
}

it.each(
  fixtures.cases.filter(fixture => ['rectangle', 'triangle'].includes(fixture.name))
)('matches the independent CGAL 6.0.1 fixture: $name', fixture => {
  const result = computeConvexStraightSkeletonOnCPU(
    {
      positions: Float32Array.from(fixture.positions),
      ringOffsets: Uint32Array.from(fixture.ringOffsets)
    },
    {maximumVertices: fixture.positions.length / 2, offsetDistance: fixture.offsetDistance}
  );
  expect(Array.from(result.status)).toEqual([
    fixture.arcs.length,
    fixture.arcs.length,
    fixture.positions.length / 2,
    0
  ]);
  const arcs = fixture.arcs.map(arc => arc.slice(0, 4));
  for (let row = 0; row < result.status[0]; row++) {
    const actual = Array.from(result.segments.subarray(row * 4, row * 4 + 4));
    const match = arcs.findIndex(arc =>
      arc.every((value, index) => Math.abs(value - actual[index]) < 1e-6)
    );
    expect(match).toBeGreaterThanOrEqual(0);
    arcs.splice(match, 1);
  }
  expect(arcs).toHaveLength(0);
  fixture.offsets
    .flat(2)
    .forEach((value, index) => expect(result.offsetPositions[index]).toBeCloseTo(value, 6));
});

it('matches a closed-form two-event clipped rectangle, including births and edge provenance', () => {
  // Clip x+y<1 from a 6 by 4 rectangle. The diagonal dies at a=1/(2-sqrt(2));
  // the surviving rectangle dies at t=2, with ridge [2,2]--[4,2].
  const collapse = 1 / (2 - Math.SQRT2);
  const result = computeConvexStraightSkeletonOnCPU(
    {positions: Float32Array.of(1, 0, 6, 0, 6, 4, 0, 4, 0, 1), ringOffsets: Uint32Array.of(0, 5)},
    {maximumVertices: 5, offsetDistance: 1.8}
  );
  expect(Array.from(result.status)).toEqual([7, 7, 4, 0]);
  expect(Array.from(result.sourceEdges.subarray(0, 14))).toEqual([
    4, 0, 3, 4, 3, 0, 0, 1, 1, 2, 2, 3, 0xffffffff, 0xffffffff
  ]);
  const expected = [
    1,
    0,
    collapse,
    collapse,
    0,
    1,
    collapse,
    collapse,
    collapse,
    collapse,
    2,
    2,
    6,
    0,
    4,
    2,
    6,
    4,
    4,
    2,
    0,
    4,
    2,
    2,
    2,
    2,
    4,
    2
  ];
  expected.forEach((value, index) => expect(result.segments[index]).toBe(Math.fround(value)));
  expect(Array.from(result.offsetPositions)).toEqual(
    Array.from(Float32Array.of(1.8, 1.8, 4.2, 1.8, 4.2, 2.2, 1.8, 2.2, 0, 0))
  );
});

it('keeps independent polygon blocks, explicit closure, clockwise winding, capacity and extinction', () => {
  const input = {
    positions: Float32Array.of(0, 0, 4, 0, 4, 2, 0, 2, 0, 0, 10, 0, 10, 8, 16, 0),
    ringOffsets: Uint32Array.of(0, 5, 8)
  };
  const result = computeConvexStraightSkeletonOnCPU(input, {
    maximumVertices: 4,
    segmentCapacity: 2,
    offsetDistance: 2
  });
  expect(Array.from(result.status)).toEqual([2, 5, 0, 2, 2, 3, 0, 2]);
  expect(Array.from(result.sourceEdges.subarray(4, 8))).toEqual([7, 5, 5, 6]);
  expect(result.offsetPositions.every(value => value === 0)).toBe(true);
});

it('uses the documented unsupported flags and clamps ring offsets', () => {
  const result = computeConvexStraightSkeletonOnCPU(
    {
      positions: Float32Array.of(0, 0, 2, 0, 1, 1, 2, 2, 0, 2),
      ringOffsets: Uint32Array.of(0, 5, 999)
    },
    {maximumVertices: 5, offsetDistance: 0}
  );
  expect(Array.from(result.status)).toEqual([0, 0, 0, 1, 0, 0, 0, 1]);
  expect(() =>
    computeConvexStraightSkeletonOnCPU(
      {positions: Float32Array.of(0), ringOffsets: Uint32Array.of(0)},
      {offsetDistance: 0}
    )
  ).toThrow(/positions/);
  expect(() =>
    computeConvexStraightSkeletonOnCPU(
      {positions: new Float32Array(0), ringOffsets: new Uint32Array(0)},
      {offsetDistance: 0}
    )
  ).toThrow(/ringOffsets/);
});

it('reports a decreasing ring as unsupported and continues with the following ring', () => {
  const positions = Float32Array.of(0, 0, 4, 0, 4, 2, 0, 2);
  const unsupported = computeConvexStraightSkeletonOnCPU(
    {positions, ringOffsets: Uint32Array.of(3, 1)},
    {maximumVertices: 4, offsetDistance: 0}
  );
  // uint32(1-3) is 0xfffffffe, exceeding the four-vertex block: no arc or offset is written.
  expect(Array.from(unsupported.status)).toEqual([0, 0, 0, 1]);
  expect(unsupported.segments.every(value => value === 0)).toBe(true);
  expect(unsupported.sourceEdges.every(value => value === 0)).toBe(true);
  expect(unsupported.offsetPositions.every(value => value === 0)).toBe(true);
  const subsequent = computeConvexStraightSkeletonOnCPU(
    {positions, ringOffsets: Uint32Array.of(3, 1, 4)},
    {maximumVertices: 4, offsetDistance: 0}
  );
  // Rows 1..3 form an ordinary right triangle: three arcs and three original offset vertices.
  expect(Array.from(subsequent.status)).toEqual([0, 0, 0, 1, 3, 3, 3, 0]);
  expect(Array.from(subsequent.offsetPositions.subarray(8, 14))).toEqual([4, 0, 4, 2, 0, 2]);
});

it('validates CPU properties at construction and retains only the vertex block layout limit', () => {
  for (const [name, value] of [
    ['offsetDistance', -1],
    ['eventTolerance', 0],
    ['maximumVertices', 2],
    ['maximumIterations', -1],
    ['segmentCapacity', 0],
    ['polygonCount', 0]
  ] as const) {
    expect(() => new CPUConvexStraightSkeletonOperation({...getProps(), [name]: value})).toThrow(
      new RegExp(`^convex-test .*${name}`)
    );
  }
  expect(
    () =>
      new CPUConvexStraightSkeletonOperation({
        ...getProps(),
        maximumVertices: 65,
        maximumIterations: 1000
      })
  ).not.toThrow();
  const ordinary = computeConvexStraightSkeletonOnCPU(
    {positions: Float32Array.of(0, 0, 4, 0, 4, 2, 0, 2), ringOffsets: Uint32Array.of(0, 4)},
    {offsetDistance: 0}
  );
  expect(ordinary.status[3]).toBe(0);
  expect(
    () =>
      new CPUConvexStraightSkeletonOperation({
        ...getProps(),
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
      })
  ).toThrow(/^convex-test/);
});

it.each([
  GPUConvexStraightSkeletonOperation,
  CPUConvexStraightSkeletonOperation
])('%s validates graph output shapes and disjoint names at construction', Operation => {
  expect(() => new Operation(getProps())).not.toThrow();
  expect(() => new Operation({...getProps(), polygonCount: 0})).toThrow(/^convex-test/);
  expect(
    () => new Operation({...getProps(), output: {...getProps().output, segments: 'positions'}})
  ).toThrow(/^convex-test/);
  expect(
    () => new Operation({...getProps(), output: {...getProps().output, sourceEdges: 'segments'}})
  ).toThrow(/^convex-test/);
});

it('executes every CPU graph port and ignores the GPU iteration budget', async () => {
  const positions = Float32Array.of(0, 0, 4, 0, 4, 2, 0, 2);
  const ringOffsets = Uint32Array.of(0, 4);
  const inputs = {
    positions: {
      format: 'float32x2' as const,
      length: 4,
      chunks: [{values: positions, length: 4}],
      pack: () => positions
    },
    ringOffsets: {
      format: 'uint32' as const,
      length: 2,
      chunks: [{values: ringOffsets, length: 2}],
      pack: () => ringOffsets
    }
  };
  const output = await new CPUConvexStraightSkeletonOperation({
    ...getProps(),
    maximumIterations: 0,
    offsetDistance: 0.5
  }).execute({inputs});
  expect(Array.from(output['status'] as Uint32Array)).toEqual([5, 5, 4, 0]);
  expect((output['segments'] as Float32Array).length).toBe(40);
  expect((output['sourceEdges'] as Uint32Array).length).toBe(20);
  expect((output['offsetPositions'] as Float32Array).length).toBe(10);
  expect(() =>
    new CPUConvexStraightSkeletonOperation({...getProps(), polygonCount: 2}).execute({inputs})
  ).toThrow(/ringOffsets/);
});
