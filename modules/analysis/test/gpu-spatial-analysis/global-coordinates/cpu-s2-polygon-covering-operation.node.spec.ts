// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeS2PolygonCoveringOnCPU,
  CPUS2PolygonCoveringOperation
} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-polygon-covering-operation';
import {
  GPUS2PolygonCoveringOperation,
  type S2PolygonCoveringOperationProps
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import fixtures from './s2-polygon-covering-operation-fixture.json';
import {OFF_AXIS_CLOCKWISE_TRIANGLE} from './off-axis-ring-fixture';

const PROPS: S2PolygonCoveringOperationProps = {
  id: 'polygon-test',
  positions: 'positions',
  ringOffsets: 'ringOffsets',
  positionCount: 4,
  ringOffsetCount: 2,
  maximumLevel: 3,
  capacity: 128,
  output: {cells: 'cells', counts: 'counts', status: 'status', visitedCount: 'visitedCount'}
};

it('matches independent Shapely gnomonic intersections: holes, complement, adaptive and uniform covers', () => {
  for (const fixture of fixtures)
    for (const capacity of [3, 512]) {
      const result = computeS2PolygonCoveringOnCPU(
        {
          positions: Float32Array.from(fixture.positions),
          ringOffsets: Uint32Array.from(fixture.ringOffsets)
        },
        {...fixture, mode: fixture.mode as 'exterior' | 'interior', capacity}
      );
      const written = Math.min(capacity, fixture.cells.length);
      expect(Array.from(result.counts), fixture.name).toEqual([written, fixture.cells.length]);
      expect(result.status[0], fixture.name).toBe(fixture.cells.length > capacity ? 2 : 0);
      expect(result.visitedCount[0], fixture.name).toBe(fixture.visitedCount);
      for (let row = 0; row < written; row++) {
        const cell = BigInt(result.cells[row * 2]) | (BigInt(result.cells[row * 2 + 1]) << 32n);
        expect(cell, fixture.name).toBe(BigInt(`0x${fixture.cells[row]}`));
      }
      expect(result.cells.subarray(written * 2).every(value => value === 0)).toBe(true);
    }
});

it('completes with omitted budgets and bounds an explicit six-visit traversal', () => {
  const fixture = fixtures.find(scene => scene.minimumLevel === 3 && !scene.complement)!;
  const result = computeS2PolygonCoveringOnCPU(
    {
      positions: Float32Array.from(fixture.positions),
      ringOffsets: Uint32Array.from(fixture.ringOffsets)
    },
    {...fixture, mode: 'exterior', capacity: 1}
  );
  expect(result.status[0]).toBe(2);
  expect(result.counts[1]).toBe(fixture.cells.length);
  expect(result.visitedCount[0]).toBeGreaterThan(6);
  const bounded = computeS2PolygonCoveringOnCPU(
    {
      positions: Float32Array.from(fixture.positions),
      ringOffsets: Uint32Array.from(fixture.ringOffsets)
    },
    {...fixture, mode: 'exterior', capacity: 1, maximumVisitedCells: 6}
  );
  // First six pops: face 0, its first child and grandchild, then leaves 0x004..., 0x00c...,
  // 0x014.... The independent Shapely fixture excludes the first leaf and includes the next two.
  expect(Array.from(bounded.counts)).toEqual([1, 2]);
  expect(bounded.status[0]).toBe(6);
  expect(bounded.visitedCount[0]).toBe(6);
});

it('returns invalid status for malformed ranges, non-unit vertices, collapsed and crossing rings', () => {
  const options = {maximumLevel: 2, capacity: 4};
  const triangle = Float32Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
  for (const ringOffsets of [Uint32Array.of(0, 2), Uint32Array.of(3, 1), Uint32Array.of(0, 4)]) {
    const result = computeS2PolygonCoveringOnCPU({positions: triangle, ringOffsets}, options);
    expect(Array.from(result.status)).toEqual([1]);
    expect(Array.from(result.counts)).toEqual([0, 0]);
    expect(Array.from(result.visitedCount)).toEqual([0]);
    expect(result.cells.every(value => value === 0)).toBe(true);
  }
  for (const positions of [
    new Float32Array(9),
    Float32Array.of(1, 0, 0, 1, 0, 0, 0, 1, 0),
    Float32Array.of(NaN, 0, 0, 0, 1, 0, 0, 0, 1)
  ])
    expect(
      computeS2PolygonCoveringOnCPU({positions, ringOffsets: Uint32Array.of(0, 3)}, options)
        .status[0]
    ).toBe(1);
  const fixture = fixtures[0];
  const crossing = Float32Array.from(
    [0, 2, 1, 3].flatMap(row => fixture.positions.slice(row * 3, row * 3 + 3))
  );
  expect(
    computeS2PolygonCoveringOnCPU({positions: crossing, ringOffsets: Uint32Array.of(0, 4)}, options)
      .status[0]
  ).toBe(1);
});

it('validates arrays and explicit budgets while allowing CPU level thirty', () => {
  expect(() =>
    computeS2PolygonCoveringOnCPU(
      {positions: new Float32Array(2), ringOffsets: Uint32Array.of(0, 0)},
      PROPS
    )
  ).toThrow(/positions/);
  expect(() =>
    computeS2PolygonCoveringOnCPU(
      {positions: new Float32Array(0), ringOffsets: new Uint32Array(1)},
      PROPS
    )
  ).toThrow(/ringOffsets/);
  const shapes = {
    positions: {format: 'float32x3' as const, length: 4},
    ringOffsets: {format: 'uint32' as const, length: 2}
  };
  expect(() =>
    constructNetworkOperationContributor(new GPUS2PolygonCoveringOperation(PROPS), shapes)
  ).not.toThrow();
  for (const Operation of [GPUS2PolygonCoveringOperation, CPUS2PolygonCoveringOperation]) {
    expect(() => new Operation({...PROPS, capacity: 0})).toThrow(/polygon-test.*length/);
    expect(() => new Operation({...PROPS, output: {...PROPS.output, cells: 'positions'}})).toThrow(
      /polygon-test/
    );
  }
  for (const maximumLevel of [-1, 0.5, NaN]) {
    expect(() => new CPUS2PolygonCoveringOperation({...PROPS, maximumLevel})).toThrow(
      /polygon-test.*maximumLevel/
    );
    expect(() =>
      constructNetworkOperationContributor(
        new GPUS2PolygonCoveringOperation({...PROPS, maximumLevel}),
        shapes
      )
    ).toThrow(/polygon-test.*maximumLevel/);
  }
  expect(
    () => new CPUS2PolygonCoveringOperation({...PROPS, maximumLevel: 30, maximumVisitedCells: 1})
  ).toThrow(/maximumVisitedCells/);
  expect(() => new CPUS2PolygonCoveringOperation({...PROPS, maximumLevel: 30})).not.toThrow();
  expect(() => new CPUS2PolygonCoveringOperation({...PROPS, maximumLevel: 31})).toThrow(
    /maximumLevel/
  );
  expect(() => new CPUS2PolygonCoveringOperation({...PROPS, minimumLevel: 4})).toThrow(
    /minimumLevel/
  );
  expect(() =>
    constructNetworkOperationContributor(
      new GPUS2PolygonCoveringOperation({...PROPS, minimumLevel: 4}),
      shapes
    )
  ).toThrow(/minimumLevel/);
  expect(() => new CPUS2PolygonCoveringOperation({...PROPS, hemisphereCenter: [0, 0, 0]})).toThrow(
    /hemisphereCenter/
  );
  expect(() =>
    constructNetworkOperationContributor(
      new GPUS2PolygonCoveringOperation({...PROPS, hemisphereCenter: [0, 0, 0]}),
      shapes
    )
  ).toThrow(/hemisphereCenter/);
});

// Independent GEOS enumeration of quadratic S2 rectangles in the x=1 chart, encoded with
// xy-to-Hilbert rotations (scratch/oracle.py). Reversal preserves each of these planar regions.
it('canonicalizes shell and hole orientation about the hemisphere pole (W2-05)', () => {
  const scenes = [
    {
      ring: [
        [-0.7, -0.4],
        [0.6, -0.4],
        [0.15, 0.65]
      ],
      level: 3,
      cells: [
        '0540000000000000',
        '0fc0000000000000',
        '1040000000000000',
        '11c0000000000000',
        '1a40000000000000',
        '1ac0000000000000'
      ]
    },
    {
      ring: [
        [-0.27, -0.36],
        [0.29, -0.36],
        [0.29, 0.36],
        [-0.27, 0.36]
      ],
      level: 4,
      cells: [
        '04d0000000000000',
        '04f0000000000000',
        '0510000000000000',
        '0530000000000000',
        '0550000000000000',
        '0570000000000000',
        '0e30000000000000',
        '0e50000000000000',
        '0f90000000000000',
        '0fb0000000000000',
        '0fd0000000000000',
        '0ff0000000000000',
        '1010000000000000',
        '1030000000000000',
        '1050000000000000',
        '1070000000000000',
        '11b0000000000000',
        '11d0000000000000',
        '1a90000000000000',
        '1ab0000000000000',
        '1ad0000000000000',
        '1af0000000000000',
        '1b10000000000000',
        '1b30000000000000'
      ]
    }
  ];
  for (const scene of scenes)
    for (const reversed of [false, true]) {
      const ring = reversed ? [...scene.ring].reverse() : scene.ring;
      const positions = Float32Array.from(
        ring.flatMap(([y, z]) => {
          const length = Math.hypot(1, y, z);
          return [1 / length, y / length, z / length];
        })
      );
      const result = computeS2PolygonCoveringOnCPU(
        {positions, ringOffsets: Uint32Array.of(0, ring.length)},
        {minimumLevel: scene.level, maximumLevel: scene.level, mode: 'interior', capacity: 128}
      );
      expect(Array.from(result.status)).toEqual([0]);
      expect(Array.from(result.counts)).toEqual([scene.cells.length, scene.cells.length]);
      expect(
        Array.from({length: result.counts[0]}, (_, row) =>
          (BigInt(result.cells[2 * row]) | (BigInt(result.cells[2 * row + 1]) << 32n))
            .toString(16)
            .padStart(16, '0')
        )
      ).toEqual(scene.cells);
    }
  for (const fixture of fixtures.filter(scene => scene.minimumLevel === 0)) {
    const rings = fixture.ringOffsets.slice(0, -1).map((begin, ring) => {
      const end = fixture.ringOffsets[ring + 1];
      const vertices = Array.from({length: end - begin}, (_, vertex) =>
        fixture.positions.slice((begin + vertex) * 3, (begin + vertex + 1) * 3)
      );
      return vertices.reverse();
    });
    const result = computeS2PolygonCoveringOnCPU(
      {
        positions: Float32Array.from(rings.flat(2)),
        ringOffsets: Uint32Array.from(fixture.ringOffsets)
      },
      {...fixture, mode: fixture.mode as 'exterior' | 'interior', capacity: 512}
    );
    expect(Array.from(result.status), fixture.name).toEqual([0]);
    expect(Array.from(result.counts), fixture.name).toEqual([
      fixture.cells.length,
      fixture.cells.length
    ]);
    expect(
      Array.from({length: result.counts[0]}, (_, row) =>
        (BigInt(result.cells[2 * row]) | (BigInt(result.cells[2 * row + 1]) << 32n)).toString(16)
      ),
      fixture.name
    ).toEqual(fixture.cells);
  }
});

it('preserves tiny off-axis shell and antipode roles in either orientation (W2-05)', () => {
  // Packed triangle from off-axis-ring-fixture.ts: it contains the positive cube corner,
  // crosses faces 0/1/2 and lies wholly in the positive octant. The complement contains
  // exactly roots 3/4/5; no root fits inside the original triangle.
  for (const reversed of [false, true])
    for (const complement of [false, true])
      for (const mode of ['exterior', 'interior'] as const) {
        const vertices = reversed
          ? [...OFF_AXIS_CLOCKWISE_TRIANGLE].reverse()
          : OFF_AXIS_CLOCKWISE_TRIANGLE;
        const result = computeS2PolygonCoveringOnCPU(
          {positions: Float32Array.from(vertices.flat()), ringOffsets: Uint32Array.of(0, 3)},
          {maximumLevel: 0, capacity: 6, mode, complement}
        );
        const expected =
          mode === 'interior'
            ? complement
              ? [0x70000000, 0x90000000, 0xb0000000]
              : []
            : complement
              ? [0x10000000, 0x30000000, 0x50000000, 0x70000000, 0x90000000, 0xb0000000]
              : [0x10000000, 0x30000000, 0x50000000];
        expect(Array.from(result.status)).toEqual([0]);
        expect(Array.from(result.counts)).toEqual([expected.length, expected.length]);
        expect(Array.from(result.cells)).toEqual([
          ...expected.flatMap(high => [0, high]),
          ...new Array((6 - expected.length) * 2).fill(0)
        ]);
      }
});

it('pins large-ring completion and route-specific exhaustion without changing the DFS prefix (CORE-118)', {
  timeout: 30000
}, () => {
  // The small circle lies strictly inside face 0 and contains its center. Exactly the four
  // level-1 children intersect it. DFS pops face 0, those children, then faces 1..5 (10 visits).
  const count = 4096;
  const positions = Float32Array.from(
    Array.from({length: count}, (_, row) => {
      const angle = (2 * Math.PI * row) / count;
      const point = [1, 0.2 * Math.cos(angle), 0.2 * Math.sin(angle)];
      return point.map(value => value / Math.hypot(...point));
    }).flat()
  );
  const ordered = [0, 0x04000000, 0, 0x0c000000, 0, 0x14000000, 0, 0x1c000000];
  for (const refinement of ['serial', 'level-synchronous'] as const)
    for (const maximumVisitedCells of [64, 6]) {
      const result = computeS2PolygonCoveringOnCPU(
        {positions, ringOffsets: Uint32Array.of(0, count)},
        {minimumLevel: 1, maximumLevel: 1, capacity: 8, refinement, maximumVisitedCells}
      );
      const exhausted = maximumVisitedCells === 6;
      const suppressed = exhausted && refinement === 'level-synchronous';
      expect(Array.from(result.status)).toEqual([exhausted ? 4 : 0]);
      expect(Array.from(result.counts)).toEqual(suppressed ? [0, 0] : [4, 4]);
      expect(Array.from(result.visitedCount)).toEqual([exhausted ? 6 : 10]);
      expect(Array.from(result.cells)).toEqual(
        suppressed ? new Array(16).fill(0) : [...ordered, ...new Array(8).fill(0)]
      );
    }
});

it('rejects unknown string options in the pure polygon form (W2-04)', () => {
  const input = {
    positions: Float32Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1),
    ringOffsets: Uint32Array.of(0, 3)
  };
  expect(() =>
    computeS2PolygonCoveringOnCPU(input, {maximumLevel: 0, capacity: 6, mode: 'bad' as never})
  ).toThrow(/s2-polygon-covering.*mode.*exterior.*interior.*bad/);
  expect(() =>
    computeS2PolygonCoveringOnCPU(input, {maximumLevel: 0, capacity: 6, refinement: 'bad' as never})
  ).toThrow(/s2-polygon-covering.*refinement.*serial.*level-synchronous.*bad/);
});

it('orients a shell and hole independently in all four combinations (review S1)', () => {
  // GEOS enumeration in the x=1 chart: square shell [-.4,.4]^2 minus [-.1,.1]^2.
  // Quadratic level-4 S2 rectangles strictly inside the region encode to these 20 IDs.
  const expected = [
    '0450000000000000',
    '04d0000000000000',
    '04f0000000000000',
    '0590000000000000',
    '05b0000000000000',
    '0e30000000000000',
    '0e50000000000000',
    '0ef0000000000000',
    '0f10000000000000',
    '0f70000000000000',
    '1090000000000000',
    '10f0000000000000',
    '1110000000000000',
    '11b0000000000000',
    '11d0000000000000',
    '1a50000000000000',
    '1a70000000000000',
    '1b10000000000000',
    '1b30000000000000',
    '1bb0000000000000'
  ];
  const square = (size: number) => [
    [1, -size, -size],
    [1, size, -size],
    [1, size, size],
    [1, -size, size]
  ];
  for (const reverseShell of [false, true])
    for (const reverseHole of [false, true]) {
      const shell = square(0.4);
      const hole = square(0.1);
      if (reverseShell) shell.reverse();
      if (reverseHole) hole.reverse();
      const positions = Float32Array.from(
        [...shell, ...hole].flatMap(point => point.map(value => value / Math.hypot(...point)))
      );
      for (const refinement of ['serial', 'level-synchronous'] as const) {
        const result = computeS2PolygonCoveringOnCPU(
          {positions, ringOffsets: Uint32Array.of(0, 4, 8)},
          {
            minimumLevel: 4,
            maximumLevel: 4,
            mode: 'interior',
            capacity: 64,
            refinement,
            maximumVisitedCells: 1024
          }
        );
        expect(Array.from(result.status)).toEqual([0]);
        expect(Array.from(result.counts)).toEqual([20, 20]);
        expect(
          Array.from({length: 20}, (_, row) =>
            (BigInt(result.cells[row * 2]) | (BigInt(result.cells[row * 2 + 1]) << 32n))
              .toString(16)
              .padStart(16, '0')
          )
        ).toEqual(expected);
      }
    }
});

it('rejects a collapsed three-vertex ring before visiting cells on both CPU routes (review N2)', () => {
  const positions = Float32Array.from(
    [-0.4, 0, 0.4].flatMap(value => {
      const length = Math.hypot(1, value);
      return [1 / length, value / length, 0];
    })
  );
  for (const refinement of ['serial', 'level-synchronous'] as const) {
    const result = computeS2PolygonCoveringOnCPU(
      {positions, ringOffsets: Uint32Array.of(0, 3)},
      {maximumLevel: 0, capacity: 6, maximumVisitedCells: 6, refinement}
    );
    expect(Array.from(result.status)).toEqual([1]);
    expect(Array.from(result.counts)).toEqual([0, 0]);
    expect(Array.from(result.visitedCount)).toEqual([0]);
    expect(Array.from(result.cells)).toEqual(new Array(12).fill(0));
  }
});
