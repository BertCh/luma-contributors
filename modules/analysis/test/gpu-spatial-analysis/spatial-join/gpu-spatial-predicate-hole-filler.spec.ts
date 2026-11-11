// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Polygon `contains` / `within` against a polygon with a hole (math review cells-joins-01).
// OGC `contains(outer, inner)` needs the interiors to intersect and no point of the inner polygon
// in the exterior of the outer one. Every expected list below follows from that definition and the
// drawn geometry; none is taken from a kernel or from the float64 oracle.

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRandom} from './spatial-join-oracle';
import {polygonWithRings, type OracleFeature, type OraclePoint} from './spatial-predicate-oracle';
import {comparePairs, runScaleJoin} from './spatial-relate-scale-harness';

const ENGINES = ['fast', 'auto', 'relate'] as const;

/** Asserts `contains(outers, inners)` and the mirrored `within` for every engine. */
async function expectContainedInners(
  label: string,
  outers: OracleFeature[],
  inners: OracleFeature[],
  containedInners: number[]
): Promise<void> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const engine of ENGINES) {
    const contains = await runScaleJoin(device, outers, inners, 'contains', {engine});
    expect(contains.pairs.sort(comparePairs), `${label} contains ${engine}`).toEqual(
      containedInners.map(inner => `0,${inner}`).sort(comparePairs)
    );
    const within = await runScaleJoin(device, inners, outers, 'within', {engine});
    expect(within.pairs.sort(comparePairs), `${label} within ${engine}`).toEqual(
      containedInners.map(inner => `${inner},0`).sort(comparePairs)
    );
    expect(contains.overflow + within.overflow, `${label} overflow ${engine}`).toBe(0);
  }
}

// A triangular hole whose vertices are not on a lattice: no edge midpoint is a float32 value on the
// edge, which is what the old midpoint probe needed.
const OBLIQUE_HOLE: OraclePoint[] = [
  [10.1, 10.2],
  [30.3, 11.7],
  [17.9, 40.4]
];

it('GPUSpatialPredicateJoin contains: a polygon that fills an oblique hole is not contained', async () => {
  const outer = polygonWithRings(
    [
      [2, 2],
      [100, 2],
      [100, 100],
      [2, 100]
    ],
    OBLIQUE_HOLE
  );
  const inners = [
    // 0 to 2: the hole itself, in three vertex orders. It only touches the outer polygon.
    polygonWithRings(OBLIQUE_HOLE),
    polygonWithRings([...OBLIQUE_HOLE].reverse()),
    polygonWithRings([OBLIQUE_HOLE[1], OBLIQUE_HOLE[2], OBLIQUE_HOLE[0]]),
    // 3: the hole plus a triangle of the outer interior across its edge (30.3, 11.7)-(17.9, 40.4).
    // Two edges run along the hole boundary and no hole vertex is strictly inside.
    polygonWithRings([OBLIQUE_HOLE[0], OBLIQUE_HOLE[1], [40.9, 45.3], OBLIQUE_HOLE[2]]),
    // 4: away from the hole, strictly inside the outer polygon.
    polygonWithRings([
      [50.1, 50.2],
      [70.3, 51.7],
      [57.9, 80.4]
    ]),
    // 5: shares the hole edge (10.1, 10.2)-(30.3, 11.7) from the outer side (the third vertex is
    // below that edge, the hole is above it).
    polygonWithRings([OBLIQUE_HOLE[0], OBLIQUE_HOLE[1], [20.7, 4.3]]),
    // 6: strictly inside the hole.
    polygonWithRings([
      [15.1, 15.2],
      [25.3, 14.7],
      [18.9, 30.4]
    ])
  ];
  await expectContainedInners('oblique hole', [outer], inners, [4, 5]);
});

it('GPUSpatialPredicateJoin contains: random oblique hole fillers are never contained', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 64 outer squares in separate 100-unit cells, each with a random triangular hole. The inner
  // features are the 64 holes, then 64 small triangles in the corner of each cell that no hole
  // reaches (a hole stays below x + y = 160 in cell coordinates, the corner triangle above 164).
  const random = createRandom(0x51ce);
  const outers: OracleFeature[] = [];
  const fillers: OracleFeature[] = [];
  const controls: OracleFeature[] = [];
  const cellCount = 64;
  for (let cell = 0; cell < cellCount; cell++) {
    const x = 100 * (cell % 8);
    const y = 100 * Math.floor(cell / 8);
    const hole: OraclePoint[] = [
      [x + 10 + 30 * random(), y + 10 + 20 * random()],
      [x + 60 + 30 * random(), y + 10 + 30 * random()],
      [x + 30 + 40 * random(), y + 60 + 30 * random()]
    ];
    outers.push(
      polygonWithRings(
        [
          [x + 1, y + 1],
          [x + 99, y + 1],
          [x + 99, y + 99],
          [x + 1, y + 99]
        ],
        hole
      )
    );
    fillers.push(polygonWithRings(hole));
    controls.push(
      polygonWithRings([
        [x + 82, y + 82],
        [x + 95, y + 83],
        [x + 88, y + 95]
      ])
    );
  }
  const expected = controls.map((_, cell) => `${cell},${cellCount + cell}`).sort(comparePairs);
  for (const engine of ENGINES) {
    const result = await runScaleJoin(device, outers, [...fillers, ...controls], 'contains', {
      engine
    });
    expect(result.overflow, `random holes overflow ${engine}`).toBe(0);
    expect(result.pairs.sort(comparePairs), `random holes ${engine}`).toEqual(expected);
  }
});

it('GPUSpatialPredicateJoin contains: a hole inscribed in the inner polygon is not contained', async () => {
  // The three hole vertices lie on three edges of the inner square, away from the edge midpoints
  // (25, 10), (40, 25) and (25, 40). The hole interior is inside the square and outside the outer
  // polygon, while every square edge stays in the closed outer polygon and no hole vertex is
  // strictly inside the square.
  const hole: OraclePoint[] = [
    [20, 10],
    [40, 30],
    [15, 40]
  ];
  const outer = polygonWithRings(
    [
      [0, 0],
      [60, 0],
      [60, 60],
      [0, 60]
    ],
    hole
  );
  const inners = [
    // 0: the square around the inscribed hole.
    polygonWithRings([
      [10, 10],
      [40, 10],
      [40, 40],
      [10, 40]
    ]),
    // 1: clear of the hole.
    polygonWithRings([
      [45, 45],
      [55, 45],
      [55, 55],
      [45, 55]
    ]),
    // 2: the hole itself.
    polygonWithRings(hole),
    // 3: the triangle cut off the square by the hole edge (20, 10)-(40, 30): it shares that edge
    // from the outer side and two half edges with nothing.
    polygonWithRings([
      [20, 10],
      [40, 10],
      [40, 30]
    ])
  ];
  await expectContainedInners('inscribed hole', [outer], inners, [1, 3]);
});

it('GPUSpatialPredicateJoin contains: a hole filler at projected-metre coordinates is not contained', async () => {
  // A 10 m hole at a northing of 5,000,000, where one float32 step is 0.5 m: an offset of a
  // thousandth of the edge (0.01 m) cannot move a point off the edge.
  const east = 500000;
  const north = 5000000;
  const square = (x: number, y: number, size: number): OraclePoint[] => [
    [x, y],
    [x + size, y],
    [x + size, y + size],
    [x, y + size]
  ];
  const outer = polygonWithRings(square(east - 100, north - 100, 200), square(east, north, 10));
  const inners = [
    // 0: the hole itself.
    polygonWithRings(square(east, north, 10)),
    // 1: clear of the hole.
    polygonWithRings(square(east + 50, north + 50, 10)),
    // 2: the hole plus the 10 m square east of it.
    polygonWithRings([
      [east, north],
      [east + 20, north],
      [east + 20, north + 10],
      [east, north + 10]
    ]),
    // 3: the 10 m square east of the hole, sharing its east edge from the outer side.
    polygonWithRings(square(east + 10, north, 10))
  ];
  await expectContainedInners('projected metres', [outer], inners, [1, 3]);
});
