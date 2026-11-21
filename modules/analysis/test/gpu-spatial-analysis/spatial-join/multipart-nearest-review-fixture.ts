// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import type {CPUSpatialJoinGeometry} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';
import type {
  GPUMultipartGeometryPort,
  GPUPackedGeometryPort
} from '../../../src/analysis-core/contracts/index';

type Format = 'uint32' | 'float32' | 'float32x2';
export type ReviewNearestCase = {
  name: string;
  queries: CPUSpatialJoinGeometry;
  features: CPUSpatialJoinGeometry;
  radius?: number;
  nearest?: boolean;
  expectedRows: number[];
  expectedDistances: number[];
  horizontalSeparation?: number;
  coincidentPoint?: number[];
};

export const NEAREST_REVIEW_CASES: ReviewNearestCase[] = [];
for (const reverse of [false, true]) {
  for (const nearest of [false, true]) {
    // Shapely distance: 2^-22 / sqrt(2) = 1.6858739404357614e-7. Row 1 contains p
    // exactly, so radius zero must exclude row 0 in either line direction.
    const point = [0.25, 0.25 + 2 ** -22];
    NEAREST_REVIEW_CASES.push({
      name: `R1 reversed=${reverse} nearest=${nearest}`,
      queries: {
        kind: 'points',
        positions: Float32Array.of(40, 20, ...point),
        featureOffsets: Uint32Array.of(0, 2)
      },
      features: {
        kind: 'lines',
        positions: Float32Array.of(
          ...(reverse ? [0, 0, 10, 10] : [10, 10, 0, 0]),
          ...point,
          1,
          point[1]
        ),
        lineOffsets: Uint32Array.of(0, 2, 4),
        featureOffsets: Uint32Array.of(0, 1, 2)
      },
      nearest,
      radius: nearest ? 0 : undefined,
      expectedRows: nearest ? [1] : [1, 0],
      expectedDistances: nearest ? [0] : [0, 1.6858739404357614e-7]
    });
  }
}

for (const separation of [1, 0]) {
  NEAREST_REVIEW_CASES.push({
    name: `R2 reversed parallel edges separation=${separation}`,
    queries: {
      kind: 'lines',
      positions: Float32Array.of(40, 30, 45, 30, 0, 0, 10, 0),
      lineOffsets: Uint32Array.of(0, 2, 4),
      featureOffsets: Uint32Array.of(0, 2)
    },
    features: {
      kind: 'lines',
      positions: Float32Array.of(-40, -30, -45, -30, 10, separation, 0, separation),
      lineOffsets: Uint32Array.of(0, 2, 4),
      featureOffsets: Uint32Array.of(0, 2)
    },
    expectedRows: [0],
    expectedDistances: [separation],
    horizontalSeparation: separation
  });
}

// The point lies exactly on the diagonal in represented float32 coordinates. Its published
// query point must remain the original point, even if projection from the distant end rounds.
const DIAGONAL_POINT = Math.fround(0.3);
NEAREST_REVIEW_CASES.push({
  name: 'R2 exact point on reversed line keeps its original coordinates',
  queries: {
    kind: 'points',
    positions: Float32Array.of(40, 20, DIAGONAL_POINT, DIAGONAL_POINT),
    featureOffsets: Uint32Array.of(0, 2)
  },
  features: {
    kind: 'lines',
    positions: Float32Array.of(-40, -30, -45, -30, 10, 10, 0, 0),
    lineOffsets: Uint32Array.of(0, 2, 4),
    featureOffsets: Uint32Array.of(0, 2)
  },
  expectedRows: [0],
  expectedDistances: [0],
  coincidentPoint: [DIAGONAL_POINT, DIAGONAL_POINT]
});
NEAREST_REVIEW_CASES.push({
  name: 'R2 line endpoint contact keeps its original coordinates',
  queries: {
    kind: 'lines',
    positions: Float32Array.of(40, 20, 45, 20, DIAGONAL_POINT, DIAGONAL_POINT, 1, 0),
    lineOffsets: Uint32Array.of(0, 2, 4),
    featureOffsets: Uint32Array.of(0, 2)
  },
  features: {
    kind: 'lines',
    positions: Float32Array.of(-40, -30, -45, -30, 10, 10, 0, 0),
    lineOffsets: Uint32Array.of(0, 2, 4),
    featureOffsets: Uint32Array.of(0, 2)
  },
  expectedRows: [0],
  expectedDistances: [0],
  coincidentPoint: [DIAGONAL_POINT, DIAGONAL_POINT]
});

for (const mixedTarget of [false, true]) {
  NEAREST_REVIEW_CASES.push({
    name: `R4 nonfinite point parts mixedTarget=${mixedTarget}`,
    queries: {
      kind: 'points',
      positions: Float32Array.of(NaN, 0, 3, 0, NaN, 0, Infinity, 0),
      featureOffsets: Uint32Array.of(0, 2, 4)
    },
    features: {
      kind: 'points',
      positions: Float32Array.from(mixedTarget ? [NaN, 0, 5, 0] : [5, 0]),
      featureOffsets: Uint32Array.of(0, mixedTarget ? 2 : 1)
    },
    expectedRows: [0, 0xffffffff],
    expectedDistances: [2, -1]
  });
}
NEAREST_REVIEW_CASES.push({
  name: 'R4 nonfinite line vertices retain only usable edges',
  queries: {
    kind: 'lines',
    positions: Float32Array.of(NaN, 0, 3, 0, 5, 0, Infinity, 0, NaN, 0),
    lineOffsets: Uint32Array.of(0, 3, 5),
    featureOffsets: Uint32Array.of(0, 1, 2)
  },
  features: {
    kind: 'points',
    positions: Float32Array.of(7, 0),
    featureOffsets: Uint32Array.of(0, 1)
  },
  expectedRows: [0, 0xffffffff],
  expectedDistances: [2, -1]
});
NEAREST_REVIEW_CASES.push({
  name: 'R4 invalid middle vertex must not create a replacement edge',
  queries: {
    kind: 'lines',
    positions: Float32Array.of(0, 0, NaN, 0, 4, 0),
    lineOffsets: Uint32Array.of(0, 3),
    featureOffsets: Uint32Array.of(0, 1)
  },
  features: {
    kind: 'points',
    positions: Float32Array.of(2, 0),
    featureOffsets: Uint32Array.of(0, 1)
  },
  // Removing the invalid vertex and connecting the survivors would incorrectly report zero.
  expectedRows: [0xffffffff],
  expectedDistances: [-1]
});

export function createReviewNearestFixture(
  graph: GPUCommandGraph,
  allocate: <ValueFormat extends Format>(
    name: string,
    format: ValueFormat,
    length: number,
    values: Float32Array | Uint32Array
  ) => GraphDataView<ValueFormat>,
  definition: ReviewNearestCase
) {
  const geometry = (
    side: string,
    packed: CPUSpatialJoinGeometry
  ): GPUPackedGeometryPort<GPUMultipartGeometryPort> => {
    const positions = allocate(
      `${side}-positions`,
      'float32x2',
      packed.positions.length / 2,
      packed.positions
    );
    const featureOffsets = allocate(
      `${side}-offsets`,
      'uint32',
      packed.featureOffsets!.length,
      packed.featureOffsets!
    );
    return packed.kind === 'points'
      ? {kind: 'points', positions, featureOffsets}
      : {
          kind: 'lines',
          positions,
          featureOffsets,
          lineOffsets: allocate(
            `${side}-lines`,
            'uint32',
            packed.lineOffsets!.length,
            packed.lineOffsets!
          )
        };
  };
  const queryCount = definition.queries.featureOffsets!.length - 1;
  const featureCount = definition.features.featureOffsets!.length - 1;
  const capacity = definition.nearest ? 1 : featureCount;
  const results: Record<string, GraphDataView> = {};
  const output = <ValueFormat extends Format>(
    name: string,
    format: ValueFormat,
    length: number
  ) => {
    const view = allocate(
      name,
      format,
      length,
      format === 'uint32'
        ? new Uint32Array(length).fill(123)
        : new Float32Array(length * (format === 'float32x2' ? 2 : 1)).fill(123)
    );
    results[name] = view;
    return view;
  };
  const radius =
    definition.radius === undefined
      ? undefined
      : allocate('radius', 'float32', 1, Float32Array.of(definition.radius));
  const join = new GPUNearestFeatureJoin({
    queries: geometry('query', definition.queries),
    features: geometry('feature', definition.features),
    radius,
    spatialSort: false,
    querySort: false,
    ...(definition.nearest ? {} : {neighborCount: capacity}),
    output: {
      ...(definition.nearest
        ? {
            nearestFeatureIds: output('rows', 'uint32', queryCount),
            nearestDistances: output('distances', 'float32', queryCount)
          }
        : {
            neighborIds: output('rows', 'uint32', queryCount * capacity),
            neighborCounts: output('counts', 'uint32', queryCount),
            neighborDistances: output('distances', 'float32', queryCount * capacity),
            neighborFootPoints: output('featurePoints', 'float32x2', queryCount * capacity),
            neighborQueryPoints: output('queryPoints', 'float32x2', queryCount * capacity)
          }),
      status: {overflow: output('overflow', 'uint32', 1)}
    }
  });
  return {join, results, radius, capacity, definition};
}

export async function assertReviewNearestFixture(
  fixture: ReturnType<typeof createReviewNearestFixture>,
  read: (view: GraphDataView) => Promise<number[]>
) {
  const {results, definition} = fixture;
  expect(await read(results['rows'])).toEqual(definition.expectedRows);
  if (results['counts']) {
    expect(await read(results['counts'])).toEqual(
      Array.from(
        {length: definition.queries.featureOffsets!.length - 1},
        (_, row) =>
          definition.expectedRows
            .slice(row * fixture.capacity, (row + 1) * fixture.capacity)
            .filter(value => value !== 0xffffffff).length
      )
    );
  }
  const distances = await read(results['distances']);
  distances.forEach((distance, index) => {
    expect(Number.isFinite(distance)).toBe(true);
    const expected = definition.expectedDistances[index];
    // Relative float32 tolerance preserves the distinction between zero and the tiny R1 gap.
    expect(Math.abs(distance - expected)).toBeLessThanOrEqual(
      Math.max(1e-14, Math.abs(expected) * 2e-5)
    );
    if (expected > 0) expect(distance).toBeGreaterThan(0);
  });
  if (results['queryPoints']) {
    for (const port of ['queryPoints', 'featurePoints']) {
      const points = await read(results[port]);
      definition.expectedRows.forEach((row, slot) => {
        if (row === 0xffffffff) {
          expect(points[slot * 2]).toBeNaN();
          expect(points[slot * 2 + 1]).toBeNaN();
        }
      });
    }
  }
  if (definition.coincidentPoint) {
    expect(await read(results['queryPoints'])).toEqual(definition.coincidentPoint);
    expect(await read(results['featurePoints'])).toEqual(definition.coincidentPoint);
  }
  if (definition.horizontalSeparation !== undefined) {
    const queryPoint = await read(results['queryPoints']);
    const featurePoint = await read(results['featurePoints']);
    // Hand oracle: any x in [0,10] is a minimizer, but both points must use the SAME x.
    for (const point of [queryPoint, featurePoint]) {
      expect(point.every(Number.isFinite)).toBe(true);
      expect(point[0]).toBeGreaterThanOrEqual(0);
      expect(point[0]).toBeLessThanOrEqual(10);
    }
    expect(queryPoint[1]).toBe(0);
    expect(featurePoint[1]).toBe(definition.horizontalSeparation);
    expect(
      Math.abs(
        Math.hypot(queryPoint[0] - featurePoint[0], queryPoint[1] - featurePoint[1]) - distances[0]
      )
    ).toBeLessThanOrEqual(1e-6);
  }
  expect(await read(results['overflow'])).toEqual([0]);
}
