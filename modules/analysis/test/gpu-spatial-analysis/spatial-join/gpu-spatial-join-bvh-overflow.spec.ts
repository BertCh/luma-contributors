// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUBufferSelection,
  GPUNearestFeatureJoin,
  GPUPointInPolygonJoin,
  GPUSpatialJoinPrepared,
  GPUSpatialPredicateJoin,
  GPU_SPATIAL_JOIN_NO_FEATURE,
  type GPUSpatialJoinGeometry
} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {GPUZonalStatistics} from '../../../src/gpu-spatial-analysis/zonal-statistics/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {buildOracleArrays, joinWithOracle, point, rectangle} from './spatial-predicate-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

/**
 * A BVH whose `leafCapacity` is below the feature count keeps only the first `leafCapacity`
 * features (no spatial sort below 256 features), so with a capacity of 2 the squares 2 and 3 are
 * dropped. Point 1 lies in square 2 and point 3 in square 3: before the BVH overflow was reported,
 * both looked unmatched, and the anti join listed them although they match.
 */
const SMALL_LEAF_CAPACITY = 2;
const FULL_LEAF_CAPACITY = 4;
const POINTS = [point(1, 1), point(21, 1), point(50, 50), point(31, 1)];
const SQUARES = [
  rectangle(0, 0, 2, 2),
  rectangle(10, 0, 2, 2),
  rectangle(20, 0, 2, 2),
  rectangle(30, 0, 2, 2)
];
const ALL_FLAGGED = [1, 1, 1, 1];
const NONE_FLAGGED = [0, 0, 0, 0];

/** CPU reference: matched pairs, and the left rows with no match (the anti join). */
const REFERENCE_PAIRS = joinWithOracle('intersects', POINTS, SQUARES);
const REFERENCE_UNMATCHED = POINTS.map((_, row) => row).filter(
  row => !REFERENCE_PAIRS.some(([left]) => left === row)
);
/** CPU reference containing feature per point, or no feature. */
const REFERENCE_CONTAINING = POINTS.map(
  (_, row) => REFERENCE_PAIRS.find(([left]) => left === row)?.[1] ?? GPU_SPATIAL_JOIN_NO_FEATURE
);

type Fixture = {
  device: Device;
  graph: GPUCommandGraph;
  buffers: Buffer[];
  upload: (data: Float32Array | Uint32Array) => Buffer;
  output: (name: string, length: number) => {buffer: Buffer; view: ReturnType<typeof view>};
  destroy: () => void;
};

function view(graph: GPUCommandGraph, name: string, buffer: Buffer, length: number) {
  return importGraphBuffer(graph, name, buffer, 'uint32', length);
}

function createFixture(device: Device, id: string): Fixture {
  const graph = new GPUCommandGraph(device, {id});
  const buffers: Buffer[] = [];
  return {
    device,
    graph,
    buffers,
    upload: data => {
      const buffer = createInputBuffer(device, data);
      buffers.push(buffer);
      return buffer;
    },
    output: (name, length) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: view(graph, name, buffer, length)};
    },
    destroy: () => {
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function createGeometry(
  fixture: Fixture,
  name: string,
  kind: 'points' | 'polygons',
  features: typeof POINTS
): GPUSpatialJoinGeometry {
  const arrays = buildOracleArrays(kind, features);
  const positions = importGraphBuffer(
    fixture.graph,
    `${name}-positions`,
    fixture.upload(arrays.positions),
    'float32x2',
    arrays.positions.length / 2
  );
  if (arrays.kind === 'points') {
    return {kind: 'points', positions};
  }
  if (arrays.kind !== 'polygons') {
    throw new Error('unexpected geometry kind');
  }
  const offsets = (suffix: string, data: Uint32Array) =>
    view(fixture.graph, `${name}-${suffix}`, fixture.upload(data), data.length);
  return {
    kind: 'polygons',
    positions,
    featureOffsets: offsets('feature-offsets', arrays.featureOffsets),
    polygonOffsets: offsets('polygon-offsets', arrays.polygonOffsets),
    ringOffsets: offsets('ring-offsets', arrays.ringOffsets)
  };
}

type PredicateResult = {
  pairs: [number, number][];
  unmatched: number[];
  truncated: number[];
  candidateOverflow: number;
  truncatedCount: number;
  overflow: number;
};

async function runPredicateJoin(
  device: Device,
  how: 'inner' | 'anti',
  leafCapacity: number,
  usePrepared = false
): Promise<PredicateResult> {
  const fixture = createFixture(device, `bvh-overflow-predicate-${how}`);
  const {graph} = fixture;
  const leftCount = POINTS.length;
  const left = createGeometry(fixture, 'left', 'points', POINTS);
  const right = createGeometry(fixture, 'right', 'polygons', SQUARES);
  const prepared = usePrepared
    ? new GPUSpatialJoinPrepared({geometry: right, leafCapacity, spatialSort: false})
    : undefined;
  if (prepared) {
    graph.add(prepared);
  }
  const leftRows = fixture.output('left-ids', 16);
  const rightRows = fixture.output('right-ids', 16);
  const count = fixture.output('count', 1);
  const pairOverflow = fixture.output('pair-overflow', 1);
  const unmatchedIds = fixture.output('unmatched-ids', leftCount);
  const unmatchedCount = fixture.output('unmatched-count', 1);
  const unmatchedOverflow = fixture.output('unmatched-overflow', 1);
  const overflow = fixture.output('overflow', 1);
  const candidateOverflow = fixture.output('candidate-overflow', 1);
  const truncatedCount = fixture.output('truncated-count', 1);
  const truncated = fixture.output('truncated', leftCount);
  graph.add(
    new GPUSpatialPredicateJoin({
      left,
      right,
      predicate: 'intersects',
      how,
      candidateCapacity: 16,
      leafCapacity: prepared ? undefined : leafCapacity,
      prepared,
      output: {
        ...(how === 'inner'
          ? {
              pairs: {
                leftRows: leftRows.view,
                rightRows: rightRows.view,
                status: {count: count.view, overflow: pairOverflow.view}
              }
            }
          : {
              unmatched: {
                ids: unmatchedIds.view,
                status: {
                  count: unmatchedCount.view,
                  overflow: unmatchedOverflow.view,
                  candidateOverflow: candidateOverflow.view
                }
              }
            }),
        candidateTruncated: truncated.view,
        status: {
          overflow: overflow.view,
          candidateOverflow: how === 'inner' ? candidateOverflow.view : undefined,
          candidateTruncatedCount: truncatedCount.view
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [pairCount] = await readUint32(count.buffer, 1);
  const [unmatchedTotal] = await readUint32(unmatchedCount.buffer, 1);
  const lefted = await readUint32(leftRows.buffer, 16);
  const righted = await readUint32(rightRows.buffer, 16);
  const result = {
    pairs:
      how === 'inner'
        ? lefted.slice(0, pairCount).map((row, slot): [number, number] => [row, righted[slot]])
        : [],
    unmatched:
      how === 'anti'
        ? (await readUint32(unmatchedIds.buffer, leftCount)).slice(0, unmatchedTotal)
        : [],
    truncated: await readUint32(truncated.buffer, leftCount),
    candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0],
    truncatedCount: (await readUint32(truncatedCount.buffer, 1))[0],
    overflow: (await readUint32(overflow.buffer, 1))[0]
  };
  compiled.destroy();
  prepared?.destroy();
  fixture.destroy();
  return result;
}

it('reference join: points 1 and 3 match only squares a two-leaf BVH drops', () => {
  expect(REFERENCE_UNMATCHED).toEqual([2]);
  expect(REFERENCE_CONTAINING).toEqual([0, 2, GPU_SPATIAL_JOIN_NO_FEATURE, 3]);
});

it('GPUSpatialPredicateJoin anti join withholds every row after a BVH leaf overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const usePrepared of [false, true]) {
    const label = usePrepared ? 'prepared' : 'built';
    const complete = await runPredicateJoin(device, 'anti', FULL_LEAF_CAPACITY, usePrepared);
    expect(complete.unmatched, label).toEqual(REFERENCE_UNMATCHED);
    expect(complete.truncated, label).toEqual(NONE_FLAGGED);
    expect(complete.candidateOverflow, label).toBe(0);
    expect(complete.truncatedCount, label).toBe(0);

    const overflowing = await runPredicateJoin(device, 'anti', SMALL_LEAF_CAPACITY, usePrepared);
    // Before 2026-11-10 the output was [1, 2, 3]: rows 1 and 3 match a dropped square. An anti
    // join must never list a row the reference join matches.
    for (const row of overflowing.unmatched) {
      expect(REFERENCE_UNMATCHED, `${label} row ${row}`).toContain(row);
    }
    expect(overflowing.unmatched, label).toEqual([]);
    expect(overflowing.truncated, label).toEqual(ALL_FLAGGED);
    expect(overflowing.truncatedCount, label).toBe(POINTS.length);
    expect(overflowing.candidateOverflow, label).toBe(1);
    expect(overflowing.overflow, label).toBe(1);
  }
});

it('GPUSpatialPredicateJoin inner join flags every row whose pairs a BVH overflow lost', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const complete = await runPredicateJoin(device, 'inner', FULL_LEAF_CAPACITY);
  expect(complete.pairs).toEqual(REFERENCE_PAIRS);
  expect(complete.truncated).toEqual(NONE_FLAGGED);

  const overflowing = await runPredicateJoin(device, 'inner', SMALL_LEAF_CAPACITY);
  const found = new Set(overflowing.pairs.map(pair => pair.join(',')));
  for (const pair of overflowing.pairs) {
    expect(REFERENCE_PAIRS.map(reference => reference.join(','))).toContain(pair.join(','));
  }
  for (const [left, right] of REFERENCE_PAIRS) {
    if (!found.has(`${left},${right}`)) {
      expect(overflowing.truncated[left], `row ${left} lost (${left}, ${right})`).toBe(1);
    }
  }
  expect(overflowing.truncated).toEqual(ALL_FLAGGED);
  expect(overflowing.candidateOverflow).toBe(1);
});

it('GPUPointInPolygonJoin flags every point after a BVH leaf overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const leafCapacity of [FULL_LEAF_CAPACITY, SMALL_LEAF_CAPACITY]) {
    const fixture = createFixture(device, `bvh-overflow-pip-${leafCapacity}`);
    const points = createGeometry(fixture, 'points', 'points', POINTS);
    const polygons = createGeometry(fixture, 'polygons', 'polygons', SQUARES);
    if (points.kind !== 'points' || polygons.kind !== 'polygons') {
      throw new Error('unexpected geometry kind');
    }
    const featureIds = fixture.output('point-feature-ids', POINTS.length);
    const overflow = fixture.output('overflow', 1);
    const candidateOverflow = fixture.output('candidate-overflow', 1);
    const truncated = fixture.output('truncated', POINTS.length);
    fixture.graph.add(
      new GPUPointInPolygonJoin({
        points,
        polygons,
        candidateCapacity: 16,
        leafCapacity,
        output: {
          pointFeatureIds: featureIds.view,
          candidateTruncated: truncated.view,
          status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
        }
      })
    );
    const compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    const assigned = await readUint32(featureIds.buffer, POINTS.length);
    const flags = await readUint32(truncated.buffer, POINTS.length);
    // Every point whose assignment differs from the reference must be flagged.
    for (const [row, feature] of assigned.entries()) {
      if (feature !== REFERENCE_CONTAINING[row]) {
        expect(flags[row], `point ${row}`).toBe(1);
      }
    }
    const overflowed = leafCapacity < SQUARES.length;
    expect(flags).toEqual(overflowed ? ALL_FLAGGED : NONE_FLAGGED);
    expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([overflowed ? 1 : 0]);
    expect(await readUint32(overflow.buffer, 1)).toEqual([overflowed ? 1 : 0]);
    if (!overflowed) {
      expect(assigned).toEqual(REFERENCE_CONTAINING);
    }
    compiled.destroy();
    fixture.destroy();
  }
});

/** Point features of the nearest joins and their queries; query 2 has no feature in range. */
const NEAREST_FEATURES = [
  [0, 0],
  [1, 0],
  [2, 0],
  [3, 0]
];
const NEAREST_QUERIES = [
  [0.001, 0],
  [2.001, 0],
  [50, 50],
  [3.001, 0]
];
const NEAREST_REFERENCE = [0, 2, GPU_SPATIAL_JOIN_NO_FEATURE, 3];

type NearestMode = 'planar' | 'great-circle' | 'neighbors' | 'great-circle-neighbors';

async function runNearestJoin(
  device: Device,
  mode: NearestMode,
  leafCapacity: number
): Promise<{ids: number[]; truncated: number[]; candidateOverflow: number; overflow: number}> {
  const fixture = createFixture(device, `bvh-overflow-nearest-${mode}`);
  const {graph} = fixture;
  const queryCount = NEAREST_QUERIES.length;
  const greatCircle = mode === 'great-circle' || mode === 'great-circle-neighbors';
  const neighbors = mode === 'neighbors' || mode === 'great-circle-neighbors';
  // Planar distances are in coordinate units; great-circle distances in meters (0.001 degree is
  // about 111 m at the equator, one degree about 111 km).
  const radius = new GPUParameterBuffer(device, {
    id: 'radius',
    format: 'float32',
    length: 1,
    values: Float32Array.of(greatCircle ? 2000 : 0.1)
  });
  const ids = fixture.output('ids', queryCount);
  const counts = fixture.output('counts', queryCount);
  const overflow = fixture.output('overflow', 1);
  const candidateOverflow = fixture.output('candidate-overflow', 1);
  const truncated = fixture.output('truncated', queryCount);
  graph.add(
    new GPUNearestFeatureJoin({
      ...(greatCircle
        ? {spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const}
        : {}),
      queries: {
        kind: 'points',
        positions: importGraphBuffer(
          graph,
          'queries',
          fixture.upload(Float32Array.from(NEAREST_QUERIES.flat())),
          'float32x2',
          queryCount
        )
      },
      features: {
        kind: 'points',
        positions: importGraphBuffer(
          graph,
          'features',
          fixture.upload(Float32Array.from(NEAREST_FEATURES.flat())),
          'float32x2',
          NEAREST_FEATURES.length
        )
      },
      radius: radius.importToGraph(graph),
      leafCapacity,
      spatialSort: false,
      ...(neighbors ? {neighborCount: 1} : {candidateCapacity: 16}),
      output: {
        ...(neighbors
          ? {neighborIds: ids.view, neighborCounts: counts.view}
          : {nearestFeatureIds: ids.view}),
        candidateTruncated: truncated.view,
        status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    ids: await readUint32(ids.buffer, queryCount),
    truncated: await readUint32(truncated.buffer, queryCount),
    candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0],
    overflow: (await readUint32(overflow.buffer, 1))[0]
  };
  compiled.destroy();
  radius.destroy();
  fixture.destroy();
  return result;
}

it('GPUNearestFeatureJoin reports a BVH leaf overflow in every mode', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const mode of ['planar', 'great-circle', 'neighbors', 'great-circle-neighbors'] as const) {
    const complete = await runNearestJoin(device, mode, FULL_LEAF_CAPACITY);
    expect(complete.ids, mode).toEqual(NEAREST_REFERENCE);
    expect(complete.truncated, mode).toEqual(NONE_FLAGGED);
    expect(complete.candidateOverflow, mode).toBe(0);

    const overflowing = await runNearestJoin(device, mode, SMALL_LEAF_CAPACITY);
    for (const [row, feature] of overflowing.ids.entries()) {
      if (feature !== NEAREST_REFERENCE[row]) {
        expect(overflowing.truncated[row], `${mode} query ${row}`).toBe(1);
      }
    }
    // Queries 1 and 3 lose their nearest feature, so the overflow is observable.
    expect(overflowing.ids, mode).not.toEqual(NEAREST_REFERENCE);
    expect(overflowing.truncated, mode).toEqual(ALL_FLAGGED);
    expect(overflowing.candidateOverflow, mode).toBe(1);
    expect(overflowing.overflow, mode).toBe(1);
  }
});

/** Rows that lose candidates to a capacity of 1: points 0, 1 and 3 have one candidate each. */
const CAPACITY_ONE_TRUNCATED = [0, 1, 0, 1];

async function runZonalStatistics(
  device: Device,
  capacities: {candidateCapacity: number; leafCapacity: number}
): Promise<{counts: number[]; truncated: number[]; candidateOverflow: number; overflow: number}> {
  const fixture = createFixture(device, 'zonal-candidate-status');
  const points = createGeometry(fixture, 'points', 'points', POINTS);
  const polygons = createGeometry(fixture, 'polygons', 'polygons', SQUARES);
  if (points.kind !== 'points' || polygons.kind !== 'polygons') {
    throw new Error('unexpected geometry kind');
  }
  const counts = fixture.output('counts', SQUARES.length);
  const truncated = fixture.output('truncated', POINTS.length);
  const overflow = fixture.output('overflow', 1);
  const candidateOverflow = fixture.output('candidate-overflow', 1);
  fixture.graph.add(
    new GPUZonalStatistics({
      features: {...polygons, ...capacities},
      positions: points.positions,
      output: {
        counts: counts.view,
        candidateTruncated: truncated.view,
        status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
      }
    })
  );
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    counts: await readUint32(counts.buffer, SQUARES.length),
    truncated: await readUint32(truncated.buffer, POINTS.length),
    candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0],
    overflow: (await readUint32(overflow.buffer, 1))[0]
  };
  compiled.destroy();
  fixture.destroy();
  return result;
}

it('GPUZonalStatistics passes the join candidate status through', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const referenceCounts = SQUARES.map(
    (_, feature) => REFERENCE_CONTAINING.filter(row => row === feature).length
  );
  const complete = await runZonalStatistics(device, {candidateCapacity: 16, leafCapacity: 4});
  expect(complete.counts).toEqual(referenceCounts);
  expect(complete.truncated).toEqual(NONE_FLAGGED);
  expect(complete.candidateOverflow).toBe(0);

  const candidateLimited = await runZonalStatistics(device, {
    candidateCapacity: 1,
    leafCapacity: 4
  });
  expect(candidateLimited.counts).toEqual([1, 0, 0, 0]);
  expect(candidateLimited.truncated).toEqual(CAPACITY_ONE_TRUNCATED);
  expect(candidateLimited.candidateOverflow).toBe(1);
  expect(candidateLimited.overflow).toBe(1);

  const leafLimited = await runZonalStatistics(device, {candidateCapacity: 16, leafCapacity: 2});
  expect(leafLimited.counts).toEqual([1, 0, 0, 0]);
  expect(leafLimited.truncated).toEqual(ALL_FLAGGED);
  expect(leafLimited.candidateOverflow).toBe(1);
  expect(leafLimited.overflow).toBe(1);
});

async function runBufferSelection(
  device: Device,
  capacities: {candidateCapacity: number; leafCapacity: number}
): Promise<{mask: number[]; truncated: number[]; candidateOverflow: number; overflow: number}> {
  const fixture = createFixture(device, 'buffer-selection-candidate-status');
  const {graph} = fixture;
  const queryCount = NEAREST_QUERIES.length;
  const distance = new GPUParameterBuffer(device, {
    id: 'distance',
    format: 'float32',
    length: 1,
    values: Float32Array.of(0.1)
  });
  const mask = fixture.output('mask', queryCount);
  const truncated = fixture.output('truncated', queryCount);
  const overflow = fixture.output('overflow', 1);
  const candidateOverflow = fixture.output('candidate-overflow', 1);
  graph.add(
    new GPUBufferSelection({
      points: {
        kind: 'points',
        positions: importGraphBuffer(
          graph,
          'points',
          fixture.upload(Float32Array.from(NEAREST_QUERIES.flat())),
          'float32x2',
          queryCount
        )
      },
      features: {
        kind: 'points',
        positions: importGraphBuffer(
          graph,
          'features',
          fixture.upload(Float32Array.from(NEAREST_FEATURES.flat())),
          'float32x2',
          NEAREST_FEATURES.length
        )
      },
      radius: distance.importToGraph(graph),
      spatialSort: false,
      ...capacities,
      output: {
        outputMask: mask.view,
        candidateTruncated: truncated.view,
        status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    mask: await readUint32(mask.buffer, queryCount),
    truncated: await readUint32(truncated.buffer, queryCount),
    candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0],
    overflow: (await readUint32(overflow.buffer, 1))[0]
  };
  compiled.destroy();
  distance.destroy();
  fixture.destroy();
  return result;
}

it('GPUBufferSelection passes the join candidate status through', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const referenceMask = NEAREST_REFERENCE.map(feature =>
    feature === GPU_SPATIAL_JOIN_NO_FEATURE ? 0 : 1
  );
  const complete = await runBufferSelection(device, {candidateCapacity: 16, leafCapacity: 4});
  expect(complete.mask).toEqual(referenceMask);
  expect(complete.truncated).toEqual(NONE_FLAGGED);
  expect(complete.candidateOverflow).toBe(0);

  const candidateLimited = await runBufferSelection(device, {
    candidateCapacity: 1,
    leafCapacity: 4
  });
  expect(candidateLimited.mask).toEqual([1, 0, 0, 0]);
  expect(candidateLimited.truncated).toEqual(CAPACITY_ONE_TRUNCATED);
  expect(candidateLimited.candidateOverflow).toBe(1);
  expect(candidateLimited.overflow).toBe(1);

  const leafLimited = await runBufferSelection(device, {candidateCapacity: 16, leafCapacity: 2});
  expect(leafLimited.mask).toEqual([1, 0, 0, 0]);
  expect(leafLimited.truncated).toEqual(ALL_FLAGGED);
  expect(leafLimited.candidateOverflow).toBe(1);
  expect(leafLimited.overflow).toBe(1);
});

it('join consumers reject a candidateTruncated view of the wrong shape', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device, 'candidate-status-rejections');
  const points = createGeometry(fixture, 'points', 'points', POINTS);
  const polygons = createGeometry(fixture, 'polygons', 'polygons', SQUARES);
  if (points.kind !== 'points' || polygons.kind !== 'polygons') {
    throw new Error('unexpected geometry kind');
  }
  const counts = fixture.output('counts', SQUARES.length);
  const shortFlags = fixture.output('short-flags', POINTS.length - 1).view;
  const pointRows = fixture.output('point-rows', POINTS.length).view;
  expect(
    () =>
      new GPUZonalStatistics({
        features: {...polygons, candidateCapacity: 16},
        positions: points.positions,
        output: {counts: counts.view, candidateTruncated: shortFlags}
      })
  ).toThrow(/^zonal-statistics output\.candidateTruncated needs one row per point \(4\)/);
  expect(
    () =>
      new GPUZonalStatistics({
        features: {kind: 'feature-rows', pointFeatureRows: pointRows, featureCount: 4},
        output: {counts: counts.view, candidateTruncated: fixture.output('flags', 4).view}
      })
  ).toThrow(/^zonal-statistics output\.candidateTruncated needs features\.kind 'polygons'/);
  const distance = new GPUParameterBuffer(device, {
    id: 'distance',
    format: 'float32',
    length: 1,
    values: Float32Array.of(1)
  });
  expect(
    () =>
      new GPUBufferSelection({
        points,
        features: {kind: 'points', positions: points.positions},
        radius: distance.importToGraph(fixture.graph),
        candidateCapacity: 16,
        output: {outputMask: pointRows, candidateTruncated: shortFlags}
      })
  ).toThrow(/^buffer-selection output\.candidateTruncated needs one row per point \(4\)/);
  expect(
    () =>
      new GPUNearestFeatureJoin({
        queries: points,
        features: {kind: 'points', positions: points.positions},
        neighborCount: 1,
        output: {
          neighborIds: fixture.output('neighbor-ids', 4).view,
          neighborCounts: fixture.output('neighbor-counts', 4).view,
          candidateTruncated: shortFlags,
          status: {overflow: fixture.output('nearest-overflow', 1).view}
        }
      })
  ).toThrow(/^nearest-feature-join output\.candidateTruncated needs one row per query/);
  distance.destroy();
  fixture.destroy();
});
