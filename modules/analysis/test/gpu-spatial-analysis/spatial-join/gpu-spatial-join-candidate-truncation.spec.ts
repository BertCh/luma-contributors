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
  GPUNearestFeatureJoin,
  GPUPointInPolygonJoin,
  GPUSpatialPredicateJoin,
  type GPUSpatialJoinGeometry
} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {buildOracleArrays, point, rectangle} from './spatial-predicate-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

/**
 * Rows 0, 1 and 3 each have two bounding-box candidates (row 2 has none), laid out in row order:
 * row 0 owns slots 0-1, row 1 owns slots 2-3 and row 3 owns slots 4-5. With a capacity of 3 the
 * slots 3, 4 and 5 are dropped, so row 1 (crosses the capacity) and row 3 (past it) are truncated.
 */
const CANDIDATE_CAPACITY = 3;
const EXPECTED_TRUNCATED = [0, 1, 0, 1];
const QUERY_POINTS = [point(6, 6), point(7, 7), point(100, 100), point(8, 8)];
const SQUARES = [rectangle(0, 0, 10, 10), rectangle(5, 5, 5, 5)];

type Fixture = {
  graph: GPUCommandGraph;
  buffers: Buffer[];
  upload: (data: Float32Array | Uint32Array) => Buffer;
  output: (name: string, length: number) => {buffer: Buffer; view: ReturnType<typeof view>};
};

function view(graph: GPUCommandGraph, name: string, buffer: Buffer, length: number) {
  return importGraphBuffer(graph, name, buffer, 'uint32', length);
}

function createFixture(device: Device): Fixture {
  const graph = new GPUCommandGraph(device, {id: 'candidate-truncation'});
  const buffers: Buffer[] = [];
  return {
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
    }
  };
}

function createGeometry(
  fixture: Fixture,
  name: string,
  kind: 'points' | 'polygons',
  features: typeof QUERY_POINTS
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

async function runPredicateJoin(
  device: Device,
  how: 'inner' | 'anti',
  candidateCapacity: number
): Promise<{
  pairs: [number, number][];
  unmatched: number[];
  truncated: number[];
  candidateOverflow: number;
  truncatedCount: number;
  overflow: number;
}> {
  const fixture = createFixture(device);
  const {graph} = fixture;
  const leftCount = QUERY_POINTS.length;
  const left = createGeometry(fixture, 'left', 'points', QUERY_POINTS);
  const right = createGeometry(fixture, 'right', 'polygons', SQUARES);
  const leftIds = fixture.output('left-ids', 16);
  const rightIds = fixture.output('right-ids', 16);
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
      candidateCapacity,
      output: {
        ...(how === 'inner'
          ? {
              pairs: {
                leftIds: leftIds.view,
                rightIds: rightIds.view,
                status: {count: count.view, overflow: pairOverflow.view}
              }
            }
          : {}),
        ...(how === 'anti'
          ? {
              unmatched: {
                ids: unmatchedIds.view,
                status: {count: unmatchedCount.view, overflow: unmatchedOverflow.view}
              }
            }
          : {}),
        candidateTruncated: truncated.view,
        status: {
          overflow: overflow.view,
          candidateOverflow: candidateOverflow.view,
          candidateTruncatedCount: truncatedCount.view
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [pairCount] = await readUint32(count.buffer, 1);
  const [unmatchedTotal] = await readUint32(unmatchedCount.buffer, 1);
  const lefted = await readUint32(leftIds.buffer, 16);
  const righted = await readUint32(rightIds.buffer, 16);
  const result = {
    pairs: lefted.slice(0, pairCount).map((row, slot): [number, number] => [row, righted[slot]]),
    unmatched: (await readUint32(unmatchedIds.buffer, leftCount)).slice(0, unmatchedTotal),
    truncated: await readUint32(truncated.buffer, leftCount),
    candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0],
    truncatedCount: (await readUint32(truncatedCount.buffer, 1))[0],
    overflow: (await readUint32(overflow.buffer, 1))[0]
  };
  compiled.destroy();
  for (const buffer of fixture.buffers) {
    buffer.destroy();
  }
  return result;
}

it('GPUSpatialPredicateJoin marks exactly the rows that lost candidates to a small capacity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const small = await runPredicateJoin(device, 'inner', CANDIDATE_CAPACITY);
  expect(small.candidateOverflow).toBe(1);
  expect(small.overflow).toBe(1);
  expect(small.truncated).toEqual(EXPECTED_TRUNCATED);
  expect(small.truncatedCount).toBe(2);
  // Slots 0-2 survive: row 0 against both squares and row 1 against the first.
  expect(small.pairs).toEqual([
    [0, 0],
    [0, 1],
    [1, 0]
  ]);

  const enough = await runPredicateJoin(device, 'inner', 6);
  expect(enough.candidateOverflow).toBe(0);
  expect(enough.overflow).toBe(0);
  expect(enough.truncated).toEqual([0, 0, 0, 0]);
  expect(enough.truncatedCount).toBe(0);
});

it('GPUSpatialPredicateJoin anti join keeps rows with dropped candidates out of the unmatched output', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Without overflow, only the point outside every square is unmatched.
  const complete = await runPredicateJoin(device, 'anti', 6);
  expect(complete.unmatched).toEqual([2]);
  expect(complete.truncatedCount).toBe(0);

  // Row 3 lost every candidate and would look unmatched; it is unknown and must not be emitted.
  const overflowing = await runPredicateJoin(device, 'anti', CANDIDATE_CAPACITY);
  expect(overflowing.candidateOverflow).toBe(1);
  expect(overflowing.truncated).toEqual(EXPECTED_TRUNCATED);
  expect(overflowing.truncatedCount).toBe(2);
  expect(overflowing.unmatched).toEqual([2]);
});

it('GPUPointInPolygonJoin flags the points whose candidates were dropped', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  const {graph} = fixture;
  const points = createGeometry(fixture, 'points', 'points', QUERY_POINTS);
  const polygons = createGeometry(fixture, 'polygons', 'polygons', SQUARES);
  if (points.kind !== 'points' || polygons.kind !== 'polygons') {
    throw new Error('unexpected geometry kind');
  }
  const featureIds = fixture.output('point-feature-ids', QUERY_POINTS.length);
  const overflow = fixture.output('overflow', 1);
  const candidateOverflow = fixture.output('candidate-overflow', 1);
  const truncated = fixture.output('truncated', QUERY_POINTS.length);
  graph.add(
    new GPUPointInPolygonJoin({
      points,
      polygons,
      candidateCapacity: CANDIDATE_CAPACITY,
      output: {
        pointFeatureIds: featureIds.view,
        candidateTruncated: truncated.view,
        status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(truncated.buffer, QUERY_POINTS.length)).toEqual(EXPECTED_TRUNCATED);
  expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([1]);
  expect(await readUint32(overflow.buffer, 1)).toEqual([1]);
  compiled.destroy();
  for (const buffer of fixture.buffers) {
    buffer.destroy();
  }
});

it('GPUNearestFeatureJoin flags the queries whose candidates were dropped', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  const {graph} = fixture;
  const queries = [
    [0.1, 0],
    [20.1, 0],
    [100, 100],
    [0.1, 0.1]
  ];
  const features = [
    [0, 0],
    [0.2, 0],
    [20, 0],
    [20.2, 0]
  ];
  const radius = new GPUParameterBuffer(device, {
    id: 'radius',
    format: 'float32',
    length: 1,
    values: Float32Array.of(1)
  });
  const nearestIds = fixture.output('nearest-ids', queries.length);
  const overflow = fixture.output('overflow', 1);
  const candidateOverflow = fixture.output('candidate-overflow', 1);
  const truncated = fixture.output('truncated', queries.length);
  graph.add(
    new GPUNearestFeatureJoin({
      queries: {
        kind: 'points',
        positions: importGraphBuffer(
          graph,
          'queries',
          fixture.upload(Float32Array.from(queries.flat())),
          'float32x2',
          queries.length
        )
      },
      features: {
        kind: 'points',
        positions: importGraphBuffer(
          graph,
          'features',
          fixture.upload(Float32Array.from(features.flat())),
          'float32x2',
          features.length
        )
      },
      radius: radius.importToGraph(graph),
      candidateCapacity: CANDIDATE_CAPACITY,
      output: {
        nearestFeatureIds: nearestIds.view,
        candidateTruncated: truncated.view,
        status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(truncated.buffer, queries.length)).toEqual(EXPECTED_TRUNCATED);
  expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([1]);
  expect(await readUint32(overflow.buffer, 1)).toEqual([1]);
  compiled.destroy();
  radius.destroy();
  for (const buffer of fixture.buffers) {
    buffer.destroy();
  }
});
