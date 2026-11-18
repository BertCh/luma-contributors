// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  fitResidentSpatialJoinCapacities,
  getInitialResidentSpatialJoinCapacities,
  getResidentSpatialJoinAggregateSource,
  getResidentSpatialJoinWorkingBytes,
  growResidentSpatialJoinCapacities,
  RESIDENT_SPATIAL_JOIN_CHUNK_ROWS,
  RESIDENT_SPATIAL_JOIN_WORKING_BYTES,
  type ResidentSpatialJoinAggregateProps
} from '../../../examples/deck/kepler-compute/src/resident-spatial-join-aggregate';

test('initial capacities chunk large inputs without capping them', () => {
  const small = getInitialResidentSpatialJoinCapacities(10, 3);
  expect(small.chunkRowCount).toBe(10);
  const large = getInitialResidentSpatialJoinCapacities(500_000_000, 1_000_000);
  expect(large.chunkRowCount).toBe(RESIDENT_SPATIAL_JOIN_CHUNK_ROWS);
  const single = getInitialResidentSpatialJoinCapacities(1000, 1);
  expect(single.candidateCapacity).toBeGreaterThanOrEqual(1000);
  expect(single.pairCapacity).toBeGreaterThanOrEqual(1000);
});

test('growth is strictly larger than an overflowed stage and bounded by candidates', () => {
  const previous = {chunkRowCount: 1000, candidateCapacity: 2016, pairCapacity: 1016};
  const grownCandidates = growResidentSpatialJoinCapacities(previous, {
    rowCount: 1000,
    candidateCount: 9000,
    requiredCount: 100,
    candidateOverflow: true
  });
  expect(grownCandidates.candidateCapacity).toBeGreaterThan(9000);
  // Matches never exceed candidates, so pairs grow with candidate density while it overflowed.
  expect(grownCandidates.pairCapacity).toBeGreaterThan(9000);
  const grownPairs = growResidentSpatialJoinCapacities(previous, {
    rowCount: 500,
    candidateCount: 1500,
    requiredCount: 3000,
    candidateOverflow: false
  });
  expect(grownPairs.candidateCapacity).toBeGreaterThanOrEqual(previous.candidateCapacity);
  expect(grownPairs.pairCapacity).toBeGreaterThan(6000);
});

test('fit halves the chunk until the working set fits the reservation', () => {
  const dense = {chunkRowCount: 1_048_576, candidateCapacity: 40_000_000, pairCapacity: 20_000_000};
  expect(getResidentSpatialJoinWorkingBytes(dense, 1000)).toBeGreaterThan(
    RESIDENT_SPATIAL_JOIN_WORKING_BYTES
  );
  const fitted = fitResidentSpatialJoinCapacities(dense, 1000);
  expect(fitted.chunkRowCount).toBeLessThan(dense.chunkRowCount);
  expect(getResidentSpatialJoinWorkingBytes(fitted, 1000)).toBeLessThanOrEqual(
    RESIDENT_SPATIAL_JOIN_WORKING_BYTES
  );
  expect(() =>
    fitResidentSpatialJoinCapacities(
      {chunkRowCount: 1, candidateCapacity: 1_000_000_000, pairCapacity: 1},
      1000
    )
  ).toThrow(/SQL/);
});

test('aggregate kernel source encodes mode, format and validity', () => {
  const view = (length: number) => ({length, byteOffset: 0}) as never;
  const props = {
    id: 'aggregate',
    technique: 'average',
    sourceFormat: 'sint32',
    leftRows: view(64),
    rightRows: view(64),
    pairCount: view(1),
    sourceValues: view(8),
    sourceValidity: view(8),
    dummy: view(1),
    outputValues: view(32),
    outputValidity: view(32)
  } as unknown as ResidentSpatialJoinAggregateProps;
  const layout = {x: 1, y: 1, z: 1} as never;
  const source = getResidentSpatialJoinAggregateSource(props, layout);
  expect(source).toContain('const MODE: u32 = 2u;');
  expect(source).toContain('const SOURCE_FORMAT: u32 = 1u;');
  expect(source).toContain('const HAS_VALIDITY: bool = true;');
  expect(source).toContain('const PAIR_CAPACITY: u32 = 64u;');
  const count = getResidentSpatialJoinAggregateSource(
    {...props, technique: 'count', sourceValidity: undefined},
    layout
  );
  expect(count).toContain('const MODE: u32 = 0u;');
  expect(count).toContain('const HAS_VALIDITY: bool = false;');
});
