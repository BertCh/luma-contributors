// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {afterAll, expect, test, vi} from 'vitest';

// Node uses isolate:false: discard previously evaluated consumers before applying this file's mocks.
vi.hoisted(() => vi.resetModules());
import {
  FlowColumnReader,
  aggregateFlowPairs,
  deduplicateFlowEndpoints,
  getFlowBounds,
  getFlowEndpointKey,
  sanitizeFlowCounts
} from '../../../examples/deck/kepler-compute/src/flow-gpu-aggregation';
import {
  makeFlowDataFromPairs,
  makeFlowLocationReader,
  makeFlowLocations,
  prepareFlowSource
} from '../../../examples/deck/kepler-compute/src/flow-source';

vi.mock('@flowmap.gl/data', () => ({
  makeLocationWeightGetter: () => () => 1,
  clusterLocations: (locations: unknown[]) => [{zoom: 0, clusters: locations.length}]
}));

// Rows: (0,0)->(1,1) x3 [rows 0,1,3], (1,1)->(0,0), a half-invalid row, and an invalid row.
const columns = {
  sourceLongitude: [0, 0, 1, 0, Number.NaN, 2],
  sourceLatitude: [0, 0, 1, 0, 0, 2],
  targetLongitude: [1, 1, 0, 1, 1, Number.POSITIVE_INFINITY],
  targetLatitude: [1, 1, 0, 1, 1, 2]
};
const counts = [2, 3, 5, 7, 11, 13];

test('float32 bit keys normalize negative zero and reject non-finite coordinates', () => {
  expect(getFlowEndpointKey(-0, 0)).toEqual(getFlowEndpointKey(0, 0));
  expect(getFlowEndpointKey(Number.NaN, 0)).toBeNull();
  expect(getFlowEndpointKey(0, Number.POSITIVE_INFINITY)).toBeNull();
  // Float64 values that round to the same float32 merge (documented difference from native).
  expect(getFlowEndpointKey(1.0000000001, 2)).toEqual(getFlowEndpointKey(1, 2));
  const key = getFlowEndpointKey(1, 2)!;
  const floats = new Float32Array(Uint32Array.of(key.latitudeBits, key.longitudeBits).buffer);
  expect(Array.from(floats)).toEqual([2, 1]);
});

test('location dedupe assigns first-seen ids, source before target, and skips non-finite endpoints', () => {
  const {endpointIds, firstEndpoints} = deduplicateFlowEndpoints(columns);
  // Row 0: source (0,0) id 1, target (1,1) id 2. Row 4 keeps its valid target (id 2), source 0.
  // Row 5: both endpoints are invalid (infinite target) or finite (source (2,2) id 3, target 0).
  expect(Array.from(endpointIds)).toEqual([1, 2, 1, 2, 2, 1, 1, 2, 0, 2, 3, 0]);
  expect(Array.from(firstEndpoints)).toEqual([0, 1, 10]);
  expect(getFlowBounds(columns)).toEqual([0, 0, 2, 2]);
});

test('pair aggregation sums counts, keeps the first row and applies the mask', () => {
  const {endpointIds} = deduplicateFlowEndpoints(columns);
  const all = aggregateFlowPairs(endpointIds, counts);
  expect(Array.from(all.sourceIds)).toEqual([1, 2]);
  expect(Array.from(all.targetIds)).toEqual([2, 1]);
  expect(Array.from(all.counts)).toEqual([12, 5]);
  expect(Array.from(all.firstRows)).toEqual([0, 2]);
  const masked = aggregateFlowPairs(endpointIds, counts, [0, 1, 0, 1, 1, 1]);
  expect(Array.from(masked.counts)).toEqual([10]);
  expect(Array.from(masked.firstRows)).toEqual([1]);
  expect(aggregateFlowPairs(endpointIds, counts, [0, 0, 0, 0, 0, 0]).sourceIds.length).toBe(0);
});

test('aggregated pairs preserve per-location totals of per-row flows', () => {
  const {endpointIds} = deduplicateFlowEndpoints(columns);
  const pairs = aggregateFlowPairs(endpointIds, counts);
  const perRow = new Map<number, number>();
  for (let row = 0; row < counts.length; row++) {
    const [sourceId, targetId] = [endpointIds[row * 2], endpointIds[row * 2 + 1]];
    if (!sourceId || !targetId) continue;
    for (const id of [sourceId, targetId]) perRow.set(id, (perRow.get(id) ?? 0) + counts[row]);
  }
  const perPair = new Map<number, number>();
  for (let pair = 0; pair < pairs.sourceIds.length; pair++)
    for (const id of [pairs.sourceIds[pair], pairs.targetIds[pair]])
      perPair.set(id, (perPair.get(id) ?? 0) + pairs.counts[pair]);
  expect(perPair).toEqual(perRow);
});

test('aggregate locations match the per-row oracle for ids and names', async () => {
  const table = new arrow.Table({
    lon0: arrow.makeVector(new Float64Array(columns.sourceLongitude)),
    lat0: arrow.makeVector(new Float64Array(columns.sourceLatitude)),
    lon1: arrow.makeVector(new Float64Array(columns.targetLongitude)),
    lat1: arrow.makeVector(new Float64Array(columns.targetLatitude)),
    count: arrow.makeVector(new Float64Array(counts)),
    origin: arrow.vectorFromArray(['A', 'B', 'C', 'D', 'E', 'F']),
    destination: arrow.vectorFromArray(['X', 'Y', null, 'W', 'V', 'U'])
  });
  const roles = {
    targetLongitude: 'lon1',
    targetLatitude: 'lat1',
    count: 'count',
    sourceName: 'origin',
    targetName: 'destination'
  };
  const coordinates = {longitude: 'lon0', latitude: 'lat0'};
  const oracle = await prepareFlowSource(table, coordinates, roles);
  const {firstEndpoints} = deduplicateFlowEndpoints(columns);
  const locations = makeFlowLocations(
    firstEndpoints,
    makeFlowLocationReader(table, coordinates, roles)
  );
  expect(locations).toEqual(oracle.locations);
  // The infinite-target row names its valid source from the first row that creates it.
  expect(locations.map(location => location.name)).toEqual(['A', 'X', 'F']);
  const unnamed = makeFlowLocations(
    Uint32Array.of(5),
    makeFlowLocationReader(table, coordinates, {targetLongitude: 'lon1', targetLatitude: 'lat1'})
  );
  expect(unnamed[0].name).toBe('Location#1');
  const flows = makeFlowDataFromPairs(
    {rowCount: 6, locations, clusterLevels: [], bounds: null, byteLength: 0},
    aggregateFlowPairs(deduplicateFlowEndpoints(columns).endpointIds, counts)
  );
  expect(flows.flows).toEqual([
    {index: 0, sourceId: 1, targetId: 2, count: 12},
    {index: 2, sourceId: 2, targetId: 1, count: 5}
  ]);
});

test('column reader walks chunks, converts to float32 and reads nulls as NaN', () => {
  const chunked = new arrow.Vector([
    arrow.makeData({type: new arrow.Float64(), data: new Float64Array([1, 2, 3])}),
    arrow.makeData({type: new arrow.Float64(), data: new Float64Array([4, 5])})
  ]);
  const reader = new FlowColumnReader(chunked);
  expect(Array.from(reader.readFloat32(2))).toEqual([1, 2]);
  expect(Array.from(reader.readFloat32(3))).toEqual([3, 4, 5]);
  const withNull = arrow.vectorFromArray([1, null, 3], new arrow.Float64());
  expect(Array.from(new FlowColumnReader(withNull).readFloat32(3))).toEqual([1, Number.NaN, 3]);
  expect(Array.from(new FlowColumnReader(null).readFloat32(2, 1))).toEqual([1, 1]);
  expect(Array.from(sanitizeFlowCounts(Float32Array.of(1, Number.NaN, Infinity)))).toEqual([
    1, 0, 0
  ]);
});

// Remove mocked dependencies and their consumers before the next shared-worker spec.
afterAll(() => {
  vi.doUnmock('@flowmap.gl/data');
  vi.resetModules();
});
