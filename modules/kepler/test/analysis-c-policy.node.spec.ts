// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  classifyPointInPolygon,
  listSpatialJoinPairs,
  planJoinCardinality,
  planJoinKeyPrecision,
  planJoinOutputColumns,
  planSpatialJoinSemantics
} from '../../../examples/deck/kepler-compute/src/analysis-join-policy';
import {
  CompensatedFloat32Sum,
  assignGroupSlots,
  planGroupByKeyChunks,
  planGroupByMeasureBatches,
  reduceGroupMeasure
} from '../../../examples/deck/kepler-compute/src/analysis-group-by-extended';
import {
  classifyRateInvalidCauses,
  decideNarrowing,
  diagnoseFloat32Narrowing
} from '../../../examples/deck/kepler-compute/src/analysis-numeric-diagnostics';
import {
  buildDorlingNeighborGrid,
  countOverlappingPairs,
  planDeclaredProjectedCartogram
} from '../../../examples/deck/kepler-compute/src/analysis-cartogram-declared-crs';
import {encodeCloudOptimizedGeoTIFF} from '../../../examples/deck/kepler-compute/src/interpolation-cog';

test('join column naming and cardinality are explicit and match the native FULL join example', () => {
  expect(
    planJoinOutputColumns(['key', 'a'], ['key', 'a', 'b'], 'key', {
      suffix: '_right',
      dropRightKey: false
    }).map(column => column.outputName)
  ).toEqual(['key', 'a', 'key_right', 'a_right', 'b']);
  const plan = planJoinCardinality(
    ['b', null, 'a', '', 'b', 'z'],
    ['a', 'b', null, 'a', 'x', 'b'],
    'full'
  );
  expect(plan).toMatchObject({
    cardinality: 'many-to-many',
    outputRowCount: 11,
    matchedPairCount: 6,
    nullKeyLeftRowCount: 1,
    nullKeyRightRowCount: 1
  });
  expect(planJoinCardinality(['a'], ['a', 'a'], 'inner').cardinality).toBe('one-to-many');
  expect(planJoinKeyPrecision('int64', [1n, 2n]).mode).toBe('exact-within-limit');
  expect(planJoinKeyPrecision('int64', [2n ** 40n]).mode).toBe('refuse');
});

test('spatial join boundary, hole and null geometry semantics', () => {
  const square = [
    [
      [0, 0],
      [4, 0],
      [4, 4],
      [0, 4]
    ],
    [
      [1, 1],
      [3, 1],
      [3, 3],
      [1, 3]
    ]
  ] as const;
  expect(classifyPointInPolygon([0.5, 0.5], square)).toBe('inside');
  expect(classifyPointInPolygon([2, 2], square)).toBe('outside');
  expect(classifyPointInPolygon([1, 2], square)).toBe('boundary');
  expect(classifyPointInPolygon([4, 4], square)).toBe('boundary');
  const points = [[0.5, 0.5], [1, 2], null, [9, 9]] as const;
  expect(listSpatialJoinPairs(points, [square, null, square], 'intersects')).toEqual([
    [0, 0],
    [0, 2],
    [1, 0],
    [1, 2]
  ]);
  expect(listSpatialJoinPairs(points, [square], 'within')).toEqual([[0, 0]]);
  expect(planSpatialJoinSemantics('within', 'EPSG:4326', 'EPSG:3857').refusal).toBeDefined();
});

test('group by chunking, hashing, extra measures and compensated sums', () => {
  expect(planGroupByKeyChunks(10000, true).ranges).toEqual([
    [0, 4096],
    [4096, 8192],
    [8192, 10001]
  ]);
  expect(
    planGroupByMeasureBatches(Array.from({length: 40}, (_, i) => i)).map(b => b.length)
  ).toEqual([16, 16, 8]);
  const keys = Array.from({length: 6000}, (_, index) =>
    index % 7 === 0 ? null : `k${index % 5000}`
  );
  const assignment = assignGroupSlots(keys, 16384);
  expect(assignment?.distinctKeys.length).toBe(new Set(keys.filter(Boolean)).size);
  expect(assignGroupSlots(keys, 1024)).toBeNull();
  expect(reduceGroupMeasure('median', [3, 1, null, 2, 10])).toBe(2.5);
  expect(reduceGroupMeasure('countDistinct', [1, 1, 2, null])).toBe(2);
  expect(reduceGroupMeasure('standardDeviation', [2, 4, 4, 4, 5, 5, 7, 9])).toBeCloseTo(2, 12);
  const compensated = new CompensatedFloat32Sum();
  let plain = 0;
  for (let index = 0; index < 100000; index++) {
    compensated.add(0.1);
    plain = Math.fround(plain + Math.fround(0.1));
  }
  expect(Math.abs(compensated.value - 10000)).toBeLessThan(Math.abs(plain - 10000));
});

test('numeric narrowing and rate causes are attributed once each', () => {
  const report = diagnoseFloat32Narrowing('float64', [0.1, 1, null, 1e300, Infinity]);
  expect(report).toMatchObject({
    inexactCount: 1,
    overflowCount: 1,
    nullCount: 1,
    nonFiniteCount: 1,
    verdict: 'refused'
  });
  expect(decideNarrowing(diagnoseFloat32Narrowing('int64', [16777217n]), false).allowed).toBe(
    false
  );
  expect(decideNarrowing(diagnoseFloat32Narrowing('int64', [16777216n]), false).allowed).toBe(true);
  const diagnostics = classifyRateInvalidCauses(
    [1, null, 2, NaN, -1, 3, 4, 5],
    [10, 5, null, 5, 5, -2, 0, 5],
    [1, 1, 1, 1, 1, 1, 1, 0]
  );
  expect(diagnostics.counts).toMatchObject({
    valid: 1,
    nullEvent: 1,
    nullPopulation: 1,
    nonFiniteEvent: 1,
    negativeEvent: 1,
    negativePopulation: 1,
    zeroPopulation: 1,
    masked: 1
  });
});

test('declared CRS overlay limit and neighbor grid agree with brute force', () => {
  const options = {
    metersPerUnit: 0.3048,
    planeOrigin: [0, 0] as const,
    anchor: [-100, 40] as const
  };
  expect(
    planDeclaredProjectedCartogram(options, {
      minimumX: -1000,
      minimumY: -1000,
      maximumX: 1000,
      maximumY: 1000
    }).ok
  ).toBe(true);
  expect(
    planDeclaredProjectedCartogram(options, {minimumX: 0, minimumY: 0, maximumX: 1e7, maximumY: 0})
      .ok
  ).toBe(false);
  let seed = 7;
  const random = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
  const xs = Array.from({length: 3000}, () => random() * 1000);
  const ys = Array.from({length: 3000}, () => random() * 1000);
  const radii = Array.from({length: 3000}, () => 1 + random() * 8);
  let brute = 0;
  for (let a = 0; a < xs.length; a++)
    for (let b = a + 1; b < xs.length; b++)
      if (Math.hypot(xs[a] - xs[b], ys[a] - ys[b]) < radii[a] + radii[b]) brute++;
  expect(countOverlappingPairs(buildDorlingNeighborGrid(xs, ys, radii), xs, ys, radii)).toBe(brute);
});

test('cloud GeoTIFF is tiled, deflated, multiband and carries overview IFDs', async () => {
  const width = 40;
  const height = 24;
  const values = new Float32Array(width * height).map((_, index) => index);
  values[5] = Number.NaN;
  const bytes = await encodeCloudOptimizedGeoTIFF({
    width,
    height,
    bounds: [0, 0, 40, 24],
    bands: [
      {name: 'a', values},
      {name: 'b', values: values.map(value => value * 2)}
    ],
    crs: {kind: 'projected', epsg: 32633},
    tileSize: 16
  });
  const view = new DataView(bytes.buffer);
  expect(view.getUint16(2, true)).toBe(42);
  const readIfd = (offset: number) => {
    const count = view.getUint16(offset, true);
    const tags = new Map<number, {type: number; count: number; value: number}>();
    for (let index = 0; index < count; index++) {
      const entry = offset + 2 + index * 12;
      tags.set(view.getUint16(entry, true), {
        type: view.getUint16(entry + 2, true),
        count: view.getUint32(entry + 4, true),
        value: view.getUint32(entry + 8, true)
      });
    }
    return {tags, next: view.getUint32(offset + 2 + count * 12, true)};
  };
  const full = readIfd(view.getUint32(4, true));
  expect(full.tags.get(256)?.value).toBe(width);
  expect(full.tags.get(277)?.value).toBe(2);
  expect(full.tags.get(259)?.value).toBe(8);
  expect(full.tags.get(324)?.count).toBe(6); // 3 x 2 tiles
  const overview = readIfd(full.next);
  expect(overview.tags.get(254)?.value).toBe(1);
  expect(overview.tags.get(256)?.value).toBe(20);
  expect(readIfd(overview.next).next).toBe(0);
  // First tile of the full image decompresses to 16x16 pixels x 2 bands x 4 bytes.
  const offsets = full.tags.get(324)!;
  const counts = full.tags.get(325)!;
  const tileOffset = view.getUint32(offsets.value, true);
  const tileLength = view.getUint32(counts.value, true);
  const stream = new Blob([bytes.slice(tileOffset, tileOffset + tileLength)])
    .stream()
    .pipeThrough(new DecompressionStream('deflate'));
  const raw = new Float32Array(await new Response(stream).arrayBuffer());
  expect(raw.length).toBe(16 * 16 * 2);
  expect(raw[2]).toBe(1);
  expect(raw[3]).toBe(2);
  expect(Number.isNaN(raw[10])).toBe(true);
});
