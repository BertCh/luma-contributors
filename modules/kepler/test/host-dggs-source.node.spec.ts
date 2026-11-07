// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {lonLatToCell, u64ToHex} from 'a5-js';
import {expect, test} from 'vitest';
import {
  appendDGGSFootprintColumn,
  decodeA5Footprint,
  decodeGeohashFootprint,
  decodeS2Footprint,
  readS2CellId
} from '../../../examples/deck/kepler-compute/src/dggs-source';

test('geohash footprints match mixed precision bounds', () => {
  const ring = decodeGeohashFootprint('"U4PRUYDT"') as Float64Array;
  expect(ring.length).toBe(10);
  expect(Array.from(ring.slice(0, 2))).toEqual(Array.from(ring.slice(8, 10)));
  const coarse = decodeGeohashFootprint('s') as Float64Array;
  expect([coarse[0], coarse[1], coarse[2], coarse[5]]).toEqual([0, 0, 45, 45]);
  expect(typeof decodeGeohashFootprint('a')).toBe('string');
  expect(typeof decodeGeohashFootprint('')).toBe('string');
});

test('S2 IDs decode exactly from tokens and Uint64 and reject invalid markers', () => {
  expect(readS2CellId('89c25')).toBe(0x89c2500000000000n);
  expect(readS2CellId(0x89c2500000000000n)).toBe(0x89c2500000000000n);
  expect(typeof readS2CellId('0')).toBe('string');
  expect(typeof readS2CellId('zz')).toBe('string');
  // Face 7 and a marker bit on an odd position are not cells.
  expect(typeof readS2CellId(0xe000000000000000n)).toBe('string');
  expect(typeof readS2CellId(0x2n)).toBe('string');
  const ring = decodeS2Footprint('89c25') as Float64Array;
  const bigintRing = decodeS2Footprint(0x89c2500000000000n) as Float64Array;
  expect(Array.from(ring)).toEqual(Array.from(bigintRing));
  expect(ring.length).toBe(10);
  // Face cells subdivide their curved edges.
  expect((decodeS2Footprint('1') as Float64Array).length).toBe(2 * (4 * 100 + 1));
});

test('A5 footprints decode Uint64 cells without number rounding', () => {
  const cell = lonLatToCell([-122.4, 37.8], 10);
  expect(cell > 2n ** 53n).toBe(true);
  const fromBigint = decodeA5Footprint(cell) as Float64Array;
  const fromHex = decodeA5Footprint(u64ToHex(cell)) as Float64Array;
  expect(fromBigint.length).toBeGreaterThanOrEqual(12);
  expect(Array.from(fromHex)).toEqual(Array.from(fromBigint));
  expect(fromBigint[0]).toBe(fromBigint[fromBigint.length - 2]);
  expect(typeof decodeA5Footprint('nothex')).toBe('string');
});

test('footprint columns preserve batches, empty batches, null rows and source IDs', () => {
  const first = new arrow.Table({id: arrow.vectorFromArray(['s', null, 'zzz!'], new arrow.Utf8())});
  const second = new arrow.Table({id: arrow.vectorFromArray(['u4pruydt'], new arrow.Utf8())});
  const batches = [first.batches[0], first.slice(0, 0).batches[0], second.batches[0]];
  const table = new arrow.Table(first.schema, batches);
  const result = appendDGGSFootprintColumn(table, 'id', 'geohash');
  expect(result.table.batches.map(batch => batch.numRows)).toEqual([3, 0, 1]);
  expect(result.summary).toMatchObject({validRowCount: 2, nullRowCount: 1});
  expect(result.summary.invalidRows).toHaveLength(1);
  const footprint = result.table.getChild('dggs_footprint')!;
  expect(footprint.get(1)).toBeNull();
  expect(footprint.get(2)).toBeNull();
  expect((footprint.get(0) as Uint8Array).byteLength).toBe(13 + 5 * 16);
  expect(result.table.schema.fields[1].metadata.get('ARROW:extension:name')).toBe('geoarrow.wkb');
  expect(Array.from(result.table.getChild('id')!)).toEqual(['s', null, 'zzz!', 'u4pruydt']);
  const roundTrip = arrow.tableFromIPC(arrow.tableToIPC(result.table, 'stream'));
  expect(roundTrip.batches.map(batch => batch.numRows)).toEqual([3, 0, 1]);
  expect(roundTrip.schema.fields[1].metadata.get('ARROW:extension:name')).toBe('geoarrow.wkb');
});

test('Uint64 cell columns keep exact identities and numbers are rejected', () => {
  const exact = 0x89c2500000000001n;
  const wide = new arrow.Table({
    id: arrow.vectorFromArray([0x89c2500000000002n, exact], new arrow.Uint64())
  });
  expect(wide.getChild('id')!.get(1)).toBe(exact);
  const result = appendDGGSFootprintColumn(wide, 'id', 's2');
  expect(result.summary.validRowCount).toBe(1);
  expect(result.summary.invalidRows).toHaveLength(1);
  const numbers = new arrow.Table({id: arrow.vectorFromArray([1, 2], new arrow.Float64())});
  expect(() => appendDGGSFootprintColumn(numbers, 'id', 'a5')).toThrow(/precision/);
});

test('S2 footprints contain the library cell center for every face and mixed levels', async () => {
  const {S2} = (await import('s2-geometry')) as unknown as {
    S2: {idToLatLng(id: string): {lat: number; lng: number}};
  };
  const identifiers = [0n, 1n, 2n, 3n, 4n, 5n].flatMap(face =>
    [
      [3n, 37n],
      [12n, 1234567n]
    ].map(
      ([level, digits]) =>
        (face << 61n) | (digits << (61n - 2n * level)) | (1n << (60n - 2n * level))
    )
  );
  for (const identifier of identifiers) {
    const ring = decodeS2Footprint(identifier) as Float64Array;
    const center = S2.idToLatLng(identifier.toString());
    let minimumLongitude = Infinity;
    let maximumLongitude = -Infinity;
    let minimumLatitude = Infinity;
    let maximumLatitude = -Infinity;
    for (let index = 0; index < ring.length; index += 2) {
      minimumLongitude = Math.min(minimumLongitude, ring[index]);
      maximumLongitude = Math.max(maximumLongitude, ring[index]);
      minimumLatitude = Math.min(minimumLatitude, ring[index + 1]);
      maximumLatitude = Math.max(maximumLatitude, ring[index + 1]);
    }
    const longitude = center.lng < minimumLongitude ? center.lng + 360 : center.lng;
    expect(longitude).toBeGreaterThanOrEqual(minimumLongitude - 1e-9);
    expect(longitude).toBeLessThanOrEqual(maximumLongitude + 1e-9);
    expect(center.lat).toBeGreaterThanOrEqual(minimumLatitude - 1e-9);
    expect(center.lat).toBeLessThanOrEqual(maximumLatitude + 1e-9);
  }
});
