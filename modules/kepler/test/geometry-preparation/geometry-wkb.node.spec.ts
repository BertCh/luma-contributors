// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {writeWKB} from '@math.gl/arrow/wkb';
import {prepareKeplerGeometrySourceFromArrow} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/geometry-gpu-vectors';
import {convertGeoArrowVectorToDenseUnion} from '@luma.gl/graph/arrow';
import oracle from './wkb-center-oracle.json';

it('raw WKB representatives match the pinned parser and all-vertex mean oracle, including closure weighting', () => {
  const first = arrow.vectorFromArray(
    oracle.cases.slice(0, 5).map(fixture => Uint8Array.from(fixture.wkb)),
    new arrow.Binary()
  );
  const empty = arrow.vectorFromArray([], new arrow.Binary());
  const last = arrow.vectorFromArray(
    oracle.cases.slice(5).map(fixture => Uint8Array.from(fixture.wkb)),
    new arrow.Binary()
  );
  const prepared = prepareKeplerGeometrySourceFromArrow(
    new arrow.Vector([...first.data, ...empty.data, ...last.data]),
    {encoding: 'geoarrow.wkb'}
  );
  expect(prepared.representativeRecipe).toBe('wkb-vertex-mean');
  expect(prepared.batches.map(batch => batch.rowCount)).toEqual([5, 0, oracle.cases.length - 5]);
  const positions = prepared.batches.flatMap(batch => [...batch.representative!.positions]);
  const validity = prepared.batches.flatMap(batch => [...batch.representative!.validity]);
  for (const [index, fixture] of oracle.cases.entries()) {
    expect(validity[index], fixture.name).toBe(fixture.expected ? 1 : 0);
    expect(positions.slice(index * 2, index * 2 + 2), fixture.name).toEqual(
      fixture.expected?.map(Math.fround) ?? [0, 0]
    );
  }
  expect(prepared.metadata.invalidRows).toHaveLength(1);
  expect(prepared.metadata.invalidRows[0].reason).toContain('GeometryCollection');
  expect(prepared.metadata.hasPointAltitude).toBe(true);
});

it('decoded dense unions retain WKB recipe only with explicit source provenance', () => {
  const fixtures = oracle.cases.filter(fixture => !('unsupported' in fixture));
  const raw = arrow.vectorFromArray(
    [null, ...fixtures.map(fixture => Uint8Array.from(fixture.wkb))],
    new arrow.Binary()
  );
  const union = convertGeoArrowVectorToDenseUnion(raw, {encoding: 'geoarrow.wkb'});
  const unlabeled = prepareKeplerGeometrySourceFromArrow(union, {encoding: 'geoarrow.geometry'});
  expect(unlabeled.representativeRecipe).toBeUndefined();
  expect(unlabeled.representativeUnsupportedReason).toContain('dense union');
  const source = prepareKeplerGeometrySourceFromArrow(union, {
    encoding: 'geoarrow.geometry',
    sourceEncoding: 'geoarrow.wkb'
  });
  expect(source.representativeRecipe).toBe('wkb-vertex-mean');
  expect(source.representativeUnsupportedReason).toBeUndefined();
  expect(source.metadata.invalidRows).toEqual([]);
  const expected = [null, ...fixtures.map(fixture => fixture.expected)];
  expect([...source.batches[0].representative!.validity]).toEqual(
    expected.map(center => (center ? 1 : 0))
  );
  expect([...source.batches[0].representative!.positions]).toEqual(
    expected.flatMap(center => center?.map(Math.fround) ?? [0, 0])
  );
});

it('WKB slices retain null, malformed, empty and measure rows without shifting valid source identities', () => {
  const point = writeWKB({type: 'Point', coordinates: [9, 8]});
  const polygon = Uint8Array.from(
    oracle.cases.find(fixture => fixture.name === 'polygon-open-ring-first-skipped-1')!.wkb
  );
  const first = arrow
    .vectorFromArray([point, null, polygon, new Uint8Array([1, 2])], new arrow.Binary())
    .slice(1);
  const empty = arrow.vectorFromArray([], new arrow.Binary());
  const last = arrow.vectorFromArray(
    [new Uint8Array(), point, writeWKB({type: 'Point', coordinates: [3, 4, 99]}, 'xym')],
    new arrow.Binary()
  );
  const source = prepareKeplerGeometrySourceFromArrow(
    new arrow.Vector([...first.data, ...empty.data, ...last.data])
  );
  expect(source.batches.map(batch => batch.rowCount)).toEqual([3, 0, 3]);
  expect(source.metadata.invalidRows.map(row => row.sourceRowIndex)).toEqual([2, 5]);
  expect(source.metadata.invalidRows[1].reason).toContain('measure');
  expect(source.batches.map(batch => [...batch.representative!.validity])).toEqual([
    [0, 1, 0],
    [],
    [0, 1, 0]
  ]);
  expect(source.batches.map(batch => [...batch.representative!.positions])).toEqual([
    [0, 0, 2.5, 2.5, 0, 0],
    [],
    [0, 0, 9, 8, 0, 0]
  ]);
  expect(source.batches.map(batch => [...batch.points.sourceRows])).toEqual([[], [], [4]]);
});

it('WKB preserves declared spatial references and accepts big endian coordinates without pretending to reproject', () => {
  const point = writeWKB({type: 'Point', coordinates: [9, 8]});
  const withSpatialReference = (srid: number) => {
    const bytes = new Uint8Array(point.byteLength + 4);
    bytes.set(point.subarray(0, 5));
    bytes.set(point.subarray(5), 9);
    const view = new DataView(bytes.buffer);
    view.setUint32(1, 0x20000001, true);
    view.setUint32(5, srid, true);
    return bytes;
  };
  const source = prepareKeplerGeometrySourceFromArrow(
    arrow.vectorFromArray([withSpatialReference(4326)], new arrow.Binary())
  );
  expect(source.metadata.crs).toBe('EPSG:4326');
  expect(source.metadata.spatialReferences).toEqual([{sourceRowIndex: 0, srids: [4326]}]);
  const mixed = prepareKeplerGeometrySourceFromArrow(
    arrow.vectorFromArray(
      [withSpatialReference(4326), withSpatialReference(3857)],
      new arrow.Binary()
    )
  );
  expect(mixed.metadata.crs).toEqual({kind: 'mixed-ewkb-srids', srids: [4326, 3857]});
  const bytes = new Uint8Array(21);
  const view = new DataView(bytes.buffer);
  view.setUint32(1, 1, false);
  view.setFloat64(5, 9, false);
  view.setFloat64(13, 8, false);
  expect([
    ...prepareKeplerGeometrySourceFromArrow(arrow.vectorFromArray([bytes], new arrow.Binary()))
      .batches[0].representative!.positions
  ]).toEqual([9, 8]);
});

it('WKB-derived union slices preserve physical batches and diagnose measure children without dropping valid siblings', () => {
  const point = writeWKB({type: 'Point', coordinates: [9, 8]});
  const polygon = Uint8Array.from(
    oracle.cases.find(fixture => fixture.name === 'polygon-open-ring-first-skipped-1')!.wkb
  );
  const first = arrow.vectorFromArray([point, polygon, null], new arrow.Binary());
  const second = arrow.vectorFromArray(
    [point, writeWKB({type: 'Point', coordinates: [1, 2, 99]}, 'xym')],
    new arrow.Binary()
  );
  const union = convertGeoArrowVectorToDenseUnion(
    new arrow.Vector([...first.data, ...second.data]),
    {encoding: 'geoarrow.wkb'}
  ).slice(1);
  const source = prepareKeplerGeometrySourceFromArrow(union, {
    encoding: 'geoarrow.geometry',
    sourceEncoding: 'geoarrow.wkb'
  });
  expect(source.batches.map(batch => batch.rowCount)).toEqual([2, 2]);
  expect(source.metadata.invalidRows.map(row => row.sourceRowIndex)).toEqual([3]);
  expect(source.metadata.invalidRows[0].reason).toContain('measure');
  expect(source.batches.map(batch => [...batch.representative!.validity])).toEqual([
    [1, 0],
    [1, 0]
  ]);
  expect(source.batches.map(batch => [...batch.representative!.positions])).toEqual([
    [2.5, 2.5, 0, 0],
    [9, 8, 0, 0]
  ]);
  expect(source.batches.map(batch => [...batch.points.sourceRows])).toEqual([[], [2]]);
});
