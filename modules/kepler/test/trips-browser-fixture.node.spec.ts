// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {runInNewContext} from 'node:vm';
import {expect, test} from 'vitest';
import {WebMercatorViewport} from '@deck.gl/core';
import {getGeographicViewportBounds} from '@luma-contributors/analysis/geospatial-analysis';
import {prepareKeplerGeoJSONTripsFromArrow} from '../../../examples/deck/kepler-compute/src/geometry-preparation/geojson-trips';
import {prepareKeplerTabularTrips} from '../../../examples/deck/kepler-compute/src/geometry-preparation/tabular-trips';
import {
  makeTimedPath,
  makeMultipartTimedPath,
  MULTIPART_PICK_POSITION,
  MULTIPART_PICK_TIME_MILLISECONDS,
  TRIP_FIXTURE_ORIGIN_MILLISECONDS
} from '../../../examples/deck/kepler-compute/scripts/trips-fixture.mjs';

function expectInsideFit(
  bounds: [number, number, number, number],
  positions: readonly number[][]
): void {
  const fitted = new WebMercatorViewport({width: 1100, height: 1000}).fitBounds([
    [bounds[0], bounds[1]],
    [bounds[2], bounds[3]]
  ]);
  // Native Kepler's no-padding fit centers latitude arithmetically. Exercise that center too.
  const native = new WebMercatorViewport({
    width: 1100,
    height: 1000,
    longitude: (bounds[0] + bounds[2]) / 2,
    latitude: (bounds[1] + bounds[3]) / 2,
    zoom: fitted.zoom,
    pitch: 0
  });
  for (const position of positions) {
    const [horizontal, vertical] = native.project([position[0], position[1], position[2] ?? 0]);
    expect(horizontal).toBeGreaterThan(0);
    expect(horizontal).toBeLessThan(native.width);
    expect(vertical).toBeGreaterThan(0);
    expect(vertical).toBeLessThan(native.height);
  }
}

test('published Arrow trip bounds include all multipart vertices and the original ground probe', () => {
  const geometry = makeMultipartTimedPath(40.77);
  const plain = makeTimedPath(40.74);
  const source = arrow
    .tableFromArrays({route: [null, JSON.stringify(plain)]})
    .concat(arrow.tableFromArrays({route: [JSON.stringify(geometry)]}));
  const prepared = prepareKeplerGeoJSONTripsFromArrow(source.getChild('route')!, {
    timestampEncoding: 'unix-millisecond'
  });
  expect(prepared.batches.map(batch => [...batch.sourceRows])).toEqual([[1], [2, 2]]);
  expect([...prepared.batches[1].valueOffsets]).toEqual([0, 3, 5]);
  expect([...prepared.batches[1].timestamps].slice(3)).toEqual([0, 20000]);
  // The previous tight uploaded bounds had north=40.779998779296875, south of the source.
  expect(
    Math.max(...prepared.batches[1].positions.filter((_, index) => index % 3 === 1))
  ).toBeLessThan(MULTIPART_PICK_POSITION[1]);
  const bounds = getGeographicViewportBounds(prepared.batches.map(batch => batch.positions))!;
  expect(bounds[0]).toBeLessThan(-74.03);
  expect(bounds[1]).toBeLessThan(40.74);
  expect(bounds[2]).toBeGreaterThan(-73.97);
  expect(bounds[3]).toBeGreaterThan(40.78);
  expectInsideFit(bounds, [
    ...plain.coordinates,
    ...geometry.coordinates.flat(),
    MULTIPART_PICK_POSITION
  ]);
});

test('browser multipart ground probe lies on the second part inside its animation trail', () => {
  const [start, end] = makeMultipartTimedPath(40.77).coordinates[1];
  const fraction = (MULTIPART_PICK_POSITION[0] - start[0]) / (end[0] - start[0]);
  expect(MULTIPART_PICK_POSITION).toEqual([-74.01, 40.78]);
  expect(fraction).toBeCloseTo(1 / 3, 10);
  expect(MULTIPART_PICK_POSITION[1]).toBeCloseTo(start[1] + fraction * (end[1] - start[1]), 10);
  expect(MULTIPART_PICK_TIME_MILLISECONDS - TRIP_FIXTURE_ORIGIN_MILLISECONDS).toBeCloseTo(
    20000 / 3,
    3
  );
  expect(MULTIPART_PICK_TIME_MILLISECONDS).toBeLessThan(TRIP_FIXTURE_ORIGIN_MILLISECONDS + 15000);
});

test('tabular cross-batch part bounds keep the grouped script northern leg inside the fit', () => {
  const prepared = prepareKeplerTabularTrips(
    [
      {
        id: 'first',
        tripIds: ['A', 'A', 'B'],
        partIds: ['leg-1', 'leg-1', 'leg-1'],
        timestamps: [0, 10000, 0],
        longitudes: [-74.03, -74, -74.03],
        latitudes: [40.74, 40.74, 40.77]
      },
      {
        id: 'second',
        tripIds: ['A', 'B', 'B', 'A', 'A'],
        partIds: ['leg-1', 'leg-1', 'leg-1', 'leg-2', 'leg-2'],
        timestamps: [20000, 10000, 20000, 0, 20000],
        longitudes: [-73.97, -74, -73.97, -74.03, -73.97],
        latitudes: [40.74, 40.77, 40.77, 40.8, 40.8]
      }
    ],
    {timestampEncoding: 'unix-millisecond'}
  );
  expect(prepared.tripCount).toBe(3);
  const bounds = getGeographicViewportBounds(prepared.batches.map(batch => batch.positions))!;
  expect(bounds[1]).toBeLessThan(40.74);
  expect(bounds[3]).toBeGreaterThan(40.8);
  expectInsideFit(bounds, [
    [-74.015, 40.74],
    [-74.015, 40.77],
    [-74.015, 40.8]
  ]);
});

test('pinned native fitting accepts padded trip bounds at the antimeridian and global limits', () => {
  const require = createRequire(import.meta.url);
  const source = readFileSync(
    join(dirname(require.resolve('@kepler.gl/utils')), '../esm/projection-utils.js'),
    'utf8'
  );
  // Execute the pinned pure functions without bootstrapping Kepler's browser-only aggregate.
  const functions = source
    .split('export function normalizeEvent')[0]
    .replace(/^import .*;$/gm, '')
    .replace(/export /g, '');
  const native = runInNewContext(`${functions}; ({validateBounds,getCenterAndZoomFromBounds})`, {
    geoViewport: require('@mapbox/geo-viewport'),
    WebMercatorViewport,
    Console: {warn() {}}
  }) as {
    validateBounds(bounds: number[]): number[] | null;
    getCenterAndZoomFromBounds(
      bounds: number[],
      dimensions: {width: number; height: number}
    ): {zoom: number; center: number[]} | null;
  };
  expect(native.validateBounds([-196.9, 9.95, 196.9, 11.05])).toBeNull();
  for (const positions of [
    [-179, 10, 179, 11],
    [180, 10, 180, 11],
    [-180, 10, -180, 11],
    [-180, -90, 180, 90]
  ]) {
    const bounds = getGeographicViewportBounds([Float32Array.from(positions)], 2)!;
    expect(native.validateBounds(bounds)).toEqual(bounds);
    const fit = native.getCenterAndZoomFromBounds(bounds, {width: 1100, height: 1000});
    expect(fit).not.toBeNull();
    expect(Number.isFinite(fit!.zoom)).toBe(true);
    expect(fit!.center.every(Number.isFinite)).toBe(true);
  }
});
