// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {runInNewContext} from 'node:vm';
import {expect, test, vi} from 'vitest';
import {WebMercatorViewport} from '@deck.gl/core';
import {getGeographicViewportBounds} from '@luma-contributors/analysis/geospatial-analysis';
import {getGeographicViewportFit} from '@luma-contributors/analysis/geospatial-analysis';
import {getPublishedTripViewportBounds} from '../../../examples/deck/kepler-compute/src/source-publication';
import {buildViewerLayers} from '../../../examples/deck/kepler-compute/src/standalone-viewer-layers';
import {getViewerInitialViewState} from '../../../examples/deck/kepler-compute/src/standalone-viewer-initial-view';
import {readViewerConfig} from '../../../examples/deck/kepler-compute/src/standalone-viewer-config';
import {createViewerDataset} from '../../../examples/deck/kepler-compute/src/standalone-viewer-data';
import {getViewStateForBounds} from '../../../examples/deck/kepler-compute/src/standalone-viewer-model';
import {latLngToCell, cellToBoundary} from 'h3-js';

function buildTripViewer(
  coordinates: number[][],
  extra: {discardedPoint?: boolean; h3Cell?: string} = {}
) {
  const geometries = [
    {type: 'LineString', coordinates},
    ...(extra.discardedPoint ? [{type: 'Point', coordinates: [0, 5]}] : [])
  ];
  const dataset = createViewerDataset(
    'trips',
    arrow.tableFromArrays({
      geometry: geometries.map(geometry => JSON.stringify(geometry)),
      hex_id: geometries.map(() => extra.h3Cell ?? null)
    })
  );
  const config = readViewerConfig({
    config: {
      visState: {
        layers: [
          {
            id: 'trip',
            type: 'trip',
            config: {dataId: 'trips', columns: {geojson: {value: 'geometry'}}, visConfig: {}}
          },
          ...(extra.h3Cell
            ? [
                {
                  id: 'cells',
                  type: 'hexagonId',
                  config: {dataId: 'trips', columns: {hex_id: {value: 'hex_id'}}, visConfig: {}}
                }
              ]
            : [])
        ]
      }
    }
  });
  const result = buildViewerLayers(config.layers, {
    datasets: new Map([['trips', dataset]]),
    filters: [],
    currentTime: 500,
    parseJson: (_dataset, _field, row) => geometries[row]
  });
  return {result, trip: result.layers.find(layer => layer.id === 'trip')!};
}

function expectProjectedInside(
  view: {longitude: number; latitude: number; zoom: number},
  positions: number[][]
) {
  const viewport = new WebMercatorViewport({...view, width: 1200, height: 800});
  for (const position of positions) {
    const pixel = viewport.project(position);
    expect(pixel.every(Number.isFinite)).toBe(true);
    expect(pixel[0]).toBeGreaterThanOrEqual(0);
    expect(pixel[0]).toBeLessThanOrEqual(1200);
    expect(pixel[1]).toBeGreaterThanOrEqual(0);
    expect(pixel[1]).toBeLessThanOrEqual(800);
  }
}

test('rendered crossing Trips interpolate locally and retain vertex timestamps (B2)', () => {
  const {result, trip} = buildTripViewer([
    [178.5, 5, 0, 50],
    [-178.5, 5, 0, 150]
  ]);
  const path = trip.props['getPath'](trip.props['data'][0]) as number[][];
  // Shortest arc is 3 degrees; its half-time longitude is the antimeridian, not Greenwich.
  expect(path[1][0] - path[0][0]).toBe(3);
  expect(trip.props['getTimestamps'](trip.props['data'][0])).toEqual([50, 150]);
  const midpoint = path[0].map((value, index) => (value + path[1][index]) / 2);
  expectProjectedInside(getViewerInitialViewState(null, result, 1200, 800), [midpoint]);
});

test('wrapped high-latitude Trips fit in Mercator and ordinary Trips keep the exact old camera (B3)', () => {
  const coordinates = [
    [179, 75, 0, 0],
    [-179, 80, 0, 1000]
  ];
  const {result} = buildTripViewer(coordinates);
  const view = getViewerInitialViewState(null, result, 1200, 800);
  const aligned = coordinates.map(([longitude, latitude]) => [
    longitude + 360 * Math.round((view.longitude - longitude) / 360),
    latitude
  ]);
  expectProjectedInside(view, aligned);
  const ordinary = buildTripViewer([
    [10, 20, 0, 0],
    [12, 22, 0, 1000]
  ]).result;
  expect(getViewerInitialViewState(null, ordinary, 1200, 800)).toEqual(
    getViewStateForBounds([10, 20, 12, 22], 1200, 800)
  );
});

test('Trips fitting excludes discarded Point rows and includes drawable H3 boundaries (B4)', () => {
  const coordinates = [
    [178.5, 5, 0, 0],
    [-178.5, 5, 0, 1000]
  ];
  const clean = buildTripViewer(coordinates).result;
  const withPoint = buildTripViewer(coordinates, {discardedPoint: true});
  expect(withPoint.trip.props['data']).toHaveLength(1);
  expect(getViewerInitialViewState(null, withPoint.result, 1200, 800)).toEqual(
    getViewerInitialViewState(null, clean, 1200, 800)
  );
  const cell = latLngToCell(1, 1, 6);
  const mixed = buildTripViewer(coordinates, {h3Cell: cell}).result;
  expect(mixed.layers).toHaveLength(2);
  const view = getViewerInitialViewState(null, mixed, 1200, 800);
  const boundary = cellToBoundary(cell).map(([latitude, longitude]) => [
    longitude + 360 * Math.round((view.longitude - longitude) / 360),
    latitude
  ]);
  expectProjectedInside(view, boundary);
});

test('the actual standalone Trips initial camera uses a compact dateline interval and honors a saved camera', () => {
  const path = {
    type: 'LineString',
    coordinates: [
      [179, 5, 0, 0],
      [-179, 5, 0, 1000]
    ]
  };
  const dataset = createViewerDataset(
    'trips',
    arrow.tableFromArrays({geometry: [JSON.stringify(path)]})
  );
  const config = readViewerConfig({
    config: {
      visState: {
        layers: [
          {
            id: 'trip',
            type: 'trip',
            config: {dataId: 'trips', columns: {geojson: {value: 'geometry'}}, visConfig: {}}
          }
        ]
      }
    }
  });
  const result = buildViewerLayers(config.layers, {
    datasets: new Map([['trips', dataset]]),
    filters: [],
    currentTime: 0,
    parseJson: () => path
  });
  expect(result.layers).toHaveLength(1);
  const view = getViewerInitialViewState(null, result, 1200, 800);
  expect(Number.isFinite(view.zoom)).toBe(true);
  expect(view.longitude).toBeCloseTo(-180, 6);
  expect(view.zoom).toBeGreaterThan(7);
  expect(
    getViewerInitialViewState(
      {longitude: 23, latitude: 45, zoom: 4, pitch: 10, bearing: 20},
      result,
      1200,
      800
    )
  ).toEqual({longitude: 23, latitude: 45, zoom: 4, pitch: 10, bearing: 20});
  expect(getViewerInitialViewState(null, {bounds: [10, 20, 12, 22]}, 1200, 800)).toEqual({
    longitude: 11,
    latitude: 21,
    zoom: Math.log2(800 / 512 / (2 / 180))
  });
  path.coordinates = [
    [10, 20, 0, 0],
    [12, 22, 0, 1000]
  ];
  const ordinary = buildViewerLayers(config.layers, {
    datasets: new Map([['trips', dataset]]),
    filters: [],
    currentTime: 0,
    parseJson: () => path
  });
  const ordinaryView = getViewerInitialViewState(null, ordinary, 1200, 800);
  expect(ordinaryView.longitude).toBe(11);
  expect(ordinaryView.latitude).toBe(21);
  expect(Number.isFinite(ordinaryView.zoom)).toBe(true);
  expect(ordinaryView.zoom).toBeGreaterThan(6);
});

test('trip XYZ fitting ignores altitude and preserves canonical ordered publication', () => {
  const positions = Float32Array.of(179, 5, 800, 181, 5, 900);
  const compact = getGeographicViewportFit([positions], 3)!;
  expect(Number.isFinite(compact.longitudeSpanDegrees)).toBe(true);
  // Two degrees separation, with the documented 5% margin on each side.
  expect(compact.longitudeSpanDegrees).toBeCloseTo(2.2, 5);
  const ordered = getPublishedTripViewportBounds([positions])!;
  expect(ordered.every(Number.isFinite)).toBe(true);
  expect(ordered[0]).toBeLessThanOrEqual(-179);
  expect(ordered[2]).toBeGreaterThanOrEqual(179);
  expect(ordered[0]).toBeLessThanOrEqual(ordered[2]);
  const ordinary = getGeographicViewportBounds([Float32Array.of(10, 20, 800, 12, 22, 900)], 3)!;
  expect(ordinary).toEqual([9.9, 19.9, 12.1, 22.1]);
  expect(getPublishedTripViewportBounds([])).toBeNull();
});
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
  const bounds = getGeographicViewportBounds(
    prepared.batches.map(batch => batch.positions),
    3
  )!;
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
  const bounds = getGeographicViewportBounds(
    prepared.batches.map(batch => batch.positions),
    3
  )!;
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

test('ordered trip publication preserves +180 endpoint and original branch rounding allowance', () => {
  const endpoint = getPublishedTripViewportBounds([Float32Array.of(178, 5, 0, 180, 5, 0)])!;
  expect(endpoint).toEqual([177.9, 5 - 5 * 2 ** -23, 180, 5 + 5 * 2 ** -23]);
  const continuous = getPublishedTripViewportBounds([Float32Array.of(360.00002, 5, 0)])!;
  // Uploaded center is 360.000030517578125; allowance is source magnitude * 2^-23.
  expect(continuous).toEqual([
    -0.000012397769751260057, 4.999999403953552, 0.00007343292600126006, 5.000000596046448
  ]);
  expect(continuous[0]).toBeLessThanOrEqual(0.00002);
  expect(continuous[2]).toBeGreaterThanOrEqual(0.00002);
  expect(endpoint[0]).toBeLessThanOrEqual(178);
  expect(endpoint[2]).toBeGreaterThanOrEqual(180);
});
test('compact dateline fit drives a capable WebMercator host', () => {
  const fit = getGeographicViewportFit([Float32Array.of(179, 5, 0, -179, 5, 0)], 3)!;
  const viewport = new WebMercatorViewport({width: 1200, height: 800}).fitBounds([
    [fit.longitudeCenterDegrees - fit.longitudeSpanDegrees / 2, fit.southDegrees],
    [fit.longitudeCenterDegrees + fit.longitudeSpanDegrees / 2, fit.northDegrees]
  ]);
  expect(Number.isFinite(viewport.zoom)).toBe(true);
  expect(viewport.longitude).toBeCloseTo(-180, 6);
  expect(viewport.zoom).toBeGreaterThan(7);
});

test('ordered trip publication does not sort longitudes for an unused compact fit', () => {
  const sorting = vi.spyOn(Array.prototype, 'sort').mockImplementation(() => {
    throw new Error('unexpected longitude sort');
  });
  let bounds: [number, number, number, number] | null;
  try {
    bounds = getPublishedTripViewportBounds([Float32Array.of(10, 20, 0, 12, 22, 0)]);
  } finally {
    sorting.mockRestore();
  }
  expect(bounds!).toEqual([9.9, 19.9, 12.1, 22.1]);
});
