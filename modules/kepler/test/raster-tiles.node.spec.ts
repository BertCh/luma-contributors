// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import type {KeplerAnalysisRecipe} from '../src/recipes/analysis-recipe';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {encodeInterpolationGeoTIFF} from '../../../examples/deck/kepler-compute/src/interpolation-geotiff';
import {
  createNumericRasterSource,
  createNumericRasterTile,
  createXyzTileSource,
  getXyzTileExtent
} from '../../../examples/deck/kepler-compute/src/raster-tile-ingest';
import {
  planRasterReprojection,
  transformLngLatToMercator,
  UnsupportedRasterCrsError
} from '../../../examples/deck/kepler-compute/src/raster-reprojection';
import {
  compileBandMath,
  evaluateBandMath,
  KEPLER_BAND_COMBINATION_EXPRESSIONS
} from '../../../examples/deck/kepler-compute/src/raster-band-math';
import {RasterBandResidency} from '../../../examples/deck/kepler-compute/src/raster-band-model';
import {
  decodeRasterPixelReadback,
  identifyRasterValue,
  planRasterPixelReadback
} from '../../../examples/deck/kepler-compute/src/raster-identify';
import {computeRasterZonalStatistics} from '../../../examples/deck/kepler-compute/src/raster-zonal';
import {readNumericGeoTIFF} from '../../../examples/deck/kepler-compute/src/raster-geotiff-reader';
import {
  canBitmapSupplyAnalysisValues,
  createBitmapResourceScope,
  encodeBitmapWorldFile,
  readBitmapNumericInput,
  resolveBitmapGeoreference,
  writeBitmapGeoreference
} from '../../../examples/deck/kepler-compute/src/bitmap-georeference';

const bounds = [-74, 40, -70, 43] as const;
const recipe: KeplerAnalysisRecipe = {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'interpolation-\u2603',
  source: {datasetId: 'samples-\ud83c\udf0d', revision: 3},
  population: 'selected',
  selectionRevision: 7,
  backendPolicy: 'require-gpu',
  operation: {
    kind: 'idw-points',
    column: 'value',
    longitudeColumn: 'longitude',
    latitudeColumn: 'latitude',
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters',
    grid: {width: 2, height: 3, bounds},
    searchRadius: 1000,
    power: 2,
    neighborCount: 4,
    minimumNeighborCount: 2,
    taper: 'none'
  }
};

// Two side-by-side 2x2 EPSG:4326 tiles covering [0,2]x[0,1]; the shared edge is lng 1.
function createMosaic() {
  const left = createNumericRasterTile({
    id: 'left',
    width: 2,
    height: 2,
    extent: [0, 0, 1, 1],
    bands: [
      {values: new Float32Array([1, 2, 3, -9999]), noDataValue: -9999},
      {values: new Float32Array([1, 1, 1, 1])}
    ]
  });
  const right = createNumericRasterTile({
    id: 'right',
    width: 2,
    height: 2,
    extent: [1, 0, 2, 1],
    bands: [
      {values: new Float32Array([5, 6, 7, 8]), noDataValue: -9999},
      {values: new Float32Array([1, 1, 1, 1])}
    ]
  });
  return createNumericRasterSource({
    id: 'mosaic',
    kind: 'single-raster',
    crs: 'EPSG:4326',
    tiles: [left, right]
  });
}

test('ingestion derives validity from nodata and non-finite samples for every band type', () => {
  const tile = createNumericRasterTile({
    id: 't',
    width: 4,
    height: 1,
    extent: [0, 0, 4, 1],
    bands: [
      {values: new Int16Array([1, -1, 3, 4]), noDataValue: -1},
      {values: new Float32Array([NaN, 2, Infinity, 4])},
      {values: new Uint8Array([0, 255, 1, 2]), noDataValue: 255},
      {values: new Float32Array([-3.4028234663852886e38, 1, 2, 3]), noDataValue: -3.4028235e38}
    ]
  });
  expect([...tile.bands[0].validity]).toEqual([1, 0, 1, 1]);
  expect([...tile.bands[1].validity]).toEqual([0, 1, 0, 1]);
  expect([...tile.bands[2].validity]).toEqual([1, 0, 1, 1]);
  expect([...tile.bands[3].validity]).toEqual([0, 1, 1, 1]);
  expect(tile.bands.map(band => band.sampleType)).toEqual(['int16', 'float32', 'uint8', 'float32']);
  expect(() =>
    createNumericRasterTile({
      id: 'bad',
      width: 2,
      height: 2,
      extent: [0, 0, 1, 1],
      bands: [{values: new Float32Array(3)}]
    })
  ).toThrow();
});

test('reprojection refuses unsupported CRSs and round-trips Web Mercator', () => {
  expect(() => planRasterReprojection('EPSG:32633')).toThrow(UnsupportedRasterCrsError);
  const plan = planRasterReprojection('EPSG:3857', 'EPSG:4326');
  const [x, y] = plan.transformTargetToSource(-73.9, 40.7);
  const [lng, lat] = plan.transformSourceToTarget(x, y);
  expect(lng).toBeCloseTo(-73.9, 9);
  expect(lat).toBeCloseTo(40.7, 9);
  expect(transformLngLatToMercator(180, 0)[0]).toBeCloseTo(20037508.342789244, 3);
});

test('identify respects tile edge ownership, nodata, outer boundary and eviction', () => {
  const source = createMosaic();
  // Pixel center of left tile, row 0, column 0.
  expect(identifyRasterValue(source, 0.25, 0.75)).toMatchObject({
    status: 'value',
    tileId: 'left',
    bandValues: [1, 1]
  });
  // lng 1 is shared: the east tile owns it, so the value is column 0 of the right tile.
  expect(identifyRasterValue(source, 1, 0.75)).toMatchObject({
    tileId: 'right',
    column: 0,
    bandValues: [5, 1]
  });
  // nodata sample (left tile, column 1, row 1).
  expect(identifyRasterValue(source, 0.75, 0.25, {bandIndexes: [1]})).toMatchObject({
    status: 'nodata',
    bandValues: [null]
  });
  // Outer east edge is still identified through the closed fallback; beyond it is outside.
  expect(identifyRasterValue(source, 2, 0.75)).toMatchObject({tileId: 'right', column: 1});
  expect(identifyRasterValue(source, 2.01, 0.75).status).toBe('outside');
  expect(
    identifyRasterValue(source, 0.25, 0.75, {isTileAvailable: tile => tile.id !== 'left'}).status
  ).toBe('evicted');
});

test('highest zoom tile wins identify and zonal never double counts parents', () => {
  const source = createXyzTileSource('xyz', [
    {
      index: {z: 0, x: 0, y: 0},
      width: 2,
      height: 2,
      bands: [{values: new Float32Array([1, 1, 1, 1])}]
    },
    {
      index: {z: 1, x: 0, y: 0},
      width: 2,
      height: 2,
      bands: [{values: new Float32Array([10, 20, 30, 40])}]
    }
  ]);
  const [west, , east, north] = getXyzTileExtent({z: 1, x: 0, y: 0});
  expect(east).toBeGreaterThan(west);
  const plan = planRasterReprojection('EPSG:3857', 'EPSG:4326');
  const [lng, lat] = plan.transformSourceToTarget(
    west + (east - west) * 0.25,
    north - (east - west) * 0.25
  );
  expect(identifyRasterValue(source, lng, lat)).toMatchObject({
    tileId: 'xyz/1/0/0',
    bandValues: [10]
  });
  const everything = computeRasterZonalStatistics(source, [
    [
      [
        [-180, 0],
        [0, 0],
        [0, 85],
        [-180, 85],
        [-180, 0]
      ]
    ]
  ]);
  expect(everything.tileCount).toBe(1);
  expect(everything.count).toBe(4);
  expect(everything.sum).toBe(100);
});

test('zonal statistics agree with identify across tile edges and nodata', () => {
  const source = createMosaic();
  const stats = computeRasterZonalStatistics(source, [
    [
      [
        [0.1, 0.1],
        [1.9, 0.1],
        [1.9, 0.9],
        [0.1, 0.9],
        [0.1, 0.1]
      ]
    ]
  ]);
  // Pixel centers: left 1,2,3,(nodata) and right 5,6,7,8.
  expect(stats).toMatchObject({
    count: 7,
    sum: 32,
    minimum: 1,
    maximum: 8,
    nodataCount: 1,
    tileCount: 2,
    isTruncated: false
  });
  expect(stats.mean).toBeCloseTo(32 / 7, 12);
  // A hole removes the right tile's upper-left pixel.
  const holed = computeRasterZonalStatistics(source, [
    [
      [
        [0, 0],
        [2, 0],
        [2, 1],
        [0, 1],
        [0, 0]
      ],
      [
        [1.1, 0.6],
        [1.4, 0.6],
        [1.4, 0.9],
        [1.1, 0.9],
        [1.1, 0.6]
      ]
    ]
  ]);
  expect(holed.count).toBe(6);
  expect(
    computeRasterZonalStatistics(
      source,
      [
        [
          [
            [0, 0],
            [2, 0],
            [2, 1],
            [0, 1],
            [0, 0]
          ]
        ]
      ],
      {maximumPixelCount: 3}
    )
  ).toMatchObject({isTruncated: true, scannedCount: 3});
  expect(
    computeRasterZonalStatistics(source, [
      [
        [
          [50, 50],
          [51, 50],
          [51, 51],
          [50, 51],
          [50, 50]
        ]
      ]
    ])
  ).toMatchObject({count: 0, mean: null});
});

test('band math propagates nodata and matches Kepler normalized difference', () => {
  const source = createMosaic();
  const left = source.tiles[0];
  const ndvi = evaluateBandMath(
    left,
    compileBandMath(KEPLER_BAND_COMBINATION_EXPRESSIONS.normalizedDifference)
  );
  // (1-1)/(1+1)=0, (2-1)/(2+1), nodata in b1 propagates.
  expect([...ndvi.validity]).toEqual([1, 1, 1, 0]);
  expect(ndvi.values[1]).toBeCloseTo(1 / 3, 6);
  expect(compileBandMath('abs(-b1) + min(b1, 2) * 2 / (b2 - b2)').bandIndexes).toEqual([1, 2]);
  const divide = evaluateBandMath(left, compileBandMath('b1 / (b2 - b2)'));
  expect([...divide.validity]).toEqual([0, 0, 0, 0]);
  expect(() => compileBandMath('b1 +')).toThrow();
  expect(() => compileBandMath('b1 ^ 2')).toThrow();
  expect(
    identifyRasterValue(source, 0.75, 0.75, {expression: compileBandMath('b1 * 2')}).derivedValue
  ).toBe(4);
});

test('residency enforces its budget, evicts least recently used bands and releases registry memory', () => {
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1000});
  const source = createMosaic();
  const destroy = vi.fn();
  const upload = vi.fn(() => ({byteLength: 20, hasPrecisionLoss: false, destroy}));
  const residency = new RasterBandResidency({
    budgetByteLength: 40,
    upload,
    memoryRegistry: registry
  });
  const [left, right] = source.tiles;
  residency.ensureBandResident(left, 1);
  residency.ensureBandResident(right, 1);
  residency.ensureBandResident(left, 1); // refresh left, so right is the oldest
  residency.ensureBandResident(left, 2);
  expect(residency.isBandResident('right', 1)).toBe(false);
  expect(residency.isTileResident(left)).toBe(true);
  expect(residency.getStatistics()).toMatchObject({
    residentBandCount: 2,
    residentByteLength: 40,
    evictionCount: 1
  });
  expect(upload).toHaveBeenCalledTimes(3);
  expect(
    identifyRasterValue(source, 1.25, 0.75, {
      isTileAvailable: tile => residency.isTileResident(tile, [1])
    }).status
  ).toBe('evicted');
  residency.destroy();
  expect(destroy).toHaveBeenCalledTimes(3);
  expect(residency.getStatistics().residentBandCount).toBe(0);
  expect(registry.memoryUsage.reservedBytes.gpu).toBe(0);
  expect(() => residency.ensureBandResident(left, 1)).toThrow();
  expect(() =>
    new RasterBandResidency({budgetByteLength: 5, upload}).ensureBandResident(left, 1)
  ).toThrow();
});

test('GPU identify readback is a fixed bounded plan', () => {
  const tile = createMosaic().tiles[0];
  const plan = planRasterPixelReadback(tile, 1, 1, 0);
  expect(plan).toMatchObject({origin: {x: 1, y: 0}, readbackByteLength: 512});
  expect(() => planRasterPixelReadback(tile, 1, 2, 0)).toThrow();
  const staging = new Uint8Array(512);
  new DataView(staging.buffer).setFloat32(0, 2.5, true);
  expect(decodeRasterPixelReadback(staging)).toBeNull();
  staging[256] = 255;
  expect(decodeRasterPixelReadback(staging)).toBe(2.5);
});

test('GeoTIFF reader round-trips the interpolation export and refuses unsupported CRS', () => {
  const values = new Float32Array([1, 2, NaN, 4, 5, 6]);
  const source = readNumericGeoTIFF(
    encodeInterpolationGeoTIFF({width: 2, height: 3, bounds, values, recipe})
  );
  const [tile] = source.tiles;
  expect(source.crs).toBe('EPSG:4326');
  expect(tile.extent).toEqual([-74, 40, -70, 43]);
  expect([...tile.bands[0].validity]).toEqual([1, 1, 0, 1, 1, 1]);
  expect(identifyRasterValue(source, -73, 42.5)).toMatchObject({status: 'value', bandValues: [1]});
  expect(identifyRasterValue(source, -73, 41.5).status).toBe('nodata');
  const bytes = encodeInterpolationGeoTIFF({width: 2, height: 3, bounds, values, recipe});
  expect(() => readNumericGeoTIFF(bytes, 'x', 10)).toThrow();
  expect(() => readNumericGeoTIFF(new Uint8Array(16))).toThrow();
});

test('bitmap georeferencing, numeric input, export and cleanup', () => {
  const georeference = resolveBitmapGeoreference({
    boundsWest: -70,
    boundsSouth: 43,
    boundsEast: -74,
    boundsNorth: 40,
    opacity: 2
  });
  expect(georeference).toMatchObject({bounds: [-74, 40, -70, 43], opacity: 1, isRotated: false});
  expect(
    resolveBitmapGeoreference([
      [-74, 40],
      [-74, 43],
      [-70, 43],
      [-70, 40]
    ]).isRotated
  ).toBe(false);
  expect(
    resolveBitmapGeoreference([
      [-74, 40],
      [-73, 43],
      [-70, 43],
      [-70, 40]
    ]).isRotated
  ).toBe(true);
  expect(() => resolveBitmapGeoreference({boundsWest: 1})).toThrow();
  expect(() => resolveBitmapGeoreference([10, 0, 5, 1])).not.toThrow();
  expect(() => resolveBitmapGeoreference([0, 0, 0, 1])).toThrow();
  expect(writeBitmapGeoreference({boundsWest: 0, other: 1} as never, georeference)).toMatchObject({
    boundsWest: -74,
    boundsNorth: 43,
    other: 1
  });
  expect(encodeBitmapWorldFile([-74, 40, -70, 43], 4, 3).split('\n')).toEqual([
    '1',
    '0',
    '0',
    '-1',
    '-73.5',
    '42.5'
  ]);
  const numeric = readBitmapNumericInput(
    encodeInterpolationGeoTIFF({width: 2, height: 3, bounds, values: new Float32Array(6), recipe}),
    'g',
    0.5
  );
  expect(numeric.georeference).toMatchObject({bounds: [-74, 40, -70, 43], opacity: 0.5});
  expect(canBitmapSupplyAnalysisValues(numeric)).toBe(true);
  expect(canBitmapSupplyAnalysisValues(undefined)).toBe(false);
  const scope = createBitmapResourceScope();
  const order: number[] = [];
  scope.track(() => order.push(1));
  scope.track(() => order.push(2));
  expect(scope.destroy()).toBe(2);
  expect(order).toEqual([2, 1]);
  expect(scope.destroy()).toBe(0);
});
