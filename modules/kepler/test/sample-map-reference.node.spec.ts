// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync, existsSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, resolve} from 'node:path';
import {runInNewContext} from 'node:vm';
import {build} from 'esbuild';
import {beforeAll, expect, test} from 'vitest';
import {assertSamplePresentation} from '../../../examples/deck/kepler-compute/scripts/sample-map-presentation.mjs';
import type {SavedConfigV1} from '@kepler.gl/schemas';
import * as arrow from 'apache-arrow';
import {makeAutoPreparePlan} from '../../../examples/deck/kepler-compute/src/auto-prepare-plan';
import {preparePublishedGeometry} from '../../../examples/deck/kepler-compute/src/source-publication';

const require = createRequire(import.meta.url);
const cacheDirectory = resolve('examples/deck/kepler-compute/scripts/kepler-demo-cache/samples');
const SAMPLE_IDS = [
  'earthquakes',
  'nyctrips',
  'ukcommute',
  'la_assessorparcels',
  'nyc_earnings',
  'mvt_population',
  'lake_lucerne_raster',
  'wms_conus_cref_qcd',
  'sftrees',
  'sfcontour',
  'nyc_census',
  'county_unemployment',
  'movement_pittsburgh',
  'world_flights',
  'fsq_places'
];
let runFixture: (serialized: string) => Promise<string>;

beforeAll(async () => {
  // Kepler declares deck/math 9/4 dependencies; the repo's default Node aliases target the
  // deck/math 10/5 app. Bundle stock public entry points into their own VM to exercise actual
  // Kepler reducers/classes without that browser bootstrap or contaminating other Node tests.
  const stockDirectory = dirname(require.resolve('@kepler.gl/layers'));
  const output = await build({
    entryPoints: [resolve('modules/kepler/test/fixtures/sample-map-reference-runner.ts')],
    bundle: true,
    platform: 'node',
    format: 'cjs',
    write: false,
    logLevel: 'silent',
    plugins: [
      {
        name: 'stock-kepler-public-dependencies',
        setup(builder) {
          builder.onResolve({filter: /^@(deck|math)\.gl\/[^/]+$/}, resolution => {
            return {path: require.resolve(resolution.path, {paths: [stockDirectory]})};
          });
          builder.onResolve({filter: /^apache-arrow$/}, () => ({
            path: require.resolve('apache-arrow', {
              paths: [dirname(require.resolve('@kepler.gl/utils'))]
            })
          }));
        }
      }
    ]
  });
  const module = {exports: {} as {runReferenceFixture(serialized: string): Promise<string>}};
  runInNewContext(output.outputFiles[0]!.text, {
    module,
    exports: module.exports,
    require,
    console,
    process,
    Buffer,
    TextEncoder,
    TextDecoder,
    URL,
    URLSearchParams,
    Response,
    Headers,
    AbortController,
    AbortSignal,
    structuredClone,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    async fetch(input: string | URL | Request) {
      const url = String(input);
      // Only the stock comparison store runs tasks. Replace its basemap-style I/O;
      // reference construction must not fetch sample data or any remote tile service.
      if (/mapbox|cartocdn/.test(url))
        return new Response(JSON.stringify({version: 8, sources: {}, layers: []}));
      throw new Error(`Unexpected reference/test service request: ${url}`);
    }
  });
  runFixture = module.exports.runReferenceFixture;
}, 60000);

function readOriginalConfig(sampleId: string, useCache = true): SavedConfigV1 {
  const path = resolve(
    cacheDirectory,
    sampleId,
    sampleId === 'world_flights' ? 'config_soei4h.json' : 'config.json'
  );
  if (useCache && existsSync(path)) return JSON.parse(readFileSync(path, 'utf8')) as SavedConfigV1;
  const type =
    sampleId === 'mvt_population'
      ? 'vectorTile'
      : sampleId === 'lake_lucerne_raster'
        ? 'rasterTile'
        : sampleId === 'wms_conus_cref_qcd'
          ? 'wms'
          : 'point';
  return {
    version: 'v1',
    config: {
      visState: {
        layers: [
          {
            id: 'layer',
            type,
            config: {
              dataId: sampleId,
              isVisible: true,
              color: [12, 34, 56],
              columns: type === 'point' ? {lat: 'latitude', lng: 'longitude', altitude: null} : {},
              visConfig: {}
            },
            visualChannels: {colorField: {name: 'value', type: 'real'}, colorScale: 'linear'}
          }
        ],
        filters: [],
        splitMaps: []
      },
      mapState: {latitude: 1, longitude: 2, zoom: 5}
    }
  } as unknown as SavedConfigV1;
}

function createFixture(sampleId: string, original: SavedConfigV1) {
  // Config-derived stand-ins keep the tests small and independent of dataset redistribution.
  // When the ignored cache is absent the synthetic configs above still cover all three bugs.
  const fields = new Map<string, {name: string; type: string; format?: string}>();
  const remoteType = original.config.visState.layers.find(layer =>
    ['vectorTile', 'rasterTile', 'wms'].includes(layer.type)
  )?.type;
  for (const layer of original.config.visState.layers) {
    for (const [role, name] of Object.entries(layer.config.columns ?? {})) {
      if (typeof name === 'string')
        fields.set(name, {name, type: role === 'geojson' ? 'geojson' : 'real'});
    }
    for (const channel of Object.values(layer.visualChannels ?? {})) {
      if (channel && typeof channel === 'object' && 'name' in channel)
        fields.set(channel.name, {name: channel.name, type: channel.type});
    }
  }
  for (const filter of original.config.visState.filters ?? []) {
    for (const name of Array.isArray(filter.name) ? filter.name : [filter.name])
      fields.set(name, {
        name,
        type: filter.type === 'timeRange' ? 'timestamp' : 'real',
        format: filter.type === 'timeRange' ? 'x' : ''
      });
  }
  const fieldList = [...fields.values()];
  const rows = remoteType
    ? []
    : [0, 1, 2].map(rowIndex => {
        const values = Object.fromEntries(
          fieldList.map(field => {
            const filter = original.config.visState.filters?.find(candidate =>
              (Array.isArray(candidate.name) ? candidate.name : [candidate.name]).includes(
                field.name
              )
            );
            const endpoints = Array.isArray(filter?.value) ? filter.value : undefined;
            const value =
              field.type === 'string'
                ? `category-${rowIndex}`
                : field.type === 'geojson'
                  ? null
                  : endpoints
                    ? endpoints[rowIndex === 0 ? 0 : 1]
                    : rowIndex + 1;
            return [field.name, value];
          })
        );
        const trip = original.config.visState.layers.some(layer => layer.type === 'trip');
        const geometry = trip
          ? {
              type: 'LineString',
              coordinates: [
                [1, 2, 3, 1567555200],
                [2, 3, 4, 1567555260]
              ]
            }
          : {
              type: 'Polygon',
              coordinates: [
                [
                  [1, 2],
                  [2, 2],
                  [2, 3],
                  [1, 2]
                ]
              ]
            };
        return fieldList.map(field =>
          field.type === 'geojson'
            ? {type: 'Feature', properties: values, geometry}
            : values[field.name]
        );
      });
  const datasetIds = new Set(original.config.visState.layers.map(layer => layer.config.dataId));
  const datasets = [...datasetIds].map(id => ({
    info: {
      id,
      label: sampleId,
      ...(remoteType
        ? {
            type:
              remoteType === 'vectorTile'
                ? 'vector-tile'
                : remoteType === 'rasterTile'
                  ? 'raster-tile'
                  : 'wms-tile'
          }
        : {})
    },
    data: {fields: fieldList, rows},
    ...(remoteType
      ? {
          metadata: {
            fields: fieldList,
            bounds: [-180, -85, 180, 85],
            pmtilesType: remoteType === 'rasterTile' ? 'raster' : undefined
          }
        }
      : {})
  }));
  return {label: sampleId, original, datasets, remote: Boolean(remoteType)};
}

test.each(
  SAMPLE_IDS
)('stock reference saves %s and matches the normal public reducer load', async sampleId => {
  const original = readOriginalConfig(sampleId);
  const fixture = createFixture(sampleId, original);
  const result = JSON.parse(await runFixture(JSON.stringify(fixture)));
  assertSamplePresentation(result.actual, result.expected, [], sampleId);
  expect(result.expected.config.visState.layers.map((layer: {id: string}) => layer.id)).toEqual(
    original.config.visState.layers.map(layer => layer.id)
  );
  if (sampleId === 'mvt_population') {
    expect(result.expected.config.visState.layers[0].visualChannels.colorField).toEqual(
      original.config.visState.layers[0]!.visualChannels['colorField']
    );
  }
}, 20000);

test.each(
  SAMPLE_IDS
)('inline basemap loads %s and builds its reference without any style fetch', async sampleId => {
  const original = readOriginalConfig(sampleId);
  const result = JSON.parse(
    await runFixture(
      JSON.stringify({
        ...createFixture(sampleId, original),
        inlineBackground: true
      })
    )
  );
  expect(result.requests).toEqual([]);
  expect(result.actual.config.mapStyle.styleType).toBe('compute-background');
  assertSamplePresentation(result.actual, result.expected, [], sampleId);
});

test.each([
  false,
  true
])('Kepler Arrow ingestion recovers multipolygon metadata before publication (use cache: %s)', async useCache => {
  const coordinate = new arrow.FixedSizeList(2, new arrow.Field('xy', new arrow.Float64()));
  const rings = new arrow.List(new arrow.Field('vertices', coordinate));
  const polygons = new arrow.List(new arrow.Field('rings', rings));
  const multipolygons = new arrow.List(new arrow.Field('polygons', polygons));
  const standIn = new arrow.Table({
    C000_02: arrow.vectorFromArray([1], new arrow.Int32()),
    geometry: arrow.vectorFromArray(
      [
        [
          [
            [
              [-73, 41],
              [-72, 41],
              [-72, 42],
              [-73, 41]
            ]
          ]
        ]
      ],
      multipolygons
    )
  });
  standIn.schema.fields
    .find(field => field.name === 'geometry')!
    .metadata.set('ARROW:extension:name', 'geoarrow.multipolygon');
  const cached = resolve(cacheDirectory, 'nyc_earnings/data.arrow');
  const table =
    useCache && existsSync(cached) ? arrow.tableFromIPC(readFileSync(cached)).slice(0, 3) : standIn;
  const original = readOriginalConfig('nyc_earnings', false);
  const layer = original.config.visState.layers[0]!;
  layer.type = 'geojson';
  layer.config.columns = {geojson: 'geometry'} as never;
  layer.visualChannels = {
    colorField: {name: 'C000_02', type: 'integer'},
    colorScale: 'quantile'
  } as never;
  const result = JSON.parse(
    await runFixture(
      JSON.stringify({
        ...createFixture('nyc_earnings', original),
        inlineBackground: true,
        arrowIpc: Array.from(arrow.tableToIPC(table))
      })
    )
  );
  expect(result.fieldEncoding).toBe('geoarrow.multipolygon');
  expect(result.containerEncoding).toBeUndefined();
  expect(result.sameBuffers).toBe(true);
  expect(result.requests).toEqual([]);
  const source = arrow.tableFromIPC(Uint8Array.from(result.restoredIpc));
  assertSamplePresentation(result.actual, result.expected, [], 'nyc_earnings native Arrow');
  const plan = makeAutoPreparePlan(source, [
    {id: 'hkc2hq', type: 'geojson', isVisible: true, columns: {geojson: 'geometry'}}
  ]);
  expect(plan.geometryColumn).toBe('geometry');
  expect(plan.warnings).toEqual([]);
  const geometry = preparePublishedGeometry(plan.table, 'geometry');
  expect(geometry.metadata.invalidRows).toEqual([]);
  expect(geometry.metadata.featureTypes.polygon).toBe(true);
  expect(geometry.batches.some(batch => batch.polygonOutlines.positions.length > 0)).toBe(true);
});

test('reference failures name the sample and dataset path', async () => {
  const original = readOriginalConfig('earthquakes');
  await expect(
    runFixture(JSON.stringify({...createFixture('earthquakes', original), missingDataset: true}))
  ).rejects.toThrow('earthquakes: reference.datasets.earthquakes: Missing restored native dataset');
});

test('fallback configs exercise null altitude, complete effect defaults and tile-field bindings without the cache', async () => {
  for (const sampleId of ['no-cache-point', 'mvt_population']) {
    const original = readOriginalConfig(sampleId, false);
    const result = JSON.parse(await runFixture(JSON.stringify(createFixture(sampleId, original))));
    assertSamplePresentation(result.actual, result.expected, [], sampleId);
  }
});

test('the reference detaches an Arrow source through stock table import and normal registration', async () => {
  const original = readOriginalConfig('arrow-fixture', false);
  const result = JSON.parse(
    await runFixture(JSON.stringify({...createFixture('arrow-fixture', original), arrow: true}))
  );
  assertSamplePresentation(result.actual, result.expected, [], 'arrow-fixture');
});

test.each([
  'earthquakes',
  'nyctrips',
  'la_assessorparcels'
])('published Arrow millisecond timestamps preserve %s filter values and plot intervals', async sampleId => {
  const original = readOriginalConfig(sampleId);
  if (!original.config.visState.filters?.some(filter => filter.type === 'timeRange')) {
    original.config.visState.filters = [
      {
        id: 'instant',
        dataId: [sampleId],
        name: ['instant'],
        type: 'timeRange',
        value: [1136073600000, 1451606400000]
      }
    ] as never;
  }
  const result = JSON.parse(
    await runFixture(
      JSON.stringify({
        ...createFixture(sampleId, original),
        publishedTime: true
      })
    )
  );
  assertSamplePresentation(result.actual, result.expected, [], sampleId);
  expect(result.expected.config.visState.filters).toEqual(result.baseline.config.visState.filters);
});

test.each([
  false,
  true
])('resident remove/re-register preserves Pittsburgh per-panel visibility (use cache: %s)', async useCache => {
  const original = readOriginalConfig('movement_pittsburgh', useCache);
  if (!original.config.visState.splitMaps?.length) {
    const first = original.config.visState.layers[0]!;
    const second = {...structuredClone(first), id: 'comparison'};
    original.config.visState.layers.push(second);
    original.config.visState.splitMaps = [
      {layers: {[first.id]: true, [second.id]: false}},
      {layers: {[first.id]: false, [second.id]: true}}
    ] as never;
  }
  const result = JSON.parse(
    await runFixture(
      JSON.stringify({
        ...createFixture('movement_pittsburgh', original),
        republish: true
      })
    )
  );
  assertSamplePresentation(result.actual, result.expected, [], 'movement_pittsburgh');
  expect(result.actual.config.visState.splitMaps).toEqual(
    result.baseline.config.visState.splitMaps
  );
});

test('eager tile metadata rejection is observed and reports its sample and layer path', async () => {
  const original = readOriginalConfig('lake_lucerne_raster', false);
  await expect(
    runFixture(
      JSON.stringify({
        ...createFixture('lake_lucerne_raster', original),
        referenceRemoteUrl: 'https://example.invalid/reference.pmtiles'
      })
    )
  ).rejects.toThrow(
    'lake_lucerne_raster: reference.remoteMetadata: visState.layers.layer.tileSource.metadata'
  );
});

test.each([
  'channels',
  'filter-type',
  'filter-source',
  'filter-value',
  'split-visibility',
  'camera',
  'dataset'
])('strict comparison rejects tampered %s after an actual stock reducer reference run', async mutation => {
  const original = readOriginalConfig('la_assessorparcels');
  const first = original.config.visState.layers[0]!;
  if (!original.config.visState.filters?.length)
    original.config.visState.filters = [
      {
        id: 'range',
        dataId: first.config.dataId,
        name: 'value',
        type: 'range',
        value: [1, 3]
      }
    ] as never;
  if (!original.config.visState.splitMaps?.length)
    original.config.visState.splitMaps = [
      {layers: {[first.id]: true}},
      {layers: {[first.id]: false}}
    ] as never;
  const result = JSON.parse(
    await runFixture(JSON.stringify(createFixture('la_assessorparcels', original)))
  );
  assertSamplePresentation(result.actual, result.expected, [], 'untampered');
  const actual = result.actual.config.visState;
  if (mutation === 'channels') actual.layers[0].visualChannels = {};
  if (mutation === 'filter-type') actual.filters[0].type = 'select';
  if (mutation === 'filter-source') actual.filters[0].dataId = ['tampered'];
  if (mutation === 'filter-value') actual.filters[0].value = [0, 1];
  if (mutation === 'split-visibility')
    actual.splitMaps[0].layers[first.id] = !actual.splitMaps[0].layers[first.id];
  if (mutation === 'camera') result.actual.config.mapState.latitude += 1;
  if (mutation === 'dataset') actual.layers[0].config.dataId = 'tampered';
  expect(() => assertSamplePresentation(result.actual, result.expected, [], 'tampered')).toThrow(
    'tampered'
  );
});
