// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Deterministic generator for native Kepler.gl fixtures. Raw files mirror Kepler demo dataset
// schemas; saved maps are produced by Kepler's own KeplerGlSchema (real layer classes, real
// dataset processors) so they are exactly the kepler.json export format.
//
// Usage: node scripts/kepler-fixtures/generate.mjs

import {createRequire} from 'node:module';
import {mkdirSync, writeFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {loadKeplerSchemas} from './load-kepler-schemas.mjs';

const directory = path.dirname(fileURLToPath(import.meta.url));
const dataDirectory = path.join(directory, 'data');
const require = createRequire(path.resolve(directory, '../../../../../package.json'));
const h3 = require('h3-js');
const S2 = require('s2-geometry').S2;
const a5 = require('a5-js');

const {KeplerGlSchema} = loadKeplerSchemas();
const {LayerClasses} = require('@kepler.gl/layers');
const {processCsvData, processGeojson} = require('@kepler.gl/processors');

mkdirSync(dataDirectory, {recursive: true});

// ---------------------------------------------------------------------------------------------
// Seeded random numbers and helpers
// ---------------------------------------------------------------------------------------------

function createRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function createGaussian(random) {
  return () => {
    const first = Math.max(random(), 1e-9);
    return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * random());
  };
}

const round = (value, digits) => Number(value.toFixed(digits));
const pad = (value, length = 2) => String(value).padStart(length, '0');
function formatTimestamp(milliseconds) {
  const date = new Date(milliseconds);
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())} ${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}:${pad(date.getUTCSeconds())}`;
}
function writeCsv(fileName, header, rows) {
  const text = `${header.join(',')}\n${rows.map(row => row.join(',')).join('\n')}\n`;
  writeFileSync(path.join(dataDirectory, fileName), text);
  return text;
}
function writeJson(fileName, value, indentation) {
  const text = JSON.stringify(value, null, indentation);
  writeFileSync(path.join(dataDirectory, fileName), `${text}\n`);
  return text;
}
function pick(random, values) {
  return values[Math.floor(random() * values.length)];
}

// ---------------------------------------------------------------------------------------------
// Raw dataset files
// ---------------------------------------------------------------------------------------------

function makeEarthquakes() {
  const random = createRandom(1);
  const gaussian = createGaussian(random);
  const centers = [
    [35.7, 140.2],
    [-33.4, -71.6],
    [61.2, -149.9],
    [-6.2, 106.8],
    [38.3, -122.3],
    [14.6, -90.5],
    [-17.8, -178.2]
  ];
  const start = Date.UTC(2018, 0, 1);
  const span = 365 * 86400000;
  const rows = [];
  for (let index = 0; index < 1500; index++) {
    const center = pick(random, centers);
    const magnitude = round(4 + Math.min(4.2, -Math.log(1 - random()) * 0.7), 1);
    rows.push([
      formatTimestamp(start + Math.floor(random() * span)),
      round(center[0] + gaussian() * 2.5, 4),
      round(center[1] + gaussian() * 3, 4),
      round(Math.abs(gaussian()) * 80 + random() * 10, 2),
      magnitude,
      pick(random, ['mb', 'ml', 'mww', 'mwc', 'md']),
      Math.floor(10 + random() * 140),
      round(20 + random() * 200, 1),
      round(random() * 12, 3),
      round(0.2 + random() * 1.2, 2),
      pick(random, ['us', 'nc', 'ak', 'ci']),
      `us${(1000000 + index * 7).toString(36)}`
    ]);
  }
  rows.sort((left, right) => (left[0] < right[0] ? -1 : 1));
  return writeCsv(
    'earthquakes.csv',
    ['DateTime', 'Latitude', 'Longitude', 'Depth', 'Magnitude', 'MagType', 'NbStations', 'Gap', 'Distance', 'RMS', 'Source', 'EventID'],
    rows
  );
}

function makeNycTrips() {
  const random = createRandom(2);
  const gaussian = createGaussian(random);
  const hubs = [
    [40.7527, -73.9772],
    [40.7484, -73.9857],
    [40.7061, -74.0087],
    [40.7794, -73.9632],
    [40.6413, -73.7781],
    [40.7769, -73.874]
  ];
  const start = Date.UTC(2015, 0, 15);
  const rows = [];
  for (let index = 0; index < 1200; index++) {
    const pickupHub = pick(random, hubs);
    const dropoffHub = pick(random, hubs);
    const pickupTime = start + Math.floor(random() * 86400000 * 2);
    const durationMinutes = 4 + Math.floor(random() * 40);
    const tripDistance = round(0.4 + random() * 9 + (pickupHub === dropoffHub ? 0 : 3), 2);
    const fareAmount = round(3 + tripDistance * 2.4 + random() * 4, 2);
    const tipAmount = round(random() < 0.3 ? 0 : fareAmount * 0.18, 2);
    rows.push([
      formatTimestamp(pickupTime),
      formatTimestamp(pickupTime + durationMinutes * 60000),
      round(pickupHub[1] + gaussian() * 0.008, 6),
      round(pickupHub[0] + gaussian() * 0.008, 6),
      round(dropoffHub[1] + gaussian() * 0.01, 6),
      round(dropoffHub[0] + gaussian() * 0.01, 6),
      tripDistance,
      fareAmount,
      1 + Math.floor(random() * 5),
      pick(random, ['CRD', 'CSH', 'CRD', 'NOC']),
      tipAmount,
      round(fareAmount + tipAmount + 0.8, 2)
    ]);
  }
  rows.sort((left, right) => (left[0] < right[0] ? -1 : 1));
  return writeCsv(
    'nyc-trips.csv',
    ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'trip_distance', 'fare_amount', 'passenger_count', 'payment_type', 'tip_amount', 'total_amount'],
    rows
  );
}

function makeSfZipGeometry() {
  const random = createRandom(3);
  const features = [];
  const columns = 9;
  const rowsCount = 7;
  const west = -122.52;
  const south = 37.7;
  const cellWidth = 0.0135;
  const cellHeight = 0.0125;
  let zipcode = 94101;
  for (let row = 0; row < rowsCount; row++) {
    for (let column = 0; column < columns; column++) {
      const jitter = () => (random() - 0.5) * 0.003;
      const x0 = west + column * cellWidth;
      const y0 = south + row * cellHeight;
      const ring = [
        [x0 + jitter(), y0 + jitter()],
        [x0 + cellWidth + jitter(), y0 + jitter()],
        [x0 + cellWidth + jitter(), y0 + cellHeight + jitter()],
        [x0 + jitter(), y0 + cellHeight + jitter()]
      ].map(([x, y]) => [round(x, 5), round(y, 5)]);
      ring.push(ring[0]);
      features.push({
        type: 'Feature',
        properties: {
          zipcode: zipcode++,
          TRACTCE: 100 + row * 40 + column,
          median_income: Math.round(35000 + random() * 140000),
          population: Math.round(2000 + random() * 38000),
          area: round(0.4 + random() * 2.2, 3)
        },
        geometry: {type: 'Polygon', coordinates: [ring]}
      });
    }
  }
  return writeJson('sf-zip-geometry.geojson', {type: 'FeatureCollection', features});
}

function makeH3Hexagons() {
  const random = createRandom(4);
  const origin = h3.latLngToCell(37.7749, -122.4194, 8);
  const cells = h3.gridDisk(origin, 14);
  const rows = cells.map((cell, index) => [
    cell,
    round(random() * 100, 3),
    Math.floor(random() * 500),
    pick(random, ['residential', 'commercial', 'industrial', 'park']),
    `cell-${index}`
  ]);
  return writeCsv('h3-hexagons.csv', ['hex_id', 'value', 'count', 'land_use', 'label'], rows);
}

const FLOW_PLACES = [
  ['San Francisco', 37.7749, -122.4194],
  ['Oakland', 37.8044, -122.2712],
  ['San Jose', 37.3382, -121.8863],
  ['Sacramento', 38.5816, -121.4944],
  ['Fresno', 36.7378, -119.7871],
  ['Los Angeles', 34.0522, -118.2437],
  ['San Diego', 32.7157, -117.1611],
  ['Reno', 39.5296, -119.8138]
];

function makeArcsFlows() {
  const random = createRandom(5);
  const rows = [];
  for (let index = 0; index < 400; index++) {
    const source = pick(random, FLOW_PLACES);
    let target = pick(random, FLOW_PLACES);
    if (target === source) target = FLOW_PLACES[(FLOW_PLACES.indexOf(source) + 1) % FLOW_PLACES.length];
    rows.push([
      round(source[1] + (random() - 0.5) * 0.05, 5),
      round(source[2] + (random() - 0.5) * 0.05, 5),
      round(target[1] + (random() - 0.5) * 0.05, 5),
      round(target[2] + (random() - 0.5) * 0.05, 5),
      1 + Math.floor(random() * 900),
      `${source[0]} to ${target[0]}`,
      source[0],
      target[0]
    ]);
  }
  return writeCsv('arcs-flows.csv', ['lat0', 'lng0', 'lat1', 'lng1', 'count', 'names', 'source_name', 'target_name'], rows);
}

function makeTripsGeojson() {
  const random = createRandom(6);
  const features = [];
  const start = Math.floor(Date.UTC(2020, 5, 1, 8) / 1000);
  for (let index = 0; index < 80; index++) {
    let longitude = -122.45 + random() * 0.06;
    let latitude = 37.74 + random() * 0.06;
    const heading = random() * Math.PI * 2;
    const coordinates = [];
    const startTime = start + Math.floor(random() * 1800);
    for (let step = 0; step < 24; step++) {
      longitude += Math.cos(heading + (random() - 0.5) * 0.9) * 0.0016;
      latitude += Math.sin(heading + (random() - 0.5) * 0.9) * 0.0013;
      coordinates.push([round(longitude, 6), round(latitude, 6), 0, startTime + step * 30]);
    }
    features.push({
      type: 'Feature',
      properties: {vendor: pick(random, ['A', 'B', 'C']), id: `trip-${index}`, speed: round(10 + random() * 40, 1)},
      geometry: {type: 'LineString', coordinates}
    });
  }
  return writeJson('trips.geojson', {type: 'FeatureCollection', features});
}

function makeCellTokens() {
  const random = createRandom(7);
  const rows = [];
  for (let index = 0; index < 240; index++) {
    const latitude = round(37.68 + random() * 0.2, 5);
    const longitude = round(-122.55 + random() * 0.25, 5);
    const s2Key = S2.latLngToKey(latitude, longitude, 14);
    const s2Identifier = BigInt(S2.keyToId(s2Key)).toString(16).padStart(16, '0').replace(/0+$/, '');
    const geohash = encodeGeohash(latitude, longitude, 7);
    const a5Token = a5.u64ToHex(a5.lonLatToCell([longitude, latitude], 14));
    rows.push([latitude, longitude, s2Identifier, geohash, a5Token, round(random() * 100, 2), Math.floor(random() * 1000)]);
  }
  return writeCsv('cell-tokens.csv', ['lat', 'lng', 's2_token', 'geohash', 'a5_cell', 'value', 'count'], rows);
}

function encodeGeohash(latitude, longitude, precision) {
  const alphabet = '0123456789bcdefghjkmnpqrstuvwxyz';
  let latitudeRange = [-90, 90];
  let longitudeRange = [-180, 180];
  let isLongitude = true;
  let bits = 0;
  let value = 0;
  let output = '';
  while (output.length < precision) {
    const range = isLongitude ? longitudeRange : latitudeRange;
    const middle = (range[0] + range[1]) / 2;
    const coordinate = isLongitude ? longitude : latitude;
    value <<= 1;
    if (coordinate >= middle) {
      value |= 1;
      range[0] = middle;
    } else {
      range[1] = middle;
    }
    isLongitude = !isLongitude;
    if (++bits === 5) {
      output += alphabet[value];
      bits = 0;
      value = 0;
    }
  }
  return output;
}

function makeIcons() {
  const random = createRandom(8);
  const rows = [];
  for (let index = 0; index < 150; index++) {
    rows.push([round(37.7 + random() * 0.15, 5), round(-122.5 + random() * 0.15, 5), pick(random, ['diamond', 'triangle', 'square']), round(random() * 10, 2)]);
  }
  return writeCsv('icons.csv', ['lat', 'lng', 'icon', 'value'], rows);
}

function makeWindField() {
  const rows = [];
  for (let row = 0; row < 14; row++) {
    for (let column = 0; column < 14; column++) {
      const latitude = round(37.5 + row * 0.03, 4);
      const longitude = round(-122.6 + column * 0.03, 4);
      rows.push([latitude, longitude, round(Math.sin(row / 2.5) * 6, 3), round(Math.cos(column / 2.5) * 6, 3)]);
    }
  }
  return writeCsv('wind-field.csv', ['lat', 'lng', 'u', 'v'], rows);
}

const rawText = {
  earthquakes: makeEarthquakes(),
  'nyc-trips': makeNycTrips(),
  'sf-zips': makeSfZipGeometry(),
  'h3-hexagons': makeH3Hexagons(),
  'arcs-flows': makeArcsFlows(),
  trips: makeTripsGeojson(),
  'cell-tokens': makeCellTokens(),
  icons: makeIcons(),
  'wind-field': makeWindField()
};

// ---------------------------------------------------------------------------------------------
// Datasets processed by Kepler's own file processors, exactly as Load Files would
// ---------------------------------------------------------------------------------------------

const DATASET_COLORS = [
  [143, 47, 191],
  [227, 26, 28],
  [18, 147, 154],
  [255, 203, 153],
  [248, 149, 112],
  [255, 153, 31],
  [31, 186, 214],
  [106, 61, 154],
  [51, 160, 44]
];
const DATASET_ORDER = Object.keys(rawText);

function processDataset(id, label, limit) {
  const isGeojson = id === 'sf-zips' || id === 'trips';
  const processed = isGeojson ? processGeojson(JSON.parse(rawText[id])) : processCsvData(rawText[id]);
  const rows = limit ? processed.rows.slice(0, limit) : processed.rows;
  return {
    id,
    label,
    color: DATASET_COLORS[DATASET_ORDER.indexOf(id) % DATASET_COLORS.length],
    allData: rows,
    fields: processed.fields.map(({name, type, format, analyzerType}) => ({name, type, format, analyzerType}))
  };
}

const LABELS = {
  earthquakes: 'earthquakes.csv',
  'nyc-trips': 'nyc-trips.csv',
  'sf-zips': 'sf-zip-geometry.geojson',
  'h3-hexagons': 'h3-hexagons.csv',
  'arcs-flows': 'arcs-flows.csv',
  trips: 'trips.geojson',
  'cell-tokens': 'cell-tokens.csv',
  icons: 'icons.csv',
  'wind-field': 'wind-field.csv'
};
const datasetCache = new Map();
function getDataset(id, limit) {
  const key = `${id}:${limit || 0}`;
  if (!datasetCache.has(key)) datasetCache.set(key, processDataset(id, LABELS[id], limit));
  return datasetCache.get(key);
}
function getField(dataset, name) {
  const fieldIndex = dataset.fields.findIndex(field => field.name === name);
  if (fieldIndex < 0) throw new Error(`Field ${name} missing from ${dataset.id}`);
  return {fieldIndex, field: dataset.fields[fieldIndex]};
}
function getFieldRef(dataset, name) {
  const {field} = getField(dataset, name);
  return {name: field.name, type: field.type};
}
function getColumns(dataset, mapping) {
  return Object.fromEntries(
    Object.entries(mapping).map(([key, name]) => [key, {value: name, fieldIdx: getField(dataset, name).fieldIndex}])
  );
}
function getColumnExtent(dataset, name) {
  const {fieldIndex} = getField(dataset, name);
  const values = dataset.allData.map(row => row[fieldIndex]).filter(value => value !== null);
  return [values.reduce((a, b) => (a < b ? a : b)), values.reduce((a, b) => (a > b ? a : b))];
}

// ---------------------------------------------------------------------------------------------
// Layers (real Kepler layer classes) and filters
// ---------------------------------------------------------------------------------------------

function makeLayer(type, {id, dataset, label, mapping, columnMode, channels = {}, visConfig = {}, color}) {
  const LayerClass = LayerClasses[type];
  const layer = new LayerClass({
    id,
    dataId: dataset.id,
    label,
    columns: getColumns(dataset, mapping),
    ...(columnMode ? {columnMode} : {}),
    ...(color ? {color} : {}),
    isVisible: true
  });
  const update = {};
  for (const [channel, [fieldName, scale]] of Object.entries(channels)) {
    update[`${channel}Field`] = getFieldRef(dataset, fieldName);
    if (scale) update[`${channel}Scale`] = scale;
  }
  layer.updateLayerConfig(update);
  layer.updateLayerVisConfig(visConfig);
  if (!layer.isValidToSave()) throw new Error(`Layer ${id} (${type}) is missing required columns`);
  return layer;
}

let filterSequence = 0;
function makeFilter(dataset, type, fieldName, value, extra = {}) {
  filterSequence++;
  return {
    dataId: [dataset.id],
    id: `fixture-filter-${filterSequence}`,
    name: [fieldName],
    type,
    value,
    enlarged: false,
    plotType: {type: 'histogram'},
    animationWindow: 'free',
    yAxis: null,
    view: 'side',
    speed: 1,
    enabled: true,
    ...extra
  };
}
function makeRangeFilter(dataset, fieldName, lowFraction, highFraction) {
  const [low, high] = getColumnExtent(dataset, fieldName);
  const value = [low + (high - low) * lowFraction, low + (high - low) * highFraction].map(entry => round(entry, 3));
  return makeFilter(dataset, 'range', fieldName, value);
}
function makeTimeFilter(dataset, fieldName, lowFraction, highFraction) {
  const [low, high] = getColumnExtent(dataset, fieldName).map(text => Date.parse(`${text.replace(' ', 'T')}Z`));
  return makeFilter(
    dataset,
    'timeRange',
    fieldName,
    [Math.round(low + (high - low) * lowFraction), Math.round(low + (high - low) * highFraction)],
    {plotType: {type: 'histogram'}, timezone: 'UTC'}
  );
}
function makeSelectFilter(dataset, fieldName, values) {
  return makeFilter(dataset, 'multiSelect', fieldName, values, {plotType: {}});
}

function makeTooltip(entries) {
  return {
    id: 'tooltip',
    enabled: true,
    config: {
      fieldsToShow: Object.fromEntries(
        Object.entries(entries).map(([dataId, names]) => [dataId, names.map(name => ({name, format: null}))])
      ),
      compareMode: false,
      compareType: 'absolute'
    }
  };
}

function saveMap({title, description, datasets, layers, filters = [], tooltip, mapState, styleType = 'dark', animationCurrentTime = null}) {
  const state = {
    visState: {
      datasets: Object.fromEntries(datasets.map(dataset => [dataset.id, dataset])),
      layers,
      layerOrder: layers.map(layer => layer.id),
      filters,
      interactionConfig: {
        tooltip,
        brush: {id: 'brush', enabled: false, config: {size: 0.5}},
        geocoder: {id: 'geocoder', enabled: false, config: {}},
        coordinate: {id: 'coordinate', enabled: false, config: {}}
      },
      layerBlending: 'normal',
      overlayBlending: 'normal',
      splitMaps: [],
      animationConfig: {currentTime: animationCurrentTime, speed: 1, timeFormat: null, timezone: null},
      editor: {features: [], visible: true},
      effects: [],
      effectOrder: [],
      charts: [],
      annotations: [],
      mapInfo: {title, description}
    },
    mapState: {bearing: 0, dragRotate: false, isSplit: false, pitch: 0, zoom: 8, ...mapState},
    mapStyle: {
      styleType,
      topLayerGroups: {},
      visibleLayerGroups: {label: true, road: true, border: false, building: true, water: true, land: true, '3d building': false},
      threeDBuildingColor: [9.665468314072013, 17.18305478057247, 31.1442867897876],
      backgroundColor: [0, 0, 0],
      mapStyles: {}
    },
    uiState: {
      mapControls: {
        mapLegend: {active: true, settings: {position: 'bottom-left', contentHeight: 150}},
        chart: {active: false},
        mapInfo: {active: false, show: true}
      },
      locale: 'en',
      theme: 'dark'
    }
  };
  const saved = KeplerGlSchema.save(state);
  saved.info.created_at = 'Sat Oct 31 2026 00:00:00 GMT+0000 (Coordinated Universal Time)';
  return saved;
}

// ---------------------------------------------------------------------------------------------
// Saved maps
// ---------------------------------------------------------------------------------------------

const earthquakes = getDataset('earthquakes');
const nycTrips = getDataset('nyc-trips');
const sfZips = getDataset('sf-zips');
const h3Hexagons = getDataset('h3-hexagons');
const arcsFlows = getDataset('arcs-flows');
const trips = getDataset('trips');

const savedMaps = {};

savedMaps['earthquakes.kepler.json'] = saveMap({
  title: 'Earthquakes',
  description: 'Point layer with range, time and multi-select filters',
  datasets: [earthquakes],
  layers: [
    makeLayer('point', {
      id: 'eq-point',
      dataset: earthquakes,
      label: 'Earthquakes',
      mapping: {lat: 'Latitude', lng: 'Longitude'},
      channels: {color: ['Magnitude', 'quantile'], size: ['Depth', 'sqrt']},
      visConfig: {radius: 8, radiusRange: [1, 40], opacity: 0.8}
    })
  ],
  filters: [
    makeRangeFilter(earthquakes, 'Magnitude', 0, 0.9),
    makeTimeFilter(earthquakes, 'DateTime', 0, 0.9),
    makeSelectFilter(earthquakes, 'MagType', ['mb', 'ml', 'mww', 'md'])
  ],
  tooltip: makeTooltip({earthquakes: ['DateTime', 'Magnitude', 'Depth', 'MagType', 'Source']}),
  mapState: {latitude: 15, longitude: -140, zoom: 1.6}
});

savedMaps['nyc-trips.kepler.json'] = saveMap({
  title: 'NYC taxi trips',
  description: 'Arc and hexagon layers with fare and time filters',
  datasets: [nycTrips],
  layers: [
    makeLayer('arc', {
      id: 'nyc-arc',
      dataset: nycTrips,
      label: 'Pickup to dropoff',
      mapping: {lat0: 'pickup_latitude', lng0: 'pickup_longitude', lat1: 'dropoff_latitude', lng1: 'dropoff_longitude'},
      channels: {size: ['fare_amount', 'linear']},
      visConfig: {opacity: 0.6, thickness: 2, targetColor: [255, 0, 255]}
    }),
    makeLayer('hexagon', {
      id: 'nyc-hexagon',
      dataset: nycTrips,
      label: 'Pickup density',
      mapping: {lat: 'pickup_latitude', lng: 'pickup_longitude'},
      visConfig: {worldUnitSize: 0.3, elevationScale: 5, enable3d: true}
    })
  ],
  filters: [
    makeRangeFilter(nycTrips, 'fare_amount', 0, 0.8),
    makeTimeFilter(nycTrips, 'tpep_pickup_datetime', 0.1, 0.9),
    makeSelectFilter(nycTrips, 'payment_type', ['CRD', 'CSH'])
  ],
  tooltip: makeTooltip({'nyc-trips': ['tpep_pickup_datetime', 'trip_distance', 'fare_amount', 'payment_type']}),
  mapState: {latitude: 40.74, longitude: -73.95, zoom: 10.5, pitch: 40, bearing: 0}
});

savedMaps['sf-zips.kepler.json'] = saveMap({
  title: 'SF zip codes',
  description: 'GeoJSON polygons with fill color, extrusion height and range filter',
  datasets: [sfZips],
  layers: [
    makeLayer('geojson', {
      id: 'sf-zip-geojson',
      dataset: sfZips,
      label: 'Zip codes',
      mapping: {geojson: '_geojson'},
      channels: {color: ['median_income', 'quantize'], height: ['population', 'linear']},
      visConfig: {opacity: 0.7, filled: true, stroked: true, enable3d: true, heightRange: [0, 400], elevationScale: 4}
    })
  ],
  filters: [makeRangeFilter(sfZips, 'median_income', 0, 0.95)],
  tooltip: makeTooltip({'sf-zips': ['zipcode', 'median_income', 'population']}),
  mapState: {latitude: 37.745, longitude: -122.46, zoom: 11, pitch: 35}
});

savedMaps['h3-hexagons.kepler.json'] = saveMap({
  title: 'H3 hexagons',
  description: 'H3 token layer with value color and count filter',
  datasets: [h3Hexagons],
  layers: [
    makeLayer('hexagonId', {
      id: 'h3-layer',
      dataset: h3Hexagons,
      label: 'H3 cells',
      mapping: {hex_id: 'hex_id'},
      channels: {color: ['value', 'quantile'], size: ['count', 'linear']},
      visConfig: {opacity: 0.8, coverage: 0.9, enable3d: true, elevationScale: 5}
    })
  ],
  filters: [makeRangeFilter(h3Hexagons, 'count', 0, 0.9), makeSelectFilter(h3Hexagons, 'land_use', ['residential', 'commercial', 'park'])],
  tooltip: makeTooltip({'h3-hexagons': ['hex_id', 'value', 'count', 'land_use']}),
  mapState: {latitude: 37.7749, longitude: -122.4194, zoom: 10.5, pitch: 30}
});

savedMaps['arcs-flows.kepler.json'] = saveMap({
  title: 'Arc flows',
  description: 'Arc, line and flow layers over an origin-destination table',
  datasets: [arcsFlows],
  layers: [
    makeLayer('arc', {
      id: 'flow-arc',
      dataset: arcsFlows,
      label: 'Arcs',
      mapping: {lat0: 'lat0', lng0: 'lng0', lat1: 'lat1', lng1: 'lng1'},
      channels: {size: ['count', 'sqrt']},
      visConfig: {opacity: 0.6, thickness: 3}
    }),
    makeLayer('line', {
      id: 'flow-line',
      dataset: arcsFlows,
      label: 'Lines',
      mapping: {lat0: 'lat0', lng0: 'lng0', lat1: 'lat1', lng1: 'lng1'},
      channels: {size: ['count', 'linear']},
      visConfig: {opacity: 0.4}
    }),
    makeLayer('flow', {
      id: 'flow-flowmap',
      dataset: arcsFlows,
      label: 'Flow map',
      columnMode: 'LAT_LNG',
      mapping: {lat0: 'lat0', lng0: 'lng0', lat1: 'lat1', lng1: 'lng1', count: 'count', sourceName: 'source_name', targetName: 'target_name'}
    })
  ],
  filters: [makeRangeFilter(arcsFlows, 'count', 0.05, 1)],
  tooltip: makeTooltip({'arcs-flows': ['names', 'count']}),
  mapState: {latitude: 36.5, longitude: -120, zoom: 5.2}
});

savedMaps['trips.kepler.json'] = saveMap({
  title: 'Trips',
  description: 'Animated GeoJSON trips with timestamps in the fourth coordinate',
  datasets: [trips],
  layers: [
    makeLayer('trip', {
      id: 'trip-layer',
      dataset: trips,
      label: 'Trips',
      mapping: {geojson: '_geojson'},
      channels: {color: ['vendor', 'ordinal']},
      visConfig: {opacity: 0.8, thickness: 2, trailLength: 180}
    })
  ],
  filters: [],
  tooltip: makeTooltip({trips: ['id', 'vendor', 'speed']}),
  mapState: {latitude: 37.77, longitude: -122.42, zoom: 12},
  animationCurrentTime: Math.floor(Date.UTC(2020, 5, 1, 8, 5) / 1000) * 1000
});

// One map that references every data-driven layer type Kepler 3.3.0-alpha.18 supports.
// Remote-source types (vectorTile, rasterTile, wms, tile3d, bitmap) need hosted data and are
// intentionally excluded; see NOTES.md.
const tokens = getDataset('cell-tokens');
// A resident publication carries one cell-ID column, so each DGGS family gets its own dataset.
const geohashTokens = {...tokens, id: 'cell-tokens-geohash', label: 'cell-tokens-geohash.csv'};
const a5Tokens = {...tokens, id: 'cell-tokens-a5', label: 'cell-tokens-a5.csv'};
const icons = getDataset('icons');
const wind = getDataset('wind-field');
const everyLayerDatasets = [
  getDataset('earthquakes', 300),
  getDataset('nyc-trips', 300),
  sfZips,
  h3Hexagons,
  getDataset('arcs-flows', 120),
  trips,
  tokens,
  geohashTokens,
  a5Tokens,
  icons,
  wind
];
const quakeSubset = everyLayerDatasets[0];
const tripSubset = everyLayerDatasets[1];
const flowSubset = everyLayerDatasets[4];
const allLayers = [
  makeLayer('point', {id: 'all-point', dataset: quakeSubset, label: 'point', mapping: {lat: 'Latitude', lng: 'Longitude'}, channels: {color: ['Magnitude', 'quantile']}}),
  makeLayer('heatmap', {id: 'all-heatmap', dataset: quakeSubset, label: 'heatmap', mapping: {lat: 'Latitude', lng: 'Longitude'}, channels: {weight: ['Magnitude', 'linear']}}),
  makeLayer('hexagon', {id: 'all-hexagon', dataset: quakeSubset, label: 'hexagon', mapping: {lat: 'Latitude', lng: 'Longitude'}, visConfig: {worldUnitSize: 400}}),
  makeLayer('grid', {id: 'all-grid', dataset: quakeSubset, label: 'grid', mapping: {lat: 'Latitude', lng: 'Longitude'}, visConfig: {worldUnitSize: 400}}),
  makeLayer('cluster', {id: 'all-cluster', dataset: quakeSubset, label: 'cluster', mapping: {lat: 'Latitude', lng: 'Longitude'}}),
  makeLayer('3D', {id: 'all-3d', dataset: quakeSubset, label: '3D', mapping: {lat: 'Latitude', lng: 'Longitude'}}),
  makeLayer('arc', {id: 'all-arc', dataset: tripSubset, label: 'arc', mapping: {lat0: 'pickup_latitude', lng0: 'pickup_longitude', lat1: 'dropoff_latitude', lng1: 'dropoff_longitude'}, channels: {size: ['fare_amount', 'linear']}}),
  makeLayer('line', {id: 'all-line', dataset: tripSubset, label: 'line', mapping: {lat0: 'pickup_latitude', lng0: 'pickup_longitude', lat1: 'dropoff_latitude', lng1: 'dropoff_longitude'}}),
  makeLayer('geojson', {id: 'all-geojson', dataset: sfZips, label: 'geojson', mapping: {geojson: '_geojson'}, channels: {color: ['median_income', 'quantize']}}),
  makeLayer('hexagonId', {id: 'all-h3', dataset: h3Hexagons, label: 'hexagonId', mapping: {hex_id: 'hex_id'}, channels: {color: ['value', 'quantile']}}),
  makeLayer('trip', {id: 'all-trip', dataset: trips, label: 'trip', mapping: {geojson: '_geojson'}}),
  makeLayer('s2', {id: 'all-s2', dataset: tokens, label: 's2', mapping: {token: 's2_token'}, channels: {color: ['value', 'quantile']}}),
  makeLayer('geohash', {id: 'all-geohash', dataset: geohashTokens, label: 'geohash', mapping: {token: 'geohash'}, channels: {color: ['value', 'quantile']}}),
  makeLayer('a5', {id: 'all-a5', dataset: a5Tokens, label: 'a5', mapping: {token: 'a5_cell'}, channels: {color: ['value', 'quantile']}}),
  makeLayer('icon', {id: 'all-icon', dataset: icons, label: 'icon', mapping: {lat: 'lat', lng: 'lng', icon: 'icon'}, channels: {color: ['value', 'quantile']}}),
  makeLayer('flow', {id: 'all-flow', dataset: flowSubset, label: 'flow', columnMode: 'LAT_LNG', mapping: {lat0: 'lat0', lng0: 'lng0', lat1: 'lat1', lng1: 'lng1', count: 'count', sourceName: 'source_name', targetName: 'target_name'}}),
  makeLayer('flowField', {id: 'all-flow-field', dataset: wind, label: 'flowField', columnMode: 'UV', mapping: {u: 'u', v: 'v', lat: 'lat', lng: 'lng'}})
];
savedMaps['all-layer-types.kepler.json'] = saveMap({
  title: 'All layer types',
  description: 'Every data-driven Kepler layer type against suitable datasets',
  datasets: everyLayerDatasets,
  layers: allLayers,
  filters: [makeRangeFilter(quakeSubset, 'Magnitude', 0, 0.95), makeSelectFilter(h3Hexagons, 'land_use', ['residential', 'commercial', 'park', 'industrial'])],
  tooltip: makeTooltip({earthquakes: ['DateTime', 'Magnitude'], 'sf-zips': ['zipcode', 'median_income'], 'h3-hexagons': ['hex_id', 'value']}),
  mapState: {latitude: 37.75, longitude: -122.4, zoom: 9, pitch: 30}
});

for (const [fileName, saved] of Object.entries(savedMaps)) writeJson(fileName, saved);

console.log(
  JSON.stringify({
    written: Object.keys(rawText).length + Object.keys(savedMaps).length,
    savedMaps: Object.keys(savedMaps),
    layerCounts: Object.fromEntries(Object.entries(savedMaps).map(([name, saved]) => [name, saved.config.config.visState.layers.length]))
  })
);
