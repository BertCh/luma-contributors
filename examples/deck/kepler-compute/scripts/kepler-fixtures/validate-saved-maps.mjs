// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Node-only validation of the saved-map fixtures with Kepler's own schema and processors.
// Usage: node scripts/kepler-fixtures/validate-saved-maps.mjs

import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFileSync, readdirSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {loadKeplerSchemas} from './load-kepler-schemas.mjs';

const directory = path.dirname(fileURLToPath(import.meta.url));
const dataDirectory = path.join(directory, 'data');
const require = createRequire(path.resolve(directory, '../../../../../package.json'));
const {KeplerGlSchema} = loadKeplerSchemas();
const {LayerClasses} = require('@kepler.gl/layers');
const {processCsvData, processGeojson, processKeplerglJSON} = require('@kepler.gl/processors');
const h3 = require('h3-js');
const S2 = require('s2-geometry').S2;
const a5 = require('a5-js');

const EXPECTED_ALL_LAYER_TYPES = [
  'point', 'heatmap', 'hexagon', 'grid', 'cluster', '3D', 'arc', 'line', 'geojson',
  'hexagonId', 'trip', 's2', 'geohash', 'a5', 'icon', 'flow', 'flowField'
];
const REMOTE_SOURCE_TYPES = ['vectorTile', 'rasterTile', 'wms', 'tile3d', 'bitmap'];
const COLUMN_FIELD_SUFFIX = 'Field';

function readJson(fileName) {
  return JSON.parse(readFileSync(path.join(dataDirectory, fileName), 'utf8'));
}

function validateRawFiles() {
  const rawChecks = [];
  for (const fileName of readdirSync(dataDirectory)) {
    const filePath = path.join(dataDirectory, fileName);
    if (fileName.endsWith('.csv')) {
      const processed = processCsvData(readFileSync(filePath, 'utf8'));
      assert(processed && processed.rows.length >= 100, `${fileName}: Kepler CSV processor returned too few rows`);
      assert(processed.fields.every(field => field.type), `${fileName}: untyped field`);
      rawChecks.push(`${fileName} rows=${processed.rows.length}`);
    } else if (fileName.endsWith('.geojson')) {
      const processed = processGeojson(JSON.parse(readFileSync(filePath, 'utf8')));
      assert(processed.rows.length >= 50, `${fileName}: too few features`);
      assert.equal(processed.fields[0].type, 'geojson', `${fileName}: _geojson field type`);
      rawChecks.push(`${fileName} features=${processed.rows.length}`);
    }
  }
  return rawChecks;
}

function validateTokens(dataset, layer) {
  const column = layer.config.columns.token || layer.config.columns.hex_id;
  const fieldIndex = dataset.fields.findIndex(field => field.name === column);
  for (const row of dataset.allData) {
    const token = row[fieldIndex];
    if (layer.type === 'hexagonId') assert(h3.isValidCell(token), `${layer.id}: invalid H3 cell ${token}`);
    else if (layer.type === 's2') {
      const {lat, lng} = S2.idToLatLng(BigInt(`0x${String(token).padEnd(16, '0')}`).toString());
      assert(Number.isFinite(lat) && Number.isFinite(lng), `${layer.id}: invalid S2 token ${token}`);
    } else if (layer.type === 'geohash') assert(/^[0-9b-hjkmnp-z]+$/.test(token), `${layer.id}: invalid geohash ${token}`);
    else if (layer.type === 'a5') {
      const [longitude, latitude] = a5.cellToLonLat(a5.hexToU64(token));
      assert(Number.isFinite(longitude) && Number.isFinite(latitude), `${layer.id}: invalid A5 cell ${token}`);
    }
  }
}

function validateSavedMap(fileName, expectations) {
  const raw = readJson(fileName);
  assert.equal(raw.info?.app, 'kepler.gl', `${fileName}: info.app must be kepler.gl`);
  assert(Array.isArray(raw.datasets) && raw.datasets.length, `${fileName}: datasets missing`);
  assert.equal(raw.config.version, 'v1');
  assert(raw.datasets.every(entry => entry.version === 'v1' && entry.data?.id && entry.data.fields && entry.data.allData));

  // The exact call Kepler's Load Files makes for a map JSON.
  const loaded = processKeplerglJSON(raw);
  assert.equal(loaded.datasets.length, raw.datasets.length, `${fileName}: dataset count`);
  assert(loaded.config, `${fileName}: KeplerGlSchema could not parse the saved config`);
  assert(loaded.info !== undefined);
  const directLoad = KeplerGlSchema.load(raw.datasets, raw.config);
  assert.equal(directLoad.datasets.length, loaded.datasets.length);

  const datasetsById = new Map(loaded.datasets.map(entry => [entry.info.id, entry.data]));
  const visState = loaded.config.visState;
  const types = [];
  for (const layer of visState.layers) {
    assert(LayerClasses[layer.type], `${fileName}: layer ${layer.id} has unsupported type ${layer.type}`);
    const dataset = datasetsById.get(layer.config.dataId);
    assert(dataset, `${fileName}: layer ${layer.id} references missing dataset ${layer.config.dataId}`);
    const fieldNames = new Set(dataset.fields.map(field => field.name));
    for (const [key, columnName] of Object.entries(layer.config.columns)) {
      assert(fieldNames.has(columnName), `${fileName}: layer ${layer.id} column ${key} -> ${columnName} missing`);
    }
    for (const [key, value] of Object.entries(layer.config)) {
      if (key.endsWith(COLUMN_FIELD_SUFFIX) && value) {
        assert(fieldNames.has(value.name), `${fileName}: layer ${layer.id} channel ${key} -> ${value.name} missing`);
      }
    }
    // Instantiating the real layer class from the parsed config proves the config is accepted.
    const columns = Object.fromEntries(
      Object.entries(layer.config.columns).map(([key, name]) => [key, {value: name, fieldIdx: dataset.fields.findIndex(field => field.name === name)}])
    );
    const instance = new LayerClasses[layer.type]({id: layer.id, ...layer.config, columns});
    assert.equal(instance.type, layer.type);
    assert(instance.isValidToSave(), `${fileName}: layer ${layer.id} is incomplete after load`);
    validateTokens({fields: dataset.fields, allData: dataset.rows}, layer);
    types.push(layer.type);
  }
  for (const filter of visState.filters) {
    for (const dataId of filter.dataId) {
      const dataset = datasetsById.get(dataId);
      assert(dataset, `${fileName}: filter ${filter.id} references missing dataset ${dataId}`);
      for (const name of filter.name) assert(dataset.fields.some(field => field.name === name), `${fileName}: filter field ${name} missing`);
    }
    assert(['range', 'timeRange', 'multiSelect', 'select'].includes(filter.type));
  }
  for (const [dataId, fields] of Object.entries(visState.interactionConfig.tooltip.fieldsToShow)) {
    const dataset = datasetsById.get(dataId);
    assert(dataset, `${fileName}: tooltip references missing dataset ${dataId}`);
    for (const {name} of fields) assert(dataset.fields.some(field => field.name === name), `${fileName}: tooltip field ${name} missing`);
  }
  // Re-saving the parsed state must reproduce the layer and filter identity (round trip).
  assert.deepEqual(visState.layers.map(layer => layer.id), raw.config.config.visState.layers.map(layer => layer.id));
  assert.deepEqual(visState.filters.map(filter => filter.id), raw.config.config.visState.filters.map(filter => filter.id));
  assert.equal(loaded.config.mapState.zoom, raw.config.config.mapState.zoom);
  assert.equal(visState.splitMaps.length, 0, `${fileName}: split maps are not supported by the resident host`);

  for (const type of expectations.types || []) assert(types.includes(type), `${fileName}: expected layer type ${type}`);
  if (expectations.exactTypes) assert.deepEqual([...types].sort(), [...expectations.exactTypes].sort());
  for (const type of REMOTE_SOURCE_TYPES) assert(!types.includes(type), `${fileName}: remote-source type ${type} is not fixture-able`);
  return {fileName, datasets: loaded.datasets.length, layers: types, filters: visState.filters.map(filter => filter.type)};
}

const results = [
  validateSavedMap('earthquakes.kepler.json', {exactTypes: ['point']}),
  validateSavedMap('nyc-trips.kepler.json', {exactTypes: ['arc', 'hexagon']}),
  validateSavedMap('sf-zips.kepler.json', {exactTypes: ['geojson']}),
  validateSavedMap('h3-hexagons.kepler.json', {exactTypes: ['hexagonId']}),
  validateSavedMap('arcs-flows.kepler.json', {exactTypes: ['arc', 'line', 'flow']}),
  validateSavedMap('trips.kepler.json', {exactTypes: ['trip']}),
  validateSavedMap('all-layer-types.kepler.json', {exactTypes: EXPECTED_ALL_LAYER_TYPES})
];
const rawChecks = validateRawFiles();
console.log(JSON.stringify({passed: true, rawChecks, results}, null, 1));
