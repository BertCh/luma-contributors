// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const website = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.resolve(website, '../examples/deck/spatial-analysis-showcase/public/data');
const target = path.join(website, 'src/gpu/data');

function readColumn(dataset, name, ArrayType = Float32Array) {
  const bytes = fs.readFileSync(path.join(source, dataset, name + '.bin'));
  return new ArrayType(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

// Each row is four little-endian uint16 values. Coordinates are fractions of the window,
// quantized to 1/10,000. Measurement units and missing-value conventions are in point-sources.md.
function writeRows(name, bounds, positions, measurement) {
  const [west, south, east, north] = bounds;
  const rows = [];
  for (let row = 0; row < positions.length / 2; row++) {
    const longitude = positions[2 * row];
    const latitude = positions[2 * row + 1];
    if (longitude < west || longitude > east || latitude < south || latitude > north) continue;
    const values = measurement(row);
    if (!values) continue;
    rows.push(
      Math.round(((longitude - west) / (east - west)) * 10000),
      Math.round(((latitude - south) / (north - south)) * 10000),
      ...values
    );
  }
  const bytes = Buffer.alloc(rows.length * 2);
  rows.forEach((value, index) => bytes.writeUInt16LE(value, index * 2));
  fs.writeFileSync(
    path.join(target, name + '.json'),
    JSON.stringify({bounds, data: bytes.toString('base64')}) + '\n'
  );
  console.log(name, rows.length / 4, 'observations');
}

const rainfall = readColumn('ghcn-stations', 'prcp');
const previousRainfall = readColumn('ghcn-stations', 'prcpPrevDay');
writeRows('helene-rain', [-92, 24, -74, 40], readColumn('ghcn-stations', 'position'), (row) => [
  Math.round(rainfall[row] * 10),
  Number.isFinite(previousRainfall[row]) ? Math.round(previousRainfall[row] * 10) : 65535
]);

const magnitude = readColumn('poopdeck-earthquakes', 'magnitude');
const seconds = readColumn('poopdeck-earthquakes', 'time', Uint32Array);
const eventType = readColumn('poopdeck-earthquakes', 'type', Uint8Array);
writeRows(
  'japan-earthquakes',
  [126, 28, 146, 42],
  readColumn('poopdeck-earthquakes', 'position'),
  (row) =>
    eventType[row] === 0 && magnitude[row] >= 4
      ? [Math.round(magnitude[row] * 100), Math.floor(seconds[row] / 86400)]
      : null
);

const coastline = JSON.parse(
  fs.readFileSync(path.join(source, 'natural-earth/ne_110m_coastline.geojson'), 'utf8')
);
const endpoints = [];
const insideJapan = ([longitude, latitude]) =>
  longitude >= 126 && longitude <= 146 && latitude >= 28 && latitude <= 42;
for (const {geometry} of coastline.features) {
  const lines = geometry.type === 'MultiLineString' ? geometry.coordinates : [geometry.coordinates];
  for (const line of lines) {
    for (let point = 1; point < line.length; point++) {
      if (insideJapan(line[point - 1]) || insideJapan(line[point])) {
        endpoints.push(line[point - 1], line[point]);
      }
    }
  }
}
fs.writeFileSync(path.join(target, 'japan-coast.json'), JSON.stringify(endpoints) + '\n');
