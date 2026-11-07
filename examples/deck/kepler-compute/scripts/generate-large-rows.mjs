// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Seeded large-row fixture generator for scale probes (100k to 1M rows).
//
// Usage: node scripts/generate-large-rows.mjs <outputPath> [rows=100000] [seed=1] [format=csv|ndjson|json]
//
// Columns: id, longitude, latitude, value_a, value_b, population. Points are drawn from a few
// Gaussian clusters plus uniform background so distance bands and kNN see varied densities;
// value_a is spatially smooth plus noise, value_b is correlated with value_a.
// The output is deterministic for a given (rows, seed). Load it in the sandbox through the file
// picker, or serve it and use the same ?rows= style query (for example ?rows=100000) for the
// built-in synthetic generator when a file upload is not needed.

import {writeFileSync} from 'node:fs';

const [outputPath, rowsArgument = '100000', seedArgument = '1', format = 'csv'] =
  process.argv.slice(2);
if (!outputPath) {
  console.error('Usage: node generate-large-rows.mjs <outputPath> [rows] [seed] [csv|ndjson|json]');
  process.exit(1);
}
const rowCount = Number(rowsArgument);
if (!Number.isInteger(rowCount) || rowCount < 1 || rowCount > 2 ** 24 - 1) {
  console.error('rows must be an integer in [1, 16777215]');
  process.exit(1);
}

function createRandom(seed) {
  let state = seed >>> 0 || 1;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
const random = createRandom(Number(seedArgument));
function nextGaussian() {
  return Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
}
const centers = Array.from({length: 8}, () => [-120 + random() * 100, 25 + random() * 25]);

const rows = [];
for (let index = 0; index < rowCount; index++) {
  let longitude;
  let latitude;
  if (random() < 0.8) {
    const center = centers[Math.floor(random() * centers.length)];
    longitude = center[0] + nextGaussian() * 1.5;
    latitude = center[1] + nextGaussian() * 1.0;
  } else {
    longitude = -125 + random() * 60;
    latitude = 20 + random() * 30;
  }
  longitude = Math.max(-179.9, Math.min(179.9, longitude));
  latitude = Math.max(-89.9, Math.min(89.9, latitude));
  const smooth = Math.sin(longitude / 7) + Math.cos(latitude / 5);
  const valueA = smooth + 0.5 * nextGaussian();
  const valueB = 0.6 * valueA + 0.4 * nextGaussian();
  const population = Math.floor(100 + random() * 10000);
  rows.push({
    id: index,
    longitude: Number(longitude.toFixed(6)),
    latitude: Number(latitude.toFixed(6)),
    value_a: Number(valueA.toFixed(5)),
    value_b: Number(valueB.toFixed(5)),
    population
  });
}

let text;
if (format === 'json') text = JSON.stringify(rows);
else if (format === 'ndjson') text = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
else {
  const header = Object.keys(rows[0]);
  text = [header.join(','), ...rows.map(row => header.map(key => row[key]).join(','))].join('\n') + '\n';
}
writeFileSync(outputPath, text);
console.log(`wrote ${rowCount} rows to ${outputPath} (${format})`);
