// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {build} from 'esbuild';

const datasetDirectory = fileURLToPath(new URL('../../../public/data/openflights/', import.meta.url));
const read = name => readFile(path.join(datasetDirectory, name));
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

function parseCsvRow(line) {
  return [...line.matchAll(/(?:^|,)(?:"((?:[^"]|"")*)"|([^,]*))/g)].map(match =>
    (match[1] ?? match[2] ?? '').replaceAll('""', '"')
  );
}

function buildPermutation(order, degrees, countryRanks) {
  const vertices = Array.from({length: degrees.length}, (_, vertex) => vertex);
  if (order === 'country') vertices.sort((first, second) => countryRanks[first] - countryRanks[second] || degrees[second] - degrees[first] || first - second);
  if (order === 'degree') vertices.sort((first, second) => degrees[second] - degrees[first] || first - second);
  const position = new Uint32Array(vertices.length);
  vertices.forEach((vertex, index) => { position[vertex] = index; });
  return position;
}

function getWholeCellSupport(extent, resolution) {
  const positionsPerCell = Math.ceil(extent / resolution);
  const cells = Math.ceil(extent / positionsPerCell);
  return {positionsPerCell, cells, visibleSpan: cells * positionsPerCell, compiledSpan: resolution * positionsPerCell};
}

// A partly filled last cell remains a whole cell; hidden compiled bins still affect GPU maximum.
assert.deepEqual(getWholeCellSupport(271, 128), {positionsPerCell: 3, cells: 91, visibleSpan: 273, compiledSpan: 384});

/** Independent scalar CPU reconstruction from the bundled OpenFlights binaries. */
export async function buildFlightMatrixWeightReference({includeEurope = false} = {}) {
  const [origins, destinations, routeRecordBytes, degreeBytes, csvBytes] = await Promise.all([
    read('origin.bin'), read('destination.bin'), read('count.bin'), read('degree.bin'), read('airports.csv')
  ]);
  assert.equal(origins.length % 4, 0);
  assert.equal(destinations.length, origins.length);
  assert.equal(routeRecordBytes.length, origins.length / 2);
  assert.equal(degreeBytes.length % 2, 0);
  const source = new Uint32Array(origins.buffer, origins.byteOffset, origins.length / 4);
  const target = new Uint32Array(destinations.buffer, destinations.byteOffset, destinations.length / 4);
  const routeRecords = new Uint16Array(routeRecordBytes.buffer, routeRecordBytes.byteOffset, routeRecordBytes.length / 2);
  const degrees = new Uint16Array(degreeBytes.buffer, degreeBytes.byteOffset, degreeBytes.length / 2);
  const airportRows = csvBytes.toString('utf8').trimEnd().split('\n').slice(1).map(parseCsvRow);
  const airportCodes = airportRows.map(row => row[1]);
  assert.equal(airportCodes.length, degrees.length);

  // Share only the declared geographic taxonomy, not the production ordering/binning algorithm.
  const geographyPath = fileURLToPath(new URL('./b11-geography.ts', import.meta.url));
  const geographySource = await readFile(geographyPath);
  const bundledGeography = await build({entryPoints: [geographyPath], bundle: true, write: false, platform: 'node', format: 'cjs', define: {'import.meta.env.BASE_URL': '"/"'}});
  const geographyModule = {exports: {}};
  new Function('module', 'exports', bundledGeography.outputFiles[0].text)(geographyModule, geographyModule.exports);
  const countries = airportRows.map(row => row[4].trim());
  const continents = countries.map(country => geographyModule.exports.getContinentIndex(country));
  const countryCounts = new Map();
  const countryContinents = new Map();
  countries.forEach((country, index) => {
    countryCounts.set(country, (countryCounts.get(country) ?? 0) + 1);
    countryContinents.set(country, continents[index]);
  });
  const rankedCountries = [...countryCounts.keys()].sort((first, second) =>
    countryContinents.get(first) - countryContinents.get(second) || countryCounts.get(second) - countryCounts.get(first) || first.localeCompare(second));
  const countryRanks = countries.map(country => rankedCountries.indexOf(country));
  const europeCount = continents.filter(continent => continent === 0).length;
  assert.equal(europeCount, 660, 'bundled Europe support');

  const offsets = new Uint32Array(degrees.length + 1);
  let routeRecordTotal = 0;
  for (let edge = 0; edge < source.length; edge++) {
    offsets[source[edge] + 1]++;
    offsets[target[edge] + 1]++;
    routeRecordTotal += routeRecords[edge];
  }
  for (let vertex = 0; vertex < degrees.length; vertex++) offsets[vertex + 1] += offsets[vertex];
  const neighbors = new Uint32Array(source.length * 2);
  const weights = new Uint32Array(source.length * 2);
  const cursor = offsets.slice(0, degrees.length);
  for (let edge = 0; edge < source.length; edge++) {
    const firstSlot = cursor[source[edge]]++;
    const secondSlot = cursor[target[edge]]++;
    neighbors[firstSlot] = target[edge];
    neighbors[secondSlot] = source[edge];
    weights[firstSlot] = routeRecords[edge];
    weights[secondSlot] = routeRecords[edge];
  }
  assert.equal(offsets.at(-1), neighbors.length);
  assert.equal(neighbors.length, source.length * 2);
  assert.equal(weights.reduce((sum, value) => sum + value, 0), routeRecordTotal * 2);
  assert.ok(routeRecordTotal * 2 <= 0xffffffff, 'symmetric route-record sum fits uint32');

  const configurations = [
    {order: 'input', resolution: 128, start: 0, extent: degrees.length},
    {order: 'input', resolution: 256, start: 0, extent: degrees.length},
    {order: 'degree', resolution: 128, start: 0, extent: degrees.length},
    {order: 'degree', resolution: 256, start: 0, extent: degrees.length},
    {order: 'degree', resolution: 128, start: Math.round(degrees.length * 0.5 - Math.round(degrees.length / 12) / 2), extent: Math.round(degrees.length / 12)}
  ];
  if (includeEurope) configurations.push({order: 'country', focus: 'Europe', resolution: 512, start: 0, extent: europeCount});
  const results = configurations.map(configuration => {
    const {order, resolution, start, extent} = configuration;
    const position = buildPermutation(order, degrees, countryRanks);
    const {positionsPerCell, cells, visibleSpan, compiledSpan} = getWholeCellSupport(extent, resolution);
    const bins = new Uint32Array(resolution * resolution);
    const weightedBins = new Uint32Array(resolution * resolution);
    for (let vertex = 0; vertex < degrees.length; vertex++) {
      const rowPosition = position[vertex];
      if (rowPosition < start || rowPosition >= start + compiledSpan) continue;
      const rowBin = Math.floor((rowPosition - start) / positionsPerCell);
      for (let slot = offsets[vertex]; slot < offsets[vertex + 1]; slot++) {
        const columnPosition = position[neighbors[slot]];
        if (columnPosition < start || columnPosition >= start + compiledSpan) continue;
        const columnBin = Math.floor((columnPosition - start) / positionsPerCell);
        const index = rowBin * resolution + columnBin;
        bins[index]++;
        weightedBins[index] += weights[slot];
      }
    }
    let occupied = 0;
    let pairTotal = 0;
    let weightedTotal = 0;
    const maximumPairs = bins.reduce((maximum, value) => Math.max(maximum, value), 0);
    const maximumWeight = weightedBins.reduce((maximum, value) => Math.max(maximum, value), 0);
    const pairClassCounts = [0, 0, 0, 0, 0, 0];
    const weightClassCounts = [0, 0, 0, 0, 0, 0];
    for (let row = 0; row < cells; row++) {
      for (let column = 0; column < cells; column++) {
        const index = row * resolution + column;
        const count = bins[index];
        const weight = weightedBins[index];
        if (count > 0) occupied++;
        pairTotal += count;
        weightedTotal += weight;
        if (count > 0) pairClassCounts[Math.min(5, count < 3 ? count - 1 : count < 6 ? 2 : count < 11 ? 3 : count < 21 ? 4 : 5)]++;
        if (weight > 0) weightClassCounts[Math.min(5, weight < 3 ? weight - 1 : weight < 6 ? 2 : weight < 11 ? 3 : weight < 21 ? 4 : 5)]++;
      }
    }
    return {
      ...configuration,
      cells,
      positionsPerCell,
      visibleSpan,
      compiledSpan,
      pairTotal,
      symmetricRouteRecordTotal: weightedTotal,
      occupied,
      maximumPairs,
      maximumRouteRecords: maximumWeight,
      pairClassCounts,
      routeRecordClassCounts: weightClassCounts
    };
  });
  for (const result of results.filter(result => result.start === 0 && result.extent === degrees.length)) {
    assert.equal(result.pairTotal, source.length * 2, 'full symmetric pair count conserves every CSR slot');
    assert.equal(result.symmetricRouteRecordTotal, routeRecordTotal * 2, 'full symmetric route-record sum conserves every CSR slot weight');
  }
  return {
    source: {
      name: 'OpenFlights routes.dat derived pair table',
      records: source.length,
      airports: degrees.length,
      routeRecordTotal,
      symmetricRouteRecordTotal: routeRecordTotal * 2,
      sha256: {
        origin: sha256(origins),
        destination: sha256(destinations),
        count: sha256(routeRecordBytes),
        degree: sha256(degreeBytes),
        airportsCsv: sha256(csvBytes),
        geographicTaxonomy: sha256(geographySource)
      }
    },
    results
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const reference = await buildFlightMatrixWeightReference();
  console.log(JSON.stringify(reference, null, 2));
}
