// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile, writeFile} from 'node:fs/promises';
import {PNG} from 'pngjs';

const sourceUrl = new URL('../public/data/grand-canyon-dem/dem.png', import.meta.url);
const manifestUrl = new URL('../public/data/grand-canyon-dem/manifest.json', import.meta.url);
const outputUrl = new URL(
  '../../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-terrain-boundaries-oracle.json',
  import.meta.url
);
const [pngBytes, manifestText] = await Promise.all([readFile(sourceUrl), readFile(manifestUrl, 'utf8')]);
const manifest = JSON.parse(manifestText);
const decoded = PNG.sync.read(pngBytes);
assert.equal(decoded.width, 2048);
assert.equal(decoded.height, 2048);

const sourceElevations = new Float32Array(decoded.width * decoded.height);
for (let cell = 0; cell < sourceElevations.length; cell++) {
  const offset = cell * 4;
  sourceElevations[cell] =
    decoded.data[offset] * 256 + decoded.data[offset + 1] + decoded.data[offset + 2] / 256 - 32768;
}

// Match createCanyonGrid(dataset, 32): sum the decoded f32 cells in row-major order and store
// each 32 × 32 block mean back to Float32Array, exactly as the showcase input is constructed.
const stride = 32;
const width = decoded.width / stride;
const height = decoded.height / stride;
const values = new Float32Array(width * height);
for (let row = 0; row < height; row++) {
  for (let column = 0; column < width; column++) {
    let sum = 0;
    for (let offsetRow = 0; offsetRow < stride; offsetRow++) {
      const sourceBase = (row * stride + offsetRow) * decoded.width + column * stride;
      for (let offsetColumn = 0; offsetColumn < stride; offsetColumn++) {
        sum += sourceElevations[sourceBase + offsetColumn];
      }
    }
    values[row * width + column] = sum / (stride * stride);
  }
}

// Independent CPU Kruskal/union-find oracle for the upper-excursion join tree. Every undirected
// 8-neighbour edge appears once in E, SE, S, SW order, matching the contributor's stable edge
// indexing. Edges are processed from the highest endpoint-minimum saddle downward. Each losing
// component elder receives that merge level; the surviving elder of each connected component is
// infinite. This computes the expected per-peak values without invoking the GPU contributor.
const parent = Uint32Array.from({length: values.length}, (_, index) => index);
const elder = Uint32Array.from({length: values.length}, (_, index) => index);
const prominence = new Float32Array(values.length);
const keySaddle = new Float32Array(values.length);
const parentPeak = new Uint32Array(values.length);
prominence.fill(Number.NaN);
keySaddle.fill(Number.NaN);
parentPeak.fill(0xffffffff);
const edges = [];
const directions = [
  [1, 0],
  [1, 1],
  [0, 1],
  [-1, 1]
];
for (let cell = 0; cell < values.length; cell++) {
  const row = Math.floor(cell / width);
  const column = cell % width;
  for (let direction = 0; direction < directions.length; direction++) {
    const [columnOffset, rowOffset] = directions[direction];
    const neighborColumn = column + columnOffset;
    const neighborRow = row + rowOffset;
    if (
      neighborColumn < 0 ||
      neighborColumn >= width ||
      neighborRow < 0 ||
      neighborRow >= height
    ) {
      continue;
    }
    const neighbor = neighborRow * width + neighborColumn;
    edges.push({index: cell * 4 + direction, first: cell, second: neighbor, saddle: Math.min(values[cell], values[neighbor])});
  }
}
edges.sort((first, second) => second.saddle - first.saddle || first.index - second.index);

function findRoot(cell) {
  let root = cell;
  while (parent[root] !== root) root = parent[root];
  while (parent[cell] !== cell) {
    const next = parent[cell];
    parent[cell] = root;
    cell = next;
  }
  return root;
}

for (const edge of edges) {
  const firstRoot = findRoot(edge.first);
  const secondRoot = findRoot(edge.second);
  if (firstRoot === secondRoot) continue;
  const firstElder = elder[firstRoot];
  const secondElder = elder[secondRoot];
  const firstHeight = values[firstElder];
  const secondHeight = values[secondElder];
  const firstWins = firstHeight > secondHeight || (firstHeight === secondHeight && firstElder < secondElder);
  const winner = firstWins ? firstElder : secondElder;
  const loser = firstWins ? secondElder : firstElder;
  keySaddle[loser] = edge.saddle;
  prominence[loser] = Math.fround(values[loser] - edge.saddle);
  parentPeak[loser] = winner;
  parent[firstRoot] = winner;
  parent[secondRoot] = winner;
  parent[winner] = winner;
  elder[winner] = winner;
}

const elderCells = new Set();
for (let cell = 0; cell < values.length; cell++) {
  const root = findRoot(cell);
  elderCells.add(elder[root]);
}
for (const cell of elderCells) {
  keySaddle[cell] = -Infinity;
  prominence[cell] = Infinity;
}
for (let cell = 0; cell < values.length; cell++) {
  if (Number.isNaN(prominence[cell])) {
    keySaddle[cell] = values[cell];
    prominence[cell] = 0;
  }
}

const finitePeaks = [];
let finitePeakCount = 0;
let infiniteElderCount = 0;
let minimum = Infinity;
let maximum = -Infinity;
for (let cell = 0; cell < values.length; cell++) {
  minimum = Math.min(minimum, values[cell]);
  maximum = Math.max(maximum, values[cell]);
  if (prominence[cell] === Infinity) infiniteElderCount++;
  if (Number.isFinite(prominence[cell]) && prominence[cell] > 0) {
    finitePeakCount++;
    finitePeaks.push({
      index: cell,
      row: Math.floor(cell / width) + 1,
      column: (cell % width) + 1,
      elevation: values[cell],
      keySaddle: keySaddle[cell],
      prominence: prominence[cell],
      parentCell: parentPeak[cell],
      parentRow: Math.floor(parentPeak[cell] / width) + 1,
      parentColumn: (parentPeak[cell] % width) + 1
    });
  }
}
finitePeaks.sort((first, second) => second.prominence - first.prominence || first.index - second.index);
const selected = finitePeaks[0];
assert.ok(selected, 'Expected at least one finite positive prominence peak');
const gridBytes = Buffer.from(values.buffer, values.byteOffset, values.byteLength);
const report = {
  dataset: 'grand-canyon-dem',
  source: manifest.properties.source,
  sourceImage: {
    width: decoded.width,
    height: decoded.height,
    sha256: createHash('sha256').update(pngBytes).digest('hex'),
    encoding: manifest.raster.encoding,
    pixelSizeMercatorM: manifest.raster.cellSizeMercatorM,
    verticalQuantizationM: manifest.raster.verticalQuantisationM
  },
  grid: {
    width,
    height,
    stride,
    cellOrdering: 'row-major; north-first',
    elevationUnit: manifest.raster.unit,
    minimumElevationM: minimum,
    maximumElevationM: maximum,
    meanElevationM: values.reduce((sum, value) => sum + value, 0) / values.length,
    sha256Float32LE: createHash('sha256').update(gridBytes).digest('hex')
  },
  prominence: {
    algorithm: 'CPU stable descending-saddle Kruskal union-find over 8-neighbor upper-excursion graph; E/SE/S/SW edges per cell; equal-height elder is lowest row-major index',
    finitePeakCount,
    infiniteComponentElderCount: infiniteElderCount,
    largestFinitePeak: selected,
    nextLargestFinitePeaks: finitePeaks.slice(1, 8)
  },
  tiling: {
    ownedRectangles: [
      {row: 0, column: 0, height: 32, width: 32},
      {row: 0, column: 32, height: 32, width: 32},
      {row: 32, column: 0, height: 32, width: 32},
      {row: 32, column: 32, height: 32, width: 32}
    ],
    note: 'D8 and branching GPU accumulations are checked against matching whole-window references in the browser script. This CPU oracle does not emulate depression filling, flat resolution, flow routing or iterative branching accumulation.'
  }
};
await writeFile(outputUrl, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report, null, 2));
