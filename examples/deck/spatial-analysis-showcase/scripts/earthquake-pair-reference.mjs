// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {EARTH_RADIUS_METERS, measureHaversineAngleRadians, toRadians} from './lib/geodesy.mjs';

// Independent double-precision pair sums on the declared float32 planar inputs. This is a
// numerical reference for the specified descriptive estimator, not a fitted stochastic model.
const directory = new URL('../public/data/poopdeck-earthquakes/', import.meta.url);
const [positions, depths, types] = await Promise.all(['position.bin', 'depth.bin', 'type.bin'].map(name => readFile(new URL(name, directory))));
const originLatitude = toRadians(36.5);
const originLongitude = toRadians(140.5);
const events = [];
for (let row = 0; row < types.length; row++) {
  if (types[row] !== 0) continue;
  const latitude = toRadians(positions.readFloatLE(row * 8 + 4));
  const longitudeDelta = toRadians(positions.readFloatLE(row * 8)) - originLongitude;
  const angle = measureHaversineAngleRadians(latitude, originLatitude, longitudeDelta);
  const scale = EARTH_RADIUS_METERS * (angle === 0 ? 1 : angle / Math.sin(angle));
  const horizontal = scale * Math.cos(latitude) * Math.sin(longitudeDelta);
  const vertical = scale * (Math.cos(originLatitude) * Math.sin(latitude) - Math.sin(originLatitude) * Math.cos(latitude) * Math.cos(longitudeDelta));
  if (Math.abs(horizontal) > 135000 || Math.abs(vertical) > 166000) continue;
  const depth = depths.readFloatLE(row * 4);
  events.push({sourceRow: row, horizontal: Math.fround(horizontal), vertical: Math.fround(vertical), depth, group: depth < 70 ? 0 : 1});
}
assert.equal(events.length, 802);
assert.equal(events.filter(event => event.group === 0).length, 673);
assert.equal(events.filter(event => event.depth === 10).length, 17);
const width = 270000;
const height = 332000;
const area = width * height;
const scenarios = [];
for (const excludeExactTen of [false, true]) for (const correction of ['none', 'translation']) for (const pair of ['all', 'shallow-to-deep', 'deep-to-shallow', 'shallow-to-shallow']) {
  const eligible = events.filter(event => !excludeExactTen || event.depth !== 10);
  const focusGroup = pair === 'all' ? null : pair === 'deep-to-shallow' ? 1 : 0;
  const neighborGroup = pair === 'all' ? null : pair === 'shallow-to-deep' ? 1 : 0;
  const focus = eligible.filter(event => focusGroup === null || event.group === focusGroup);
  const neighbors = eligible.filter(event => neighborGroup === null || event.group === neighborGroup);
  const meanProduct = focus.reduce((sum, event) => sum + event.depth, 0) / focus.length * neighbors.reduce((sum, event) => sum + event.depth, 0) / neighbors.length;
  const shells = Array.from({length: 16}, () => ({weight: 0, products: 0, count: 0}));
  for (const first of focus) for (const second of neighbors) {
    if (first.sourceRow === second.sourceRow) continue;
    const horizontal = Math.abs(first.horizontal - second.horizontal);
    const vertical = Math.abs(first.vertical - second.vertical);
    const distance = Math.hypot(horizontal, vertical);
    if (distance > 80000) continue;
    const bin = Math.max(0, Math.ceil(distance / 5000) - 1);
    const weight = correction === 'none' ? 1 : area / ((width - horizontal) * (height - vertical));
    shells[bin].weight += weight;
    shells[bin].products += weight * first.depth * second.depth;
    shells[bin].count++;
  }
  let cumulativeWeight = 0;
  let cumulativeCount = 0;
  const bins = shells.map((shell, index) => {
    cumulativeWeight += shell.weight;
    cumulativeCount += shell.count;
    const radius = (index + 1) * 5000;
    const lower = index * 5000;
    const scale = area / focus.length / neighbors.length;
    const kStatistic = cumulativeWeight * scale;
    return {radius, k: kStatistic, lMinusR: Math.sqrt(kStatistic / Math.PI) - radius, pairCorrelation: shell.weight > 0 ? shell.weight * scale / (Math.PI * (radius ** 2 - lower ** 2)) : null, markCorrelation: shell.weight > 0 ? shell.products / shell.weight / meanProduct : null, cumulativeCount};
  });
  scenarios.push({excludeExactTen, correction, pair, eligible: eligible.length, focus: focus.length, neighbors: neighbors.length, bins});
}
const outputFile = process.argv[2] ?? '/tmp/earthquake-pair-reference.json';
await writeFile(outputFile, JSON.stringify({events: events.length, area, scenarios}, null, 2) + '\n');
console.log(JSON.stringify(scenarios.map(({excludeExactTen, correction, pair, bins}) => ({excludeExactTen, correction, pair, terminal: bins.at(-1)}))));
