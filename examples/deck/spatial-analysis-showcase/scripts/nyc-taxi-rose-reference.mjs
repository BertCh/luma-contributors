// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFile, writeFile} from 'node:fs/promises';
import {addMetersToLngLat, getDistanceScales} from '@math.gl/web-mercator';

const dataDirectory = new URL('../public/data/poopdeck-nyc-taxi-paths/', import.meta.url);
const outputUrl = new URL(
  '../../../../dev-docs/research/showcase-overhaul-2026-10-25/nyc-taxi-rose-reference.json',
  import.meta.url
);
const playhead = 600;
const binCount = 36;
const classCount = 5;

async function readTyped(name, Type) {
  const bytes = await readFile(new URL(name, dataDirectory));
  return new Type(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

const manifest = JSON.parse(await readFile(new URL('manifest.json', dataDirectory), 'utf8'));
const [pathOffsets, lonLat, timestamps, fares] = await Promise.all([
  readTyped('pathOffsets.bin', Uint32Array),
  readTyped('vertices.bin', Float32Array),
  readTyped('time.bin', Uint32Array),
  readTyped('fare_amount.bin', Float32Array)
]);

const origin = [
  (manifest.bbox[0] + manifest.bbox[2]) / 2,
  (manifest.bbox[1] + manifest.bbox[3]) / 2
];
const scales = getDistanceScales({longitude: origin[0], latitude: origin[1], highPrecision: true});
const metersPerDegree = [
  scales.unitsPerDegree[0] * scales.metersPerUnit[0],
  scales.unitsPerDegree[1] * scales.metersPerUnit[1]
];
const positions = new Float32Array(lonLat.length);
for (let row = 0; row < lonLat.length / 2; row++) {
  const longitude = lonLat[row * 2];
  const latitude = lonLat[row * 2 + 1];
  let x = (longitude - origin[0]) * metersPerDegree[0];
  let y = (latitude - origin[1]) * metersPerDegree[1];
  const [estimatedLongitude, estimatedLatitude] = addMetersToLngLat(origin, [x, y]);
  x += (longitude - estimatedLongitude) * metersPerDegree[0];
  y += (latitude - estimatedLatitude) * metersPerDegree[1];
  positions[row * 2] = x;
  positions[row * 2 + 1] = y;
}

function tidy(value) {
  return Number(value.toPrecision(12));
}

function quantileBreaks(values, requestedClasses) {
  const sorted = Array.from(values).filter(Number.isFinite).sort((first, second) => first - second);
  const breaks = [];
  for (let index = 1; index < requestedClasses; index++) {
    const position = ((sorted.length - 1) * index) / requestedClasses;
    const lower = Math.floor(position);
    const upper = Math.min(sorted.length - 1, lower + 1);
    breaks.push(tidy(sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower)));
  }
  return breaks.filter((value, index) => index === 0 || value > breaks[index - 1]);
}

function classIndex(value, breaks) {
  let low = 0;
  let high = breaks.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (breaks[middle] <= value) low = middle + 1;
    else high = middle;
  }
  return low;
}

const breaks = quantileBreaks(fares, classCount);
const observations = [];
for (let track = 0; track < pathOffsets.length - 1; track++) {
  const start = pathOffsets[track];
  const end = pathOffsets[track + 1];
  if (start >= end || timestamps[start] > playhead || timestamps[end - 1] < playhead) continue;
  let low = start + 1;
  let high = end;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (timestamps[middle] > playhead) high = middle;
    else low = middle + 1;
  }
  const segmentRow = low === end ? end - 2 : low - 1;
  if (segmentRow < start || segmentRow + 1 >= end) continue;
  const dx = Math.fround(positions[(segmentRow + 1) * 2] - positions[segmentRow * 2]);
  const dy = Math.fround(positions[(segmentRow + 1) * 2 + 1] - positions[segmentRow * 2 + 1]);
  if (dx === 0 && dy === 0) continue;
  const heading = Math.fround(Math.atan2(dy, dx));
  const headingX = Math.fround(Math.cos(heading));
  const headingY = Math.fround(Math.sin(heading));
  const length = Math.fround(Math.sqrt(Math.fround(dx * dx) + Math.fround(dy * dy)));
  observations.push({
    group: classIndex(fares[track], breaks),
    headingX,
    headingY,
    length
  });
}

function summarize(mode, weighting, grouping) {
  const binTotal = mode === 'directional' ? binCount : binCount / 2;
  const groupTotal = grouping === 'all' ? 1 : classCount;
  const bins = Array.from({length: groupTotal}, () => new Array(binTotal).fill(0));
  const rows = Array.from({length: groupTotal}, () => []);
  for (const observation of observations) {
    const group = grouping === 'all' ? 0 : observation.group;
    let angle = Math.atan2(observation.headingY, observation.headingX);
    if (angle < 0) angle += Math.PI * 2;
    if (mode === 'axial' && angle >= Math.PI) angle -= Math.PI;
    const period = mode === 'axial' ? Math.PI : Math.PI * 2;
    const centeredBinAngle = (angle + period / (2 * binTotal)) % period;
    const bin = Math.min(binTotal - 1, Math.floor(centeredBinAngle * binTotal / period + 1e-6));
    const weight = weighting === 'equal' ? 1 : observation.length;
    bins[group][bin] += weight;
    rows[group].push({angle: mode === 'axial' ? angle * 2 : angle, weight});
  }
  const groups = rows.map((groupRows, group) => {
    let sumWeight = 0;
    let cosine = 0;
    let sine = 0;
    for (const row of groupRows) {
      sumWeight += row.weight;
      cosine += row.weight * Math.cos(row.angle);
      sine += row.weight * Math.sin(row.angle);
    }
    return {
      count: groupRows.length,
      totalWeight: sumWeight,
      meanAngle: (Math.atan2(sine, cosine) + Math.PI * 2) % (Math.PI * 2) / (mode === 'axial' ? 2 : 1),
      resultantLength: sumWeight > 0 ? Math.hypot(cosine, sine) / sumWeight : null,
      bins: bins[group]
    };
  });
  return {mode, weighting, grouping, groups};
}

const results = {};
for (const mode of ['directional', 'axial']) {
  for (const weighting of ['equal', 'segment-length']) {
    for (const grouping of ['all', 'fare-class']) {
      const result = summarize(mode, weighting, grouping);
      results[`${mode}/${weighting}/${grouping}`] = result;
    }
  }
}
const reference = {
  source: 'Bundled poopdeck-nyc-taxi-paths bytes; independent CPU interpolation and circular summaries',
  playheadSecondsAfterDatasetOrigin: playhead,
  origin,
  fareQuantileBreaks: breaks,
  activeNonzeroHeadingRows: observations.length,
  supportedFareClasses: classCount,
  compassDisplayOrder: 'North clockwise; GPU bins are remapped from positive-x counter-clockwise; axial bins mirror antipodal sectors at half weight.',
  results
};
await writeFile(outputUrl, `${JSON.stringify(reference, null, 2)}\n`);
console.log(`Wrote independent taxi rose reference: ${outputUrl.pathname}`);
