// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFileSync, writeFileSync} from 'node:fs';
import earcut, {deviation} from 'earcut';

const source = new URL('../../../public/data/chicago-tracts/', import.meta.url);
const destination = new URL(
  '../../../../../../website/src/gpu/data/chicago-tracts.json',
  import.meta.url
);
const {features} = JSON.parse(readFileSync(new URL('tracts.geojson', source)));
const columns = Object.fromEntries(
  ['poverty150Pct', 'unemployedPct', 'age65', 'population'].map(name => [
    name,
    readFileSync(new URL(name + '.bin', source))
  ])
);

// Whole Census shells in a local equirectangular frame around Chicago's Near West Side.
const origin = [-87.67, 41.878];
const metersPerDegree = 111195;
const longitudeScale = metersPerDegree * Math.cos((origin[1] * Math.PI) / 180);
function project([longitude, latitude]) {
  return [
    Math.round((longitude - origin[0]) * longitudeScale),
    Math.round((latitude - origin[1]) * metersPerDegree)
  ];
}

const tracts = [];
const edges = new Map();
for (const {geometry, properties} of features) {
  if (geometry.type !== 'Polygon' || geometry.coordinates.length !== 1) continue;
  const {index, GEOID: geoid} = properties;
  const attributes = Object.fromEntries(
    Object.entries(columns).map(([name, buffer]) => [name, buffer.readFloatLE(index * 4)])
  );
  if (
    Object.values(attributes).some(value => !Number.isFinite(value)) ||
    attributes.population <= 0
  )
    continue;
  const ring = geometry.coordinates[0].slice(0, -1);
  const positions = ring.flatMap(project);
  const triangles = earcut(positions);
  if (deviation(positions, [], 2, triangles) > 1e-6 || triangles.length === 0) continue;
  // Use the area-weighted centroid of the triangulated footprint.
  let area = 0,
    sumX = 0,
    sumY = 0;
  for (let offset = 0; offset < triangles.length; offset += 3) {
    const [a, b, c] = triangles
      .slice(offset, offset + 3)
      .map(vertex => positions.slice(vertex * 2, vertex * 2 + 2));
    const twiceArea = Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
    area += twiceArea;
    sumX += (twiceArea * (a[0] + b[0] + c[0])) / 3;
    sumY += (twiceArea * (a[1] + b[1] + c[1])) / 3;
  }
  if (!area) continue;
  const center = [Math.round(sumX / area), Math.round(sumY / area)];
  const neighbors = new Set();
  const row = tracts.length;
  for (let corner = 0; corner < ring.length; corner++) {
    const key = [ring[corner].join(','), ring[(corner + 1) % ring.length].join(',')]
      .sort()
      .join('|');
    if (edges.has(key)) {
      const other = edges.get(key);
      if (other !== row) {
        tracts[other].neighbors.add(row);
        neighbors.add(other);
      }
    } else {
      edges.set(key, row);
    }
  }
  tracts.push({
    geoid,
    positions,
    triangles,
    center,
    neighbors,
    // Central, east-west-elongated area suited to the landscape map panel.
    distance: Math.hypot(center[0] / 1.6, center[1]),
    population: attributes.population,
    age65: attributes.age65,
    povertyPercent: +attributes.poverty150Pct.toFixed(1),
    unemploymentPercent: +attributes.unemployedPct.toFixed(1)
  });
}
const order = [];
const included = new Set();
const byDistance = [...tracts.keys()].sort(
  (first, second) => tracts[first].distance - tracts[second].distance
);
const frontier = new Set([byDistance[0]]);
while (order.length < 156) {
  const row = byDistance.find(candidate => frontier.has(candidate));
  assert.notEqual(row, undefined);
  order.push(row);
  included.add(row);
  frontier.delete(row);
  for (const neighbor of tracts[row].neighbors) {
    if (!included.has(neighbor)) frontier.add(neighbor);
  }
}
const rows = order.map(row => {
  const {distance, neighbors, ...fields} = tracts[row];
  return {
    ...fields,
    neighbors: [...neighbors]
      .filter(other => included.has(other))
      .map(other => order.indexOf(other))
      .sort((a, b) => a - b)
  };
});
writeFileSync(destination, JSON.stringify(rows) + '\n');
